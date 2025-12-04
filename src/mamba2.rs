// Mamba2 (State Space Duality) Implementation
// Based on "Transformers are SSMs" (Dao & Gu, 2024)
// Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py
//
// Implementation Notes:
// - Follows HuggingFace's `torch_forward` (naive) path, NOT the CUDA kernel path
// - Uses batched matmul instead of 6D broadcast operations for efficiency
// - No custom Triton/CUDA kernels (Candle doesn't support them)
// - Performance is comparable to HF's torch_forward, which is actually fast
// - Future: Native Rust CUDA kernels via cudarc when needed
//
// Dimension notation conventions:
// - B: batch size
// - L or seq_len: sequence length
// - H or num_heads: number of attention heads
// - D or head_dim: dimension per head
// - N or state_size: SSM state dimension
// - C or chunk_size: chunk size for SSD decomposition

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{linear, Conv1d, Conv1dConfig, Linear, Module, VarBuilder};

// Numerical stability constants
const EPS: f64 = 1e-6;
const MAX_EXP_INPUT: f32 = 50.0;  // exp(50) ≈ 5e21, safe for f32
const MIN_EXP_INPUT: f32 = -50.0; // exp(-50) ≈ 2e-22, above f32 denorm
const MASK_NEG_VALUE: f32 = -1e9; // Large negative for masked positions (avoids -inf NaN issues)

/// Softplus activation: softplus(x) = log(1 + exp(x))
/// Preserves input dtype by using affine instead of scalar addition
fn softplus(x: &Tensor, beta: f64) -> Result<Tensor> {
    let beta_x = x.affine(beta, 0.0)?;
    let exp = beta_x.exp()?;
    // Use affine(1.0, 1.0) instead of (+ 1.0) to preserve dtype
    let one_plus_exp = exp.affine(1.0, 1.0)?;
    one_plus_exp.log()?.affine(1.0 / beta, 0.0)
}

/// Pad tensor on a specific dimension (preserves input dtype)
fn pad_1d(tensor: &Tensor, padding: (usize, usize), dim: usize, value: f32) -> Result<Tensor> {
    let (left, right) = padding;
    if left == 0 && right == 0 {
        return Ok(tensor.clone());
    }

    let shape = tensor.dims();
    let dtype = tensor.dtype();
    let mut left_shape = shape.to_vec();
    let mut right_shape = shape.to_vec();
    left_shape[dim] = left;
    right_shape[dim] = right;

    let mut parts = Vec::new();
    if left > 0 {
        parts.push(Tensor::full(value, left_shape.as_slice(), tensor.device())?.to_dtype(dtype)?);
    }
    parts.push(tensor.clone());
    if right > 0 {
        parts.push(Tensor::full(value, right_shape.as_slice(), tensor.device())?.to_dtype(dtype)?);
    }

    if parts.len() == 1 {
        Ok(parts.into_iter().next().unwrap())
    } else {
        Tensor::cat(&parts.iter().collect::<Vec<_>>(), dim)
    }
}

/// Helper: Pad tensor on seq_len dimension (dim=1)
fn pad_tensor_by_size(input: &Tensor, pad_size: usize) -> Result<Tensor> {
    if pad_size == 0 {
        return Ok(input.clone());
    }

    let shape = input.dims();
    match shape.len() {
        3 => {
            // [B, L, D] -> [B, L+pad, D]
            let zeros = Tensor::zeros(&[shape[0], pad_size, shape[2]], input.dtype(), input.device())?;
            Tensor::cat(&[input, &zeros], 1)
        }
        4 => {
            // [B, L, H, D] -> [B, L+pad, H, D]
            let zeros = Tensor::zeros(&[shape[0], pad_size, shape[2], shape[3]], input.dtype(), input.device())?;
            Tensor::cat(&[input, &zeros], 1)
        }
        _ => candle_core::bail!("pad_tensor_by_size: unsupported tensor rank {}", shape.len()),
    }
}

/// Reshape tensor into chunks for SSD algorithm
/// [B, L, ...] -> [B, L/chunk_size, chunk_size, ...]
fn reshape_into_chunks(input: &Tensor, pad_size: usize, chunk_size: usize) -> Result<Tensor> {
    let padded = pad_tensor_by_size(input, pad_size)?;
    let shape = padded.dims();

    match shape.len() {
        3 => {
            // [B, L, D] -> [B, n_chunks, chunk_size, D]
            let (b, l, d) = (shape[0], shape[1], shape[2]);
            padded.reshape(&[b, l / chunk_size, chunk_size, d])
        }
        4 => {
            // [B, L, H, D] -> [B, n_chunks, chunk_size, H, D]
            let (b, l, h, d) = (shape[0], shape[1], shape[2], shape[3]);
            padded.reshape(&[b, l / chunk_size, chunk_size, h, d])
        }
        _ => candle_core::bail!("reshape_into_chunks: unsupported tensor rank {}", shape.len()),
    }
}

/// Stable segment sum using cumsum difference method
/// Computes lower triangular matrix where L[i,j] = sum(input[j+1..=i])
/// Input: [..., chunk_size]
/// Output: [..., chunk_size, chunk_size]
fn segment_sum(input: &Tensor) -> Result<Tensor> {
    let shape = input.dims();
    let chunk_size = shape[shape.len() - 1];

    // Input validation
    if chunk_size == 0 {
        candle_core::bail!("segment_sum: chunk_size cannot be zero");
    }

    // Ensure input is contiguous before operations
    let input = input.contiguous()?;

    // 1. Compute cumulative sum along last dimension
    let cumsum = input.cumsum(D::Minus1)?;

    // 2. Compute difference matrix: cumsum[i] - cumsum[j]
    // Flatten batch dims for simpler processing
    let batch_dims: Vec<usize> = shape[..shape.len()-1].to_vec();
    let batch_size: usize = if batch_dims.is_empty() { 1 } else { batch_dims.iter().product() };

    // Reshape cumsum to [batch_size, chunk_size]
    let cumsum_flat = cumsum.reshape(&[batch_size, chunk_size])?;

    // For each batch element, compute outer difference: cumsum[i] - cumsum[j]
    // Using explicit loops to avoid broadcast contiguity issues in Candle
    let mut diff_rows = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let row = cumsum_flat.narrow(0, b, 1)?; // [1, chunk_size]
        let row_i = row.transpose(0, 1)?; // [chunk_size, 1]
        let row_j = row.clone(); // [1, chunk_size]
        // diff[i,j] = cumsum[i] - cumsum[j], clamped for numerical stability
        let diff = row_i.broadcast_sub(&row_j)?
            .clamp(MIN_EXP_INPUT, MAX_EXP_INPUT)?
            .contiguous()?; // [chunk_size, chunk_size]
        diff_rows.push(diff);
    }

    // Stack all batch elements
    let refs: Vec<&Tensor> = diff_rows.iter().collect();
    let diff = Tensor::stack(&refs, 0)?; // [batch_size, chunk_size, chunk_size]

    // 3. Apply lower triangular mask (j <= i)
    // Create mask: 1 where j <= i, 0 elsewhere
    let mask_data: Vec<f32> = (0..chunk_size)
        .flat_map(|i| (0..chunk_size).map(move |j| if j <= i { 1.0 } else { 0.0 }))
        .collect();
    let device = diff.device();
    let dtype = diff.dtype();
    let mask = Tensor::from_vec(mask_data, &[1, chunk_size, chunk_size], device)?.to_dtype(dtype)?;

    // 4. Apply mask: where mask=1 use diff, where mask=0 use large negative (exp -> ~0)
    let neg_large = Tensor::full(MASK_NEG_VALUE, &[1, chunk_size, chunk_size], device)?.to_dtype(dtype)?;

    // result = diff * mask + neg_large * (1 - mask)
    let masked_diff = diff.broadcast_mul(&mask)?;
    let one_minus_mask = mask.affine(-1.0, 1.0)?;
    let masked_neg = neg_large.broadcast_mul(&one_minus_mask)?;
    let result = masked_diff.broadcast_add(&masked_neg)?;

    // Reshape back to original batch shape: [..., chunk_size, chunk_size]
    let mut final_shape = batch_dims;
    final_shape.push(chunk_size);
    final_shape.push(chunk_size);
    result.reshape(final_shape.as_slice())
}

/// Gated RMS Normalization
/// Combines RMS normalization with gating mechanism (SiLU activation)
#[derive(Debug, Clone)]
pub struct MambaRMSNormGated {
    weight: Tensor,
    eps: f64,
}

impl MambaRMSNormGated {
    pub fn new(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(hidden_size, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, hidden_states: &Tensor, gate: Option<&Tensor>) -> Result<Tensor> {
        let dtype = hidden_states.dtype();
        let mut hs = hidden_states.to_dtype(DType::F32)?;

        // Apply gating if provided
        if let Some(g) = gate {
            let g_f32 = g.to_dtype(DType::F32)?;
            let g_silu = candle_nn::ops::silu(&g_f32)?;
            hs = hs.broadcast_mul(&g_silu)?;
        }

        // RMS normalization
        let variance = hs.sqr()?.mean_keepdim(D::Minus1)?;
        let eps_tensor = Tensor::new(&[self.eps as f32], hs.device())?;
        let var_eps = variance.broadcast_add(&eps_tensor)?;
        let hs_normed = hs.broadcast_div(&var_eps.sqrt()?)?;

        // Apply weight and convert back to original dtype
        let result = hs_normed.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?.to_dtype(dtype)?;
        Ok(result)
    }
}

/// Mamba2 Mixer - Core SSD (State Space Duality) implementation
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Mamba2Mixer {
    // Configuration
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    intermediate_size: usize,
    ssm_state_size: usize,
    chunk_size: usize,
    n_groups: usize,
    conv_kernel_size: usize,
    conv_dim: usize,
    time_step_min: f64,
    time_step_max: f64,

    // Learnable parameters
    in_proj: Linear,
    conv1d: Conv1d,
    dt_bias: Tensor,
    a_log: Tensor,
    d: Tensor,
    norm: MambaRMSNormGated,
    out_proj: Linear,
}

impl Mamba2Mixer {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        state_size: usize,
        chunk_size: usize,
        n_groups: usize,
        conv_kernel: usize,
        expand: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let intermediate_size = hidden_size * expand;
        let conv_dim = intermediate_size + 2 * n_groups * state_size;
        let projection_size = intermediate_size + conv_dim + num_heads;

        // Input projection
        let in_proj = linear(hidden_size, projection_size, vb.pp("in_proj"))?;

        // Depthwise Conv1D (groups = conv_dim for depthwise)
        let conv_cfg = Conv1dConfig {
            groups: conv_dim,
            padding: conv_kernel - 1,
            ..Default::default()
        };
        let conv1d = candle_nn::conv1d(conv_dim, conv_dim, conv_kernel, conv_cfg, vb.pp("conv1d"))?;

        // Time step bias (dt_bias)
        let dt_bias = vb.get(num_heads, "dt_bias")?;

        // S4D initialization for A: A_log initialized to log(1..num_heads)
        // This gives A = -exp(A_log) values that create stable exponential decay
        let a_log = if vb.contains_tensor("A_log") {
            // Load from checkpoint if available
            vb.get(num_heads, "A_log")?
        } else {
            // Initialize with S4D pattern: log(1), log(2), ..., log(num_heads)
            let a_init: Vec<f32> = (1..=num_heads)
                .map(|i| (i as f64).ln() as f32)
                .collect();
            let a_init_tensor = Tensor::from_vec(a_init, num_heads, vb.device())?
                .to_dtype(vb.dtype())?;  // Match model dtype
            // Store in varmap for training
            vb.get_with_hints(num_heads, "A_log", candle_nn::init::Init::Const(0.0))?
                .broadcast_add(&a_init_tensor)?
        };

        // D skip connection (residual multiplier per head)
        let d = vb.get(num_heads, "D")?;

        // Gated RMS norm
        let norm = MambaRMSNormGated::new(intermediate_size, EPS, vb.pp("norm"))?;

        // Output projection
        let out_proj = linear(intermediate_size, hidden_size, vb.pp("out_proj"))?;

        Ok(Self {
            num_heads,
            head_dim,
            hidden_size,
            intermediate_size,
            ssm_state_size: state_size,
            chunk_size,
            n_groups,
            conv_kernel_size: conv_kernel,
            conv_dim,
            time_step_min: 0.001,
            time_step_max: 0.1,
            in_proj,
            conv1d,
            dt_bias,
            a_log,
            d,
            norm,
            out_proj,
        })
    }

    /// Forward pass using SSD algorithm (batched matmul approach)
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (_batch_size, seq_len, _) = hidden_states.dims3()?;
        let dtype = hidden_states.dtype();

        // 1. Input projection
        let projected = self.in_proj.forward(hidden_states)?;

        // Split projections
        let d_mlp = (projected.dim(D::Minus1)? - 2 * self.intermediate_size
                    - 2 * self.n_groups * self.ssm_state_size - self.num_heads) / 2;

        let splits = vec![d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads];
        let parts = self.split_projection(&projected, &splits)?;
        let (_skip1, _skip2, gate, hidden_states_b_c, dt) = (
            &parts[0], &parts[1], &parts[2], &parts[3], &parts[4]
        );

        // 2. Conv1D (depthwise)
        // Transpose for conv: [B, L, C] -> [B, C, L]
        let hidden_states_b_c_t = hidden_states_b_c.transpose(1, 2)?;
        let conv_out = self.conv1d.forward(&hidden_states_b_c_t)?;

        // Crop to seq_len and transpose back
        let conv_out = conv_out.narrow(2, 0, seq_len)?;
        let hidden_states_b_c = conv_out.transpose(1, 2)?;

        // Apply SiLU activation and ensure dtype preserved
        let hidden_states_b_c = candle_nn::ops::silu(&hidden_states_b_c)?.to_dtype(dtype)?;

        // Split into hidden_states, B, C
        let hidden_states = hidden_states_b_c.narrow(D::Minus1, 0, self.intermediate_size)?;
        let b = hidden_states_b_c.narrow(
            D::Minus1,
            self.intermediate_size,
            self.n_groups * self.ssm_state_size
        )?;
        let c = hidden_states_b_c.narrow(
            D::Minus1,
            self.intermediate_size + self.n_groups * self.ssm_state_size,
            self.n_groups * self.ssm_state_size
        )?;

        // 3. SSM transformation (SSD algorithm)
        let y = self.ssd_forward(&hidden_states, &b, &c, dt)?.to_dtype(dtype)?;

        // 4. Gated normalization
        let scan_output = self.norm.forward(&y, Some(gate))?.to_dtype(dtype)?;

        // 5. Output projection
        let output = self.out_proj.forward(&scan_output)?;
        output.to_dtype(dtype)
    }

    /// Core SSD (State Space Duality) algorithm
    /// Uses model dtype for tensor core acceleration (BF16/F16/F32)
    /// Only exp() operations stay in F32 to prevent overflow
    fn ssd_forward(
        &self,
        hidden_states: &Tensor,
        b: &Tensor,
        c: &Tensor,
        dt: &Tensor,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let input_dtype = hidden_states.dtype();  // Store model dtype for tensor cores

        // Compute A (always negative for stability)
        // A = -exp(A_log), clamped for numerical stability
        // exp() stays in F32 to prevent overflow, then convert back to model dtype
        let a_exp = self.a_log.to_dtype(DType::F32)?.exp()?;
        let a_clamped = a_exp.clamp(1e-6_f32, 1e6_f32)?;
        let a = a_clamped.affine(-1.0, 0.0)?.to_dtype(input_dtype)?;

        // Apply softplus to dt and clamp
        // Note: softplus/exp might return F32 on some backends, so convert back to input dtype
        let dt = dt.broadcast_add(&self.dt_bias.to_dtype(dt.dtype())?.reshape(&[1, 1, self.num_heads])?)?;
        let dt = softplus(&dt, 1.0)?.to_dtype(input_dtype)?;
        let dt = dt.clamp(self.time_step_min as f32, self.time_step_max as f32)?;

        // Reshape for multi-head processing - use model dtype for tensor cores
        let hidden_states = hidden_states.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?;
        let mut b_reshaped = b.reshape(&[batch_size, seq_len, self.n_groups, self.ssm_state_size])?;
        let mut c_reshaped = c.reshape(&[batch_size, seq_len, self.n_groups, self.ssm_state_size])?;

        // Repeat B and C for all heads (group-wise)
        let repeat_factor = self.num_heads / self.n_groups;
        b_reshaped = self.repeat_interleave(&b_reshaped, repeat_factor, 2)?;
        c_reshaped = self.repeat_interleave(&c_reshaped, repeat_factor, 2)?;

        // Calculate padding for chunks
        let pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size;

        // D residual (skip connection)
        let d_residual = self.d.reshape(&[1, 1, self.num_heads, 1])?
            .broadcast_mul(&pad_tensor_by_size(&hidden_states, pad_size)?)?;

        // Discretize hidden_states and A
        let hidden_states = hidden_states.broadcast_mul(&dt.unsqueeze(D::Minus1)?)?;
        let a_discrete = a.broadcast_as(&[1, 1, self.num_heads])?.broadcast_mul(&dt)?;

        // Reshape into chunks
        let hidden_states_chunks = reshape_into_chunks(&hidden_states, pad_size, self.chunk_size)?;
        let a_chunks = reshape_into_chunks(&a_discrete, pad_size, self.chunk_size)?;
        let b_chunks = reshape_into_chunks(&b_reshaped, pad_size, self.chunk_size)?;
        let c_chunks = reshape_into_chunks(&c_reshaped, pad_size, self.chunk_size)?;

        // Permute A for processing: [B, n_chunks, chunk_size, num_heads] -> [B, num_heads, n_chunks, chunk_size]
        let a_chunks = a_chunks.permute((0, 3, 1, 2))?.contiguous()?;
        let a_cumsum = a_chunks.cumsum(D::Minus1)?;

        // 1. Intra-chunk computation (diagonal blocks) - 90% of computation
        // Clamp segment_sum before exp to prevent overflow/underflow
        let l_pre_exp = segment_sum(&a_chunks)?.clamp(MIN_EXP_INPUT, MAX_EXP_INPUT)?;
        let l = l_pre_exp.exp()?.contiguous()?;

        // Compute G (attention-like weights): C * B^T contracted over state dimension
        // Shape: [B, n_chunks, chunk_size, chunk_size, num_heads]
        let g = self.compute_g(&c_chunks, &b_chunks)?.contiguous()?;

        // Compute M: G * L (apply causal mask-like pattern)
        let l_perm = l.permute((0, 2, 3, 4, 1))?.contiguous()?; // [B, n_chunks, chunk_size, chunk_size, num_heads]
        let m = g.broadcast_mul(&l_perm)?.contiguous()?;

        // Compute Y_diag: apply M to hidden_states
        let y_diag = self.compute_y_diag(&m, &hidden_states_chunks)?;

        // 2. Compute states for each chunk (B terms of factorization)
        let states = self.compute_chunk_states(&a_cumsum, &b_chunks, &hidden_states_chunks)?;

        // 3. Inter-chunk recurrence (A terms) - 10% of computation, sequential
        let states_with_initial = self.add_initial_state(&states)?;
        let decay_chunk = self.compute_decay_chunk(&a_cumsum)?;
        let new_states = self.apply_inter_chunk_recurrence(&decay_chunk, &states_with_initial)?;
        let states = new_states.narrow(1, 0, new_states.dim(1)? - 1)?;

        // 4. State-to-output conversion (C terms)
        let y_off = self.compute_y_off(&c_chunks, &states, &a_cumsum)?;

        // Combine diagonal and off-diagonal contributions
        let mut y = y_diag.broadcast_add(&y_off)?;

        // Reshape back: [B, n_chunks, chunk_size, num_heads, head_dim] -> [B, seq_len, num_heads, head_dim]
        let total_len = seq_len + pad_size;
        y = y.reshape(&[batch_size, total_len, self.num_heads, self.head_dim])?;

        // Add D residual
        y = y.broadcast_add(&d_residual)?;

        // Crop padding
        if pad_size > 0 {
            y = y.narrow(1, 0, seq_len)?;
        }

        // Reshape to output: [B, seq_len, num_heads * head_dim]
        y = y.reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;

        Ok(y)
    }

    // Helper: Split projected tensor
    fn split_projection(&self, tensor: &Tensor, sizes: &[usize]) -> Result<Vec<Tensor>> {
        let mut result = Vec::new();
        let mut offset = 0;
        for &size in sizes {
            result.push(tensor.narrow(D::Minus1, offset, size)?);
            offset += size;
        }
        Ok(result)
    }

    // Helper: Repeat interleave (like PyTorch's repeat_interleave)
    fn repeat_interleave(&self, tensor: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
        let shape = tensor.dims();
        let mut result_shape = shape.to_vec();
        result_shape[dim] *= repeats;

        // Create indices for gathering
        let indices: Vec<u32> = (0..shape[dim])
            .flat_map(|i| std::iter::repeat(i as u32).take(repeats))
            .collect();
        let indices_tensor = Tensor::from_vec(indices, shape[dim] * repeats, tensor.device())?;

        tensor.index_select(&indices_tensor, dim)
    }

    // Helper: Compute G matrix (attention-like weights from B and C)
    fn compute_g(&self, c_chunks: &Tensor, b_chunks: &Tensor) -> Result<Tensor> {
        // c_chunks: [B, n_chunks, chunk_size, num_heads, state_size]
        // b_chunks: [B, n_chunks, chunk_size, num_heads, state_size]
        // output: [B, n_chunks, chunk_size, chunk_size, num_heads]

        let shape = c_chunks.dims();
        let (b, n_chunks, chunk_size, num_heads, state_size) = (
            shape[0], shape[1], shape[2], shape[3], shape[4]
        );

        // Reshape for batched matmul
        // C: [B*n_chunks*chunk_size, num_heads, state_size, 1]
        let c_flat = c_chunks.reshape(&[b * n_chunks * chunk_size, num_heads, state_size, 1])?;
        // B: [B*n_chunks*chunk_size, num_heads, 1, state_size]
        let b_flat = b_chunks.reshape(&[b * n_chunks * chunk_size, num_heads, 1, state_size])?;

        // Matmul: [num_heads, state_size, 1] @ [num_heads, 1, state_size] = [num_heads, state_size, state_size]
        let g_intermediate = c_flat.broadcast_mul(&b_flat)?;

        // Sum over state dimension
        let g = g_intermediate.sum(D::Minus1)?; // [B*n_chunks*chunk_size, num_heads, state_size]
        let g = g.sum(D::Minus1)?; // [B*n_chunks*chunk_size, num_heads]

        // Reshape back
        g.reshape(&[b, n_chunks, chunk_size, 1, num_heads])?
            .broadcast_as(&[b, n_chunks, chunk_size, chunk_size, num_heads])
    }

    // Helper: Compute Y_diag
    fn compute_y_diag(&self, m: &Tensor, hidden_states_chunks: &Tensor) -> Result<Tensor> {
        // m: [B, n_chunks, chunk_size, chunk_size, num_heads]
        // hidden_states: [B, n_chunks, chunk_size, num_heads, head_dim]
        // output: [B, n_chunks, chunk_size, num_heads, head_dim]

        // Avoid 6D broadcast_mul by using matmul approach
        // M is like attention weights: [B, n_chunks, L, S, num_heads]
        // HS is like values: [B, n_chunks, S, num_heads, head_dim]

        let shape = m.dims();
        let (b, n_chunks, l, s, num_heads) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
        let head_dim = hidden_states_chunks.dim(D::Minus1)?;

        // Reshape M: [B*n_chunks*num_heads, L, S]
        let m_reshaped = m.permute((0, 1, 4, 2, 3))?.contiguous()?
            .reshape(&[b * n_chunks * num_heads, l, s])?;

        // Reshape HS: [B*n_chunks*num_heads, S, head_dim]
        let hs_reshaped = hidden_states_chunks.permute((0, 1, 3, 2, 4))?.contiguous()?
            .reshape(&[b * n_chunks * num_heads, s, head_dim])?;

        // Matmul: [B*n_chunks*num_heads, L, S] @ [B*n_chunks*num_heads, S, head_dim]
        //       = [B*n_chunks*num_heads, L, head_dim]
        let result = m_reshaped.matmul(&hs_reshaped)?;

        // Reshape back: [B, n_chunks, num_heads, L, head_dim] -> [B, n_chunks, L, num_heads, head_dim]
        result.reshape(&[b, n_chunks, num_heads, l, head_dim])?
            .permute((0, 1, 3, 2, 4))
    }

    // Helper: Compute chunk states
    fn compute_chunk_states(
        &self,
        a_cumsum: &Tensor,
        b_chunks: &Tensor,
        hidden_states_chunks: &Tensor,
    ) -> Result<Tensor> {
        // Compute decay within chunks (clamp for numerical stability)
        let a_last = a_cumsum.narrow(D::Minus1, self.chunk_size - 1, 1)?;
        let decay_states_temp = a_last.broadcast_sub(&a_cumsum)?;
        let decay_states_clamped = decay_states_temp.clamp(MIN_EXP_INPUT, MAX_EXP_INPUT)?;
        let decay_states = decay_states_clamped.exp()?;

        // Permute decay for broadcasting
        let decay_perm = decay_states.permute((0, 2, 3, 1))?.contiguous()?;
        let b_decay = b_chunks.broadcast_mul(&decay_perm.unsqueeze(D::Minus1)?)?;

        // Compute states: [B, n_chunks, num_heads, head_dim, state_size]
        // b_decay: [B, n_chunks, chunk_size, num_heads, state_size]
        // hidden_states_chunks: [B, n_chunks, chunk_size, num_heads, head_dim]
        // Want: B.T @ X summed over chunk_size → [B, n_chunks, num_heads, state_size, head_dim]

        let shape = b_decay.dims();
        let (b, n_chunks, chunk_size, num_heads, state_size) = (
            shape[0], shape[1], shape[2], shape[3], shape[4]
        );
        let head_dim = hidden_states_chunks.dim(D::Minus1)?;

        // Reshape for batched matmul
        // b_decay: [B*n_chunks*num_heads, chunk_size, state_size]
        let b_reshaped = b_decay.permute((0, 1, 3, 2, 4))?.contiguous()?
            .reshape(&[b * n_chunks * num_heads, chunk_size, state_size])?;

        // hidden_states: [B*n_chunks*num_heads, chunk_size, head_dim]
        let hs_reshaped = hidden_states_chunks.permute((0, 1, 3, 2, 4))?.contiguous()?
            .reshape(&[b * n_chunks * num_heads, chunk_size, head_dim])?;

        // Matmul: b_decay.T @ hidden_states
        // [B*n_chunks*num_heads, state_size, chunk_size] @ [B*n_chunks*num_heads, chunk_size, head_dim]
        // = [B*n_chunks*num_heads, state_size, head_dim]
        let b_t = b_reshaped.transpose(1, 2)?;
        let states_flat = b_t.matmul(&hs_reshaped)?;

        // Reshape back: [B, n_chunks, num_heads, state_size, head_dim] -> [B, n_chunks, num_heads, head_dim, state_size]
        states_flat.reshape(&[b, n_chunks, num_heads, state_size, head_dim])?
            .transpose(D::Minus2, D::Minus1)
    }

    // Helper: Add initial zero state
    fn add_initial_state(&self, states: &Tensor) -> Result<Tensor> {
        let shape = states.dims();
        let zero_state = Tensor::zeros(&[shape[0], 1, shape[2], shape[3], shape[4]], states.dtype(), states.device())?;
        Tensor::cat(&[&zero_state, states], 1)
    }

    // Helper: Compute decay between chunks
    fn compute_decay_chunk(&self, a_cumsum: &Tensor) -> Result<Tensor> {
        // Extract last element of each chunk
        let a_last = a_cumsum.narrow(D::Minus1, self.chunk_size - 1, 1)?;

        // Pad for segment sum
        let a_last_squeezed = a_last.squeeze(D::Minus1)?;
        let a_last_padded = pad_1d(&a_last_squeezed, (1, 0), 2, 0.0)?;

        // Clamp before exp for numerical stability
        let decay_pre_exp = segment_sum(&a_last_padded)?.clamp(MIN_EXP_INPUT, MAX_EXP_INPUT)?;
        let decay_chunk = decay_pre_exp.exp()?;
        decay_chunk.permute((0, 2, 1, 3))?.contiguous() // Transpose for application
    }

    // Helper: Apply inter-chunk recurrence
    fn apply_inter_chunk_recurrence(&self, decay_chunk: &Tensor, states: &Tensor) -> Result<Tensor> {
        // decay_chunk: [B, num_heads, n_chunks+1, n_chunks+1] (segment_sum output)
        // states: [B, n_chunks+1, num_heads, head_dim, state_size]

        let states_shape = states.dims();
        let (b, n_chunks_plus_1, num_heads, head_dim, state_size) = (
            states_shape[0], states_shape[1], states_shape[2], states_shape[3], states_shape[4]
        );

        // decay_chunk is [B, num_heads, n_chunks+1, n_chunks+1]
        // We need to apply decay to states where:
        // new_state[i] = sum_j(decay[i,j] * state[j])

        // For batched matmul approach:
        // decay_perm: [B, n_chunks+1, n_chunks+1, num_heads]
        // states: [B, n_chunks+1, num_heads, head_dim, state_size]
        // output: [B, n_chunks+1, num_heads, head_dim, state_size]

        // Reshape for batched matmul:
        // decay: [B * num_heads, n_chunks+1, n_chunks+1]
        // states: [B * num_heads, n_chunks+1, head_dim * state_size]
        // result: [B * num_heads, n_chunks+1, head_dim * state_size]

        // Permute decay to [B, num_heads, n_chunks+1, n_chunks+1]
        let decay_for_matmul = decay_chunk.contiguous()?
            .reshape(&[b * num_heads, n_chunks_plus_1, n_chunks_plus_1])?;

        // Permute states to [B, num_heads, n_chunks+1, head_dim * state_size]
        let states_for_matmul = states.permute((0, 2, 1, 3, 4))?.contiguous()?
            .reshape(&[b * num_heads, n_chunks_plus_1, head_dim * state_size])?;

        // Matmul: [B*num_heads, n_chunks+1, n_chunks+1] @ [B*num_heads, n_chunks+1, head_dim*state_size]
        let result = decay_for_matmul.matmul(&states_for_matmul)?;

        // Reshape back: [B, num_heads, n_chunks+1, head_dim, state_size]
        result.reshape(&[b, num_heads, n_chunks_plus_1, head_dim, state_size])?
            .permute((0, 2, 1, 3, 4)) // -> [B, n_chunks+1, num_heads, head_dim, state_size]
    }

    // Helper: Compute Y_off
    fn compute_y_off(
        &self,
        c_chunks: &Tensor,
        states: &Tensor,
        a_cumsum: &Tensor,
    ) -> Result<Tensor> {
        // c_chunks: [B, n_chunks, chunk_size, num_heads, state_size]
        // states: [B, n_chunks, num_heads, head_dim, state_size] (narrowed from apply_inter_chunk_recurrence)
        // a_cumsum: [B, num_heads, n_chunks, chunk_size]
        // output: [B, n_chunks, chunk_size, num_heads, head_dim]

        let c_shape = c_chunks.dims();
        let (b, n_chunks, chunk_size, num_heads, state_size) = (
            c_shape[0], c_shape[1], c_shape[2], c_shape[3], c_shape[4]
        );
        let head_dim = states.dim(3)?;

        // Compute state decay: exp(a_cumsum), clamped for numerical stability
        let a_cumsum_clamped = a_cumsum.clamp(MIN_EXP_INPUT, MAX_EXP_INPUT)?;
        let state_decay = a_cumsum_clamped.exp()?.contiguous()?;

        // Use batched matmul for C @ states
        // C: [B, n_chunks, chunk_size, num_heads, state_size]
        // states: [B, n_chunks, num_heads, head_dim, state_size]
        // Want: C @ states.T -> [B, n_chunks, chunk_size, num_heads, head_dim]

        // Reshape for matmul:
        // C: [B*n_chunks*chunk_size*num_heads, 1, state_size]
        // states: [B*n_chunks*num_heads, state_size, head_dim]

        // First, C @ states.T for each (batch, chunk, position)
        // Actually we want: for each (b, chunk, pos), c[num_heads, state_size] @ states[b, chunk, num_heads, head_dim, state_size].T

        // Simpler approach: loop-free using einsum-like logic
        // c_chunks: [B, n_chunks, chunk_size, num_heads, state_size]
        // states: [B, n_chunks, num_heads, head_dim, state_size]
        // Expand states to [B, n_chunks, 1, num_heads, head_dim, state_size]
        // Expand c to [B, n_chunks, chunk_size, num_heads, 1, state_size]
        // Multiply and sum over state_size

        // Reshape for batched matmul
        // c: [B*n_chunks*num_heads, chunk_size, state_size]
        let c_reshaped = c_chunks.permute((0, 1, 3, 2, 4))?.contiguous()?
            .reshape(&[b * n_chunks * num_heads, chunk_size, state_size])?;

        // states: [B*n_chunks*num_heads, head_dim, state_size] -> transpose to [B*n_chunks*num_heads, state_size, head_dim]
        let states_reshaped = states.permute((0, 1, 2, 4, 3))?.contiguous()?
            .reshape(&[b * n_chunks * num_heads, state_size, head_dim])?;

        // Matmul: [B*n_chunks*num_heads, chunk_size, state_size] @ [B*n_chunks*num_heads, state_size, head_dim]
        //       = [B*n_chunks*num_heads, chunk_size, head_dim]
        let c_times_states = c_reshaped.matmul(&states_reshaped)?;

        // Reshape back: [B, n_chunks, num_heads, chunk_size, head_dim]
        let c_times_states = c_times_states.reshape(&[b, n_chunks, num_heads, chunk_size, head_dim])?
            .permute((0, 1, 3, 2, 4))?.contiguous()?; // -> [B, n_chunks, chunk_size, num_heads, head_dim]

        // Apply decay: state_decay is [B, num_heads, n_chunks, chunk_size]
        // Permute to [B, n_chunks, chunk_size, num_heads] and expand to [B, n_chunks, chunk_size, num_heads, 1]
        let decay_perm = state_decay.permute((0, 2, 3, 1))?.contiguous()?;

        // Expand decay for broadcasting: [B, n_chunks, chunk_size, num_heads, 1]
        let decay_expanded = decay_perm.unsqueeze(D::Minus1)?.contiguous()?;

        c_times_states.broadcast_mul(&decay_expanded)
    }
}

/// Mamba2 Block (combines norm + mixer + residual)
#[derive(Debug, Clone)]
pub struct Mamba2Block {
    norm: candle_nn::LayerNorm,
    mixer: Mamba2Mixer,
}

impl Mamba2Block {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        state_size: usize,
        chunk_size: usize,
        n_groups: usize,
        conv_kernel: usize,
        expand: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm = candle_nn::layer_norm(hidden_size, EPS, vb.pp("norm"))?;
        let mixer = Mamba2Mixer::new(
            hidden_size,
            num_heads,
            head_dim,
            state_size,
            chunk_size,
            n_groups,
            conv_kernel,
            expand,
            vb.pp("mixer"),
        )?;

        Ok(Self { norm, mixer })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let normalized = self.norm.forward(x)?;
        let output = self.mixer.forward(&normalized)?;
        residual.broadcast_add(&output)
    }
}
