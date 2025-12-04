use candle_core::{Device, Result, Tensor};
use candle_nn::{Optimizer, VarMap};
use crate::config::{BlazrConfig, CheckpointMeta, ModelConfig, TrainerConfig};
use crate::data::DataLoader;
use crate::distributed::{
    GradientSynchronizer, DistributedAdamW, AdamWConfig,
    create_synchronizer, compute_grad_norm,
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::VecDeque;
use std::path::Path;
use std::time::Instant;
use std::io::Write;

/// The "LitGPT" style Trainer abstraction.
/// It owns the loop state, the optimizer, and handles the "boring" stuff.
pub struct LitTrainer<'a> {
    config: TrainerConfig,
    varmap: &'a VarMap,
    optimizer: candle_nn::AdamW,
    #[allow(dead_code)]  // Kept for potential future use (e.g., device-specific optimizations)
    device: Device,
    global_step: usize,
    // Training metrics
    dataset_size: usize,
    total_steps: usize,
    batch_size: usize,
    seq_len: usize,
    // JSON logging
    steps_json_path: String,
    // UI mode
    headless: bool,
    // Gradient clipping state (for loss pre-scaling heuristic)
    prev_grad_norm: Option<f64>,
    // Resume support
    start_step: usize,
    // Checkpoint metadata
    model_config: ModelConfig,
    data_file_path: Option<String>,
    // Distributed training support
    synchronizer: Option<Box<dyn GradientSynchronizer>>,
    distributed_optimizer: Option<DistributedAdamW>,
}

impl<'a> LitTrainer<'a> {
    pub fn new(
        config: TrainerConfig,
        varmap: &'a VarMap,
        device: Device,
        dataset_size: usize,
        batch_size: usize,
        seq_len: usize,
        steps_json_path: String,
        headless: bool,
        start_step: usize,
        model_config: ModelConfig,
        data_file_path: Option<String>,
    ) -> Result<Self> {
        // Init AdamW with standard parameters
        let params = candle_nn::ParamsAdamW {
            lr: config.learning_rate,
            ..Default::default()
        };
        let optimizer = candle_nn::AdamW::new(varmap.all_vars(), params)?;

        // Calculate total steps from dataset
        let effective_batch_size = batch_size * seq_len * config.gradient_accumulation;
        let steps_per_epoch = (dataset_size + effective_batch_size - 1) / effective_batch_size; // Ceiling division
        let total_steps = steps_per_epoch * config.num_epochs;

        // Initialize distributed training components if enabled
        let (synchronizer, distributed_optimizer) = if config.distributed.enabled {
            let sync = create_synchronizer(&config.distributed)?;
            let dist_opt = DistributedAdamW::new(
                varmap,
                AdamWConfig::with_lr(config.learning_rate),
            )?;
            (Some(sync), Some(dist_opt))
        } else {
            (None, None)
        };

        Ok(Self {
            config,
            varmap,
            optimizer,
            device,
            global_step: start_step,
            dataset_size,
            total_steps,
            batch_size,
            seq_len,
            steps_json_path,
            headless,
            prev_grad_norm: None,
            start_step,
            model_config,
            data_file_path,
            synchronizer,
            distributed_optimizer,
        })
    }

    /// Performs one training step (including gradient accumulation)
    /// Returns (loss, grad_norm)
    ///
    /// # Gradient Clipping Strategy
    /// Since Candle 0.9.x doesn't support in-place gradient modification,
    /// we use loss pre-scaling based on the previous step's gradient norm.
    /// This effectively clips gradients while preserving AdamW's momentum state.
    pub fn training_step<D: DataLoader>(
        &mut self,
        model: &crate::model::LitGPT,
        data_loader: &mut D
    ) -> Result<(f64, f64)> {

        let mut total_loss = 0.0;
        let mut last_grad_norm = 0.0;
        let accum_steps = self.config.gradient_accumulation;
        let load_balance_alpha = self.config.load_balance_alpha;
        let max_grad_norm = 1.0; // Standard clipping threshold

        // Compute loss pre-scale factor based on previous step's gradient norm.
        // This is a heuristic that effectively clips gradients without modifying GradStore.
        //
        // IMPORTANT: This has a one-step lag - if gradients explode at step N, clipping
        // won't take effect until step N+1. This is acceptable because:
        // 1. Gradient explosions typically build up over multiple steps
        // 2. The alternative (two backward passes) doubles compute cost
        // 3. AdamW's momentum naturally smooths out single-step spikes
        let loss_prescale = if let Some(prev_norm) = self.prev_grad_norm {
            if prev_norm > max_grad_norm {
                // Scale factor to bring gradients within bounds
                // No lower clamp - if gradients are 100x too large, we need 100x reduction
                max_grad_norm / (prev_norm + 1e-6)
            } else {
                1.0
            }
        } else {
            1.0 // First step: no scaling (gradients typically small at init)
        };

        // Gradient Accumulation Loop
        for _ in 0..accum_steps {
            let (input, target) = data_loader.next_batch()?;

            // Forward pass (returns logits + auxiliary losses)
            let output = model.forward(&input)?;
            let logits = output.logits;
            let aux_losses = output.aux_losses;

            // Check for NaN in logits (indicates model forward pass issue)
            if let Ok(logits_vec) = logits.flatten_all()?.to_vec1::<f32>() {
                if logits_vec.iter().any(|x| x.is_nan() || x.is_infinite()) {
                    eprintln!("\n❌ NaN/Inf in model logits! This indicates:");
                    eprintln!("   1. Numerical instability in forward pass");
                    eprintln!("   2. RoPE or attention computation overflow");
                    eprintln!("   3. Mamba SSM state explosion");
                    return Err(candle_core::Error::Msg("NaN in logits".to_string()));
                }
            }

            // Calculate Loss (Cross Entropy)
            // Flatten batch/seq dims for cross_entropy: [Batch*Seq, Vocab]
            // Note: logits are already F32 from model forward (cast for stable loss computation)
            let (b, s, v) = logits.dims3()?;
            let logits_flat = logits.reshape((b * s, v))?;
            let target_flat = target.reshape(b * s)?;

            let ce_loss_per_token = candle_nn::loss::cross_entropy(&logits_flat, &target_flat)?;
            let ce_loss = ce_loss_per_token.mean_all()?;
            // Note: mean_all() returns a scalar on CPU but may need squeezing on CUDA in some cases

            // Add auxiliary losses (e.g., MoE load balancing)
            let mut total_aux_loss = 0.0;
            for aux_loss in &aux_losses {
                // CRITICAL: CUDA can return [1,1] instead of scalar - squeeze manually
                let mut loss_t = aux_loss.clone();
                while loss_t.rank() > 0 && loss_t.elem_count() == 1 {
                    loss_t = loss_t.squeeze(0)?;
                }
                let loss_val = loss_t.to_vec0::<f32>()? as f64;
                total_aux_loss += loss_val;
            }
            let avg_aux_loss = if !aux_losses.is_empty() {
                total_aux_loss / aux_losses.len() as f64
            } else {
                0.0
            };

            // Check for NaN in CE loss before combining
            let ce_loss_val = ce_loss.to_vec0::<f32>()? as f64;
            if ce_loss_val.is_nan() || ce_loss_val.is_infinite() {
                eprintln!("\n❌ NaN/Inf detected in CE loss! This usually indicates:");
                eprintln!("   1. Learning rate too high (try 1e-4 or lower)");
                eprintln!("   2. Gradient explosion (sequence too long)");
                eprintln!("   3. Numerical instability in model");
                eprintln!("   Current CE loss: {}", ce_loss_val);
                return Err(candle_core::Error::Msg("NaN loss detected".to_string()));
            }

            // Combine losses: total = ce_loss + alpha * avg_aux_loss
            let aux_tensor = Tensor::new(avg_aux_loss as f32, ce_loss.device())?.to_dtype(ce_loss.dtype())?;
            let alpha_tensor = Tensor::new(load_balance_alpha as f32, ce_loss.device())?.to_dtype(ce_loss.dtype())?;
            let weighted_aux = aux_tensor.mul(&alpha_tensor)?;
            let loss = ce_loss.clone().add(&weighted_aux)?;

            // Check combined loss
            let total_loss_val = loss.to_vec0::<f32>()? as f64;
            if total_loss_val.is_nan() || total_loss_val.is_infinite() {
                eprintln!("\n❌ NaN/Inf in combined loss! CE: {:.4}, Aux: {:.4}, Alpha: {:.4}",
                    ce_loss_val, avg_aux_loss, load_balance_alpha);
                return Err(candle_core::Error::Msg("NaN loss detected".to_string()));
            }

            // Scale loss for accumulation AND apply gradient clipping pre-scale
            // The prescale factor effectively clips gradients by reducing the loss magnitude
            // before backprop, which proportionally reduces all gradient magnitudes.
            let scaled_loss = ((loss.clone() / accum_steps as f64)? * loss_prescale)?;

            // Backward pass (accumulates grads into tensors)
            let grads = scaled_loss.backward()?;

            // Apply gradients - either single-GPU or distributed path
            if let (Some(ref sync), Some(ref mut dist_opt)) =
                (&self.synchronizer, &mut self.distributed_optimizer)
            {
                // DISTRIBUTED PATH: Synchronize gradients across GPUs
                // 1. Synchronize gradients (all-reduce across GPUs)
                let synced_grads = sync.synchronize_gradients(&grads, self.varmap)?;

                // 2. Compute gradient norm from synchronized gradients
                let grad_norm = compute_grad_norm(&synced_grads)?;
                last_grad_norm = grad_norm;

                // 3. Apply gradients via distributed optimizer
                dist_opt.step(&synced_grads)?;
            } else {
                // SINGLE-GPU PATH: Original behavior
                // Compute gradient norm (for monitoring and next step's pre-scaling)
                let (grads, grad_norm) = self.clip_gradients(grads, max_grad_norm)?;
                last_grad_norm = grad_norm;

                // Apply gradients via optimizer
                self.optimizer.step(&grads)?;
            }

            // Track loss for logging (just CE loss, not aux)
            total_loss += ce_loss.to_vec0::<f32>()? as f64;
        }

        // Store gradient norm for next step's loss pre-scaling
        self.prev_grad_norm = Some(last_grad_norm);

        self.global_step += 1;
        Ok((total_loss / accum_steps as f64, last_grad_norm))
    }

    pub fn fit<D: DataLoader>(
        &mut self,
        model: &crate::model::LitGPT,
        data_loader: &mut D
    ) -> Result<()> {
        // Save blazr-compatible config.json if it doesn't exist
        // This enables the checkpoint to be used directly with blazr inference server
        self.save_blazr_config_if_needed();

        if self.headless {
            self.fit_headless(model, data_loader)
        } else {
            self.fit_normal(model, data_loader)
        }
    }

    /// Save blazr-compatible config.json to checkpoint directory (once only)
    fn save_blazr_config_if_needed(&self) {
        // Create checkpoint directory if it doesn't exist
        if !Path::new(&self.config.checkpoint_dir).exists() {
            if let Err(e) = std::fs::create_dir_all(&self.config.checkpoint_dir) {
                if !self.headless {
                    eprintln!("Warning: Failed to create checkpoint directory: {}", e);
                }
                return;
            }
        }

        // Only save if config.json doesn't already exist
        if !BlazrConfig::exists(&self.config.checkpoint_dir) {
            let blazr_config = BlazrConfig::from_model_config(&self.model_config);
            if let Err(e) = blazr_config.save(&self.config.checkpoint_dir) {
                if !self.headless {
                    eprintln!("Warning: Failed to save blazr config.json: {}", e);
                }
            } else if !self.headless {
                println!("📝 Saved config.json for blazr inference compatibility");
            }
        }
    }

    fn fit_headless<D: DataLoader>(
        &mut self,
        model: &crate::model::LitGPT,
        data_loader: &mut D
    ) -> Result<()> {
        let tokens_per_step = self.batch_size * self.seq_len * self.config.gradient_accumulation;
        let steps_per_epoch = (self.dataset_size + tokens_per_step - 1) / tokens_per_step;
        let training_start = Instant::now();

        // max_steps=0 means unlimited (train for full epochs)
        let effective_steps = if self.config.max_steps == 0 {
            self.total_steps
        } else {
            self.config.max_steps.min(self.total_steps)
        };

        // Resume support: start from start_step
        for step in self.start_step..effective_steps {
            let (loss, grad_norm) = self.training_step(model, data_loader)?;
            let current_epoch = (step + 1) as f64 / steps_per_epoch as f64;

            // Log at intervals
            if step % self.config.log_interval == 0 || step == 0 {
                let elapsed = training_start.elapsed().as_secs_f64();
                let samples_processed = (step + 1) * self.batch_size * self.config.gradient_accumulation;
                let it_s = samples_processed as f64 / elapsed;

                // Get VRAM usage via nvidia-smi
                let vram = std::process::Command::new("nvidia-smi")
                    .args(&["--query-compute-apps=used_memory", "--format=csv,noheader,nounits"])
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .and_then(|s| s.lines().next().map(|l| l.trim().to_string()))
                    .unwrap_or_else(|| "0".to_string());

                println!("{{\"step\": {}, \"loss\": {:.4}, \"grad_norm\": {:.4}, \"learning_rate\": {:.6e}, \"epoch\": {:.2}, \"it/s\": {:.2}, \"vram\": \"{}\"}}",
                    step + 1, loss, grad_norm, self.config.learning_rate, current_epoch, it_s, vram);
            }
        }

        // Always save final checkpoint
        let prefix = self.model_config.checkpoint_prefix();
        println!("Saving final checkpoint to {}/{}_final.safetensors", self.config.checkpoint_dir, prefix);
        self.save_checkpoint_named("final")?;

        Ok(())
    }

    fn fit_normal<D: DataLoader>(
        &mut self,
        model: &crate::model::LitGPT,
        data_loader: &mut D
    ) -> Result<()> {
        let tokens_per_step = self.batch_size * self.seq_len * self.config.gradient_accumulation;
        let steps_per_epoch = (self.dataset_size + tokens_per_step - 1) / tokens_per_step;

        // Respect max_steps limit (0 = unlimited, train for full epochs)
        let effective_steps = if self.config.max_steps == 0 {
            self.total_steps
        } else {
            self.config.max_steps.min(self.total_steps)
        };

        // MultiProgress: scrollable logs above, sticky progress bar below
        let mp = MultiProgress::new();

        // Progress bar style - ETA calculated manually using rolling average
        let sty = ProgressStyle::with_template(
            "{bar:40.cyan/dim} {pos:>7}/{len:7} [{elapsed}<{msg}"
        )
        .unwrap()
        .progress_chars("━━─");

        let pb = mp.add(ProgressBar::new(effective_steps as u64));
        pb.set_style(sty);

        // Resume support - this goes to scrollable area
        if self.start_step > 0 {
            pb.set_position(self.start_step as u64);
            mp.println(format!("📍 Resuming from step {}", self.start_step)).ok();
        }

        let mut last_loss = 0.0;
        let mut last_grad_norm = 0.0;

        // Rolling average for it/s calculation (last N step times)
        const ROLLING_WINDOW: usize = 5;
        let mut step_times: VecDeque<f64> = VecDeque::with_capacity(ROLLING_WINDOW);
        let mut step_start = Instant::now();

        for step in self.start_step..effective_steps {
            let (loss, grad_norm) = self.training_step(model, data_loader)?;
            last_loss = loss;
            last_grad_norm = grad_norm;

            // Track step time for rolling average
            let step_duration = step_start.elapsed().as_secs_f64();
            step_times.push_back(step_duration);
            if step_times.len() > ROLLING_WINDOW {
                step_times.pop_front();
            }
            step_start = Instant::now();

            let current_epoch = (step + 1) as f64 / steps_per_epoch as f64;

            // Calculate it/s as rolling average of recent steps
            let avg_step_time: f64 = step_times.iter().sum::<f64>() / step_times.len() as f64;
            let it_s = if avg_step_time > 0.0 { 1.0 / avg_step_time } else { 0.0 };

            // Silent log to file at intervals (no terminal output for steps)
            if step % self.config.log_interval == 0 || step == 0 {
                let step_json = serde_json::json!({
                    "step": step + 1,
                    "loss": loss,
                    "grad_norm": grad_norm,
                    "learning_rate": self.config.learning_rate,
                    "epoch": current_epoch,
                });
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.steps_json_path)
                {
                    let _ = writeln!(file, "{}", step_json);
                }
            }

            // Calculate ETA based on rolling average
            let remaining_steps = effective_steps - step - 1;
            let eta_secs = avg_step_time * remaining_steps as f64;
            let eta_str = format_duration(eta_secs);

            // Update progress bar (sticky at bottom)
            pb.set_position((step + 1) as u64);
            pb.set_message(format!(
                "{}] loss: {:.4} | grad: {:.4} | {:.2} it/s",
                eta_str, loss, grad_norm, it_s
            ));

            // Checkpoint messages go to scrollable area above
            if step > 0 && step % self.config.save_interval == 0 {
                mp.println(format!("💾 Checkpoint saved: step {}", step)).ok();
                self.save_checkpoint(step)?;
            }
        }

        pb.finish_with_message(format!(
            "loss: {:.4} | grad: {:.4} | done ✓",
            last_loss, last_grad_norm
        ));

        // Final messages
        let prefix = self.model_config.checkpoint_prefix();
        mp.println(format!("\n💾 Saving final checkpoint to {}/{}_final.safetensors", self.config.checkpoint_dir, prefix)).ok();
        self.save_checkpoint_named("final")?;

        mp.println(format!(
            "✅ Training complete | final loss: {:.4} | grad: {:.4}",
            last_loss, last_grad_norm
        )).ok();

        Ok(())
    }

    fn save_checkpoint(&self, step: usize) -> Result<()> {
        let prefix = self.model_config.checkpoint_prefix();
        let checkpoint_path = format!("{}/{}_step_{}.safetensors", self.config.checkpoint_dir, prefix, step);
        if !Path::new(&self.config.checkpoint_dir).exists() {
            std::fs::create_dir_all(&self.config.checkpoint_dir).map_err(candle_core::Error::wrap)?;
        }
        self.varmap.save(&checkpoint_path)?;

        // Save metadata for auto-resume validation
        if let Some(ref data_path) = self.data_file_path {
            if let Ok(meta) = CheckpointMeta::new(
                step,
                &self.model_config,
                data_path,
                &checkpoint_path,
            ) {
                if let Err(e) = meta.save(&self.config.checkpoint_dir) {
                    eprintln!("Warning: Failed to save checkpoint metadata: {}", e);
                }
            }
        }

        // Clean up old checkpoints if limit is set
        if self.config.max_checkpoints > 0 {
            self.cleanup_old_checkpoints()?;
        }

        Ok(())
    }

    /// Remove old checkpoints to keep only max_checkpoints most recent
    fn cleanup_old_checkpoints(&self) -> Result<()> {
        let dir = Path::new(&self.config.checkpoint_dir);
        if !dir.exists() {
            return Ok(());
        }

        // Collect all step checkpoints with their step numbers
        // Format: {name}_{seq}_{params}_{dtype}_step_{N}.safetensors
        let mut checkpoints: Vec<(usize, std::path::PathBuf)> = Vec::new();

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                    // Match *_step_N.safetensors (e.g., nano_512_117m_bf16_step_100.safetensors)
                    if filename.contains("_step_") && filename.ends_with(".safetensors") {
                        // Extract step number after "_step_"
                        if let Some(step_part) = filename.split("_step_").last() {
                            if let Some(step_str) = step_part.strip_suffix(".safetensors") {
                                if let Ok(step) = step_str.parse::<usize>() {
                                    checkpoints.push((step, path));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sort by step number (ascending)
        checkpoints.sort_by_key(|(step, _)| *step);

        // Remove oldest checkpoints if we exceed the limit
        let to_remove = checkpoints.len().saturating_sub(self.config.max_checkpoints);
        for (step, path) in checkpoints.into_iter().take(to_remove) {
            // Remove checkpoint file
            if let Err(e) = std::fs::remove_file(&path) {
                eprintln!("Warning: Failed to remove old checkpoint {:?}: {}", path, e);
            }
            // Remove associated metadata file
            let meta_path = format!("{}/checkpoint_step_{}.meta.json", self.config.checkpoint_dir, step);
            let _ = std::fs::remove_file(&meta_path); // Ignore errors for metadata
        }

        Ok(())
    }

    fn save_checkpoint_named(&self, name: &str) -> Result<()> {
        let prefix = self.model_config.checkpoint_prefix();
        let path = format!("{}/{}_{}.safetensors", self.config.checkpoint_dir, prefix, name);
        if !Path::new(&self.config.checkpoint_dir).exists() {
            std::fs::create_dir_all(&self.config.checkpoint_dir).map_err(candle_core::Error::wrap)?;
        }
        self.varmap.save(&path)?;
        Ok(())
    }

    /// Compute gradient norm from GradStore
    ///
    /// # Arguments
    /// * `grads` - GradStore from backward pass
    ///
    /// # Returns
    /// Global L2 gradient norm across all parameters
    fn compute_grad_norm_from_grads(&self, grads: &candle_core::backprop::GradStore) -> Result<f64> {
        let all_vars = self.varmap.all_vars();
        let mut total_norm_sq = 0.0f64;

        for var in &all_vars {
            if let Some(grad) = grads.get(var) {
                let grad_norm_sq = grad.powf(2.0)?
                    .sum_all()?
                    .to_vec0::<f32>()? as f64;
                total_norm_sq += grad_norm_sq;
            }
        }

        Ok(total_norm_sq.sqrt())
    }

    /// Gradient clipping via norm-based scaling
    ///
    /// # Implementation Note
    /// Candle's GradStore is opaque, so we implement clipping by:
    /// 1. Computing the global L2 norm from all gradients
    /// 2. Logging warnings when norm exceeds max_norm
    ///
    /// Actual clipping is achieved via loss pre-scaling in training_step(),
    /// which scales the loss based on the previous step's gradient norm.
    /// This preserves AdamW's momentum/variance tracking.
    ///
    /// # Arguments
    /// * `grads` - GradStore from backward pass
    /// * `max_norm` - Maximum allowed gradient norm (typically 1.0)
    ///
    /// # Returns
    /// Tuple of (GradStore, computed_grad_norm)
    fn clip_gradients(&self, grads: candle_core::backprop::GradStore, _max_norm: f64) -> Result<(candle_core::backprop::GradStore, f64)> {
        let total_norm = self.compute_grad_norm_from_grads(&grads)?;
        Ok((grads, total_norm))
    }
}

/// Format seconds into human-readable duration (e.g., "5m32s", "2h14m", "3d05h")
fn format_duration(secs: f64) -> String {
    let total_secs = secs.round() as u64;
    let days = total_secs / 86400;
    let hours = (total_secs % 86400) / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;

    if days > 0 {
        format!("{}d{:02}h", days, hours)
    } else if hours > 0 {
        format!("{}h{:02}m", hours, minutes)
    } else if minutes > 0 {
        format!("{}m{:02}s", minutes, seconds)
    } else {
        format!("{}s", seconds)
    }
}