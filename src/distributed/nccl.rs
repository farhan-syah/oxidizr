//! NCCL-based gradient synchronization backend
//!
//! This module provides high-performance gradient synchronization using NVIDIA's
//! NCCL (NVIDIA Collective Communications Library) for all-reduce operations.
//!
//! # Requirements
//!
//! - CUDA 12.x
//! - NCCL library installed on the system
//! - Compile with `--features nccl`
//!
//! # Performance
//!
//! NCCL provides optimal bandwidth utilization for GPU-to-GPU communication:
//! - NVLink: up to 600 GB/s bidirectional
//! - PCIe: ~12-32 GB/s depending on generation
//!
//! For 4+ GPUs, NCCL is significantly faster than CPU staging.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarMap;
use cudarc::driver::{CudaContext, CudaStream};
use cudarc::nccl::{Comm, ReduceOp};
use std::sync::{Arc, Mutex};

use super::sync::{GradientSynchronizer, SyncedGradient};

/// NCCL-based gradient synchronizer
///
/// Uses NCCL all-reduce operations for high-performance gradient synchronization
/// across multiple GPUs.
///
/// # Thread Safety
///
/// This struct wraps NCCL communicators in a Mutex for thread safety. While the
/// GradientSynchronizer trait requires Send+Sync, NCCL operations are serialized
/// to ensure correctness.
pub struct NcclSynchronizer {
    /// GPU devices participating in training (Candle devices)
    devices: Vec<Device>,
    /// cudarc CUDA streams (hold references to keep contexts alive)
    streams: Vec<Arc<CudaStream>>,
    /// NCCL communicators (wrapped in Mutex for thread safety)
    comms: Mutex<Vec<Comm>>,
    /// Number of GPUs
    world_size: usize,
}

// SAFETY: The Mutex ensures that NCCL operations are serialized
unsafe impl Send for NcclSynchronizer {}
unsafe impl Sync for NcclSynchronizer {}

impl NcclSynchronizer {
    /// Create a new NCCL synchronizer for the specified GPUs
    ///
    /// # Arguments
    ///
    /// * `gpu_ids` - GPU ordinal IDs to use (e.g., [0, 1, 2, 3])
    ///
    /// # Returns
    ///
    /// New synchronizer instance or error if initialization fails
    pub fn new(gpu_ids: &[usize]) -> Result<Self> {
        let world_size = gpu_ids.len();
        if world_size < 2 {
            return Err(candle_core::Error::Msg(
                "NcclSynchronizer requires at least 2 GPUs. Use CpuStagingSynchronizer for single GPU.".to_string()
            ));
        }

        // Create Candle devices
        let devices: Vec<Device> = gpu_ids
            .iter()
            .map(|&id| Device::new_cuda(id))
            .collect::<Result<Vec<_>>>()?;

        // Create cudarc CUDA contexts and streams
        let streams: Vec<Arc<CudaStream>> = gpu_ids
            .iter()
            .map(|&id| {
                CudaContext::new(id)
                    .map(|ctx| ctx.default_stream())
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create CUDA context: {:?}", e)))?;

        // Initialize NCCL communicators from streams
        let comms = Comm::from_devices(streams.clone())
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create NCCL comms: {:?}", e)))?;

        log::info!(
            "NCCL synchronizer initialized with {} GPUs: {:?}",
            world_size,
            gpu_ids
        );

        Ok(Self {
            devices,
            streams,
            comms: Mutex::new(comms),
            world_size,
        })
    }

    /// Get the primary device (GPU 0)
    #[allow(dead_code)]
    pub fn primary_device(&self) -> &Device {
        &self.devices[0]
    }

    /// Perform NCCL all-reduce on a tensor using rank 0's communicator
    ///
    /// This sums the tensor values across all GPUs and divides by world_size.
    fn all_reduce_tensor(&self, tensor: &Tensor, comms: &[Comm]) -> Result<Tensor> {
        let dtype = tensor.dtype();

        // Flatten tensor for all-reduce
        let flat = tensor.flatten_all()?;

        // Get raw data and perform all-reduce based on dtype
        match dtype {
            DType::F32 => {
                let data: Vec<f32> = flat.to_vec1()?;
                let reduced = self.all_reduce_f32(&data, comms)?;
                Tensor::from_vec(reduced, tensor.shape(), tensor.device())
            }
            DType::F16 => {
                // Convert to f32 for all-reduce, then back
                let f32_tensor = flat.to_dtype(DType::F32)?;
                let data: Vec<f32> = f32_tensor.to_vec1()?;
                let reduced = self.all_reduce_f32(&data, comms)?;
                let result = Tensor::from_vec(reduced, tensor.shape(), tensor.device())?;
                result.to_dtype(DType::F16)
            }
            DType::BF16 => {
                // Convert to f32 for all-reduce, then back
                let f32_tensor = flat.to_dtype(DType::F32)?;
                let data: Vec<f32> = f32_tensor.to_vec1()?;
                let reduced = self.all_reduce_f32(&data, comms)?;
                let result = Tensor::from_vec(reduced, tensor.shape(), tensor.device())?;
                result.to_dtype(DType::BF16)
            }
            _ => Err(candle_core::Error::Msg(format!(
                "NCCL all-reduce not supported for dtype {:?}",
                dtype
            ))),
        }
    }

    /// All-reduce f32 data using NCCL
    ///
    /// Uses rank 0's stream and communicator for the operation.
    fn all_reduce_f32(&self, data: &[f32], comms: &[Comm]) -> Result<Vec<f32>> {
        // Use the first GPU's stream and communicator
        let stream = &self.streams[0];
        let comm = &comms[0];

        // Copy data to GPU
        let gpu_data = stream
            .clone_htod(data)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to copy to GPU: {:?}", e)))?;

        // Allocate output buffer
        let mut output = stream
            .alloc_zeros::<f32>(data.len())
            .map_err(|e| candle_core::Error::Msg(format!("Failed to allocate output: {:?}", e)))?;

        // Perform all-reduce (sum across all GPUs)
        comm.all_reduce(&gpu_data, &mut output, &ReduceOp::Sum)
            .map_err(|e| candle_core::Error::Msg(format!("NCCL all_reduce failed: {:?}", e)))?;

        // Copy result back to CPU
        let mut result: Vec<f32> = stream
            .clone_dtoh(&output)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to copy from GPU: {:?}", e)))?;

        // Average by dividing by world size
        let scale = 1.0 / self.world_size as f32;
        for v in &mut result {
            *v *= scale;
        }

        Ok(result)
    }
}

impl GradientSynchronizer for NcclSynchronizer {
    fn synchronize_gradients(
        &self,
        grads: &candle_core::backprop::GradStore,
        varmap: &VarMap,
    ) -> Result<Vec<SyncedGradient>> {
        let data = varmap.data().lock().unwrap();
        let mut result = Vec::with_capacity(data.len());

        // Lock the communicators for the duration of synchronization
        let comms = self.comms.lock().unwrap();

        for (name, var) in data.iter() {
            if let Some(grad) = grads.get(var) {
                // Perform NCCL all-reduce to average gradients
                let synced_grad = self.all_reduce_tensor(&grad, &comms)?;
                result.push((name.clone(), var.clone(), synced_grad));
            }
        }

        Ok(result)
    }

    fn local_rank(&self) -> usize {
        0 // Primary GPU in single-process mode
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn barrier(&self) -> Result<()> {
        // NCCL doesn't have an explicit barrier, but we can synchronize
        // by doing a dummy all-reduce on a small tensor
        let comms = self.comms.lock().unwrap();
        let stream = &self.streams[0];
        let comm = &comms[0];

        let data = stream
            .alloc_zeros::<f32>(1)
            .map_err(|e| candle_core::Error::Msg(format!("Barrier alloc failed: {:?}", e)))?;
        let mut output = stream
            .alloc_zeros::<f32>(1)
            .map_err(|e| candle_core::Error::Msg(format!("Barrier alloc failed: {:?}", e)))?;
        comm.all_reduce(&data, &mut output, &ReduceOp::Sum)
            .map_err(|e| candle_core::Error::Msg(format!("Barrier failed: {:?}", e)))?;

        Ok(())
    }
}
