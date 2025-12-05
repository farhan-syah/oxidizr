//! CLI subcommands for oxidizr
//!
//! This module contains the implementation of all CLI subcommands:
//! - `train` - Train a model (default command)
//! - `pack` - Package a trained model for distribution
//! - `push` - Push a packaged model to HuggingFace (feature-gated)

pub mod train;
pub mod pack;

#[cfg(feature = "huggingface")]
pub mod push;

pub use train::{TrainArgs, run_train};
pub use pack::{PackArgs, run_pack};

#[cfg(feature = "huggingface")]
pub use push::{PushArgs, run_push};
