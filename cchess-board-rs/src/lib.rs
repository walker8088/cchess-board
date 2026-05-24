pub mod base_onnx;
pub mod classifier;
pub mod detector;
pub mod rtmpose;

#[cfg(feature = "py")]
pub mod python;

pub use detector::ChessboardDetector;
