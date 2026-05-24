use std::path::Path;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum BaseOnnxError {
    #[error("ONNX runtime error: {0}")]
    OnnxError(String),

    #[error("Image load error: {0}")]
    ImageError(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimMismatch { expected: String, actual: String },
}

/// BGR/RGB image stored as contiguous [H][W][C] u8 values.
#[derive(Clone, Debug)]
pub struct Image {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

impl Image {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            data: vec![0u8; width * height * 3],
            width,
            height,
        }
    }

    pub fn from_vec(data: Vec<u8>, width: usize, height: usize) -> Result<Self, BaseOnnxError> {
        if data.len() != width * height * 3 {
            return Err(BaseOnnxError::ImageError(format!(
                "Data length {} does not match {}x{}x3",
                data.len(),
                width,
                height
            )));
        }
        Ok(Self {
            data,
            width,
            height,
        })
    }

    #[inline]
    pub fn get_pixel(&self, x: usize, y: usize) -> [u8; 3] {
        let idx = (y * self.width + x) * 3;
        [self.data[idx], self.data[idx + 1], self.data[idx + 2]]
    }

    #[inline]
    pub fn set_pixel(&mut self, x: usize, y: usize, v: [u8; 3]) {
        let idx = (y * self.width + x) * 3;
        self.data[idx] = v[0];
        self.data[idx + 1] = v[1];
        self.data[idx + 2] = v[2];
    }

    /// Load image from file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, BaseOnnxError> {
        let img = image::open(path).map_err(|e| BaseOnnxError::ImageError(e.to_string()))?;
        let rgb = img.to_rgb8();
        let (w, h) = rgb.dimensions();
        // Convert RGB to BGR
        let data: Vec<u8> = rgb
            .as_raw()
            .chunks(3)
            .flat_map(|p| [p[2], p[1], p[0]])
            .collect();
        Ok(Self {
            data,
            width: w as usize,
            height: h as usize,
        })
    }

    /// Save as RGB image (auto-detects format from extension).
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), BaseOnnxError> {
        let path = path.as_ref();
        let mut rgb_data = Vec::with_capacity(self.data.len());
        for chunk in self.data.chunks(3) {
            rgb_data.extend_from_slice(&[chunk[2], chunk[1], chunk[0]]);
        }
        let rgb = image::RgbImage::from_raw(self.width as u32, self.height as u32, rgb_data)
            .ok_or_else(|| BaseOnnxError::ImageError("Failed to create image buffer".into()))?;
        rgb.save(path)
            .map_err(|e| BaseOnnxError::ImageError(e.to_string()))
    }

    /// BGR to RGB conversion.
    pub fn bgr_to_rgb(&self) -> Self {
        let data = self
            .data
            .chunks(3)
            .flat_map(|p| [p[2], p[1], p[0]])
            .collect();
        Self {
            data,
            width: self.width,
            height: self.height,
        }
    }

    /// RGB to BGR conversion.
    pub fn rgb_to_bgr(&self) -> Self {
        self.bgr_to_rgb()
    }

    /// Crop region from image.
    pub fn crop(&self, x: usize, y: usize, w: usize, h: usize) -> Self {
        let x0 = x.min(self.width);
        let y0 = y.min(self.height);
        let w = w.min(self.width - x0);
        let h = h.min(self.height - y0);
        let mut data = Vec::with_capacity(w * h * 3);
        for row in y0..y0 + h {
            let row_start = (row * self.width + x0) * 3;
            let row_end = row_start + w * 3;
            data.extend_from_slice(&self.data[row_start..row_end]);
        }
        Self {
            data,
            width: w,
            height: h,
        }
    }

    /// Resize with bilinear interpolation.
    pub fn resize(&self, new_w: usize, new_h: usize) -> Self {
        let mut out = Self::new(new_w, new_h);
        let scale_x = self.width as f32 / new_w as f32;
        let scale_y = self.height as f32 / new_h as f32;

        for y in 0..new_h {
            for x in 0..new_w {
                let src_x = (x as f32 + 0.5) * scale_x - 0.5;
                let src_y = (y as f32 + 0.5) * scale_y - 0.5;

                let x0 = src_x.floor() as usize;
                let y0 = src_y.floor() as usize;
                let x1 = (x0 + 1).min(self.width - 1);
                let y1 = (y0 + 1).min(self.height - 1);
                let x0 = x0.min(self.width - 1);
                let y0 = y0.min(self.height - 1);

                let fx = src_x - x0 as f32;
                let fy = src_y - y0 as f32;

                let p00 = self.get_pixel(x0, y0);
                let p10 = self.get_pixel(x1, y0);
                let p01 = self.get_pixel(x0, y1);
                let p11 = self.get_pixel(x1, y1);

                let mut pixel = [0u8; 3];
                for c in 0..3 {
                    let val = (p00[c] as f32 * (1.0 - fx) * (1.0 - fy)
                        + p10[c] as f32 * fx * (1.0 - fy)
                        + p01[c] as f32 * (1.0 - fx) * fy
                        + p11[c] as f32 * fx * fy)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    pixel[c] = val;
                }
                out.set_pixel(x, y, pixel);
            }
        }
        out
    }
}

/// Represents either a file path or an existing Image.
pub enum ImageSource<'a> {
    Path(&'a Path),
    Image(&'a Image),
}

/// Load an image from a source.
pub fn load_image(source: &ImageSource) -> Result<Image, BaseOnnxError> {
    match source {
        ImageSource::Path(p) => Image::load(p),
        ImageSource::Image(img) => Ok((*img).clone()),
    }
}

/// Create a tract-onnx model from a model path.
pub fn create_model<P: AsRef<Path>>(
    model_path: P,
) -> Result<tract_onnx::prelude::TypedModel, BaseOnnxError> {
    use tract_onnx::prelude::{Framework, InferenceModelExt};
    tract_onnx::onnx()
        .model_for_path(model_path)
        .map_err(|e| BaseOnnxError::OnnxError(e.to_string()))?
        .into_optimized()
        .map_err(|e| BaseOnnxError::OnnxError(e.to_string()))
}

/// Re-export ndarray dims
pub mod nd {
    pub use ndarray::Dim;
    pub type Dim4 = Dim<[usize; 4]>;
}

pub fn check_images_list(_images: &[ImageSource]) -> Result<(), BaseOnnxError> {
    Ok(())
}
