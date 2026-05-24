use ndarray::Array4;
use tract_onnx::prelude::tvec;

use crate::base_onnx::{create_model, BaseOnnxError, Image};

/// Mapping from class name to short symbol (FEN-style).
const DICT_CATE_NAMES: [(&str, &str); 16] = [
    ("point", "."),
    ("other", "x"),
    ("red_king", "K"),
    ("red_advisor", "A"),
    ("red_bishop", "B"),
    ("red_knight", "N"),
    ("red_rook", "R"),
    ("red_cannon", "C"),
    ("red_pawn", "P"),
    ("black_king", "k"),
    ("black_advisor", "a"),
    ("black_bishop", "b"),
    ("black_knight", "n"),
    ("black_rook", "r"),
    ("black_cannon", "c"),
    ("black_pawn", "p"),
];

const MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const STD: [f32; 3] = [58.395, 57.12, 57.375];

pub struct ClassifierResult {
    pub label_names: Vec<Vec<String>>,
    pub label_short: Vec<Vec<String>>,
    pub confidence: Vec<Vec<f32>>,
    pub layout_str: String,
}

pub type OnnxModel = tract_onnx::prelude::SimplePlan<
    tract_onnx::prelude::TypedFact,
    Box<dyn tract_onnx::prelude::TypedOp>,
    tract_onnx::prelude::TypedModel,
>;

/// ONNX classifier for Chinese chess pieces on a 10x9 grid.
pub struct ClassifierOnnx {
    model: OnnxModel,
    pub input_size: (usize, usize),
    pub crop_size: (usize, usize),
}

impl ClassifierOnnx {
    pub fn new<P: AsRef<std::path::Path>>(
        model_path: P,
        input_size: (usize, usize),
        crop_size: (usize, usize),
    ) -> Result<Self, BaseOnnxError> {
        let model = create_model(model_path)?;
        let plan = tract_onnx::prelude::SimplePlan::new(model)
            .map_err(|e| BaseOnnxError::OnnxError(e.to_string()))?;
        Ok(Self {
            model: plan,
            input_size,
            crop_size,
        })
    }

    fn center_crop(&self, img: &Image) -> Image {
        let (cw, ch) = self.crop_size;
        let start_x = (img.width / 2).saturating_sub(cw / 2);
        let start_y = (img.height / 2).saturating_sub(ch / 2);
        img.crop(start_x, start_y, cw, ch)
    }

    pub fn preprocess_image(&self, img_bgr: &Image) -> Result<Array4<f32>, BaseOnnxError> {
        let img_rgb = if img_bgr.width != self.crop_size.0 || img_bgr.height != self.crop_size.1 {
            self.center_crop(img_bgr).bgr_to_rgb()
        } else {
            img_bgr.bgr_to_rgb()
        };

        let resized = img_rgb.resize(self.input_size.0, self.input_size.1);

        let (w, h) = (self.input_size.0, self.input_size.1);
        let mut array = Array4::<f32>::zeros((1, 3, h, w));

        for y in 0..h {
            for x in 0..w {
                let pixel = resized.get_pixel(x, y);
                for c in 0..3 {
                    let val = pixel[c] as f32;
                    array[[0, c, y, x]] = (val - MEAN[c]) / STD[c];
                }
            }
        }
        Ok(array)
    }

    pub fn pred(&self, img_bgr: &Image) -> Result<ClassifierResult, BaseOnnxError> {
        use tract_onnx::prelude::Tensor;

        let input = self.preprocess_image(img_bgr)?;
        let shape = input.shape();
        let tensor = Tensor::from_shape(shape, input.as_slice().unwrap())
            .map_err(|e| BaseOnnxError::OnnxError(e.to_string()))?;

        let result = self
            .model
            .run(tvec![tensor.into()])
            .map_err(|e| BaseOnnxError::OnnxError(e.to_string()))?;

        let output = result
            .first()
            .ok_or_else(|| BaseOnnxError::OnnxError("No output from model".into()))?;
        let tensor_f32 = output
            .to_array_view::<f32>()
            .map_err(|e| BaseOnnxError::OnnxError(e.to_string()))?;

        let data: Vec<f32> = tensor_f32.iter().cloned().collect();
        let expected_len = 1 * 90 * 16;
        if data.len() != expected_len {
            return Err(BaseOnnxError::DimMismatch {
                expected: format!("{}", expected_len),
                actual: format!("{}", data.len()),
            });
        }

        let mut label_indexes = vec![0usize; 90];
        let mut confidences = vec![0.0f32; 90];

        for i in 0..90 {
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..16 {
                let val = data[i * 16 + j];
                if val > max_val {
                    max_val = val;
                    max_idx = j;
                }
            }
            label_indexes[i] = max_idx;
            confidences[i] = max_val;
        }

        let class_keys: Vec<&str> = DICT_CATE_NAMES.iter().map(|(k, _)| *k).collect();
        let label_names: Vec<String> = label_indexes
            .iter()
            .map(|&idx| class_keys[idx].to_string())
            .collect();
        let label_short: Vec<String> = label_indexes
            .iter()
            .map(|&idx| DICT_CATE_NAMES[idx].1.to_string())
            .collect();

        let label_names_10x9: Vec<Vec<String>> = (0..10)
            .map(|i| label_names[i * 9..(i + 1) * 9].to_vec())
            .collect();
        let label_short_10x9: Vec<Vec<String>> = (0..10)
            .map(|i| label_short[i * 9..(i + 1) * 9].to_vec())
            .collect();
        let confidence_10x9: Vec<Vec<f32>> = (0..10)
            .map(|i| confidences[i * 9..(i + 1) * 9].to_vec())
            .collect();

        let layout_str = label_short_10x9
            .iter()
            .map(|row| row.join(""))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(ClassifierResult {
            label_names: label_names_10x9,
            label_short: label_short_10x9,
            confidence: confidence_10x9,
            layout_str,
        })
    }
}
