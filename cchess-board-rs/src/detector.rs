use std::path::{Path, PathBuf};

use crate::base_onnx::{BaseOnnxError, Image};
use crate::classifier::{ClassifierOnnx, ClassifierResult};
use crate::rtmpose::{
    get_perspective_transform, perspective_transform_points, warp_perspective, PoseResult,
    RtmposeOnnx,
};

const BONE_NAMES_DETECTOR: [&str; 4] = ["A0", "A8", "J0", "J8"];

pub fn check_keypoints(keypoints: &[[f32; 2]]) -> Result<(), BaseOnnxError> {
    if keypoints.len() != BONE_NAMES_DETECTOR.len() {
        return Err(BaseOnnxError::DimMismatch {
            expected: format!("({}, 2)", BONE_NAMES_DETECTOR.len()),
            actual: format!("({}, 2)", keypoints.len()),
        });
    }
    Ok(())
}

pub fn perspective_transform(
    image: &Image,
    src_points: [[f32; 2]; 4],
    keypoints: &[[f32; 2]],
    dst_size: (usize, usize),
) -> Result<(Image, Vec<[f32; 2]>, [[f32; 2]; 4]), BaseOnnxError> {
    check_keypoints(keypoints)?;

    let padding: f32 = 50.0;
    let (dw, dh) = (dst_size.0 as f32, dst_size.1 as f32);

    let dst_corners: [[f32; 2]; 4] = [
        [padding, padding],
        [dw - padding, padding],
        [padding, dh - padding],
        [dw - padding, dh - padding],
    ];

    let matrix = get_perspective_transform(src_points, dst_corners);

    let result = warp_perspective(image, &matrix, dst_size.0, dst_size.1)?;

    let transformed_keypoints = perspective_transform_points(keypoints, &matrix);

    Ok((result, transformed_keypoints, dst_corners))
}

pub fn get_board_corner_points(keypoints: &[[f32; 2]]) -> Result<[[f32; 2]; 4], BaseOnnxError> {
    check_keypoints(keypoints)?;

    let a0 = keypoints[BONE_NAMES_DETECTOR.iter().position(|&n| n == "A0").unwrap()];
    let a8 = keypoints[BONE_NAMES_DETECTOR.iter().position(|&n| n == "A8").unwrap()];
    let j0 = keypoints[BONE_NAMES_DETECTOR.iter().position(|&n| n == "J0").unwrap()];
    let j8 = keypoints[BONE_NAMES_DETECTOR.iter().position(|&n| n == "J8").unwrap()];

    let min_x = a0[0].min(a8[0]).min(j0[0]).min(j8[0]);
    let min_y = a0[1].min(a8[1]).min(j0[1]).min(j8[1]);
    let max_x = a0[0].max(a8[0]).max(j0[0]).max(j8[0]);
    let max_y = a0[1].max(a8[1]).max(j0[1]).max(j8[1]);

    Ok([
        [min_x, min_y],
        [max_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ])
}

pub fn corner_points_to_rect(corner_points: &[[f32; 2]; 4]) -> ((usize, usize), (usize, usize)) {
    let a0 = corner_points[0];
    let a8 = corner_points[1];
    let j0 = corner_points[2];
    let j8 = corner_points[3];

    let min_x = a0[0].min(a8[0]).min(j0[0]).min(j8[0]);
    let min_y = a0[1].min(a8[1]).min(j0[1]).min(j8[1]);
    let max_x = a0[0].max(a8[0]).max(j0[0]).max(j8[0]);
    let max_y = a0[1].max(a8[1]).max(j0[1]).max(j8[1]);

    let grid_x_half = (max_x - min_x) / 8.0;
    let grid_y_half = (max_y - min_y) / 9.0;

    let min_x = (0.0f32.max(min_x - grid_x_half)) as usize;
    let max_x = (0.0f32.max(max_x + grid_x_half)) as usize;
    let min_y = (0.0f32.max(min_y - grid_y_half)) as usize;
    let max_y = (0.0f32.max(max_y + grid_y_half)) as usize;

    ((min_x, min_y), (max_x, max_y))
}

pub fn extract_chessboard(
    img_bgr: &Image,
    keypoints: &[[f32; 2]],
) -> Result<(Image, Vec<[f32; 2]>, [[f32; 2]; 4]), BaseOnnxError> {
    let source_corner_points = get_board_corner_points(keypoints)?;
    perspective_transform(img_bgr, source_corner_points, keypoints, (450, 500))
}

pub struct ChessboardDetector {
    pub pose: RtmposeOnnx,
    pub classifier: ClassifierOnnx,
    pub current_image: Option<Image>,
    pub current_filename: Option<PathBuf>,
}

impl ChessboardDetector {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, BaseOnnxError> {
        let model_path = model_path.as_ref();
        let pose_path = model_path.join("pose.onnx");
        let classifier_path = model_path.join("layout.onnx");

        let pose = RtmposeOnnx::new(&pose_path, (256, 256), 1.25)?;
        let classifier = ClassifierOnnx::new(&classifier_path, (280, 315), (400, 450))?;

        Ok(Self {
            pose,
            classifier,
            current_image: None,
            current_filename: None,
        })
    }

    pub fn pred_keypoints(&self, image_bgr: &Image) -> Result<PoseResult, BaseOnnxError> {
        let height = image_bgr.height as f32;
        let width = image_bgr.width as f32;
        let bbox: [f32; 4] = [0.0, 0.0, width, height];
        self.pose.pred(image_bgr, &bbox)
    }

    pub fn get_board_rect(
        &self,
        img: &Image,
    ) -> Result<(((usize, usize), (usize, usize)), Image), BaseOnnxError> {
        let pose_result = self.pred_keypoints(img)?;
        check_keypoints(&pose_result.keypoints)?;

        let corner_points = get_board_corner_points(&pose_result.keypoints)?;
        let board_rect = corner_points_to_rect(&corner_points);

        let ((min_x, min_y), (max_x, max_y)) = board_rect;
        let cropped = img.crop(min_x, min_y, max_x - min_x, max_y - min_y);

        Ok((board_rect, cropped))
    }

    pub fn img_to_labels(&self, image_bgr: &Image) -> Result<ClassifierResult, BaseOnnxError> {
        self.classifier.pred(image_bgr)
    }

    pub fn pred_detect_board_and_classifier(
        &self,
        image_bgr: &Image,
    ) -> Result<(Image, ClassifierResult), BaseOnnxError> {
        let pose_result = self.pred_keypoints(image_bgr)?;
        let (transformed_image, _, _) = extract_chessboard(image_bgr, &pose_result.keypoints)?;
        let classifier_result = self.classifier.pred(&transformed_image)?;
        Ok((transformed_image, classifier_result))
    }

    /// Note: FEN conversion (`labels_to_fen`) excluded per user request.
    pub fn cv_image_to_fen(&self, img_bgr: &Image) -> Result<ClassifierResult, BaseOnnxError> {
        let (_, cell_labels) = self.pred_detect_board_and_classifier(img_bgr)?;
        Ok(cell_labels)
    }

    /// Note: FEN conversion excluded per user request.
    pub fn transformed_img_to_fen(
        &self,
        image_bgr: &Image,
    ) -> Result<ClassifierResult, BaseOnnxError> {
        self.classifier.pred(image_bgr)
    }

    pub fn img_to_board<P: AsRef<Path>>(
        &mut self,
        image_file: P,
    ) -> Result<(Image, ClassifierResult), BaseOnnxError> {
        let path = image_file.as_ref();
        let img = Image::load(path)?;

        self.current_image = Some(img.clone());
        self.current_filename = Some(path.to_path_buf());
        self.pred_detect_board_and_classifier(&img)
    }

    pub fn img_board_mark_to<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        image_file: P,
        mark_file: Q,
    ) -> Result<bool, BaseOnnxError> {
        let path = image_file.as_ref();
        let img = Image::load(path)?;

        let (_board_rect, cropped_image) = self.get_board_rect(&img)?;
        let labels = self.img_to_labels(&cropped_image)?;
        println!("layout: {}", labels.layout_str);

        cropped_image.save(mark_file)?;

        Ok(true)
    }
}
