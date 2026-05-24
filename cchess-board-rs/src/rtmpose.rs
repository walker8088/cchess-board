use ndarray::Array2;
use tract_onnx::prelude::tvec;

use crate::base_onnx::{create_model, BaseOnnxError, Image};

/// Keypoint names for the chessboard (4 corners).
pub const BONE_NAMES: [&str; 4] = ["A0", "A8", "J0", "J8"];

/// Skeleton links for visualization.
pub const SKELETON_LINKS: [&str; 3] = ["A0-A8", "A8-J8", "J8-J0"];

/// Normalize mean and std.
const MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const STD: [f32; 3] = [58.395, 57.12, 57.375];

pub struct PoseResult {
    pub keypoints: Vec<[f32; 2]>,
    pub scores: Vec<f32>,
}

/// 2x3 affine transformation matrix stored in row-major [2][3].
pub type AffineMatrix = [[f32; 3]; 2];

/// 3x3 perspective transformation matrix stored in row-major [3][3].
pub type PerspectiveMatrix = [[f32; 3]; 3];

pub type OnnxModel = tract_onnx::prelude::SimplePlan<
    tract_onnx::prelude::TypedFact,
    Box<dyn tract_onnx::prelude::TypedOp>,
    tract_onnx::prelude::TypedModel,
>;

/// RTMPose ONNX model for detecting chessboard corners.
pub struct RtmposeOnnx {
    model: OnnxModel,
    pub input_size: (usize, usize),
    pub padding: f32,
    pub bone_colors: Vec<[u8; 3]>,
}

impl RtmposeOnnx {
    pub fn new<P: AsRef<std::path::Path>>(
        model_path: P,
        input_size: (usize, usize),
        padding: f32,
    ) -> Result<Self, BaseOnnxError> {
        let model = create_model(model_path)?;
        let plan = tract_onnx::prelude::SimplePlan::new(model)
            .map_err(|e| BaseOnnxError::OnnxError(e.to_string()))?;

        let bone_colors: Vec<[u8; 3]> = BONE_NAMES
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let r = ((i * 73 + 17) % 256) as u8;
                let g = ((i * 137 + 43) % 256) as u8;
                let b = ((i * 199 + 89) % 256) as u8;
                [r, g, b]
            })
            .collect();

        Ok(Self {
            model: plan,
            input_size,
            padding,
            bone_colors,
        })
    }

    pub fn get_bbox_center_scale(&self, bbox: &[f32; 4]) -> ([f32; 2], [f32; 2]) {
        let [x1, y1, x2, y2] = *bbox;
        let center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0];
        let w = x2 - x1;
        let h = y2 - y1;
        let scale = [w * self.padding, h * self.padding];
        (center, scale)
    }

    /// Compute affine transform from 3 src points to 3 dst points.
    pub fn get_affine_transform(src: [[f32; 2]; 3], dst: [[f32; 2]; 3]) -> AffineMatrix {
        // We want M such that for each point:
        //   dst_x = M[0][0]*src_x + M[0][1]*src_y + M[0][2]
        //   dst_y = M[1][0]*src_x + M[1][1]*src_y + M[1][2]
        let mut m = [[0.0f32; 3]; 2];

        // Solve for x coefficients: M[0][0]*sx + M[0][1]*sy = dx - M[0][2]
        // Using first two points for M[0][2] elimination:
        let sx0 = src[0][0];
        let sy0 = src[0][1];
        let sx1 = src[1][0];
        let sy1 = src[1][1];
        let sx2 = src[2][0];
        let sy2 = src[2][1];
        let dx0 = dst[0][0];
        let dy0 = dst[0][1];
        let dx1 = dst[1][0];
        let dy1 = dst[1][1];
        let dx2 = dst[2][0];
        let dy2 = dst[2][1];

        // For x: solve [[sx0,sy0,1],[sx1,sy1,1],[sx2,sy2,1]] * [mx00, mx01, mx02] = [dx0,dx1,dx2]
        // Use Gaussian elimination
        let mut a = [
            [sx0, sy0, 1.0, dx0],
            [sx1, sy1, 1.0, dx1],
            [sx2, sy2, 1.0, dx2],
        ];

        // Eliminate
        for i in 0..3 {
            let pivot = a[i][i];
            if pivot.abs() < 1e-12 {
                continue;
            }
            for j in 0..4 {
                a[i][j] /= pivot;
            }
            for k in 0..3 {
                if k != i {
                    let factor = a[k][i];
                    for j in 0..4 {
                        a[k][j] -= factor * a[i][j];
                    }
                }
            }
        }

        m[0][0] = a[0][3];
        m[0][1] = a[1][3];
        m[0][2] = a[2][3];

        // For y
        let mut b = [
            [sx0, sy0, 1.0, dy0],
            [sx1, sy1, 1.0, dy1],
            [sx2, sy2, 1.0, dy2],
        ];
        for i in 0..3 {
            let pivot = b[i][i];
            if pivot.abs() < 1e-12 {
                continue;
            }
            for j in 0..4 {
                b[i][j] /= pivot;
            }
            for k in 0..3 {
                if k != i {
                    let factor = b[k][i];
                    for j in 0..4 {
                        b[k][j] -= factor * b[i][j];
                    }
                }
            }
        }

        m[1][0] = b[0][3];
        m[1][1] = b[1][3];
        m[1][2] = b[2][3];

        m
    }

    /// Get the 3rd point for affine transform (rotate vector a-b by 90° CCW around b).
    fn get_3rd_point(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        [b[0] - dy, b[1] + dx]
    }

    /// Rotate a 2D point by angle in radians.
    fn rotate_point(pt: [f32; 2], angle_rad: f32) -> [f32; 2] {
        let (s, c) = angle_rad.sin_cos();
        [c * pt[0] - s * pt[1], s * pt[0] + c * pt[1]]
    }

    /// Calculate affine transformation matrix (mimics OpenCV's behavior).
    pub fn get_warp_matrix(
        center: [f32; 2],
        scale: [f32; 2],
        rot: f32,
        output_size: (usize, usize),
        shift: (f32, f32),
        inv: bool,
        fix_aspect_ratio: bool,
    ) -> AffineMatrix {
        let (dst_w, dst_h) = (output_size.0 as f32, output_size.1 as f32);
        let rot_rad = rot.to_radians();

        let src_dir = Self::rotate_point([-scale[0] * 0.5, 0.0], rot_rad);
        let dst_dir = [-dst_w * 0.5, 0.0];

        let s0 = [
            center[0] + scale[0] * shift.0,
            center[1] + scale[1] * shift.1,
        ];
        let s1 = [
            center[0] + src_dir[0] + scale[0] * shift.0,
            center[1] + src_dir[1] + scale[1] * shift.1,
        ];
        let d0 = [dst_w * 0.5, dst_h * 0.5];
        let d1 = [dst_w * 0.5 + dst_dir[0], dst_h * 0.5 + dst_dir[1]];

        let (src_2, dst_2) = if fix_aspect_ratio {
            (Self::get_3rd_point(s0, s1), Self::get_3rd_point(d0, d1))
        } else {
            let src_dir_2 = Self::rotate_point([0.0, -scale[1] * 0.5], rot_rad);
            let dst_dir_2 = [0.0, -dst_h * 0.5];
            (
                [
                    center[0] + src_dir_2[0] + scale[0] * shift.0,
                    center[1] + src_dir_2[1] + scale[1] * shift.1,
                ],
                [dst_w * 0.5 + dst_dir_2[0], dst_h * 0.5 + dst_dir_2[1]],
            )
        };

        let src_pts = [s0, s1, src_2];
        let dst_pts = [d0, d1, dst_2];

        let m = if inv {
            Self::get_affine_transform(dst_pts, src_pts)
        } else {
            Self::get_affine_transform(src_pts, dst_pts)
        };
        m
    }

    pub fn get_warp_size_with_input_size(
        &self,
        bbox_center: [f32; 2],
        bbox_scale: [f32; 2],
        inv: bool,
    ) -> AffineMatrix {
        let (w, h) = self.input_size;
        let aspect_ratio = w as f32 / h as f32;
        let [scale_w, scale_h] = bbox_scale;
        let adjusted_scale = if scale_w > scale_h * aspect_ratio {
            [scale_w, scale_w / aspect_ratio]
        } else {
            [scale_h * aspect_ratio, scale_h]
        };

        Self::get_warp_matrix(
            bbox_center,
            adjusted_scale,
            0.0,
            (w, h),
            (0.0, 0.0),
            inv,
            true,
        )
    }

    /// Apply top-down affine transform using bilinear interpolation.
    pub fn topdown_affine(
        &self,
        img: &Image,
        bbox_center: [f32; 2],
        bbox_scale: [f32; 2],
    ) -> Result<Image, BaseOnnxError> {
        let warp_mat = self.get_warp_size_with_input_size(bbox_center, bbox_scale, false);
        let (w, h) = self.input_size;
        warp_affine(img, &warp_mat, w, h)
    }

    pub fn get_simcc_maximum(
        &self,
        simcc_x: &[[f32; 512]],
        simcc_y: &[[f32; 512]],
    ) -> (Vec<[f32; 2]>, Vec<f32>) {
        let (input_w, input_h) = (self.input_size.0 as f32, self.input_size.1 as f32);
        let n = simcc_x.len();
        let mut keypoints = Vec::with_capacity(n);
        let mut scores = Vec::with_capacity(n);

        for i in 0..n {
            let mut max_x_idx = 0usize;
            let mut max_x_val = simcc_x[i][0];
            for j in 1..512 {
                if simcc_x[i][j] > max_x_val {
                    max_x_val = simcc_x[i][j];
                    max_x_idx = j;
                }
            }
            let mut max_y_idx = 0usize;
            let mut max_y_val = simcc_y[i][0];
            for j in 1..512 {
                if simcc_y[i][j] > max_y_val {
                    max_y_val = simcc_y[i][j];
                    max_y_idx = j;
                }
            }
            let x_coord = max_x_idx as f32 / (input_w * 2.0);
            let y_coord = max_y_idx as f32 / (input_h * 2.0);
            keypoints.push([x_coord, y_coord]);
            scores.push(max_x_val * max_y_val);
        }
        (keypoints, scores)
    }

    pub fn preprocess_image(
        &self,
        img_bgr: &Image,
        bbox_center: [f32; 2],
        bbox_scale: [f32; 2],
    ) -> Result<ndarray::Array4<f32>, BaseOnnxError> {
        let affine_img = self.topdown_affine(img_bgr, bbox_center, bbox_scale)?;
        let img_rgb = affine_img.bgr_to_rgb();
        let (w, h) = (self.input_size.0, self.input_size.1);
        let mut array = ndarray::Array4::<f32>::zeros((1, 3, h, w));

        for y in 0..h {
            for x in 0..w {
                let pixel = img_rgb.get_pixel(x, y);
                for c in 0..3 {
                    let val = pixel[c] as f32;
                    array[[0, c, y, x]] = (val - MEAN[c]) / STD[c];
                }
            }
        }
        Ok(array)
    }

    pub fn pred(&self, image: &Image, bbox: &[f32; 4]) -> Result<PoseResult, BaseOnnxError> {
        let (center, scale) = self.get_bbox_center_scale(bbox);
        let input = self.preprocess_image(image, center, scale)?;
        let shape = input.shape();
        let tensor = tract_onnx::prelude::Tensor::from_shape(shape, input.as_slice().unwrap())
            .map_err(|e| BaseOnnxError::OnnxError(e.to_string()))?;

        let outputs = self
            .model
            .run(tvec![tensor.into()])
            .map_err(|e| BaseOnnxError::OnnxError(e.to_string()))?;

        if outputs.len() < 2 {
            return Err(BaseOnnxError::OnnxError(
                "Expected 2 outputs (simcc_x, simcc_y)".into(),
            ));
        }

        // Extract simcc_x (first output)
        let tensor_x = outputs[0]
            .to_array_view::<f32>()
            .map_err(|e| BaseOnnxError::OnnxError(e.to_string()))?;
        let data_x: Vec<f32> = tensor_x.iter().cloned().collect();
        let n = tensor_x.shape()[1];
        let mut simcc_x: Vec<[f32; 512]> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = [0.0f32; 512];
            for j in 0..512 {
                row[j] = data_x[i * 512 + j];
            }
            simcc_x.push(row);
        }

        // Extract simcc_y (second output)
        let tensor_y = outputs[1]
            .to_array_view::<f32>()
            .map_err(|e| BaseOnnxError::OnnxError(e.to_string()))?;
        let data_y: Vec<f32> = tensor_y.iter().cloned().collect();
        let n = tensor_y.shape()[1];
        let mut simcc_y: Vec<[f32; 512]> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = [0.0f32; 512];
            for j in 0..512 {
                row[j] = data_y[i * 512 + j];
            }
            simcc_y.push(row);
        }

        let (keypoints_norm, scores) = self.get_simcc_maximum(&simcc_x, &simcc_y);
        let keypoints = self.transform_keypoints_to_original(&keypoints_norm, center, scale)?;

        Ok(PoseResult { keypoints, scores })
    }

    pub fn transform_keypoints_to_original(
        &self,
        keypoints: &[[f32; 2]],
        center: [f32; 2],
        scale: [f32; 2],
    ) -> Result<Vec<[f32; 2]>, BaseOnnxError> {
        let (ow, oh) = (self.input_size.0 as f32, self.input_size.1 as f32);

        let target_coords: Vec<[f32; 2]> =
            keypoints.iter().map(|&[x, y]| [x * ow, y * oh]).collect();

        let inv_mat = self.get_warp_size_with_input_size(center, scale, true);

        let mut result = Vec::with_capacity(target_coords.len());
        for &[px, py] in &target_coords {
            let new_x = inv_mat[0][0] * px + inv_mat[0][1] * py + inv_mat[0][2];
            let new_y = inv_mat[1][0] * px + inv_mat[1][1] * py + inv_mat[1][2];
            result.push([new_x, new_y]);
        }
        Ok(result)
    }
}

/// Apply affine transformation with bilinear interpolation.
/// warp_mat is a 2x3 matrix: [[a,b,c],[d,e,f]]
/// Output point: x' = a*x + b*y + c, y' = d*x + e*y + f
/// We do inverse mapping: for each output pixel, find source location.
pub fn warp_affine(
    img: &Image,
    warp_mat: &AffineMatrix,
    dst_w: usize,
    dst_h: usize,
) -> Result<Image, BaseOnnxError> {
    // Compute inverse of warp_mat for inverse mapping
    let inv_mat = invert_affine_2x3(warp_mat)
        .ok_or_else(|| BaseOnnxError::ImageError("Cannot invert affine transform".into()))?;

    let mut out = Image::new(dst_w, dst_h);
    let max_x = (img.width - 1) as f32;
    let max_y = (img.height - 1) as f32;

    for y in 0..dst_h {
        for x in 0..dst_w {
            // Map output coords to source coords using inverse matrix
            let sx = inv_mat[0][0] * x as f32 + inv_mat[0][1] * y as f32 + inv_mat[0][2];
            let sy = inv_mat[1][0] * x as f32 + inv_mat[1][1] * y as f32 + inv_mat[1][2];

            if sx < 0.0 || sx > max_x || sy < 0.0 || sy > max_y {
                out.set_pixel(x, y, [0, 0, 0]);
                continue;
            }

            let x0 = sx.floor() as usize;
            let y0 = sy.floor() as usize;
            let x1 = (x0 + 1).min(img.width - 1);
            let y1 = (y0 + 1).min(img.height - 1);
            let x0 = x0.min(img.width - 1);
            let y0 = y0.min(img.height - 1);

            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;

            let p00 = img.get_pixel(x0, y0);
            let p10 = img.get_pixel(x1, y0);
            let p01 = img.get_pixel(x0, y1);
            let p11 = img.get_pixel(x1, y1);

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
    Ok(out)
}

/// Invert a 2x3 affine matrix.
/// Given M = [[a,b,c],[d,e,f]], find M_inv such that M_inv * M = I (for the 2D transform).
pub fn invert_affine_2x3(m: &AffineMatrix) -> Option<AffineMatrix> {
    // M = [[a,b,c],[d,e,f]]
    // We want to solve for M_inv = [[A,B,C],[D,E,F]] such that:
    // For point p: M_inv(M(p)) = p
    // The 2x2 linear part: L = [[a,b],[d,e]]
    // L_inv = 1/det * [[e,-b],[-d,a]]
    let a = m[0][0];
    let b = m[0][1];
    let d = m[1][0];
    let e = m[1][1];
    let det = a * e - b * d;
    if det.abs() < 1e-10 {
        return None;
    }
    let inv_det = 1.0 / det;

    // Inverse of the 2x2 part
    let a_inv = e * inv_det;
    let b_inv = -b * inv_det;
    let d_inv = -d * inv_det;
    let e_inv = a * inv_det;

    // Translation part: C = -A*c - B*f, F = -D*c - E*f
    let c = m[0][2];
    let f = m[1][2];
    let c_inv = -(a_inv * c + b_inv * f);
    let f_inv = -(d_inv * c + e_inv * f);

    Some([[a_inv, b_inv, c_inv], [d_inv, e_inv, f_inv]])
}

/// Compute perspective transform matrix from 4 src points to 4 dst points.
/// Returns a 3x3 matrix in row-major order.
pub fn get_perspective_transform(src: [[f32; 2]; 4], dst: [[f32; 2]; 4]) -> PerspectiveMatrix {
    // Solve the 8 equations for the 8 unknowns (h00..h21, with h22=1)
    // For each point (sx, sy) -> (dx, dy):
    //   dx = (h00*sx + h01*sy + h02) / (h20*sx + h21*sy + 1)
    //   dy = (h10*sx + h11*sy + h12) / (h20*sx + h21*sy + 1)
    // Rearranged:
    //   h00*sx + h01*sy + h02 - h20*sx*dx - h21*sy*dx = dx
    //   h10*sx + h11*sy + h12 - h20*sx*dy - h21*sy*dy = dy

    let mut a = Array2::<f32>::zeros((8, 8));
    let mut b = Array2::<f32>::zeros((8, 1));

    for i in 0..4 {
        let sx = src[i][0];
        let sy = src[i][1];
        let dx = dst[i][0];
        let dy = dst[i][1];

        // Row 2*i: x equation
        a[[2 * i, 0]] = sx;
        a[[2 * i, 1]] = sy;
        a[[2 * i, 2]] = 1.0;
        a[[2 * i, 3]] = 0.0;
        a[[2 * i, 4]] = 0.0;
        a[[2 * i, 5]] = 0.0;
        a[[2 * i, 6]] = -sx * dx;
        a[[2 * i, 7]] = -sy * dx;
        b[[2 * i, 0]] = dx;

        // Row 2*i+1: y equation
        a[[2 * i + 1, 0]] = 0.0;
        a[[2 * i + 1, 1]] = 0.0;
        a[[2 * i + 1, 2]] = 0.0;
        a[[2 * i + 1, 3]] = sx;
        a[[2 * i + 1, 4]] = sy;
        a[[2 * i + 1, 5]] = 1.0;
        a[[2 * i + 1, 6]] = -sx * dy;
        a[[2 * i + 1, 7]] = -sy * dy;
        b[[2 * i + 1, 0]] = dy;
    }

    let h = solve_linear_system(&a, &b).unwrap_or(vec![0.0; 8]);

    [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]]
}

/// Apply perspective transform with bilinear interpolation.
pub fn warp_perspective(
    img: &Image,
    mat: &PerspectiveMatrix,
    dst_w: usize,
    dst_h: usize,
) -> Result<Image, BaseOnnxError> {
    // Invert the perspective matrix
    let inv_mat = invert_perspective_3x3(mat)
        .ok_or_else(|| BaseOnnxError::ImageError("Cannot invert perspective transform".into()))?;

    let mut out = Image::new(dst_w, dst_h);
    let max_x = (img.width - 1) as f32;
    let max_y = (img.height - 1) as f32;

    for y in 0..dst_h {
        for x in 0..dst_w {
            // Map output to source using inverse matrix
            let sx = inv_mat[0][0] * x as f32 + inv_mat[0][1] * y as f32 + inv_mat[0][2];
            let sy = inv_mat[1][0] * x as f32 + inv_mat[1][1] * y as f32 + inv_mat[1][2];
            let w = inv_mat[2][0] * x as f32 + inv_mat[2][1] * y as f32 + inv_mat[2][2];

            let w_inv = if w.abs() < 1e-10 { 1.0 } else { 1.0 / w };
            let src_x = sx * w_inv;
            let src_y = sy * w_inv;

            if src_x < 0.0 || src_x > max_x || src_y < 0.0 || src_y > max_y {
                out.set_pixel(x, y, [0, 0, 0]);
                continue;
            }

            let x0 = src_x.floor() as usize;
            let y0 = src_y.floor() as usize;
            let x1 = (x0 + 1).min(img.width - 1);
            let y1 = (y0 + 1).min(img.height - 1);
            let x0 = x0.min(img.width - 1);
            let y0 = y0.min(img.height - 1);

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            let p00 = img.get_pixel(x0, y0);
            let p10 = img.get_pixel(x1, y0);
            let p01 = img.get_pixel(x0, y1);
            let p11 = img.get_pixel(x1, y1);

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
    Ok(out)
}

/// Transform points using a perspective matrix.
pub fn perspective_transform_points(points: &[[f32; 2]], mat: &PerspectiveMatrix) -> Vec<[f32; 2]> {
    points
        .iter()
        .map(|&[px, py]| {
            let w = mat[2][0] * px + mat[2][1] * py + mat[2][2];
            let w_inv = if w.abs() < 1e-10 { 1.0 } else { 1.0 / w };
            let nx = (mat[0][0] * px + mat[0][1] * py + mat[0][2]) * w_inv;
            let ny = (mat[1][0] * px + mat[1][1] * py + mat[1][2]) * w_inv;
            [nx, ny]
        })
        .collect()
}

/// Invert a 3x3 matrix using Gauss-Jordan elimination.
pub fn invert_perspective_3x3(m: &PerspectiveMatrix) -> Option<PerspectiveMatrix> {
    let mut augmented = [
        [m[0][0], m[0][1], m[0][2], 1.0, 0.0, 0.0],
        [m[1][0], m[1][1], m[1][2], 0.0, 1.0, 0.0],
        [m[2][0], m[2][1], m[2][2], 0.0, 0.0, 1.0],
    ];

    for i in 0..3 {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..3 {
            if augmented[k][i].abs() > augmented[max_row][i].abs() {
                max_row = k;
            }
        }
        augmented.swap(i, max_row);

        let pivot = augmented[i][i];
        if pivot.abs() < 1e-12 {
            return None;
        }
        for j in 0..6 {
            augmented[i][j] /= pivot;
        }
        for k in 0..3 {
            if k != i {
                let factor = augmented[k][i];
                for j in 0..6 {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }

    Some([
        [augmented[0][3], augmented[0][4], augmented[0][5]],
        [augmented[1][3], augmented[1][4], augmented[1][5]],
        [augmented[2][3], augmented[2][4], augmented[2][5]],
    ])
}

/// Solve a linear system Ax = b using Gaussian elimination.
/// A is n×n, b is n×1, returns x as Vec<f32>.
pub fn solve_linear_system(a: &Array2<f32>, b: &Array2<f32>) -> Option<Vec<f32>> {
    let n = a.nrows();
    let mut aug = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(n + 1);
        for j in 0..n {
            row.push(a[[i, j]]);
        }
        row.push(b[[i, 0]]);
        aug.push(row);
    }

    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }
        aug.swap(i, max_row);

        let pivot = aug[i][i];
        if pivot.abs() < 1e-12 {
            return None;
        }
        for j in 0..=n {
            aug[i][j] /= pivot;
        }
        for k in 0..n {
            if k != i {
                let factor = aug[k][i];
                for j in 0..=n {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }

    let mut x = vec![0.0f32; n];
    for i in 0..n {
        x[i] = aug[i][n];
    }
    Some(x)
}
