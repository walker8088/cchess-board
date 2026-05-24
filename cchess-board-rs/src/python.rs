use ndarray::Array3;
use numpy::{PyArray3, PyArrayMethods, PyUntypedArrayMethods, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::base_onnx::{BaseOnnxError, Image};
use crate::detector::{
    check_keypoints, corner_points_to_rect, extract_chessboard, get_board_corner_points,
    perspective_transform,
};
use crate::rtmpose::{
    get_perspective_transform, invert_affine_2x3, invert_perspective_3x3,
    perspective_transform_points, warp_affine, warp_perspective,
};

// ─── Error Conversion ─────────────────────────────────────────────────────

fn to_py_err(e: BaseOnnxError) -> PyErr {
    match e {
        BaseOnnxError::OnnxError(msg) => PyRuntimeError::new_err(msg),
        BaseOnnxError::ImageError(msg) => PyValueError::new_err(msg),
        BaseOnnxError::DimMismatch { expected, actual } => PyValueError::new_err(format!(
            "Dimension mismatch: expected {expected}, got {actual}"
        )),
    }
}

// ─── Python Image Wrapper ─────────────────────────────────────────────────

/// A BGR image backed by a contiguous [H,W,3] u8 array.
#[pyclass(name = "Image")]
#[derive(Clone)]
pub struct PyImage {
    inner: Image,
}

#[pymethods]
impl PyImage {
    #[new]
    fn new(width: usize, height: usize) -> Self {
        Self {
            inner: Image::new(width, height),
        }
    }

    #[staticmethod]
    fn from_array(arr: Bound<'_, PyArray3<u8>>) -> PyResult<Self> {
        let shape = arr.shape();
        if shape.len() != 3 || shape[2] != 3 {
            return Err(PyValueError::new_err(format!(
                "Expected array of shape (H, W, 3), got {:?}",
                shape
            )));
        }
        let h = shape[0];
        let w = shape[1];
        let data = arr
            .to_vec()
            .map_err(|e| PyValueError::new_err(format!("Failed to convert array to vec: {e}")))?;
        Ok(Self {
            inner: Image::from_vec(data, w, h).map_err(to_py_err)?,
        })
    }

    fn to_array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let arr = Array3::from_shape_vec(
            (self.inner.height, self.inner.width, 3),
            self.inner.data.clone(),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.to_pyarray_bound(py).to_owned())
    }

    fn to_rgb_array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let rgb = self.inner.bgr_to_rgb();
        let arr = Array3::from_shape_vec((rgb.height, rgb.width, 3), rgb.data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.to_pyarray_bound(py).to_owned())
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let img = Image::load(path).map_err(to_py_err)?;
        Ok(Self { inner: img })
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(to_py_err)
    }

    #[getter]
    fn width(&self) -> usize {
        self.inner.width
    }

    #[getter]
    fn height(&self) -> usize {
        self.inner.height
    }

    fn crop(&self, x: usize, y: usize, w: usize, h: usize) -> Self {
        Self {
            inner: self.inner.crop(x, y, w, h),
        }
    }

    fn resize(&self, new_w: usize, new_h: usize) -> Self {
        Self {
            inner: self.inner.resize(new_w, new_h),
        }
    }

    fn bgr_to_rgb(&self) -> Self {
        Self {
            inner: self.inner.bgr_to_rgb(),
        }
    }

    fn rgb_to_bgr(&self) -> Self {
        Self {
            inner: self.inner.rgb_to_bgr(),
        }
    }

    fn get_pixel(&self, x: usize, y: usize) -> [u8; 3] {
        self.inner.get_pixel(x, y)
    }

    fn __repr__(&self) -> String {
        format!(
            "Image(width={}, height={}, data_len={})",
            self.inner.width,
            self.inner.height,
            self.inner.data.len()
        )
    }
}

// ─── Geometric Transforms ─────────────────────────────────────────────────

/// Compute perspective transform matrix from 4 src points to 4 dst points.
///
/// Args:
///     src: List of 4 [x,y] source points
///     dst: List of 4 [x,y] destination points
/// Returns:
///     3x3 perspective matrix as list of lists
#[pyfunction]
fn py_get_perspective_transform(src: [[f32; 2]; 4], dst: [[f32; 2]; 4]) -> [[f32; 3]; 3] {
    get_perspective_transform(src, dst)
}

/// Apply perspective transform to an image.
///
/// Args:
///     img: Source Image
///     mat: 3x3 perspective matrix
///     dst_w: Output width
///     dst_h: Output height
/// Returns:
///     Transformed Image
#[pyfunction]
fn py_warp_perspective(
    img: &PyImage,
    mat: [[f32; 3]; 3],
    dst_w: usize,
    dst_h: usize,
) -> PyResult<PyImage> {
    let result = warp_perspective(&img.inner, &mat, dst_w, dst_h).map_err(to_py_err)?;
    Ok(PyImage { inner: result })
}

/// Apply affine transform to an image.
///
/// Args:
///     img: Source Image
///     mat: 2x3 affine matrix
///     dst_w: Output width
///     dst_h: Output height
/// Returns:
///     Transformed Image
#[pyfunction]
fn py_warp_affine(
    img: &PyImage,
    mat: [[f32; 3]; 2],
    dst_w: usize,
    dst_h: usize,
) -> PyResult<PyImage> {
    let result = warp_affine(&img.inner, &mat, dst_w, dst_h).map_err(to_py_err)?;
    Ok(PyImage { inner: result })
}

/// Transform points using a perspective matrix.
///
/// Args:
///     points: List of [x,y] points
///     mat: 3x3 perspective matrix
/// Returns:
///     List of transformed [x,y] points
#[pyfunction]
fn py_perspective_transform_points(points: Vec<[f32; 2]>, mat: [[f32; 3]; 3]) -> Vec<[f32; 2]> {
    perspective_transform_points(&points, &mat)
}

/// Invert a 3x3 perspective matrix.
///
/// Args:
///     mat: 3x3 perspective matrix
/// Returns:
///     Inverted 3x3 matrix, or None if singular
#[pyfunction]
fn py_invert_perspective_3x3(mat: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    invert_perspective_3x3(&mat)
}

/// Invert a 2x3 affine matrix.
///
/// Args:
///     mat: 2x3 affine matrix
/// Returns:
///     Inverted 2x3 matrix, or None if singular
#[pyfunction]
fn py_invert_affine_2x3(mat: [[f32; 3]; 2]) -> Option<[[f32; 3]; 2]> {
    invert_affine_2x3(&mat)
}

// ─── Detector Functions ──────────────────────────────────────────────────

/// Validate that keypoints has exactly 4 points.
///
/// Args:
///     keypoints: List of [x,y] points
/// Returns:
///     None if valid, raises ValueError otherwise
#[pyfunction]
fn py_check_keypoints(keypoints: Vec<[f32; 2]>) -> PyResult<()> {
    check_keypoints(&keypoints).map_err(to_py_err)
}

/// Compute the axis-aligned bounding box corners from 4 keypoints.
///
/// The keypoints are expected to be in order: [A0, A8, J0, J8]
/// corresponding to the four corners of a chessboard.
///
/// Args:
///     keypoints: List of 4 [x,y] points
/// Returns:
///     4 corners: [TL, TR, BL, BR] as [min_x,min_y], [max_x,min_y], [min_x,max_y], [max_x,max_y]
#[pyfunction]
fn py_get_board_corner_points(keypoints: Vec<[f32; 2]>) -> PyResult<[[f32; 2]; 4]> {
    get_board_corner_points(&keypoints).map_err(to_py_err)
}

/// Expand corner points to a rectangular region with half-grid padding.
///
/// Args:
///     corner_points: 4 corners [[x,y], ...]
/// Returns:
///     Tuple of ((min_x, min_y), (max_x, max_y))
#[pyfunction]
fn py_corner_points_to_rect(corner_points: [[f32; 2]; 4]) -> ((usize, usize), (usize, usize)) {
    corner_points_to_rect(&corner_points)
}

/// Extract and rectify a chessboard from an image given keypoints.
///
/// Args:
///     img: Source Image
///     keypoints: List of 4 [x,y] corner keypoints
/// Returns:
///     Tuple of (rectified Image, transformed keypoints, destination corners)
#[pyfunction]
fn py_extract_chessboard(
    img: &PyImage,
    keypoints: Vec<[f32; 2]>,
) -> PyResult<(PyImage, Vec<[f32; 2]>, [[f32; 2]; 4])> {
    let (result, kps, corners) = extract_chessboard(&img.inner, &keypoints).map_err(to_py_err)?;
    Ok((PyImage { inner: result }, kps, corners))
}

/// Full perspective transform pipeline: src points + keypoints -> rectified image.
///
/// Args:
///     img: Source Image
///     src_points: 4 source corner points [[x,y], ...]
///     keypoints: 4 keypoints for validation [[x,y], ...]
///     dst_size: (width, height) of output image
/// Returns:
///     Tuple of (rectified Image, transformed keypoints, destination corners)
#[pyfunction]
fn py_perspective_transform(
    img: &PyImage,
    src_points: [[f32; 2]; 4],
    keypoints: Vec<[f32; 2]>,
    dst_size: (usize, usize),
) -> PyResult<(PyImage, Vec<[f32; 2]>, [[f32; 2]; 4])> {
    let (result, kps, corners) =
        perspective_transform(&img.inner, src_points, &keypoints, dst_size).map_err(to_py_err)?;
    Ok((PyImage { inner: result }, kps, corners))
}

// ─── Module Definition ───────────────────────────────────────────────────

/// Pure Rust Chinese chess board detection utilities.
///
/// Provides image manipulation, geometric transforms, and chessboard
/// extraction — all without OpenCV or ONNX runtime dependencies.
#[pymodule]
fn cchess_board_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyImage>()?;
    m.add_function(wrap_pyfunction!(py_get_perspective_transform, m)?)?;
    m.add_function(wrap_pyfunction!(py_warp_perspective, m)?)?;
    m.add_function(wrap_pyfunction!(py_warp_affine, m)?)?;
    m.add_function(wrap_pyfunction!(py_perspective_transform_points, m)?)?;
    m.add_function(wrap_pyfunction!(py_invert_perspective_3x3, m)?)?;
    m.add_function(wrap_pyfunction!(py_invert_affine_2x3, m)?)?;
    m.add_function(wrap_pyfunction!(py_check_keypoints, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_board_corner_points, m)?)?;
    m.add_function(wrap_pyfunction!(py_corner_points_to_rect, m)?)?;
    m.add_function(wrap_pyfunction!(py_extract_chessboard, m)?)?;
    m.add_function(wrap_pyfunction!(py_perspective_transform, m)?)?;
    Ok(())
}
