"""Test suite for cchess_board_rs Python bindings."""

import cchess_board_rs
import numpy as np
import pytest
from cchess_board_rs import (
    Image,
    py_check_keypoints,
    py_corner_points_to_rect,
    py_extract_chessboard,
    py_get_board_corner_points,
    py_get_perspective_transform,
    py_invert_affine_2x3,
    py_invert_perspective_3x3,
    py_perspective_transform,
    py_perspective_transform_points,
    py_warp_affine,
    py_warp_perspective,
)

EPS = 1e-4

# ─── Image Tests ───────────────────────────────────────────────────────────


class TestImage:
    def test_new_creates_zeroed(self):
        img = Image(10, 20)
        assert img.width == 10
        assert img.height == 20
        arr = img.to_rgb_array()
        assert arr.shape == (20, 10, 3)
        assert np.all(arr == 0)

    def test_from_array(self):
        data = np.arange(30, dtype=np.uint8).reshape(5, 2, 3)
        img = Image.from_array(data)
        assert img.width == 2
        assert img.height == 5
        # Convert back and verify
        recovered = img.to_array()
        np.testing.assert_array_equal(recovered, data)

    def test_from_array_wrong_shape(self):
        data = np.arange(20, dtype=np.uint8).reshape(5, 4)
        # PyO3 raises TypeError when array shape doesn't match expected PyArray3
        with pytest.raises(TypeError):
            Image.from_array(data)

    def test_get_pixel(self):
        data = np.zeros((3, 3, 3), dtype=np.uint8)
        data[1, 2] = [100, 150, 200]
        img = Image.from_array(data)
        pixel = img.get_pixel(2, 1)
        assert list(pixel) == [100, 150, 200]

    def test_bgr_to_rgb_roundtrip(self):
        data = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        img = Image.from_array(data.copy())
        rgb = img.bgr_to_rgb()
        bgr = rgb.rgb_to_bgr()
        original = bgr.to_array()
        np.testing.assert_array_equal(original, data)

    def test_resize(self):
        data = np.full((10, 10, 3), 128, dtype=np.uint8)
        img = Image.from_array(data)
        resized = img.resize(5, 5)
        assert resized.width == 5
        assert resized.height == 5

    def test_crop(self):
        data = np.arange(300, dtype=np.uint8).reshape(10, 10, 3)
        img = Image.from_array(data)
        cropped = img.crop(2, 3, 4, 5)
        assert cropped.width == 4
        assert cropped.height == 5
        # First pixel of cropped should be data[3, 2]
        recovered = cropped.to_array()
        np.testing.assert_array_equal(recovered[0, 0], data[3, 2])

    def test_repr(self):
        img = Image(5, 5)
        repr_str = repr(img)
        assert "width=5" in repr_str
        assert "height=5" in repr_str


# ─── Affine Transform Tests ────────────────────────────────────────────────


class TestAffineTransform:
    def test_invert_affine_identity(self):
        mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        inv = py_invert_affine_2x3(mat)
        assert inv is not None
        for i in range(2):
            for j in range(3):
                assert abs(inv[i][j] - mat[i][j]) < EPS

    def test_invert_affine_translation(self):
        mat = [[1.0, 0.0, 100.0], [0.0, 1.0, 50.0]]
        inv = py_invert_affine_2x3(mat)
        assert inv is not None
        assert abs(inv[0][2] - (-100.0)) < EPS
        assert abs(inv[1][2] - (-50.0)) < EPS

    def test_invert_affine_scale(self):
        mat = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
        inv = py_invert_affine_2x3(mat)
        assert inv is not None
        assert abs(inv[0][0] - 0.5) < EPS
        assert abs(inv[1][1] - 1.0 / 3.0) < EPS

    def test_invert_affine_singular(self):
        mat = [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]
        assert py_invert_affine_2x3(mat) is None

    def test_warp_affine_identity(self):
        data = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        img = Image.from_array(data)
        mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        result = py_warp_affine(img, mat, 10, 10)
        out = result.to_array()
        # Should be very close to original
        diff = np.abs(out.astype(int) - data.astype(int)).max()
        assert diff <= 2

    def test_warp_affine_translation(self):
        data = np.full((10, 10, 3), [100, 150, 200], dtype=np.uint8)
        img = Image.from_array(data)
        mat = [[1.0, 0.0, 3.0], [0.0, 1.0, 2.0]]
        result = py_warp_affine(img, mat, 10, 10)
        # Pixel at (5,5) should come from (2,3)
        assert result.get_pixel(5, 5) == img.get_pixel(2, 3)

    def test_warp_affine_out_of_bounds_black(self):
        data = np.full((5, 5, 3), 255, dtype=np.uint8)
        img = Image.from_array(data)
        mat = [[1.0, 0.0, 100.0], [0.0, 1.0, 100.0]]
        result = py_warp_affine(img, mat, 5, 5)
        out = result.to_array()
        assert np.all(out == 0)


# ─── Perspective Transform Tests ───────────────────────────────────────────


class TestPerspectiveTransform:
    def test_get_perspective_transform_identity(self):
        src = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        dst = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        mat = py_get_perspective_transform(src, dst)
        # Should be identity
        assert abs(mat[0][0] - mat[2][2]) < EPS
        assert abs(mat[0][1]) < EPS
        assert abs(mat[1][1] - mat[2][2]) < EPS

    def test_get_perspective_transform_translation(self):
        src = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        dst = [[10.0, 20.0], [11.0, 20.0], [10.0, 21.0], [11.0, 21.0]]
        mat = py_get_perspective_transform(src, dst)
        pts = py_perspective_transform_points([[0.0, 0.0]], mat)
        assert abs(pts[0][0] - 10.0) < EPS
        assert abs(pts[0][1] - 20.0) < EPS

    def test_perspective_transform_points(self):
        src = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        dst = [[10.0, 10.0], [20.0, 10.0], [10.0, 20.0], [20.0, 20.0]]
        mat = py_get_perspective_transform(src, dst)
        pts = py_perspective_transform_points([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], mat)
        assert len(pts) == 3
        assert abs(pts[0][0] - 10.0) < EPS
        assert abs(pts[1][0] - 15.0) < EPS
        assert abs(pts[2][0] - 20.0) < EPS

    def test_invert_perspective_identity(self):
        mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        inv = py_invert_perspective_3x3(mat)
        assert inv is not None
        for i in range(3):
            for j in range(3):
                assert abs(inv[i][j] - mat[i][j]) < EPS

    def test_invert_perspective_roundtrip(self):
        mat = [[1.5, 0.2, 10.0], [-0.1, 1.3, 5.0], [0.001, 0.002, 1.0]]
        inv = py_invert_perspective_3x3(mat)
        assert inv is not None
        inv_inv = py_invert_perspective_3x3(inv)
        assert inv_inv is not None
        for i in range(3):
            for j in range(3):
                assert abs(inv_inv[i][j] - mat[i][j]) < 0.001

    def test_invert_perspective_singular(self):
        mat = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        assert py_invert_perspective_3x3(mat) is None

    def test_warp_perspective_identity(self):
        data = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        img = Image.from_array(data)
        mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        result = py_warp_perspective(img, mat, 10, 10)
        out = result.to_array()
        diff = np.abs(out.astype(int) - data.astype(int)).max()
        assert diff <= 2

    def test_warp_perspective_corner_mapping(self):
        # Create image with distinct corner colors
        data = np.zeros((100, 100, 3), dtype=np.uint8)
        data[0, 0] = [255, 0, 0]  # TL: Red
        data[0, 99] = [0, 255, 0]  # TR: Green
        data[99, 0] = [0, 0, 255]  # BL: Blue
        data[99, 99] = [255, 255, 0]  # BR: Yellow
        img = Image.from_array(data)

        src = [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]]
        dst = [[0.0, 0.0], [50.0, 0.0], [0.0, 50.0], [50.0, 50.0]]
        mat = py_get_perspective_transform(src, dst)
        result = py_warp_perspective(img, mat, 50, 50)

        tl = result.get_pixel(0, 0)
        assert tl[0] > tl[1] and tl[0] > tl[2], "TL should be red"

    def test_perspective_transform_roundtrip(self):
        src = [[10.0, 20.0], [100.0, 15.0], [25.0, 80.0], [90.0, 75.0]]
        dst = [[5.0, 10.0], [200.0, 5.0], [15.0, 150.0], [180.0, 140.0]]
        mat = py_get_perspective_transform(src, dst)
        inv = py_invert_perspective_3x3(mat)
        assert inv is not None

        transformed = py_perspective_transform_points(src, mat)
        recovered = py_perspective_transform_points(transformed, inv)

        for i in range(4):
            assert abs(recovered[i][0] - src[i][0]) < 0.01
            assert abs(recovered[i][1] - src[i][1]) < 0.01


# ─── Detector Tests ────────────────────────────────────────────────────────


class TestDetector:
    def test_check_keypoints_valid(self):
        keypoints = [[10.0, 20.0], [100.0, 15.0], [25.0, 80.0], [90.0, 75.0]]
        py_check_keypoints(keypoints)  # Should not raise

    def test_check_keypoints_invalid(self):
        with pytest.raises(ValueError):
            py_check_keypoints([[10.0, 20.0], [100.0, 15.0], [25.0, 80.0]])

    def test_get_board_corner_points(self):
        keypoints = [
            [10.0, 20.0],
            [100.0, 20.0],
            [10.0, 100.0],
            [100.0, 100.0],
        ]
        corners = py_get_board_corner_points(keypoints)
        assert corners[0] == [10.0, 20.0]
        assert corners[1] == [100.0, 20.0]
        assert corners[2] == [10.0, 100.0]
        assert corners[3] == [100.0, 100.0]

    def test_corner_points_to_rect(self):
        corners = [[0.0, 0.0], [400.0, 0.0], [0.0, 400.0], [400.0, 400.0]]
        ((min_x, min_y), (max_x, max_y)) = py_corner_points_to_rect(corners)
        assert min_x == 0
        assert max_x == 450
        assert min_y == 0
        assert max_y == 444

    def test_extract_chessboard(self):
        data = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        img = Image.from_array(data)
        keypoints = [[50.0, 50.0], [450.0, 50.0], [50.0, 450.0], [450.0, 450.0]]
        result, kps, dst_corners = py_extract_chessboard(img, keypoints)
        assert result.width == 450
        assert result.height == 500
        assert len(kps) == 4
        assert dst_corners[0] == [50.0, 50.0]

    def test_perspective_transform_full(self):
        data = np.full((200, 200, 3), 128, dtype=np.uint8)
        img = Image.from_array(data)
        src_points = [[0.0, 0.0], [200.0, 0.0], [0.0, 200.0], [200.0, 200.0]]
        keypoints = [[0.0, 0.0], [200.0, 0.0], [0.0, 200.0], [200.0, 200.0]]
        result, kps, dst_corners = py_perspective_transform(
            img, src_points, keypoints, (300, 300)
        )
        assert result.width == 300
        assert result.height == 300
        assert len(kps) == 4
        assert dst_corners[0] == [50.0, 50.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
