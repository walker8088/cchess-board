use cchess_board_rs::base_onnx::{BaseOnnxError, Image};
use cchess_board_rs::detector::{
    check_keypoints, corner_points_to_rect, extract_chessboard, get_board_corner_points,
    perspective_transform,
};

// ─── Keypoint Validation ───────────────────────────────────────────────────

#[test]
fn test_check_keypoints_valid() {
    let keypoints = [[10.0, 20.0], [100.0, 15.0], [25.0, 80.0], [90.0, 75.0]];
    assert!(check_keypoints(&keypoints).is_ok());
}

#[test]
fn test_check_keypoints_too_few() {
    let keypoints = [[10.0, 20.0], [100.0, 15.0], [25.0, 80.0]];
    let result = check_keypoints(&keypoints);
    assert!(result.is_err());
    match result.unwrap_err() {
        BaseOnnxError::DimMismatch { expected, actual } => {
            assert!(expected.contains("4"));
            assert!(actual.contains("3"));
        }
        _ => panic!("Expected DimMismatch error"),
    }
}

#[test]
fn test_check_keypoints_too_many() {
    let keypoints = [
        [10.0, 20.0],
        [100.0, 15.0],
        [25.0, 80.0],
        [90.0, 75.0],
        [50.0, 50.0],
    ];
    assert!(check_keypoints(&keypoints).is_err());
}

#[test]
fn test_check_keypoints_empty() {
    let keypoints: [[f32; 2]; 0] = [];
    assert!(check_keypoints(&keypoints).is_err());
}

// ─── Board Corner Points ──────────────────────────────────────────────────

#[test]
fn test_get_board_corner_points_aligned() {
    // Keypoints in order: A0, A8, J0, J8 (top-left, top-right, bottom-left, bottom-right)
    let keypoints = [
        [10.0, 20.0],   // A0
        [100.0, 20.0],  // A8
        [10.0, 100.0],  // J0
        [100.0, 100.0], // J8
    ];
    let corners = get_board_corner_points(&keypoints).unwrap();
    // Should return: [TL, TR, BL, BR] as [min_x, min_y], [max_x, min_y], [min_x, max_y], [max_x, max_y]
    assert_eq!(corners[0], [10.0, 20.0]); // TL
    assert_eq!(corners[1], [100.0, 20.0]); // TR
    assert_eq!(corners[2], [10.0, 100.0]); // BL
    assert_eq!(corners[3], [100.0, 100.0]); // BR
}

#[test]
fn test_get_board_corner_points_skewed() {
    // Skewed keypoints (like a real camera perspective)
    let keypoints = [
        [15.0, 25.0],   // A0 - slightly inset
        [95.0, 18.0],   // A8 - slightly higher
        [20.0, 95.0],   // J0 - slightly right
        [100.0, 105.0], // J8 - slightly lower
    ];
    let corners = get_board_corner_points(&keypoints).unwrap();
    assert_eq!(corners[0], [15.0, 18.0]); // TL: min_x=15, min_y=18
    assert_eq!(corners[1], [100.0, 18.0]); // TR: max_x=100, min_y=18
    assert_eq!(corners[2], [15.0, 105.0]); // BL: min_x=15, max_y=105
    assert_eq!(corners[3], [100.0, 105.0]); // BR: max_x=100, max_y=105
}

#[test]
fn test_get_board_corner_points_invalid_keypoints() {
    let keypoints = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]; // Only 3 points
    assert!(get_board_corner_points(&keypoints).is_err());
}

#[test]
fn test_get_board_corner_points_negative_coords() {
    let keypoints = [[-10.0, -20.0], [50.0, -15.0], [-5.0, 60.0], [55.0, 65.0]];
    let corners = get_board_corner_points(&keypoints).unwrap();
    assert_eq!(corners[0], [-10.0, -20.0]);
    assert_eq!(corners[1], [55.0, -20.0]);
    assert_eq!(corners[2], [-10.0, 65.0]);
    assert_eq!(corners[3], [55.0, 65.0]);
}

// ─── Corner Points to Rect ────────────────────────────────────────────────

#[test]
fn test_corner_points_to_rect_square() {
    // 400x400 square centered at origin offset
    let corners = [[0.0, 0.0], [400.0, 0.0], [0.0, 400.0], [400.0, 400.0]];
    let ((min_x, min_y), (max_x, max_y)) = corner_points_to_rect(&corners);
    // grid_x_half = 400/8 = 50, grid_y_half = 400/9 ≈ 44.44
    // Expanded by half grid: min_x = max(0, 0-50) = 0, max_x = 400+50 = 450
    // min_y = max(0, 0-44) = 0, max_y = 400+44 = 444
    assert_eq!(min_x, 0);
    assert_eq!(max_x, 450);
    assert_eq!(min_y, 0);
    assert_eq!(max_y, 444);
}

#[test]
fn test_corner_points_to_rect_offset() {
    // 80x90 rect offset by (100, 200)
    let corners = [
        [100.0, 200.0],
        [180.0, 200.0],
        [100.0, 290.0],
        [180.0, 290.0],
    ];
    let ((min_x, min_y), (max_x, max_y)) = corner_points_to_rect(&corners);
    // grid_x_half = 80/8 = 10, grid_y_half = 90/9 = 10
    // min_x = max(0, 100-10) = 90
    // max_x = 180+10 = 190
    // min_y = max(0, 200-10) = 190
    // max_y = 290+10 = 300
    assert_eq!(min_x, 90);
    assert_eq!(max_x, 190);
    assert_eq!(min_y, 190);
    assert_eq!(max_y, 300);
}

#[test]
fn test_corner_points_to_rect_near_edge_clamps() {
    // Corners near (0,0) should not produce negative coordinates
    let corners = [[5.0, 3.0], [13.0, 3.0], [5.0, 12.0], [13.0, 12.0]];
    let ((min_x, min_y), (max_x, max_y)) = corner_points_to_rect(&corners);
    // grid_x_half = 8/8 = 1, grid_y_half = 9/9 = 1
    // min_x = max(0, 5-1) = 4, min_y = max(0, 3-1) = 2
    assert_eq!(min_x, 4);
    assert_eq!(min_y, 2);
    assert_eq!(max_x, 14);
    assert_eq!(max_y, 13);
}

#[test]
fn test_corner_points_to_rect_small_board() {
    // Very small board (8x9 pixels - one pixel per cell)
    let corners = [[0.0, 0.0], [8.0, 0.0], [0.0, 9.0], [8.0, 9.0]];
    let ((min_x, min_y), (max_x, max_y)) = corner_points_to_rect(&corners);
    // grid_x_half = 1, grid_y_half = 1
    // Expanded: min_x = max(0, 0-1) = 0, max_x = 8+1 = 9
    // min_y = max(0, 0-1) = 0, max_y = 9+1 = 10
    assert_eq!(min_x, 0);
    assert_eq!(max_x, 9);
    assert_eq!(min_y, 0);
    assert_eq!(max_y, 10);
}

// ─── Extract Chessboard ──────────────────────────────────────────────────

#[test]
fn test_extract_chessboard_invalid_keypoints() {
    let img = Image::new(100, 100);
    let keypoints = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]; // Only 3
    assert!(extract_chessboard(&img, &keypoints).is_err());
}

#[test]
fn test_extract_chessboard_valid_returns_correct_size() {
    let mut img = Image::new(500, 500);
    // Fill with recognizable pattern
    for y in 0..500 {
        for x in 0..500 {
            img.set_pixel(x, y, [x as u8, y as u8, 128]);
        }
    }
    // Keypoints defining a chessboard region
    let keypoints = [[50.0, 50.0], [450.0, 50.0], [50.0, 450.0], [450.0, 450.0]];
    let (result, transformed_kps, dst_corners) = extract_chessboard(&img, &keypoints).unwrap();
    // Output should be 450x500 as defined in the function
    assert_eq!(result.width, 450);
    assert_eq!(result.height, 500);
    // Transformed keypoints should have same count
    assert_eq!(transformed_kps.len(), 4);
    // Dest corners should have padding applied
    assert_eq!(dst_corners[0], [50.0, 50.0]); // padding, padding
    assert_eq!(dst_corners[1], [400.0, 50.0]); // dw-padding, padding
    assert_eq!(dst_corners[2], [50.0, 450.0]); // padding, dh-padding
    assert_eq!(dst_corners[3], [400.0, 450.0]); // dw-padding, dh-padding
}

#[test]
fn test_extract_chessboard_preserves_content() {
    let mut img = Image::new(200, 200);
    // Fill with uniform color for simplicity
    for y in 0..200 {
        for x in 0..200 {
            img.set_pixel(x, y, [100, 150, 200]);
        }
    }
    let keypoints = [[20.0, 30.0], [180.0, 25.0], [25.0, 170.0], [175.0, 175.0]];
    let (result, _, _) = extract_chessboard(&img, &keypoints).unwrap();
    // Check center pixel is approximately the source color
    let center = result.get_pixel(225, 250);
    // Due to interpolation, values should be close
    for c in 0..3 {
        let diff = (center[c] as i32 - [100, 150, 200][c] as i32).abs();
        assert!(diff <= 5, "Center pixel channel {} diff={}", c, diff);
    }
}

// ─── Perspective Transform ───────────────────────────────────────────────

#[test]
fn test_perspective_transform_invalid_keypoints() {
    let img = Image::new(100, 100);
    let src_points = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
    let keypoints = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]; // Only 3
    assert!(perspective_transform(&img, src_points, &keypoints, (100, 100)).is_err());
}

#[test]
fn test_perspective_transform_identity() {
    let mut img = Image::new(100, 100);
    for y in 0..100 {
        for x in 0..100 {
            img.set_pixel(x, y, [x as u8, y as u8, 50]);
        }
    }
    let src_points = [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]];
    let keypoints = [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]];
    // Use (200, 200) so padding=50 doesn't collapse corners
    let (result, _, dst_corners) =
        perspective_transform(&img, src_points, &keypoints, (200, 200)).unwrap();

    assert_eq!(result.width, 200);
    assert_eq!(result.height, 200);
    // Corners should have padding: [50,50], [150,50], [50,150], [150,150]
    assert_eq!(dst_corners[0], [50.0, 50.0]);
    assert_eq!(dst_corners[1], [150.0, 50.0]);
    assert_eq!(dst_corners[2], [50.0, 150.0]);
    assert_eq!(dst_corners[3], [150.0, 150.0]);
    // Check center content is approximately preserved
    let center_src = img.get_pixel(50, 50);
    let center_dst = result.get_pixel(100, 100);
    for c in 0..3 {
        let diff = (center_dst[c] as i32 - center_src[c] as i32).abs();
        assert!(diff <= 3, "Center pixel channel {} diff={}", c, diff);
    }
}
