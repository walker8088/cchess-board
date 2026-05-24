use cchess_board_rs::base_onnx::Image;
use cchess_board_rs::rtmpose::{
    get_perspective_transform, invert_affine_2x3, invert_perspective_3x3,
    perspective_transform_points, solve_linear_system, warp_affine, warp_perspective, AffineMatrix,
    PerspectiveMatrix,
};
use ndarray::Array2;

const EPSILON: f32 = 1e-4;

// ─── Affine Matrix Inversion ────────────────────────────────────────────────

#[test]
fn test_invert_affine_identity() {
    // Identity transform: [[1,0,0],[0,1,0]]
    let mat: AffineMatrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let inv = invert_affine_2x3(&mat).unwrap();
    for i in 0..2 {
        for j in 0..3 {
            assert!(
                (inv[i][j] - mat[i][j]).abs() < EPSILON,
                "inv[{}][{}] = {} != {} = mat[{}][{}]",
                i,
                j,
                inv[i][j],
                mat[i][j],
                i,
                j
            );
        }
    }
}

#[test]
fn test_invert_affine_translation() {
    // Translation by (100, 50): [[1,0,100],[0,1,50]]
    let mat: AffineMatrix = [[1.0, 0.0, 100.0], [0.0, 1.0, 50.0]];
    let inv = invert_affine_2x3(&mat).unwrap();
    // Inverse should be translation by (-100, -50)
    assert!((inv[0][2] - (-100.0)).abs() < EPSILON);
    assert!((inv[1][2] - (-50.0)).abs() < EPSILON);
    // Linear part should remain identity
    assert!((inv[0][0] - 1.0).abs() < EPSILON);
    assert!((inv[0][1] - 0.0).abs() < EPSILON);
    assert!((inv[1][0] - 0.0).abs() < EPSILON);
    assert!((inv[1][1] - 1.0).abs() < EPSILON);
}

#[test]
fn test_invert_affine_scale() {
    // Scale by (2, 3): [[2,0,0],[0,3,0]]
    let mat: AffineMatrix = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]];
    let inv = invert_affine_2x3(&mat).unwrap();
    assert!((inv[0][0] - 0.5).abs() < EPSILON);
    assert!((inv[1][1] - 1.0 / 3.0).abs() < EPSILON);
}

#[test]
fn test_invert_affine_scale_and_translate() {
    // Scale by 2 and translate by (10, 20): [[2,0,10],[0,2,20]]
    let mat: AffineMatrix = [[2.0, 0.0, 10.0], [0.0, 2.0, 20.0]];
    let inv = invert_affine_2x3(&mat).unwrap();
    // Inverse: scale by 0.5, translate by (-5, -10)
    assert!((inv[0][0] - 0.5).abs() < EPSILON);
    assert!((inv[1][1] - 0.5).abs() < EPSILON);
    assert!((inv[0][2] - (-5.0)).abs() < EPSILON);
    assert!((inv[1][2] - (-10.0)).abs() < EPSILON);
}

#[test]
fn test_invert_affine_singular_returns_none() {
    // Singular matrix: [[0,0,0],[0,0,0]]
    let mat: AffineMatrix = [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]];
    assert!(invert_affine_2x3(&mat).is_none());
}

#[test]
fn test_invert_affine_roundtrip() {
    // Apply matrix then inverse should give identity
    let mat: AffineMatrix = [[1.5, 0.3, 10.0], [-0.2, 1.2, -5.0]];
    let inv = invert_affine_2x3(&mat).unwrap();
    let inv_inv = invert_affine_2x3(&inv).unwrap();
    for i in 0..2 {
        for j in 0..3 {
            assert!(
                (inv_inv[i][j] - mat[i][j]).abs() < EPSILON,
                "Double inversion mismatch at [{i}][{j}]"
            );
        }
    }
}

// ─── Linear System Solver ───────────────────────────────────────────────────

#[test]
fn test_solve_linear_system_2x2() {
    // 2x + 3y = 8, 4x + y = 6 => x=1, y=2
    let mut a = Array2::<f32>::zeros((2, 2));
    a[[0, 0]] = 2.0;
    a[[0, 1]] = 3.0;
    a[[1, 0]] = 4.0;
    a[[1, 1]] = 1.0;
    let mut b = Array2::<f32>::zeros((2, 1));
    b[[0, 0]] = 8.0;
    b[[1, 0]] = 6.0;
    let x = solve_linear_system(&a, &b).unwrap();
    assert!((x[0] - 1.0).abs() < EPSILON);
    assert!((x[1] - 2.0).abs() < EPSILON);
}

#[test]
fn test_solve_linear_system_3x3() {
    // x + y + z = 6, 2y + z = 5, 3z = 3 => z=1, y=2, x=3
    let mut a = Array2::<f32>::zeros((3, 3));
    a[[0, 0]] = 1.0;
    a[[0, 1]] = 1.0;
    a[[0, 2]] = 1.0;
    a[[1, 0]] = 0.0;
    a[[1, 1]] = 2.0;
    a[[1, 2]] = 1.0;
    a[[2, 0]] = 0.0;
    a[[2, 1]] = 0.0;
    a[[2, 2]] = 3.0;
    let mut b = Array2::<f32>::zeros((3, 1));
    b[[0, 0]] = 6.0;
    b[[1, 0]] = 5.0;
    b[[2, 0]] = 3.0;
    let x = solve_linear_system(&a, &b).unwrap();
    assert!((x[0] - 3.0).abs() < EPSILON);
    assert!((x[1] - 2.0).abs() < EPSILON);
    assert!((x[2] - 1.0).abs() < EPSILON);
}

#[test]
fn test_solve_linear_system_identity() {
    let a = Array2::<f32>::eye(4);
    let mut b = Array2::<f32>::zeros((4, 1));
    b[[0, 0]] = 1.0;
    b[[1, 0]] = 2.0;
    b[[2, 0]] = 3.0;
    b[[3, 0]] = 4.0;
    let x = solve_linear_system(&a, &b).unwrap();
    for i in 0..4 {
        assert!((x[i] - (i as f32 + 1.0)).abs() < EPSILON);
    }
}

#[test]
fn test_solve_linear_system_singular_returns_none() {
    let mut a = Array2::<f32>::zeros((2, 2));
    a[[0, 0]] = 1.0;
    a[[0, 1]] = 2.0;
    a[[1, 0]] = 2.0;
    a[[1, 1]] = 4.0; // Row 2 = 2 * Row 1, singular
    let b = Array2::<f32>::from_elem((2, 1), 1.0);
    assert!(solve_linear_system(&a, &b).is_none());
}

// ─── Perspective Transform ─────────────────────────────────────────────────

#[test]
fn test_get_perspective_transform_identity() {
    let src = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let dst = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let mat = get_perspective_transform(src, dst);
    // Should be identity (up to scale)
    assert!((mat[0][0] - mat[2][2]).abs() < EPSILON);
    assert!(mat[0][1].abs() < EPSILON);
    assert!(mat[0][2].abs() < EPSILON);
    assert!(mat[1][0].abs() < EPSILON);
    assert!((mat[1][1] - mat[2][2]).abs() < EPSILON);
    assert!(mat[1][2].abs() < EPSILON);
    assert!(mat[2][0].abs() < EPSILON);
    assert!(mat[2][1].abs() < EPSILON);
}

#[test]
fn test_get_perspective_transform_translation() {
    let src = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let dst = [[10.0, 20.0], [11.0, 20.0], [10.0, 21.0], [11.0, 21.0]];
    let mat = get_perspective_transform(src, dst);
    // Apply to (0,0) should give (10, 20)
    let pts = perspective_transform_points(&[[0.0, 0.0]], &mat);
    assert!((pts[0][0] - 10.0).abs() < EPSILON);
    assert!((pts[0][1] - 20.0).abs() < EPSILON);
    // Apply to (1,1) should give (11, 21)
    let pts = perspective_transform_points(&[[1.0, 1.0]], &mat);
    assert!((pts[0][0] - 11.0).abs() < EPSILON);
    assert!((pts[0][1] - 21.0).abs() < EPSILON);
}

#[test]
fn test_get_perspective_transform_scale() {
    let src = [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]];
    let dst = [[0.0, 0.0], [200.0, 0.0], [0.0, 200.0], [200.0, 200.0]];
    let mat = get_perspective_transform(src, dst);
    // Scale by 2
    let pts = perspective_transform_points(&[[50.0, 50.0]], &mat);
    assert!((pts[0][0] - 100.0).abs() < 0.01);
    assert!((pts[0][1] - 100.0).abs() < 0.01);
}

#[test]
fn test_perspective_transform_points_multiple() {
    let src = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let dst = [[10.0, 10.0], [20.0, 10.0], [10.0, 20.0], [20.0, 20.0]];
    let mat = get_perspective_transform(src, dst);
    let pts = perspective_transform_points(&[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], &mat);
    assert_eq!(pts.len(), 3);
    assert!((pts[0][0] - 10.0).abs() < EPSILON);
    assert!((pts[1][0] - 15.0).abs() < EPSILON);
    assert!((pts[2][0] - 20.0).abs() < EPSILON);
}

// ─── Perspective Matrix Inversion ───────────────────────────────────────────

#[test]
fn test_invert_perspective_identity() {
    let mat: PerspectiveMatrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let inv = invert_perspective_3x3(&mat).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!((inv[i][j] - mat[i][j]).abs() < EPSILON);
        }
    }
}

#[test]
fn test_invert_perspective_roundtrip() {
    let mat: PerspectiveMatrix = [[1.5, 0.2, 10.0], [-0.1, 1.3, 5.0], [0.001, 0.002, 1.0]];
    let inv = invert_perspective_3x3(&mat).unwrap();
    let inv_inv = invert_perspective_3x3(&inv).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (inv_inv[i][j] - mat[i][j]).abs() < 0.001,
                "Double inversion mismatch at [{i}][{j}]"
            );
        }
    }
}

#[test]
fn test_invert_perspective_singular_returns_none() {
    let mat: PerspectiveMatrix = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    assert!(invert_perspective_3x3(&mat).is_none());
}

// ─── Warp Affine ────────────────────────────────────────────────────────────

#[test]
fn test_warp_affine_identity() {
    let mut img = Image::new(5, 5);
    for y in 0..5 {
        for x in 0..5 {
            img.set_pixel(x, y, [(x * 50) as u8, (y * 40) as u8, 100]);
        }
    }
    let mat: AffineMatrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let result = warp_affine(&img, &mat, 5, 5).unwrap();
    assert_eq!(result.width, 5);
    assert_eq!(result.height, 5);
    // Should be nearly identical to original
    for y in 0..5 {
        for x in 0..5 {
            let orig = img.get_pixel(x, y);
            let warped = result.get_pixel(x, y);
            for c in 0..3 {
                let diff = (warped[c] as i32 - orig[c] as i32).abs();
                assert!(
                    diff <= 2,
                    "Warp identity pixel ({},{}) channel {} diff={}",
                    x,
                    y,
                    c,
                    diff
                );
            }
        }
    }
}

#[test]
fn test_warp_affine_translation() {
    let mut img = Image::new(10, 10);
    // Fill with a distinctive pattern
    for y in 0..10 {
        for x in 0..10 {
            img.set_pixel(x, y, [200, 100, 50]);
        }
    }
    // Translate by (3, 2)
    let mat: AffineMatrix = [[1.0, 0.0, 3.0], [0.0, 1.0, 2.0]];
    let result = warp_affine(&img, &mat, 10, 10).unwrap();
    // Pixel at (5,5) in result should come from (2,3) in source
    let src_pixel = img.get_pixel(2, 3);
    let dst_pixel = result.get_pixel(5, 5);
    assert_eq!(dst_pixel, src_pixel);
}

#[test]
fn test_warp_affine_out_of_bounds_black() {
    let mut img = Image::new(5, 5);
    for y in 0..5 {
        for x in 0..5 {
            img.set_pixel(x, y, [255, 255, 255]);
        }
    }
    // Large translation should push source out of bounds
    let mat: AffineMatrix = [[1.0, 0.0, 100.0], [0.0, 1.0, 100.0]];
    let result = warp_affine(&img, &mat, 5, 5).unwrap();
    // All pixels should be black
    for y in 0..5 {
        for x in 0..5 {
            assert_eq!(result.get_pixel(x, y), [0, 0, 0]);
        }
    }
}

#[test]
fn test_warp_affine_scale_down() {
    let mut img = Image::new(10, 10);
    for y in 0..10 {
        for x in 0..10 {
            img.set_pixel(x, y, [x as u8 * 25, y as u8 * 25, 128]);
        }
    }
    // Scale down by 2: output (x,y) maps to source (2x, 2y)
    let mat: AffineMatrix = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]];
    let result = warp_affine(&img, &mat, 5, 5).unwrap();
    assert_eq!(result.width, 5);
    assert_eq!(result.height, 5);
    // Output (2,2) maps to source (4,4) via inverse matrix
    let src_pixel = img.get_pixel(4, 4);
    let dst_pixel = result.get_pixel(2, 2);
    for c in 0..3 {
        let diff = (dst_pixel[c] as i32 - src_pixel[c] as i32).abs();
        assert!(
            diff <= 2,
            "Scale warp pixel (2,2) channel {} diff={}",
            c,
            diff
        );
    }
}

// ─── Warp Perspective ──────────────────────────────────────────────────────

#[test]
fn test_warp_perspective_identity() {
    let mut img = Image::new(6, 6);
    for y in 0..6 {
        for x in 0..6 {
            img.set_pixel(x, y, [(x * 40) as u8, (y * 35) as u8, 150]);
        }
    }
    let mat: PerspectiveMatrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let result = warp_perspective(&img, &mat, 6, 6).unwrap();
    for y in 0..6 {
        for x in 0..6 {
            let orig = img.get_pixel(x, y);
            let warped = result.get_pixel(x, y);
            for c in 0..3 {
                let diff = (warped[c] as i32 - orig[c] as i32).abs();
                assert!(
                    diff <= 2,
                    "Perspective identity pixel ({},{}) channel {} diff={}",
                    x,
                    y,
                    c,
                    diff
                );
            }
        }
    }
}

#[test]
fn test_warp_perspective_maps_corners_correctly() {
    // Create a 100x100 image with distinct corner colors
    let mut img = Image::new(100, 100);
    img.set_pixel(0, 0, [255, 0, 0]); // TL: Red
    img.set_pixel(99, 0, [0, 255, 0]); // TR: Green
    img.set_pixel(0, 99, [0, 0, 255]); // BL: Blue
    img.set_pixel(99, 99, [255, 255, 0]); // BR: Yellow
                                          // Fill interior with gray
    for y in 1..99 {
        for x in 1..99 {
            img.set_pixel(x, y, [128, 128, 128]);
        }
    }

    // Map to 50x50 output
    let src = [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]];
    let dst = [[0.0, 0.0], [50.0, 0.0], [0.0, 50.0], [50.0, 50.0]];
    let mat = get_perspective_transform(src, dst);
    let result = warp_perspective(&img, &mat, 50, 50).unwrap();

    // Check corners of output
    let tl = result.get_pixel(0, 0);
    let br = result.get_pixel(49, 49);
    // TL should be predominantly red
    assert!(tl[0] > tl[1] && tl[0] > tl[2], "TL should be red");
    // BR should have high R and G (yellow-ish)
    assert!(br[0] > 100 && br[1] > 100, "BR should be yellow-ish");
}

#[test]
fn test_warp_perspective_out_of_bounds_black() {
    let mut img = Image::new(10, 10);
    for y in 0..10 {
        for x in 0..10 {
            img.set_pixel(x, y, [255, 255, 255]);
        }
    }
    // Extreme perspective should push everything out of bounds
    let mat: PerspectiveMatrix = [[0.001, 0.0, 500.0], [0.0, 0.001, 500.0], [0.0, 0.0, 1.0]];
    let result = warp_perspective(&img, &mat, 10, 10).unwrap();
    for y in 0..10 {
        for x in 0..10 {
            assert_eq!(
                result.get_pixel(x, y),
                [0, 0, 0],
                "Pixel ({},{}) should be black",
                x,
                y
            );
        }
    }
}

// ─── Combined: Perspective Transform Roundtrip ─────────────────────────────

#[test]
fn test_perspective_transform_roundtrip_points() {
    let src = [[10.0, 20.0], [100.0, 15.0], [25.0, 80.0], [90.0, 75.0]];
    let dst = [[5.0, 10.0], [200.0, 5.0], [15.0, 150.0], [180.0, 140.0]];
    let mat = get_perspective_transform(src, dst);
    let inv = invert_perspective_3x3(&mat).unwrap();

    // Transform src to dst, then back
    let transformed = perspective_transform_points(&src, &mat);
    let recovered = perspective_transform_points(&transformed, &inv);

    for i in 0..4 {
        assert!(
            (recovered[i][0] - src[i][0]).abs() < 0.01,
            "X mismatch for point {}: expected {}, got {}",
            i,
            src[i][0],
            recovered[i][0]
        );
        assert!(
            (recovered[i][1] - src[i][1]).abs() < 0.01,
            "Y mismatch for point {}: expected {}, got {}",
            i,
            src[i][1],
            recovered[i][1]
        );
    }
}

#[test]
fn test_perspective_transform_chessboard_corners() {
    // Simulate chessboard corner detection
    let src = [[50.0, 60.0], [400.0, 55.0], [45.0, 350.0], [395.0, 345.0]];
    let dst = [[50.0, 50.0], [450.0, 50.0], [50.0, 500.0], [450.0, 500.0]];
    let mat = get_perspective_transform(src, dst);

    // Transform the corners
    let transformed = perspective_transform_points(&src, &mat);

    // Check they map close to dst
    for i in 0..4 {
        assert!(
            (transformed[i][0] - dst[i][0]).abs() < 0.1,
            "Corner {} X: expected {}, got {}",
            i,
            dst[i][0],
            transformed[i][0]
        );
        assert!(
            (transformed[i][1] - dst[i][1]).abs() < 0.1,
            "Corner {} Y: expected {}, got {}",
            i,
            dst[i][1],
            transformed[i][1]
        );
    }
}
