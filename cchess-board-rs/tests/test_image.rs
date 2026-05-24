use cchess_board_rs::base_onnx::Image;

// ─── Construction ───────────────────────────────────────────────────────────

#[test]
fn test_image_new_creates_zeroed_buffer() {
    let img = Image::new(10, 20);
    assert_eq!(img.width, 10);
    assert_eq!(img.height, 20);
    assert_eq!(img.data.len(), 10 * 20 * 3);
    assert!(img.data.iter().all(|&b| b == 0));
}

#[test]
fn test_image_from_vec_valid() {
    let data = vec![255u8; 12]; // 2x2x3
    let img = Image::from_vec(data.clone(), 2, 2).unwrap();
    assert_eq!(img.width, 2);
    assert_eq!(img.height, 2);
    assert_eq!(img.data, data);
}

#[test]
fn test_image_from_vec_invalid_length() {
    let data = vec![0u8; 10]; // Not divisible by 3 for any WxH
    assert!(Image::from_vec(data, 2, 2).is_err());
}

#[test]
fn test_image_from_vec_mismatched_dimensions() {
    let data = vec![0u8; 18]; // 3x2x3
    assert!(Image::from_vec(data, 3, 3).is_err()); // expects 27 bytes
}

// ─── Pixel Access ───────────────────────────────────────────────────────────

#[test]
fn test_get_set_pixel() {
    let mut img = Image::new(5, 5);
    img.set_pixel(2, 3, [100, 150, 200]);
    let px = img.get_pixel(2, 3);
    assert_eq!(px, [100, 150, 200]);
}

#[test]
fn test_get_set_pixel_corners() {
    let mut img = Image::new(3, 3);
    img.set_pixel(0, 0, [1, 2, 3]);
    img.set_pixel(2, 0, [4, 5, 6]);
    img.set_pixel(0, 2, [7, 8, 9]);
    img.set_pixel(2, 2, [10, 11, 12]);
    assert_eq!(img.get_pixel(0, 0), [1, 2, 3]);
    assert_eq!(img.get_pixel(2, 0), [4, 5, 6]);
    assert_eq!(img.get_pixel(0, 2), [7, 8, 9]);
    assert_eq!(img.get_pixel(2, 2), [10, 11, 12]);
}

// ─── Color Conversion ──────────────────────────────────────────────────────

#[test]
fn test_bgr_to_rgb_conversion() {
    let mut img = Image::new(2, 1);
    img.set_pixel(0, 0, [100, 150, 200]); // BGR
    img.set_pixel(1, 0, [0, 128, 255]); // BGR
    let rgb = img.bgr_to_rgb();
    assert_eq!(rgb.get_pixel(0, 0), [200, 150, 100]); // RGB
    assert_eq!(rgb.get_pixel(1, 0), [255, 128, 0]); // RGB
}

#[test]
fn test_rgb_to_bgr_is_inverse_of_bgr_to_rgb() {
    let mut img = Image::new(2, 2);
    img.set_pixel(0, 0, [10, 20, 30]);
    img.set_pixel(1, 1, [255, 128, 64]);
    let rgb = img.bgr_to_rgb();
    let bgr = rgb.rgb_to_bgr();
    assert_eq!(bgr.get_pixel(0, 0), img.get_pixel(0, 0));
    assert_eq!(bgr.get_pixel(1, 1), img.get_pixel(1, 1));
}

#[test]
fn test_round_trip_bgr_rgb_bgr() {
    let mut img = Image::new(3, 3);
    for y in 0..3 {
        for x in 0..3 {
            img.set_pixel(x, y, [(x * 30) as u8, (y * 40) as u8, 100]);
        }
    }
    let rgb = img.bgr_to_rgb();
    let recovered = rgb.rgb_to_bgr();
    assert_eq!(recovered.data, img.data);
}

// ─── Crop ───────────────────────────────────────────────────────────────────

#[test]
fn test_crop_basic() {
    let mut img = Image::new(10, 10);
    for y in 0..10 {
        for x in 0..10 {
            img.set_pixel(x, y, [x as u8, y as u8, 50]);
        }
    }
    let cropped = img.crop(2, 3, 4, 5);
    assert_eq!(cropped.width, 4);
    assert_eq!(cropped.height, 5);
    assert_eq!(cropped.get_pixel(0, 0), img.get_pixel(2, 3));
    assert_eq!(cropped.get_pixel(3, 4), img.get_pixel(5, 7));
}

#[test]
fn test_crop_full_image() {
    let mut img = Image::new(5, 5);
    for y in 0..5 {
        for x in 0..5 {
            img.set_pixel(x, y, [x as u8, y as u8, 0]);
        }
    }
    let cropped = img.crop(0, 0, 5, 5);
    assert_eq!(cropped.data, img.data);
}

#[test]
fn test_crop_clamped_to_bounds() {
    let img = Image::new(5, 5);
    let cropped = img.crop(3, 3, 10, 10);
    assert_eq!(cropped.width, 2);
    assert_eq!(cropped.height, 2);
}

#[test]
fn test_crop_zero_size() {
    let img = Image::new(5, 5);
    let cropped = img.crop(2, 2, 0, 0);
    assert_eq!(cropped.width, 0);
    assert_eq!(cropped.height, 0);
    assert_eq!(cropped.data.len(), 0);
}

// ─── Resize ─────────────────────────────────────────────────────────────────

#[test]
fn test_resize_downscale() {
    let mut img = Image::new(4, 4);
    for y in 0..4 {
        for x in 0..4 {
            img.set_pixel(x, y, [100, 100, 100]);
        }
    }
    let resized = img.resize(2, 2);
    assert_eq!(resized.width, 2);
    assert_eq!(resized.height, 2);
    // All pixels should be 100 since source is uniform
    for y in 0..2 {
        for x in 0..2 {
            assert_eq!(resized.get_pixel(x, y), [100, 100, 100]);
        }
    }
}

#[test]
fn test_resize_preserves_color_gradient() {
    let mut img = Image::new(10, 1);
    for x in 0..10 {
        img.set_pixel(x, 0, [x as u8 * 25, 0, 0]);
    }
    let resized = img.resize(5, 1);
    // Check that values increase monotonically (gradient preserved)
    for x in 0..4 {
        let curr = resized.get_pixel(x, 0)[0];
        let next = resized.get_pixel(x + 1, 0)[0];
        assert!(next >= curr, "Gradient should be preserved");
    }
}

#[test]
fn test_resize_same_size() {
    let mut img = Image::new(3, 3);
    for y in 0..3 {
        for x in 0..3 {
            img.set_pixel(x, y, [x as u8 * 50, y as u8 * 50, 100]);
        }
    }
    let resized = img.resize(3, 3);
    // Values should be very close (bilinear interpolation may introduce small diffs)
    for y in 0..3 {
        for x in 0..3 {
            let orig = img.get_pixel(x, y);
            let res = resized.get_pixel(x, y);
            for c in 0..3 {
                let diff = (res[c] as i32 - orig[c] as i32).abs();
                assert!(
                    diff <= 1,
                    "Pixel ({},{}) channel {}: orig={}, resized={}, diff={}",
                    x,
                    y,
                    c,
                    orig[c],
                    res[c],
                    diff
                );
            }
        }
    }
}

#[test]
fn test_resize_upscale() {
    let mut img = Image::new(2, 2);
    img.set_pixel(0, 0, [0, 0, 0]);
    img.set_pixel(1, 0, [255, 0, 0]);
    img.set_pixel(0, 1, [0, 255, 0]);
    img.set_pixel(1, 1, [0, 0, 255]);
    let resized = img.resize(4, 4);
    assert_eq!(resized.width, 4);
    assert_eq!(resized.height, 4);
    // Corner values should approximately match
    // Note: bilinear interpolation at corners should be close to source
    let tl = resized.get_pixel(0, 0);
    let br = resized.get_pixel(3, 3);
    assert!(tl[0] < 128, "Top-left should be mostly red=0");
    assert!(br[2] > 128, "Bottom-right should be mostly blue=255");
}
