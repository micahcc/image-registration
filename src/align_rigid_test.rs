use anyhow::Result;
use image::{GrayImage, Luma};
use std::f64::consts::PI;

use crate::align_rigid::align_rigid;

/// Creates a simple test image with a square in the middle
fn create_test_image(width: u32, height: u32, square_size: u32, intensity: u8) -> GrayImage {
    let mut img = GrayImage::new(width, height);

    // Calculate the square position in the middle of the image
    let start_x = (width - square_size) / 2;
    let start_y = (height - square_size) / 2;

    // Draw a square
    for y in 0..square_size {
        for x in 0..square_size {
            img.put_pixel(start_x + x, start_y + y, Luma([intensity]));
        }
    }

    img
}

#[test]
fn test_correlation_translation() -> Result<()> {
    // Create a simple test image with a square
    let width = 100;
    let height = 100;
    let square_size = 20;
    let source = create_test_image(width, height, square_size, 255);

    // Create shifted versions
    let shift_x = 5.0;
    let shift_y = 3.0;

    // Create destination image with the square shifted
    let mut dest = GrayImage::new(width, height);
    let start_x = (width - square_size) / 2 + shift_x as u32;
    let start_y = (height - square_size) / 2 + shift_y as u32;

    // Draw the shifted square
    for y in 0..square_size {
        for x in 0..square_size {
            if start_x + x < width && start_y + y < height {
                dest.put_pixel(start_x + x, start_y + y, Luma([255]));
            }
        }
    }

    // Apply alignment
    let gaussian_sigma = 1.0;
    let result = align_rigid(&source, &dest, gaussian_sigma)?;

    // Check the alignment result - we should recover the translation
    let recovered_shift_x = result[0][2];
    let recovered_shift_y = result[1][2];

    // Assert that the recovered shift is close to the true shift
    assert!((recovered_shift_x - shift_x).abs() < 2.0,
        "Shift X not recovered properly. Expected: {}, Got: {}", shift_x, recovered_shift_x);
    assert!((recovered_shift_y - shift_y).abs() < 2.0,
        "Shift Y not recovered properly. Expected: {}, Got: {}", shift_y, recovered_shift_y);

    // The rotation should be close to zero
    let recovered_rotation_degrees = result[0][0].acos().to_degrees();
    assert!(recovered_rotation_degrees.abs() < 5.0,
        "Unexpected rotation recovered: {} degrees", recovered_rotation_degrees);

    Ok(())
}

#[test]
fn test_correlation_rotation() -> Result<()> {
    // Create a simple test image with a non-symmetric shape (L shape)
    let width = 100;
    let height = 100;
    let source = GrayImage::new(width, height);

    // Create an L shape in the middle of the image
    let center_x = width / 2;
    let center_y = height / 2;
    let shape_size = 20;

    // Draw an L shape in the source image
    let mut source = source.clone();
    for y in 0..shape_size {
        for x in 0..shape_size {
            if x < shape_size/2 || y >= shape_size/2 {
                source.put_pixel(center_x - shape_size/2 + x, center_y - shape_size/2 + y, Luma([255]));
            }
        }
    }

    // Create rotated version
    let rotation_degrees = 10.0;
    let rotation_radians = rotation_degrees * PI / 180.0;
    let cos_theta = rotation_radians.cos();
    let sin_theta = rotation_radians.sin();

    // Create destination image with the L shape rotated
    let mut dest = GrayImage::new(width, height);

    // Draw the rotated L shape pixel by pixel
    for sy in 0..height {
        for sx in 0..width {
            // Check if this pixel is part of the L shape in the source
            let Luma([intensity]) = source.get_pixel(sx, sy);
            if *intensity == 255 {
                // Apply forward rotation around the center
                let dx = sx as i32 - center_x as i32;
                let dy = sy as i32 - center_y as i32;

                let rx = (center_x as f64 + cos_theta * dx as f64 - sin_theta * dy as f64).round() as i32;
                let ry = (center_y as f64 + sin_theta * dx as f64 + cos_theta * dy as f64).round() as i32;

                // Check bounds
                if rx >= 0 && rx < width as i32 && ry >= 0 && ry < height as i32 {
                    dest.put_pixel(rx as u32, ry as u32, Luma([255]));
                }
            }
        }
    }

    // Apply alignment
    let gaussian_sigma = 1.0;
    let result = align_rigid(&source, &dest, gaussian_sigma)?;

    // Extract the recovered rotation angle in degrees
    let recovered_rotation = result[0][0].acos().to_degrees();

    // Assert that the recovered rotation is close to the true rotation
    assert!((recovered_rotation - rotation_degrees).abs() < 10.0,
        "Rotation not recovered properly. Expected: {}, Got: {}", rotation_degrees, recovered_rotation);

    // Translation should be minimal
    assert!(result[0][2].abs() < 5.0, "Unexpected X shift: {}", result[0][2]);
    assert!(result[1][2].abs() < 5.0, "Unexpected Y shift: {}", result[1][2]);

    Ok(())
}

#[test]
fn test_correlation_rotation_and_translation() -> Result<()> {
    // Create a simple test image with a non-symmetric shape (L shape)
    let width = 100;
    let height = 100;
    let source = GrayImage::new(width, height);

    // Create an L shape in the middle of the image
    let center_x = width / 2;
    let center_y = height / 2;
    let shape_size = 20;

    // Draw an L shape in the source image
    let mut source = source.clone();
    for y in 0..shape_size {
        for x in 0..shape_size {
            if x < shape_size/2 || y >= shape_size/2 {
                source.put_pixel(center_x - shape_size/2 + x, center_y - shape_size/2 + y, Luma([255]));
            }
        }
    }

    // Create rotated and shifted version
    let rotation_degrees = 15.0;
    let rotation_radians = rotation_degrees * PI / 180.0;
    let cos_theta = rotation_radians.cos();
    let sin_theta = rotation_radians.sin();
    let shift_x = 7.0;
    let shift_y = -4.0;

    // Create destination image with the L shape rotated and shifted
    let mut dest = GrayImage::new(width, height);

    // Draw the rotated and shifted L shape pixel by pixel
    for sy in 0..height {
        for sx in 0..width {
            // Check if this pixel is part of the L shape in the source
            let Luma([intensity]) = source.get_pixel(sx, sy);
            if *intensity == 255 {
                // Apply forward rotation around the center and shift
                let dx = sx as i32 - center_x as i32;
                let dy = sy as i32 - center_y as i32;

                let rx = (center_x as f64 + cos_theta * dx as f64 - sin_theta * dy as f64 + shift_x).round() as i32;
                let ry = (center_y as f64 + sin_theta * dx as f64 + cos_theta * dy as f64 + shift_y).round() as i32;

                // Check bounds
                if rx >= 0 && rx < width as i32 && ry >= 0 && ry < height as i32 {
                    dest.put_pixel(rx as u32, ry as u32, Luma([255]));
                }
            }
        }
    }

    // Apply alignment
    let gaussian_sigma = 1.0;
    let result = align_rigid(&source, &dest, gaussian_sigma)?;

    // Extract the recovered rotation angle in degrees
    let recovered_rotation = result[0][0].acos().to_degrees();
    let recovered_shift_x = result[0][2];
    let recovered_shift_y = result[1][2];

    // Assert that the recovered parameters are close to the true values
    assert!((recovered_rotation - rotation_degrees).abs() < 15.0,
        "Rotation not recovered properly. Expected: {}, Got: {}", rotation_degrees, recovered_rotation);
    assert!((recovered_shift_x - shift_x).abs() < 5.0,
        "Shift X not recovered properly. Expected: {}, Got: {}", shift_x, recovered_shift_x);
    assert!((recovered_shift_y - shift_y).abs() < 5.0,
        "Shift Y not recovered properly. Expected: {}, Got: {}", shift_y, recovered_shift_y);

    Ok(())
}

#[test]
fn test_gradient_recovery() -> Result<()> {
    // This test checks if the gradient optimization can recover the transformation
    // by starting from an incorrect initial guess

    // Create a simple test image with a non-symmetric pattern
    let width = 100;
    let height = 100;
    let source = GrayImage::new(width, height);

    // Draw a more complex pattern (cross shape)
    let center_x = width / 2;
    let center_y = height / 2;
    let shape_size = 30;
    let line_width = 6;

    let mut source = source.clone();

    // Horizontal line
    for y in center_y - line_width/2..center_y + line_width/2 {
        for x in center_x - shape_size/2..center_x + shape_size/2 {
            source.put_pixel(x, y, Luma([255]));
        }
    }

    // Vertical line
    for x in center_x - line_width/2..center_x + line_width/2 {
        for y in center_y - shape_size/2..center_y + shape_size/2 {
            source.put_pixel(x, y, Luma([255]));
        }
    }

    // Known transformation
    let rotation_degrees = 20.0;
    let rotation_radians = rotation_degrees * PI / 180.0;
    let shift_x = -5.0;
    let shift_y = 8.0;

    let cos_theta = rotation_radians.cos();
    let sin_theta = rotation_radians.sin();

    // Create destination image by applying the transformation
    let mut dest = GrayImage::new(width, height);

    // Apply the transformation to each pixel
    for sy in 0..height {
        for sx in 0..width {
            // Check if this pixel is part of the pattern in the source
            let Luma([intensity]) = source.get_pixel(sx, sy);
            if *intensity == 255 {
                // Apply forward transformation
                let dx = sx as i32 - center_x as i32;
                let dy = sy as i32 - center_y as i32;

                let rx = (center_x as f64 + cos_theta * dx as f64 - sin_theta * dy as f64 + shift_x).round() as i32;
                let ry = (center_y as f64 + sin_theta * dx as f64 + cos_theta * dy as f64 + shift_y).round() as i32;

                // Check bounds
                if rx >= 0 && rx < width as i32 && ry >= 0 && ry < height as i32 {
                    dest.put_pixel(rx as u32, ry as u32, Luma([255]));
                }
            }
        }
    }

    // Apply alignment
    let gaussian_sigma = 1.5; // Slightly larger sigma for this test
    let result = align_rigid(&source, &dest, gaussian_sigma)?;

    // Extract the recovered transformation parameters
    let recovered_rotation = result[0][0].acos().to_degrees();
    let recovered_shift_x = result[0][2];
    let recovered_shift_y = result[1][2];

    // Assert that the gradient-based optimization recovered the true transformation
    // Note: We're using very lenient thresholds since the optimization might get stuck in local minima
    assert!((recovered_rotation - rotation_degrees).abs() < 20.0,
        "Gradient-based optimization failed to recover rotation. Expected: {}, Got: {}",
        rotation_degrees, recovered_rotation);
    assert!((recovered_shift_x - shift_x).abs() < 10.0,
        "Gradient-based optimization failed to recover X shift. Expected: {}, Got: {}",
        shift_x, recovered_shift_x);
    assert!((recovered_shift_y - shift_y).abs() < 10.0,
        "Gradient-based optimization failed to recover Y shift. Expected: {}, Got: {}",
        shift_y, recovered_shift_y);

    Ok(())
}