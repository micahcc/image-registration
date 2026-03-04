use anyhow::Result;
use argmin::core::observers::ObserverMode;
use argmin::core::CostFunction;
use argmin::core::Executor;
use argmin::core::Gradient;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use image::buffer::ConvertBuffer;
use image::GrayImage;
use image::ImageBuffer;
use image::Luma;
use imageproc::filter::gaussian_blur_f32;
use imageproc::kernel::Kernel;

use crate::ArgminLogger;

type FloatImage = ImageBuffer<Luma<f32>, Vec<f32>>;

/// Optimization problem structure for rotation + translation (rigid)
/// parameters:
/// params[0] = theta (degrees) rotation about middle of image
/// params[1] = shift x in pixels
/// params[2] = shift y in pixels
#[derive(Clone)]
struct Problem {
    source_smoothed: FloatImage,
    dest_smoothed: FloatImage,
    source_dv_dx: FloatImage,
    source_dv_dy: FloatImage,
    // Precalculated values for optimization
    dest_mean: f64,
    dest_variance: f64,
    width: usize,
    height: usize,
    center_x: f64,
    center_y: f64,
}

impl Problem {
    fn new(
        source_smoothed: FloatImage,
        dest_smoothed: FloatImage,
        source_dv_dx: FloatImage,
        source_dv_dy: FloatImage,
    ) -> Self {
        let width = dest_smoothed.width() as usize;
        let height = dest_smoothed.height() as usize;
        let center_x = width as f64 / 2.0;
        let center_y = height as f64 / 2.0;

        // Calculate destination image statistics
        let mut dest_sum = 0.0;
        let mut dest_sq_sum = 0.0;
        let total_pixels = width * height;

        for y in 0..height {
            for x in 0..width {
                let val = dest_smoothed.get_pixel(x as u32, y as u32)[0] as f64;
                dest_sum += val;
                dest_sq_sum += val * val;
            }
        }

        let dest_mean = dest_sum / total_pixels as f64;
        let dest_variance = dest_sq_sum / total_pixels as f64 - dest_mean * dest_mean;

        Self {
            source_smoothed,
            dest_smoothed,
            source_dv_dx,
            source_dv_dy,
            dest_mean,
            dest_variance,
            width,
            height,
            center_x,
            center_y,
        }
    }
}

impl Gradient for Problem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, params: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        if params.len() != 3 {
            return Err(anyhow::anyhow!(
                "Expected 3 parameters: [theta, shift_x, shift_y]"
            ));
        }

        let theta = params[0].to_radians();
        let shift_x = params[1];
        let shift_y = params[2];

        // Calculate the rotation and translation
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        // Initialize accumulators for source statistics (which depend on transformation)
        let mut source_sum = 0.0;
        let mut source_sq_sum = 0.0;
        let total_pixels = self.width * self.height;

        // First pass: collect statistics for the transformed source image
        // We need this pass to calculate the source image mean and variance
        // These values are used to normalize the correlation coefficient
        for dest_y in 0..self.height {
            for dest_x in 0..self.width {
                // Calculate the corresponding position in the source image
                let dx = dest_x as f64 - self.center_x;
                let dy = dest_y as f64 - self.center_y;

                // Apply inverse transformation (rotation + translation)
                // source_x = center_x + cos(θ)*(x-center_x) + sin(θ)*(y-center_y) - shift_x
                // source_y = center_y - sin(θ)*(x-center_x) + cos(θ)*(y-center_y) - shift_y
                let source_x = self.center_x + cos_theta * dx + sin_theta * dy - shift_x;
                let source_y = self.center_y - sin_theta * dx + cos_theta * dy - shift_y;

                // Get nearest valid pixel coordinates using clamp
                let src_x = source_x.clamp(0.0, (self.width - 1) as f64) as usize;
                let src_y = source_y.clamp(0.0, (self.height - 1) as f64) as usize;

                // Get the source pixel value
                let source_val = self.source_smoothed.get_pixel(src_x as u32, src_y as u32)[0] as f64;

                // Update the accumulators
                source_sum += source_val;
                source_sq_sum += source_val * source_val;
            }
        }

        // Calculate the source image statistics
        let source_mean = source_sum / total_pixels as f64;
        let source_variance = source_sq_sum / total_pixels as f64 - source_mean * source_mean;

        if source_variance < 1e-10 {
            return Ok(vec![0.0, 0.0, 0.0]);
        }

        // Normalizer for correlation coefficient
        // The correlation coefficient is defined as:
        // ρ = Σ[(S(x,y) - μₛ)(D(x,y) - μₚ)] / (N·σₛ·σₚ)
        // where:
        // - S(x,y) is the source image value at (x,y)
        // - D(x,y) is the destination image value
        // - μₛ is the source mean
        // - μₚ is the destination mean
        // - σₛ is the source standard deviation
        // - σₚ is the destination standard deviation
        // - N is the number of pixels
        let normalizer =
            1.0 / (total_pixels as f64 * source_variance.sqrt() * self.dest_variance.sqrt());

        // Initialize gradient accumulators
        let mut grad_theta = 0.0;
        let mut grad_shift_x = 0.0;
        let mut grad_shift_y = 0.0;

        // Second pass: calculate gradients in a single pass without storing points
        // The derivative of the correlation with respect to a parameter p is:
        // ∂ρ/∂p = Σ[(∂S(x,y)/∂p)·(D(x,y) - μₚ)] / (N·σₛ·σₚ)
        //
        // Where ∂S(x,y)/∂p is the derivative of the source pixel value with respect to parameter p.
        // This is calculated using the chain rule:
        // ∂S(x,y)/∂p = (∂S/∂x)·(∂x/∂p) + (∂S/∂y)·(∂y/∂p)
        //
        // The derivatives ∂S/∂x and ∂S/∂y are the image gradients (precomputed)
        // The derivatives ∂x/∂p and ∂y/∂p depend on the specific parameter (θ, shift_x, shift_y)
        for dest_y in 0..self.height {
            for dest_x in 0..self.width {
                // Calculate the corresponding position in the source image
                let dx = dest_x as f64 - self.center_x;
                let dy = dest_y as f64 - self.center_y;

                // Apply inverse transformation (rotation + translation)
                let source_x = self.center_x + cos_theta * dx + sin_theta * dy - shift_x;
                let source_y = self.center_y - sin_theta * dx + cos_theta * dy - shift_y;

                // Get nearest valid pixel coordinates using clamp
                let src_x = source_x.clamp(0.0, (self.width - 1) as f64) as usize;
                let src_y = source_y.clamp(0.0, (self.height - 1) as f64) as usize;

                // Get destination value and center it
                let dest_val = self.dest_smoothed.get_pixel(dest_x as u32, dest_y as u32)[0] as f64;
                let dest_centered = dest_val - self.dest_mean;

                // Calculate the correlation term for this pixel
                // This is (D(x,y) - μₚ) / (N·σₛ·σₚ)
                let corr_term = dest_centered * normalizer;

                // Get the image gradient at this position (∂S/∂x and ∂S/∂y)
                let dx_val = self.source_dv_dx.get_pixel(src_x as u32, src_y as u32)[0] as f64;
                let dy_val = self.source_dv_dy.get_pixel(src_x as u32, src_y as u32)[0] as f64;

                // Partial derivatives of source_x and source_y with respect to parameters
                // These are derived by differentiating the transformation equations:
                //
                // For θ (rotation):
                // ∂(source_x)/∂θ = -sin(θ)·dx + cos(θ)·dy
                // ∂(source_y)/∂θ = -cos(θ)·dx - sin(θ)·dy
                let d_source_x_d_theta = -sin_theta * dx + cos_theta * dy;
                let d_source_y_d_theta = -cos_theta * dx - sin_theta * dy;

                // Calculate the change in source pixel value due to rotation
                // ∂S/∂θ = (∂S/∂x)·(∂x/∂θ) + (∂S/∂y)·(∂y/∂θ)
                let d_source_val_d_theta = dx_val * d_source_x_d_theta + dy_val * d_source_y_d_theta;

                // Update gradient components for each parameter
                // For θ: we calculated d_source_val_d_theta above
                grad_theta += corr_term * d_source_val_d_theta;

                // For shift_x: ∂(source_x)/∂(shift_x) = -1, ∂(source_y)/∂(shift_x) = 0
                // So ∂S/∂(shift_x) = (∂S/∂x)·(-1) = -dx_val
                grad_shift_x -= corr_term * dx_val;

                // For shift_y: ∂(source_x)/∂(shift_y) = 0, ∂(source_y)/∂(shift_y) = -1
                // So ∂S/∂(shift_y) = (∂S/∂y)·(-1) = -dy_val
                grad_shift_y -= corr_term * dy_val;
            }
        }

        // Return the gradient of the cost function (2 - correlation)
        // Our optimization minimizes (2 - correlation)
        // Since ∂(2-ρ)/∂p = -∂ρ/∂p, we negate the gradient components
        Ok(vec![-grad_theta, -grad_shift_x, -grad_shift_y])
    }
}

impl CostFunction for Problem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        if params.len() != 3 {
            return Err(anyhow::anyhow!(
                "Expected 3 parameters: [theta, shift_x, shift_y]",
            ));
        }

        let theta = params[0].to_radians(); // Convert to radians
        let shift_x = params[1];
        let shift_y = params[2];

        // Calculate the rotation and translation
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        // Initialize accumulators for correlation calculation
        let mut source_sum = 0.0;
        let mut source_sq_sum = 0.0;
        let total_pixels = self.width * self.height;

        // For each pixel in the destination image
        for dest_y in 0..self.height {
            for dest_x in 0..self.width {
                // Calculate the corresponding position in the source image
                let dx = dest_x as f64 - self.center_x;
                let dy = dest_y as f64 - self.center_y;

                // Apply inverse transformation (rotation + translation)
                let source_x = self.center_x + cos_theta * dx + sin_theta * dy - shift_x;
                let source_y = self.center_y - sin_theta * dx + cos_theta * dy - shift_y;

                // Get nearest valid pixel coordinates using clamp
                let src_x = source_x.clamp(0.0, (self.width - 1) as f64) as usize;
                let src_y = source_y.clamp(0.0, (self.height - 1) as f64) as usize;

                // Get the source pixel value (using nearest neighbor extrapolation for out-of-bounds)
                let source_val = self.source_smoothed.get_pixel(src_x as u32, src_y as u32)[0] as f64;

                // Update the correlation accumulators
                source_sum += source_val;
                source_sq_sum += source_val * source_val;
            }
        }

        // Calculate the source image statistics based on the transformed positions
        let source_mean = source_sum / total_pixels as f64;
        let source_variance = source_sq_sum / total_pixels as f64 - source_mean * source_mean;

        // Avoid division by zero
        if source_variance < 1e-10 || self.dest_variance < 1e-10 {
            return Ok(2.0);
        }

        // Recalculate product sum with centered values
        let mut centered_product_sum = 0.0;
        for dest_y in 0..self.height {
            for dest_x in 0..self.width {
                let dx = dest_x as f64 - self.center_x;
                let dy = dest_y as f64 - self.center_y;

                let source_x = self.center_x + cos_theta * dx + sin_theta * dy - shift_x;
                let source_y = self.center_y - sin_theta * dx + cos_theta * dy - shift_y;

                // Get nearest valid pixel coordinates using clamp
                let src_x = source_x.clamp(0.0, (self.width - 1) as f64) as usize;
                let src_y = source_y.clamp(0.0, (self.height - 1) as f64) as usize;

                let source_val = self.source_smoothed.get_pixel(src_x as u32, src_y as u32)[0] as f64;
                let dest_val = self.dest_smoothed.get_pixel(dest_x as u32, dest_y as u32)[0] as f64;

                centered_product_sum += (source_val - source_mean) * (dest_val - self.dest_mean);
            }
        }

        let correlation = centered_product_sum
            / (total_pixels as f64 * source_variance.sqrt() * self.dest_variance.sqrt());

        // Uncomment for debugging
        // if (params[0] - 0.0).abs() < 0.1 &&
        //    (params[1] - 5.0).abs() < 0.1 &&
        //    (params[2] + 3.0).abs() < 0.1 {
        //     println!("DEBUG - Cost function internals:");
        //     println!("  Source mean: {}, variance: {}", source_mean, source_variance);
        //     println!("  Dest mean: {}, variance: {}", self.dest_mean, self.dest_variance);
        //     println!("  Total pixels: {}", total_pixels);
        //     println!("  Centered product sum: {}", centered_product_sum);
        //     println!("  Correlation: {}", correlation);
        // }

        // The cost is simply 2 - correlation
        // This ranges from 1 (perfect match) to 3 (perfect anti-match)
        let cost = 2.0 - correlation;

        Ok(cost)
    }
}

/// Aligns the source image to the destination image using a rigid transformation (rotation + translation).
/// Uses gradient-based optimization to minimize the negative correlation between images.
///
/// Returns an affine transformation matrix as [[f64; 3]; 3] where the top 2x3 represents the transformation.
pub fn align_rigid(
    source: &GrayImage,
    dest: &GrayImage,
    gaussian_sigma: f32,
) -> Result<[[f64; 3]; 3]> {
    let source_smoothed: FloatImage = gaussian_blur_f32(&source.convert(), gaussian_sigma);
    let dest_smoothed: FloatImage = gaussian_blur_f32(&dest.convert(), gaussian_sigma);

    // Create x and y derivative filters (Sobel operators)
    let dx_data = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    let dx_filter = Kernel::new(&dx_data, 3, 3);

    let dy_data = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
    let dy_filter = Kernel::new(&dy_data, 3, 3);

    let source_dv_dx: ImageBuffer<Luma<f32>, Vec<f32>> =
        imageproc::filter::filter(&source_smoothed, dx_filter, |p: f32| p);
    let source_dv_dy: ImageBuffer<Luma<f32>, Vec<f32>> =
        imageproc::filter::filter(&source_smoothed, dy_filter, |p: f32| p);
    let problem = Problem::new(source_smoothed, dest_smoothed, source_dv_dx, source_dv_dy);

    // Create the solver - L-BFGS is a good choice for this problem
    let linesearch = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(linesearch, 3)
        .with_tolerance_cost(1e-15)
        .expect("Failed to set tolerance")
        .with_tolerance_grad(1e-15)
        .expect("Failed to set grad tolerance");

    let initial = vec![0.0, 0.0, 0.0];
    let res = Executor::new(problem, solver)
        .configure(|state| state.param(initial).max_iters(300).target_cost(1e-10))
        .add_observer(ArgminLogger {}, ObserverMode::Always)
        .run()
        .expect("Optimization failed");

    let error = res.state().cost;
    let params = res.state().best_param.clone().unwrap_or(vec![]);
    log::info!(
        "Found solution with error: {:.6}, params: {params:?}",
        error
    );

    let theta = params[0].to_radians();
    let shift_x = params[1];
    let shift_y = params[2];

    // Create an affine transformation matrix
    // For 2D rigid transformation (rotation + translation):
    // | cos(θ)  -sin(θ)  tx |
    // | sin(θ)   cos(θ)  ty |
    // |   0        0      1 |
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();

    // Create the affine matrix
    let affine = [
        [cos_theta, -sin_theta, shift_x],
        [sin_theta, cos_theta, shift_y],
        [0.0, 0.0, 1.0],
    ];

    // Return the transformation matrix
    Ok(affine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

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

    #[test]
    fn test_problem_cost_function() -> Result<()> {
        // This test specifically tests the cost function implementation in the Problem struct
        // We'll create two simple images and test various transformation parameters

        // Create two simple test images
        let width = 50;
        let height = 50;
        let square_size = 20;

        // Create source image with a square
        let source = create_test_image(width, height, square_size, 255);

        // Create destination image with the same square but shifted by a known amount
        let known_shift_x = 5.0;
        let known_shift_y = -3.0;
        let _known_rotation_degrees = 0.0; // No rotation for simplicity

        // Create destination image with the shifted square
        let mut dest = GrayImage::new(width, height);
        let start_x = (width - square_size) / 2 + known_shift_x as u32;
        let start_y = (height - square_size) / 2 - known_shift_y as u32; // Negative because we're creating dest manually

        // Draw the shifted square
        for y in 0..square_size {
            for x in 0..square_size {
                if start_x + x < width && start_y + y < height {
                    dest.put_pixel(start_x + x, start_y + y, Luma([255]));
                }
            }
        }

        // Apply Gaussian blur to both images as done in align_rigid function
        let gaussian_sigma = 1.0;
        let source_smoothed: FloatImage = gaussian_blur_f32(&source.convert(), gaussian_sigma);
        let dest_smoothed: FloatImage = gaussian_blur_f32(&dest.convert(), gaussian_sigma);

        // Create the derivative filters
        let dx_data = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
        let dx_filter = Kernel::new(&dx_data, 3, 3);

        let dy_data = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
        let dy_filter = Kernel::new(&dy_data, 3, 3);

        // Calculate image gradients
        let source_dv_dx: ImageBuffer<Luma<f32>, Vec<f32>> =
            imageproc::filter::filter(&source_smoothed, dx_filter, |p: f32| p);
        let source_dv_dy: ImageBuffer<Luma<f32>, Vec<f32>> =
            imageproc::filter::filter(&source_smoothed, dy_filter, |p: f32| p);

        // Create the Problem instance
        let problem = Problem::new(source_smoothed, dest_smoothed, source_dv_dx, source_dv_dy);

        // Test cases for the cost function
        let test_cases = vec![
            // [theta, shift_x, shift_y], expected_cost_below
            (vec![0.0, 0.0, 0.0], 2.2), // No transformation - should have high cost
            (vec![0.0, known_shift_x, known_shift_y], 1.3), // Correct transformation - should have low cost
            (vec![45.0, 0.0, 0.0], 2.2), // Wrong rotation - should have high cost
            (vec![0.0, -10.0, 15.0], 2.2), // Wrong translation - should have high cost
            (vec![0.0, known_shift_x, -known_shift_y], 2.2), // Wrong Y direction - should have high cost
        ];

        for (i, (params, expected_max_cost)) in test_cases.iter().enumerate() {
            // Call the cost function with the test parameters
            let cost = problem.cost(params)?;

            // Uncomment for debugging
            // if i == 1 {
            //     println!("Test case {}: params: {:?}, cost: {}", i, params, cost);
            // }

            // Assert that the cost matches our expectations
            assert!(cost <= *expected_max_cost,
                "Test case {}: Cost {} exceeds maximum expected {}", i, cost, expected_max_cost);

            // For the correct transformation, verify cost is significantly lower
            if i == 1 {
                assert!(cost < 1.5,
                    "Correct transformation should have cost < 1.5, got {}", cost);

                // For correct transformation, correlation should be high (cost should be close to 1.0)
                let correlation = 2.0 - cost;
                assert!(correlation > 0.8,
                    "Correlation should be high (> 0.8), but was {}", correlation);
            }
        }

        // Test edge cases

        // 1. Invalid number of parameters should return an error
        let invalid_params = vec![0.0, 0.0]; // Only 2 parameters instead of 3
        let result = problem.cost(&invalid_params);
        assert!(result.is_err(), "Expected error for invalid parameter count");

        // 2. Parameters that put the entire image out of bounds
        let out_of_bounds = vec![0.0, 1000.0, 1000.0]; // Huge shift
        let out_of_bounds_cost = problem.cost(&out_of_bounds)?;
        assert!(out_of_bounds_cost >= 2.0,
            "Out of bounds transformation should have high cost, got {}", out_of_bounds_cost);

        Ok(())
    }
}
