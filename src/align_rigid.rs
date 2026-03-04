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
        let mut valid_pixel_count = 0;

        // First pass: collect statistics for the transformed source image
        for dest_y in 0..self.height {
            for dest_x in 0..self.width {
                // Calculate the corresponding position in the source image
                let dx = dest_x as f64 - self.center_x;
                let dy = dest_y as f64 - self.center_y;

                // Apply inverse transformation (rotation + translation)
                let source_x = self.center_x + cos_theta * dx + sin_theta * dy - shift_x;
                let source_y = self.center_y - sin_theta * dx + cos_theta * dy - shift_y;

                // Check if the pixel is within bounds
                if source_x >= 0.0
                    && source_x < self.width as f64
                    && source_y >= 0.0
                    && source_y < self.height as f64
                {
                    // Get integer coordinates for accessing pixels
                    let src_x = source_x as usize;
                    let src_y = source_y as usize;

                    // Get the source pixel value
                    let source_val =
                        self.source_smoothed.get_pixel(src_x as u32, src_y as u32)[0] as f64;

                    // Update the accumulators
                    source_sum += source_val;
                    source_sq_sum += source_val * source_val;
                    valid_pixel_count += 1;
                }
            }
        }

        if valid_pixel_count == 0 {
            return Ok(vec![0.0, 0.0, 0.0]);
        }

        // Calculate the source image statistics
        let source_mean = source_sum / valid_pixel_count as f64;
        let source_variance = source_sq_sum / valid_pixel_count as f64 - source_mean * source_mean;

        if source_variance < 1e-10 {
            return Ok(vec![0.0, 0.0, 0.0]);
        }

        // Normalizer for correlation coefficient
        let normalizer =
            1.0 / (valid_pixel_count as f64 * source_variance.sqrt() * self.dest_variance.sqrt());

        // Initialize gradient accumulators
        let mut grad_theta = 0.0;
        let mut grad_shift_x = 0.0;
        let mut grad_shift_y = 0.0;

        // Second pass: calculate gradients in a single pass without storing points
        for dest_y in 0..self.height {
            for dest_x in 0..self.width {
                // Calculate the corresponding position in the source image
                let dx = dest_x as f64 - self.center_x;
                let dy = dest_y as f64 - self.center_y;

                // Apply inverse transformation (rotation + translation)
                let source_x = self.center_x + cos_theta * dx + sin_theta * dy - shift_x;
                let source_y = self.center_y - sin_theta * dx + cos_theta * dy - shift_y;

                // Check if the pixel is within bounds
                if source_x >= 0.0
                    && source_x < self.width as f64
                    && source_y >= 0.0
                    && source_y < self.height as f64
                {
                    // Get integer coordinates for accessing pixels
                    let src_x = source_x as usize;
                    let src_y = source_y as usize;

                    // Get destination value and center it
                    let dest_val = self.dest_smoothed.get_pixel(dest_x as u32, dest_y as u32)[0] as f64;
                    let dest_centered = dest_val - self.dest_mean;

                    // Calculate the correlation term for this pixel
                    let corr_term = dest_centered * normalizer;

                    // Get the image gradient at this position
                    let dx_val = self.source_dv_dx.get_pixel(src_x as u32, src_y as u32)[0] as f64;
                    let dy_val = self.source_dv_dy.get_pixel(src_x as u32, src_y as u32)[0] as f64;

                    // Partial derivatives of source_x and source_y with respect to parameters
                    // For theta:
                    // d(source_x)/d(theta) = -sin_theta * dx + cos_theta * dy
                    // d(source_y)/d(theta) = -cos_theta * dx - sin_theta * dy
                    let d_source_x_d_theta = -sin_theta * dx + cos_theta * dy;
                    let d_source_y_d_theta = -cos_theta * dx - sin_theta * dy;

                    // Calculate the change in source pixel value due to rotation
                    let d_source_val_d_theta = dx_val * d_source_x_d_theta + dy_val * d_source_y_d_theta;

                    // Update gradient components for each parameter
                    grad_theta += corr_term * d_source_val_d_theta;
                    grad_shift_x -= corr_term * dx_val; // d(source_x)/d(shift_x) = -1
                    grad_shift_y -= corr_term * dy_val; // d(source_y)/d(shift_y) = -1
                }
            }
        }

        // Return the gradient of the cost function (2 - correlation)
        // Since d(2 - correlation)/d(param) = -d(correlation)/d(param)
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
        let mut product_sum = 0.0;
        let mut valid_pixel_count = 0;
        let mut out_of_bounds_count = 0;

        // For each pixel in the destination image
        for dest_y in 0..self.height {
            for dest_x in 0..self.width {
                // Calculate the corresponding position in the source image
                let dx = dest_x as f64 - self.center_x;
                let dy = dest_y as f64 - self.center_y;

                // Apply inverse transformation (rotation + translation)
                let source_x = self.center_x + cos_theta * dx + sin_theta * dy - shift_x;
                let source_y = self.center_y - sin_theta * dx + cos_theta * dy - shift_y;

                // Get the destination pixel value and center it with precalculated mean
                let dest_val = self.dest_smoothed.get_pixel(dest_x as u32, dest_y as u32)[0] as f64;
                let dest_centered = dest_val - self.dest_mean;

                // Check if the pixel is within bounds
                if source_x >= 0.0
                    && source_x < self.width as f64
                    && source_y >= 0.0
                    && source_y < self.height as f64
                {
                    // Get integer coordinates for accessing pixels
                    let src_x = source_x as usize;
                    let src_y = source_y as usize;

                    // Get the source pixel value
                    let source_val =
                        self.source_smoothed.get_pixel(src_x as u32, src_y as u32)[0] as f64;

                    // Update the correlation accumulators
                    source_sum += source_val;
                    source_sq_sum += source_val * source_val;
                    product_sum += source_val * dest_val;
                    valid_pixel_count += 1;
                } else {
                    // Point is out of bounds, count it separately
                    out_of_bounds_count += 1;
                }
            }
        }

        // If no valid pixels, return a cost of 2.0 very bad
        if valid_pixel_count == 0 {
            return Ok(2.0);
        }

        // Calculate the source image statistics based on the transformed positions
        let source_mean = source_sum / valid_pixel_count as f64;
        let source_variance = source_sq_sum / valid_pixel_count as f64 - source_mean * source_mean;

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

                if source_x >= 0.0
                    && source_x < self.width as f64
                    && source_y >= 0.0
                    && source_y < self.height as f64
                {
                    let src_x = source_x as usize;
                    let src_y = source_y as usize;

                    let source_val =
                        self.source_smoothed.get_pixel(src_x as u32, src_y as u32)[0] as f64;
                    let dest_val =
                        self.dest_smoothed.get_pixel(dest_x as u32, dest_y as u32)[0] as f64;

                    centered_product_sum +=
                        (source_val - source_mean) * (dest_val - self.dest_mean);
                }
            }
        }

        let correlation = centered_product_sum
            / (valid_pixel_count as f64 * source_variance.sqrt() * self.dest_variance.sqrt());

        // Calculate the total pixels
        let total_pixels = self.width * self.height;

        // Calculate the weighted cost:
        // - For valid pixels: 2 - correlation (ranges from 1 to 3, where 1 is perfect match)
        // - For out-of-bounds pixels: loss of 2 (as specified)
        let valid_pixel_ratio = valid_pixel_count as f64 / total_pixels as f64;
        let out_of_bounds_ratio = out_of_bounds_count as f64 / total_pixels as f64;

        // Combine costs: (valid pixel contribution) + (out-of-bounds contribution)
        let cost = valid_pixel_ratio * (2.0 - correlation) + out_of_bounds_ratio * 2.0;

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
