use dust_ukf_rs::{get_sigmas_ukf, DIM};
use nalgebra::{SMatrix, SVector};

fn main() {
    // Create a test mean
    let mean = SVector::<f64, DIM>::from_element(0.0);

    // Create a near-singular covariance matrix (very small eigenvalues)
    let mut covar = SMatrix::<f64, DIM, DIM>::identity();
    for (diagonal, value) in covar
        .as_mut_slice()
        .iter_mut()
        .step_by(DIM + 1)
        .zip([1e-15, 1e-14, 1.0, 1.0, 1.0, 1.0])
    {
        *diagonal = value;
    }

    // This should succeed with PSD repair fallback
    match get_sigmas_ukf(&mean, &covar) {
        Some(sigmas) => {
            println!("✓ PSD repair successful!");
            println!("Generated {} sigma points", sigmas.nrows());
            println!("First sigma point: {:?}", sigmas.row(0));
        }
        None => {
            println!("✗ Failed to generate sigma points (even with repair)");
        }
    }

    // Create a well-conditioned covariance matrix
    let good_covar = SMatrix::<f64, DIM, DIM>::identity();

    match get_sigmas_ukf(&mean, &good_covar) {
        Some(sigmas) => {
            println!("\n✓ Normal case (no repair needed)");
            println!("Generated {} sigma points", sigmas.nrows());
        }
        None => {
            println!("\n✗ Failed on well-conditioned matrix (unexpected)");
        }
    }
}
