use rayon::prelude::*;

pub mod cuda;

pub fn matmul_f64(left: &[f64], right: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    assert_eq!(left.len(), m * k, "left matrix length mismatch");
    assert_eq!(right.len(), k * n, "right matrix length mismatch");
    let mut output = vec![0.0; m * n];
    if m == 0 || k == 0 || n == 0 {
        return output;
    }

    // SAFETY: all pointers come from valid slices with the checked lengths above.
    // Matrices are row-major contiguous. Strides are expressed in elements, and
    // the output slice is uniquely borrowed for the duration of the call.
    unsafe {
        matrixmultiply::dgemm(
            m,
            k,
            n,
            1.0,
            left.as_ptr(),
            k as isize,
            1,
            right.as_ptr(),
            n as isize,
            1,
            0.0,
            output.as_mut_ptr(),
            n as isize,
            1,
        );
    }
    output
}

pub fn batched_matmul_f64(
    left_batches: &[Vec<f64>],
    right_batches: &[Vec<f64>],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Vec<f64>> {
    assert_eq!(left_batches.len(), right_batches.len(), "batch mismatch");
    left_batches
        .par_iter()
        .zip(right_batches.par_iter())
        .map(|(left, right)| matmul_f64(left, right, m, k, n))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_f64_matches_small_fixture() {
        let left = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let right = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        assert_eq!(
            matmul_f64(&left, &right, 2, 3, 2),
            vec![58.0, 64.0, 139.0, 154.0]
        );
    }
}
