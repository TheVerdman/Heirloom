#![deny(unsafe_op_in_unsafe_fn)]
#![warn(rustdoc::broken_intra_doc_links)]

//! Audited kernel boundaries for Heirloom.
//!
//! Public CPU wrappers validate all dimension products, input lengths,
//! allocations, and pointer strides before entering `matrixmultiply`. The
//! [`cuda`] module owns CUDA Driver/NCCL handles and validates logical shapes
//! before loading PTX or launching device code.

use rayon::prelude::*;
use std::fmt;

pub mod cuda;

/// An input-validation or allocation failure at the safe kernel boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelError {
    message: String,
}

impl KernelError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for KernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for KernelError {}

/// Result type returned by safe CPU-kernel wrappers.
pub type KernelResult<T> = std::result::Result<T, KernelError>;

/// Multiplies two row-major `f64` matrices through a checked safe boundary.
///
/// Before the unsafe matrix kernel is entered, this function proves that all
/// matrix element counts fit in `usize`, input slices have the exact required
/// lengths, output allocation succeeds, and row strides fit in `isize`.
pub fn matmul_f64(
    left: &[f64],
    right: &[f64],
    m: usize,
    k: usize,
    n: usize,
) -> KernelResult<Vec<f64>> {
    let left_len = checked_matrix_len(m, k, "left")?;
    let right_len = checked_matrix_len(k, n, "right")?;
    let output_len = checked_matrix_len(m, n, "output")?;
    ensure_matrix_len(left, left_len, "left")?;
    ensure_matrix_len(right, right_len, "right")?;

    let mut output = Vec::new();
    output.try_reserve_exact(output_len).map_err(|error| {
        KernelError::new(format!(
            "matrix output allocation for {output_len} f64 elements failed: {error}"
        ))
    })?;
    output.resize(output_len, 0.0);
    if m == 0 || k == 0 || n == 0 {
        return Ok(output);
    }

    let left_row_stride = checked_stride(k, "left row")?;
    let right_row_stride = checked_stride(n, "right row")?;
    let output_row_stride = checked_stride(n, "output row")?;

    // SAFETY: checked products above prove the exact m*k, k*n, and m*n slice
    // lengths without wraparound. Non-zero dimensions ensure each pointer has
    // addressable elements. Row strides fit in isize, matrices are row-major
    // contiguous, and output is uniquely borrowed for the entire call.
    unsafe {
        matrixmultiply::dgemm(
            m,
            k,
            n,
            1.0,
            left.as_ptr(),
            left_row_stride,
            1,
            right.as_ptr(),
            right_row_stride,
            1,
            0.0,
            output.as_mut_ptr(),
            output_row_stride,
            1,
        );
    }
    Ok(output)
}

/// Applies [`matmul_f64`] to equally sized batches.
pub fn batched_matmul_f64(
    left_batches: &[Vec<f64>],
    right_batches: &[Vec<f64>],
    m: usize,
    k: usize,
    n: usize,
) -> KernelResult<Vec<Vec<f64>>> {
    if left_batches.len() != right_batches.len() {
        return Err(KernelError::new(format!(
            "matrix batch mismatch: left has {} batches, right has {}",
            left_batches.len(),
            right_batches.len()
        )));
    }
    left_batches
        .par_iter()
        .zip(right_batches.par_iter())
        .map(|(left, right)| matmul_f64(left, right, m, k, n))
        .collect()
}

fn checked_matrix_len(rows: usize, cols: usize, role: &str) -> KernelResult<usize> {
    rows.checked_mul(cols).ok_or_else(|| {
        KernelError::new(format!(
            "{role} matrix element count overflow: {rows} * {cols}"
        ))
    })
}

fn ensure_matrix_len(data: &[f64], expected: usize, role: &str) -> KernelResult<()> {
    if data.len() != expected {
        return Err(KernelError::new(format!(
            "{role} matrix length mismatch: expected {expected}, got {}",
            data.len()
        )));
    }
    Ok(())
}

fn checked_stride(value: usize, role: &str) -> KernelResult<isize> {
    isize::try_from(value).map_err(|_| {
        KernelError::new(format!(
            "{role} stride {value} does not fit in the kernel isize stride type"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_f64_matches_small_fixture() {
        let left = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let right = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        assert_eq!(
            matmul_f64(&left, &right, 2, 3, 2).unwrap(),
            vec![58.0, 64.0, 139.0, 154.0]
        );
    }

    #[test]
    fn matmul_f64_rejects_invalid_lengths_and_batch_counts() {
        let error = matmul_f64(&[1.0], &[1.0], 2, 1, 1).unwrap_err();
        assert!(error.to_string().contains("left matrix length mismatch"));

        let error = batched_matmul_f64(&[vec![1.0]], &[], 1, 1, 1).unwrap_err();
        assert!(error.to_string().contains("matrix batch mismatch"));
    }

    #[test]
    fn matmul_f64_zero_dimensions_have_explicit_shapes() {
        assert_eq!(matmul_f64(&[], &[1.0; 6], 0, 3, 2).unwrap(), vec![]);
        assert_eq!(matmul_f64(&[], &[], 2, 0, 3).unwrap(), vec![0.0; 6]);
        assert_eq!(matmul_f64(&[1.0; 6], &[], 2, 3, 0).unwrap(), vec![]);
    }

    #[test]
    fn matmul_f64_rejects_each_overflowing_shape_product() {
        assert!(matmul_f64(&[], &[], usize::MAX, 2, 0)
            .unwrap_err()
            .to_string()
            .contains("left matrix element count overflow"));
        assert!(matmul_f64(&[], &[], 0, usize::MAX, 2)
            .unwrap_err()
            .to_string()
            .contains("right matrix element count overflow"));
        assert!(matmul_f64(&[], &[], usize::MAX, 0, 2)
            .unwrap_err()
            .to_string()
            .contains("output matrix element count overflow"));
    }

    #[test]
    fn matrix_stride_conversion_rejects_values_above_isize() {
        assert!(checked_stride(usize::MAX, "fixture")
            .unwrap_err()
            .to_string()
            .contains("does not fit"));
    }
}
