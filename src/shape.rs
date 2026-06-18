use crate::{Result, TensorError};

pub(crate) fn checked_numel(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            TensorError::Shape(format!("shape {:?} overflows usize element count", shape))
        })
    })
}

pub(crate) fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut running = 1usize;
    for index in (0..shape.len()).rev() {
        strides[index] = running;
        running = running.saturating_mul(shape[index]);
    }
    strides
}

pub(crate) fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    strides == contiguous_strides(shape)
}

pub(crate) fn has_internal_overlap(shape: &[usize], strides: &[usize]) -> bool {
    shape
        .iter()
        .zip(strides.iter())
        .any(|(dim, stride)| *dim > 1 && *stride == 0)
}

pub(crate) fn logical_offset(index: &[usize], strides: &[usize], offset: usize) -> usize {
    offset
        + index
            .iter()
            .zip(strides.iter())
            .map(|(idx, stride)| idx * stride)
            .sum::<usize>()
}

pub(crate) fn flatten_index(index: &[usize], shape: &[usize]) -> usize {
    logical_offset(index, &contiguous_strides(shape), 0)
}

pub(crate) fn for_each_index<F>(shape: &[usize], mut f: F)
where
    F: FnMut(&[usize]),
{
    if shape.contains(&0) {
        return;
    }
    if shape.is_empty() {
        f(&[]);
        return;
    }

    let mut index = vec![0; shape.len()];
    'outer: loop {
        f(&index);
        for dim in (0..shape.len()).rev() {
            index[dim] += 1;
            if index[dim] < shape[dim] {
                continue 'outer;
            }
            index[dim] = 0;
        }
        break;
    }
}

pub(crate) fn broadcast_shapes(left: &[usize], right: &[usize]) -> Result<Vec<usize>> {
    let rank = left.len().max(right.len());
    let mut out = vec![1; rank];
    for (out_dim, out_dim_size) in out.iter_mut().enumerate() {
        let left_dim = dim_from_right(left, rank, out_dim);
        let right_dim = dim_from_right(right, rank, out_dim);
        *out_dim_size = match (left_dim, right_dim) {
            (a, b) if a == b => a,
            (1, b) => b,
            (a, 1) => a,
            (a, b) => {
                return Err(TensorError::Shape(format!(
                    "cannot broadcast shapes {:?} and {:?}: dimension {} has {} vs {}",
                    left, right, out_dim, a, b
                )))
            }
        };
    }
    checked_numel(&out)?;
    Ok(out)
}

pub(crate) fn broadcast_flat_index(
    output_index: &[usize],
    output_shape: &[usize],
    input_shape: &[usize],
) -> usize {
    if input_shape.is_empty() {
        return 0;
    }

    let leading = output_shape.len() - input_shape.len();
    let mut input_index = vec![0; input_shape.len()];
    for input_dim in 0..input_shape.len() {
        input_index[input_dim] = if input_shape[input_dim] == 1 {
            0
        } else {
            output_index[input_dim + leading]
        };
    }
    flatten_index(&input_index, input_shape)
}

pub(crate) fn expand_strides(
    input_shape: &[usize],
    input_strides: &[usize],
    output_shape: &[usize],
) -> Result<Vec<usize>> {
    if output_shape.len() < input_shape.len() {
        return Err(TensorError::Shape(format!(
            "cannot expand shape {:?} to lower-rank shape {:?}",
            input_shape, output_shape
        )));
    }

    let mut output_strides = vec![0; output_shape.len()];
    let leading = output_shape.len() - input_shape.len();
    for output_dim in 0..output_shape.len() {
        if output_dim < leading {
            output_strides[output_dim] = 0;
            continue;
        }

        let input_dim = output_dim - leading;
        let input_size = input_shape[input_dim];
        let output_size = output_shape[output_dim];
        output_strides[output_dim] = if input_size == output_size {
            input_strides[input_dim]
        } else if input_size == 1 {
            0
        } else {
            return Err(TensorError::Shape(format!(
                "cannot expand shape {:?} to {:?}: dim {} has {} vs {}",
                input_shape, output_shape, input_dim, input_size, output_size
            )));
        };
    }

    checked_numel(output_shape)?;
    Ok(output_strides)
}

pub(crate) fn normalize_dim(rank: usize, dim: isize) -> Result<usize> {
    let normalized = if dim < 0 { rank as isize + dim } else { dim };
    if normalized < 0 || normalized >= rank as isize {
        return Err(TensorError::Shape(format!(
            "dim {dim} out of range for tensor rank {rank}"
        )));
    }
    Ok(normalized as usize)
}

pub(crate) fn validate_permutation(rank: usize, dims: &[usize]) -> Result<()> {
    if dims.len() != rank {
        return Err(TensorError::Shape(format!(
            "permutation rank mismatch: tensor rank {rank}, dims {:?}",
            dims
        )));
    }

    let mut seen = vec![false; rank];
    for &dim in dims {
        if dim >= rank {
            return Err(TensorError::Shape(format!(
                "permutation dim {dim} out of range for rank {rank}"
            )));
        }
        if seen[dim] {
            return Err(TensorError::Shape(format!(
                "permutation contains duplicate dim {dim}: {:?}",
                dims
            )));
        }
        seen[dim] = true;
    }
    Ok(())
}

pub(crate) fn max_storage_offset(shape: &[usize], strides: &[usize], offset: usize) -> usize {
    offset
        + shape
            .iter()
            .zip(strides.iter())
            .map(|(dim, stride)| dim.saturating_sub(1).saturating_mul(*stride))
            .sum::<usize>()
}

fn dim_from_right(shape: &[usize], output_rank: usize, output_dim: usize) -> usize {
    let leading = output_rank - shape.len();
    if output_dim < leading {
        1
    } else {
        shape[output_dim - leading]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_has_one_element() {
        assert_eq!(numel(&[]), 1);
        assert_eq!(contiguous_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn broadcasting_rejects_incompatible_shapes() {
        let err = broadcast_shapes(&[2, 3], &[2, 2]).unwrap_err();
        assert!(matches!(err, TensorError::Shape(_)));
    }

    #[test]
    fn expand_strides_insert_zero_strides_for_broadcasted_axes() {
        assert_eq!(
            expand_strides(&[3, 1], &[1, 1], &[2, 3, 4]).unwrap(),
            vec![0, 1, 0]
        );
    }

    #[test]
    fn negative_dims_normalize_against_rank() {
        assert_eq!(normalize_dim(3, -1).unwrap(), 2);
        assert!(normalize_dim(3, -4).is_err());
    }
}
