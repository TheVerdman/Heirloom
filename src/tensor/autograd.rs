//! Private autograd graph nodes and backward formulas.
//!
//! Each operation saves only the tensors or metadata required by its
//! derivative. Saved tensor versions are checked during backward so an
//! in-place mutation cannot silently invalidate a formula. Kernel-facing shape
//! arithmetic is checked before offsets or lengths reach CPU/CUDA code.

use super::Tensor;
use crate::dispatch::BinaryOp;
use crate::extension::CustomUnaryOp;
use crate::shape::{
    broadcast_flat_index, broadcast_shapes, checked_numel, flatten_index, for_each_index, numel,
};
use crate::{DType, Device, Result, TensorError};
use heirloom_kernels::cuda::{
    CudaBuffer, MemoryProductKeySelectedScoreDims, MemorySelectedScoreDims, MemoryWeightedValueDims,
};
use std::collections::HashSet;

#[derive(Clone)]
pub(super) struct SavedTensor {
    tensor: Tensor,
    version: usize,
    purpose: &'static str,
}

#[derive(Clone)]
pub(super) enum GradFn {
    Binary {
        op: BinaryOp,
        left: SavedTensor,
        right: SavedTensor,
        output_shape: Vec<usize>,
    },
    Matmul {
        left: SavedTensor,
        right: SavedTensor,
    },
    MatmulBias {
        left: SavedTensor,
        right: SavedTensor,
        bias: SavedTensor,
    },
    Permute {
        input: Tensor,
        dims: Vec<usize>,
    },
    Relu {
        input: SavedTensor,
    },
    Gelu {
        input: SavedTensor,
    },
    LayerNormLastDim {
        input: SavedTensor,
        weight: SavedTensor,
        bias: SavedTensor,
        eps: f64,
    },
    Embedding {
        weight: SavedTensor,
        indices: EmbeddingIndices,
        embedding_dim: usize,
    },
    MemoryWeightedValue {
        indices: EmbeddingIndices,
        weights: SavedTensor,
        values: SavedTensor,
        dims: MemoryWeightedValueDims,
    },
    MemorySelectedScores {
        indices: EmbeddingIndices,
        query: SavedTensor,
        keys: SavedTensor,
        dims: MemorySelectedScoreDims,
    },
    MemoryProductKeySelectedScores {
        indices: EmbeddingIndices,
        query: SavedTensor,
        left_keys: SavedTensor,
        right_keys: SavedTensor,
        dims: MemoryProductKeySelectedScoreDims,
    },
    MaskedFill {
        input: Tensor,
        mask: Vec<bool>,
    },
    SoftmaxDim {
        output_data: Vec<f64>,
        output_shape: Vec<usize>,
        dim: usize,
        input: Tensor,
    },
    CausalSelfAttention {
        query: SavedTensor,
        key: SavedTensor,
        value: SavedTensor,
        attention: CausalAttentionWeights,
        n_heads: usize,
        amp_bf16_tensor_core: bool,
    },
    CrossEntropy {
        logits: SavedTensor,
        targets: CrossEntropyTargets,
    },
    CustomUnary {
        op: CustomUnaryOp,
        input: SavedTensor,
        output_data: Vec<f64>,
        output_shape: Vec<usize>,
    },
    Cast {
        input: Tensor,
    },
    Sum {
        input: Tensor,
    },
    Mean {
        input: Tensor,
    },
    ReduceDim {
        input: Tensor,
        dim: usize,
        keepdim: bool,
        kind: ReduceDimKind,
    },
    View {
        input: Tensor,
    },
    Expand {
        input: Tensor,
        output_shape: Vec<usize>,
    },
    Narrow {
        input: Tensor,
        dim: usize,
        start: usize,
    },
    Copy {
        input: Tensor,
    },
}

#[derive(Clone)]
pub(super) enum EmbeddingIndices {
    Cpu(Vec<usize>),
    Cuda(SavedTensor),
}

#[derive(Clone)]
pub(super) enum CrossEntropyTargets {
    Cpu(Vec<usize>),
    Cuda(SavedTensor),
}

#[derive(Clone)]
pub(super) enum CausalAttentionWeights {
    Cpu(Vec<f64>),
    Cuda(SavedTensor),
    CudaFlash {
        output: SavedTensor,
        row_max: SavedTensor,
        row_denom: SavedTensor,
    },
}

#[derive(Clone, Copy)]
pub(super) enum ReduceDimKind {
    Sum,
    Mean,
}

impl SavedTensor {
    pub(super) fn new(tensor: &Tensor, purpose: &'static str) -> Self {
        Self {
            tensor: tensor.clone(),
            version: tensor.storage_version(),
            purpose,
        }
    }

    fn unpack(&self) -> Result<Tensor> {
        let current = self.tensor.storage_version();
        if current != self.version {
            return Err(TensorError::Autograd(format!(
                "saved tensor for {} was modified in-place before backward: saved version {}, current version {}",
                self.purpose, self.version, current
            )));
        }
        Ok(self.tensor.clone())
    }

    fn tensor(&self) -> Tensor {
        self.tensor.clone()
    }
}

impl GradFn {
    fn parents(&self) -> Vec<Tensor> {
        match self {
            Self::Binary { left, right, .. } => vec![left.tensor(), right.tensor()],
            Self::Matmul { left, right } => vec![left.tensor(), right.tensor()],
            Self::MatmulBias { left, right, bias } => {
                vec![left.tensor(), right.tensor(), bias.tensor()]
            }
            Self::Permute { input, .. }
            | Self::SoftmaxDim { input, .. }
            | Self::MaskedFill { input, .. }
            | Self::Sum { input }
            | Self::Mean { input }
            | Self::ReduceDim { input, .. }
            | Self::View { input }
            | Self::Expand { input, .. }
            | Self::Narrow { input, .. }
            | Self::Copy { input } => vec![input.clone()],
            Self::Relu { input } => vec![input.tensor()],
            Self::Gelu { input } => vec![input.tensor()],
            Self::LayerNormLastDim {
                input,
                weight,
                bias,
                ..
            } => vec![input.tensor(), weight.tensor(), bias.tensor()],
            Self::Embedding { weight, .. } => vec![weight.tensor()],
            Self::MemoryWeightedValue {
                weights, values, ..
            } => vec![weights.tensor(), values.tensor()],
            Self::MemorySelectedScores { query, keys, .. } => {
                vec![query.tensor(), keys.tensor()]
            }
            Self::MemoryProductKeySelectedScores {
                query,
                left_keys,
                right_keys,
                ..
            } => vec![query.tensor(), left_keys.tensor(), right_keys.tensor()],
            Self::CrossEntropy { logits, .. } => vec![logits.tensor()],
            Self::CustomUnary { input, .. } => vec![input.tensor()],
            Self::Cast { input } => vec![input.clone()],
            Self::CausalSelfAttention {
                query, key, value, ..
            } => vec![query.tensor(), key.tensor(), value.tensor()],
        }
    }

    pub(super) fn backward(&self, grad_output: &[f64]) -> Result<Vec<(Tensor, Vec<f64>)>> {
        match self {
            Self::Binary {
                op,
                left,
                right,
                output_shape,
            } => backward_binary(*op, left, right, output_shape, grad_output),
            Self::Matmul { left, right } => backward_matmul(left, right, grad_output),
            Self::MatmulBias { .. } => Err(TensorError::Autograd(
                "CPU backward is not implemented for fused Tensor Core matmul+bias".to_string(),
            )),
            Self::Permute { input, dims } => backward_permute(input, dims, grad_output),
            Self::Relu { input } => backward_relu(input, grad_output),
            Self::Gelu { input } => backward_gelu(input, grad_output),
            Self::LayerNormLastDim {
                input,
                weight,
                bias,
                eps,
            } => backward_layer_norm_last_dim(input, weight, bias, *eps, grad_output),
            Self::Embedding {
                weight,
                indices,
                embedding_dim,
            } => backward_embedding(weight, indices, *embedding_dim, grad_output),
            Self::MemoryWeightedValue {
                indices,
                weights,
                values,
                dims,
            } => backward_memory_weighted_value(indices, weights, values, *dims, grad_output),
            Self::MemorySelectedScores {
                indices,
                query,
                keys,
                dims,
            } => backward_memory_selected_scores(indices, query, keys, *dims, grad_output),
            Self::MemoryProductKeySelectedScores {
                indices,
                query,
                left_keys,
                right_keys,
                dims,
            } => backward_memory_product_key_selected_scores(
                indices,
                query,
                left_keys,
                right_keys,
                *dims,
                grad_output,
            ),
            Self::MaskedFill { input, mask } => backward_masked_fill(input, mask, grad_output),
            Self::SoftmaxDim {
                output_data,
                output_shape,
                dim,
                input,
            } => backward_softmax_dim(input, output_data, output_shape, *dim, grad_output),
            Self::CrossEntropy { logits, targets } => {
                backward_cross_entropy(logits, targets, grad_output)
            }
            Self::CausalSelfAttention {
                query,
                key,
                value,
                attention,
                n_heads,
                amp_bf16_tensor_core: _,
            } => {
                backward_causal_self_attention(query, key, value, attention, *n_heads, grad_output)
            }
            Self::CustomUnary {
                op,
                input,
                output_data,
                output_shape,
            } => backward_custom_unary(*op, input, output_data, output_shape, grad_output),
            Self::Cast { input } => {
                if !input.requires_grad() {
                    return Ok(Vec::new());
                }
                Ok(vec![(input.clone(), grad_output.to_vec())])
            }
            Self::Sum { input } => {
                if !input.requires_grad() {
                    return Ok(Vec::new());
                }
                Ok(vec![(input.clone(), vec![grad_output[0]; input.numel()])])
            }
            Self::Mean { input } => {
                if !input.requires_grad() {
                    return Ok(Vec::new());
                }
                let scale = grad_output[0] / input.numel() as f64;
                Ok(vec![(input.clone(), vec![scale; input.numel()])])
            }
            Self::ReduceDim {
                input,
                dim,
                keepdim,
                kind,
            } => backward_reduce_dim(input, *dim, *keepdim, *kind, grad_output),
            Self::View { input } | Self::Copy { input } => {
                if !input.requires_grad() {
                    return Ok(Vec::new());
                }
                Ok(vec![(input.clone(), grad_output.to_vec())])
            }
            Self::Expand {
                input,
                output_shape,
            } => {
                if !input.requires_grad() {
                    return Ok(Vec::new());
                }
                Ok(vec![(
                    input.clone(),
                    unbroadcast_gradient_f64(grad_output, output_shape, &input.shape()),
                )])
            }
            Self::Narrow { input, dim, start } => backward_narrow(input, *dim, *start, grad_output),
        }
    }

    pub(super) fn backward_cuda_f32(
        &self,
        grad_output: &CudaBuffer,
    ) -> Result<Vec<(Tensor, CudaBuffer)>> {
        match self {
            Self::Binary {
                op,
                left,
                right,
                output_shape,
            } => backward_binary_cuda_f32(*op, left, right, output_shape, grad_output),
            Self::Matmul { left, right } => backward_matmul_cuda_f32(left, right, grad_output),
            Self::MatmulBias { left, right, bias } => {
                backward_matmul_bias_cuda_f32(left, right, bias, grad_output)
            }
            Self::Permute { input, dims } => backward_permute_cuda_f32(input, dims, grad_output),
            Self::Relu { input } => backward_relu_cuda_f32(input, grad_output),
            Self::Gelu { input } => backward_gelu_cuda_f32(input, grad_output),
            Self::LayerNormLastDim {
                input,
                weight,
                bias,
                eps,
            } => backward_layer_norm_last_dim_cuda_f32(input, weight, bias, *eps, grad_output),
            Self::Embedding {
                weight,
                indices,
                embedding_dim,
            } => backward_embedding_cuda_f32(weight, indices, *embedding_dim, grad_output),
            Self::MemoryWeightedValue {
                indices,
                weights,
                values,
                dims,
            } => backward_memory_weighted_value_cuda_f32(
                indices,
                weights,
                values,
                *dims,
                grad_output,
            ),
            Self::MemorySelectedScores {
                indices,
                query,
                keys,
                dims,
            } => backward_memory_selected_scores_cuda_f32(indices, query, keys, *dims, grad_output),
            Self::MemoryProductKeySelectedScores {
                indices,
                query,
                left_keys,
                right_keys,
                dims,
            } => backward_memory_product_key_selected_scores_cuda_f32(
                indices,
                query,
                left_keys,
                right_keys,
                *dims,
                grad_output,
            ),
            Self::CrossEntropy { logits, targets } => {
                backward_cross_entropy_cuda_f32(logits, targets, grad_output)
            }
            Self::CausalSelfAttention {
                query,
                key,
                value,
                attention,
                n_heads,
                amp_bf16_tensor_core,
            } => backward_causal_self_attention_cuda_f32(
                query,
                key,
                value,
                attention,
                *n_heads,
                *amp_bf16_tensor_core,
                grad_output,
            ),
            Self::SoftmaxDim {
                output_shape,
                dim,
                input,
                ..
            } => backward_softmax_dim_cuda_f32(input, output_shape, *dim, grad_output),
            Self::Sum { input } => backward_sum_cuda_f32(input, grad_output),
            Self::Mean { input } => backward_mean_cuda_f32(input, grad_output),
            Self::Cast { input } => backward_cast_cuda_f32(input, grad_output),
            Self::View { input } | Self::Copy { input } => {
                backward_view_cuda_f32(input, grad_output)
            }
            other => Err(TensorError::Autograd(format!(
                "CUDA backward is not implemented for {}",
                other.name()
            ))),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Binary { .. } => "binary",
            Self::Matmul { .. } => "matmul",
            Self::MatmulBias { .. } => "matmul_bias",
            Self::Permute { .. } => "permute",
            Self::Relu { .. } => "relu",
            Self::Gelu { .. } => "gelu",
            Self::LayerNormLastDim { .. } => "layer_norm_last_dim",
            Self::Embedding { .. } => "embedding",
            Self::MemoryWeightedValue { .. } => "memory_weighted_value",
            Self::MemorySelectedScores { .. } => "memory_selected_scores",
            Self::MemoryProductKeySelectedScores { .. } => "memory_product_key_selected_scores",
            Self::MaskedFill { .. } => "masked_fill",
            Self::SoftmaxDim { .. } => "softmax_dim",
            Self::CausalSelfAttention { .. } => "causal_self_attention",
            Self::CrossEntropy { .. } => "cross_entropy",
            Self::CustomUnary { .. } => "custom_unary",
            Self::Cast { .. } => "cast",
            Self::Sum { .. } => "sum",
            Self::Mean { .. } => "mean",
            Self::ReduceDim { .. } => "reduce_dim",
            Self::View { .. } => "view",
            Self::Expand { .. } => "expand",
            Self::Narrow { .. } => "narrow",
            Self::Copy { .. } => "copy",
        }
    }
}

fn backward_binary_cuda_f32(
    op: BinaryOp,
    left: &SavedTensor,
    right: &SavedTensor,
    output_shape: &[usize],
    grad_output: &CudaBuffer,
) -> Result<Vec<(Tensor, CudaBuffer)>> {
    let left = left.unpack()?;
    let right = right.unpack()?;
    if left.dtype() != DType::F32 || right.dtype() != DType::F32 {
        return Err(TensorError::Autograd(
            "CUDA add backward currently supports only f32 tensors".to_string(),
        ));
    }
    if left.device() != right.device() || !matches!(left.device(), Device::Cuda(_)) {
        return Err(TensorError::Autograd(format!(
            "CUDA add backward expected matching CUDA devices, got {:?} and {:?}",
            left.device(),
            right.device()
        )));
    }
    let left_shape = left.shape();
    let right_shape = right.shape();
    let bias_add_broadcast = op == BinaryOp::Add
        && left_shape == output_shape
        && output_shape.len() == 2
        && right_shape.len() == 1
        && right_shape[0] == output_shape[1];
    if !bias_add_broadcast && (left_shape != output_shape || right_shape != output_shape) {
        return Err(TensorError::Autograd(format!(
            "CUDA binary backward currently supports same-shape gradients or add bias broadcast [rows, cols] + [cols], left={:?} right={:?} output={:?}",
            left_shape,
            right_shape,
            output_shape
        )));
    }
    if grad_output.len() != numel(output_shape) {
        return Err(TensorError::Autograd(format!(
            "CUDA add backward grad_output length {} does not match output shape {:?}",
            grad_output.len(),
            output_shape
        )));
    }
    let mut grads = Vec::new();
    if bias_add_broadcast {
        if left.requires_grad() {
            grads.push((left.clone(), grad_output.clone()));
        }
        if right.requires_grad() {
            let grad_right = heirloom_kernels::cuda::bias_add_backward_bias_f32_buffer(
                grad_output,
                output_shape[0],
                output_shape[1],
            )
            .map_err(cuda_error)?;
            grads.push((right, grad_right));
        }
        return Ok(grads);
    }

    let left_buffer = left.cuda_f32_full_storage_buffer("CUDA binary backward left")?;
    let right_buffer = right.cuda_f32_full_storage_buffer("CUDA binary backward right")?;

    if left.requires_grad() {
        let grad_left = match op {
            BinaryOp::Add | BinaryOp::Sub => grad_output.clone(),
            BinaryOp::Mul => heirloom_kernels::cuda::mul_f32_buffers(grad_output, &right_buffer)
                .map_err(cuda_error)?,
            BinaryOp::Div => heirloom_kernels::cuda::div_f32_buffers(grad_output, &right_buffer)
                .map_err(cuda_error)?,
        };
        grads.push((left.clone(), grad_left));
    }
    if right.requires_grad() {
        let grad_right = match op {
            BinaryOp::Add => grad_output.clone(),
            BinaryOp::Sub => {
                heirloom_kernels::cuda::neg_f32_buffer(grad_output).map_err(cuda_error)?
            }
            BinaryOp::Mul => heirloom_kernels::cuda::mul_f32_buffers(grad_output, &left_buffer)
                .map_err(cuda_error)?,
            BinaryOp::Div => heirloom_kernels::cuda::div_backward_rhs_f32_buffer(
                &left_buffer,
                &right_buffer,
                grad_output,
            )
            .map_err(cuda_error)?,
        };
        grads.push((right, grad_right));
    }
    Ok(grads)
}

fn backward_matmul_cuda_f32(
    left: &SavedTensor,
    right: &SavedTensor,
    grad_output: &CudaBuffer,
) -> Result<Vec<(Tensor, CudaBuffer)>> {
    let left = left.unpack()?;
    let right = right.unpack()?;
    if !matches!(left.dtype(), DType::F32 | DType::BFloat16)
        || !matches!(right.dtype(), DType::F32 | DType::BFloat16)
    {
        return Err(TensorError::Autograd(format!(
            "CUDA matmul backward expected f32 or bf16 inputs, got {:?} and {:?}",
            left.dtype(),
            right.dtype()
        )));
    }
    if left.device() != right.device() || !matches!(left.device(), Device::Cuda(_)) {
        return Err(TensorError::Autograd(format!(
            "CUDA matmul backward expected matching CUDA devices, got {:?} and {:?}",
            left.device(),
            right.device()
        )));
    }
    let left_shape = left.shape();
    let right_shape = right.shape();
    if left_shape.len() != 2 || right_shape.len() != 2 {
        return Err(TensorError::Autograd(format!(
            "CUDA matmul backward currently supports only rank-2 tensors, got {:?} and {:?}",
            left_shape, right_shape
        )));
    }
    let (m, k) = (left_shape[0], left_shape[1]);
    let (right_k, n) = (right_shape[0], right_shape[1]);
    if k != right_k {
        return Err(TensorError::Autograd(format!(
            "CUDA matmul backward shape mismatch: left {:?} right {:?}",
            left_shape, right_shape
        )));
    }
    let output_len = m.checked_mul(n).ok_or_else(|| {
        TensorError::Autograd(format!(
            "CUDA matmul backward output shape overflows: {m} * {n}"
        ))
    })?;
    if grad_output.len() != output_len {
        return Err(TensorError::Autograd(format!(
            "CUDA matmul backward grad_output length {} does not match output shape [{m}, {n}]",
            grad_output.len()
        )));
    }

    let (left_buffer, left_layout) = left.cuda_matrix_layout("CUDA matmul backward left")?;
    let (right_buffer, right_layout) = right.cuda_matrix_layout("CUDA matmul backward right")?;
    if let Some(grads) = backward_matmul_cuda_bf16_tensor_core(
        &left,
        &right,
        grad_output,
        m,
        k,
        n,
        left_layout,
        right_layout,
        &left_buffer,
        &right_buffer,
    )? {
        return Ok(grads);
    }
    let left_buffer = if left.dtype() == DType::BFloat16 {
        heirloom_kernels::cuda::bf16_to_f32_buffer(&left_buffer).map_err(cuda_error)?
    } else {
        left_buffer
    };
    let right_buffer = if right.dtype() == DType::BFloat16 {
        heirloom_kernels::cuda::bf16_to_f32_buffer(&right_buffer).map_err(cuda_error)?
    } else {
        right_buffer
    };
    let dims = heirloom_kernels::cuda::MatmulStridedDims {
        left: left_layout,
        right: right_layout,
    };
    let mut grads = Vec::new();
    if left.requires_grad() {
        let grad_left = heirloom_kernels::cuda::matmul_strided_grad_left_f32_buffers(
            grad_output,
            &right_buffer,
            dims,
        )
        .map_err(cuda_error)?;
        grads.push((left.clone(), grad_left));
    }
    if right.requires_grad() {
        let grad_right = heirloom_kernels::cuda::matmul_strided_grad_right_f32_buffers(
            &left_buffer,
            grad_output,
            dims,
        )
        .map_err(cuda_error)?;
        grads.push((right, grad_right));
    }
    Ok(grads)
}

fn backward_matmul_bias_cuda_f32(
    left: &SavedTensor,
    right: &SavedTensor,
    bias: &SavedTensor,
    grad_output: &CudaBuffer,
) -> Result<Vec<(Tensor, CudaBuffer)>> {
    let left_tensor = left.unpack()?;
    let right_tensor = right.unpack()?;
    let bias_tensor = bias.unpack()?;
    let left_shape = left_tensor.shape();
    let right_shape = right_tensor.shape();
    if left_shape.len() != 2 || right_shape.len() != 2 {
        return Err(TensorError::Autograd(format!(
            "CUDA fused matmul+bias backward currently supports only rank-2 tensors, got {:?} and {:?}",
            left_shape, right_shape
        )));
    }
    let (m, k) = (left_shape[0], left_shape[1]);
    let (right_k, n) = (right_shape[0], right_shape[1]);
    if k != right_k {
        return Err(TensorError::Autograd(format!(
            "CUDA fused matmul+bias backward shape mismatch: left {:?} right {:?}",
            left_shape, right_shape
        )));
    }
    if bias_tensor.dtype() != DType::F32 || !matches!(bias_tensor.device(), Device::Cuda(_)) {
        return Err(TensorError::Autograd(format!(
            "CUDA fused matmul+bias backward expected CUDA f32 bias, got {:?} on {:?}",
            bias_tensor.dtype(),
            bias_tensor.device()
        )));
    }
    if bias_tensor.shape() != [n] {
        return Err(TensorError::Autograd(format!(
            "CUDA fused matmul+bias backward expected bias shape [{n}], got {:?}",
            bias_tensor.shape()
        )));
    }
    let output_len = m.checked_mul(n).ok_or_else(|| {
        TensorError::Autograd(format!(
            "CUDA fused matmul+bias backward output shape overflows: {m} * {n}"
        ))
    })?;
    if grad_output.len() != output_len {
        return Err(TensorError::Autograd(format!(
            "CUDA fused matmul+bias backward grad_output length {} does not match output shape [{m}, {n}]",
            grad_output.len()
        )));
    }

    let mut grads = backward_matmul_cuda_f32(left, right, grad_output)?;
    if bias_tensor.requires_grad() {
        let grad_bias =
            heirloom_kernels::cuda::bias_add_backward_bias_f32_buffer(grad_output, m, n)
                .map_err(cuda_error)?;
        grads.push((bias_tensor, grad_bias));
    }
    Ok(grads)
}

#[allow(clippy::too_many_arguments)]
fn backward_matmul_cuda_bf16_tensor_core(
    left: &Tensor,
    right: &Tensor,
    grad_output: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
    left_layout: heirloom_kernels::cuda::MatrixLayout,
    right_layout: heirloom_kernels::cuda::MatrixLayout,
    left_buffer: &CudaBuffer,
    right_buffer: &CudaBuffer,
) -> Result<Option<Vec<(Tensor, CudaBuffer)>>> {
    if left.dtype() != DType::BFloat16 || right.dtype() != DType::BFloat16 {
        return Ok(None);
    }
    let Device::Cuda(device_id) = left.device() else {
        return Ok(None);
    };
    let device_supported =
        heirloom_kernels::cuda::device_supports_bf16_tensor_cores(device_id as i32)
            .map_err(cuda_error)?;
    let grad_left_supported = !left.requires_grad()
        || heirloom_kernels::cuda::bf16_tensor_core_matmul_shape_supported(m, n, k);
    let grad_right_supported = !right.requires_grad()
        || heirloom_kernels::cuda::bf16_tensor_core_matmul_shape_supported(k, m, n);

    if !(device_supported && grad_left_supported && grad_right_supported) {
        if heirloom_kernels::cuda::tensor_core_requirement_enabled() {
            return Err(TensorError::Device(format!(
                "HEIRLOOM_REQUIRE_TENSOR_CORES=1 requires BF16 Tensor Core matmul backward, \
                 but support was device_supported={device_supported} \
                 grad_left_supported={grad_left_supported} grad_right_supported={grad_right_supported} \
                 for shapes left=[{m},{k}] right=[{k},{n}]"
            )));
        }
        return Ok(None);
    }

    let grad_output_bf16 =
        heirloom_kernels::cuda::f32_to_bf16_buffer(grad_output).map_err(cuda_error)?;
    let mut grads = Vec::new();
    if left.requires_grad() {
        let grad_left = if heirloom_kernels::cuda::tensor_core_normal_rhs_gemm_enabled()
            && is_simple_transposed_bf16_matrix_layout(right_layout)
            && n.checked_mul(k) == Some(right_buffer.len())
            && heirloom_kernels::cuda::bf16_tensor_core_matmul_exact_tile_shape_supported(m, n, k)
        {
            heirloom_kernels::cuda::matmul_bf16_tensor_core_normal_rhs_f32_buffers_backward(
                &grad_output_bf16,
                right_buffer,
                m,
                n,
                k,
            )
            .map_err(cuda_error)?
        } else {
            let right_logical =
                materialize_bf16_matrix_layout_if_needed(right_buffer, right_layout)
                    .map_err(cuda_error)?;
            heirloom_kernels::cuda::matmul_bf16_tensor_core_rhs_t_f32_buffers_backward(
                &grad_output_bf16,
                &right_logical,
                m,
                n,
                k,
            )
            .map_err(cuda_error)?
        };
        grads.push((left.clone(), grad_left));
    }
    if right.requires_grad() {
        let left_logical = materialize_bf16_matrix_layout_if_needed(left_buffer, left_layout)
            .map_err(cuda_error)?;
        let (left_t, grad_output_t) = heirloom_kernels::cuda::transpose2d_pair_bf16_buffers(
            &left_logical,
            m,
            k,
            &grad_output_bf16,
            m,
            n,
        )
        .map_err(cuda_error)?;
        let grad_right =
            heirloom_kernels::cuda::matmul_bf16_tensor_core_rhs_t_f32_buffers_backward(
                &left_t,
                &grad_output_t,
                k,
                m,
                n,
            )
            .map_err(cuda_error)?;
        grads.push((right.clone(), grad_right));
    }
    Ok(Some(grads))
}

fn is_simple_transposed_bf16_matrix_layout(layout: heirloom_kernels::cuda::MatrixLayout) -> bool {
    layout.offset == 0 && layout.row_stride == 1 && layout.col_stride == layout.rows
}

fn materialize_bf16_matrix_layout_if_needed(
    buffer: &CudaBuffer,
    layout: heirloom_kernels::cuda::MatrixLayout,
) -> heirloom_kernels::cuda::CudaResult<CudaBuffer> {
    if layout.offset == 0 && layout.col_stride == 1 && layout.row_stride == layout.cols {
        return Ok(buffer.clone());
    }
    heirloom_kernels::cuda::materialize_matrix_layout_bf16_buffer(buffer, layout)
}

fn backward_permute_cuda_f32(
    input: &Tensor,
    dims: &[usize],
    grad_output: &CudaBuffer,
) -> Result<Vec<(Tensor, CudaBuffer)>> {
    if !input.requires_grad() {
        return Ok(Vec::new());
    }
    if !matches!(input.dtype(), DType::F32 | DType::BFloat16)
        || !matches!(input.device(), Device::Cuda(_))
    {
        return Err(TensorError::Autograd(format!(
            "CUDA permute backward expected CUDA f32/bf16 input, got {:?} on {:?}",
            input.dtype(),
            input.device()
        )));
    }
    let input_shape = input.shape();
    if input_shape.len() != 2 || dims != [1, 0] {
        return Err(TensorError::Autograd(format!(
            "CUDA permute backward currently supports only rank-2 transpose dims [1, 0], got input {:?} dims {:?}",
            input_shape,
            dims
        )));
    }
    if grad_output.len() != input.numel() {
        let output_shape = dims.iter().map(|&dim| input_shape[dim]).collect::<Vec<_>>();
        return Err(TensorError::Autograd(format!(
            "CUDA permute backward grad_output length {} does not match output shape {:?}",
            grad_output.len(),
            output_shape
        )));
    }
    let grad_input =
        heirloom_kernels::cuda::transpose2d_f32_buffer(grad_output, input_shape[1], input_shape[0])
            .map_err(cuda_error)?;
    Ok(vec![(input.clone(), grad_input)])
}

fn backward_view_cuda_f32(
    input: &Tensor,
    grad_output: &CudaBuffer,
) -> Result<Vec<(Tensor, CudaBuffer)>> {
    if !input.requires_grad() {
        return Ok(Vec::new());
    }
    if !matches!(input.dtype(), DType::F32 | DType::BFloat16)
        || !matches!(input.device(), Device::Cuda(_))
    {
        return Err(TensorError::Autograd(format!(
            "CUDA view backward expected CUDA f32/bf16 input, got {:?} on {:?}",
            input.dtype(),
            input.device()
        )));
    }
    if grad_output.len() != input.numel() {
        return Err(TensorError::Autograd(format!(
            "CUDA view backward grad_output length {} does not match input shape {:?}",
            grad_output.len(),
            input.shape()
        )));
    }
    Ok(vec![(input.clone(), grad_output.clone())])
}

fn backward_cast_cuda_f32(
    input: &Tensor,
    grad_output: &CudaBuffer,
) -> Result<Vec<(Tensor, CudaBuffer)>> {
    if !input.requires_grad() {
        return Ok(Vec::new());
    }
    if input.device() == Device::Cpu {
        return Err(TensorError::Autograd(
            "CUDA cast backward received a CPU input".to_string(),
        ));
    }
    if grad_output.len() != input.numel() {
        return Err(TensorError::Autograd(format!(
            "CUDA cast backward expected grad length {} for input shape {:?}, got {}",
            input.numel(),
            input.shape(),
            grad_output.len()
        )));
    }
    Ok(vec![(input.clone(), grad_output.clone())])
}

fn backward_relu_cuda_f32(
    input: &SavedTensor,
    grad_output: &CudaBuffer,
) -> Result<Vec<(Tensor, CudaBuffer)>> {
    let input = input.unpack()?;
    if !input.requires_grad() {
        return Ok(Vec::new());
    }
    if input.dtype() != DType::F32 || !matches!(input.device(), Device::Cuda(_)) {
        return Err(TensorError::Autograd(format!(
            "CUDA relu backward expected CUDA f32 input, got {:?} on {:?}",
            input.dtype(),
            input.device()
        )));
    }
    if grad_output.len() != input.numel() {
        return Err(TensorError::Autograd(format!(
            "CUDA relu backward grad_output length {} does not match input shape {:?}",
            grad_output.len(),
            input.shape()
        )));
    }
    let input_buffer = input.cuda_f32_full_storage_buffer("CUDA relu backward input")?;
    let grad_input = heirloom_kernels::cuda::relu_backward_f32_buffer(&input_buffer, grad_output)
        .map_err(cuda_error)?;
    Ok(vec![(input, grad_input)])
}

fn backward_gelu_cuda_f32(
    input: &SavedTensor,
    grad_output: &CudaBuffer,
) -> Result<Vec<(Tensor, CudaBuffer)>> {
    let input = input.unpack()?;
    if !input.requires_grad() {
        return Ok(Vec::new());
    }
    if input.dtype() != DType::F32 || !matches!(input.device(), Device::Cuda(_)) {
        return Err(TensorError::Autograd(format!(
            "CUDA gelu backward expected CUDA f32 input, got {:?} on {:?}",
            input.dtype(),
            input.device()
        )));
    }
    if grad_output.len() != input.numel() {
        return Err(TensorError::Autograd(format!(
            "CUDA gelu backward grad_output length {} does not match input shape {:?}",
            grad_output.len(),
            input.shape()
        )));
    }
    let input_buffer = input.cuda_f32_full_storage_buffer("CUDA gelu backward input")?;
    let grad_input = heirloom_kernels::cuda::gelu_backward_f32_buffer(&input_buffer, grad_output)
        .map_err(cuda_error)?;
    Ok(vec![(input, grad_input)])
}

fn backward_layer_norm_last_dim_cuda_f32(
    input: &SavedTensor,
    weight: &SavedTensor,
    bias: &SavedTensor,
    eps: f64,
    grad_output: &CudaBuffer,
) -> Result<Vec<(Tensor, CudaBuffer)>> {
    let input = input.unpack()?;
    let weight = weight.unpack()?;
    let bias = bias.unpack()?;
    if input.dtype() != DType::F32 || weight.dtype() != DType::F32 || bias.dtype() != DType::F32 {
        return Err(TensorError::Autograd(format!(
            "CUDA layer_norm backward expected f32 tensors, got {:?}, {:?}, {:?}",
            input.dtype(),
            weight.dtype(),
            bias.dtype()
        )));
    }
    if input.device() != weight.device()
        || input.device() != bias.device()
        || !matches!(input.device(), Device::Cuda(_))
    {
        return Err(TensorError::Autograd(format!(
            "CUDA layer_norm backward expected matching CUDA devices, got {:?}, {:?}, {:?}",
            input.device(),
            weight.device(),
            bias.device()
        )));
    }
    let input_shape = input.shape();
    let features = *input_shape.last().ok_or_else(|| {
        TensorError::Autograd("CUDA layer_norm backward requires rank >= 1 input".to_string())
    })?;
    if features == 0 {
        return Err(TensorError::Autograd(
            "CUDA layer_norm backward requires features > 0".to_string(),
        ));
    }
    if weight.shape() != vec![features] || bias.shape() != vec![features] {
        return Err(TensorError::Autograd(format!(
            "CUDA layer_norm backward expected weight/bias shape [{features}], got {:?} and {:?}",
            weight.shape(),
            bias.shape()
        )));
    }
    if grad_output.len() != input.numel() {
        return Err(TensorError::Autograd(format!(
            "CUDA layer_norm backward grad_output length {} does not match input shape {:?}",
            grad_output.len(),
            input_shape
        )));
    }
    let rows = input.numel() / features;
    let input_buffer = input.cuda_f32_full_storage_buffer("CUDA layer_norm backward input")?;
    let mut grads = Vec::new();

    if input.requires_grad() {
        let weight_buffer =
            weight.cuda_f32_full_storage_buffer("CUDA layer_norm backward weight")?;
        let grad_input = heirloom_kernels::cuda::layer_norm_backward_input_f32_buffers(
            &input_buffer,
            &weight_buffer,
            grad_output,
            rows,
            features,
            eps as f32,
        )
        .map_err(cuda_error)?;
        grads.push((input.clone(), grad_input));
    }

    if weight.requires_grad() || bias.requires_grad() {
        let (grad_weight, grad_bias) =
            heirloom_kernels::cuda::layer_norm_backward_weight_bias_f32_buffers(
                &input_buffer,
                grad_output,
                rows,
                features,
                eps as f32,
            )
            .map_err(cuda_error)?;
        if weight.requires_grad() {
            grads.push((weight, grad_weight));
        }
        if bias.requires_grad() {
            grads.push((bias, grad_bias));
        }
    }

    Ok(grads)
}

fn backward_embedding_cuda_f32(
    weight: &SavedTensor,
    indices: &EmbeddingIndices,
    embedding_dim: usize,
    grad_output: &CudaBuffer,
) -> Result<Vec<(Tensor, CudaBuffer)>> {
    let weight = weight.unpack()?;
    if !weight.requires_grad() {
        return Ok(Vec::new());
    }
    let indices = match indices {
        EmbeddingIndices::Cuda(indices) => indices.unpack()?,
        EmbeddingIndices::Cpu(_) => {
            return Err(TensorError::Autograd(
                "CUDA embedding backward expected saved CUDA indices".to_string(),
            ))
        }
    };
    if weight.dtype() != DType::F32 || !matches!(weight.device(), Device::Cuda(_)) {
        return Err(TensorError::Autograd(format!(
            "CUDA embedding backward expected CUDA f32 weight, got {:?} on {:?}",
            weight.dtype(),
            weight.device()
        )));
    }
    if indices.dtype() != DType::I64 || indices.device() != weight.device() {
        return Err(TensorError::Autograd(format!(
            "CUDA embedding backward expected CUDA i64 indices on {:?}, got {:?} on {:?}",
            weight.device(),
            indices.dtype(),
            indices.device()
        )));
    }
    let weight_shape = weight.shape();
    if weight_shape.len() != 2 || weight_shape[1] != embedding_dim {
        return Err(TensorError::Autograd(format!(
            "CUDA embedding backward saved weight shape {:?} does not match embedding_dim {embedding_dim}",
            weight_shape
        )));
    }
    let expected_grad_len = indices.numel().checked_mul(embedding_dim).ok_or_else(|| {
        TensorError::Autograd(format!(
            "CUDA embedding backward grad_output shape overflow: {} * {embedding_dim}",
            indices.numel()
        ))
    })?;
    if grad_output.len() != expected_grad_len {
        return Err(TensorError::Autograd(format!(
            "CUDA embedding backward grad_output length {} does not match indices {:?} and embedding_dim {embedding_dim}",
            grad_output.len(),
            indices.shape()
        )));
    }

    let index_buffer = indices.cuda_i64_full_storage_buffer("CUDA embedding backward indices")?;
    let grad_weight = heirloom_kernels::cuda::embedding_backward_f32_i64_buffers(
        &index_buffer,
        grad_output,
        weight_shape[0],
        embedding_dim,
    )
    .map_err(cuda_error)?;
    Ok(vec![(weight, grad_weight)])
}

fn backward_memory_weighted_value_cuda_f32(
    indices: &EmbeddingIndices,
    weights: &SavedTensor,
    values: &SavedTensor,
    dims: MemoryWeightedValueDims,
    grad_output: &CudaBuffer,
) -> Result<Vec<(Tensor, CudaBuffer)>> {
    let weights = weights.unpack()?;
    let values = values.unpack()?;
    let indices = match indices {
        EmbeddingIndices::Cuda(indices) => indices.unpack()?,
        EmbeddingIndices::Cpu(_) => {
            return Err(TensorError::Autograd(
                "CUDA memory weighted value backward expected saved CUDA indices".to_string(),
            ))
        }
    };
    validate_memory_weighted_value_autograd_shapes(&weights, &values, dims, grad_output.len())?;
    if indices.dtype() != DType::I64
        || indices.device() != weights.device()
        || indices.device() != values.device()
    {
        return Err(TensorError::Autograd(format!(
            "CUDA memory weighted value backward expected CUDA i64 indices and matching f32 tensors, got indices {:?} on {:?}, weights {:?} on {:?}, values {:?} on {:?}",
            indices.dtype(),
            indices.device(),
            weights.dtype(),
            weights.device(),
            values.dtype(),
            values.device()
        )));
    }
    if !matches!(weights.device(), Device::Cuda(_))
        || weights.dtype() != DType::F32
        || values.dtype() != DType::F32
    {
        return Err(TensorError::Autograd(format!(
            "CUDA memory weighted value backward expected f32 CUDA weights/values, got weights {:?} on {:?}, values {:?} on {:?}",
            weights.dtype(),
            weights.device(),
            values.dtype(),
            values.device()
        )));
    }
    if indices.shape() != vec![dims.tokens, dims.top_k] {
        return Err(TensorError::Autograd(format!(
            "CUDA memory weighted value backward expected indices [{}, {}], got {:?}",
            dims.tokens,
            dims.top_k,
            indices.shape()
        )));
    }
    let index_buffer =
        indices.cuda_i64_full_storage_buffer("CUDA memory weighted value backward indices")?;
    let weights_buffer =
        weights.cuda_f32_full_storage_buffer("CUDA memory weighted value backward weights")?;
    let values_buffer =
        values.cuda_f32_full_storage_buffer("CUDA memory weighted value backward values")?;
    let mut grads = Vec::new();
    if weights.requires_grad() {
        let grad_weights =
            heirloom_kernels::cuda::memory_weighted_value_backward_weights_f32_i64_buffers(
                &index_buffer,
                &values_buffer,
                grad_output,
                dims,
            )
            .map_err(cuda_error)?;
        grads.push((weights.clone(), grad_weights));
    }
    if values.requires_grad() {
        let grad_values =
            heirloom_kernels::cuda::memory_weighted_value_backward_values_f32_i64_buffers(
                &index_buffer,
                &weights_buffer,
                grad_output,
                dims,
            )
            .map_err(cuda_error)?;
        grads.push((values, grad_values));
    }
    Ok(grads)
}

fn backward_memory_selected_scores_cuda_f32(
    indices: &EmbeddingIndices,
    query: &SavedTensor,
    keys: &SavedTensor,
    dims: MemorySelectedScoreDims,
    grad_output: &CudaBuffer,
) -> Result<Vec<(Tensor, CudaBuffer)>> {
    let query = query.unpack()?;
    let keys = keys.unpack()?;
    let indices = match indices {
        EmbeddingIndices::Cuda(indices) => indices.unpack()?,
        EmbeddingIndices::Cpu(_) => {
            return Err(TensorError::Autograd(
                "CUDA memory selected scores backward expected saved CUDA indices".to_string(),
            ))
        }
    };
    validate_memory_selected_score_autograd_shapes(&query, &keys, dims, grad_output.len())?;
    if indices.dtype() != DType::I64
        || indices.device() != query.device()
        || indices.device() != keys.device()
    {
        return Err(TensorError::Autograd(format!(
            "CUDA memory selected scores backward expected CUDA i64 indices and matching f32 tensors, got indices {:?} on {:?}, query {:?} on {:?}, keys {:?} on {:?}",
            indices.dtype(),
            indices.device(),
            query.dtype(),
            query.device(),
            keys.dtype(),
            keys.device()
        )));
    }
    if !matches!(query.device(), Device::Cuda(_))
        || query.dtype() != DType::F32
        || keys.dtype() != DType::F32
    {
        return Err(TensorError::Autograd(format!(
            "CUDA memory selected scores backward expected f32 CUDA query/keys, got query {:?} on {:?}, keys {:?} on {:?}",
            query.dtype(),
            query.device(),
            keys.dtype(),
            keys.device()
        )));
    }
    if indices.shape() != vec![dims.tokens, dims.top_k] {
        return Err(TensorError::Autograd(format!(
            "CUDA memory selected scores backward expected indices [{}, {}], got {:?}",
            dims.tokens,
            dims.top_k,
            indices.shape()
        )));
    }
    let index_buffer =
        indices.cuda_i64_full_storage_buffer("CUDA memory selected scores backward indices")?;
    let query_buffer =
        query.cuda_f32_full_storage_buffer("CUDA memory selected scores backward query")?;
    let key_buffer =
        keys.cuda_f32_full_storage_buffer("CUDA memory selected scores backward keys")?;
    let mut grads = Vec::new();
    if query.requires_grad() {
        let grad_query =
            heirloom_kernels::cuda::memory_selected_scores_backward_query_f32_i64_buffers(
                &index_buffer,
                &key_buffer,
                grad_output,
                dims,
            )
            .map_err(cuda_error)?;
        grads.push((query.clone(), grad_query));
    }
    if keys.requires_grad() {
        let grad_keys =
            heirloom_kernels::cuda::memory_selected_scores_backward_keys_f32_i64_buffers(
                &index_buffer,
                &query_buffer,
                grad_output,
                dims,
            )
            .map_err(cuda_error)?;
        grads.push((keys, grad_keys));
    }
    Ok(grads)
}

fn backward_memory_product_key_selected_scores_cuda_f32(
    indices: &EmbeddingIndices,
    query: &SavedTensor,
    left_keys: &SavedTensor,
    right_keys: &SavedTensor,
    dims: MemoryProductKeySelectedScoreDims,
    grad_output: &CudaBuffer,
) -> Result<Vec<(Tensor, CudaBuffer)>> {
    let query = query.unpack()?;
    let left_keys = left_keys.unpack()?;
    let right_keys = right_keys.unpack()?;
    let indices = match indices {
        EmbeddingIndices::Cuda(indices) => indices.unpack()?,
        EmbeddingIndices::Cpu(_) => {
            return Err(TensorError::Autograd(
                "CUDA product-key selected scores backward expected saved CUDA indices".to_string(),
            ))
        }
    };
    validate_memory_product_key_selected_score_autograd_shapes(
        &query,
        &left_keys,
        &right_keys,
        dims,
        grad_output.len(),
    )?;
    if indices.dtype() != DType::I64
        || indices.device() != query.device()
        || indices.device() != left_keys.device()
        || indices.device() != right_keys.device()
    {
        return Err(TensorError::Autograd(format!(
            "CUDA product-key selected scores backward expected CUDA i64 indices and matching f32 tensors, got indices {:?} on {:?}, query {:?} on {:?}, left {:?} on {:?}, right {:?} on {:?}",
            indices.dtype(),
            indices.device(),
            query.dtype(),
            query.device(),
            left_keys.dtype(),
            left_keys.device(),
            right_keys.dtype(),
            right_keys.device()
        )));
    }
    if !matches!(query.device(), Device::Cuda(_))
        || query.dtype() != DType::F32
        || left_keys.dtype() != DType::F32
        || right_keys.dtype() != DType::F32
    {
        return Err(TensorError::Autograd(format!(
            "CUDA product-key selected scores backward expected f32 CUDA query/half-keys, got query {:?} on {:?}, left {:?} on {:?}, right {:?} on {:?}",
            query.dtype(),
            query.device(),
            left_keys.dtype(),
            left_keys.device(),
            right_keys.dtype(),
            right_keys.device()
        )));
    }
    if indices.shape() != vec![dims.tokens, dims.top_k] {
        return Err(TensorError::Autograd(format!(
            "CUDA product-key selected scores backward expected indices [{}, {}], got {:?}",
            dims.tokens,
            dims.top_k,
            indices.shape()
        )));
    }
    let index_buffer = indices
        .cuda_i64_full_storage_buffer("CUDA product-key selected scores backward indices")?;
    let query_buffer =
        query.cuda_f32_full_storage_buffer("CUDA product-key selected scores backward query")?;
    let left_buffer = left_keys
        .cuda_f32_full_storage_buffer("CUDA product-key selected scores backward left keys")?;
    let right_buffer = right_keys
        .cuda_f32_full_storage_buffer("CUDA product-key selected scores backward right keys")?;
    let mut grads = Vec::new();
    if query.requires_grad() {
        let grad_query =
            heirloom_kernels::cuda::memory_product_key_selected_scores_backward_query_f32_i64_buffers(
                &index_buffer,
                &left_buffer,
                &right_buffer,
                grad_output,
                dims,
            )
            .map_err(cuda_error)?;
        grads.push((query.clone(), grad_query));
    }
    if left_keys.requires_grad() {
        let grad_left =
            heirloom_kernels::cuda::memory_product_key_selected_scores_backward_half_keys_f32_i64_buffers(
                &index_buffer,
                &query_buffer,
                grad_output,
                dims,
                true,
            )
            .map_err(cuda_error)?;
        grads.push((left_keys, grad_left));
    }
    if right_keys.requires_grad() {
        let grad_right =
            heirloom_kernels::cuda::memory_product_key_selected_scores_backward_half_keys_f32_i64_buffers(
                &index_buffer,
                &query_buffer,
                grad_output,
                dims,
                false,
            )
            .map_err(cuda_error)?;
        grads.push((right_keys, grad_right));
    }
    Ok(grads)
}

fn validate_memory_weighted_value_autograd_shapes(
    weights: &Tensor,
    values: &Tensor,
    dims: MemoryWeightedValueDims,
    grad_output_len: usize,
) -> Result<()> {
    if weights.shape() != vec![dims.tokens, dims.top_k] {
        return Err(TensorError::Autograd(format!(
            "memory weighted value backward expected weights [{}, {}], got {:?}",
            dims.tokens,
            dims.top_k,
            weights.shape()
        )));
    }
    if values.shape() != vec![dims.slots, dims.value_dim] {
        return Err(TensorError::Autograd(format!(
            "memory weighted value backward expected values [{}, {}], got {:?}",
            dims.slots,
            dims.value_dim,
            values.shape()
        )));
    }
    if grad_output_len != dims.tokens * dims.value_dim {
        return Err(TensorError::Autograd(format!(
            "memory weighted value backward grad_output length {} does not match [{}, {}]",
            grad_output_len, dims.tokens, dims.value_dim
        )));
    }
    Ok(())
}

fn validate_memory_selected_score_autograd_shapes(
    query: &Tensor,
    keys: &Tensor,
    dims: MemorySelectedScoreDims,
    grad_output_len: usize,
) -> Result<()> {
    if query.shape() != vec![dims.tokens, dims.key_dim] {
        return Err(TensorError::Autograd(format!(
            "memory selected scores backward expected query [{}, {}], got {:?}",
            dims.tokens,
            dims.key_dim,
            query.shape()
        )));
    }
    if keys.shape() != vec![dims.slots, dims.key_dim] {
        return Err(TensorError::Autograd(format!(
            "memory selected scores backward expected keys [{}, {}], got {:?}",
            dims.slots,
            dims.key_dim,
            keys.shape()
        )));
    }
    if grad_output_len != dims.tokens * dims.top_k {
        return Err(TensorError::Autograd(format!(
            "memory selected scores backward grad_output length {} does not match [{}, {}]",
            grad_output_len, dims.tokens, dims.top_k
        )));
    }
    Ok(())
}

fn validate_memory_product_key_selected_score_autograd_shapes(
    query: &Tensor,
    left_keys: &Tensor,
    right_keys: &Tensor,
    dims: MemoryProductKeySelectedScoreDims,
    grad_output_len: usize,
) -> Result<()> {
    if dims.key_dim == 0 || !dims.key_dim.is_multiple_of(2) {
        return Err(TensorError::Autograd(format!(
            "product-key selected scores backward expected even non-zero key_dim, got {}",
            dims.key_dim
        )));
    }
    let half_dim = dims.key_dim / 2;
    if query.shape() != vec![dims.tokens, dims.key_dim] {
        return Err(TensorError::Autograd(format!(
            "product-key selected scores backward expected query [{}, {}], got {:?}",
            dims.tokens,
            dims.key_dim,
            query.shape()
        )));
    }
    if left_keys.shape() != vec![dims.side, half_dim] {
        return Err(TensorError::Autograd(format!(
            "product-key selected scores backward expected left keys [{}, {}], got {:?}",
            dims.side,
            half_dim,
            left_keys.shape()
        )));
    }
    if right_keys.shape() != vec![dims.side, half_dim] {
        return Err(TensorError::Autograd(format!(
            "product-key selected scores backward expected right keys [{}, {}], got {:?}",
            dims.side,
            half_dim,
            right_keys.shape()
        )));
    }
    if grad_output_len != dims.tokens * dims.top_k {
        return Err(TensorError::Autograd(format!(
            "product-key selected scores backward grad_output length {} does not match [{}, {}]",
            grad_output_len, dims.tokens, dims.top_k
        )));
    }
    Ok(())
}

fn backward_cross_entropy_cuda_f32(
    logits: &SavedTensor,
    targets: &CrossEntropyTargets,
    grad_output: &CudaBuffer,
) -> Result<Vec<(Tensor, CudaBuffer)>> {
    let logits = logits.unpack()?;
    if !logits.requires_grad() {
        return Ok(Vec::new());
    }
    let targets = match targets {
        CrossEntropyTargets::Cuda(targets) => targets.unpack()?,
        CrossEntropyTargets::Cpu(_) => {
            return Err(TensorError::Autograd(
                "CUDA cross entropy backward expected saved CUDA targets".to_string(),
            ))
        }
    };
    if logits.dtype() != DType::F32 || !matches!(logits.device(), Device::Cuda(_)) {
        return Err(TensorError::Autograd(format!(
            "CUDA cross entropy backward expected CUDA f32 logits, got {:?} on {:?}",
            logits.dtype(),
            logits.device()
        )));
    }
    if targets.dtype() != DType::I64 || targets.device() != logits.device() {
        return Err(TensorError::Autograd(format!(
            "CUDA cross entropy backward expected CUDA i64 targets on {:?}, got {:?} on {:?}",
            logits.device(),
            targets.dtype(),
            targets.device()
        )));
    }
    let shape = logits.shape();
    if shape.len() != 2 {
        return Err(TensorError::Autograd(format!(
            "CUDA cross entropy backward expected rank-2 logits, got {:?}",
            shape
        )));
    }
    let (batch, classes) = (shape[0], shape[1]);
    if targets.numel() != batch {
        return Err(TensorError::Autograd(format!(
            "CUDA cross entropy backward target length {} does not match batch {batch}",
            targets.numel()
        )));
    }
    if grad_output.len() != 1 {
        return Err(TensorError::Autograd(format!(
            "CUDA cross entropy backward expected scalar grad_output, got length {}",
            grad_output.len()
        )));
    }
    let logits_buffer =
        logits.cuda_f32_full_storage_buffer("CUDA cross entropy backward logits")?;
    let target_buffer =
        targets.cuda_i64_full_storage_buffer("CUDA cross entropy backward targets")?;
    let grad_logits = heirloom_kernels::cuda::cross_entropy_backward_f32_i64_buffers(
        &logits_buffer,
        &target_buffer,
        grad_output,
        batch,
        classes,
    )
    .map_err(cuda_error)?;
    Ok(vec![(logits, grad_logits)])
}

fn backward_causal_self_attention_cuda_f32(
    query: &SavedTensor,
    key: &SavedTensor,
    value: &SavedTensor,
    attention: &CausalAttentionWeights,
    n_heads: usize,
    amp_bf16_tensor_core: bool,
    grad_output: &CudaBuffer,
) -> Result<Vec<(Tensor, CudaBuffer)>> {
    let query = query.unpack()?;
    let key = key.unpack()?;
    let value = value.unpack()?;
    if query.dtype() != DType::F32 || key.dtype() != DType::F32 || value.dtype() != DType::F32 {
        return Err(TensorError::Autograd(format!(
            "CUDA causal attention backward expected f32 tensors, got {:?}, {:?}, {:?}",
            query.dtype(),
            key.dtype(),
            value.dtype()
        )));
    }
    if query.device() != key.device()
        || query.device() != value.device()
        || !matches!(query.device(), Device::Cuda(_))
    {
        return Err(TensorError::Autograd(format!(
            "CUDA causal attention backward expected matching CUDA devices, got {:?}, {:?}, {:?}",
            query.device(),
            key.device(),
            value.device()
        )));
    }
    let shape = query.shape();
    if shape.len() != 3 || key.shape() != shape || value.shape() != shape {
        return Err(TensorError::Autograd(format!(
            "CUDA causal attention backward expects query/key/value shape [batch, time, channels], got {:?}, {:?}, {:?}",
            shape,
            key.shape(),
            value.shape()
        )));
    }
    if n_heads == 0 {
        return Err(TensorError::Autograd(
            "CUDA causal attention backward requires n_heads > 0".to_string(),
        ));
    }
    let (batch, time, channels) = (shape[0], shape[1], shape[2]);
    if batch == 0 || time == 0 || channels == 0 {
        return Err(TensorError::Autograd(format!(
            "CUDA causal attention backward requires non-empty batch/time/channels, got {shape:?}"
        )));
    }
    if channels % n_heads != 0 {
        return Err(TensorError::Autograd(format!(
            "CUDA causal attention backward channels {channels} must be divisible by n_heads {n_heads}"
        )));
    }
    if grad_output.len() != query.numel() {
        return Err(TensorError::Autograd(format!(
            "CUDA causal attention backward grad_output length {} does not match query shape {:?}",
            grad_output.len(),
            shape
        )));
    }

    let query_buffer =
        query.cuda_f32_full_storage_buffer("CUDA causal attention backward query")?;
    let key_buffer = key.cuda_f32_full_storage_buffer("CUDA causal attention backward key")?;
    let value_buffer =
        value.cuda_f32_full_storage_buffer("CUDA causal attention backward value")?;
    let dims = heirloom_kernels::cuda::CausalAttentionDims {
        batch,
        time,
        channels,
        n_heads,
    };
    let mut grads = Vec::new();

    if let CausalAttentionWeights::CudaFlash {
        output,
        row_max,
        row_denom,
    } = attention
    {
        let output = output.unpack()?;
        let row_max = row_max.unpack()?;
        let row_denom = row_denom.unpack()?;
        let row_shape = vec![batch, n_heads, time];
        if output.dtype() != DType::F32
            || row_max.dtype() != DType::F32
            || row_denom.dtype() != DType::F32
            || output.device() != query.device()
            || row_max.device() != query.device()
            || row_denom.device() != query.device()
            || output.shape() != shape
            || row_max.shape() != row_shape
            || row_denom.shape() != row_shape
        {
            return Err(TensorError::Autograd(format!(
                "CUDA flash causal attention backward expected f32 output {:?} and row stats {:?} on {:?}, got output {:?} {:?} on {:?}, row_max {:?} {:?} on {:?}, row_denom {:?} {:?} on {:?}",
                shape,
                row_shape,
                query.device(),
                output.dtype(),
                output.shape(),
                output.device(),
                row_max.dtype(),
                row_max.shape(),
                row_max.device(),
                row_denom.dtype(),
                row_denom.shape(),
                row_denom.device()
            )));
        }
        let tensor_core_backward_supported = amp_bf16_tensor_core
            && heirloom_kernels::cuda::flash_bf16_tensor_core_attention_backward_enabled()
            && heirloom_kernels::cuda::causal_attention_bf16_flash_tensor_core_backward_shape_supported(dims)
            && match query.device() {
                Device::Cuda(device_id) => {
                    heirloom_kernels::cuda::device_supports_bf16_tensor_cores(device_id as i32)
                        .map_err(cuda_error)?
                }
                Device::Cpu => false,
            };
        if tensor_core_backward_supported {
            let output_buffer = output
                .cuda_f32_full_storage_buffer("CUDA flash causal attention backward output")?;
            let row_max_buffer = row_max
                .cuda_f32_full_storage_buffer("CUDA flash causal attention backward row_max")?;
            let row_denom_buffer = row_denom
                .cuda_f32_full_storage_buffer("CUDA flash causal attention backward row_denom")?;
            match heirloom_kernels::cuda::causal_attention_bf16_flash_tensor_core_backward_f32_buffers(
                &query_buffer,
                &key_buffer,
                &value_buffer,
                &output_buffer,
                &row_max_buffer,
                &row_denom_buffer,
                grad_output,
                dims,
            ) {
                Ok((grad_query, grad_key, grad_value)) => {
                    if query.requires_grad() {
                        grads.push((query.clone(), grad_query));
                    }
                    if key.requires_grad() {
                        grads.push((key.clone(), grad_key));
                    }
                    if value.requires_grad() {
                        grads.push((value, grad_value));
                    }
                    return Ok(grads);
                }
                Err(err) => {
                    heirloom_kernels::cuda::record_flash_bf16_tensor_core_attention_backward_fallback();
                    if heirloom_kernels::cuda::require_flash_bf16_attention_enabled() {
                        heirloom_kernels::cuda::record_flash_bf16_attention_hard_require_failure();
                        return Err(TensorError::Autograd(
                            heirloom_kernels::cuda::flash_bf16_attention_required_error(
                                &err.to_string(),
                            )
                            .to_string(),
                        ));
                    }
                    return Err(cuda_error(err));
                }
            }
        }
        heirloom_kernels::cuda::record_flash_bf16_tensor_core_attention_backward_request();
        heirloom_kernels::cuda::record_flash_bf16_tensor_core_attention_backward_fallback();
        if heirloom_kernels::cuda::require_flash_bf16_attention_enabled() {
            heirloom_kernels::cuda::record_flash_bf16_attention_hard_require_failure();
            return Err(TensorError::Autograd(
                heirloom_kernels::cuda::flash_bf16_attention_required_error(
                    "flash BF16 Tensor Core attention backward is disabled or unsupported for this shape/device",
                )
                .to_string(),
            ));
        }
        return Err(TensorError::Autograd(
            "saved flash causal attention state requires HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_BACKWARD=1 and sm80 BF16 Tensor Core support for backward"
                .to_string(),
        ));
    }

    let attention = match attention {
        CausalAttentionWeights::Cuda(attention) => attention.unpack()?,
        CausalAttentionWeights::Cpu(_) => {
            return Err(TensorError::Autograd(
                "CUDA causal attention backward expected saved CUDA attention".to_string(),
            ))
        }
        CausalAttentionWeights::CudaFlash { .. } => unreachable!("handled above"),
    };
    if query.device() != attention.device() {
        return Err(TensorError::Autograd(format!(
            "CUDA causal attention backward expected saved attention on {:?}, got {:?}",
            query.device(),
            attention.device()
        )));
    }
    let attention_shape = vec![batch, n_heads, time, time];
    if attention.shape() != attention_shape {
        return Err(TensorError::Autograd(format!(
            "CUDA causal attention backward expected saved attention shape {:?}, got {:?}",
            attention_shape,
            attention.shape()
        )));
    }
    let attention_buffer =
        attention.cuda_f32_full_storage_buffer("CUDA causal attention backward attention")?;

    let tensor_core_backward_supported = amp_bf16_tensor_core
        && heirloom_kernels::cuda::causal_attention_bf16_tensor_core_shape_supported(dims)
        && match query.device() {
            Device::Cuda(device_id) => {
                heirloom_kernels::cuda::device_supports_bf16_tensor_cores(device_id as i32)
                    .map_err(cuda_error)?
            }
            Device::Cpu => false,
        };
    if amp_bf16_tensor_core && tensor_core_backward_supported {
        let (grad_query, grad_key, grad_value) =
            heirloom_kernels::cuda::causal_attention_bf16_tensor_core_backward_f32_buffers(
                &query_buffer,
                &key_buffer,
                &value_buffer,
                &attention_buffer,
                grad_output,
                dims,
            )
            .map_err(cuda_error)?;
        if query.requires_grad() {
            grads.push((query.clone(), grad_query));
        }
        if key.requires_grad() {
            grads.push((key.clone(), grad_key));
        }
        if value.requires_grad() {
            grads.push((value, grad_value));
        }
        return Ok(grads);
    }
    if amp_bf16_tensor_core && heirloom_kernels::cuda::attention_tensor_core_requirement_enabled() {
        return Err(TensorError::Autograd(format!(
            "HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES=1 requires BF16 Tensor Core attention backward, \
             but support was amp_bf16_tensor_core={amp_bf16_tensor_core} \
             shape_supported={} for query shape {:?} n_heads={n_heads} device={:?}",
            heirloom_kernels::cuda::causal_attention_bf16_tensor_core_shape_supported(dims),
            query.shape(),
            query.device()
        )));
    }

    if query.requires_grad() {
        let grad_query = heirloom_kernels::cuda::causal_attention_backward_query_f32_buffers(
            &query_buffer,
            &key_buffer,
            &value_buffer,
            &attention_buffer,
            grad_output,
            dims,
        )
        .map_err(cuda_error)?;
        grads.push((query.clone(), grad_query));
    }
    if key.requires_grad() {
        let grad_key = heirloom_kernels::cuda::causal_attention_backward_key_f32_buffers(
            &query_buffer,
            &key_buffer,
            &value_buffer,
            &attention_buffer,
            grad_output,
            dims,
        )
        .map_err(cuda_error)?;
        grads.push((key.clone(), grad_key));
    }
    if value.requires_grad() {
        let grad_value = heirloom_kernels::cuda::causal_attention_backward_value_f32_buffers(
            &attention_buffer,
            grad_output,
            dims,
        )
        .map_err(cuda_error)?;
        grads.push((value, grad_value));
    }

    Ok(grads)
}

fn backward_sum_cuda_f32(
    input: &Tensor,
    grad_output: &CudaBuffer,
) -> Result<Vec<(Tensor, CudaBuffer)>> {
    if !input.requires_grad() {
        return Ok(Vec::new());
    }
    if grad_output.len() != 1 {
        return Err(TensorError::Autograd(format!(
            "CUDA sum backward expected scalar grad_output, got length {}",
            grad_output.len()
        )));
    }
    let grad_input =
        heirloom_kernels::cuda::fill_from_scalar_f32_buffer(grad_output, input.numel(), 1.0)
            .map_err(cuda_error)?;
    Ok(vec![(input.clone(), grad_input)])
}

fn backward_mean_cuda_f32(
    input: &Tensor,
    grad_output: &CudaBuffer,
) -> Result<Vec<(Tensor, CudaBuffer)>> {
    if !input.requires_grad() {
        return Ok(Vec::new());
    }
    if input.numel() == 0 {
        return Err(TensorError::Autograd(
            "CUDA mean backward is undefined for empty tensors".to_string(),
        ));
    }
    if grad_output.len() != 1 {
        return Err(TensorError::Autograd(format!(
            "CUDA mean backward expected scalar grad_output, got length {}",
            grad_output.len()
        )));
    }
    let scale = 1.0 / input.numel() as f32;
    let grad_input =
        heirloom_kernels::cuda::fill_from_scalar_f32_buffer(grad_output, input.numel(), scale)
            .map_err(cuda_error)?;
    Ok(vec![(input.clone(), grad_input)])
}

fn backward_binary(
    op: BinaryOp,
    left: &SavedTensor,
    right: &SavedTensor,
    output_shape: &[usize],
    grad_output: &[f64],
) -> Result<Vec<(Tensor, Vec<f64>)>> {
    let left = left.unpack()?;
    let right = right.unpack()?;
    let left_shape = left.shape();
    let right_shape = right.shape();
    let left_data = left.data_f64();
    let right_data = right.data_f64();
    let mut left_grad = vec![0.0; numel(&left_shape)];
    let mut right_grad = vec![0.0; numel(&right_shape)];
    let mut output_flat = 0;

    for_each_index(output_shape, |output_index| {
        let left_index = broadcast_flat_index(output_index, output_shape, &left_shape);
        let right_index = broadcast_flat_index(output_index, output_shape, &right_shape);
        let go = grad_output[output_flat];
        let lhs = left_data[left_index];
        let rhs = right_data[right_index];

        if left.requires_grad() {
            left_grad[left_index] += match op {
                BinaryOp::Add | BinaryOp::Sub => go,
                BinaryOp::Mul => go * rhs,
                BinaryOp::Div => go / rhs,
            };
        }
        if right.requires_grad() {
            right_grad[right_index] += match op {
                BinaryOp::Add => go,
                BinaryOp::Sub => -go,
                BinaryOp::Mul => go * lhs,
                BinaryOp::Div => -go * lhs / (rhs * rhs),
            };
        }
        output_flat += 1;
    });

    let mut grads = Vec::new();
    if left.requires_grad() {
        grads.push((left, left_grad));
    }
    if right.requires_grad() {
        grads.push((right, right_grad));
    }
    Ok(grads)
}

fn backward_matmul(
    left: &SavedTensor,
    right: &SavedTensor,
    grad_output: &[f64],
) -> Result<Vec<(Tensor, Vec<f64>)>> {
    let left = left.unpack()?;
    let right = right.unpack()?;
    let left_shape = left.shape();
    let right_shape = right.shape();
    let left_batch_shape = &left_shape[..left_shape.len() - 2];
    let right_batch_shape = &right_shape[..right_shape.len() - 2];
    let batch_shape = broadcast_shapes(left_batch_shape, right_batch_shape)?;
    let (m, k) = (
        left_shape[left_shape.len() - 2],
        left_shape[left_shape.len() - 1],
    );
    let n = right_shape[right_shape.len() - 1];
    let left_matrix_len = m.checked_mul(k).ok_or_else(|| {
        TensorError::Autograd(format!(
            "matmul backward left matrix size overflows: {m} * {k}"
        ))
    })?;
    let right_matrix_len = k.checked_mul(n).ok_or_else(|| {
        TensorError::Autograd(format!(
            "matmul backward right matrix size overflows: {k} * {n}"
        ))
    })?;
    let output_matrix_len = m.checked_mul(n).ok_or_else(|| {
        TensorError::Autograd(format!(
            "matmul backward output matrix size overflows: {m} * {n}"
        ))
    })?;
    let left_data = left.data_f64();
    let right_data = right.data_f64();
    let mut grad_left = left.requires_grad().then(|| vec![0.0; left.numel()]);
    let mut grad_right = right.requires_grad().then(|| vec![0.0; right.numel()]);
    let mut grads = Vec::new();

    let batch_count = checked_numel(&batch_shape).map_err(|error| {
        TensorError::Autograd(format!("matmul backward batch shape is invalid: {error}"))
    })?;
    let expected_grad_len = batch_count.checked_mul(output_matrix_len).ok_or_else(|| {
        TensorError::Autograd("matmul backward batched output size overflows usize".to_string())
    })?;
    if grad_output.len() != expected_grad_len {
        return Err(TensorError::Autograd(format!(
            "matmul backward gradient length {} does not match expected {expected_grad_len}",
            grad_output.len()
        )));
    }
    let mut batch_indices = Vec::with_capacity(batch_count);
    for_each_index(&batch_shape, |batch_index| {
        batch_indices.push(batch_index.to_vec());
    });
    for (output_batch, batch_index) in batch_indices.iter().enumerate() {
        let left_batch = broadcast_flat_index(batch_index, &batch_shape, left_batch_shape);
        let right_batch = broadcast_flat_index(batch_index, &batch_shape, right_batch_shape);
        let left_base = left_batch.checked_mul(left_matrix_len).ok_or_else(|| {
            TensorError::Autograd("matmul backward left batch offset overflows usize".to_string())
        })?;
        let right_base = right_batch.checked_mul(right_matrix_len).ok_or_else(|| {
            TensorError::Autograd("matmul backward right batch offset overflows usize".to_string())
        })?;
        let output_base = output_batch.checked_mul(output_matrix_len).ok_or_else(|| {
            TensorError::Autograd("matmul backward output batch offset overflows usize".to_string())
        })?;

        for row in 0..m {
            for shared in 0..k {
                for col in 0..n {
                    let go = grad_output[output_base + row * n + col];
                    if let Some(grad_left) = &mut grad_left {
                        grad_left[left_base + row * k + shared] +=
                            go * right_data[right_base + shared * n + col];
                    }
                    if let Some(grad_right) = &mut grad_right {
                        grad_right[right_base + shared * n + col] +=
                            left_data[left_base + row * k + shared] * go;
                    }
                }
            }
        }
    }

    if let Some(grad_left) = grad_left {
        grads.push((left.clone(), grad_left));
    }
    if let Some(grad_right) = grad_right {
        grads.push((right, grad_right));
    }

    Ok(grads)
}

fn backward_permute(
    input: &Tensor,
    dims: &[usize],
    grad_output: &[f64],
) -> Result<Vec<(Tensor, Vec<f64>)>> {
    if !input.requires_grad() {
        return Ok(Vec::new());
    }
    let input_shape = input.shape();
    let output_shape = dims.iter().map(|&dim| input_shape[dim]).collect::<Vec<_>>();
    let mut grad_input = vec![0.0; input.numel()];
    let mut output_flat = 0;

    for_each_index(&output_shape, |output_index| {
        let mut input_index = vec![0; input_shape.len()];
        for (output_dim, &input_dim) in dims.iter().enumerate() {
            input_index[input_dim] = output_index[output_dim];
        }
        grad_input[flatten_index(&input_index, &input_shape)] += grad_output[output_flat];
        output_flat += 1;
    });
    Ok(vec![(input.clone(), grad_input)])
}

fn backward_relu(input: &SavedTensor, grad_output: &[f64]) -> Result<Vec<(Tensor, Vec<f64>)>> {
    let input = input.unpack()?;
    if !input.requires_grad() {
        return Ok(Vec::new());
    }
    let input_data = input.data_f64();
    let grad_input = input_data
        .iter()
        .zip(grad_output.iter())
        .map(|(value, grad)| if *value > 0.0 { *grad } else { 0.0 })
        .collect::<Vec<_>>();
    Ok(vec![(input, grad_input)])
}

fn backward_gelu(input: &SavedTensor, grad_output: &[f64]) -> Result<Vec<(Tensor, Vec<f64>)>> {
    let input = input.unpack()?;
    if !input.requires_grad() {
        return Ok(Vec::new());
    }
    let input_data = input.data_f64();
    let grad_input = input_data
        .iter()
        .zip(grad_output.iter())
        .map(|(value, grad)| grad * gelu_derivative(*value))
        .collect::<Vec<_>>();
    Ok(vec![(input, grad_input)])
}

fn backward_layer_norm_last_dim(
    input: &SavedTensor,
    weight: &SavedTensor,
    bias: &SavedTensor,
    eps: f64,
    grad_output: &[f64],
) -> Result<Vec<(Tensor, Vec<f64>)>> {
    let input = input.unpack()?;
    let weight = weight.unpack()?;
    let bias = bias.unpack()?;
    let input_shape = input.shape();
    let features = *input_shape.last().ok_or_else(|| {
        TensorError::Shape("layer_norm_last_dim requires rank >= 1 input".to_string())
    })?;
    let rows = input.numel() / features;
    let input_data = input.data_f64();
    let weight_data = weight.data_f64();

    let mut grad_input = input.requires_grad().then(|| vec![0.0; input.numel()]);
    let mut grad_weight = weight.requires_grad().then(|| vec![0.0; features]);
    let mut grad_bias = bias.requires_grad().then(|| vec![0.0; features]);

    for row in 0..rows {
        let start = row * features;
        let row_values = &input_data[start..start + features];
        let mean = row_values.iter().sum::<f64>() / features as f64;
        let variance = row_values
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f64>()
            / features as f64;
        let rstd = 1.0 / (variance + eps).sqrt();
        let mut sum_dxhat = 0.0;
        let mut sum_dxhat_xhat = 0.0;
        let mut xhat = vec![0.0; features];
        let mut dxhat = vec![0.0; features];

        for col in 0..features {
            xhat[col] = (row_values[col] - mean) * rstd;
            let go = grad_output[start + col];
            dxhat[col] = go * weight_data[col];
            sum_dxhat += dxhat[col];
            sum_dxhat_xhat += dxhat[col] * xhat[col];
            if let Some(grad_weight) = &mut grad_weight {
                grad_weight[col] += go * xhat[col];
            }
            if let Some(grad_bias) = &mut grad_bias {
                grad_bias[col] += go;
            }
        }

        if let Some(grad_input) = &mut grad_input {
            for col in 0..features {
                grad_input[start + col] = rstd
                    * (features as f64 * dxhat[col] - sum_dxhat - xhat[col] * sum_dxhat_xhat)
                    / features as f64;
            }
        }
    }

    let mut grads = Vec::new();
    if let Some(grad_input) = grad_input {
        grads.push((input, grad_input));
    }
    if let Some(grad_weight) = grad_weight {
        grads.push((weight, grad_weight));
    }
    if let Some(grad_bias) = grad_bias {
        grads.push((bias, grad_bias));
    }
    Ok(grads)
}

fn backward_embedding(
    weight: &SavedTensor,
    indices: &EmbeddingIndices,
    embedding_dim: usize,
    grad_output: &[f64],
) -> Result<Vec<(Tensor, Vec<f64>)>> {
    let weight = weight.unpack()?;
    if !weight.requires_grad() {
        return Ok(Vec::new());
    }
    let indices = match indices {
        EmbeddingIndices::Cpu(indices) => indices.as_slice(),
        EmbeddingIndices::Cuda(_) => {
            return Err(TensorError::Autograd(
                "CPU embedding backward expected saved CPU indices".to_string(),
            ))
        }
    };
    let mut grad_weight = vec![0.0; weight.numel()];
    for (position, &index) in indices.iter().enumerate() {
        let grad_start = position * embedding_dim;
        let weight_start = index * embedding_dim;
        for dim in 0..embedding_dim {
            grad_weight[weight_start + dim] += grad_output[grad_start + dim];
        }
    }
    Ok(vec![(weight, grad_weight)])
}

fn backward_memory_weighted_value(
    indices: &EmbeddingIndices,
    weights: &SavedTensor,
    values: &SavedTensor,
    dims: MemoryWeightedValueDims,
    grad_output: &[f64],
) -> Result<Vec<(Tensor, Vec<f64>)>> {
    let weights = weights.unpack()?;
    let values = values.unpack()?;
    let indices = match indices {
        EmbeddingIndices::Cpu(indices) => indices.as_slice(),
        EmbeddingIndices::Cuda(_) => {
            return Err(TensorError::Autograd(
                "CPU memory weighted value backward expected saved CPU indices".to_string(),
            ))
        }
    };
    validate_memory_weighted_value_autograd_shapes(&weights, &values, dims, grad_output.len())?;
    if indices.len() != dims.tokens * dims.top_k {
        return Err(TensorError::Autograd(format!(
            "memory weighted value backward expected {} indices, got {}",
            dims.tokens * dims.top_k,
            indices.len()
        )));
    }
    let value_data = values.data_f64();
    let weight_data = weights.data_f64();
    let mut grads = Vec::new();
    if weights.requires_grad() {
        let mut grad_weights = vec![0.0; dims.tokens * dims.top_k];
        for token in 0..dims.tokens {
            for selected in 0..dims.top_k {
                let index_pos = token * dims.top_k + selected;
                let row = indices[index_pos];
                let mut grad = 0.0;
                for dim in 0..dims.value_dim {
                    grad += grad_output[token * dims.value_dim + dim]
                        * value_data[row * dims.value_dim + dim];
                }
                grad_weights[index_pos] = grad;
            }
        }
        grads.push((weights.clone(), grad_weights));
    }
    if values.requires_grad() {
        let mut grad_values = vec![0.0; dims.slots * dims.value_dim];
        for token in 0..dims.tokens {
            for selected in 0..dims.top_k {
                let index_pos = token * dims.top_k + selected;
                let row = indices[index_pos];
                let weight = weight_data[index_pos];
                for dim in 0..dims.value_dim {
                    grad_values[row * dims.value_dim + dim] +=
                        weight * grad_output[token * dims.value_dim + dim];
                }
            }
        }
        grads.push((values, grad_values));
    }
    Ok(grads)
}

fn backward_memory_selected_scores(
    indices: &EmbeddingIndices,
    query: &SavedTensor,
    keys: &SavedTensor,
    dims: MemorySelectedScoreDims,
    grad_output: &[f64],
) -> Result<Vec<(Tensor, Vec<f64>)>> {
    let query = query.unpack()?;
    let keys = keys.unpack()?;
    let indices = match indices {
        EmbeddingIndices::Cpu(indices) => indices.as_slice(),
        EmbeddingIndices::Cuda(_) => {
            return Err(TensorError::Autograd(
                "CPU memory selected scores backward expected saved CPU indices".to_string(),
            ))
        }
    };
    validate_memory_selected_score_autograd_shapes(&query, &keys, dims, grad_output.len())?;
    if indices.len() != dims.tokens * dims.top_k {
        return Err(TensorError::Autograd(format!(
            "memory selected scores backward expected {} indices, got {}",
            dims.tokens * dims.top_k,
            indices.len()
        )));
    }
    let query_data = query.data_f64();
    let key_data = keys.data_f64();
    let mut grads = Vec::new();
    if query.requires_grad() {
        let mut grad_query = vec![0.0; dims.tokens * dims.key_dim];
        for token in 0..dims.tokens {
            for selected in 0..dims.top_k {
                let index_pos = token * dims.top_k + selected;
                let row = indices[index_pos];
                let grad = grad_output[index_pos];
                for dim in 0..dims.key_dim {
                    grad_query[token * dims.key_dim + dim] +=
                        grad * key_data[row * dims.key_dim + dim];
                }
            }
        }
        grads.push((query.clone(), grad_query));
    }
    if keys.requires_grad() {
        let mut grad_keys = vec![0.0; dims.slots * dims.key_dim];
        for token in 0..dims.tokens {
            for selected in 0..dims.top_k {
                let index_pos = token * dims.top_k + selected;
                let row = indices[index_pos];
                let grad = grad_output[index_pos];
                for dim in 0..dims.key_dim {
                    grad_keys[row * dims.key_dim + dim] +=
                        grad * query_data[token * dims.key_dim + dim];
                }
            }
        }
        grads.push((keys, grad_keys));
    }
    Ok(grads)
}

fn backward_memory_product_key_selected_scores(
    indices: &EmbeddingIndices,
    query: &SavedTensor,
    left_keys: &SavedTensor,
    right_keys: &SavedTensor,
    dims: MemoryProductKeySelectedScoreDims,
    grad_output: &[f64],
) -> Result<Vec<(Tensor, Vec<f64>)>> {
    let query = query.unpack()?;
    let left_keys = left_keys.unpack()?;
    let right_keys = right_keys.unpack()?;
    let indices = match indices {
        EmbeddingIndices::Cpu(indices) => indices.as_slice(),
        EmbeddingIndices::Cuda(_) => {
            return Err(TensorError::Autograd(
                "CPU product-key selected scores backward expected saved CPU indices".to_string(),
            ))
        }
    };
    validate_memory_product_key_selected_score_autograd_shapes(
        &query,
        &left_keys,
        &right_keys,
        dims,
        grad_output.len(),
    )?;
    if indices.len() != dims.tokens * dims.top_k {
        return Err(TensorError::Autograd(format!(
            "product-key selected scores backward expected {} indices, got {}",
            dims.tokens * dims.top_k,
            indices.len()
        )));
    }
    let slots = dims.side.checked_mul(dims.side).ok_or_else(|| {
        TensorError::Autograd("product-key selected scores side^2 overflow".to_string())
    })?;
    let half_dim = dims.key_dim / 2;
    let query_data = query.data_f64();
    let left_data = left_keys.data_f64();
    let right_data = right_keys.data_f64();
    let mut grads = Vec::new();
    if query.requires_grad() {
        let mut grad_query = vec![0.0; dims.tokens * dims.key_dim];
        for token in 0..dims.tokens {
            for selected in 0..dims.top_k {
                let index_pos = token * dims.top_k + selected;
                let slot = indices[index_pos];
                if slot >= slots {
                    return Err(TensorError::Autograd(format!(
                        "product-key selected scores backward index {slot} out of range for slots {slots}"
                    )));
                }
                let left = slot / dims.side;
                let right = slot % dims.side;
                let grad = grad_output[index_pos];
                for dim in 0..half_dim {
                    grad_query[token * dims.key_dim + dim] +=
                        grad * left_data[left * half_dim + dim];
                    grad_query[token * dims.key_dim + half_dim + dim] +=
                        grad * right_data[right * half_dim + dim];
                }
            }
        }
        grads.push((query.clone(), grad_query));
    }
    if left_keys.requires_grad() {
        let mut grad_left = vec![0.0; dims.side * half_dim];
        for token in 0..dims.tokens {
            for selected in 0..dims.top_k {
                let index_pos = token * dims.top_k + selected;
                let slot = indices[index_pos];
                if slot >= slots {
                    return Err(TensorError::Autograd(format!(
                        "product-key selected scores backward index {slot} out of range for slots {slots}"
                    )));
                }
                let left = slot / dims.side;
                let grad = grad_output[index_pos];
                for dim in 0..half_dim {
                    grad_left[left * half_dim + dim] +=
                        grad * query_data[token * dims.key_dim + dim];
                }
            }
        }
        grads.push((left_keys, grad_left));
    }
    if right_keys.requires_grad() {
        let mut grad_right = vec![0.0; dims.side * half_dim];
        for token in 0..dims.tokens {
            for selected in 0..dims.top_k {
                let index_pos = token * dims.top_k + selected;
                let slot = indices[index_pos];
                if slot >= slots {
                    return Err(TensorError::Autograd(format!(
                        "product-key selected scores backward index {slot} out of range for slots {slots}"
                    )));
                }
                let right = slot % dims.side;
                let grad = grad_output[index_pos];
                for dim in 0..half_dim {
                    grad_right[right * half_dim + dim] +=
                        grad * query_data[token * dims.key_dim + half_dim + dim];
                }
            }
        }
        grads.push((right_keys, grad_right));
    }
    Ok(grads)
}

fn backward_masked_fill(
    input: &Tensor,
    mask: &[bool],
    grad_output: &[f64],
) -> Result<Vec<(Tensor, Vec<f64>)>> {
    if !input.requires_grad() {
        return Ok(Vec::new());
    }
    let grad_input = grad_output
        .iter()
        .zip(mask.iter())
        .map(|(grad, masked)| if *masked { 0.0 } else { *grad })
        .collect::<Vec<_>>();
    Ok(vec![(input.clone(), grad_input)])
}

fn backward_softmax_dim(
    input: &Tensor,
    output_data: &[f64],
    output_shape: &[usize],
    dim: usize,
    grad_output: &[f64],
) -> Result<Vec<(Tensor, Vec<f64>)>> {
    if !input.requires_grad() {
        return Ok(Vec::new());
    }

    let mut grad_input = vec![0.0; output_data.len()];
    let mut base_shape = output_shape.to_vec();
    base_shape[dim] = 1;
    for_each_index(&base_shape, |base_index| {
        let mut dot = 0.0;
        for axis_index in 0..output_shape[dim] {
            let mut index = base_index.to_vec();
            index[dim] = axis_index;
            let flat = flatten_index(&index, output_shape);
            dot += grad_output[flat] * output_data[flat];
        }

        for axis_index in 0..output_shape[dim] {
            let mut index = base_index.to_vec();
            index[dim] = axis_index;
            let flat = flatten_index(&index, output_shape);
            grad_input[flat] = output_data[flat] * (grad_output[flat] - dot);
        }
    });
    Ok(vec![(input.clone(), grad_input)])
}

fn backward_softmax_dim_cuda_f32(
    input: &Tensor,
    output_shape: &[usize],
    dim: usize,
    grad_output: &CudaBuffer,
) -> Result<Vec<(Tensor, CudaBuffer)>> {
    if !input.requires_grad() {
        return Ok(Vec::new());
    }
    if input.dtype() != DType::F32 || !matches!(input.device(), Device::Cuda(_)) {
        return Err(TensorError::Autograd(format!(
            "CUDA softmax_dim backward expected f32 CUDA input, got {:?} on {:?}",
            input.dtype(),
            input.device()
        )));
    }
    if output_shape.len() != 2 || dim != 1 {
        return Err(TensorError::Autograd(format!(
            "CUDA softmax_dim backward currently supports rank-2 dim=1 only, got shape={output_shape:?} dim={dim}"
        )));
    }
    if input.shape() != output_shape {
        return Err(TensorError::Autograd(format!(
            "CUDA softmax_dim backward expected input shape {:?}, got {:?}",
            output_shape,
            input.shape()
        )));
    }
    if grad_output.len() != numel(output_shape) {
        return Err(TensorError::Autograd(format!(
            "CUDA softmax_dim backward grad_output length {} does not match output shape {:?}",
            grad_output.len(),
            output_shape
        )));
    }
    let input_buffer = input.cuda_f32_full_storage_buffer("CUDA softmax_dim backward input")?;
    let grad_input = heirloom_kernels::cuda::softmax_dim1_backward_f32_buffer(
        &input_buffer,
        grad_output,
        output_shape[0],
        output_shape[1],
    )
    .map_err(cuda_error)?;
    Ok(vec![(input.clone(), grad_input)])
}

fn backward_cross_entropy(
    logits: &SavedTensor,
    targets: &CrossEntropyTargets,
    grad_output: &[f64],
) -> Result<Vec<(Tensor, Vec<f64>)>> {
    let logits = logits.unpack()?;
    if !logits.requires_grad() {
        return Ok(Vec::new());
    }
    let targets = match targets {
        CrossEntropyTargets::Cpu(targets) => targets.as_slice(),
        CrossEntropyTargets::Cuda(_) => {
            return Err(TensorError::Autograd(
                "CPU cross entropy backward expected saved CPU targets".to_string(),
            ))
        }
    };

    let shape = logits.shape();
    let (batch, classes) = (shape[0], shape[1]);
    let softmax = softmax_data(&logits.data_f64(), &shape, 1);
    let mut grad_logits = softmax;
    for row in 0..batch {
        grad_logits[row * classes + targets[row]] -= 1.0;
    }
    let scale = grad_output[0] / batch as f64;
    for value in &mut grad_logits {
        *value *= scale;
    }
    Ok(vec![(logits, grad_logits)])
}

fn backward_causal_self_attention(
    query: &SavedTensor,
    key: &SavedTensor,
    value: &SavedTensor,
    attention: &CausalAttentionWeights,
    n_heads: usize,
    grad_output: &[f64],
) -> Result<Vec<(Tensor, Vec<f64>)>> {
    let query = query.unpack()?;
    let key = key.unpack()?;
    let value = value.unpack()?;
    let attention =
        match attention {
            CausalAttentionWeights::Cpu(attention) => attention.as_slice(),
            CausalAttentionWeights::Cuda(_) => {
                return Err(TensorError::Autograd(
                    "CPU causal attention backward expected saved CPU attention".to_string(),
                ))
            }
            CausalAttentionWeights::CudaFlash { .. } => return Err(TensorError::Autograd(
                "CPU causal attention backward expected saved CPU attention, got CUDA flash state"
                    .to_string(),
            )),
        };
    let shape = query.shape();
    let (batch, time, channels) = (shape[0], shape[1], shape[2]);
    let head_dim = channels / n_heads;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let query_data = query.data_f64();
    let key_data = key.data_f64();
    let value_data = value.data_f64();

    let mut grad_query = query.requires_grad().then(|| vec![0.0; query.numel()]);
    let mut grad_key = key.requires_grad().then(|| vec![0.0; key.numel()]);
    let mut grad_value = value.requires_grad().then(|| vec![0.0; value.numel()]);

    for b in 0..batch {
        for h in 0..n_heads {
            for t in 0..time {
                let mut grad_attention = vec![0.0; t + 1];
                for s in 0..=t {
                    let mut acc = 0.0;
                    for d in 0..head_dim {
                        let out_index = qkv_index(b, t, h, d, time, channels, head_dim);
                        let value_index = qkv_index(b, s, h, d, time, channels, head_dim);
                        acc += grad_output[out_index] * value_data[value_index];
                        if let Some(grad_value) = &mut grad_value {
                            let attn = attention[attention_index(b, h, t, s, n_heads, time)];
                            grad_value[value_index] += attn * grad_output[out_index];
                        }
                    }
                    grad_attention[s] = acc;
                }

                let mut dot = 0.0;
                for s in 0..=t {
                    let attn = attention[attention_index(b, h, t, s, n_heads, time)];
                    dot += grad_attention[s] * attn;
                }

                for s in 0..=t {
                    let attn = attention[attention_index(b, h, t, s, n_heads, time)];
                    let grad_score = attn * (grad_attention[s] - dot);
                    for d in 0..head_dim {
                        let query_index = qkv_index(b, t, h, d, time, channels, head_dim);
                        let key_index = qkv_index(b, s, h, d, time, channels, head_dim);
                        if let Some(grad_query) = &mut grad_query {
                            grad_query[query_index] += grad_score * key_data[key_index] * scale;
                        }
                        if let Some(grad_key) = &mut grad_key {
                            grad_key[key_index] += grad_score * query_data[query_index] * scale;
                        }
                    }
                }
            }
        }
    }

    let mut grads = Vec::new();
    if let Some(grad_query) = grad_query {
        grads.push((query, grad_query));
    }
    if let Some(grad_key) = grad_key {
        grads.push((key, grad_key));
    }
    if let Some(grad_value) = grad_value {
        grads.push((value, grad_value));
    }
    Ok(grads)
}

fn backward_custom_unary(
    op: CustomUnaryOp,
    input: &SavedTensor,
    output_data: &[f64],
    output_shape: &[usize],
    grad_output: &[f64],
) -> Result<Vec<(Tensor, Vec<f64>)>> {
    let input = input.unpack()?;
    if !input.requires_grad() {
        return Ok(Vec::new());
    }
    if grad_output.len() != input.numel() {
        return Err(TensorError::Autograd(format!(
            "custom unary op {} received gradient length {} for input shape {:?}",
            op.name(),
            grad_output.len(),
            input.shape()
        )));
    }

    let input_data = input.data_f64();
    let grad_input = op.backward(
        &input_data,
        output_data,
        grad_output,
        output_shape,
        input.dtype(),
        input.device(),
    )?;
    if grad_input.len() != input.numel() {
        return Err(TensorError::Autograd(format!(
            "custom unary op {} returned gradient length {} for input shape {:?} with {} elements",
            op.name(),
            grad_input.len(),
            input.shape(),
            input.numel()
        )));
    }
    Ok(vec![(input, grad_input)])
}

fn backward_reduce_dim(
    input: &Tensor,
    dim: usize,
    keepdim: bool,
    kind: ReduceDimKind,
    grad_output: &[f64],
) -> Result<Vec<(Tensor, Vec<f64>)>> {
    if !input.requires_grad() {
        return Ok(Vec::new());
    }

    let input_shape = input.shape();
    let output_shape = reduced_shape(&input_shape, dim, keepdim);
    let scale = match kind {
        ReduceDimKind::Sum => 1.0,
        ReduceDimKind::Mean => 1.0 / input_shape[dim] as f64,
    };
    let mut grad_input = vec![0.0; input.numel()];
    let mut input_flat = 0;

    for_each_index(&input_shape, |input_index| {
        let output_index = reduced_index(input_index, dim, keepdim);
        let output_flat = flatten_index(&output_index, &output_shape);
        grad_input[input_flat] = grad_output[output_flat] * scale;
        input_flat += 1;
    });
    Ok(vec![(input.clone(), grad_input)])
}

fn backward_narrow(
    input: &Tensor,
    dim: usize,
    start: usize,
    grad_output: &[f64],
) -> Result<Vec<(Tensor, Vec<f64>)>> {
    if !input.requires_grad() {
        return Ok(Vec::new());
    }
    let input_shape = input.shape();
    let mut output_shape = input_shape.clone();
    let outer_numel = input_shape_without_dim_numel(&input_shape, dim)?;
    output_shape[dim] = if outer_numel == 0 {
        0
    } else {
        if !grad_output.len().is_multiple_of(outer_numel) {
            return Err(TensorError::Autograd(format!(
                "narrow backward gradient length {} is not divisible by outer element count {outer_numel}",
                grad_output.len()
            )));
        }
        grad_output.len() / outer_numel
    };
    let mut grad_input = vec![0.0; input.numel()];
    let mut output_flat = 0;

    for_each_index(&output_shape, |output_index| {
        let mut input_index = output_index.to_vec();
        input_index[dim] += start;
        grad_input[flatten_index(&input_index, &input_shape)] += grad_output[output_flat];
        output_flat += 1;
    });
    Ok(vec![(input.clone(), grad_input)])
}

fn input_shape_without_dim_numel(shape: &[usize], dim: usize) -> Result<usize> {
    let reduced_shape = shape
        .iter()
        .enumerate()
        .filter_map(|(index, size)| (index != dim).then_some(*size))
        .collect::<Vec<_>>();
    checked_numel(&reduced_shape).map_err(|error| {
        TensorError::Autograd(format!("narrow backward outer shape is invalid: {error}"))
    })
}

pub(super) fn reduced_shape(input_shape: &[usize], dim: usize, keepdim: bool) -> Vec<usize> {
    if keepdim {
        let mut output_shape = input_shape.to_vec();
        output_shape[dim] = 1;
        output_shape
    } else {
        input_shape
            .iter()
            .enumerate()
            .filter_map(|(index, size)| (index != dim).then_some(*size))
            .collect()
    }
}

pub(super) fn reduced_index(input_index: &[usize], dim: usize, keepdim: bool) -> Vec<usize> {
    if keepdim {
        let mut output_index = input_index.to_vec();
        output_index[dim] = 0;
        output_index
    } else {
        input_index
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (index != dim).then_some(*value))
            .collect()
    }
}

pub(super) fn softmax_data(input: &[f64], shape: &[usize], dim: usize) -> Vec<f64> {
    let mut output = vec![0.0; input.len()];
    let mut base_shape = shape.to_vec();
    base_shape[dim] = 1;

    for_each_index(&base_shape, |base_index| {
        let mut max = f64::NEG_INFINITY;
        for axis_index in 0..shape[dim] {
            let mut index = base_index.to_vec();
            index[dim] = axis_index;
            max = max.max(input[flatten_index(&index, shape)]);
        }

        let mut denom = 0.0;
        for axis_index in 0..shape[dim] {
            let mut index = base_index.to_vec();
            index[dim] = axis_index;
            let flat = flatten_index(&index, shape);
            let shifted_exp = (input[flat] - max).exp();
            output[flat] = shifted_exp;
            denom += shifted_exp;
        }

        for axis_index in 0..shape[dim] {
            let mut index = base_index.to_vec();
            index[dim] = axis_index;
            output[flatten_index(&index, shape)] /= denom;
        }
    });

    output
}

fn gelu_derivative(x: f64) -> f64 {
    let coeff = (2.0 / std::f64::consts::PI).sqrt();
    let inner = coeff * (x + 0.044_715 * x * x * x);
    let tanh_inner = inner.tanh();
    let sech2 = 1.0 - tanh_inner * tanh_inner;
    0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * coeff * (1.0 + 3.0 * 0.044_715 * x * x)
}

fn qkv_index(
    batch: usize,
    time_index: usize,
    head: usize,
    head_offset: usize,
    time: usize,
    channels: usize,
    head_dim: usize,
) -> usize {
    batch * time * channels + time_index * channels + head * head_dim + head_offset
}

fn attention_index(
    batch: usize,
    head: usize,
    query_time: usize,
    key_time: usize,
    n_heads: usize,
    time: usize,
) -> usize {
    ((batch * n_heads + head) * time + query_time) * time + key_time
}

pub(super) fn build_topo(
    tensor: &Tensor,
    seen: &mut HashSet<usize>,
    topo: &mut Vec<Tensor>,
) -> Result<()> {
    if !seen.insert(tensor.id()) {
        return Ok(());
    }
    if tensor.graph_released() {
        return Err(TensorError::Autograd(format!(
            "autograd graph for tensor {} has already been released; rerun the forward pass or call backward_retain_graph/backward_with_grad_retain_graph on the first backward",
            tensor.id()
        )));
    }
    if let Some(grad_fn) = tensor.grad_fn() {
        for parent in grad_fn.parents() {
            build_topo(&parent, seen, topo)?;
        }
    }
    topo.push(tensor.clone());
    Ok(())
}

fn unbroadcast_gradient_f64(
    grad_output: &[f64],
    output_shape: &[usize],
    input_shape: &[usize],
) -> Vec<f64> {
    let mut grad_input = vec![0.0; numel(input_shape)];
    let mut output_linear = 0;
    for_each_index(output_shape, |output_index| {
        let input_linear = broadcast_flat_index(output_index, output_shape, input_shape);
        grad_input[input_linear] += grad_output[output_linear];
        output_linear += 1;
    });
    grad_input
}

fn cuda_error(error: heirloom_kernels::cuda::CudaError) -> TensorError {
    TensorError::Device(error.to_string())
}
