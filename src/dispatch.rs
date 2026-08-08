use crate::shape::{broadcast_flat_index, broadcast_shapes, checked_numel, for_each_index, numel};
use crate::{DType, Device};
use crate::{Result, TensorError};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operator {
    Add,
    Sub,
    Mul,
    Div,
    Matmul,
    Relu,
    Gelu,
    SoftmaxDim,
    CrossEntropyForLogits,
    Sum,
    Mean,
    SumDim,
    MeanDim,
    LayerNormLastDim,
    Embedding,
    MaskedFill,
    ArgmaxLastDim,
    CausalSelfAttention,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperatorKind {
    Binary,
    Unary,
    MatrixMultiply,
    Loss,
    ReductionAll,
    ReductionDim,
    Normalization,
    Indexing,
    Masking,
    Selection,
    Attention,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DTypeRule {
    ArithmeticPromotion,
    FloatingPromotion,
    PreserveFloating,
    SumReduction,
    MeanReduction,
    EmbeddingLookup,
    MaskPreserve,
    ArgmaxIndex,
    SameFloatingInputs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AliasPolicy {
    FreshOutput,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AutogradPolicy {
    FloatingInputs,
    NonDifferentiable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OperatorInfo {
    pub operator: Operator,
    pub name: &'static str,
    pub kind: OperatorKind,
    pub dtype_rule: DTypeRule,
    pub alias_policy: AliasPolicy,
    pub autograd_policy: AutogradPolicy,
}

impl OperatorInfo {
    pub fn returns_fresh_output(&self) -> bool {
        matches!(self.alias_policy, AliasPolicy::FreshOutput)
    }

    pub fn should_record_autograd(&self, output_dtype: DType, input_requires_grad: bool) -> bool {
        match self.autograd_policy {
            AutogradPolicy::FloatingInputs => input_requires_grad && output_dtype.is_floating(),
            AutogradPolicy::NonDifferentiable => false,
        }
    }
}

pub fn operator_catalog() -> &'static [OperatorInfo] {
    OPERATOR_CATALOG
}

const OPERATOR_CATALOG: &[OperatorInfo] = &[
    OperatorInfo {
        operator: Operator::Add,
        name: "aten.add",
        kind: OperatorKind::Binary,
        dtype_rule: DTypeRule::ArithmeticPromotion,
        alias_policy: AliasPolicy::FreshOutput,
        autograd_policy: AutogradPolicy::FloatingInputs,
    },
    OperatorInfo {
        operator: Operator::Sub,
        name: "aten.sub",
        kind: OperatorKind::Binary,
        dtype_rule: DTypeRule::ArithmeticPromotion,
        alias_policy: AliasPolicy::FreshOutput,
        autograd_policy: AutogradPolicy::FloatingInputs,
    },
    OperatorInfo {
        operator: Operator::Mul,
        name: "aten.mul",
        kind: OperatorKind::Binary,
        dtype_rule: DTypeRule::ArithmeticPromotion,
        alias_policy: AliasPolicy::FreshOutput,
        autograd_policy: AutogradPolicy::FloatingInputs,
    },
    OperatorInfo {
        operator: Operator::Div,
        name: "aten.div",
        kind: OperatorKind::Binary,
        dtype_rule: DTypeRule::ArithmeticPromotion,
        alias_policy: AliasPolicy::FreshOutput,
        autograd_policy: AutogradPolicy::FloatingInputs,
    },
    OperatorInfo {
        operator: Operator::Matmul,
        name: "aten.matmul",
        kind: OperatorKind::MatrixMultiply,
        dtype_rule: DTypeRule::FloatingPromotion,
        alias_policy: AliasPolicy::FreshOutput,
        autograd_policy: AutogradPolicy::FloatingInputs,
    },
    OperatorInfo {
        operator: Operator::Relu,
        name: "aten.relu",
        kind: OperatorKind::Unary,
        dtype_rule: DTypeRule::PreserveFloating,
        alias_policy: AliasPolicy::FreshOutput,
        autograd_policy: AutogradPolicy::FloatingInputs,
    },
    OperatorInfo {
        operator: Operator::Gelu,
        name: "aten.gelu",
        kind: OperatorKind::Unary,
        dtype_rule: DTypeRule::PreserveFloating,
        alias_policy: AliasPolicy::FreshOutput,
        autograd_policy: AutogradPolicy::FloatingInputs,
    },
    OperatorInfo {
        operator: Operator::SoftmaxDim,
        name: "aten.softmax.dim",
        kind: OperatorKind::Unary,
        dtype_rule: DTypeRule::PreserveFloating,
        alias_policy: AliasPolicy::FreshOutput,
        autograd_policy: AutogradPolicy::FloatingInputs,
    },
    OperatorInfo {
        operator: Operator::CrossEntropyForLogits,
        name: "heirloom.cross_entropy_for_logits",
        kind: OperatorKind::Loss,
        dtype_rule: DTypeRule::PreserveFloating,
        alias_policy: AliasPolicy::FreshOutput,
        autograd_policy: AutogradPolicy::FloatingInputs,
    },
    OperatorInfo {
        operator: Operator::Sum,
        name: "aten.sum",
        kind: OperatorKind::ReductionAll,
        dtype_rule: DTypeRule::SumReduction,
        alias_policy: AliasPolicy::FreshOutput,
        autograd_policy: AutogradPolicy::FloatingInputs,
    },
    OperatorInfo {
        operator: Operator::Mean,
        name: "aten.mean",
        kind: OperatorKind::ReductionAll,
        dtype_rule: DTypeRule::MeanReduction,
        alias_policy: AliasPolicy::FreshOutput,
        autograd_policy: AutogradPolicy::FloatingInputs,
    },
    OperatorInfo {
        operator: Operator::SumDim,
        name: "aten.sum.dim",
        kind: OperatorKind::ReductionDim,
        dtype_rule: DTypeRule::SumReduction,
        alias_policy: AliasPolicy::FreshOutput,
        autograd_policy: AutogradPolicy::FloatingInputs,
    },
    OperatorInfo {
        operator: Operator::MeanDim,
        name: "aten.mean.dim",
        kind: OperatorKind::ReductionDim,
        dtype_rule: DTypeRule::MeanReduction,
        alias_policy: AliasPolicy::FreshOutput,
        autograd_policy: AutogradPolicy::FloatingInputs,
    },
    OperatorInfo {
        operator: Operator::LayerNormLastDim,
        name: "aten.layer_norm.last_dim",
        kind: OperatorKind::Normalization,
        dtype_rule: DTypeRule::SameFloatingInputs,
        alias_policy: AliasPolicy::FreshOutput,
        autograd_policy: AutogradPolicy::FloatingInputs,
    },
    OperatorInfo {
        operator: Operator::Embedding,
        name: "aten.embedding",
        kind: OperatorKind::Indexing,
        dtype_rule: DTypeRule::EmbeddingLookup,
        alias_policy: AliasPolicy::FreshOutput,
        autograd_policy: AutogradPolicy::FloatingInputs,
    },
    OperatorInfo {
        operator: Operator::MaskedFill,
        name: "aten.masked_fill",
        kind: OperatorKind::Masking,
        dtype_rule: DTypeRule::MaskPreserve,
        alias_policy: AliasPolicy::FreshOutput,
        autograd_policy: AutogradPolicy::FloatingInputs,
    },
    OperatorInfo {
        operator: Operator::ArgmaxLastDim,
        name: "aten.argmax.last_dim",
        kind: OperatorKind::Selection,
        dtype_rule: DTypeRule::ArgmaxIndex,
        alias_policy: AliasPolicy::FreshOutput,
        autograd_policy: AutogradPolicy::NonDifferentiable,
    },
    OperatorInfo {
        operator: Operator::CausalSelfAttention,
        name: "heirloom.causal_self_attention",
        kind: OperatorKind::Attention,
        dtype_rule: DTypeRule::SameFloatingInputs,
        alias_policy: AliasPolicy::FreshOutput,
        autograd_policy: AutogradPolicy::FloatingInputs,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinaryOp {
    pub(crate) fn operator(self) -> Operator {
        match self {
            Self::Add => Operator::Add,
            Self::Sub => Operator::Sub,
            Self::Mul => Operator::Mul,
            Self::Div => Operator::Div,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TensorMeta {
    pub(crate) dtype: DType,
    pub(crate) device: Device,
}

impl TensorMeta {
    pub(crate) fn new(dtype: DType, device: Device) -> Self {
        Self { dtype, device }
    }
}

pub(crate) struct DispatchOutput {
    pub(crate) data: Vec<f64>,
    pub(crate) shape: Vec<usize>,
}

type BinaryKernel = fn(BinaryOp, &[f64], &[usize], &[f64], &[usize]) -> Result<DispatchOutput>;
type MatmulKernel = fn(&[f64], &[usize], &[f64], &[usize]) -> Result<DispatchOutput>;
type UnaryKernel = fn(&[f64], &[usize]) -> DispatchOutput;
type ReductionKernel = fn(&[f64]) -> Result<DispatchOutput>;

#[derive(Clone, Copy)]
pub(crate) struct ResolvedBinaryOp {
    pub(crate) schema: &'static OperatorInfo,
    pub(crate) output_dtype: DType,
    pub(crate) kernel: BinaryKernel,
}

#[derive(Clone, Copy)]
pub(crate) struct ResolvedMatmulOp {
    pub(crate) schema: &'static OperatorInfo,
    pub(crate) output_dtype: DType,
    pub(crate) kernel: MatmulKernel,
}

#[derive(Clone, Copy)]
pub(crate) struct ResolvedUnaryOp {
    pub(crate) schema: &'static OperatorInfo,
    pub(crate) output_dtype: DType,
    pub(crate) kernel: Option<UnaryKernel>,
}

#[derive(Clone, Copy)]
pub(crate) struct ResolvedReductionOp {
    pub(crate) schema: &'static OperatorInfo,
    pub(crate) output_dtype: DType,
    pub(crate) kernel: Option<ReductionKernel>,
}

#[derive(Clone, Copy)]
pub(crate) struct ResolvedTypedOp {
    pub(crate) schema: &'static OperatorInfo,
    pub(crate) output_dtype: DType,
}

#[derive(Debug)]
pub(crate) struct KernelRegistry {
    schemas: &'static [OperatorInfo],
}

impl KernelRegistry {
    pub(crate) fn builtin() -> &'static Self {
        &BUILTIN_REGISTRY
    }

    pub(crate) fn schema(&self, operator: Operator) -> &'static OperatorInfo {
        self.schemas
            .iter()
            .find(|schema| schema.operator == operator)
            .expect("all builtin operators must have a schema")
    }

    pub(crate) fn resolve_binary(
        &self,
        op: BinaryOp,
        left: TensorMeta,
        right: TensorMeta,
    ) -> Result<ResolvedBinaryOp> {
        ensure_same_device(left, right)?;
        ensure_cpu_kernel_device(left.device, self.schema(op.operator()).name)?;
        ensure_arithmetic_dtype(left.dtype, op)?;
        ensure_arithmetic_dtype(right.dtype, op)?;
        Ok(ResolvedBinaryOp {
            schema: self.schema(op.operator()),
            output_dtype: promote_binary_dtype(left.dtype, right.dtype, op),
            kernel: binary_cpu,
        })
    }

    pub(crate) fn resolve_matmul(
        &self,
        left: TensorMeta,
        right: TensorMeta,
    ) -> Result<ResolvedMatmulOp> {
        ensure_same_device(left, right)?;
        ensure_cpu_kernel_device(left.device, "aten.matmul")?;
        ensure_floating_dtype(left.dtype, "matmul")?;
        ensure_floating_dtype(right.dtype, "matmul")?;
        Ok(ResolvedMatmulOp {
            schema: self.schema(Operator::Matmul),
            output_dtype: promote_floating_dtype(left.dtype, right.dtype),
            kernel: matmul_cpu,
        })
    }

    pub(crate) fn resolve_unary(
        &self,
        operator: Operator,
        input: TensorMeta,
    ) -> Result<ResolvedUnaryOp> {
        let schema = self.schema(operator);
        ensure_cpu_kernel_device(input.device, schema.name)?;
        match schema.dtype_rule {
            DTypeRule::PreserveFloating => {
                ensure_floating_dtype(input.dtype, schema.name)?;
                Ok(ResolvedUnaryOp {
                    schema,
                    output_dtype: input.dtype,
                    kernel: unary_kernel(operator),
                })
            }
            other => Err(TensorError::DType(format!(
                "{} cannot be resolved as a unary op with dtype rule {:?}",
                schema.name, other
            ))),
        }
    }

    pub(crate) fn resolve_reduction(
        &self,
        operator: Operator,
        input: TensorMeta,
    ) -> Result<ResolvedReductionOp> {
        let schema = self.schema(operator);
        ensure_cpu_kernel_device(input.device, schema.name)?;
        let output_dtype = match schema.dtype_rule {
            DTypeRule::SumReduction => sum_output_dtype(input.dtype),
            DTypeRule::MeanReduction => mean_output_dtype(input.dtype),
            other => {
                return Err(TensorError::DType(format!(
                    "{} cannot be resolved as a reduction with dtype rule {:?}",
                    schema.name, other
                )))
            }
        };
        Ok(ResolvedReductionOp {
            schema,
            output_dtype,
            kernel: reduction_kernel(operator),
        })
    }

    pub(crate) fn resolve_layer_norm_last_dim(
        &self,
        input: TensorMeta,
        weight: TensorMeta,
        bias: TensorMeta,
    ) -> Result<ResolvedTypedOp> {
        let schema = self.schema(Operator::LayerNormLastDim);
        ensure_same_floating_inputs(schema, &[input, weight, bias])?;
        ensure_cpu_kernel_device(input.device, schema.name)?;
        Ok(ResolvedTypedOp {
            schema,
            output_dtype: input.dtype,
        })
    }

    pub(crate) fn resolve_embedding(
        &self,
        indices: TensorMeta,
        weight: TensorMeta,
    ) -> Result<ResolvedTypedOp> {
        let schema = self.schema(Operator::Embedding);
        ensure_same_device(indices, weight)?;
        ensure_cpu_kernel_device(weight.device, schema.name)?;
        ensure_exact_dtype(indices.dtype, DType::I64, schema.name, "indices")?;
        ensure_floating_dtype(weight.dtype, "embedding weight")?;
        Ok(ResolvedTypedOp {
            schema,
            output_dtype: weight.dtype,
        })
    }

    pub(crate) fn resolve_masked_fill(
        &self,
        input: TensorMeta,
        mask: TensorMeta,
    ) -> Result<ResolvedTypedOp> {
        let schema = self.schema(Operator::MaskedFill);
        ensure_same_device(input, mask)?;
        ensure_cpu_kernel_device(input.device, schema.name)?;
        ensure_exact_dtype(mask.dtype, DType::Bool, schema.name, "mask")?;
        Ok(ResolvedTypedOp {
            schema,
            output_dtype: input.dtype,
        })
    }

    pub(crate) fn resolve_argmax_last_dim(&self, input: TensorMeta) -> Result<ResolvedTypedOp> {
        let schema = self.schema(Operator::ArgmaxLastDim);
        ensure_cpu_kernel_device(input.device, schema.name)?;
        ensure_orderable_dtype(input.dtype, schema.name)?;
        Ok(ResolvedTypedOp {
            schema,
            output_dtype: DType::I64,
        })
    }

    pub(crate) fn resolve_causal_self_attention(
        &self,
        query: TensorMeta,
        key: TensorMeta,
        value: TensorMeta,
    ) -> Result<ResolvedTypedOp> {
        let schema = self.schema(Operator::CausalSelfAttention);
        ensure_same_floating_inputs(schema, &[query, key, value])?;
        ensure_cpu_kernel_device(query.device, schema.name)?;
        Ok(ResolvedTypedOp {
            schema,
            output_dtype: query.dtype,
        })
    }
}

static BUILTIN_REGISTRY: KernelRegistry = KernelRegistry {
    schemas: OPERATOR_CATALOG,
};

pub(crate) fn resolve_binary(
    op: BinaryOp,
    left: TensorMeta,
    right: TensorMeta,
) -> Result<ResolvedBinaryOp> {
    KernelRegistry::builtin().resolve_binary(op, left, right)
}

pub(crate) fn resolve_matmul(left: TensorMeta, right: TensorMeta) -> Result<ResolvedMatmulOp> {
    KernelRegistry::builtin().resolve_matmul(left, right)
}

pub(crate) fn resolve_unary(operator: Operator, input: TensorMeta) -> Result<ResolvedUnaryOp> {
    KernelRegistry::builtin().resolve_unary(operator, input)
}

pub(crate) fn resolve_reduction(
    operator: Operator,
    input: TensorMeta,
) -> Result<ResolvedReductionOp> {
    KernelRegistry::builtin().resolve_reduction(operator, input)
}

pub(crate) fn resolve_layer_norm_last_dim(
    input: TensorMeta,
    weight: TensorMeta,
    bias: TensorMeta,
) -> Result<ResolvedTypedOp> {
    KernelRegistry::builtin().resolve_layer_norm_last_dim(input, weight, bias)
}

pub(crate) fn resolve_embedding(
    indices: TensorMeta,
    weight: TensorMeta,
) -> Result<ResolvedTypedOp> {
    KernelRegistry::builtin().resolve_embedding(indices, weight)
}

pub(crate) fn resolve_masked_fill(input: TensorMeta, mask: TensorMeta) -> Result<ResolvedTypedOp> {
    KernelRegistry::builtin().resolve_masked_fill(input, mask)
}

pub(crate) fn resolve_argmax_last_dim(input: TensorMeta) -> Result<ResolvedTypedOp> {
    KernelRegistry::builtin().resolve_argmax_last_dim(input)
}

pub(crate) fn resolve_causal_self_attention(
    query: TensorMeta,
    key: TensorMeta,
    value: TensorMeta,
) -> Result<ResolvedTypedOp> {
    KernelRegistry::builtin().resolve_causal_self_attention(query, key, value)
}

fn binary_cpu(
    op: BinaryOp,
    left_data: &[f64],
    left_shape: &[usize],
    right_data: &[f64],
    right_shape: &[usize],
) -> Result<DispatchOutput> {
    let output_shape = broadcast_shapes(left_shape, right_shape)?;
    let mut out = Vec::with_capacity(numel(&output_shape));

    for_each_index(&output_shape, |index| {
        let left_index = broadcast_flat_index(index, &output_shape, left_shape);
        let right_index = broadcast_flat_index(index, &output_shape, right_shape);
        let left = left_data[left_index];
        let right = right_data[right_index];
        out.push(match op {
            BinaryOp::Add => left + right,
            BinaryOp::Sub => left - right,
            BinaryOp::Mul => left * right,
            BinaryOp::Div => left / right,
        });
    });

    Ok(DispatchOutput {
        data: out,
        shape: output_shape,
    })
}

fn matmul_cpu(
    left_data: &[f64],
    left_shape: &[usize],
    right_data: &[f64],
    right_shape: &[usize],
) -> Result<DispatchOutput> {
    if left_shape.len() < 2 || right_shape.len() < 2 {
        return Err(TensorError::Shape(format!(
            "matmul requires tensors with rank >= 2, got {:?} and {:?}",
            left_shape, right_shape
        )));
    }
    let left_batch_shape = &left_shape[..left_shape.len() - 2];
    let right_batch_shape = &right_shape[..right_shape.len() - 2];
    let batch_shape = broadcast_shapes(left_batch_shape, right_batch_shape)?;

    let (m, k) = (
        left_shape[left_shape.len() - 2],
        left_shape[left_shape.len() - 1],
    );
    let (k_right, n) = (
        right_shape[right_shape.len() - 2],
        right_shape[right_shape.len() - 1],
    );
    if k != k_right {
        return Err(TensorError::Shape(format!(
            "matmul inner dimensions must match, got {:?} and {:?}",
            left_shape, right_shape
        )));
    }

    let batch_count = checked_numel(&batch_shape)?;
    let left_matrix_len = m.checked_mul(k).ok_or_else(|| {
        TensorError::Shape(format!("matmul left matrix size overflows: {m} * {k}"))
    })?;
    let right_matrix_len = k.checked_mul(n).ok_or_else(|| {
        TensorError::Shape(format!("matmul right matrix size overflows: {k} * {n}"))
    })?;
    let output_matrix_len = m.checked_mul(n).ok_or_else(|| {
        TensorError::Shape(format!("matmul output matrix size overflows: {m} * {n}"))
    })?;
    let output_len = batch_count.checked_mul(output_matrix_len).ok_or_else(|| {
        TensorError::Shape(format!(
            "matmul batched output size overflows: {batch_count} * {output_matrix_len}"
        ))
    })?;
    let mut left_batches = Vec::with_capacity(batch_count);
    let mut right_batches = Vec::with_capacity(batch_count);
    let mut batch_indices = Vec::with_capacity(batch_count);
    for_each_index(&batch_shape, |batch_index| {
        batch_indices.push(batch_index.to_vec());
    });
    for batch_index in &batch_indices {
        let left_batch = broadcast_flat_index(batch_index, &batch_shape, left_batch_shape);
        let right_batch = broadcast_flat_index(batch_index, &batch_shape, right_batch_shape);
        let left_start = left_batch.checked_mul(left_matrix_len).ok_or_else(|| {
            TensorError::Shape("matmul left batch offset overflows usize".to_string())
        })?;
        let right_start = right_batch.checked_mul(right_matrix_len).ok_or_else(|| {
            TensorError::Shape("matmul right batch offset overflows usize".to_string())
        })?;
        let left_end = left_start.checked_add(left_matrix_len).ok_or_else(|| {
            TensorError::Shape("matmul left batch range overflows usize".to_string())
        })?;
        let right_end = right_start.checked_add(right_matrix_len).ok_or_else(|| {
            TensorError::Shape("matmul right batch range overflows usize".to_string())
        })?;
        left_batches.push(
            left_data
                .get(left_start..left_end)
                .ok_or_else(|| {
                    TensorError::Shape(format!(
                        "matmul left batch range {left_start}..{left_end} exceeds data length {}",
                        left_data.len()
                    ))
                })?
                .to_vec(),
        );
        right_batches.push(
            right_data
                .get(right_start..right_end)
                .ok_or_else(|| {
                    TensorError::Shape(format!(
                        "matmul right batch range {right_start}..{right_end} exceeds data length {}",
                        right_data.len()
                    ))
                })?
                .to_vec(),
        );
    }
    let mut out = Vec::new();
    out.try_reserve_exact(output_len).map_err(|error| {
        TensorError::Shape(format!(
            "matmul output allocation for {output_len} elements failed: {error}"
        ))
    })?;
    let batches = heirloom_kernels::batched_matmul_f64(&left_batches, &right_batches, m, k, n)
        .map_err(|error| TensorError::InvalidOperation(format!("CPU matmul failed: {error}")))?;
    for batch in batches {
        out.extend(batch);
    }

    let mut shape = batch_shape;
    shape.push(m);
    shape.push(n);
    Ok(DispatchOutput { data: out, shape })
}

fn relu_cpu(input: &[f64], shape: &[usize]) -> DispatchOutput {
    DispatchOutput {
        data: input
            .iter()
            .map(|value| if *value > 0.0 { *value } else { 0.0 })
            .collect(),
        shape: shape.to_vec(),
    }
}

fn sum_cpu(input: &[f64]) -> Result<DispatchOutput> {
    Ok(DispatchOutput {
        data: vec![input.iter().sum::<f64>()],
        shape: Vec::new(),
    })
}

fn mean_cpu(input: &[f64]) -> Result<DispatchOutput> {
    if input.is_empty() {
        return Err(TensorError::InvalidOperation(
            "mean of an empty tensor is undefined in this prototype".to_string(),
        ));
    }
    Ok(DispatchOutput {
        data: vec![input.iter().sum::<f64>() / input.len() as f64],
        shape: Vec::new(),
    })
}

fn unary_kernel(operator: Operator) -> Option<UnaryKernel> {
    match operator {
        Operator::Relu => Some(relu_cpu),
        Operator::Gelu | Operator::SoftmaxDim | Operator::CrossEntropyForLogits => None,
        _ => None,
    }
}

fn reduction_kernel(operator: Operator) -> Option<ReductionKernel> {
    match operator {
        Operator::Sum => Some(sum_cpu),
        Operator::Mean => Some(mean_cpu),
        Operator::SumDim | Operator::MeanDim => None,
        _ => None,
    }
}

fn ensure_same_device(left: TensorMeta, right: TensorMeta) -> Result<()> {
    if left.device != right.device {
        return Err(TensorError::Device(format!(
            "expected matching devices, got {:?} and {:?}",
            left.device, right.device
        )));
    }
    Ok(())
}

fn ensure_cpu_kernel_device(device: Device, op_name: &str) -> Result<()> {
    if device != Device::Cpu {
        return Err(TensorError::Device(format!(
            "{op_name} has no registered CUDA kernel yet for device {:?}; copy explicitly to CPU or implement the CUDA kernel path",
            device
        )));
    }
    Ok(())
}

fn ensure_same_dtype(left: TensorMeta, right: TensorMeta, op_name: &str) -> Result<()> {
    if left.dtype != right.dtype {
        return Err(TensorError::DType(format!(
            "{op_name} expected matching dtypes, got {:?} and {:?}",
            left.dtype, right.dtype
        )));
    }
    Ok(())
}

fn ensure_same_floating_inputs(schema: &OperatorInfo, inputs: &[TensorMeta]) -> Result<()> {
    let Some((&first, rest)) = inputs.split_first() else {
        return Err(TensorError::InvalidOperation(format!(
            "{} requires at least one input",
            schema.name
        )));
    };
    ensure_floating_dtype(first.dtype, schema.name)?;
    for input in rest {
        ensure_same_device(first, *input)?;
        ensure_same_dtype(first, *input, schema.name)?;
        ensure_floating_dtype(input.dtype, schema.name)?;
    }
    Ok(())
}

fn ensure_exact_dtype(actual: DType, expected: DType, op_name: &str, role: &str) -> Result<()> {
    if actual != expected {
        return Err(TensorError::DType(format!(
            "{op_name} {role} expected dtype {:?}, got {:?}",
            expected, actual
        )));
    }
    Ok(())
}

fn ensure_orderable_dtype(dtype: DType, _op_name: &str) -> Result<()> {
    match dtype {
        DType::F32 | DType::BFloat16 | DType::F64 | DType::I64 | DType::Bool => Ok(()),
    }
}

fn ensure_floating_dtype(dtype: DType, op_name: &str) -> Result<()> {
    if !dtype.is_floating() {
        return Err(TensorError::DType(format!(
            "{op_name} requires floating tensors, got {dtype:?}"
        )));
    }
    Ok(())
}

fn ensure_arithmetic_dtype(dtype: DType, op: BinaryOp) -> Result<()> {
    if matches!(dtype, DType::Bool) && matches!(op, BinaryOp::Sub | BinaryOp::Div) {
        return Err(TensorError::DType(format!(
            "{} does not support {:?} for bool tensors",
            KernelRegistry::builtin().schema(op.operator()).name,
            op
        )));
    }
    Ok(())
}

fn promote_floating_dtype(left: DType, right: DType) -> DType {
    if matches!(left, DType::F64) || matches!(right, DType::F64) {
        DType::F64
    } else if matches!(left, DType::F32) || matches!(right, DType::F32) {
        DType::F32
    } else {
        DType::BFloat16
    }
}

fn promote_binary_dtype(left: DType, right: DType, op: BinaryOp) -> DType {
    if matches!(op, BinaryOp::Div) {
        if left.is_floating() || right.is_floating() {
            return promote_floating_dtype(left, right);
        }
        return DType::F64;
    }
    if left.is_floating() || right.is_floating() {
        return promote_floating_dtype(left, right);
    }
    DType::I64
}

fn sum_output_dtype(dtype: DType) -> DType {
    match dtype {
        DType::F32 => DType::F32,
        DType::BFloat16 => DType::BFloat16,
        DType::F64 => DType::F64,
        DType::I64 | DType::Bool => DType::I64,
    }
}

fn mean_output_dtype(dtype: DType) -> DType {
    match dtype {
        DType::F32 => DType::F32,
        DType::BFloat16 => DType::BFloat16,
        DType::F64 => DType::F64,
        DType::I64 | DType::Bool => DType::F64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn operator_catalog_has_unique_names_and_operators() {
        let mut names = HashSet::new();
        let mut operators = HashSet::new();
        for schema in operator_catalog() {
            assert!(names.insert(schema.name), "duplicate name {}", schema.name);
            assert!(
                operators.insert(schema.operator),
                "duplicate operator {:?}",
                schema.operator
            );
        }
        assert_eq!(operator_catalog().len(), 18);
    }

    #[test]
    fn binary_resolution_promotes_and_rejects_bool_edges() {
        let cpu_f32 = TensorMeta::new(DType::F32, Device::Cpu);
        let cpu_f64 = TensorMeta::new(DType::F64, Device::Cpu);
        let cpu_i64 = TensorMeta::new(DType::I64, Device::Cpu);
        let cpu_bool = TensorMeta::new(DType::Bool, Device::Cpu);

        assert_eq!(
            resolve_binary(BinaryOp::Add, cpu_f32, cpu_f64)
                .unwrap()
                .output_dtype,
            DType::F64
        );
        assert_eq!(
            resolve_binary(BinaryOp::Mul, cpu_i64, cpu_i64)
                .unwrap()
                .output_dtype,
            DType::I64
        );
        assert_eq!(
            resolve_binary(BinaryOp::Div, cpu_i64, cpu_i64)
                .unwrap()
                .output_dtype,
            DType::F64
        );
        assert!(resolve_binary(BinaryOp::Div, cpu_bool, cpu_bool).is_err());
    }

    #[test]
    fn floating_only_resolution_rejects_integer_matmul_and_relu() {
        let cpu_i64 = TensorMeta::new(DType::I64, Device::Cpu);
        assert!(resolve_matmul(cpu_i64, cpu_i64).is_err());
        assert!(resolve_unary(Operator::Relu, cpu_i64).is_err());
        assert!(resolve_unary(Operator::Gelu, cpu_i64).is_err());
    }

    #[test]
    fn reduction_resolution_matches_prototype_dtype_rules() {
        let cpu_bool = TensorMeta::new(DType::Bool, Device::Cpu);
        assert_eq!(
            resolve_reduction(Operator::Sum, cpu_bool)
                .unwrap()
                .output_dtype,
            DType::I64
        );
        assert_eq!(
            resolve_reduction(Operator::Mean, cpu_bool)
                .unwrap()
                .output_dtype,
            DType::F64
        );
    }

    #[test]
    fn layer_norm_resolution_requires_same_floating_dtype() {
        let cpu_f32 = TensorMeta::new(DType::F32, Device::Cpu);
        let cpu_f64 = TensorMeta::new(DType::F64, Device::Cpu);
        let cpu_i64 = TensorMeta::new(DType::I64, Device::Cpu);

        let resolved = resolve_layer_norm_last_dim(cpu_f32, cpu_f32, cpu_f32).unwrap();
        assert_eq!(resolved.schema.operator, Operator::LayerNormLastDim);
        assert_eq!(resolved.output_dtype, DType::F32);
        assert!(resolve_layer_norm_last_dim(cpu_f32, cpu_f64, cpu_f32).is_err());
        assert!(resolve_layer_norm_last_dim(cpu_i64, cpu_i64, cpu_i64).is_err());
    }

    #[test]
    fn embedding_resolution_separates_index_and_weight_dtype() {
        let cpu_f32 = TensorMeta::new(DType::F32, Device::Cpu);
        let cpu_i64 = TensorMeta::new(DType::I64, Device::Cpu);

        let resolved = resolve_embedding(cpu_i64, cpu_f32).unwrap();
        assert_eq!(resolved.schema.operator, Operator::Embedding);
        assert_eq!(resolved.output_dtype, DType::F32);
        assert!(resolve_embedding(cpu_f32, cpu_f32).is_err());
        assert!(resolve_embedding(cpu_i64, cpu_i64).is_err());
    }

    #[test]
    fn masked_fill_resolution_preserves_input_dtype_and_requires_bool_mask() {
        let cpu_f32 = TensorMeta::new(DType::F32, Device::Cpu);
        let cpu_i64 = TensorMeta::new(DType::I64, Device::Cpu);
        let cpu_bool = TensorMeta::new(DType::Bool, Device::Cpu);

        let resolved = resolve_masked_fill(cpu_i64, cpu_bool).unwrap();
        assert_eq!(resolved.schema.operator, Operator::MaskedFill);
        assert_eq!(resolved.output_dtype, DType::I64);
        assert!(resolve_masked_fill(cpu_f32, cpu_i64).is_err());
    }

    #[test]
    fn argmax_resolution_returns_indices_and_disables_autograd() {
        let cpu_f32 = TensorMeta::new(DType::F32, Device::Cpu);

        let resolved = resolve_argmax_last_dim(cpu_f32).unwrap();
        assert_eq!(resolved.schema.operator, Operator::ArgmaxLastDim);
        assert_eq!(resolved.output_dtype, DType::I64);
        assert!(!resolved.schema.should_record_autograd(DType::I64, true));
    }

    #[test]
    fn causal_attention_resolution_requires_same_floating_dtype() {
        let cpu_f32 = TensorMeta::new(DType::F32, Device::Cpu);
        let cpu_f64 = TensorMeta::new(DType::F64, Device::Cpu);
        let cpu_i64 = TensorMeta::new(DType::I64, Device::Cpu);

        let resolved = resolve_causal_self_attention(cpu_f32, cpu_f32, cpu_f32).unwrap();
        assert_eq!(resolved.schema.operator, Operator::CausalSelfAttention);
        assert_eq!(resolved.output_dtype, DType::F32);
        assert!(resolve_causal_self_attention(cpu_f32, cpu_f64, cpu_f32).is_err());
        assert!(resolve_causal_self_attention(cpu_i64, cpu_i64, cpu_i64).is_err());
    }
}
