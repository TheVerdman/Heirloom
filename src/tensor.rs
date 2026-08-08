//! Tensor metadata, storage views, operations, and reverse-mode autograd entry
//! points.

use crate::dispatch::{self, BinaryOp, Operator, OperatorInfo, TensorMeta};
use crate::extension::CustomUnaryOp;
use crate::grad_mode::should_track_grad;
use crate::shape::{
    broadcast_flat_index, broadcast_shapes, checked_max_storage_offset, checked_numel,
    contiguous_strides, flatten_index, for_each_index, has_internal_overlap, is_contiguous,
    logical_offset, normalize_dim, numel, validate_permutation,
};
use crate::storage::{bf16_bits_to_f32, f32_to_bf16_bits, DType, Device, Storage, StorageData};
use crate::{Result, TensorError};
use heirloom_kernels::cuda::{CudaBuffer, NcclAllReduceStats, NcclCommunicator};
use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

mod autograd;

use autograd::{
    build_topo, reduced_index, reduced_shape, softmax_data, CausalAttentionWeights,
    CrossEntropyTargets, EmbeddingIndices, GradFn, ReduceDimKind, SavedTensor,
};

static NEXT_TENSOR_ID: AtomicUsize = AtomicUsize::new(1);

/// A reference-counted n-dimensional array with explicit dtype, device, layout,
/// and autograd metadata.
///
/// Cloning a tensor clones the handle, not its storage. View operations may
/// share storage with different shapes, strides, and offsets. Mutating
/// operations therefore perform alias/version checks, while shape and stride
/// arithmetic is validated before dispatch. Floating-point tensors can opt
/// into reverse-mode autograd with the constructor's `requires_grad` argument.
#[derive(Clone)]
pub struct Tensor {
    inner: Rc<RefCell<TensorInner>>,
}

struct TensorInner {
    id: usize,
    storage: Rc<Storage>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    storage_offset: usize,
    dtype: DType,
    device: Device,
    requires_grad: bool,
    grad: Option<GradTensor>,
    grad_fn: Option<GradFn>,
    grad_fn_released: bool,
    retain_grad: bool,
    grad_hooks: Vec<GradHookEntry>,
    next_grad_hook_id: usize,
}

type GradHook = Rc<dyn Fn(&[f64]) -> Result<Vec<f64>>>;

#[derive(Clone)]
struct GradHookEntry {
    id: usize,
    hook: GradHook,
}

#[derive(Clone)]
struct GradTensor {
    storage: Rc<Storage>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    storage_offset: usize,
    dtype: DType,
    device: Device,
}

impl GradTensor {
    fn zeros_for_layout(shape: &[usize], strides: &[usize], storage_offset: usize) -> Result<Self> {
        let storage_len = if numel(shape) == 0 {
            0
        } else {
            checked_max_storage_offset(shape, strides, storage_offset)?
                .checked_add(1)
                .ok_or_else(|| {
                    TensorError::Autograd("gradient storage length overflows usize".to_string())
                })?
        };
        Ok(Self {
            storage: Storage::new_f64(vec![0.0; storage_len]),
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            storage_offset,
            dtype: DType::F64,
            device: Device::Cpu,
        })
    }

    fn zeros_for_tensor_layout(
        shape: &[usize],
        strides: &[usize],
        storage_offset: usize,
    ) -> Result<Self> {
        if has_internal_overlap(shape, strides) {
            Self::zeros_for_layout(shape, &contiguous_strides(shape), 0)
        } else {
            Self::zeros_for_layout(shape, strides, storage_offset)
        }
    }

    fn from_cuda_f32_full(
        shape: &[usize],
        _strides: &[usize],
        _storage_offset: usize,
        device: Device,
        buffer: CudaBuffer,
    ) -> Result<Self> {
        let Device::Cuda(device_id) = device else {
            return Err(TensorError::Device(format!(
                "CUDA gradient expected CUDA device, got {device:?}"
            )));
        };
        if buffer.len() != numel(shape) {
            return Err(TensorError::Autograd(format!(
                "CUDA gradient accumulation expected dense logical gradient for shape {shape:?}, got buffer_len={}",
                buffer.len()
            )));
        }
        Ok(Self {
            storage: Storage::new_cuda_from_buffer(device_id, DType::F32, numel(shape), buffer)?,
            shape: shape.to_vec(),
            strides: contiguous_strides(shape),
            storage_offset: 0,
            dtype: DType::F32,
            device,
        })
    }

    fn add_logical(&self, grad: &[f64]) -> Result<()> {
        if grad.len() != numel(&self.shape) {
            return Err(TensorError::Autograd(format!(
                "gradient length {} does not match gradient tensor shape {:?}",
                grad.len(),
                self.shape
            )));
        }

        let mut grad_index = 0;
        self.storage.with_data_mut(|data| {
            let StorageData::F64(values) = data else {
                unreachable!("gradient tensors always use f64 storage");
            };
            for_each_index(&self.shape, |index| {
                let storage_index = logical_offset(index, &self.strides, self.storage_offset);
                values[storage_index] += grad[grad_index];
                grad_index += 1;
            });
        });
        Ok(())
    }

    fn add_cuda_f32_full(&mut self, incoming: &CudaBuffer) -> Result<()> {
        if self.dtype != DType::F32 || !matches!(self.device, Device::Cuda(_)) {
            return Err(TensorError::Autograd(format!(
                "expected CUDA f32 gradient tensor, got {:?} on {:?}",
                self.dtype, self.device
            )));
        }
        if !is_contiguous(&self.shape, &self.strides)
            || self.storage_offset != 0
            || self.storage.len() != numel(&self.shape)
            || incoming.len() != numel(&self.shape)
        {
            return Err(TensorError::Autograd(format!(
                "CUDA gradient accumulation currently requires contiguous full-storage gradients, shape={:?} strides={:?} offset={} storage_len={} incoming_len={}",
                self.shape,
                self.strides,
                self.storage_offset,
                self.storage.len(),
                incoming.len()
            )));
        }
        let existing = self.storage.cuda_buffer(DType::F32, self.device)?;
        let updated =
            heirloom_kernels::cuda::add_f32_buffers(&existing, incoming).map_err(cuda_error)?;
        let Device::Cuda(device_id) = self.device else {
            unreachable!("CUDA f32 gradient checked above");
        };
        self.storage =
            Storage::new_cuda_from_buffer(device_id, DType::F32, numel(&self.shape), updated)?;
        Ok(())
    }

    fn cuda_f32_full_storage_buffer(&self) -> Result<CudaBuffer> {
        if self.dtype != DType::F32 || !matches!(self.device, Device::Cuda(_)) {
            return Err(TensorError::Autograd(format!(
                "expected CUDA f32 gradient tensor, got {:?} on {:?}",
                self.dtype, self.device
            )));
        }
        if !is_contiguous(&self.shape, &self.strides)
            || self.storage_offset != 0
            || self.storage.len() != numel(&self.shape)
        {
            return Err(TensorError::Autograd(format!(
                "CUDA gradient read currently requires contiguous full-storage gradients, shape={:?} strides={:?} offset={} storage_len={}",
                self.shape,
                self.strides,
                self.storage_offset,
                self.storage.len()
            )));
        }
        self.storage.cuda_buffer(DType::F32, self.device)
    }

    fn logical_data_f64(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(numel(&self.shape));
        self.storage.with_data(|data| {
            for_each_index(&self.shape, |index| {
                out.push(data.value_as_f64(logical_offset(
                    index,
                    &self.strides,
                    self.storage_offset,
                )));
            });
        });
        out
    }

    fn as_tensor(&self) -> Tensor {
        Tensor::from_parts(
            Rc::clone(&self.storage),
            self.shape.clone(),
            self.strides.clone(),
            self.storage_offset,
            self.dtype,
            self.device,
            false,
            None,
        )
    }
}

impl Tensor {
    pub fn from_vec(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Result<Self> {
        Self::from_f32(data, shape, requires_grad)
    }

    pub fn from_f32(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Result<Self> {
        Self::from_storage(
            Storage::new_f32(data),
            shape,
            DType::F32,
            Device::Cpu,
            requires_grad,
            None,
        )
    }

    pub fn from_bf16_bits(data: Vec<u16>, shape: &[usize], requires_grad: bool) -> Result<Self> {
        Self::from_storage(
            Storage::new_bf16_bits(data),
            shape,
            DType::BFloat16,
            Device::Cpu,
            requires_grad,
            None,
        )
    }

    pub fn from_f64(data: Vec<f64>, shape: &[usize], requires_grad: bool) -> Result<Self> {
        Self::from_storage(
            Storage::new_f64(data),
            shape,
            DType::F64,
            Device::Cpu,
            requires_grad,
            None,
        )
    }

    pub fn from_i64(data: Vec<i64>, shape: &[usize], requires_grad: bool) -> Result<Self> {
        Self::from_storage(
            Storage::new_i64(data),
            shape,
            DType::I64,
            Device::Cpu,
            requires_grad,
            None,
        )
    }

    pub fn from_bool(data: Vec<bool>, shape: &[usize], requires_grad: bool) -> Result<Self> {
        Self::from_storage(
            Storage::new_bool(data),
            shape,
            DType::Bool,
            Device::Cpu,
            requires_grad,
            None,
        )
    }

    fn from_storage(
        storage: Rc<Storage>,
        shape: &[usize],
        dtype: DType,
        device: Device,
        requires_grad: bool,
        grad_fn: Option<GradFn>,
    ) -> Result<Self> {
        ensure_grad_dtype(dtype, requires_grad)?;
        let expected = checked_numel(shape)?;
        if expected != storage.len() {
            return Err(TensorError::Shape(format!(
                "data length {} does not match shape {:?} with {} elements",
                storage.len(),
                shape,
                expected
            )));
        }
        debug_assert_eq!(dtype, storage.dtype());
        debug_assert_eq!(device, storage.device());

        Ok(Self::from_parts(
            storage,
            shape.to_vec(),
            contiguous_strides(shape),
            0,
            dtype,
            device,
            requires_grad,
            grad_fn,
        ))
    }

    pub fn zeros(shape: &[usize], requires_grad: bool) -> Result<Self> {
        let len = checked_numel(shape)?;
        Self::from_vec(vec![0.0; len], shape, requires_grad)
    }

    pub fn ones(shape: &[usize], requires_grad: bool) -> Result<Self> {
        let len = checked_numel(shape)?;
        Self::from_vec(vec![1.0; len], shape, requires_grad)
    }

    pub fn zeros_with_dtype(shape: &[usize], dtype: DType, requires_grad: bool) -> Result<Self> {
        Self::zeros_on_device(shape, dtype, Device::Cpu, requires_grad)
    }

    pub fn ones_with_dtype(shape: &[usize], dtype: DType, requires_grad: bool) -> Result<Self> {
        Self::ones_on_device(shape, dtype, Device::Cpu, requires_grad)
    }

    pub fn zeros_on_device(
        shape: &[usize],
        dtype: DType,
        device: Device,
        requires_grad: bool,
    ) -> Result<Self> {
        let len = checked_numel(shape)?;
        let tensor = match dtype {
            DType::F32 => Self::from_f32(vec![0.0; len], shape, requires_grad),
            DType::BFloat16 => {
                Self::from_bf16_bits(vec![f32_to_bf16_bits(0.0); len], shape, requires_grad)
            }
            DType::F64 => Self::from_f64(vec![0.0; len], shape, requires_grad),
            DType::I64 => Self::from_i64(vec![0; len], shape, requires_grad),
            DType::Bool => Self::from_bool(vec![false; len], shape, requires_grad),
        }?;
        tensor.to_device(device)
    }

    pub fn cuda_f32_zeros_on_device(shape: &[usize], device_id: usize) -> Result<Self> {
        let len = checked_numel(shape)?;
        let device_ordinal = i32::try_from(device_id).map_err(|_| {
            TensorError::Device(format!(
                "CUDA device id {device_id} does not fit into a CUDA ordinal"
            ))
        })?;
        let buffer = heirloom_kernels::cuda::fill_constant_f32_buffer(device_ordinal, len, 0.0)
            .map_err(cuda_error)?;
        Self::from_storage(
            Storage::new_cuda_from_buffer(device_id, DType::F32, len, buffer)?,
            shape,
            DType::F32,
            Device::Cuda(device_id),
            false,
            None,
        )
    }

    pub fn ones_on_device(
        shape: &[usize],
        dtype: DType,
        device: Device,
        requires_grad: bool,
    ) -> Result<Self> {
        let len = checked_numel(shape)?;
        let tensor = match dtype {
            DType::F32 => Self::from_f32(vec![1.0; len], shape, requires_grad),
            DType::BFloat16 => {
                Self::from_bf16_bits(vec![f32_to_bf16_bits(1.0); len], shape, requires_grad)
            }
            DType::F64 => Self::from_f64(vec![1.0; len], shape, requires_grad),
            DType::I64 => Self::from_i64(vec![1; len], shape, requires_grad),
            DType::Bool => Self::from_bool(vec![true; len], shape, requires_grad),
        }?;
        tensor.to_device(device)
    }

    pub fn cuda_i64_arange(device_id: usize, len: usize) -> Result<Self> {
        let device_ordinal = i32::try_from(device_id).map_err(|_| {
            TensorError::Device(format!(
                "CUDA device id {device_id} does not fit into a CUDA ordinal"
            ))
        })?;
        let buffer =
            heirloom_kernels::cuda::i64_arange_buffer(device_ordinal, len).map_err(cuda_error)?;
        Ok(Self::from_parts(
            Storage::new_cuda_from_buffer(device_id, DType::I64, len, buffer)?,
            vec![len],
            contiguous_strides(&[len]),
            0,
            DType::I64,
            Device::Cuda(device_id),
            false,
            None,
        ))
    }

    pub fn scalar(value: f32, requires_grad: bool) -> Result<Self> {
        Self::from_vec(vec![value], &[], requires_grad)
    }

    pub fn scalar_f64(value: f64, requires_grad: bool) -> Result<Self> {
        Self::from_f64(vec![value], &[], requires_grad)
    }

    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        storage: Rc<Storage>,
        shape: Vec<usize>,
        strides: Vec<usize>,
        storage_offset: usize,
        dtype: DType,
        device: Device,
        requires_grad: bool,
        grad_fn: Option<GradFn>,
    ) -> Self {
        Self {
            inner: Rc::new(RefCell::new(TensorInner {
                id: NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed),
                storage,
                shape,
                strides,
                storage_offset,
                dtype,
                device,
                requires_grad,
                grad: None,
                grad_fn,
                grad_fn_released: false,
                retain_grad: false,
                grad_hooks: Vec::new(),
                next_grad_hook_id: 1,
            })),
        }
    }

    fn from_f64_values_with_dtype(
        data: Vec<f64>,
        shape: &[usize],
        dtype: DType,
        requires_grad: bool,
        grad_fn: Option<GradFn>,
    ) -> Result<Self> {
        Self::from_storage(
            storage_from_f64_values(data, dtype),
            shape,
            dtype,
            Device::Cpu,
            requires_grad,
            grad_fn,
        )
    }

    fn from_cpu_storage_like(
        storage: Rc<Storage>,
        shape: &[usize],
        dtype: DType,
        requires_grad: bool,
    ) -> Result<Self> {
        Self::from_storage(storage, shape, dtype, Device::Cpu, requires_grad, None)
    }

    pub fn id(&self) -> usize {
        self.inner.borrow().id
    }

    pub fn dtype(&self) -> DType {
        self.inner.borrow().dtype
    }

    pub fn device(&self) -> Device {
        self.inner.borrow().device
    }

    pub fn shape(&self) -> Vec<usize> {
        self.inner.borrow().shape.clone()
    }

    pub fn strides(&self) -> Vec<usize> {
        self.inner.borrow().strides.clone()
    }

    pub fn storage_offset(&self) -> usize {
        self.inner.borrow().storage_offset
    }

    pub fn storage_version(&self) -> usize {
        self.inner.borrow().storage.version()
    }

    pub fn ndim(&self) -> usize {
        self.inner.borrow().shape.len()
    }

    pub fn numel(&self) -> usize {
        numel(&self.inner.borrow().shape)
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.borrow().requires_grad
    }

    pub fn set_requires_grad(&self, requires_grad: bool) {
        self.try_set_requires_grad(requires_grad)
            .expect("set_requires_grad failed");
    }

    pub fn try_set_requires_grad(&self, requires_grad: bool) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        ensure_grad_dtype(inner.dtype, requires_grad)?;
        inner.requires_grad = requires_grad;
        if !requires_grad {
            inner.grad = None;
            inner.grad_fn = None;
            inner.grad_fn_released = false;
            inner.retain_grad = false;
        }
        Ok(())
    }

    pub fn is_leaf(&self) -> bool {
        let inner = self.inner.borrow();
        inner.requires_grad && inner.grad_fn.is_none() && !inner.grad_fn_released
    }

    pub fn is_contiguous(&self) -> bool {
        let inner = self.inner.borrow();
        is_contiguous(&inner.shape, &inner.strides)
    }

    pub fn has_internal_overlap(&self) -> bool {
        let inner = self.inner.borrow();
        has_internal_overlap(&inner.shape, &inner.strides)
    }

    pub fn data(&self) -> Vec<f32> {
        self.data_f64()
            .into_iter()
            .map(|value| value as f32)
            .collect()
    }

    pub fn data_f64(&self) -> Vec<f64> {
        self.try_data_f64()
            .expect("failed to materialize tensor data as f64")
    }

    fn try_data_f64(&self) -> Result<Vec<f64>> {
        let (storage, shape, strides, offset) = self.layout_snapshot();
        let base = storage.with_data(StorageData::to_f64_vec)?;
        let mut out = Vec::with_capacity(numel(&shape));
        for_each_index(&shape, |index| {
            out.push(base[logical_offset(index, &strides, offset)]);
        });
        Ok(out)
    }

    pub fn data_f32(&self) -> Result<Vec<f32>> {
        ensure_dtype(self.dtype(), DType::F32, "data_f32")?;
        let (storage, shape, strides, offset) = self.layout_snapshot();
        let base = storage.with_data(StorageData::to_f32_vec)?;
        let mut out = Vec::with_capacity(numel(&shape));
        for_each_index(&shape, |index| {
            out.push(base[logical_offset(index, &strides, offset)]);
        });
        Ok(out)
    }

    pub fn data_bf16_bits(&self) -> Result<Vec<u16>> {
        ensure_dtype(self.dtype(), DType::BFloat16, "data_bf16_bits")?;
        let (storage, shape, strides, offset) = self.layout_snapshot();
        let base = storage.with_data(StorageData::to_bf16_bits_vec)?;
        let mut out = Vec::with_capacity(numel(&shape));
        for_each_index(&shape, |index| {
            out.push(base[logical_offset(index, &strides, offset)]);
        });
        Ok(out)
    }

    pub fn data_f64_exact(&self) -> Result<Vec<f64>> {
        ensure_dtype(self.dtype(), DType::F64, "data_f64_exact")?;
        self.try_data_f64()
    }

    pub fn data_i64(&self) -> Result<Vec<i64>> {
        ensure_dtype(self.dtype(), DType::I64, "data_i64")?;
        let (storage, shape, strides, offset) = self.layout_snapshot();
        let base = storage.with_data(StorageData::to_i64_vec)?;
        let mut out = Vec::with_capacity(numel(&shape));
        for_each_index(&shape, |index| {
            out.push(base[logical_offset(index, &strides, offset)]);
        });
        Ok(out)
    }

    pub fn data_bool(&self) -> Result<Vec<bool>> {
        ensure_dtype(self.dtype(), DType::Bool, "data_bool")?;
        let (storage, shape, strides, offset) = self.layout_snapshot();
        let base = storage.with_data(StorageData::to_bool_vec)?;
        let mut out = Vec::with_capacity(numel(&shape));
        for_each_index(&shape, |index| {
            out.push(base[logical_offset(index, &strides, offset)]);
        });
        Ok(out)
    }

    pub fn topk_indices_dim1(&self, k: usize) -> Result<Self> {
        ensure_dtype(self.dtype(), DType::F32, "topk_indices_dim1")?;
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(TensorError::Shape(format!(
                "topk_indices_dim1 expected rank-2 tensor, got {shape:?}"
            )));
        }
        if k == 0 || k > shape[1] {
            return Err(TensorError::Shape(format!(
                "topk_indices_dim1 expected 0 < k <= {}, got {k}",
                shape[1]
            )));
        }
        match self.device() {
            Device::Cpu => self.topk_indices_dim1_cpu(k),
            Device::Cuda(device_id) => {
                let scores = self.cuda_f32_full_storage_buffer("topk_indices_dim1 scores")?;
                let output = heirloom_kernels::cuda::memory_topk_indices_f32(
                    &scores,
                    heirloom_kernels::cuda::MemoryTopkDims {
                        rows: shape[0],
                        cols: shape[1],
                        top_k: k,
                    },
                )
                .map_err(cuda_error)?;
                Self::from_storage(
                    Storage::new_cuda_from_buffer(device_id, DType::I64, shape[0] * k, output)?,
                    &[shape[0], k],
                    DType::I64,
                    Device::Cuda(device_id),
                    false,
                    None,
                )
            }
        }
    }

    fn topk_indices_dim1_cpu(&self, k: usize) -> Result<Self> {
        let shape = self.shape();
        let rows = shape[0];
        let cols = shape[1];
        let data = self.data_f32()?;
        let mut indices = Vec::with_capacity(rows * k);
        for row in 0..rows {
            let row_data = &data[row * cols..(row + 1) * cols];
            let mut scored = row_data
                .iter()
                .copied()
                .enumerate()
                .collect::<Vec<(usize, f32)>>();
            scored.sort_by(|(left_index, left_score), (right_index, right_score)| {
                right_score
                    .partial_cmp(left_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left_index.cmp(right_index))
            });
            indices.extend(scored.iter().take(k).map(|(index, _)| *index as i64));
        }
        Self::from_i64(indices, &[rows, k], false)
    }

    pub fn concat_i64_flat(tensors: &[Self]) -> Result<Self> {
        let Some(first) = tensors.first() else {
            return Err(TensorError::InvalidOperation(
                "concat_i64_flat requires at least one tensor".to_string(),
            ));
        };
        let device = first.device();
        let mut total = 0usize;
        for (index, tensor) in tensors.iter().enumerate() {
            if tensor.dtype() != DType::I64 {
                return Err(TensorError::DType(format!(
                    "concat_i64_flat expected i64 tensors, got tensor[{index}] {:?}",
                    tensor.dtype()
                )));
            }
            if tensor.device() != device {
                return Err(TensorError::Device(format!(
                    "concat_i64_flat expected all tensors on {device:?}, got tensor[{index}] on {:?}",
                    tensor.device()
                )));
            }
            total = total.checked_add(tensor.numel()).ok_or_else(|| {
                TensorError::Shape("concat_i64_flat total length overflow".to_string())
            })?;
        }
        match device {
            Device::Cpu => {
                let mut data = Vec::with_capacity(total);
                for tensor in tensors {
                    data.extend(tensor.data_i64()?);
                }
                Self::from_i64(data, &[total], false)
            }
            Device::Cuda(device_id) => {
                let buffers = tensors
                    .iter()
                    .map(|tensor| tensor.cuda_i64_full_storage_buffer("concat_i64_flat input"))
                    .collect::<Result<Vec<_>>>()?;
                let output =
                    heirloom_kernels::cuda::concat_i64_buffers(&buffers).map_err(cuda_error)?;
                Self::from_storage(
                    Storage::new_cuda_from_buffer(device_id, DType::I64, total, output)?,
                    &[total],
                    DType::I64,
                    device,
                    false,
                    None,
                )
            }
        }
    }

    /// Copies the logical tensor to `device`, preserving dtype and gradient
    /// intent while materializing non-contiguous layouts as needed.
    pub fn to_device(&self, device: Device) -> Result<Self> {
        if self.device() == device {
            return Ok(self.clone());
        }
        match (self.device(), device) {
            (Device::Cpu, Device::Cuda(device_id)) => self.copy_cpu_to_cuda(device_id),
            (Device::Cuda(_), Device::Cpu) => self.copy_cuda_to_cpu(),
            (Device::Cuda(source), Device::Cuda(target)) => Err(TensorError::Device(format!(
                "CUDA-to-CUDA tensor copy from cuda:{source} to cuda:{target} requires peer/D2D copy support, which is not implemented yet"
            ))),
            (Device::Cpu, Device::Cpu) => Ok(self.clone()),
        }
    }

    pub fn cuda(&self, device_id: usize) -> Result<Self> {
        self.to_device(Device::Cuda(device_id))
    }

    pub fn cpu(&self) -> Result<Self> {
        self.to_device(Device::Cpu)
    }

    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if self.dtype() == dtype {
            return Ok(self.clone());
        }
        if self.device() != Device::Cpu {
            return self.cuda_to_dtype(dtype);
        }
        let requires_grad = should_track_grad(self.requires_grad() && dtype.is_floating());
        let grad_fn = requires_grad.then(|| GradFn::Cast {
            input: self.clone(),
        });
        Self::from_f64_values_with_dtype(
            self.try_data_f64()?,
            &self.shape(),
            dtype,
            requires_grad,
            grad_fn,
        )
    }

    fn cuda_to_dtype(&self, dtype: DType) -> Result<Self> {
        let device = self.device();
        let Device::Cuda(device_id) = device else {
            unreachable!("cuda_to_dtype called only for CUDA tensors");
        };
        let source_dtype = self.dtype();
        let output = match (source_dtype, dtype) {
            (DType::F32, DType::BFloat16) => {
                let input = self.cuda_f32_full_storage_buffer("CUDA f32->bf16 cast input")?;
                heirloom_kernels::cuda::f32_to_bf16_buffer(&input).map_err(cuda_error)?
            }
            (DType::BFloat16, DType::F32) => {
                let input = self.cuda_bf16_full_storage_buffer("CUDA bf16->f32 cast input")?;
                heirloom_kernels::cuda::bf16_to_f32_buffer(&input).map_err(cuda_error)?
            }
            _ => {
                return Err(TensorError::Device(format!(
                    "CUDA dtype conversion from {:?} to {:?} is not implemented; only f32<->bf16 casts are supported",
                    source_dtype, dtype
                )))
            }
        };
        let shape = self.shape();
        let requires_grad = should_track_grad(self.requires_grad() && dtype.is_floating());
        let grad_fn = requires_grad.then(|| GradFn::Cast {
            input: self.clone(),
        });
        Self::from_storage(
            Storage::new_cuda_from_buffer(device_id, dtype, self.numel(), output)?,
            &shape,
            dtype,
            device,
            requires_grad,
            grad_fn,
        )
    }

    pub(crate) fn cuda_f32_bf16_roundtrip(&self) -> Result<Self> {
        if self.device() == Device::Cpu {
            return Err(TensorError::Device(
                "cuda_f32_bf16_roundtrip requires a CUDA tensor".to_string(),
            ));
        }
        if self.dtype() != DType::F32 {
            return Err(TensorError::DType(format!(
                "cuda_f32_bf16_roundtrip requires f32 tensor, got {:?}",
                self.dtype()
            )));
        }
        let device = self.device();
        let Device::Cuda(device_id) = device else {
            unreachable!("cuda_f32_bf16_roundtrip checked CUDA device");
        };
        let input = self.cuda_f32_full_storage_buffer("CUDA f32->bf16->f32 roundtrip input")?;
        let output =
            heirloom_kernels::cuda::f32_bf16_roundtrip_f32_buffer(&input).map_err(cuda_error)?;
        let shape = self.shape();
        let requires_grad = should_track_grad(self.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::Cast {
            input: self.clone(),
        });
        Self::from_storage(
            Storage::new_cuda_from_buffer(device_id, DType::F32, self.numel(), output)?,
            &shape,
            DType::F32,
            device,
            requires_grad,
            grad_fn,
        )
    }

    fn copy_cpu_to_cuda(&self, device_id: usize) -> Result<Self> {
        let dtype = self.dtype();
        let shape = self.shape();
        let requires_grad = self.requires_grad();
        let cpu_storage = storage_from_f64_values(self.try_data_f64()?, dtype);
        let cuda_storage =
            cpu_storage.with_data(|data| Storage::new_cuda_from_cpu(device_id, dtype, data))?;
        Self::from_storage(
            cuda_storage,
            &shape,
            dtype,
            Device::Cuda(device_id),
            requires_grad,
            None,
        )
    }

    fn copy_cuda_to_cpu(&self) -> Result<Self> {
        let (storage, dtype, shape, requires_grad) = {
            let inner = self.inner.borrow();
            (
                Rc::clone(&inner.storage),
                inner.dtype,
                inner.shape.clone(),
                inner.requires_grad,
            )
        };
        let cpu_storage = storage.cpu_clone()?;
        let logical_storage = storage_from_f64_values(self.try_data_f64()?, dtype);
        if self.is_contiguous() && self.storage_offset() == 0 && cpu_storage.len() == self.numel() {
            Self::from_cpu_storage_like(cpu_storage, &shape, dtype, requires_grad)
        } else {
            Self::from_cpu_storage_like(logical_storage, &shape, dtype, requires_grad)
        }
    }

    pub fn copy_from(&self, source: &Self) -> Result<()> {
        ensure_same_dtype_device(self, source)?;
        if self.shape() != source.shape() {
            return Err(TensorError::Shape(format!(
                "copy_from requires matching shapes, got destination {:?} and source {:?}",
                self.shape(),
                source.shape()
            )));
        }
        match source.dtype() {
            DType::F32 => self.copy_from_data(&source.data_f32()?),
            DType::BFloat16 => self.copy_from_data_bf16_bits(&source.data_bf16_bits()?),
            DType::F64 => self.copy_from_data_f64(&source.data_f64_exact()?),
            DType::I64 => self.copy_from_data_i64(&source.data_i64()?),
            DType::Bool => self.copy_from_data_bool(&source.data_bool()?),
        }
    }

    pub fn copy_from_data(&self, source: &[f32]) -> Result<()> {
        ensure_dtype(self.dtype(), DType::F32, "copy_from_data")?;
        self.copy_from_f64_values(source.iter().map(|value| *value as f64), source.len())
    }

    pub fn copy_from_data_f64(&self, source: &[f64]) -> Result<()> {
        ensure_dtype(self.dtype(), DType::F64, "copy_from_data_f64")?;
        self.copy_from_f64_values(source.iter().copied(), source.len())
    }

    pub fn copy_from_data_bf16(&self, source: &[f32]) -> Result<()> {
        ensure_dtype(self.dtype(), DType::BFloat16, "copy_from_data_bf16")?;
        self.copy_from_f64_values(source.iter().map(|value| *value as f64), source.len())
    }

    pub fn copy_from_data_bf16_bits(&self, source: &[u16]) -> Result<()> {
        ensure_dtype(self.dtype(), DType::BFloat16, "copy_from_data_bf16_bits")?;
        self.copy_from_f64_values(
            source
                .iter()
                .map(|bits| crate::storage::bf16_bits_to_f32(*bits) as f64),
            source.len(),
        )
    }

    pub fn copy_from_data_i64(&self, source: &[i64]) -> Result<()> {
        ensure_dtype(self.dtype(), DType::I64, "copy_from_data_i64")?;
        self.copy_from_f64_values(source.iter().map(|value| *value as f64), source.len())
    }

    pub fn copy_from_data_bool(&self, source: &[bool]) -> Result<()> {
        ensure_dtype(self.dtype(), DType::Bool, "copy_from_data_bool")?;
        self.copy_from_f64_values(
            source.iter().map(|value| if *value { 1.0 } else { 0.0 }),
            source.len(),
        )
    }

    fn copy_from_f64_values(
        &self,
        source: impl Iterator<Item = f64>,
        source_len: usize,
    ) -> Result<()> {
        if source_len != self.numel() {
            return Err(TensorError::Shape(format!(
                "copy_from_data length {} does not match tensor shape {:?}",
                source_len,
                self.shape()
            )));
        }
        if self.has_internal_overlap() {
            return Err(TensorError::InvalidOperation(
                "cannot copy into tensor with internal overlap".to_string(),
            ));
        }
        if self.device() != Device::Cpu {
            return Err(TensorError::Device(format!(
                "copy_from_data into {:?} requires CUDA copy kernels for tensor views and is not implemented yet",
                self.device()
            )));
        }

        let (storage, shape, strides, offset) = self.layout_snapshot();
        let mut source_index = 0;
        let source = source.collect::<Vec<_>>();
        storage.with_data_mut(|data| {
            for_each_index(&shape, |index| {
                let storage_index = logical_offset(index, &strides, offset);
                data.set_from_f64(storage_index, source[source_index]);
                source_index += 1;
            });
        });
        Ok(())
    }

    pub fn grad(&self) -> Option<Vec<f32>> {
        self.grad_f64()
            .map(|grad| grad.into_iter().map(|value| value as f32).collect())
    }

    pub fn grad_f64(&self) -> Option<Vec<f64>> {
        self.inner
            .borrow()
            .grad
            .as_ref()
            .map(GradTensor::logical_data_f64)
    }

    pub fn grad_tensor(&self) -> Option<Self> {
        self.inner.borrow().grad.as_ref().map(GradTensor::as_tensor)
    }

    pub(crate) fn cuda_grad_f32_buffer(&self) -> Result<Option<CudaBuffer>> {
        self.inner
            .borrow()
            .grad
            .as_ref()
            .map(GradTensor::cuda_f32_full_storage_buffer)
            .transpose()
    }

    pub fn zero_grad(&self) {
        self.inner.borrow_mut().grad = None;
    }

    pub fn retain_grad(&self) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        if !inner.requires_grad {
            return Err(TensorError::Autograd(
                "retain_grad requires a tensor that requires gradients".to_string(),
            ));
        }
        inner.retain_grad = true;
        Ok(())
    }

    pub fn retains_grad(&self) -> bool {
        self.inner.borrow().retain_grad
    }

    pub fn register_grad_hook<F>(&self, hook: F) -> usize
    where
        F: Fn(&[f64]) -> Result<Vec<f64>> + 'static,
    {
        let mut inner = self.inner.borrow_mut();
        let id = inner.next_grad_hook_id;
        inner.next_grad_hook_id += 1;
        inner.grad_hooks.push(GradHookEntry {
            id,
            hook: Rc::new(hook),
        });
        id
    }

    pub fn remove_grad_hook(&self, hook_id: usize) -> bool {
        let mut inner = self.inner.borrow_mut();
        let original_len = inner.grad_hooks.len();
        inner.grad_hooks.retain(|entry| entry.id != hook_id);
        inner.grad_hooks.len() != original_len
    }

    pub fn detach(&self) -> Self {
        let (storage, shape, strides, offset, dtype, device) = {
            let inner = self.inner.borrow();
            (
                Rc::clone(&inner.storage),
                inner.shape.clone(),
                inner.strides.clone(),
                inner.storage_offset,
                inner.dtype,
                inner.device,
            )
        };
        Self::from_parts(storage, shape, strides, offset, dtype, device, false, None)
    }

    fn grad_fn(&self) -> Option<GradFn> {
        self.inner.borrow().grad_fn.clone()
    }

    fn graph_released(&self) -> bool {
        self.inner.borrow().grad_fn_released
    }

    fn release_grad_fn(&self) {
        let mut inner = self.inner.borrow_mut();
        if inner.grad_fn.is_some() {
            inner.grad_fn = None;
            inner.grad_fn_released = true;
        }
    }

    fn should_keep_grad_after_backward(&self) -> bool {
        let inner = self.inner.borrow();
        if !inner.requires_grad {
            return false;
        }
        let is_leaf = inner.grad_fn.is_none() && !inner.grad_fn_released;
        is_leaf || inner.retain_grad
    }

    fn clear_transient_grad_after_backward(&self) {
        if !self.should_keep_grad_after_backward() {
            self.zero_grad();
        }
    }

    fn layout_snapshot(&self) -> (Rc<Storage>, Vec<usize>, Vec<usize>, usize) {
        let inner = self.inner.borrow();
        (
            Rc::clone(&inner.storage),
            inner.shape.clone(),
            inner.strides.clone(),
            inner.storage_offset,
        )
    }

    pub(super) fn cuda_f32_full_storage_buffer(&self, role: &str) -> Result<CudaBuffer> {
        let (storage, shape, strides, offset, dtype, device) = {
            let inner = self.inner.borrow();
            (
                Rc::clone(&inner.storage),
                inner.shape.clone(),
                inner.strides.clone(),
                inner.storage_offset,
                inner.dtype,
                inner.device,
            )
        };
        if dtype != DType::F32 && dtype != DType::BFloat16 {
            return Err(TensorError::DType(format!(
                "{role} expected f32 or bf16, got {dtype:?}"
            )));
        }
        if !matches!(device, Device::Cuda(_)) {
            return Err(TensorError::Device(format!(
                "{role} expected CUDA tensor, got {device:?}"
            )));
        }
        if !is_contiguous(&shape, &strides) || offset != 0 || storage.len() != numel(&shape) {
            return Err(TensorError::Device(format!(
                "{role} requires a contiguous full-storage CUDA tensor; shape={shape:?} strides={strides:?} offset={offset} storage_len={}",
                storage.len()
            )));
        }
        storage.cuda_buffer(DType::F32, device)
    }

    pub(super) fn cuda_bf16_full_storage_buffer(&self, role: &str) -> Result<CudaBuffer> {
        let (storage, shape, strides, offset, dtype, device) = {
            let inner = self.inner.borrow();
            (
                Rc::clone(&inner.storage),
                inner.shape.clone(),
                inner.strides.clone(),
                inner.storage_offset,
                inner.dtype,
                inner.device,
            )
        };
        if dtype != DType::BFloat16 {
            return Err(TensorError::DType(format!(
                "{role} expected bf16, got {dtype:?}"
            )));
        }
        if !matches!(device, Device::Cuda(_)) {
            return Err(TensorError::Device(format!(
                "{role} expected CUDA tensor, got {device:?}"
            )));
        }
        if !is_contiguous(&shape, &strides) || offset != 0 || storage.len() != numel(&shape) {
            return Err(TensorError::Device(format!(
                "{role} requires a contiguous full-storage CUDA tensor; shape={shape:?} strides={strides:?} offset={offset} storage_len={}",
                storage.len()
            )));
        }
        storage.cuda_buffer(DType::BFloat16, device)
    }

    pub(super) fn cuda_matrix_layout(
        &self,
        role: &str,
    ) -> Result<(CudaBuffer, heirloom_kernels::cuda::MatrixLayout)> {
        let (storage, shape, strides, offset, dtype, device) = {
            let inner = self.inner.borrow();
            (
                Rc::clone(&inner.storage),
                inner.shape.clone(),
                inner.strides.clone(),
                inner.storage_offset,
                inner.dtype,
                inner.device,
            )
        };
        if !matches!(dtype, DType::F32 | DType::BFloat16) {
            return Err(TensorError::DType(format!(
                "{role} expected f32 or bf16, got {dtype:?}"
            )));
        }
        if !matches!(device, Device::Cuda(_)) {
            return Err(TensorError::Device(format!(
                "{role} expected CUDA tensor, got {device:?}"
            )));
        }
        if shape.len() != 2 {
            return Err(TensorError::Shape(format!(
                "{role} expected rank-2 CUDA tensor, got {shape:?}"
            )));
        }
        if numel(&shape) > 0
            && checked_max_storage_offset(&shape, &strides, offset)? >= storage.len()
        {
            return Err(TensorError::Shape(format!(
                "{role} layout shape={shape:?} strides={strides:?} offset={offset} exceeds storage length {}",
                storage.len()
            )));
        }
        let buffer = storage.cuda_buffer(dtype, device)?;
        Ok((
            buffer,
            heirloom_kernels::cuda::MatrixLayout {
                rows: shape[0],
                cols: shape[1],
                row_stride: strides[0],
                col_stride: strides[1],
                offset,
            },
        ))
    }

    pub(super) fn cuda_i64_full_storage_buffer(&self, role: &str) -> Result<CudaBuffer> {
        let (storage, shape, strides, offset, dtype, device) = {
            let inner = self.inner.borrow();
            (
                Rc::clone(&inner.storage),
                inner.shape.clone(),
                inner.strides.clone(),
                inner.storage_offset,
                inner.dtype,
                inner.device,
            )
        };
        if dtype != DType::I64 {
            return Err(TensorError::DType(format!(
                "{role} expected i64, got {dtype:?}"
            )));
        }
        if !matches!(device, Device::Cuda(_)) {
            return Err(TensorError::Device(format!(
                "{role} expected CUDA tensor, got {device:?}"
            )));
        }
        if !is_contiguous(&shape, &strides) || offset != 0 || storage.len() != numel(&shape) {
            return Err(TensorError::Device(format!(
                "{role} requires a contiguous full-storage CUDA tensor; shape={shape:?} strides={strides:?} offset={offset} storage_len={}",
                storage.len()
            )));
        }
        storage.cuda_buffer(DType::I64, device)
    }

    pub(super) fn cuda_bool_full_storage_buffer(&self, role: &str) -> Result<CudaBuffer> {
        let (storage, shape, strides, offset, dtype, device) = {
            let inner = self.inner.borrow();
            (
                Rc::clone(&inner.storage),
                inner.shape.clone(),
                inner.strides.clone(),
                inner.storage_offset,
                inner.dtype,
                inner.device,
            )
        };
        if dtype != DType::Bool {
            return Err(TensorError::DType(format!(
                "{role} expected bool, got {dtype:?}"
            )));
        }
        if !matches!(device, Device::Cuda(_)) {
            return Err(TensorError::Device(format!(
                "{role} expected CUDA tensor, got {device:?}"
            )));
        }
        if !is_contiguous(&shape, &strides) || offset != 0 || storage.len() != numel(&shape) {
            return Err(TensorError::Device(format!(
                "{role} requires a contiguous full-storage CUDA tensor; shape={shape:?} strides={strides:?} offset={offset} storage_len={}",
                storage.len()
            )));
        }
        storage.cuda_buffer(DType::Bool, device)
    }

    pub(crate) fn cuda_i64_memory_access_counts(&self, memory_slots: usize) -> Result<Vec<u64>> {
        if self.ndim() != 1 {
            return Err(TensorError::Shape(format!(
                "CUDA memory access counting expects rank-1 selected rows, got {:?}",
                self.shape()
            )));
        }
        let selected_rows =
            self.cuda_i64_full_storage_buffer("CUDA memory access count selected rows")?;
        let counts =
            heirloom_kernels::cuda::memory_access_count_rows_i64_u64(&selected_rows, memory_slots)
                .map_err(cuda_error)?;
        counts.to_u64().map_err(cuda_error)
    }

    pub fn cuda_row_union_mask_nccl(
        &self,
        rows: usize,
        communicator: &mut heirloom_kernels::cuda::NcclCommunicator,
    ) -> Result<(Self, heirloom_kernels::cuda::NcclAllReduceStats)> {
        if self.ndim() != 1 {
            return Err(TensorError::Shape(format!(
                "CUDA row-union mask expects rank-1 selected rows, got {:?}",
                self.shape()
            )));
        }
        let Device::Cuda(device_id) = self.device() else {
            return Err(TensorError::Device(format!(
                "CUDA row-union mask expects CUDA selected rows, got {:?}",
                self.device()
            )));
        };
        if rows == 0 {
            return Err(TensorError::Shape(
                "CUDA row-union mask requires at least one row".to_string(),
            ));
        }
        let selected_rows =
            self.cuda_i64_full_storage_buffer("CUDA row-union mask selected rows")?;
        let mask = heirloom_kernels::cuda::memory_selected_rows_to_f32_mask(&selected_rows, rows)
            .map_err(cuda_error)?;
        let stats = communicator
            .all_reduce_sum_in_place_f32(&mask)
            .map_err(cuda_error)?;
        let mask = heirloom_kernels::cuda::f32_mask_to_bool_buffer(&mask).map_err(cuda_error)?;
        let tensor = Self::from_parts(
            Storage::new_cuda_from_buffer(device_id, DType::Bool, rows, mask)?,
            vec![rows],
            contiguous_strides(&[rows]),
            0,
            DType::Bool,
            Device::Cuda(device_id),
            false,
            None,
        );
        Ok((tensor, stats))
    }

    pub fn cuda_selected_rows_mask(&self, rows: usize) -> Result<Self> {
        if self.ndim() != 1 {
            return Err(TensorError::Shape(format!(
                "CUDA selected-rows mask expects rank-1 selected rows, got {:?}",
                self.shape()
            )));
        }
        let Device::Cuda(device_id) = self.device() else {
            return Err(TensorError::Device(format!(
                "CUDA selected-rows mask expects CUDA selected rows, got {:?}",
                self.device()
            )));
        };
        if rows == 0 {
            return Err(TensorError::Shape(
                "CUDA selected-rows mask requires at least one row".to_string(),
            ));
        }
        let selected_rows =
            self.cuda_i64_full_storage_buffer("CUDA selected-rows mask selected rows")?;
        let mask = heirloom_kernels::cuda::memory_selected_rows_to_f32_mask(&selected_rows, rows)
            .map_err(cuda_error)?;
        let mask = heirloom_kernels::cuda::f32_mask_to_bool_buffer(&mask).map_err(cuda_error)?;
        Ok(Self::from_parts(
            Storage::new_cuda_from_buffer(device_id, DType::Bool, rows, mask)?,
            vec![rows],
            contiguous_strides(&[rows]),
            0,
            DType::Bool,
            Device::Cuda(device_id),
            false,
            None,
        ))
    }

    pub fn cuda_bool_and(&self, other: &Self) -> Result<Self> {
        if self.dtype() != DType::Bool || other.dtype() != DType::Bool {
            return Err(TensorError::DType(format!(
                "CUDA bool-and expects bool tensors, got {:?} and {:?}",
                self.dtype(),
                other.dtype()
            )));
        }
        let Device::Cuda(device_id) = self.device() else {
            return Err(TensorError::Device(format!(
                "CUDA bool-and expects CUDA tensors, got {:?}",
                self.device()
            )));
        };
        if other.device() != Device::Cuda(device_id) {
            return Err(TensorError::Device(format!(
                "CUDA bool-and expects both masks on cuda:{device_id}, got {:?}",
                other.device()
            )));
        }
        if self.shape() != other.shape() {
            return Err(TensorError::Shape(format!(
                "CUDA bool-and expects matching shapes, got {:?} and {:?}",
                self.shape(),
                other.shape()
            )));
        }
        let left = self.cuda_bool_full_storage_buffer("CUDA bool-and left mask")?;
        let right = other.cuda_bool_full_storage_buffer("CUDA bool-and right mask")?;
        let output = heirloom_kernels::cuda::bool_and_buffers(&left, &right).map_err(cuda_error)?;
        Ok(Self::from_parts(
            Storage::new_cuda_from_buffer(device_id, DType::Bool, self.numel(), output)?,
            self.shape(),
            contiguous_strides(&self.shape()),
            0,
            DType::Bool,
            Device::Cuda(device_id),
            false,
            None,
        ))
    }

    pub fn cuda_bool_mask_to_i64_indices(&self) -> Result<Self> {
        if self.dtype() != DType::Bool {
            return Err(TensorError::DType(format!(
                "CUDA bool mask compaction expects bool tensor, got {:?}",
                self.dtype()
            )));
        }
        let Device::Cuda(device_id) = self.device() else {
            return Err(TensorError::Device(format!(
                "CUDA bool mask compaction expects CUDA tensor, got {:?}",
                self.device()
            )));
        };
        if self.ndim() != 1 {
            return Err(TensorError::Shape(format!(
                "CUDA bool mask compaction expects rank-1 mask, got {:?}",
                self.shape()
            )));
        }
        let mask = self.cuda_bool_full_storage_buffer("CUDA bool mask compaction input")?;
        let (indices, count) =
            heirloom_kernels::cuda::bool_mask_to_i64_indices(&mask).map_err(cuda_error)?;
        Ok(Self::from_parts(
            Storage::new_cuda_from_buffer(device_id, DType::I64, count, indices)?,
            vec![count],
            contiguous_strides(&[count]),
            0,
            DType::I64,
            Device::Cuda(device_id),
            false,
            None,
        ))
    }

    pub fn cuda_memory_gather_selected_rows_f32_i64(&self, selected_rows: &Self) -> Result<Self> {
        if self.dtype() != DType::F32 {
            return Err(TensorError::DType(format!(
                "CUDA selected-row gather expects an f32 table, got {:?}",
                self.dtype()
            )));
        }
        if selected_rows.dtype() != DType::I64 {
            return Err(TensorError::DType(format!(
                "CUDA selected-row gather expects i64 selected rows, got {:?}",
                selected_rows.dtype()
            )));
        }
        let Device::Cuda(device_id) = self.device() else {
            return Err(TensorError::Device(format!(
                "CUDA selected-row gather expects a CUDA table, got {:?}",
                self.device()
            )));
        };
        if selected_rows.device() != Device::Cuda(device_id) {
            return Err(TensorError::Device(format!(
                "CUDA selected-row gather expects selected rows on cuda:{device_id}, got {:?}",
                selected_rows.device()
            )));
        }
        if self.ndim() != 2 || selected_rows.ndim() != 1 {
            return Err(TensorError::Shape(format!(
                "CUDA selected-row gather expects table [rows, row_dim] and selected rows [n], got {:?} and {:?}",
                self.shape(),
                selected_rows.shape()
            )));
        }
        let shape = self.shape();
        let selected_count = selected_rows.numel();
        let table = self.cuda_f32_full_storage_buffer("CUDA selected-row gather table")?;
        let rows =
            selected_rows.cuda_i64_full_storage_buffer("CUDA selected-row gather selected rows")?;
        let output = heirloom_kernels::cuda::memory_gather_selected_rows_f32_i64(
            &table,
            &rows,
            heirloom_kernels::cuda::SelectedRowsDims {
                selected_rows: selected_count,
                rows: shape[0],
                row_dim: shape[1],
            },
        )
        .map_err(cuda_error)?;
        Ok(Self::from_parts(
            Storage::new_cuda_from_buffer(
                device_id,
                DType::F32,
                selected_count * shape[1],
                output,
            )?,
            vec![selected_count, shape[1]],
            contiguous_strides(&[selected_count, shape[1]]),
            0,
            DType::F32,
            Device::Cuda(device_id),
            false,
            None,
        ))
    }

    pub fn cuda_memory_gather_selected_grad_rows_f32_i64(
        &self,
        selected_rows: &Self,
    ) -> Result<Option<Self>> {
        if self.dtype() != DType::F32 {
            return Err(TensorError::DType(format!(
                "CUDA selected gradient-row gather expects an f32 parameter, got {:?}",
                self.dtype()
            )));
        }
        let Device::Cuda(device_id) = self.device() else {
            return Err(TensorError::Device(format!(
                "CUDA selected gradient-row gather expects a CUDA parameter, got {:?}",
                self.device()
            )));
        };
        if selected_rows.dtype() != DType::I64 || selected_rows.device() != Device::Cuda(device_id)
        {
            return Err(TensorError::Device(format!(
                "CUDA selected gradient-row gather expects i64 selected rows on cuda:{device_id}, got {:?} on {:?}",
                selected_rows.dtype(),
                selected_rows.device()
            )));
        }
        if self.ndim() != 2 || selected_rows.ndim() != 1 {
            return Err(TensorError::Shape(format!(
                "CUDA selected gradient-row gather expects parameter [rows, row_dim] and selected rows [n], got {:?} and {:?}",
                self.shape(),
                selected_rows.shape()
            )));
        }
        let Some(grad) = self.cuda_grad_f32_buffer()? else {
            return Ok(None);
        };
        let shape = self.shape();
        let selected_count = selected_rows.numel();
        let rows =
            selected_rows.cuda_i64_full_storage_buffer("CUDA selected gradient-row gather rows")?;
        let output = heirloom_kernels::cuda::memory_gather_selected_rows_f32_i64(
            &grad,
            &rows,
            heirloom_kernels::cuda::SelectedRowsDims {
                selected_rows: selected_count,
                rows: shape[0],
                row_dim: shape[1],
            },
        )
        .map_err(cuda_error)?;
        Ok(Some(Self::from_parts(
            Storage::new_cuda_from_buffer(
                device_id,
                DType::F32,
                selected_count * shape[1],
                output,
            )?,
            vec![selected_count, shape[1]],
            contiguous_strides(&[selected_count, shape[1]]),
            0,
            DType::F32,
            Device::Cuda(device_id),
            false,
            None,
        )))
    }

    pub(crate) fn cuda_memory_product_key_topk_indices_from_side_scores(
        &self,
        right_scores: &Self,
        slots: usize,
        key_dim: usize,
        top_k: usize,
        beam: usize,
    ) -> Result<Self> {
        if self.device() != right_scores.device() {
            return Err(TensorError::Device(format!(
                "CUDA product-key side-score lookup expected left/right scores on the same device, got {:?} and {:?}",
                self.device(),
                right_scores.device()
            )));
        }
        let device = self.device();
        let Device::Cuda(device_id) = device else {
            return Err(TensorError::Device(format!(
                "CUDA product-key side-score lookup expected CUDA scores, got {device:?}"
            )));
        };
        if self.dtype() != DType::F32 || right_scores.dtype() != DType::F32 {
            return Err(TensorError::DType(format!(
                "CUDA product-key side-score lookup expects f32 scores, got {:?}/{:?}",
                self.dtype(),
                right_scores.dtype()
            )));
        }
        let left_shape = self.shape();
        let right_shape = right_scores.shape();
        if left_shape.len() != 2 || right_shape != left_shape {
            return Err(TensorError::Shape(format!(
                "CUDA product-key side-score lookup expected matching [tokens, side] scores, got {left_shape:?} and {right_shape:?}"
            )));
        }
        let dims = heirloom_kernels::cuda::MemoryProductKeyDims {
            tokens: left_shape[0],
            slots,
            key_dim,
            top_k,
            beam,
        };
        let left = self.cuda_f32_full_storage_buffer("CUDA product-key left side scores")?;
        let right =
            right_scores.cuda_f32_full_storage_buffer("CUDA product-key right side scores")?;
        let output = heirloom_kernels::cuda::memory_product_key_topk_indices_from_side_scores_f32(
            &left, &right, dims,
        )
        .map_err(cuda_error)?;
        Self::from_storage(
            Storage::new_cuda_from_buffer(device_id, DType::I64, dims.tokens * dims.top_k, output)?,
            &[dims.tokens, dims.top_k],
            DType::I64,
            device,
            false,
            None,
        )
    }

    pub(crate) fn memory_product_key_half_rows(&self, side: usize) -> Result<(Self, Self)> {
        if self.dtype() != DType::I64 {
            return Err(TensorError::DType(format!(
                "product-key row splitting expects i64 selected rows, got {:?}",
                self.dtype()
            )));
        }
        if side == 0 {
            return Err(TensorError::InvalidOperation(
                "product-key row splitting requires non-zero side".to_string(),
            ));
        }
        match self.device() {
            Device::Cpu => {
                let slots = side
                    .checked_mul(side)
                    .ok_or_else(|| TensorError::Shape("product-key side^2 overflow".to_string()))?;
                let rows = self
                    .data_i64()?
                    .into_iter()
                    .map(|row| {
                        if row < 0 || row as usize >= slots {
                            Err(TensorError::Shape(format!(
                                "product-key selected row {row} out of range for slots {slots}"
                            )))
                        } else {
                            Ok(row as usize)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                let left = rows.iter().map(|row| (row / side) as i64).collect();
                let right = rows.iter().map(|row| (row % side) as i64).collect();
                let shape = self.shape();
                Ok((
                    Self::from_i64(left, &shape, false)?,
                    Self::from_i64(right, &shape, false)?,
                ))
            }
            Device::Cuda(device_id) => {
                let rows =
                    self.cuda_i64_full_storage_buffer("CUDA product-key split selected rows")?;
                let (left, right) =
                    heirloom_kernels::cuda::memory_product_key_split_rows_i64(&rows, side)
                        .map_err(cuda_error)?;
                let shape = self.shape();
                Ok((
                    Self::from_storage(
                        Storage::new_cuda_from_buffer(device_id, DType::I64, self.numel(), left)?,
                        &shape,
                        DType::I64,
                        self.device(),
                        false,
                        None,
                    )?,
                    Self::from_storage(
                        Storage::new_cuda_from_buffer(device_id, DType::I64, self.numel(), right)?,
                        &shape,
                        DType::I64,
                        self.device(),
                        false,
                        None,
                    )?,
                ))
            }
        }
    }

    fn add_grad(&self, grad: Vec<f64>) -> Result<()> {
        if self.device() != Device::Cpu {
            return self.add_grad_cuda_f32_from_f64(grad);
        }
        self.add_grad_cpu(grad)
    }

    fn add_grad_cpu(&self, mut grad: Vec<f64>) -> Result<()> {
        let (shape, strides, storage_offset, hooks) = {
            let inner = self.inner.borrow();
            (
                inner.shape.clone(),
                inner.strides.clone(),
                inner.storage_offset,
                inner.grad_hooks.clone(),
            )
        };
        if grad.len() != numel(&shape) {
            return Err(TensorError::Autograd(format!(
                "gradient length {} does not match tensor shape {:?}",
                grad.len(),
                shape
            )));
        }

        for hook in hooks {
            let updated = (hook.hook)(&grad)?;
            if updated.len() != grad.len() {
                return Err(TensorError::Autograd(format!(
                    "gradient hook {} returned {} values for tensor shape {:?} with {} gradient values",
                    hook.id,
                    updated.len(),
                    shape,
                    grad.len()
                )));
            }
            grad = updated;
        }

        let mut inner = self.inner.borrow_mut();
        match &mut inner.grad {
            Some(existing) => existing.add_logical(&grad)?,
            None => {
                let grad_tensor =
                    GradTensor::zeros_for_tensor_layout(&shape, &strides, storage_offset)?;
                grad_tensor.add_logical(&grad)?;
                inner.grad = Some(grad_tensor);
            }
        }
        Ok(())
    }

    fn add_grad_cuda_f32_from_f64(&self, grad: Vec<f64>) -> Result<()> {
        if grad.len() != self.numel() {
            return Err(TensorError::Autograd(format!(
                "gradient length {} does not match tensor shape {:?}",
                grad.len(),
                self.shape()
            )));
        }
        let Device::Cuda(device_id) = self.device() else {
            unreachable!("add_grad_cuda_f32_from_f64 called only for CUDA tensors");
        };
        let grad_f32 = grad
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let buffer = CudaBuffer::from_f32(
            i32::try_from(device_id).map_err(|_| {
                TensorError::Device(format!(
                    "CUDA device id {device_id} does not fit into a CUDA ordinal"
                ))
            })?,
            &grad_f32,
        )
        .map_err(cuda_error)?;
        self.add_grad_cuda_f32_buffer(buffer)
    }

    fn add_grad_cuda_f32_buffer(&self, grad: CudaBuffer) -> Result<()> {
        let (shape, strides, storage_offset, hooks, dtype, device) = {
            let inner = self.inner.borrow();
            (
                inner.shape.clone(),
                inner.strides.clone(),
                inner.storage_offset,
                inner.grad_hooks.clone(),
                inner.dtype,
                inner.device,
            )
        };
        if !matches!(dtype, DType::F32 | DType::BFloat16) || !matches!(device, Device::Cuda(_)) {
            return Err(TensorError::Autograd(format!(
                "CUDA gradient accumulation currently supports only CUDA f32/bf16 tensors, got {:?} on {:?}",
                dtype, device
            )));
        }
        if !hooks.is_empty() {
            return Err(TensorError::Autograd(
                "tensor gradient hooks for CUDA tensors are not implemented without CPU materialization"
                    .to_string(),
            ));
        }
        if grad.len() != numel(&shape) {
            return Err(TensorError::Autograd(format!(
                "CUDA gradient length {} does not match tensor shape {:?}",
                grad.len(),
                shape
            )));
        }

        let mut inner = self.inner.borrow_mut();
        match &mut inner.grad {
            Some(existing) => existing.add_cuda_f32_full(&grad)?,
            None => {
                inner.grad = Some(GradTensor::from_cuda_f32_full(
                    &shape,
                    &strides,
                    storage_offset,
                    device,
                    grad,
                )?);
            }
        }
        Ok(())
    }

    /// Runs reverse-mode autodiff from this scalar and accumulates leaf
    /// gradients.
    ///
    /// The output must contain exactly one element and require gradients. The
    /// graph is released after traversal; use [`Tensor::backward_retain_graph`]
    /// only when a second traversal is intentional.
    pub fn backward(&self) -> Result<()> {
        if !self.requires_grad() {
            return Err(TensorError::Autograd(
                "cannot call backward on a tensor that does not require gradients".to_string(),
            ));
        }
        if self.numel() != 1 {
            return Err(TensorError::Autograd(format!(
                "backward requires a scalar output, got shape {:?}",
                self.shape()
            )));
        }
        self.backward_with_grad_f64_impl(vec![1.0], false)
    }

    pub fn backward_retain_graph(&self) -> Result<()> {
        if !self.requires_grad() {
            return Err(TensorError::Autograd(
                "cannot call backward on a tensor that does not require gradients".to_string(),
            ));
        }
        if self.numel() != 1 {
            return Err(TensorError::Autograd(format!(
                "backward requires a scalar output, got shape {:?}",
                self.shape()
            )));
        }
        self.backward_with_grad_f64_impl(vec![1.0], true)
    }

    pub fn backward_with_grad(&self, seed_grad: Vec<f32>) -> Result<()> {
        self.backward_with_grad_f64(seed_grad.into_iter().map(f64::from).collect())
    }

    pub fn backward_with_grad_retain_graph(&self, seed_grad: Vec<f32>) -> Result<()> {
        self.backward_with_grad_f64_retain_graph(seed_grad.into_iter().map(f64::from).collect())
    }

    pub fn backward_with_grad_f64(&self, seed_grad: Vec<f64>) -> Result<()> {
        self.backward_with_grad_f64_impl(seed_grad, false)
    }

    pub fn backward_with_grad_f64_retain_graph(&self, seed_grad: Vec<f64>) -> Result<()> {
        self.backward_with_grad_f64_impl(seed_grad, true)
    }

    fn backward_with_grad_f64_impl(&self, seed_grad: Vec<f64>, retain_graph: bool) -> Result<()> {
        if seed_grad.len() != self.numel() {
            return Err(TensorError::Autograd(format!(
                "seed gradient length {} does not match output shape {:?}",
                seed_grad.len(),
                self.shape()
            )));
        }

        let mut topo = Vec::new();
        let mut seen = HashSet::new();
        build_topo(self, &mut seen, &mut topo)?;

        self.add_grad(seed_grad)?;
        for tensor in topo.iter().rev() {
            if tensor.device() != Device::Cpu {
                if let Some(grad_output) = tensor.cuda_grad_f32_buffer()? {
                    if let Some(grad_fn) = tensor.grad_fn() {
                        for (parent, parent_grad) in grad_fn.backward_cuda_f32(&grad_output)? {
                            if parent.requires_grad() {
                                parent.add_grad_cuda_f32_buffer(parent_grad)?;
                            }
                        }
                    }
                }
            } else if let Some(grad_output) = tensor.grad_f64() {
                if let Some(grad_fn) = tensor.grad_fn() {
                    for (parent, parent_grad) in grad_fn.backward(&grad_output)? {
                        if parent.requires_grad() {
                            parent.add_grad(parent_grad)?;
                        }
                    }
                }
            }
            tensor.clear_transient_grad_after_backward();
        }
        if !retain_graph {
            for tensor in &topo {
                tensor.release_grad_fn();
            }
        }
        Ok(())
    }

    pub fn apply_sgd(&self, lr: f32) -> Result<()> {
        if lr <= 0.0 {
            return Err(TensorError::InvalidOperation(format!(
                "learning rate must be positive, got {lr}"
            )));
        }
        ensure_floating_tensor(self, "apply_sgd")?;
        if self.device() != Device::Cpu {
            return self.apply_sgd_cuda(lr);
        }
        let Some(grad) = self.grad_f64() else {
            return Ok(());
        };
        self.apply_gradient_update(&grad, -(lr as f64))
    }

    fn apply_sgd_cuda(&self, lr: f32) -> Result<()> {
        let Some(grad) = self.cuda_grad_f32_buffer()? else {
            return Ok(());
        };
        let (storage, shape, strides, offset, dtype, device) = {
            let inner = self.inner.borrow();
            (
                Rc::clone(&inner.storage),
                inner.shape.clone(),
                inner.strides.clone(),
                inner.storage_offset,
                inner.dtype,
                inner.device,
            )
        };
        if dtype != DType::F32 {
            return Err(TensorError::DType(format!(
                "CUDA apply_sgd currently supports f32 parameters only, got {dtype:?}"
            )));
        }
        if !matches!(device, Device::Cuda(_)) {
            return Err(TensorError::Device(format!(
                "CUDA apply_sgd expected CUDA parameter, got {device:?}"
            )));
        }
        if !is_contiguous(&shape, &strides) || offset != 0 || storage.len() != numel(&shape) {
            return Err(TensorError::Device(format!(
                "CUDA apply_sgd requires a contiguous full-storage parameter; shape={shape:?} strides={strides:?} offset={offset} storage_len={}",
                storage.len()
            )));
        }
        if grad.len() != storage.len() {
            return Err(TensorError::Shape(format!(
                "CUDA gradient update length {} does not match tensor shape {:?}",
                grad.len(),
                shape
            )));
        }

        let param = storage.cuda_buffer(DType::F32, device)?;
        heirloom_kernels::cuda::sgd_update_f32_buffer(&param, &grad, lr).map_err(cuda_error)?;
        storage.bump_version();
        Ok(())
    }

    pub(crate) fn cuda_grad_sum_squares_f32(&self) -> Result<Option<f32>> {
        let Some(grad) = self.cuda_grad_f32_buffer()? else {
            return Ok(None);
        };
        let sum_squares =
            heirloom_kernels::cuda::sum_squares_f32_buffer(&grad).map_err(cuda_error)?;
        let values = sum_squares.to_f32().map_err(cuda_error)?;
        Ok(values.first().copied())
    }

    pub(crate) fn cuda_f32_sum_squares(&self) -> Result<f32> {
        let buffer = self.cuda_f32_full_storage_buffer("CUDA f32 sum-of-squares")?;
        let sum_squares =
            heirloom_kernels::cuda::sum_squares_f32_buffer(&buffer).map_err(cuda_error)?;
        let values = sum_squares.to_f32().map_err(cuda_error)?;
        values.first().copied().ok_or_else(|| {
            TensorError::Device(
                "CUDA f32 sum-of-squares returned an empty scalar buffer".to_string(),
            )
        })
    }

    pub fn cuda_all_reduce_sum_in_place_f32_nccl(
        &self,
        communicator: &mut NcclCommunicator,
        scale_after_sum: f32,
    ) -> Result<NcclAllReduceStats> {
        if !scale_after_sum.is_finite() {
            return Err(TensorError::InvalidOperation(format!(
                "CUDA f32 NCCL all-reduce expected finite post-scale, got {scale_after_sum}"
            )));
        }
        let (storage, shape, strides, offset, dtype, device) = {
            let inner = self.inner.borrow();
            (
                Rc::clone(&inner.storage),
                inner.shape.clone(),
                inner.strides.clone(),
                inner.storage_offset,
                inner.dtype,
                inner.device,
            )
        };
        if dtype != DType::F32 {
            return Err(TensorError::DType(format!(
                "CUDA f32 NCCL all-reduce expected f32 tensor, got {dtype:?}"
            )));
        }
        if !matches!(device, Device::Cuda(_)) {
            return Err(TensorError::Device(format!(
                "CUDA f32 NCCL all-reduce expected CUDA tensor, got {device:?}"
            )));
        }
        if !is_contiguous(&shape, &strides) || offset != 0 || storage.len() != numel(&shape) {
            return Err(TensorError::Device(format!(
                "CUDA f32 NCCL all-reduce requires a contiguous full-storage tensor; shape={shape:?} strides={strides:?} offset={offset} storage_len={}",
                storage.len()
            )));
        }
        let buffer = storage.cuda_buffer(DType::F32, device)?;
        let stats = communicator
            .all_reduce_sum_in_place_f32(&buffer)
            .map_err(cuda_error)?;
        if stats.calls > 0 && scale_after_sum != 1.0 {
            heirloom_kernels::cuda::scale_in_place_f32_buffer(&buffer, scale_after_sum)
                .map_err(cuda_error)?;
        }
        if stats.calls > 0 {
            storage.bump_version();
        }
        Ok(stats)
    }

    pub(crate) fn apply_adamw_cuda_f32(
        &self,
        m: &CudaBuffer,
        v: &CudaBuffer,
        adamw: heirloom_kernels::cuda::AdamWParams,
    ) -> Result<bool> {
        let Some(grad) = self.cuda_grad_f32_buffer()? else {
            return Ok(false);
        };
        let (storage, shape, strides, offset, dtype, device) = {
            let inner = self.inner.borrow();
            (
                Rc::clone(&inner.storage),
                inner.shape.clone(),
                inner.strides.clone(),
                inner.storage_offset,
                inner.dtype,
                inner.device,
            )
        };
        if dtype != DType::F32 {
            return Err(TensorError::DType(format!(
                "CUDA AdamW currently supports f32 parameters only, got {dtype:?}"
            )));
        }
        if !matches!(device, Device::Cuda(_)) {
            return Err(TensorError::Device(format!(
                "CUDA AdamW expected CUDA parameter, got {device:?}"
            )));
        }
        if !is_contiguous(&shape, &strides) || offset != 0 || storage.len() != numel(&shape) {
            return Err(TensorError::Device(format!(
                "CUDA AdamW requires a contiguous full-storage parameter; shape={shape:?} strides={strides:?} offset={offset} storage_len={}",
                storage.len()
            )));
        }
        if grad.len() != storage.len() {
            return Err(TensorError::Shape(format!(
                "CUDA AdamW gradient length {} does not match tensor shape {:?}",
                grad.len(),
                shape
            )));
        }

        let param = storage.cuda_buffer(DType::F32, device)?;
        heirloom_kernels::cuda::adamw_update_f32_buffers(&param, &grad, m, v, adamw)
            .map_err(cuda_error)?;
        storage.bump_version();
        Ok(true)
    }

    pub(crate) fn apply_adamw_cuda_sparse_rows_f32(
        &self,
        selected_rows: &Self,
        row_mask: Option<&Self>,
        m: &CudaBuffer,
        v: &CudaBuffer,
        dims: heirloom_kernels::cuda::SparseAdamWRowsDims,
        adamw: heirloom_kernels::cuda::AdamWParams,
    ) -> Result<bool> {
        let Some(grad) = self.cuda_grad_f32_buffer()? else {
            return Ok(false);
        };
        let (storage, shape, strides, offset, dtype, device) = {
            let inner = self.inner.borrow();
            (
                Rc::clone(&inner.storage),
                inner.shape.clone(),
                inner.strides.clone(),
                inner.storage_offset,
                inner.dtype,
                inner.device,
            )
        };
        if dtype != DType::F32 {
            return Err(TensorError::DType(format!(
                "CUDA sparse-row AdamW currently supports f32 parameters only, got {dtype:?}"
            )));
        }
        if !matches!(device, Device::Cuda(_)) {
            return Err(TensorError::Device(format!(
                "CUDA sparse-row AdamW expected CUDA parameter, got {device:?}"
            )));
        }
        if selected_rows.dtype() != DType::I64 || selected_rows.device() != device {
            return Err(TensorError::Device(format!(
                "CUDA sparse-row AdamW expected i64 selected rows on {device:?}, got {:?} on {:?}",
                selected_rows.dtype(),
                selected_rows.device()
            )));
        }
        if let Some(row_mask) = row_mask {
            if row_mask.dtype() != DType::Bool || row_mask.device() != device {
                return Err(TensorError::Device(format!(
                    "CUDA sparse-row AdamW expected bool row_mask on {device:?}, got {:?} on {:?}",
                    row_mask.dtype(),
                    row_mask.device()
                )));
            }
            if row_mask.shape() != vec![dims.rows] {
                return Err(TensorError::Shape(format!(
                    "CUDA sparse-row AdamW expected row_mask shape [{}], got {:?}",
                    dims.rows,
                    row_mask.shape()
                )));
            }
        }
        if selected_rows.numel() != dims.selected_rows {
            return Err(TensorError::Shape(format!(
                "CUDA sparse-row AdamW selected_rows length {} does not match dims {:?}",
                selected_rows.numel(),
                dims
            )));
        }
        if shape != vec![dims.rows, dims.row_dim] {
            return Err(TensorError::Shape(format!(
                "CUDA sparse-row AdamW expected parameter shape [{}, {}], got {:?}",
                dims.rows, dims.row_dim, shape
            )));
        }
        if !is_contiguous(&shape, &strides) || offset != 0 || storage.len() != numel(&shape) {
            return Err(TensorError::Device(format!(
                "CUDA sparse-row AdamW requires a contiguous full-storage parameter; shape={shape:?} strides={strides:?} offset={offset} storage_len={}",
                storage.len()
            )));
        }
        if grad.len() != storage.len() {
            return Err(TensorError::Shape(format!(
                "CUDA sparse-row AdamW gradient length {} does not match tensor shape {:?}",
                grad.len(),
                shape
            )));
        }

        let param = storage.cuda_buffer(DType::F32, device)?;
        let selected_rows_buffer =
            selected_rows.cuda_i64_full_storage_buffer("CUDA sparse-row AdamW selected rows")?;
        let row_mask_buffer = row_mask
            .map(|mask| mask.cuda_bool_full_storage_buffer("CUDA sparse-row AdamW row mask"))
            .transpose()?;
        let compact_grad = heirloom_kernels::cuda::memory_gather_selected_rows_f32_i64(
            &grad,
            &selected_rows_buffer,
            heirloom_kernels::cuda::SelectedRowsDims {
                selected_rows: dims.selected_rows,
                rows: dims.rows,
                row_dim: dims.row_dim,
            },
        )
        .map_err(cuda_error)?;
        heirloom_kernels::cuda::memory_sparse_adamw_compact_rows_f32_i64_buffers(
            heirloom_kernels::cuda::SparseAdamWCompactRowsBuffers {
                param: &param,
                compact_grad: &compact_grad,
                m,
                v,
                selected_rows: &selected_rows_buffer,
                row_mask: row_mask_buffer.as_ref(),
            },
            dims,
            adamw,
        )
        .map_err(cuda_error)?;
        storage.bump_version();
        Ok(true)
    }

    pub(crate) fn apply_adamw_cuda_sparse_rows_compact_grad_f32(
        &self,
        selected_rows: &Self,
        compact_grad_rows: &Self,
        m: &CudaBuffer,
        v: &CudaBuffer,
        dims: heirloom_kernels::cuda::SparseAdamWRowsDims,
        adamw: heirloom_kernels::cuda::AdamWParams,
    ) -> Result<bool> {
        let (storage, shape, strides, offset, dtype, device) = {
            let inner = self.inner.borrow();
            (
                Rc::clone(&inner.storage),
                inner.shape.clone(),
                inner.strides.clone(),
                inner.storage_offset,
                inner.dtype,
                inner.device,
            )
        };
        if dtype != DType::F32 {
            return Err(TensorError::DType(format!(
                "CUDA compact sparse-row AdamW currently supports f32 parameters only, got {dtype:?}"
            )));
        }
        if !matches!(device, Device::Cuda(_)) {
            return Err(TensorError::Device(format!(
                "CUDA compact sparse-row AdamW expected CUDA parameter, got {device:?}"
            )));
        }
        if selected_rows.dtype() != DType::I64 || selected_rows.device() != device {
            return Err(TensorError::Device(format!(
                "CUDA compact sparse-row AdamW expected i64 selected rows on {device:?}, got {:?} on {:?}",
                selected_rows.dtype(),
                selected_rows.device()
            )));
        }
        if compact_grad_rows.dtype() != DType::F32 || compact_grad_rows.device() != device {
            return Err(TensorError::Device(format!(
                "CUDA compact sparse-row AdamW expected f32 compact gradients on {device:?}, got {:?} on {:?}",
                compact_grad_rows.dtype(),
                compact_grad_rows.device()
            )));
        }
        if selected_rows.numel() != dims.selected_rows {
            return Err(TensorError::Shape(format!(
                "CUDA compact sparse-row AdamW selected_rows length {} does not match dims {:?}",
                selected_rows.numel(),
                dims
            )));
        }
        if compact_grad_rows.shape() != vec![dims.selected_rows, dims.row_dim] {
            return Err(TensorError::Shape(format!(
                "CUDA compact sparse-row AdamW expected compact gradient shape [{}, {}], got {:?}",
                dims.selected_rows,
                dims.row_dim,
                compact_grad_rows.shape()
            )));
        }
        if shape != vec![dims.rows, dims.row_dim] {
            return Err(TensorError::Shape(format!(
                "CUDA compact sparse-row AdamW expected parameter shape [{}, {}], got {:?}",
                dims.rows, dims.row_dim, shape
            )));
        }
        if !is_contiguous(&shape, &strides) || offset != 0 || storage.len() != numel(&shape) {
            return Err(TensorError::Device(format!(
                "CUDA compact sparse-row AdamW requires a contiguous full-storage parameter; shape={shape:?} strides={strides:?} offset={offset} storage_len={}",
                storage.len()
            )));
        }

        let param = storage.cuda_buffer(DType::F32, device)?;
        let selected_rows_buffer = selected_rows
            .cuda_i64_full_storage_buffer("CUDA compact sparse-row AdamW selected rows")?;
        let compact_grad_buffer = compact_grad_rows
            .cuda_f32_full_storage_buffer("CUDA compact sparse-row AdamW compact gradients")?;
        heirloom_kernels::cuda::memory_sparse_adamw_compact_rows_f32_i64_buffers(
            heirloom_kernels::cuda::SparseAdamWCompactRowsBuffers {
                param: &param,
                compact_grad: &compact_grad_buffer,
                m,
                v,
                selected_rows: &selected_rows_buffer,
                row_mask: None,
            },
            dims,
            adamw,
        )
        .map_err(cuda_error)?;
        storage.bump_version();
        Ok(true)
    }

    fn apply_gradient_update(&self, grad: &[f64], scale: f64) -> Result<()> {
        if grad.len() != self.numel() {
            return Err(TensorError::Shape(format!(
                "gradient update length {} does not match tensor shape {:?}",
                grad.len(),
                self.shape()
            )));
        }
        if self.has_internal_overlap() {
            return Err(TensorError::InvalidOperation(
                "cannot update tensor with internal overlap in-place".to_string(),
            ));
        }
        if self.device() != Device::Cpu {
            return Err(TensorError::Device(format!(
                "gradient update on {:?} requires CUDA optimizer kernels, which are not implemented yet",
                self.device()
            )));
        }

        let (storage, shape, strides, offset) = self.layout_snapshot();
        let mut grad_index = 0;
        storage.with_data_mut(|data| -> Result<()> {
            for_each_index(&shape, |index| {
                let storage_index = logical_offset(index, &strides, offset);
                match data {
                    StorageData::F32(values) => {
                        values[storage_index] += (scale * grad[grad_index]) as f32;
                    }
                    StorageData::BF16(values) => {
                        let updated = bf16_bits_to_f32(values[storage_index]) as f64
                            + scale * grad[grad_index];
                        values[storage_index] = f32_to_bf16_bits(updated as f32);
                    }
                    StorageData::F64(values) => {
                        values[storage_index] += scale * grad[grad_index];
                    }
                    StorageData::I64(_) | StorageData::Bool(_) | StorageData::Cuda { .. } => {
                        unreachable!("non-floating tensor passed apply_gradient_update");
                    }
                }
                grad_index += 1;
            });
            Ok(())
        })?;
        Ok(())
    }

    pub fn add_(&self, other: &Self) -> Result<()> {
        if self.has_internal_overlap() {
            return Err(TensorError::InvalidOperation(
                "cannot write in-place to tensor with internal overlap".to_string(),
            ));
        }
        ensure_same_dtype_device(self, other)?;
        let resolved = dispatch::resolve_binary(
            BinaryOp::Add,
            TensorMeta::new(self.dtype(), self.device()),
            TensorMeta::new(other.dtype(), other.device()),
        )?;
        if resolved.output_dtype != self.dtype() {
            return Err(TensorError::DType(format!(
                "in-place add would require dtype promotion from {:?} to {:?}",
                self.dtype(),
                resolved.output_dtype
            )));
        }
        let output_shape = broadcast_shapes(&self.shape(), &other.shape())?;
        if output_shape != self.shape() {
            return Err(TensorError::Shape(format!(
                "in-place add cannot change shape {:?} with rhs shape {:?}",
                self.shape(),
                other.shape()
            )));
        }

        let rhs = other.data_f64();
        let rhs_shape = other.shape();
        let (storage, shape, strides, offset) = self.layout_snapshot();
        storage.with_data_mut(|data| {
            for_each_index(&shape, |index| {
                let left_storage_index = logical_offset(index, &strides, offset);
                let rhs_index = broadcast_flat_index(index, &shape, &rhs_shape);
                let updated = data.value_as_f64(left_storage_index) + rhs[rhs_index];
                data.set_from_f64(left_storage_index, updated);
            });
        });
        Ok(())
    }

    pub fn view(&self, new_shape: &[usize]) -> Result<Self> {
        checked_numel(new_shape)?;
        if numel(new_shape) != self.numel() {
            return Err(TensorError::Shape(format!(
                "cannot view shape {:?} as {:?}",
                self.shape(),
                new_shape
            )));
        }
        if !self.is_contiguous() {
            return Err(TensorError::InvalidOperation(
                "view only supports contiguous tensors; use reshape for copy fallback".to_string(),
            ));
        }

        let (storage, offset, dtype, device, requires_grad) = {
            let inner = self.inner.borrow();
            (
                Rc::clone(&inner.storage),
                inner.storage_offset,
                inner.dtype,
                inner.device,
                should_track_grad(inner.requires_grad),
            )
        };
        let grad_fn = requires_grad.then(|| GradFn::View {
            input: self.clone(),
        });

        Ok(Self::from_parts(
            storage,
            new_shape.to_vec(),
            contiguous_strides(new_shape),
            offset,
            dtype,
            device,
            requires_grad,
            grad_fn,
        ))
    }

    pub fn expand(&self, output_shape: &[usize]) -> Result<Self> {
        checked_numel(output_shape)?;
        let (storage, input_shape, input_strides, offset, dtype, device, requires_grad) = {
            let inner = self.inner.borrow();
            (
                Rc::clone(&inner.storage),
                inner.shape.clone(),
                inner.strides.clone(),
                inner.storage_offset,
                inner.dtype,
                inner.device,
                should_track_grad(inner.requires_grad),
            )
        };
        let output_strides =
            crate::shape::expand_strides(&input_shape, &input_strides, output_shape)?;
        let grad_fn = requires_grad.then(|| GradFn::Expand {
            input: self.clone(),
            output_shape: output_shape.to_vec(),
        });

        Ok(Self::from_parts(
            storage,
            output_shape.to_vec(),
            output_strides,
            offset,
            dtype,
            device,
            requires_grad,
            grad_fn,
        ))
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        checked_numel(new_shape)?;
        if numel(new_shape) != self.numel() {
            return Err(TensorError::Shape(format!(
                "cannot reshape {:?} as {:?}",
                self.shape(),
                new_shape
            )));
        }
        if self.is_contiguous() {
            self.view(new_shape)
        } else {
            self.contiguous()?.view(new_shape)
        }
    }

    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        if self.device() != Device::Cpu {
            return Err(TensorError::Device(format!(
                "contiguous for {:?} requires a device copy kernel, which is not implemented yet",
                self.device()
            )));
        }

        let data = self.data_f64();
        let shape = self.shape();
        let dtype = self.dtype();
        let requires_grad = should_track_grad(self.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::Copy {
            input: self.clone(),
        });
        Self::from_f64_values_with_dtype(data, &shape, dtype, requires_grad, grad_fn)
    }

    pub fn as_strided(
        &self,
        shape: &[usize],
        strides: &[usize],
        storage_offset: usize,
    ) -> Result<Self> {
        checked_numel(shape)?;
        if shape.len() != strides.len() {
            return Err(TensorError::Shape(format!(
                "as_strided shape {:?} and strides {:?} must have same rank",
                shape, strides
            )));
        }
        if self.requires_grad() && should_track_grad(true) {
            return Err(TensorError::Autograd(
                "differentiable as_strided is not implemented; overlapping alias semantics are hard"
                    .to_string(),
            ));
        }

        let (storage, dtype, device) = {
            let inner = self.inner.borrow();
            (Rc::clone(&inner.storage), inner.dtype, inner.device)
        };
        if numel(shape) > 0
            && checked_max_storage_offset(shape, strides, storage_offset)? >= storage.len()
        {
            return Err(TensorError::Shape(format!(
                "as_strided view shape {:?}, strides {:?}, offset {} exceeds storage length {}",
                shape,
                strides,
                storage_offset,
                storage.len()
            )));
        }

        Ok(Self::from_parts(
            storage,
            shape.to_vec(),
            strides.to_vec(),
            storage_offset,
            dtype,
            device,
            false,
            None,
        ))
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let shape = self.shape();
        if dim >= shape.len() {
            return Err(TensorError::Shape(format!(
                "narrow dim {dim} out of range for shape {:?}",
                shape
            )));
        }
        if start.checked_add(len).is_none_or(|end| end > shape[dim]) {
            return Err(TensorError::Shape(format!(
                "narrow start {start} len {len} out of range for dim {} of shape {:?}",
                dim, shape
            )));
        }

        let (storage, mut new_shape, strides, base_offset, dtype, device, requires_grad) = {
            let inner = self.inner.borrow();
            (
                Rc::clone(&inner.storage),
                inner.shape.clone(),
                inner.strides.clone(),
                inner.storage_offset,
                inner.dtype,
                inner.device,
                should_track_grad(inner.requires_grad),
            )
        };
        let offset_delta = start.checked_mul(strides[dim]).ok_or_else(|| {
            TensorError::Shape(format!(
                "narrow offset overflows for start {start} and stride {}",
                strides[dim]
            ))
        })?;
        let offset = base_offset.checked_add(offset_delta).ok_or_else(|| {
            TensorError::Shape(format!(
                "narrow storage offset overflows: base {base_offset}, delta {offset_delta}"
            ))
        })?;
        new_shape[dim] = len;
        let grad_fn = requires_grad.then(|| GradFn::Narrow {
            input: self.clone(),
            dim,
            start,
        });

        Ok(Self::from_parts(
            storage,
            new_shape,
            strides,
            offset,
            dtype,
            device,
            requires_grad,
            grad_fn,
        ))
    }

    pub fn permute(&self, dims: &[usize]) -> Result<Self> {
        let shape = self.shape();
        validate_permutation(shape.len(), dims)?;

        let (storage, strides, offset, dtype, device, requires_grad) = {
            let inner = self.inner.borrow();
            (
                Rc::clone(&inner.storage),
                inner.strides.clone(),
                inner.storage_offset,
                inner.dtype,
                inner.device,
                should_track_grad(inner.requires_grad),
            )
        };
        let new_shape = dims.iter().map(|&dim| shape[dim]).collect::<Vec<_>>();
        let new_strides = dims.iter().map(|&dim| strides[dim]).collect::<Vec<_>>();
        let grad_fn = requires_grad.then(|| GradFn::Permute {
            input: self.clone(),
            dims: dims.to_vec(),
        });

        Ok(Self::from_parts(
            storage,
            new_shape,
            new_strides,
            offset,
            dtype,
            device,
            requires_grad,
            grad_fn,
        ))
    }

    pub fn transpose(&self) -> Result<Self> {
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(TensorError::Shape(format!(
                "transpose currently supports only rank-2 tensors, got shape {:?}",
                shape
            )));
        }
        self.permute(&[1, 0])
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        self.binary_op(other, BinaryOp::Add)
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        self.binary_op(other, BinaryOp::Sub)
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        self.binary_op(other, BinaryOp::Mul)
    }

    pub fn div(&self, other: &Self) -> Result<Self> {
        self.binary_op(other, BinaryOp::Div)
    }

    fn binary_op(&self, other: &Self, op: BinaryOp) -> Result<Self> {
        if self.device() != Device::Cpu || other.device() != Device::Cpu {
            return self.cuda_binary_op(other, op);
        }

        let resolved = dispatch::resolve_binary(
            op,
            TensorMeta::new(self.dtype(), self.device()),
            TensorMeta::new(other.dtype(), other.device()),
        )?;
        debug_assert_eq!(resolved.schema.operator, op.operator());
        ensure_fresh_output(resolved.schema)?;
        let left_data = self.data_f64();
        let right_data = other.data_f64();
        let left_shape = self.shape();
        let right_shape = other.shape();
        let output = (resolved.kernel)(op, &left_data, &left_shape, &right_data, &right_shape)?;

        let requires_grad = should_record_grad(
            resolved.schema,
            resolved.output_dtype,
            self.requires_grad() || other.requires_grad(),
        );
        let grad_fn = requires_grad.then(|| GradFn::Binary {
            op,
            left: SavedTensor::new(self, "binary left operand"),
            right: SavedTensor::new(other, "binary right operand"),
            output_shape: output.shape.clone(),
        });
        Self::from_f64_values_with_dtype(
            output.data,
            &output.shape,
            resolved.output_dtype,
            requires_grad,
            grad_fn,
        )
    }

    fn cuda_binary_op(&self, other: &Self, op: BinaryOp) -> Result<Self> {
        ensure_same_dtype_device(self, other)?;
        if self.dtype() != DType::F32 {
            return Err(TensorError::DType(format!(
                "CUDA binary ops currently support only f32 tensors, got {:?}",
                self.dtype()
            )));
        }
        let device = self.device();
        let Device::Cuda(device_id) = device else {
            unreachable!("cuda_binary_op called only for CUDA tensors");
        };
        let left_shape = self.shape();
        let right_shape = other.shape();
        let (shape, output) = if left_shape == right_shape {
            let left = self.cuda_f32_full_storage_buffer("CUDA binary left")?;
            let right = other.cuda_f32_full_storage_buffer("CUDA binary right")?;
            let output = match op {
                BinaryOp::Add => heirloom_kernels::cuda::add_f32_buffers(&left, &right),
                BinaryOp::Sub => heirloom_kernels::cuda::sub_f32_buffers(&left, &right),
                BinaryOp::Mul => heirloom_kernels::cuda::mul_f32_buffers(&left, &right),
                BinaryOp::Div => heirloom_kernels::cuda::div_f32_buffers(&left, &right),
            }
            .map_err(cuda_error)?;
            (left_shape, output)
        } else if op == BinaryOp::Add
            && left_shape.len() == 2
            && right_shape.len() == 1
            && left_shape[1] == right_shape[0]
        {
            let left = self.cuda_f32_full_storage_buffer("CUDA bias-add matrix")?;
            let right = other.cuda_f32_full_storage_buffer("CUDA bias-add bias")?;
            let output = heirloom_kernels::cuda::bias_add_f32_buffers(
                &left,
                &right,
                left_shape[0],
                left_shape[1],
            )
            .map_err(cuda_error)?;
            (left_shape, output)
        } else {
            return Err(TensorError::Shape(format!(
                "CUDA binary ops currently support equal shapes or add bias broadcast [rows, cols] + [cols], got {:?} and {:?}",
                left_shape, right_shape
            )));
        };
        let requires_grad = should_track_grad(self.requires_grad() || other.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::Binary {
            op,
            left: SavedTensor::new(self, "CUDA binary left operand"),
            right: SavedTensor::new(other, "CUDA binary right operand"),
            output_shape: shape.clone(),
        });
        Self::from_storage(
            Storage::new_cuda_from_buffer(device_id, DType::F32, self.numel(), output)?,
            &shape,
            DType::F32,
            device,
            requires_grad,
            grad_fn,
        )
    }

    /// Multiplies rank-2 tensors after validating shapes, layouts, devices, and
    /// all matrix-size/stride conversions used by the selected kernel.
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        if self.device() != Device::Cpu || other.device() != Device::Cpu {
            return self.cuda_matmul(other);
        }

        let resolved = dispatch::resolve_matmul(
            TensorMeta::new(self.dtype(), self.device()),
            TensorMeta::new(other.dtype(), other.device()),
        )?;
        debug_assert_eq!(resolved.schema.operator, Operator::Matmul);
        ensure_fresh_output(resolved.schema)?;
        let left_data = self.data_f64();
        let right_data = other.data_f64();
        let left_shape = self.shape();
        let right_shape = other.shape();
        let output = (resolved.kernel)(&left_data, &left_shape, &right_data, &right_shape)?;

        let requires_grad = should_record_grad(
            resolved.schema,
            resolved.output_dtype,
            self.requires_grad() || other.requires_grad(),
        );
        let grad_fn = requires_grad.then(|| GradFn::Matmul {
            left: SavedTensor::new(self, "matmul left operand"),
            right: SavedTensor::new(other, "matmul right operand"),
        });
        Self::from_f64_values_with_dtype(
            output.data,
            &output.shape,
            resolved.output_dtype,
            requires_grad,
            grad_fn,
        )
    }

    fn cuda_matmul(&self, other: &Self) -> Result<Self> {
        ensure_same_dtype_device(self, other)?;
        if self.dtype() != DType::F32 {
            return Err(TensorError::DType(format!(
                "CUDA matmul currently supports only f32 tensors, got {:?}",
                self.dtype()
            )));
        }
        let left_shape = self.shape();
        let right_shape = other.shape();
        if left_shape.len() != 2 || right_shape.len() != 2 {
            return Err(TensorError::Shape(format!(
                "CUDA matmul currently supports only rank-2 tensors, got {:?} and {:?}",
                left_shape, right_shape
            )));
        }
        let (m, k) = (left_shape[0], left_shape[1]);
        let (right_k, n) = (right_shape[0], right_shape[1]);
        if k != right_k {
            return Err(TensorError::Shape(format!(
                "matmul shape mismatch: left {:?} right {:?}",
                left_shape, right_shape
            )));
        }

        let device = self.device();
        let Device::Cuda(device_id) = device else {
            unreachable!("cuda_matmul called only for CUDA tensors");
        };
        let (left, left_layout) = self.cuda_matrix_layout("CUDA matmul left")?;
        let (right, right_layout) = other.cuda_matrix_layout("CUDA matmul right")?;
        let output = heirloom_kernels::cuda::matmul_strided_f32_buffers(
            &left,
            &right,
            heirloom_kernels::cuda::MatmulStridedDims {
                left: left_layout,
                right: right_layout,
            },
        )
        .map_err(cuda_error)?;
        let output_shape = vec![m, n];
        let requires_grad = should_track_grad(self.requires_grad() || other.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::Matmul {
            left: SavedTensor::new(self, "CUDA matmul left operand"),
            right: SavedTensor::new(other, "CUDA matmul right operand"),
        });
        Self::from_storage(
            Storage::new_cuda_from_buffer(device_id, DType::F32, numel(&output_shape), output)?,
            &output_shape,
            DType::F32,
            device,
            requires_grad,
            grad_fn,
        )
    }

    pub(crate) fn matmul_bf16_tensor_core_rhs_t(&self, rhs: &Self) -> Result<Self> {
        ensure_same_dtype_device(self, rhs)?;
        if self.dtype() != DType::BFloat16 {
            return Err(TensorError::DType(format!(
                "Tensor Core BF16 matmul expects bf16 CUDA tensors, got {:?}",
                self.dtype()
            )));
        }
        let left_shape = self.shape();
        let right_shape = rhs.shape();
        if left_shape.len() != 2 || right_shape.len() != 2 {
            return Err(TensorError::Shape(format!(
                "Tensor Core BF16 RHS^T matmul expects rank-2 tensors, got {:?} and {:?}",
                left_shape, right_shape
            )));
        }
        let (m, k) = (left_shape[0], left_shape[1]);
        let (n, right_k) = (right_shape[0], right_shape[1]);
        if k != right_k {
            return Err(TensorError::Shape(format!(
                "Tensor Core BF16 RHS^T matmul shape mismatch: left {:?} right {:?}",
                left_shape, right_shape
            )));
        }
        if !heirloom_kernels::cuda::bf16_tensor_core_matmul_shape_supported(m, k, n) {
            return Err(TensorError::Shape(format!(
                "Tensor Core BF16 RHS^T matmul requires non-zero dims, got M={m} K={k} N={n}"
            )));
        }

        let device = self.device();
        let Device::Cuda(device_id) = device else {
            return Err(TensorError::Device(format!(
                "Tensor Core BF16 matmul requires CUDA tensors, got {device:?}"
            )));
        };
        let left = self.cuda_bf16_full_storage_buffer("Tensor Core BF16 matmul left")?;
        let right = rhs.cuda_bf16_full_storage_buffer("Tensor Core BF16 matmul RHS^T")?;
        let output = heirloom_kernels::cuda::matmul_bf16_tensor_core_rhs_t_f32_buffers(
            &left, &right, m, k, n,
        )
        .map_err(cuda_error)?;
        let output_shape = vec![m, n];
        let requires_grad = should_track_grad(self.requires_grad() || rhs.requires_grad());
        let grad_fn = if requires_grad {
            let rhs_t = rhs.transpose()?;
            Some(GradFn::Matmul {
                left: SavedTensor::new(self, "Tensor Core BF16 matmul left operand"),
                right: SavedTensor::new(&rhs_t, "Tensor Core BF16 matmul RHS^T operand"),
            })
        } else {
            None
        };
        Self::from_storage(
            Storage::new_cuda_from_buffer(device_id, DType::F32, numel(&output_shape), output)?,
            &output_shape,
            DType::F32,
            device,
            requires_grad,
            grad_fn,
        )
    }

    pub(crate) fn matmul_bf16_tensor_core_rhs_t_bias(
        &self,
        rhs: &Self,
        bias: &Self,
    ) -> Result<Self> {
        if self.device() != rhs.device() || self.device() != bias.device() {
            return Err(TensorError::Device(format!(
                "Tensor Core BF16 fused matmul+bias expects matching devices, got {:?}, {:?}, {:?}",
                self.device(),
                rhs.device(),
                bias.device()
            )));
        }
        if self.dtype() != DType::BFloat16 || rhs.dtype() != DType::BFloat16 {
            return Err(TensorError::DType(format!(
                "Tensor Core BF16 fused matmul+bias expects bf16 left/RHS tensors, got {:?} and {:?}",
                self.dtype(),
                rhs.dtype()
            )));
        }
        if bias.dtype() != DType::F32 {
            return Err(TensorError::DType(format!(
                "Tensor Core BF16 fused matmul+bias expects f32 bias, got {:?}",
                bias.dtype()
            )));
        }
        let left_shape = self.shape();
        let right_shape = rhs.shape();
        let bias_shape = bias.shape();
        if left_shape.len() != 2 || right_shape.len() != 2 {
            return Err(TensorError::Shape(format!(
                "Tensor Core BF16 fused matmul+bias expects rank-2 tensors, got {:?} and {:?}",
                left_shape, right_shape
            )));
        }
        let (m, k) = (left_shape[0], left_shape[1]);
        let (n, right_k) = (right_shape[0], right_shape[1]);
        if k != right_k {
            return Err(TensorError::Shape(format!(
                "Tensor Core BF16 fused matmul+bias shape mismatch: left {:?} right {:?}",
                left_shape, right_shape
            )));
        }
        if bias_shape != [n] {
            return Err(TensorError::Shape(format!(
                "Tensor Core BF16 fused matmul+bias expected bias shape [{n}], got {:?}",
                bias_shape
            )));
        }
        if !heirloom_kernels::cuda::bf16_tensor_core_matmul_exact_tile_shape_supported(m, k, n) {
            return Err(TensorError::Shape(format!(
                "Tensor Core BF16 fused matmul+bias requires exact tile shapes M%16=0 K%16=0 N%8=0, got M={m} K={k} N={n}"
            )));
        }

        let device = self.device();
        let Device::Cuda(device_id) = device else {
            return Err(TensorError::Device(format!(
                "Tensor Core BF16 fused matmul+bias requires CUDA tensors, got {device:?}"
            )));
        };
        let left = self.cuda_bf16_full_storage_buffer("Tensor Core BF16 fused matmul+bias left")?;
        let right =
            rhs.cuda_bf16_full_storage_buffer("Tensor Core BF16 fused matmul+bias RHS^T")?;
        let bias_buffer =
            bias.cuda_f32_full_storage_buffer("Tensor Core BF16 fused matmul+bias bias")?;
        let output = heirloom_kernels::cuda::matmul_bf16_tensor_core_rhs_t_bias_f32_buffers(
            &left,
            &right,
            &bias_buffer,
            m,
            k,
            n,
        )
        .map_err(cuda_error)?;
        let output_shape = vec![m, n];
        let requires_grad =
            should_track_grad(self.requires_grad() || rhs.requires_grad() || bias.requires_grad());
        let grad_fn = if requires_grad {
            let rhs_t = rhs.transpose()?;
            Some(GradFn::MatmulBias {
                left: SavedTensor::new(self, "Tensor Core BF16 fused matmul+bias left operand"),
                right: SavedTensor::new(&rhs_t, "Tensor Core BF16 fused matmul+bias RHS^T operand"),
                bias: SavedTensor::new(bias, "Tensor Core BF16 fused matmul+bias bias"),
            })
        } else {
            None
        };
        Self::from_storage(
            Storage::new_cuda_from_buffer(device_id, DType::F32, numel(&output_shape), output)?,
            &output_shape,
            DType::F32,
            device,
            requires_grad,
            grad_fn,
        )
    }

    pub fn relu(&self) -> Result<Self> {
        if self.device() != Device::Cpu {
            return self.cuda_relu();
        }

        let resolved =
            dispatch::resolve_unary(Operator::Relu, TensorMeta::new(self.dtype(), self.device()))?;
        debug_assert_eq!(resolved.schema.operator, Operator::Relu);
        ensure_fresh_output(resolved.schema)?;
        let shape = self.shape();
        let input = self.data_f64();
        let kernel = resolved
            .kernel
            .expect("relu schema must resolve to a unary kernel");
        let output = kernel(&input, &shape);
        let requires_grad =
            should_record_grad(resolved.schema, resolved.output_dtype, self.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::Relu {
            input: SavedTensor::new(self, "relu input"),
        });
        Self::from_f64_values_with_dtype(
            output.data,
            &output.shape,
            resolved.output_dtype,
            requires_grad,
            grad_fn,
        )
    }

    fn cuda_relu(&self) -> Result<Self> {
        if self.dtype() != DType::F32 {
            return Err(TensorError::DType(format!(
                "CUDA relu currently supports only f32 tensors, got {:?}",
                self.dtype()
            )));
        }
        let device = self.device();
        let Device::Cuda(device_id) = device else {
            unreachable!("cuda_relu called only for CUDA tensors");
        };
        let input = self.cuda_f32_full_storage_buffer("CUDA relu input")?;
        let output = heirloom_kernels::cuda::relu_f32_buffer(&input).map_err(cuda_error)?;
        let shape = self.shape();
        let requires_grad = should_track_grad(self.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::Relu {
            input: SavedTensor::new(self, "CUDA relu input"),
        });
        Self::from_storage(
            Storage::new_cuda_from_buffer(device_id, DType::F32, self.numel(), output)?,
            &shape,
            DType::F32,
            device,
            requires_grad,
            grad_fn,
        )
    }

    pub fn gelu(&self) -> Result<Self> {
        if self.device() != Device::Cpu {
            return self.cuda_gelu();
        }

        let resolved =
            dispatch::resolve_unary(Operator::Gelu, TensorMeta::new(self.dtype(), self.device()))?;
        debug_assert_eq!(resolved.schema.operator, Operator::Gelu);
        ensure_fresh_output(resolved.schema)?;
        let shape = self.shape();
        let output = self
            .data_f64()
            .into_iter()
            .map(gelu_value)
            .collect::<Vec<_>>();
        let requires_grad =
            should_record_grad(resolved.schema, resolved.output_dtype, self.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::Gelu {
            input: SavedTensor::new(self, "gelu input"),
        });
        Self::from_f64_values_with_dtype(
            output,
            &shape,
            resolved.output_dtype,
            requires_grad,
            grad_fn,
        )
    }

    fn cuda_gelu(&self) -> Result<Self> {
        if self.dtype() != DType::F32 {
            return Err(TensorError::DType(format!(
                "CUDA gelu currently supports only f32 tensors, got {:?}",
                self.dtype()
            )));
        }
        let device = self.device();
        let Device::Cuda(device_id) = device else {
            unreachable!("cuda_gelu called only for CUDA tensors");
        };
        let input = self.cuda_f32_full_storage_buffer("CUDA gelu input")?;
        let output = heirloom_kernels::cuda::gelu_f32_buffer(&input).map_err(cuda_error)?;
        let shape = self.shape();
        let requires_grad = should_track_grad(self.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::Gelu {
            input: SavedTensor::new(self, "CUDA gelu input"),
        });
        Self::from_storage(
            Storage::new_cuda_from_buffer(device_id, DType::F32, self.numel(), output)?,
            &shape,
            DType::F32,
            device,
            requires_grad,
            grad_fn,
        )
    }

    pub fn layer_norm_last_dim(&self, weight: &Self, bias: &Self, eps: f64) -> Result<Self> {
        if self.device() != Device::Cpu
            || weight.device() != Device::Cpu
            || bias.device() != Device::Cpu
        {
            return self.cuda_layer_norm_last_dim(weight, bias, eps);
        }

        let resolved = dispatch::resolve_layer_norm_last_dim(
            TensorMeta::new(self.dtype(), self.device()),
            TensorMeta::new(weight.dtype(), weight.device()),
            TensorMeta::new(bias.dtype(), bias.device()),
        )?;
        debug_assert_eq!(resolved.schema.operator, Operator::LayerNormLastDim);
        ensure_fresh_output(resolved.schema)?;
        if eps <= 0.0 {
            return Err(TensorError::InvalidOperation(format!(
                "layer_norm_last_dim requires positive eps, got {eps}"
            )));
        }
        let shape = self.shape();
        let Some(&features) = shape.last() else {
            return Err(TensorError::Shape(
                "layer_norm_last_dim requires rank >= 1 input".to_string(),
            ));
        };
        if features == 0 {
            return Err(TensorError::InvalidOperation(
                "layer_norm_last_dim requires features > 0".to_string(),
            ));
        }
        if weight.shape() != vec![features] || bias.shape() != vec![features] {
            return Err(TensorError::Shape(format!(
                "layer_norm_last_dim expected weight/bias shape [{features}], got {:?} and {:?}",
                weight.shape(),
                bias.shape()
            )));
        }

        let rows = self.numel() / features;
        let input = self.data_f64();
        let weight_data = weight.data_f64();
        let bias_data = bias.data_f64();
        let mut output = vec![0.0; input.len()];
        for row in 0..rows {
            let start = row * features;
            let row_values = &input[start..start + features];
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
            for col in 0..features {
                output[start + col] =
                    (row_values[col] - mean) * rstd * weight_data[col] + bias_data[col];
            }
        }

        let requires_grad = should_record_grad(
            resolved.schema,
            resolved.output_dtype,
            self.requires_grad() || weight.requires_grad() || bias.requires_grad(),
        );
        let grad_fn = requires_grad.then(|| GradFn::LayerNormLastDim {
            input: SavedTensor::new(self, "layer_norm input"),
            weight: SavedTensor::new(weight, "layer_norm weight"),
            bias: SavedTensor::new(bias, "layer_norm bias"),
            eps,
        });
        Self::from_f64_values_with_dtype(
            output,
            &shape,
            resolved.output_dtype,
            requires_grad,
            grad_fn,
        )
    }

    fn cuda_layer_norm_last_dim(&self, weight: &Self, bias: &Self, eps: f64) -> Result<Self> {
        if eps <= 0.0 || !eps.is_finite() {
            return Err(TensorError::InvalidOperation(format!(
                "layer_norm_last_dim requires finite positive eps, got {eps}"
            )));
        }
        if self.device() != weight.device() || self.device() != bias.device() {
            return Err(TensorError::Device(format!(
                "CUDA layer_norm_last_dim expected matching CUDA devices, got {:?}, {:?}, {:?}",
                self.device(),
                weight.device(),
                bias.device()
            )));
        }
        let device = self.device();
        let Device::Cuda(device_id) = device else {
            return Err(TensorError::Device(format!(
                "CUDA layer_norm_last_dim requires CUDA tensors, got {device:?}"
            )));
        };
        if self.dtype() != DType::F32 || weight.dtype() != DType::F32 || bias.dtype() != DType::F32
        {
            return Err(TensorError::DType(format!(
                "CUDA layer_norm_last_dim currently supports only f32 tensors, got {:?}, {:?}, {:?}",
                self.dtype(),
                weight.dtype(),
                bias.dtype()
            )));
        }
        let shape = self.shape();
        let Some(&features) = shape.last() else {
            return Err(TensorError::Shape(
                "layer_norm_last_dim requires rank >= 1 input".to_string(),
            ));
        };
        if features == 0 {
            return Err(TensorError::InvalidOperation(
                "CUDA layer_norm_last_dim requires features > 0".to_string(),
            ));
        }
        if weight.shape() != vec![features] || bias.shape() != vec![features] {
            return Err(TensorError::Shape(format!(
                "layer_norm_last_dim expected weight/bias shape [{features}], got {:?} and {:?}",
                weight.shape(),
                bias.shape()
            )));
        }
        let rows = self.numel() / features;
        let input = self.cuda_f32_full_storage_buffer("CUDA layer_norm input")?;
        let weight_buffer = weight.cuda_f32_full_storage_buffer("CUDA layer_norm weight")?;
        let bias_buffer = bias.cuda_f32_full_storage_buffer("CUDA layer_norm bias")?;
        let output = heirloom_kernels::cuda::layer_norm_forward_f32_buffers(
            &input,
            &weight_buffer,
            &bias_buffer,
            rows,
            features,
            eps as f32,
        )
        .map_err(cuda_error)?;
        let requires_grad = should_track_grad(
            self.requires_grad() || weight.requires_grad() || bias.requires_grad(),
        );
        let grad_fn = requires_grad.then(|| GradFn::LayerNormLastDim {
            input: SavedTensor::new(self, "CUDA layer_norm input"),
            weight: SavedTensor::new(weight, "CUDA layer_norm weight"),
            bias: SavedTensor::new(bias, "CUDA layer_norm bias"),
            eps,
        });
        Self::from_storage(
            Storage::new_cuda_from_buffer(device_id, DType::F32, self.numel(), output)?,
            &shape,
            DType::F32,
            device,
            requires_grad,
            grad_fn,
        )
    }

    pub fn embedding(&self, weight: &Self) -> Result<Self> {
        if self.device() != Device::Cpu || weight.device() != Device::Cpu {
            return self.cuda_embedding(weight);
        }

        let resolved = dispatch::resolve_embedding(
            TensorMeta::new(self.dtype(), self.device()),
            TensorMeta::new(weight.dtype(), weight.device()),
        )?;
        debug_assert_eq!(resolved.schema.operator, Operator::Embedding);
        ensure_fresh_output(resolved.schema)?;
        let weight_shape = weight.shape();
        if weight_shape.len() != 2 {
            return Err(TensorError::Shape(format!(
                "embedding weight must have shape [vocab, dim], got {:?}",
                weight_shape
            )));
        }
        let (vocab_size, embedding_dim) = (weight_shape[0], weight_shape[1]);
        let indices = self
            .data_i64()?
            .into_iter()
            .map(|index| {
                if index < 0 || index as usize >= vocab_size {
                    Err(TensorError::Shape(format!(
                        "embedding index {index} out of range for vocab size {vocab_size}"
                    )))
                } else {
                    Ok(index as usize)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let weight_data = weight.data_f64();
        let mut output = Vec::with_capacity(indices.len() * embedding_dim);
        for &index in &indices {
            let start = index * embedding_dim;
            output.extend_from_slice(&weight_data[start..start + embedding_dim]);
        }

        let mut output_shape = self.shape();
        output_shape.push(embedding_dim);
        let requires_grad = should_record_grad(
            resolved.schema,
            resolved.output_dtype,
            weight.requires_grad(),
        );
        let grad_fn = requires_grad.then(|| GradFn::Embedding {
            weight: SavedTensor::new(weight, "embedding weight"),
            indices: EmbeddingIndices::Cpu(indices),
            embedding_dim,
        });
        Self::from_f64_values_with_dtype(
            output,
            &output_shape,
            resolved.output_dtype,
            requires_grad,
            grad_fn,
        )
    }

    fn cuda_embedding(&self, weight: &Self) -> Result<Self> {
        if self.device() != weight.device() {
            return Err(TensorError::Device(format!(
                "CUDA embedding expected indices and weight on the same device, got {:?} and {:?}",
                self.device(),
                weight.device()
            )));
        }
        let device = self.device();
        let Device::Cuda(device_id) = device else {
            return Err(TensorError::Device(format!(
                "CUDA embedding requires CUDA tensors, got {device:?}"
            )));
        };
        if self.dtype() != DType::I64 {
            return Err(TensorError::DType(format!(
                "CUDA embedding indices must be i64, got {:?}",
                self.dtype()
            )));
        }
        if weight.dtype() != DType::F32 {
            return Err(TensorError::DType(format!(
                "CUDA embedding currently supports only f32 weights, got {:?}",
                weight.dtype()
            )));
        }

        let weight_shape = weight.shape();
        if weight_shape.len() != 2 {
            return Err(TensorError::Shape(format!(
                "embedding weight must have shape [vocab, dim], got {:?}",
                weight_shape
            )));
        }
        let (vocab_size, embedding_dim) = (weight_shape[0], weight_shape[1]);
        if embedding_dim == 0 {
            return Err(TensorError::Shape(
                "CUDA embedding currently requires embedding_dim > 0".to_string(),
            ));
        }
        let indices = self.cuda_i64_full_storage_buffer("CUDA embedding indices")?;
        let weight_buffer = weight.cuda_f32_full_storage_buffer("CUDA embedding weight")?;
        let output = heirloom_kernels::cuda::embedding_f32_i64_buffers(
            &indices,
            &weight_buffer,
            vocab_size,
            embedding_dim,
        )
        .map_err(cuda_error)?;

        let mut output_shape = self.shape();
        output_shape.push(embedding_dim);
        let requires_grad = should_track_grad(weight.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::Embedding {
            weight: SavedTensor::new(weight, "CUDA embedding weight"),
            indices: EmbeddingIndices::Cuda(SavedTensor::new(self, "CUDA embedding indices")),
            embedding_dim,
        });
        Self::from_storage(
            Storage::new_cuda_from_buffer(device_id, DType::F32, numel(&output_shape), output)?,
            &output_shape,
            DType::F32,
            device,
            requires_grad,
            grad_fn,
        )
    }

    pub fn memory_selected_scores(&self, query: &Self, keys: &Self) -> Result<Self> {
        if self.device() != query.device() || self.device() != keys.device() {
            return Err(TensorError::Device(format!(
                "memory_selected_scores expected indices, query, and keys on the same device, got {:?}, {:?}, {:?}",
                self.device(),
                query.device(),
                keys.device()
            )));
        }
        let index_shape = self.shape();
        let query_shape = query.shape();
        let key_shape = keys.shape();
        if index_shape.len() != 2 || query_shape.len() != 2 || key_shape.len() != 2 {
            return Err(TensorError::Shape(format!(
                "memory_selected_scores expected indices [tokens, top_k], query [tokens, key_dim], keys [slots, key_dim], got indices={index_shape:?} query={query_shape:?} keys={key_shape:?}"
            )));
        }
        if query_shape[0] != index_shape[0] || key_shape[1] != query_shape[1] {
            return Err(TensorError::Shape(format!(
                "memory_selected_scores expected matching tokens/key_dim, got indices={index_shape:?} query={query_shape:?} keys={key_shape:?}"
            )));
        }
        if self.dtype() != DType::I64 || query.dtype() != DType::F32 || keys.dtype() != DType::F32 {
            return Err(TensorError::DType(format!(
                "memory_selected_scores expected i64 indices and f32 query/keys, got {:?}, {:?}, {:?}",
                self.dtype(),
                query.dtype(),
                keys.dtype()
            )));
        }
        let dims = heirloom_kernels::cuda::MemorySelectedScoreDims {
            tokens: index_shape[0],
            top_k: index_shape[1],
            slots: key_shape[0],
            key_dim: query_shape[1],
        };
        if self.device() != Device::Cpu {
            return self.cuda_memory_selected_scores(query, keys, dims);
        }

        let indices = self
            .data_i64()?
            .into_iter()
            .map(|index| {
                if index < 0 || index as usize >= dims.slots {
                    Err(TensorError::Shape(format!(
                        "memory_selected_scores index {index} out of range for slots {}",
                        dims.slots
                    )))
                } else {
                    Ok(index as usize)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let query_data = query.data_f64();
        let key_data = keys.data_f64();
        let mut output = vec![0.0; dims.tokens * dims.top_k];
        for token in 0..dims.tokens {
            for selected in 0..dims.top_k {
                let index_pos = token * dims.top_k + selected;
                let row = indices[index_pos];
                let mut score = 0.0;
                for dim in 0..dims.key_dim {
                    score +=
                        query_data[token * dims.key_dim + dim] * key_data[row * dims.key_dim + dim];
                }
                output[index_pos] = score;
            }
        }
        let requires_grad = should_track_grad(query.requires_grad() || keys.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::MemorySelectedScores {
            indices: EmbeddingIndices::Cpu(indices),
            query: SavedTensor::new(query, "memory selected scores query"),
            keys: SavedTensor::new(keys, "memory selected scores key table"),
            dims,
        });
        Self::from_f64_values_with_dtype(
            output,
            &[dims.tokens, dims.top_k],
            DType::F32,
            requires_grad,
            grad_fn,
        )
    }

    fn cuda_memory_selected_scores(
        &self,
        query: &Self,
        keys: &Self,
        dims: heirloom_kernels::cuda::MemorySelectedScoreDims,
    ) -> Result<Self> {
        let device = self.device();
        let Device::Cuda(device_id) = device else {
            return Err(TensorError::Device(format!(
                "CUDA memory_selected_scores requires CUDA tensors, got {device:?}"
            )));
        };
        let indices = self.cuda_i64_full_storage_buffer("CUDA memory selected scores indices")?;
        let query_buffer =
            query.cuda_f32_full_storage_buffer("CUDA memory selected scores query")?;
        let key_buffer = keys.cuda_f32_full_storage_buffer("CUDA memory selected scores keys")?;
        let output = heirloom_kernels::cuda::memory_selected_scores_forward_f32_i64_buffers(
            &indices,
            &query_buffer,
            &key_buffer,
            dims,
        )
        .map_err(cuda_error)?;
        let requires_grad = should_track_grad(query.requires_grad() || keys.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::MemorySelectedScores {
            indices: EmbeddingIndices::Cuda(SavedTensor::new(
                self,
                "CUDA memory selected scores indices",
            )),
            query: SavedTensor::new(query, "CUDA memory selected scores query"),
            keys: SavedTensor::new(keys, "CUDA memory selected scores key table"),
            dims,
        });
        Self::from_storage(
            Storage::new_cuda_from_buffer(device_id, DType::F32, dims.tokens * dims.top_k, output)?,
            &[dims.tokens, dims.top_k],
            DType::F32,
            device,
            requires_grad,
            grad_fn,
        )
    }

    pub(crate) fn memory_product_key_selected_scores(
        &self,
        query: &Self,
        left_keys: &Self,
        right_keys: &Self,
    ) -> Result<Self> {
        if self.device() != query.device()
            || self.device() != left_keys.device()
            || self.device() != right_keys.device()
        {
            return Err(TensorError::Device(format!(
                "memory_product_key_selected_scores expected indices/query/half-keys on the same device, got {:?}, {:?}, {:?}, {:?}",
                self.device(),
                query.device(),
                left_keys.device(),
                right_keys.device()
            )));
        }
        let index_shape = self.shape();
        let query_shape = query.shape();
        let left_shape = left_keys.shape();
        let right_shape = right_keys.shape();
        if index_shape.len() != 2
            || query_shape.len() != 2
            || left_shape.len() != 2
            || right_shape.len() != 2
        {
            return Err(TensorError::Shape(format!(
                "memory_product_key_selected_scores expected indices [tokens, top_k], query [tokens, key_dim], and half-keys [side, half_dim], got indices={index_shape:?} query={query_shape:?} left={left_shape:?} right={right_shape:?}"
            )));
        }
        let expected_key_dim = left_shape[1].checked_mul(2).ok_or_else(|| {
            TensorError::Shape(format!(
                "memory_product_key_selected_scores half-key dimension {} cannot be doubled",
                left_shape[1]
            ))
        })?;
        if query_shape[0] != index_shape[0]
            || left_shape != right_shape
            || query_shape[1] != expected_key_dim
        {
            return Err(TensorError::Shape(format!(
                "memory_product_key_selected_scores expected matching tokens and key_dim=2*half_dim, got indices={index_shape:?} query={query_shape:?} left={left_shape:?} right={right_shape:?}"
            )));
        }
        if self.dtype() != DType::I64
            || query.dtype() != DType::F32
            || left_keys.dtype() != DType::F32
            || right_keys.dtype() != DType::F32
        {
            return Err(TensorError::DType(format!(
                "memory_product_key_selected_scores expected i64 indices and f32 query/half-keys, got {:?}, {:?}, {:?}, {:?}",
                self.dtype(),
                query.dtype(),
                left_keys.dtype(),
                right_keys.dtype()
            )));
        }
        let dims = heirloom_kernels::cuda::MemoryProductKeySelectedScoreDims {
            tokens: index_shape[0],
            top_k: index_shape[1],
            side: left_shape[0],
            key_dim: query_shape[1],
        };
        if self.device() != Device::Cpu {
            return self
                .cuda_memory_product_key_selected_scores(query, left_keys, right_keys, dims);
        }

        let slots = dims.side.checked_mul(dims.side).ok_or_else(|| {
            TensorError::Shape("product-key selected scores side^2 overflow".to_string())
        })?;
        let indices = self
            .data_i64()?
            .into_iter()
            .map(|index| {
                if index < 0 || index as usize >= slots {
                    Err(TensorError::Shape(format!(
                        "memory_product_key_selected_scores index {index} out of range for slots {slots}"
                    )))
                } else {
                    Ok(index as usize)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let query_data = query.data_f64();
        let left_data = left_keys.data_f64();
        let right_data = right_keys.data_f64();
        let half_dim = dims.key_dim / 2;
        let mut output = vec![0.0; dims.tokens * dims.top_k];
        for token in 0..dims.tokens {
            for selected in 0..dims.top_k {
                let index_pos = token * dims.top_k + selected;
                let slot = indices[index_pos];
                let left = slot / dims.side;
                let right = slot % dims.side;
                let mut score = 0.0;
                for dim in 0..half_dim {
                    score +=
                        query_data[token * dims.key_dim + dim] * left_data[left * half_dim + dim];
                    score += query_data[token * dims.key_dim + half_dim + dim]
                        * right_data[right * half_dim + dim];
                }
                output[index_pos] = score;
            }
        }
        let requires_grad = should_track_grad(
            query.requires_grad() || left_keys.requires_grad() || right_keys.requires_grad(),
        );
        let grad_fn = requires_grad.then(|| GradFn::MemoryProductKeySelectedScores {
            indices: EmbeddingIndices::Cpu(indices),
            query: SavedTensor::new(query, "product-key selected scores query"),
            left_keys: SavedTensor::new(left_keys, "product-key selected scores left key table"),
            right_keys: SavedTensor::new(right_keys, "product-key selected scores right key table"),
            dims,
        });
        Self::from_f64_values_with_dtype(
            output,
            &[dims.tokens, dims.top_k],
            DType::F32,
            requires_grad,
            grad_fn,
        )
    }

    fn cuda_memory_product_key_selected_scores(
        &self,
        query: &Self,
        left_keys: &Self,
        right_keys: &Self,
        dims: heirloom_kernels::cuda::MemoryProductKeySelectedScoreDims,
    ) -> Result<Self> {
        let device = self.device();
        let Device::Cuda(device_id) = device else {
            return Err(TensorError::Device(format!(
                "CUDA memory_product_key_selected_scores requires CUDA tensors, got {device:?}"
            )));
        };
        let indices =
            self.cuda_i64_full_storage_buffer("CUDA product-key selected scores indices")?;
        let query_buffer =
            query.cuda_f32_full_storage_buffer("CUDA product-key selected scores query")?;
        let left_buffer =
            left_keys.cuda_f32_full_storage_buffer("CUDA product-key selected scores left keys")?;
        let right_buffer = right_keys
            .cuda_f32_full_storage_buffer("CUDA product-key selected scores right keys")?;
        let output =
            heirloom_kernels::cuda::memory_product_key_selected_scores_forward_f32_i64_buffers(
                &indices,
                &query_buffer,
                &left_buffer,
                &right_buffer,
                dims,
            )
            .map_err(cuda_error)?;
        let requires_grad = should_track_grad(
            query.requires_grad() || left_keys.requires_grad() || right_keys.requires_grad(),
        );
        let grad_fn = requires_grad.then(|| GradFn::MemoryProductKeySelectedScores {
            indices: EmbeddingIndices::Cuda(SavedTensor::new(
                self,
                "CUDA product-key selected scores indices",
            )),
            query: SavedTensor::new(query, "CUDA product-key selected scores query"),
            left_keys: SavedTensor::new(left_keys, "CUDA product-key selected scores left keys"),
            right_keys: SavedTensor::new(right_keys, "CUDA product-key selected scores right keys"),
            dims,
        });
        Self::from_storage(
            Storage::new_cuda_from_buffer(device_id, DType::F32, dims.tokens * dims.top_k, output)?,
            &[dims.tokens, dims.top_k],
            DType::F32,
            device,
            requires_grad,
            grad_fn,
        )
    }

    pub fn memory_weighted_value(&self, weights: &Self, values: &Self) -> Result<Self> {
        if self.device() != weights.device() || self.device() != values.device() {
            return Err(TensorError::Device(format!(
                "memory_weighted_value expected indices, weights, and values on the same device, got {:?}, {:?}, {:?}",
                self.device(),
                weights.device(),
                values.device()
            )));
        }
        let index_shape = self.shape();
        let weight_shape = weights.shape();
        let value_shape = values.shape();
        if index_shape.len() != 2 || weight_shape != index_shape || value_shape.len() != 2 {
            return Err(TensorError::Shape(format!(
                "memory_weighted_value expected indices/weights [tokens, top_k] and values [slots, value_dim], got indices={index_shape:?} weights={weight_shape:?} values={value_shape:?}"
            )));
        }
        if self.dtype() != DType::I64
            || weights.dtype() != DType::F32
            || values.dtype() != DType::F32
        {
            return Err(TensorError::DType(format!(
                "memory_weighted_value expected i64 indices and f32 weights/values, got {:?}, {:?}, {:?}",
                self.dtype(),
                weights.dtype(),
                values.dtype()
            )));
        }
        let dims = heirloom_kernels::cuda::MemoryWeightedValueDims {
            tokens: index_shape[0],
            top_k: index_shape[1],
            slots: value_shape[0],
            value_dim: value_shape[1],
        };
        if self.device() != Device::Cpu {
            return self.cuda_memory_weighted_value(weights, values, dims);
        }

        let indices = self
            .data_i64()?
            .into_iter()
            .map(|index| {
                if index < 0 || index as usize >= dims.slots {
                    Err(TensorError::Shape(format!(
                        "memory_weighted_value index {index} out of range for slots {}",
                        dims.slots
                    )))
                } else {
                    Ok(index as usize)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let weight_data = weights.data_f64();
        let value_data = values.data_f64();
        let mut output = vec![0.0; dims.tokens * dims.value_dim];
        for token in 0..dims.tokens {
            for selected in 0..dims.top_k {
                let index_pos = token * dims.top_k + selected;
                let row = indices[index_pos];
                let weight = weight_data[index_pos];
                for dim in 0..dims.value_dim {
                    output[token * dims.value_dim + dim] +=
                        weight * value_data[row * dims.value_dim + dim];
                }
            }
        }
        let requires_grad = should_track_grad(weights.requires_grad() || values.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::MemoryWeightedValue {
            indices: EmbeddingIndices::Cpu(indices),
            weights: SavedTensor::new(weights, "memory weighted value weights"),
            values: SavedTensor::new(values, "memory weighted value table"),
            dims,
        });
        Self::from_f64_values_with_dtype(
            output,
            &[dims.tokens, dims.value_dim],
            DType::F32,
            requires_grad,
            grad_fn,
        )
    }

    fn cuda_memory_weighted_value(
        &self,
        weights: &Self,
        values: &Self,
        dims: heirloom_kernels::cuda::MemoryWeightedValueDims,
    ) -> Result<Self> {
        let device = self.device();
        let Device::Cuda(device_id) = device else {
            return Err(TensorError::Device(format!(
                "CUDA memory_weighted_value requires CUDA tensors, got {device:?}"
            )));
        };
        let indices = self.cuda_i64_full_storage_buffer("CUDA memory weighted value indices")?;
        let weights_buffer =
            weights.cuda_f32_full_storage_buffer("CUDA memory weighted value weights")?;
        let values_buffer =
            values.cuda_f32_full_storage_buffer("CUDA memory weighted value values")?;
        let output = heirloom_kernels::cuda::memory_weighted_value_forward_f32_i64_buffers(
            &indices,
            &weights_buffer,
            &values_buffer,
            dims,
        )
        .map_err(cuda_error)?;
        let requires_grad = should_track_grad(weights.requires_grad() || values.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::MemoryWeightedValue {
            indices: EmbeddingIndices::Cuda(SavedTensor::new(
                self,
                "CUDA memory weighted value indices",
            )),
            weights: SavedTensor::new(weights, "CUDA memory weighted value weights"),
            values: SavedTensor::new(values, "CUDA memory weighted value table"),
            dims,
        });
        Self::from_storage(
            Storage::new_cuda_from_buffer(
                device_id,
                DType::F32,
                dims.tokens * dims.value_dim,
                output,
            )?,
            &[dims.tokens, dims.value_dim],
            DType::F32,
            device,
            requires_grad,
            grad_fn,
        )
    }

    pub fn masked_fill(&self, mask: &Self, value: f64) -> Result<Self> {
        let resolved = dispatch::resolve_masked_fill(
            TensorMeta::new(self.dtype(), self.device()),
            TensorMeta::new(mask.dtype(), mask.device()),
        )?;
        debug_assert_eq!(resolved.schema.operator, Operator::MaskedFill);
        ensure_fresh_output(resolved.schema)?;
        if self.shape() != mask.shape() {
            return Err(TensorError::Shape(format!(
                "masked_fill currently requires identical shapes, got {:?} and {:?}",
                self.shape(),
                mask.shape()
            )));
        }
        let mask_data = mask.data_bool()?;
        let mut output = self.data_f64();
        for (item, masked) in output.iter_mut().zip(mask_data.iter()) {
            if *masked {
                *item = value;
            }
        }
        let shape = self.shape();
        let requires_grad =
            should_record_grad(resolved.schema, resolved.output_dtype, self.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::MaskedFill {
            input: self.clone(),
            mask: mask_data,
        });
        Self::from_f64_values_with_dtype(
            output,
            &shape,
            resolved.output_dtype,
            requires_grad,
            grad_fn,
        )
    }

    pub fn argmax_last_dim(&self) -> Result<Self> {
        let resolved =
            dispatch::resolve_argmax_last_dim(TensorMeta::new(self.dtype(), self.device()))?;
        debug_assert_eq!(resolved.schema.operator, Operator::ArgmaxLastDim);
        ensure_fresh_output(resolved.schema)?;
        let shape = self.shape();
        let Some(&classes) = shape.last() else {
            return Err(TensorError::Shape(
                "argmax_last_dim requires rank >= 1 input".to_string(),
            ));
        };
        if classes == 0 {
            return Err(TensorError::InvalidOperation(
                "argmax_last_dim over empty last dimension is undefined".to_string(),
            ));
        }
        let rows = self.numel() / classes;
        let data = self.data_f64();
        let mut output = Vec::with_capacity(rows);
        for row in 0..rows {
            let start = row * classes;
            let mut best_index = 0usize;
            let mut best_value = data[start];
            for class in 1..classes {
                let value = data[start + class];
                if value > best_value {
                    best_value = value;
                    best_index = class;
                }
            }
            output.push(best_index as i64);
        }
        let output_shape = shape[..shape.len() - 1].to_vec();
        debug_assert!(!should_record_grad(
            resolved.schema,
            resolved.output_dtype,
            self.requires_grad()
        ));
        Self::from_i64(output, &output_shape, false)
    }

    pub fn causal_self_attention(&self, key: &Self, value: &Self, n_heads: usize) -> Result<Self> {
        if self.device() != Device::Cpu
            || key.device() != Device::Cpu
            || value.device() != Device::Cpu
        {
            return self.cuda_causal_self_attention(key, value, n_heads);
        }

        let resolved = dispatch::resolve_causal_self_attention(
            TensorMeta::new(self.dtype(), self.device()),
            TensorMeta::new(key.dtype(), key.device()),
            TensorMeta::new(value.dtype(), value.device()),
        )?;
        debug_assert_eq!(resolved.schema.operator, Operator::CausalSelfAttention);
        ensure_fresh_output(resolved.schema)?;
        if n_heads == 0 {
            return Err(TensorError::InvalidOperation(
                "causal_self_attention requires n_heads > 0".to_string(),
            ));
        }
        let shape = self.shape();
        if shape.len() != 3 || key.shape() != shape || value.shape() != shape {
            return Err(TensorError::Shape(format!(
                "causal_self_attention expects query/key/value shape [batch, time, channels], got {:?}, {:?}, {:?}",
                shape,
                key.shape(),
                value.shape()
            )));
        }
        let (batch, time, channels) = (shape[0], shape[1], shape[2]);
        if channels % n_heads != 0 {
            return Err(TensorError::Shape(format!(
                "channels {channels} must be divisible by n_heads {n_heads}"
            )));
        }
        let head_dim = channels / n_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let query = self.data_f64();
        let key_data = key.data_f64();
        let value_data = value.data_f64();
        let attention_len = batch
            .checked_mul(n_heads)
            .and_then(|value| value.checked_mul(time))
            .and_then(|value| value.checked_mul(time))
            .ok_or_else(|| {
                TensorError::Shape(format!(
                    "causal attention score size overflows: batch={batch}, heads={n_heads}, time={time}"
                ))
            })?;
        let mut attention = Vec::new();
        attention
            .try_reserve_exact(attention_len)
            .map_err(|error| {
                TensorError::Shape(format!(
                "causal attention score allocation for {attention_len} elements failed: {error}"
            ))
            })?;
        attention.resize(attention_len, 0.0);
        let mut output = vec![0.0; self.numel()];

        for b in 0..batch {
            for h in 0..n_heads {
                for t in 0..time {
                    let mut max_score = f64::NEG_INFINITY;
                    for s in 0..=t {
                        let mut score = 0.0;
                        for d in 0..head_dim {
                            score += query[qkv_index(b, t, h, d, time, channels, head_dim)]
                                * key_data[qkv_index(b, s, h, d, time, channels, head_dim)];
                        }
                        max_score = max_score.max(score * scale);
                    }

                    let mut denom = 0.0;
                    for s in 0..=t {
                        let mut score = 0.0;
                        for d in 0..head_dim {
                            score += query[qkv_index(b, t, h, d, time, channels, head_dim)]
                                * key_data[qkv_index(b, s, h, d, time, channels, head_dim)];
                        }
                        let weight = (score * scale - max_score).exp();
                        attention[attention_index(b, h, t, s, n_heads, time)] = weight;
                        denom += weight;
                    }

                    for s in 0..=t {
                        let attn_index = attention_index(b, h, t, s, n_heads, time);
                        attention[attn_index] /= denom;
                        for d in 0..head_dim {
                            output[qkv_index(b, t, h, d, time, channels, head_dim)] += attention
                                [attn_index]
                                * value_data[qkv_index(b, s, h, d, time, channels, head_dim)];
                        }
                    }
                }
            }
        }

        let requires_grad = should_record_grad(
            resolved.schema,
            resolved.output_dtype,
            self.requires_grad() || key.requires_grad() || value.requires_grad(),
        );
        let grad_fn = requires_grad.then(|| GradFn::CausalSelfAttention {
            query: SavedTensor::new(self, "causal attention query"),
            key: SavedTensor::new(key, "causal attention key"),
            value: SavedTensor::new(value, "causal attention value"),
            attention: CausalAttentionWeights::Cpu(attention),
            n_heads,
            amp_bf16_tensor_core: false,
        });
        Self::from_f64_values_with_dtype(
            output,
            &shape,
            resolved.output_dtype,
            requires_grad,
            grad_fn,
        )
    }

    pub fn causal_self_attention_amp_bf16_tensor_core(
        &self,
        key: &Self,
        value: &Self,
        n_heads: usize,
    ) -> Result<Self> {
        ensure_same_dtype_device(self, key)?;
        ensure_same_dtype_device(self, value)?;
        if self.dtype() != DType::F32 {
            return Err(TensorError::DType(format!(
                "CUDA AMP Tensor Core causal_self_attention currently supports only f32 query/key/value tensors, got {:?}",
                self.dtype()
            )));
        }
        if n_heads == 0 {
            return Err(TensorError::InvalidOperation(
                "causal_self_attention requires n_heads > 0".to_string(),
            ));
        }
        let shape = self.shape();
        if shape.len() != 3 || key.shape() != shape || value.shape() != shape {
            return Err(TensorError::Shape(format!(
                "CUDA AMP Tensor Core causal_self_attention expects query/key/value shape [batch, time, channels], got {:?}, {:?}, {:?}",
                shape,
                key.shape(),
                value.shape()
            )));
        }
        let (batch, time, channels) = (shape[0], shape[1], shape[2]);
        if channels % n_heads != 0 {
            return Err(TensorError::Shape(format!(
                "channels {channels} must be divisible by n_heads {n_heads}"
            )));
        }

        let device = self.device();
        let Device::Cuda(device_id) = device else {
            return Err(TensorError::Device(format!(
                "CUDA AMP Tensor Core causal_self_attention requires CUDA tensors, got {device:?}"
            )));
        };
        let dims = heirloom_kernels::cuda::CausalAttentionDims {
            batch,
            time,
            channels,
            n_heads,
        };
        if !heirloom_kernels::cuda::causal_attention_bf16_tensor_core_shape_supported(dims) {
            return Err(TensorError::Shape(format!(
                "CUDA AMP Tensor Core causal_self_attention requires valid non-zero time/head_dim, got time={} head_dim={}",
                time,
                channels / n_heads
            )));
        }

        let query_buffer =
            self.cuda_f32_full_storage_buffer("CUDA AMP Tensor Core causal attention query")?;
        let key_buffer =
            key.cuda_f32_full_storage_buffer("CUDA AMP Tensor Core causal attention key")?;
        let value_buffer =
            value.cuda_f32_full_storage_buffer("CUDA AMP Tensor Core causal attention value")?;
        let requires_grad =
            should_track_grad(self.requires_grad() || key.requires_grad() || value.requires_grad());

        if heirloom_kernels::cuda::flash_bf16_attention_enabled() {
            if requires_grad {
                if heirloom_kernels::cuda::flash_bf16_tensor_core_attention_enabled()
                    && heirloom_kernels::cuda::flash_bf16_tensor_core_attention_backward_enabled()
                    && heirloom_kernels::cuda::causal_attention_bf16_flash_tensor_core_backward_shape_supported(dims)
                {
                    match heirloom_kernels::cuda::causal_attention_bf16_flash_tensor_core_forward_f32_buffers(
                        &query_buffer,
                        &key_buffer,
                        &value_buffer,
                        dims,
                    ) {
                        Ok((output, row_max, row_denom)) => {
                            let output_storage = Storage::new_cuda_from_buffer(
                                device_id,
                                DType::F32,
                                self.numel(),
                                output,
                            )?;
                            let row_shape = vec![batch, n_heads, time];
                            let row_len = numel(&row_shape);
                            let saved_output = Self::from_storage(
                                output_storage.clone(),
                                &shape,
                                DType::F32,
                                device,
                                false,
                                None,
                            )?;
                            let row_max_tensor = Self::from_storage(
                                Storage::new_cuda_from_buffer(
                                    device_id,
                                    DType::F32,
                                    row_len,
                                    row_max,
                                )?,
                                &row_shape,
                                DType::F32,
                                device,
                                false,
                                None,
                            )?;
                            let row_denom_tensor = Self::from_storage(
                                Storage::new_cuda_from_buffer(
                                    device_id,
                                    DType::F32,
                                    row_len,
                                    row_denom,
                                )?,
                                &row_shape,
                                DType::F32,
                                device,
                                false,
                                None,
                            )?;
                            let grad_fn = Some(GradFn::CausalSelfAttention {
                                query: SavedTensor::new(
                                    self,
                                    "CUDA AMP Tensor Core flash causal attention query",
                                ),
                                key: SavedTensor::new(
                                    key,
                                    "CUDA AMP Tensor Core flash causal attention key",
                                ),
                                value: SavedTensor::new(
                                    value,
                                    "CUDA AMP Tensor Core flash causal attention value",
                                ),
                                attention: CausalAttentionWeights::CudaFlash {
                                    output: SavedTensor::new(
                                        &saved_output,
                                        "CUDA AMP Tensor Core flash causal attention output",
                                    ),
                                    row_max: SavedTensor::new(
                                        &row_max_tensor,
                                        "CUDA AMP Tensor Core flash causal attention row_max",
                                    ),
                                    row_denom: SavedTensor::new(
                                        &row_denom_tensor,
                                        "CUDA AMP Tensor Core flash causal attention row_denom",
                                    ),
                                },
                                n_heads,
                                amp_bf16_tensor_core: true,
                            });
                            return Self::from_storage(
                                output_storage,
                                &shape,
                                DType::F32,
                                device,
                                true,
                                grad_fn,
                            );
                        }
                        Err(err) => {
                            heirloom_kernels::cuda::record_flash_bf16_attention_fallback(false);
                            heirloom_kernels::cuda::record_flash_bf16_tensor_core_attention_backward_fallback();
                            if heirloom_kernels::cuda::require_flash_bf16_attention_enabled() {
                                heirloom_kernels::cuda::record_flash_bf16_attention_hard_require_failure();
                                return Err(cuda_error(
                                    heirloom_kernels::cuda::flash_bf16_attention_required_error(
                                        &err.to_string(),
                                    ),
                                ));
                            }
                        }
                    }
                } else {
                    if heirloom_kernels::cuda::flash_bf16_tensor_core_attention_enabled() {
                        heirloom_kernels::cuda::record_flash_bf16_tensor_core_attention_request();
                        heirloom_kernels::cuda::record_flash_bf16_attention_fallback(false);
                        heirloom_kernels::cuda::record_flash_bf16_tensor_core_attention_backward_fallback();
                    } else {
                        heirloom_kernels::cuda::record_flash_bf16_attention_request();
                        heirloom_kernels::cuda::record_flash_bf16_attention_fallback(true);
                    }
                    if heirloom_kernels::cuda::require_flash_bf16_attention_enabled() {
                        heirloom_kernels::cuda::record_flash_bf16_attention_hard_require_failure();
                        return Err(cuda_error(
                            heirloom_kernels::cuda::flash_bf16_attention_required_error(
                                "autograd requires HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TENSOR_CORE=1 and HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_BACKWARD=1",
                            ),
                        ));
                    }
                }
            } else {
                if heirloom_kernels::cuda::flash_bf16_tensor_core_attention_enabled() {
                    match heirloom_kernels::cuda::causal_attention_bf16_flash_tensor_core_forward_f32_buffers(
                        &query_buffer,
                        &key_buffer,
                        &value_buffer,
                        dims,
                    ) {
                        Ok((output, _row_max, _row_denom)) => {
                            return Self::from_storage(
                                Storage::new_cuda_from_buffer(
                                    device_id,
                                    DType::F32,
                                    self.numel(),
                                    output,
                                )?,
                                &shape,
                                DType::F32,
                                device,
                                false,
                                None,
                            );
                        }
                        Err(err) => {
                            heirloom_kernels::cuda::record_flash_bf16_attention_fallback(false);
                            if heirloom_kernels::cuda::require_flash_bf16_attention_enabled() {
                                heirloom_kernels::cuda::record_flash_bf16_attention_hard_require_failure();
                                return Err(cuda_error(
                                    heirloom_kernels::cuda::flash_bf16_attention_required_error(
                                        &err.to_string(),
                                    ),
                                ));
                            }
                        }
                    }
                }
                match heirloom_kernels::cuda::causal_attention_bf16_flash_forward_f32_buffers(
                    &query_buffer,
                    &key_buffer,
                    &value_buffer,
                    dims,
                ) {
                    Ok(output) => {
                        return Self::from_storage(
                            Storage::new_cuda_from_buffer(
                                device_id,
                                DType::F32,
                                self.numel(),
                                output,
                            )?,
                            &shape,
                            DType::F32,
                            device,
                            false,
                            None,
                        );
                    }
                    Err(err) => {
                        heirloom_kernels::cuda::record_flash_bf16_attention_fallback(true);
                        if heirloom_kernels::cuda::require_flash_bf16_attention_enabled() {
                            heirloom_kernels::cuda::record_flash_bf16_attention_hard_require_failure();
                            return Err(cuda_error(
                                heirloom_kernels::cuda::flash_bf16_attention_required_error(
                                    &err.to_string(),
                                ),
                            ));
                        }
                    }
                }
            }
        }

        let (output, attention) =
            heirloom_kernels::cuda::causal_attention_bf16_tensor_core_forward_f32_buffers(
                &query_buffer,
                &key_buffer,
                &value_buffer,
                dims,
            )
            .map_err(cuda_error)?;
        let attention_shape = vec![batch, n_heads, time, time];
        let attention_tensor = Self::from_storage(
            Storage::new_cuda_from_buffer(
                device_id,
                DType::F32,
                numel(&attention_shape),
                attention,
            )?,
            &attention_shape,
            DType::F32,
            device,
            false,
            None,
        )?;
        let grad_fn = requires_grad.then(|| GradFn::CausalSelfAttention {
            query: SavedTensor::new(self, "CUDA AMP Tensor Core causal attention query"),
            key: SavedTensor::new(key, "CUDA AMP Tensor Core causal attention key"),
            value: SavedTensor::new(value, "CUDA AMP Tensor Core causal attention value"),
            attention: CausalAttentionWeights::Cuda(SavedTensor::new(
                &attention_tensor,
                "CUDA AMP Tensor Core causal attention weights",
            )),
            n_heads,
            amp_bf16_tensor_core: true,
        });
        Self::from_storage(
            Storage::new_cuda_from_buffer(device_id, DType::F32, self.numel(), output)?,
            &shape,
            DType::F32,
            device,
            requires_grad,
            grad_fn,
        )
    }

    fn cuda_causal_self_attention(&self, key: &Self, value: &Self, n_heads: usize) -> Result<Self> {
        ensure_same_dtype_device(self, key)?;
        ensure_same_dtype_device(self, value)?;
        if self.dtype() != DType::F32 {
            return Err(TensorError::DType(format!(
                "CUDA causal_self_attention currently supports only f32 tensors, got {:?}",
                self.dtype()
            )));
        }
        if n_heads == 0 {
            return Err(TensorError::InvalidOperation(
                "causal_self_attention requires n_heads > 0".to_string(),
            ));
        }
        let shape = self.shape();
        if shape.len() != 3 || key.shape() != shape || value.shape() != shape {
            return Err(TensorError::Shape(format!(
                "CUDA causal_self_attention expects query/key/value shape [batch, time, channels], got {:?}, {:?}, {:?}",
                shape,
                key.shape(),
                value.shape()
            )));
        }
        let (batch, time, channels) = (shape[0], shape[1], shape[2]);
        if batch == 0 || time == 0 || channels == 0 {
            return Err(TensorError::Shape(format!(
                "CUDA causal_self_attention requires non-empty batch/time/channels, got {shape:?}"
            )));
        }
        if channels % n_heads != 0 {
            return Err(TensorError::Shape(format!(
                "channels {channels} must be divisible by n_heads {n_heads}"
            )));
        }

        let device = self.device();
        let Device::Cuda(device_id) = device else {
            unreachable!("cuda_causal_self_attention called only for CUDA tensors");
        };
        let query_buffer = self.cuda_f32_full_storage_buffer("CUDA causal attention query")?;
        let key_buffer = key.cuda_f32_full_storage_buffer("CUDA causal attention key")?;
        let value_buffer = value.cuda_f32_full_storage_buffer("CUDA causal attention value")?;
        let dims = heirloom_kernels::cuda::CausalAttentionDims {
            batch,
            time,
            channels,
            n_heads,
        };
        let (output, attention) = heirloom_kernels::cuda::causal_attention_forward_f32_buffers(
            &query_buffer,
            &key_buffer,
            &value_buffer,
            dims,
        )
        .map_err(cuda_error)?;
        let attention_shape = vec![batch, n_heads, time, time];
        let attention_tensor = Self::from_storage(
            Storage::new_cuda_from_buffer(
                device_id,
                DType::F32,
                numel(&attention_shape),
                attention,
            )?,
            &attention_shape,
            DType::F32,
            device,
            false,
            None,
        )?;
        let requires_grad =
            should_track_grad(self.requires_grad() || key.requires_grad() || value.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::CausalSelfAttention {
            query: SavedTensor::new(self, "CUDA causal attention query"),
            key: SavedTensor::new(key, "CUDA causal attention key"),
            value: SavedTensor::new(value, "CUDA causal attention value"),
            attention: CausalAttentionWeights::Cuda(SavedTensor::new(
                &attention_tensor,
                "CUDA causal attention weights",
            )),
            n_heads,
            amp_bf16_tensor_core: false,
        });
        Self::from_storage(
            Storage::new_cuda_from_buffer(device_id, DType::F32, self.numel(), output)?,
            &shape,
            DType::F32,
            device,
            requires_grad,
            grad_fn,
        )
    }

    pub fn apply_custom_unary(&self, op: CustomUnaryOp) -> Result<Self> {
        ensure_floating_tensor(self, op.name())?;
        let shape = self.shape();
        let input = self.data_f64();
        let output = op.forward(&input, &shape, self.dtype(), self.device())?;
        if output.len() != self.numel() {
            return Err(TensorError::Shape(format!(
                "custom unary op {} returned {} elements for input shape {:?} with {} elements",
                op.name(),
                output.len(),
                shape,
                self.numel()
            )));
        }

        let requires_grad = should_track_grad(self.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::CustomUnary {
            op,
            input: SavedTensor::new(self, "custom unary input"),
            output_data: output.clone(),
            output_shape: shape.clone(),
        });
        Self::from_f64_values_with_dtype(output, &shape, self.dtype(), requires_grad, grad_fn)
    }

    pub fn softmax_dim(&self, dim: usize) -> Result<Self> {
        if self.device() != Device::Cpu {
            let shape = self.shape();
            validate_dim_for_rank(shape.len(), dim, "softmax")?;
            if shape[dim] == 0 {
                return Err(TensorError::InvalidOperation(format!(
                    "softmax over empty dim {dim} of shape {:?} is undefined",
                    shape
                )));
            }
            return self.cuda_softmax_dim(dim, self.dtype());
        }

        let resolved = dispatch::resolve_unary(
            Operator::SoftmaxDim,
            TensorMeta::new(self.dtype(), self.device()),
        )?;
        debug_assert_eq!(resolved.schema.operator, Operator::SoftmaxDim);
        ensure_fresh_output(resolved.schema)?;
        let shape = self.shape();
        validate_dim_for_rank(shape.len(), dim, "softmax")?;
        if shape[dim] == 0 {
            return Err(TensorError::InvalidOperation(format!(
                "softmax over empty dim {dim} of shape {:?} is undefined",
                shape
            )));
        }
        let output = softmax_data(&self.data_f64(), &shape, dim);
        let requires_grad =
            should_record_grad(resolved.schema, resolved.output_dtype, self.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::SoftmaxDim {
            output_data: output.clone(),
            output_shape: shape.clone(),
            dim,
            input: self.clone(),
        });
        Self::from_f64_values_with_dtype(
            output,
            &shape,
            resolved.output_dtype,
            requires_grad,
            grad_fn,
        )
    }

    fn cuda_softmax_dim(&self, dim: usize, output_dtype: DType) -> Result<Self> {
        if self.dtype() != DType::F32 || output_dtype != DType::F32 {
            return Err(TensorError::DType(format!(
                "CUDA softmax_dim currently supports f32 -> f32, got {:?} -> {:?}",
                self.dtype(),
                output_dtype
            )));
        }
        let shape = self.shape();
        if shape.len() != 2 || dim != 1 {
            return Err(TensorError::Device(format!(
                "CUDA softmax_dim currently supports rank-2 dim=1 only, got shape={shape:?} dim={dim}"
            )));
        }
        let device = self.device();
        let Device::Cuda(device_id) = device else {
            unreachable!("cuda_softmax_dim called only for CUDA tensors");
        };
        let input = self.cuda_f32_full_storage_buffer("CUDA softmax_dim input")?;
        let output = heirloom_kernels::cuda::softmax_dim1_f32_buffer(&input, shape[0], shape[1])
            .map_err(cuda_error)?;
        let requires_grad = should_track_grad(self.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::SoftmaxDim {
            output_data: Vec::new(),
            output_shape: shape.clone(),
            dim,
            input: self.clone(),
        });
        Self::from_storage(
            Storage::new_cuda_from_buffer(device_id, DType::F32, self.numel(), output)?,
            &shape,
            DType::F32,
            device,
            requires_grad,
            grad_fn,
        )
    }

    pub fn softmax_dim_signed(&self, dim: isize) -> Result<Self> {
        self.softmax_dim(normalize_dim(self.ndim(), dim)?)
    }

    pub fn sum(&self) -> Result<Self> {
        if self.device() != Device::Cpu {
            return self.cuda_reduce_all(Operator::Sum);
        }

        let resolved = dispatch::resolve_reduction(
            Operator::Sum,
            TensorMeta::new(self.dtype(), self.device()),
        )?;
        debug_assert_eq!(resolved.schema.operator, Operator::Sum);
        ensure_fresh_output(resolved.schema)?;
        let input = self.data_f64();
        let kernel = resolved
            .kernel
            .expect("sum schema must resolve to a reduction kernel");
        let output = kernel(&input)?;
        let requires_grad =
            should_record_grad(resolved.schema, resolved.output_dtype, self.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::Sum {
            input: self.clone(),
        });
        Self::from_f64_values_with_dtype(
            output.data,
            &output.shape,
            resolved.output_dtype,
            requires_grad,
            grad_fn,
        )
    }

    pub fn mean(&self) -> Result<Self> {
        if self.device() != Device::Cpu {
            return self.cuda_reduce_all(Operator::Mean);
        }

        let resolved = dispatch::resolve_reduction(
            Operator::Mean,
            TensorMeta::new(self.dtype(), self.device()),
        )?;
        debug_assert_eq!(resolved.schema.operator, Operator::Mean);
        ensure_fresh_output(resolved.schema)?;
        let input = self.data_f64();
        let kernel = resolved
            .kernel
            .expect("mean schema must resolve to a reduction kernel");
        let output = kernel(&input)?;
        let requires_grad =
            should_record_grad(resolved.schema, resolved.output_dtype, self.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::Mean {
            input: self.clone(),
        });
        Self::from_f64_values_with_dtype(
            output.data,
            &output.shape,
            resolved.output_dtype,
            requires_grad,
            grad_fn,
        )
    }

    fn cuda_reduce_all(&self, operator: Operator) -> Result<Self> {
        if self.dtype() != DType::F32 {
            return Err(TensorError::DType(format!(
                "CUDA reductions currently support only f32 tensors, got {:?}",
                self.dtype()
            )));
        }
        if operator == Operator::Mean && self.numel() == 0 {
            return Err(TensorError::InvalidOperation(
                "mean of an empty tensor is undefined in this prototype".to_string(),
            ));
        }

        let device = self.device();
        let Device::Cuda(device_id) = device else {
            unreachable!("cuda_reduce_all called only for CUDA tensors");
        };
        let input = self.cuda_f32_full_storage_buffer("CUDA reduction input")?;
        let output = match operator {
            Operator::Sum => heirloom_kernels::cuda::sum_f32_buffer(&input),
            Operator::Mean => heirloom_kernels::cuda::mean_f32_buffer(&input),
            other => {
                return Err(TensorError::InvalidOperation(format!(
                    "unsupported CUDA reduction operator {:?}",
                    other
                )))
            }
        }
        .map_err(cuda_error)?;
        let requires_grad = should_track_grad(self.requires_grad());
        let grad_fn = requires_grad.then(|| match operator {
            Operator::Sum => GradFn::Sum {
                input: self.clone(),
            },
            Operator::Mean => GradFn::Mean {
                input: self.clone(),
            },
            _ => unreachable!("CUDA reduction operator checked above"),
        });
        Self::from_storage(
            Storage::new_cuda_from_buffer(device_id, DType::F32, 1, output)?,
            &[],
            DType::F32,
            device,
            requires_grad,
            grad_fn,
        )
    }

    pub fn sum_dim(&self, dim: usize, keepdim: bool) -> Result<Self> {
        self.reduce_dim(dim, keepdim, ReduceDimKind::Sum)
    }

    pub fn mean_dim(&self, dim: usize, keepdim: bool) -> Result<Self> {
        self.reduce_dim(dim, keepdim, ReduceDimKind::Mean)
    }

    pub fn sum_dim_signed(&self, dim: isize, keepdim: bool) -> Result<Self> {
        self.reduce_dim(
            normalize_dim(self.ndim(), dim)?,
            keepdim,
            ReduceDimKind::Sum,
        )
    }

    pub fn mean_dim_signed(&self, dim: isize, keepdim: bool) -> Result<Self> {
        self.reduce_dim(
            normalize_dim(self.ndim(), dim)?,
            keepdim,
            ReduceDimKind::Mean,
        )
    }

    fn reduce_dim(&self, dim: usize, keepdim: bool, kind: ReduceDimKind) -> Result<Self> {
        let operator = match kind {
            ReduceDimKind::Sum => Operator::SumDim,
            ReduceDimKind::Mean => Operator::MeanDim,
        };
        let resolved =
            dispatch::resolve_reduction(operator, TensorMeta::new(self.dtype(), self.device()))?;
        debug_assert_eq!(resolved.schema.operator, operator);
        ensure_fresh_output(resolved.schema)?;
        let input_shape = self.shape();
        if dim >= input_shape.len() {
            return Err(TensorError::Shape(format!(
                "reduction dim {dim} out of range for shape {:?}",
                input_shape
            )));
        }
        if matches!(kind, ReduceDimKind::Mean) && input_shape[dim] == 0 {
            return Err(TensorError::InvalidOperation(format!(
                "mean over empty dim {dim} of shape {:?} is undefined",
                input_shape
            )));
        }

        let output_shape = reduced_shape(&input_shape, dim, keepdim);
        let mut output = vec![0.0; numel(&output_shape)];
        let input = self.data_f64();
        let mut input_flat = 0;
        for_each_index(&input_shape, |input_index| {
            let output_index = reduced_index(input_index, dim, keepdim);
            let output_flat = flatten_index(&output_index, &output_shape);
            output[output_flat] += input[input_flat];
            input_flat += 1;
        });
        if matches!(kind, ReduceDimKind::Mean) {
            let scale = input_shape[dim] as f64;
            for value in &mut output {
                *value /= scale;
            }
        }

        let requires_grad =
            should_record_grad(resolved.schema, resolved.output_dtype, self.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::ReduceDim {
            input: self.clone(),
            dim,
            keepdim,
            kind,
        });
        Self::from_f64_values_with_dtype(
            output,
            &output_shape,
            resolved.output_dtype,
            requires_grad,
            grad_fn,
        )
    }

    pub fn mse_loss(&self, target: &Self) -> Result<Self> {
        if self.shape() != target.shape() {
            return Err(TensorError::Shape(format!(
                "mse_loss requires identical shapes, got {:?} and {:?}",
                self.shape(),
                target.shape()
            )));
        }
        if self.numel() == 0 {
            return Err(TensorError::InvalidOperation(
                "mse_loss of an empty tensor is undefined".to_string(),
            ));
        }

        let diff = self.sub(target)?;
        diff.mul(&diff)?.mean()
    }

    pub fn cross_entropy_for_logits(&self, targets: &[usize]) -> Result<Self> {
        if self.device() != Device::Cpu {
            return self.cuda_cross_entropy_for_logits(targets);
        }

        let resolved = dispatch::resolve_unary(
            Operator::CrossEntropyForLogits,
            TensorMeta::new(self.dtype(), self.device()),
        )?;
        debug_assert_eq!(resolved.schema.operator, Operator::CrossEntropyForLogits);
        ensure_fresh_output(resolved.schema)?;
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(TensorError::Shape(format!(
                "cross_entropy_for_logits expects rank-2 logits [batch, classes], got {:?}",
                shape
            )));
        }
        let (batch, classes) = (shape[0], shape[1]);
        if batch == 0 || classes == 0 {
            return Err(TensorError::InvalidOperation(format!(
                "cross_entropy_for_logits requires non-empty batch/classes, got {:?}",
                shape
            )));
        }
        if targets.len() != batch {
            return Err(TensorError::Shape(format!(
                "target length {} must equal batch size {}",
                targets.len(),
                batch
            )));
        }
        for &target in targets {
            if target >= classes {
                return Err(TensorError::Shape(format!(
                    "target class {target} out of range for {classes} classes"
                )));
            }
        }

        let logits = self.data_f64();
        let mut total = 0.0;
        for row in 0..batch {
            let start = row * classes;
            let row_values = &logits[start..start + classes];
            let max = row_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp = row_values
                .iter()
                .map(|value| (*value - max).exp())
                .sum::<f64>();
            total += -row_values[targets[row]] + max + sum_exp.ln();
        }

        let requires_grad =
            should_record_grad(resolved.schema, resolved.output_dtype, self.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::CrossEntropy {
            logits: SavedTensor::new(self, "cross entropy logits"),
            targets: CrossEntropyTargets::Cpu(targets.to_vec()),
        });
        Self::from_f64_values_with_dtype(
            vec![total / batch as f64],
            &[],
            resolved.output_dtype,
            requires_grad,
            grad_fn,
        )
    }

    pub fn cross_entropy_for_logits_tensor(&self, targets: &Self) -> Result<Self> {
        if targets.dtype() != DType::I64 {
            return Err(TensorError::DType(format!(
                "cross_entropy_for_logits_tensor expected i64 targets, got {:?}",
                targets.dtype()
            )));
        }
        if self.device() != targets.device() {
            return Err(TensorError::Device(format!(
                "cross_entropy_for_logits_tensor requires logits and targets on the same device, got {:?} and {:?}",
                self.device(),
                targets.device()
            )));
        }
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(TensorError::Shape(format!(
                "cross_entropy_for_logits expects rank-2 logits [batch, classes], got {:?}",
                shape
            )));
        }
        let target_shape = targets.shape();
        if target_shape != vec![shape[0]] {
            return Err(TensorError::Shape(format!(
                "cross_entropy_for_logits_tensor expected target shape [{}], got {:?}",
                shape[0], target_shape
            )));
        }
        if self.device() != Device::Cpu {
            return self.cuda_cross_entropy_for_logits_tensor(targets);
        }

        let target_ids = targets
            .data_i64()?
            .into_iter()
            .map(|target| {
                if target < 0 {
                    Err(TensorError::Shape(format!(
                        "target class {target} out of range for {} classes",
                        shape[1]
                    )))
                } else {
                    Ok(target as usize)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        self.cross_entropy_for_logits(&target_ids)
    }

    fn cuda_cross_entropy_for_logits(&self, targets: &[usize]) -> Result<Self> {
        let device = self.device();
        let Device::Cuda(device_id) = device else {
            return Err(TensorError::Device(format!(
                "CUDA cross_entropy_for_logits requires CUDA logits, got {device:?}"
            )));
        };
        if self.dtype() != DType::F32 {
            return Err(TensorError::DType(format!(
                "CUDA cross_entropy_for_logits currently supports only f32 logits, got {:?}",
                self.dtype()
            )));
        }
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(TensorError::Shape(format!(
                "cross_entropy_for_logits expects rank-2 logits [batch, classes], got {:?}",
                shape
            )));
        }
        let (batch, classes) = (shape[0], shape[1]);
        if batch == 0 || classes == 0 {
            return Err(TensorError::InvalidOperation(format!(
                "cross_entropy_for_logits requires non-empty batch/classes, got {:?}",
                shape
            )));
        }
        if targets.len() != batch {
            return Err(TensorError::Shape(format!(
                "target length {} must equal batch size {}",
                targets.len(),
                batch
            )));
        }
        let target_ids = targets
            .iter()
            .map(|&target| {
                if target >= classes {
                    Err(TensorError::Shape(format!(
                        "target class {target} out of range for {classes} classes"
                    )))
                } else {
                    Ok(target as i64)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let target_buffer =
            CudaBuffer::from_i64(device_id as i32, &target_ids).map_err(cuda_error)?;
        let target_tensor = Self::from_storage(
            Storage::new_cuda_from_buffer(device_id, DType::I64, batch, target_buffer)?,
            &[batch],
            DType::I64,
            device,
            false,
            None,
        )?;
        self.cuda_cross_entropy_for_logits_tensor(&target_tensor)
    }

    fn cuda_cross_entropy_for_logits_tensor(&self, targets: &Self) -> Result<Self> {
        let device = self.device();
        let Device::Cuda(device_id) = device else {
            return Err(TensorError::Device(format!(
                "CUDA cross_entropy_for_logits requires CUDA logits, got {device:?}"
            )));
        };
        if self.dtype() != DType::F32 {
            return Err(TensorError::DType(format!(
                "CUDA cross_entropy_for_logits currently supports only f32 logits, got {:?}",
                self.dtype()
            )));
        }
        if targets.dtype() != DType::I64 || targets.device() != device {
            return Err(TensorError::DType(format!(
                "CUDA cross_entropy_for_logits expected i64 CUDA targets on {:?}, got {:?} on {:?}",
                device,
                targets.dtype(),
                targets.device()
            )));
        }
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(TensorError::Shape(format!(
                "cross_entropy_for_logits expects rank-2 logits [batch, classes], got {:?}",
                shape
            )));
        }
        let (batch, classes) = (shape[0], shape[1]);
        if batch == 0 || classes == 0 {
            return Err(TensorError::InvalidOperation(format!(
                "cross_entropy_for_logits requires non-empty batch/classes, got {:?}",
                shape
            )));
        }
        if targets.shape() != vec![batch] {
            return Err(TensorError::Shape(format!(
                "target shape {:?} must equal [{batch}]",
                targets.shape()
            )));
        }
        let logits = self.cuda_f32_full_storage_buffer("CUDA cross entropy logits")?;
        let output = heirloom_kernels::cuda::cross_entropy_forward_f32_i64_buffers(
            &logits,
            &targets.cuda_i64_full_storage_buffer("CUDA cross entropy targets")?,
            batch,
            classes,
        )
        .map_err(cuda_error)?;
        let requires_grad = should_track_grad(self.requires_grad());
        let grad_fn = requires_grad.then(|| GradFn::CrossEntropy {
            logits: SavedTensor::new(self, "CUDA cross entropy logits"),
            targets: CrossEntropyTargets::Cuda(SavedTensor::new(
                targets,
                "CUDA cross entropy targets",
            )),
        });
        Self::from_storage(
            Storage::new_cuda_from_buffer(device_id, DType::F32, 1, output)?,
            &[],
            DType::F32,
            device,
            requires_grad,
            grad_fn,
        )
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("id", &self.id())
            .field("shape", &self.shape())
            .field("strides", &self.strides())
            .field("storage_offset", &self.storage_offset())
            .field("storage_version", &self.storage_version())
            .field("dtype", &self.dtype())
            .field("device", &self.device())
            .field("requires_grad", &self.requires_grad())
            .field("data", &self.data())
            .finish()
    }
}

fn validate_dim_for_rank(rank: usize, dim: usize, op_name: &str) -> Result<()> {
    if dim >= rank {
        return Err(TensorError::Shape(format!(
            "{op_name} dim {dim} out of range for tensor rank {rank}"
        )));
    }
    Ok(())
}

fn ensure_same_dtype_device(left: &Tensor, right: &Tensor) -> Result<()> {
    if left.dtype() != right.dtype() {
        return Err(TensorError::DType(format!(
            "expected matching dtypes, got {:?} and {:?}",
            left.dtype(),
            right.dtype()
        )));
    }
    if left.device() != right.device() {
        return Err(TensorError::Device(format!(
            "expected matching devices, got {:?} and {:?}",
            left.device(),
            right.device()
        )));
    }
    Ok(())
}

fn ensure_dtype(actual: DType, expected: DType, op_name: &str) -> Result<()> {
    if actual != expected {
        return Err(TensorError::DType(format!(
            "{op_name} expected dtype {:?}, got {:?}",
            expected, actual
        )));
    }
    Ok(())
}

fn ensure_grad_dtype(dtype: DType, requires_grad: bool) -> Result<()> {
    if requires_grad && !dtype.is_floating() {
        return Err(TensorError::Autograd(format!(
            "only floating tensors can require gradients, got dtype {:?}",
            dtype
        )));
    }
    Ok(())
}

fn ensure_floating_tensor(tensor: &Tensor, op_name: &str) -> Result<()> {
    if !tensor.dtype().is_floating() {
        return Err(TensorError::DType(format!(
            "{op_name} requires floating tensors, got {:?}",
            tensor.dtype()
        )));
    }
    Ok(())
}

fn ensure_fresh_output(schema: &OperatorInfo) -> Result<()> {
    if schema.returns_fresh_output() {
        Ok(())
    } else {
        Err(TensorError::InvalidOperation(format!(
            "{} is not declared as a fresh-output operator",
            schema.name
        )))
    }
}

fn should_record_grad(
    schema: &OperatorInfo,
    output_dtype: DType,
    input_requires_grad: bool,
) -> bool {
    should_track_grad(schema.should_record_autograd(output_dtype, input_requires_grad))
}

fn gelu_value(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044_715 * x * x * x)).tanh())
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

fn storage_from_f64_values(data: Vec<f64>, dtype: DType) -> Rc<Storage> {
    match dtype {
        DType::F32 => Storage::new_f32(data.into_iter().map(|value| value as f32).collect()),
        DType::BFloat16 => Storage::new_bf16_bits(
            data.into_iter()
                .map(|value| f32_to_bf16_bits(value as f32))
                .collect(),
        ),
        DType::F64 => Storage::new_f64(data),
        DType::I64 => Storage::new_i64(data.into_iter().map(|value| value as i64).collect()),
        DType::Bool => Storage::new_bool(data.into_iter().map(|value| value != 0.0).collect()),
    }
}

fn cuda_error(error: heirloom_kernels::cuda::CudaError) -> TensorError {
    TensorError::Device(error.to_string())
}

#[cfg(test)]
mod safety_tests {
    use super::*;

    #[test]
    fn product_key_half_dimension_overflow_returns_shape_error() {
        let indices = Tensor::from_i64(Vec::new(), &[0, 1], false).unwrap();
        let query = Tensor::from_f32(Vec::new(), &[0, 0], false).unwrap();
        let left = Tensor::from_f32(Vec::new(), &[0, usize::MAX], false).unwrap();
        let right = Tensor::from_f32(Vec::new(), &[0, usize::MAX], false).unwrap();

        let error = indices
            .memory_product_key_selected_scores(&query, &left, &right)
            .expect_err("overflowing doubled half-key dimension must be rejected");

        assert!(error.to_string().contains("cannot be doubled"));
    }
}
