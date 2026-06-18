use crate::{amp, Result, TensorError};
use heirloom_kernels::cuda::CudaBuffer;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    BFloat16,
    F64,
    I64,
    Bool,
}

impl DType {
    pub(crate) fn is_floating(self) -> bool {
        matches!(self, Self::F32 | Self::BFloat16 | Self::F64)
    }
}

#[derive(Clone, Debug)]
pub(crate) enum StorageData {
    F32(Vec<f32>),
    BF16(Vec<u16>),
    F64(Vec<f64>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
    Cuda {
        dtype: DType,
        device: Device,
        len: usize,
        buffer: CudaBuffer,
    },
}

impl StorageData {
    pub(crate) fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::BF16(_) => DType::BFloat16,
            Self::F64(_) => DType::F64,
            Self::I64(_) => DType::I64,
            Self::Bool(_) => DType::Bool,
            Self::Cuda { dtype, .. } => *dtype,
        }
    }

    pub(crate) fn device(&self) -> Device {
        match self {
            Self::F32(_) | Self::BF16(_) | Self::F64(_) | Self::I64(_) | Self::Bool(_) => {
                Device::Cpu
            }
            Self::Cuda { device, .. } => *device,
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::F32(data) => data.len(),
            Self::BF16(data) => data.len(),
            Self::F64(data) => data.len(),
            Self::I64(data) => data.len(),
            Self::Bool(data) => data.len(),
            Self::Cuda { len, .. } => *len,
        }
    }

    pub(crate) fn value_as_f64(&self, index: usize) -> f64 {
        match self {
            Self::F32(data) => data[index] as f64,
            Self::BF16(data) => bf16_bits_to_f32(data[index]) as f64,
            Self::F64(data) => data[index],
            Self::I64(data) => data[index] as f64,
            Self::Bool(data) => {
                if data[index] {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Cuda { .. } => self
                .to_cpu_storage()
                .expect("failed to materialize CUDA storage")
                .value_as_f64(index),
        }
    }

    pub(crate) fn set_from_f64(&mut self, index: usize, value: f64) {
        match self {
            Self::F32(data) => data[index] = value as f32,
            Self::BF16(data) => data[index] = f32_to_bf16_bits(value as f32),
            Self::F64(data) => data[index] = value,
            Self::I64(data) => data[index] = value as i64,
            Self::Bool(data) => data[index] = value != 0.0,
            Self::Cuda { .. } => {
                panic!("CUDA storage cannot be mutated through CPU scalar access")
            }
        }
    }

    pub(crate) fn to_cpu_storage(&self) -> Result<Self> {
        match self {
            Self::F32(data) => Ok(Self::F32(data.clone())),
            Self::BF16(data) => Ok(Self::BF16(data.clone())),
            Self::F64(data) => Ok(Self::F64(data.clone())),
            Self::I64(data) => Ok(Self::I64(data.clone())),
            Self::Bool(data) => Ok(Self::Bool(data.clone())),
            Self::Cuda {
                dtype,
                device,
                len,
                buffer,
            } => {
                amp::record_cuda_host_staging(
                    "StorageData::to_cpu_storage",
                    *dtype,
                    *device,
                    *len,
                )?;
                let expected_device = match device {
                    Device::Cpu => {
                        return Err(TensorError::Device(
                            "CUDA storage carried a CPU device tag".to_string(),
                        ))
                    }
                    Device::Cuda(device_id) => *device_id,
                };
                if usize::try_from(buffer.device_ordinal()).ok() != Some(expected_device) {
                    return Err(TensorError::Device(format!(
                        "CUDA storage device tag {:?} does not match buffer ordinal {}",
                        device,
                        buffer.device_ordinal()
                    )));
                }
                if buffer.len() != *len {
                    return Err(TensorError::Device(format!(
                        "CUDA storage length {} does not match buffer length {}",
                        len,
                        buffer.len()
                    )));
                }
                match dtype {
                    DType::F32 => Ok(Self::F32(buffer.to_f32().map_err(cuda_error)?)),
                    DType::BFloat16 => Ok(Self::BF16(buffer.to_u16().map_err(cuda_error)?)),
                    DType::F64 => Ok(Self::F64(buffer.to_f64().map_err(cuda_error)?)),
                    DType::I64 => Ok(Self::I64(buffer.to_i64().map_err(cuda_error)?)),
                    DType::Bool => Ok(Self::Bool(
                        buffer
                            .to_u8()
                            .map_err(cuda_error)?
                            .into_iter()
                            .map(|value| value != 0)
                            .collect(),
                    )),
                }
            }
        }
    }

    pub(crate) fn to_f64_vec(&self) -> Result<Vec<f64>> {
        match self.to_cpu_storage()? {
            Self::F32(data) => Ok(data.into_iter().map(f64::from).collect()),
            Self::BF16(data) => Ok(data
                .into_iter()
                .map(|bits| bf16_bits_to_f32(bits) as f64)
                .collect()),
            Self::F64(data) => Ok(data),
            Self::I64(data) => Ok(data.into_iter().map(|value| value as f64).collect()),
            Self::Bool(data) => Ok(data
                .into_iter()
                .map(|value| if value { 1.0 } else { 0.0 })
                .collect()),
            Self::Cuda { .. } => unreachable!("to_cpu_storage removes CUDA storage"),
        }
    }

    pub(crate) fn to_f32_vec(&self) -> Result<Vec<f32>> {
        match self.to_cpu_storage()? {
            Self::F32(data) => Ok(data),
            Self::BF16(data) => Ok(data.into_iter().map(bf16_bits_to_f32).collect()),
            Self::F64(data) => Ok(data.into_iter().map(|value| value as f32).collect()),
            Self::I64(data) => Ok(data.into_iter().map(|value| value as f32).collect()),
            Self::Bool(data) => Ok(data
                .into_iter()
                .map(|value| if value { 1.0 } else { 0.0 })
                .collect()),
            Self::Cuda { .. } => unreachable!("to_cpu_storage removes CUDA storage"),
        }
    }

    pub(crate) fn to_i64_vec(&self) -> Result<Vec<i64>> {
        match self.to_cpu_storage()? {
            Self::F32(data) => Ok(data.into_iter().map(|value| value as i64).collect()),
            Self::BF16(data) => Ok(data
                .into_iter()
                .map(|bits| bf16_bits_to_f32(bits) as i64)
                .collect()),
            Self::F64(data) => Ok(data.into_iter().map(|value| value as i64).collect()),
            Self::I64(data) => Ok(data),
            Self::Bool(data) => Ok(data.into_iter().map(i64::from).collect()),
            Self::Cuda { .. } => unreachable!("to_cpu_storage removes CUDA storage"),
        }
    }

    pub(crate) fn to_bool_vec(&self) -> Result<Vec<bool>> {
        match self.to_cpu_storage()? {
            Self::F32(data) => Ok(data.into_iter().map(|value| value != 0.0).collect()),
            Self::BF16(data) => Ok(data
                .into_iter()
                .map(|bits| bf16_bits_to_f32(bits) != 0.0)
                .collect()),
            Self::F64(data) => Ok(data.into_iter().map(|value| value != 0.0).collect()),
            Self::I64(data) => Ok(data.into_iter().map(|value| value != 0).collect()),
            Self::Bool(data) => Ok(data),
            Self::Cuda { .. } => unreachable!("to_cpu_storage removes CUDA storage"),
        }
    }

    pub(crate) fn to_bf16_bits_vec(&self) -> Result<Vec<u16>> {
        match self.to_cpu_storage()? {
            Self::F32(data) => Ok(data.into_iter().map(f32_to_bf16_bits).collect()),
            Self::BF16(data) => Ok(data),
            Self::F64(data) => Ok(data
                .into_iter()
                .map(|value| f32_to_bf16_bits(value as f32))
                .collect()),
            Self::I64(data) => Ok(data
                .into_iter()
                .map(|value| f32_to_bf16_bits(value as f32))
                .collect()),
            Self::Bool(data) => Ok(data
                .into_iter()
                .map(|value| f32_to_bf16_bits(if value { 1.0 } else { 0.0 }))
                .collect()),
            Self::Cuda { .. } => unreachable!("to_cpu_storage removes CUDA storage"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda(usize),
}

#[derive(Debug)]
pub(crate) struct Storage {
    data: RefCell<StorageData>,
    version: Cell<usize>,
}

impl Storage {
    pub(crate) fn new_f32(data: Vec<f32>) -> Rc<Self> {
        Self::new(StorageData::F32(data))
    }

    pub(crate) fn new_bf16_bits(data: Vec<u16>) -> Rc<Self> {
        Self::new(StorageData::BF16(data))
    }

    pub(crate) fn new_f64(data: Vec<f64>) -> Rc<Self> {
        Self::new(StorageData::F64(data))
    }

    pub(crate) fn new_i64(data: Vec<i64>) -> Rc<Self> {
        Self::new(StorageData::I64(data))
    }

    pub(crate) fn new_bool(data: Vec<bool>) -> Rc<Self> {
        Self::new(StorageData::Bool(data))
    }

    pub(crate) fn new_cuda_from_cpu(
        device_id: usize,
        dtype: DType,
        cpu_storage: &StorageData,
    ) -> Result<Rc<Self>> {
        if cpu_storage.device() != Device::Cpu || cpu_storage.dtype() != dtype {
            return Err(TensorError::Device(format!(
                "new_cuda_from_cpu expected CPU {:?} storage, got {:?} on {:?}",
                dtype,
                cpu_storage.dtype(),
                cpu_storage.device()
            )));
        }
        let device_ordinal = cuda_device_ordinal(device_id)?;
        let buffer = match dtype {
            DType::F32 => CudaBuffer::from_f32(device_ordinal, &cpu_storage.to_f32_vec()?)
                .map_err(cuda_error)?,
            DType::BFloat16 => {
                CudaBuffer::from_u16(device_ordinal, &cpu_storage.to_bf16_bits_vec()?)
                    .map_err(cuda_error)?
            }
            DType::F64 => CudaBuffer::from_f64(device_ordinal, &cpu_storage.to_f64_vec()?)
                .map_err(cuda_error)?,
            DType::I64 => CudaBuffer::from_i64(device_ordinal, &cpu_storage.to_i64_vec()?)
                .map_err(cuda_error)?,
            DType::Bool => CudaBuffer::from_u8(
                device_ordinal,
                &cpu_storage
                    .to_bool_vec()?
                    .into_iter()
                    .map(u8::from)
                    .collect::<Vec<_>>(),
            )
            .map_err(cuda_error)?,
        };
        Ok(Self::new(StorageData::Cuda {
            dtype,
            device: Device::Cuda(device_id),
            len: cpu_storage.len(),
            buffer,
        }))
    }

    pub(crate) fn new_cuda_from_buffer(
        device_id: usize,
        dtype: DType,
        len: usize,
        buffer: CudaBuffer,
    ) -> Result<Rc<Self>> {
        let device_ordinal = cuda_device_ordinal(device_id)?;
        if buffer.device_ordinal() != device_ordinal {
            return Err(TensorError::Device(format!(
                "CUDA buffer ordinal {} does not match tensor device cuda:{device_id}",
                buffer.device_ordinal()
            )));
        }
        if buffer.len() != len {
            return Err(TensorError::Device(format!(
                "CUDA buffer length {} does not match storage length {len}",
                buffer.len()
            )));
        }
        let expected_element_size = match dtype {
            DType::F32 => std::mem::size_of::<f32>(),
            DType::BFloat16 => std::mem::size_of::<u16>(),
            DType::F64 => std::mem::size_of::<f64>(),
            DType::I64 => std::mem::size_of::<i64>(),
            DType::Bool => std::mem::size_of::<u8>(),
        };
        if buffer.element_size() != expected_element_size {
            return Err(TensorError::Device(format!(
                "CUDA buffer element size {} does not match {:?} element size {}",
                buffer.element_size(),
                dtype,
                expected_element_size
            )));
        }
        Ok(Self::new(StorageData::Cuda {
            dtype,
            device: Device::Cuda(device_id),
            len,
            buffer,
        }))
    }

    fn new(data: StorageData) -> Rc<Self> {
        Rc::new(Self {
            data: RefCell::new(data),
            version: Cell::new(0),
        })
    }

    pub(crate) fn dtype(&self) -> DType {
        self.data.borrow().dtype()
    }

    pub(crate) fn device(&self) -> Device {
        self.data.borrow().device()
    }

    pub(crate) fn len(&self) -> usize {
        self.data.borrow().len()
    }

    pub(crate) fn cpu_clone(&self) -> Result<Rc<Self>> {
        Ok(Self::new(self.data.borrow().to_cpu_storage()?))
    }

    pub(crate) fn cuda_buffer(&self, dtype: DType, device: Device) -> Result<CudaBuffer> {
        let data = self.data.borrow();
        match &*data {
            StorageData::Cuda {
                dtype: actual_dtype,
                device: actual_device,
                buffer,
                ..
            } if *actual_dtype == dtype && *actual_device == device => Ok(buffer.clone()),
            StorageData::Cuda {
                dtype: actual_dtype,
                device: actual_device,
                ..
            } => Err(TensorError::Device(format!(
                "CUDA storage expected {:?} on {:?}, got {:?} on {:?}",
                dtype, device, actual_dtype, actual_device
            ))),
            other => Err(TensorError::Device(format!(
                "expected CUDA storage for {:?} on {:?}, got {:?} on {:?}",
                dtype,
                device,
                other.dtype(),
                other.device()
            ))),
        }
    }

    pub(crate) fn version(&self) -> usize {
        self.version.get()
    }

    pub(crate) fn bump_version(&self) {
        self.version.set(self.version.get() + 1);
    }

    pub(crate) fn with_data<R>(&self, f: impl FnOnce(&StorageData) -> R) -> R {
        let data = self.data.borrow();
        f(&data)
    }

    pub(crate) fn with_data_mut<R>(&self, f: impl FnOnce(&mut StorageData) -> R) -> R {
        let mut data = self.data.borrow_mut();
        let result = f(&mut data);
        self.bump_version();
        result
    }
}

pub(crate) fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fff + lsb;
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

pub(crate) fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn cuda_device_ordinal(device_id: usize) -> Result<i32> {
    i32::try_from(device_id).map_err(|_| {
        TensorError::Device(format!(
            "CUDA device id {device_id} does not fit into a CUDA ordinal"
        ))
    })
}

fn cuda_error(error: heirloom_kernels::cuda::CudaError) -> TensorError {
    TensorError::Device(error.to_string())
}
