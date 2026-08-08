use heirloom::{DType, Device, Tensor, TensorError};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};

#[pyclass(name = "Tensor", unsendable, skip_from_py_object)]
#[derive(Clone)]
struct PyTensor {
    inner: Tensor,
}

#[pymethods]
impl PyTensor {
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        dtype_name(self.inner.dtype())
    }

    #[getter]
    fn device(&self) -> String {
        device_name(self.inner.device())
    }

    #[getter]
    fn strides(&self) -> Vec<usize> {
        self.inner.strides()
    }

    #[getter]
    fn storage_offset(&self) -> usize {
        self.inner.storage_offset()
    }

    #[getter]
    fn numel(&self) -> usize {
        self.inner.numel()
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    #[setter]
    fn set_requires_grad(&self, requires_grad: bool) -> PyResult<()> {
        self.inner
            .try_set_requires_grad(requires_grad)
            .map_err(pyerr)
    }

    #[getter]
    fn is_leaf(&self) -> bool {
        self.inner.is_leaf()
    }

    #[getter]
    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    fn tolist(&self) -> Vec<f64> {
        self.inner.data_f64()
    }

    fn data_f32(&self) -> PyResult<Vec<f32>> {
        self.inner.data_f32().map_err(pyerr)
    }

    fn data_f64(&self) -> PyResult<Vec<f64>> {
        self.inner.data_f64_exact().map_err(pyerr)
    }

    fn data_i64(&self) -> PyResult<Vec<i64>> {
        self.inner.data_i64().map_err(pyerr)
    }

    fn data_bool(&self) -> PyResult<Vec<bool>> {
        self.inner.data_bool().map_err(pyerr)
    }

    fn numpy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let numpy = py.import("numpy")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", numpy_dtype_name(self.inner.dtype()))?;
        let array = numpy
            .getattr("array")?
            .call((self.tolist(),), Some(&kwargs))?;
        Ok(array
            .call_method1("reshape", (self.inner.shape(),))?
            .unbind())
    }

    fn to_device(&self, device: &str) -> PyResult<Self> {
        wrap_tensor(self.inner.to_device(parse_device(device)?))
    }

    fn cuda(&self, device_id: usize) -> PyResult<Self> {
        wrap_tensor(self.inner.cuda(device_id))
    }

    fn cpu(&self) -> PyResult<Self> {
        wrap_tensor(self.inner.cpu())
    }

    fn to_dtype(&self, dtype: &str) -> PyResult<Self> {
        wrap_tensor(self.inner.to_dtype(parse_dtype(dtype)?))
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    fn retain_grad(&self) -> PyResult<()> {
        self.inner.retain_grad().map_err(pyerr)
    }

    fn grad(&self) -> Option<Self> {
        self.inner.grad_tensor().map(|inner| Self { inner })
    }

    #[pyo3(signature = (seed_grad = None, retain_graph = false))]
    fn backward(&self, seed_grad: Option<Vec<f32>>, retain_graph: bool) -> PyResult<()> {
        match (seed_grad, retain_graph) {
            (Some(seed), true) => self.inner.backward_with_grad_retain_graph(seed),
            (Some(seed), false) => self.inner.backward_with_grad(seed),
            (None, true) => self.inner.backward_retain_graph(),
            (None, false) => self.inner.backward(),
        }
        .map_err(pyerr)
    }

    fn add(&self, other: PyRef<'_, PyTensor>) -> PyResult<Self> {
        wrap_tensor(self.inner.add(&other.inner))
    }

    fn sub(&self, other: PyRef<'_, PyTensor>) -> PyResult<Self> {
        wrap_tensor(self.inner.sub(&other.inner))
    }

    fn mul(&self, other: PyRef<'_, PyTensor>) -> PyResult<Self> {
        wrap_tensor(self.inner.mul(&other.inner))
    }

    fn div(&self, other: PyRef<'_, PyTensor>) -> PyResult<Self> {
        wrap_tensor(self.inner.div(&other.inner))
    }

    fn matmul(&self, other: PyRef<'_, PyTensor>) -> PyResult<Self> {
        wrap_tensor(self.inner.matmul(&other.inner))
    }

    fn relu(&self) -> PyResult<Self> {
        wrap_tensor(self.inner.relu())
    }

    fn gelu(&self) -> PyResult<Self> {
        wrap_tensor(self.inner.gelu())
    }

    fn sum(&self) -> PyResult<Self> {
        wrap_tensor(self.inner.sum())
    }

    fn mean(&self) -> PyResult<Self> {
        wrap_tensor(self.inner.mean())
    }

    #[pyo3(signature = (dim, keepdim = false))]
    fn sum_dim(&self, dim: usize, keepdim: bool) -> PyResult<Self> {
        wrap_tensor(self.inner.sum_dim(dim, keepdim))
    }

    #[pyo3(signature = (dim, keepdim = false))]
    fn mean_dim(&self, dim: usize, keepdim: bool) -> PyResult<Self> {
        wrap_tensor(self.inner.mean_dim(dim, keepdim))
    }

    fn mse_loss(&self, target: PyRef<'_, PyTensor>) -> PyResult<Self> {
        wrap_tensor(self.inner.mse_loss(&target.inner))
    }

    fn view(&self, shape: Vec<usize>) -> PyResult<Self> {
        wrap_tensor(self.inner.view(&shape))
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        wrap_tensor(self.inner.reshape(&shape))
    }

    fn expand(&self, shape: Vec<usize>) -> PyResult<Self> {
        wrap_tensor(self.inner.expand(&shape))
    }

    fn contiguous(&self) -> PyResult<Self> {
        wrap_tensor(self.inner.contiguous())
    }

    fn narrow(&self, dim: usize, start: usize, len: usize) -> PyResult<Self> {
        wrap_tensor(self.inner.narrow(dim, start, len))
    }

    fn permute(&self, dims: Vec<usize>) -> PyResult<Self> {
        wrap_tensor(self.inner.permute(&dims))
    }

    fn transpose(&self) -> PyResult<Self> {
        wrap_tensor(self.inner.transpose())
    }

    fn softmax_dim(&self, dim: usize) -> PyResult<Self> {
        wrap_tensor(self.inner.softmax_dim(dim))
    }

    fn layer_norm_last_dim(
        &self,
        weight: PyRef<'_, PyTensor>,
        bias: PyRef<'_, PyTensor>,
        eps: f64,
    ) -> PyResult<Self> {
        wrap_tensor(
            self.inner
                .layer_norm_last_dim(&weight.inner, &bias.inner, eps),
        )
    }

    fn embedding(&self, weight: PyRef<'_, PyTensor>) -> PyResult<Self> {
        wrap_tensor(self.inner.embedding(&weight.inner))
    }

    fn cross_entropy_for_logits(&self, targets: PyRef<'_, PyTensor>) -> PyResult<Self> {
        wrap_tensor(self.inner.cross_entropy_for_logits_tensor(&targets.inner))
    }

    fn causal_self_attention(
        &self,
        key: PyRef<'_, PyTensor>,
        value: PyRef<'_, PyTensor>,
        n_heads: usize,
    ) -> PyResult<Self> {
        wrap_tensor(
            self.inner
                .causal_self_attention(&key.inner, &value.inner, n_heads),
        )
    }

    fn __add__(&self, other: PyRef<'_, PyTensor>) -> PyResult<Self> {
        self.add(other)
    }

    fn __sub__(&self, other: PyRef<'_, PyTensor>) -> PyResult<Self> {
        self.sub(other)
    }

    fn __mul__(&self, other: PyRef<'_, PyTensor>) -> PyResult<Self> {
        self.mul(other)
    }

    fn __truediv__(&self, other: PyRef<'_, PyTensor>) -> PyResult<Self> {
        self.div(other)
    }

    fn __matmul__(&self, other: PyRef<'_, PyTensor>) -> PyResult<Self> {
        self.matmul(other)
    }

    fn __repr__(&self) -> String {
        format!(
            "heirloom_py.Tensor(shape={:?}, dtype='{}', device='{}', requires_grad={})",
            self.inner.shape(),
            dtype_name(self.inner.dtype()),
            device_name(self.inner.device()),
            self.inner.requires_grad()
        )
    }
}

#[pyfunction(signature = (data, shape, dtype = "f32", requires_grad = false, device = "cpu"))]
fn tensor(
    data: Vec<f64>,
    shape: Vec<usize>,
    dtype: &str,
    requires_grad: bool,
    device: &str,
) -> PyResult<PyTensor> {
    let dtype = parse_dtype(dtype)?;
    let device = parse_device(device)?;
    wrap_tensor(tensor_from_f64(data, &shape, dtype, requires_grad, device))
}

#[pyfunction(signature = (shape, dtype = "f32", requires_grad = false, device = "cpu"))]
fn zeros(shape: Vec<usize>, dtype: &str, requires_grad: bool, device: &str) -> PyResult<PyTensor> {
    wrap_tensor(Tensor::zeros_on_device(
        &shape,
        parse_dtype(dtype)?,
        parse_device(device)?,
        requires_grad,
    ))
}

#[pyfunction(signature = (shape, dtype = "f32", requires_grad = false, device = "cpu"))]
fn ones(shape: Vec<usize>, dtype: &str, requires_grad: bool, device: &str) -> PyResult<PyTensor> {
    wrap_tensor(Tensor::ones_on_device(
        &shape,
        parse_dtype(dtype)?,
        parse_device(device)?,
        requires_grad,
    ))
}

#[pyfunction(signature = (value, dtype = "f32", requires_grad = false, device = "cpu"))]
fn scalar(value: f64, dtype: &str, requires_grad: bool, device: &str) -> PyResult<PyTensor> {
    tensor(vec![value], Vec::new(), dtype, requires_grad, device)
}

#[pyfunction(signature = (array, requires_grad = false, device = "cpu"))]
fn from_numpy(array: &Bound<'_, PyAny>, requires_grad: bool, device: &str) -> PyResult<PyTensor> {
    let shape = array.getattr("shape")?.extract::<Vec<usize>>()?;
    let dtype_object = array.getattr("dtype")?;
    let dtype_string = dtype_object.str()?.to_str()?.to_string();
    let dtype = parse_numpy_dtype(&dtype_string)?;
    let flat = array.call_method0("ravel")?.call_method0("tolist")?;
    let tensor = match dtype {
        DType::F32 => Tensor::from_f32(flat.extract::<Vec<f32>>()?, &shape, requires_grad),
        DType::F64 => Tensor::from_f64(flat.extract::<Vec<f64>>()?, &shape, requires_grad),
        DType::I64 => Tensor::from_i64(flat.extract::<Vec<i64>>()?, &shape, requires_grad),
        DType::Bool => Tensor::from_bool(flat.extract::<Vec<bool>>()?, &shape, requires_grad),
        DType::BFloat16 => {
            return Err(PyValueError::new_err(
                "NumPy does not expose a stable bfloat16 dtype in this binding",
            ));
        }
    }
    .map_err(pyerr)?;
    wrap_tensor(tensor.to_device(parse_device(device)?))
}

#[pymodule]
fn heirloom_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(scalar, m)?)?;
    m.add_function(wrap_pyfunction!(from_numpy, m)?)?;
    m.add("version", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

fn wrap_tensor(result: heirloom::Result<Tensor>) -> PyResult<PyTensor> {
    result.map(|inner| PyTensor { inner }).map_err(pyerr)
}

fn tensor_from_f64(
    data: Vec<f64>,
    shape: &[usize],
    dtype: DType,
    requires_grad: bool,
    device: Device,
) -> heirloom::Result<Tensor> {
    let cpu = match dtype {
        DType::F32 => Tensor::from_f32(
            data.into_iter().map(|value| value as f32).collect(),
            shape,
            requires_grad,
        )?,
        DType::BFloat16 => Tensor::from_f32(
            data.into_iter().map(|value| value as f32).collect(),
            shape,
            requires_grad,
        )?
        .to_dtype(DType::BFloat16)?,
        DType::F64 => Tensor::from_f64(data, shape, requires_grad)?,
        DType::I64 => Tensor::from_i64(
            data.into_iter().map(|value| value as i64).collect(),
            shape,
            requires_grad,
        )?,
        DType::Bool => Tensor::from_bool(
            data.into_iter().map(|value| value != 0.0).collect(),
            shape,
            requires_grad,
        )?,
    };
    cpu.to_device(device)
}

fn parse_dtype(dtype: &str) -> PyResult<DType> {
    match dtype.to_ascii_lowercase().as_str() {
        "f32" | "float32" => Ok(DType::F32),
        "bf16" | "bfloat16" => Ok(DType::BFloat16),
        "f64" | "float64" | "double" => Ok(DType::F64),
        "i64" | "int64" | "long" => Ok(DType::I64),
        "bool" | "boolean" => Ok(DType::Bool),
        other => Err(PyValueError::new_err(format!(
            "unsupported dtype '{other}'; expected f32, bf16, f64, i64, or bool"
        ))),
    }
}

fn parse_numpy_dtype(dtype: &str) -> PyResult<DType> {
    match dtype {
        "float32" => Ok(DType::F32),
        "float64" => Ok(DType::F64),
        "int64" => Ok(DType::I64),
        "bool" => Ok(DType::Bool),
        other => Err(PyValueError::new_err(format!(
            "unsupported NumPy dtype '{other}'; expected float32, float64, int64, or bool"
        ))),
    }
}

fn parse_device(device: &str) -> PyResult<Device> {
    if device == "cpu" {
        return Ok(Device::Cpu);
    }
    let Some(ordinal) = device.strip_prefix("cuda:") else {
        return Err(PyValueError::new_err(format!(
            "unsupported device '{device}'; expected 'cpu' or 'cuda:<id>'"
        )));
    };
    let id = ordinal.parse::<usize>().map_err(|_| {
        PyValueError::new_err(format!(
            "unsupported CUDA device '{device}'; expected 'cuda:<non-negative-id>'"
        ))
    })?;
    Ok(Device::Cuda(id))
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::BFloat16 => "bf16",
        DType::F64 => "f64",
        DType::I64 => "i64",
        DType::Bool => "bool",
    }
}

fn numpy_dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 | DType::BFloat16 => "float32",
        DType::F64 => "float64",
        DType::I64 => "int64",
        DType::Bool => "bool",
    }
}

fn device_name(device: Device) -> String {
    match device {
        Device::Cpu => "cpu".to_string(),
        Device::Cuda(id) => format!("cuda:{id}"),
    }
}

fn pyerr(error: TensorError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}
