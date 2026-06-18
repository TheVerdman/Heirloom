use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::fmt;
use std::ops::Deref;
use std::ptr;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

type CUdevice = c_int;
type CUresult = c_int;
type CUdeviceptr = u64;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUstream = *mut c_void;
type CUevent = *mut c_void;
type NcclResult = c_int;
type NcclComm = *mut c_void;

const CUDA_SUCCESS: CUresult = 0;
const CUDA_ERROR_NOT_READY: CUresult = 600;
const CU_EVENT_DISABLE_TIMING: c_uint = 2;
const NCCL_SUCCESS: NcclResult = 0;
const RTLD_NOW: c_int = 2;
#[cfg(target_os = "linux")]
const RTLD_GLOBAL: c_int = 0x100;
#[cfg(target_os = "linux")]
const NCCL_DLOPEN_FLAGS: c_int = RTLD_NOW | RTLD_GLOBAL;
#[cfg(not(target_os = "linux"))]
const NCCL_DLOPEN_FLAGS: c_int = RTLD_NOW;
const CU_JIT_INFO_LOG_BUFFER: c_uint = 3;
const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: c_uint = 4;
const CU_JIT_ERROR_LOG_BUFFER: c_uint = 5;
const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_uint = 6;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
const NCCL_FLOAT32: c_int = 7;
const NCCL_SUM: c_int = 0;

static BF16_MMA_PROBE_CALLS: AtomicUsize = AtomicUsize::new(0);
static BF16_TENSOR_CORE_MATMUL_CALLS: AtomicUsize = AtomicUsize::new(0);
static BF16_TENSOR_CORE_MATMUL_FORWARD_CALLS: AtomicUsize = AtomicUsize::new(0);
static BF16_TENSOR_CORE_MATMUL_BACKWARD_CALLS: AtomicUsize = AtomicUsize::new(0);
static BF16_SCALAR_MATMUL_FALLBACK_CALLS: AtomicUsize = AtomicUsize::new(0);
static BF16_TENSOR_CORE_ATTENTION_FORWARD_CALLS: AtomicUsize = AtomicUsize::new(0);
static BF16_TENSOR_CORE_ATTENTION_QK_MATMUL_CALLS: AtomicUsize = AtomicUsize::new(0);
static BF16_TENSOR_CORE_ATTENTION_AV_MATMUL_CALLS: AtomicUsize = AtomicUsize::new(0);
static BF16_TENSOR_CORE_ATTENTION_BACKWARD_CALLS: AtomicUsize = AtomicUsize::new(0);
static BF16_TENSOR_CORE_ATTENTION_SCORE_GRAD_MATMUL_CALLS: AtomicUsize = AtomicUsize::new(0);
static BF16_TENSOR_CORE_ATTENTION_DQ_MATMUL_CALLS: AtomicUsize = AtomicUsize::new(0);
static BF16_TENSOR_CORE_ATTENTION_DK_MATMUL_CALLS: AtomicUsize = AtomicUsize::new(0);
static BF16_TENSOR_CORE_ATTENTION_DV_MATMUL_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_KERNEL_LAUNCH_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_KERNEL_LAUNCH_ELEMENTS: AtomicUsize = AtomicUsize::new(0);
static CUDA_KERNEL_LAUNCH_FAMILIES: OnceLock<
    Mutex<HashMap<&'static str, KernelLaunchFamilyStats>>,
> = OnceLock::new();
static CUDA_HOST_SYNC_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_STREAM_CREATE_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_STREAM_SYNC_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_EVENT_CREATE_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_EVENT_RECORD_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_EVENT_QUERY_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_EVENT_ELAPSED_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_H2D_BYTES: AtomicUsize = AtomicUsize::new(0);
static CUDA_D2H_BYTES: AtomicUsize = AtomicUsize::new(0);
static CUDA_D2D_BYTES: AtomicUsize = AtomicUsize::new(0);
static CUDA_ALLOC_ACTIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static CUDA_ALLOC_RESERVED_BYTES: AtomicUsize = AtomicUsize::new(0);
static CUDA_ALLOC_HIGH_WATER_BYTES: AtomicUsize = AtomicUsize::new(0);
static CUDA_ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_ALLOC_CACHE_HITS: AtomicUsize = AtomicUsize::new(0);
static CUDA_ALLOC_FREES: AtomicUsize = AtomicUsize::new(0);
static CUDA_ALLOC_DEFERRED_FREES: AtomicUsize = AtomicUsize::new(0);
static CUDA_ALLOC_PENDING_RECLAIMS: AtomicUsize = AtomicUsize::new(0);
static CUDA_MODULE_LOAD_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_MODULE_CACHE_HITS: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_PADDED_TILES: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_REMAINDER_TILES: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_CTA_GEMM_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_CTA_TILES: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_CTA_WARPS_LAUNCHED: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_MMA_WARP_TILES: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_STAGED_CTA_GEMM_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_STAGED_CTA_GEMM_ELAPSED_US: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_SHARED_STAGE_TILES: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_SHARED_STAGE_BYTES: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_WIDE_SWIZZLED_CTA_GEMM_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_WIDE_SWIZZLED_CTA_GEMM_ELAPSED_US: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_SWIZZLED_STAGE_TILES: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_SWIZZLED_STAGE_BYTES: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_LDMATRIX_GEMM_REQUESTED_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_LDMATRIX_GEMM_EXECUTED_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_LDMATRIX_GEMM_STAGED_FALLBACK_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_LDMATRIX_GEMM_HARD_REQUIRE_FAILURES: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_LDMATRIX_GEMM_INSTRUCTIONS: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_LDMATRIX_GEMM_ELAPSED_US: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_CP_ASYNC_GEMM_REQUESTED_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_CP_ASYNC_GEMM_EXECUTED_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_CP_ASYNC_GEMM_STAGED_FALLBACK_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_CP_ASYNC_GEMM_HARD_REQUIRE_FAILURES: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_CP_ASYNC_GEMM_INSTRUCTIONS: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_CP_ASYNC_GEMM_ELAPSED_US: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM_ELAPSED_US: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_LEGACY_WARP_GEMM_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_LEGACY_WARP_GEMM_ELAPSED_US: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_ATTENTION_PADDED_TILES: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_ATTENTION_REMAINDER_TILES: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_ATTENTION_SCALAR_FALLBACKS: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_ATTENTION_LDMATRIX_REQUESTED_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_ATTENTION_LDMATRIX_GLOBAL_FALLBACK_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_ATTENTION_CP_ASYNC_REQUESTED_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_TENSOR_CORE_ATTENTION_CP_ASYNC_GLOBAL_FALLBACK_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_BF16_ATTENTION_MATERIALIZED_REFERENCE_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_ATTENTION_REQUESTED_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_ATTENTION_EXECUTED_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_ATTENTION_FALLBACK_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_ATTENTION_SCALAR_FALLBACK_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_ATTENTION_QK_TILE_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_ATTENTION_AV_TILE_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_ATTENTION_RAGGED_TILE_COUNT: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_ATTENTION_CAUSAL_MASKED_TILE_COUNT: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_ATTENTION_ELAPSED_US: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_ATTENTION_HARD_REQUIRE_FAILURES: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_SCALAR_STREAMING_REQUESTED_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_SCALAR_STREAMING_EXECUTED_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_SCALAR_STREAMING_QK_TILE_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_SCALAR_STREAMING_AV_TILE_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_SCALAR_STREAMING_ELAPSED_US: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_REQUESTED_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_EXECUTED_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_FALLBACK_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_QK_MMA_TILE_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_AV_MMA_TILE_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_RAGGED_TILE_COUNT: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_CAUSAL_MASKED_TILE_COUNT: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_ELAPSED_US: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_REQUESTED_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_EXECUTED_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_FALLBACK_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_ROW_DOT_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_QK_RECOMPUTE_MMA_TILE_CALLS: AtomicUsize =
    AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_DP_MMA_TILE_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_DQ_MMA_TILE_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_DK_MMA_TILE_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_DV_MMA_TILE_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_SCALAR_TILE_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_RAGGED_TILE_COUNT: AtomicUsize = AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_CAUSAL_MASKED_TILE_COUNT: AtomicUsize =
    AtomicUsize::new(0);
static CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_ELAPSED_US: AtomicUsize = AtomicUsize::new(0);
static CUDA_MEMORY_LOOKUP_REJECTED_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_MEMORY_QUERY_KEY_SCORE_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_MEMORY_TOPK_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_MEMORY_PRODUCT_KEY_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_MEMORY_SOFTMAX_TOPK_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_MEMORY_WEIGHTED_VALUE_FORWARD_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_MEMORY_WEIGHTED_VALUE_BACKWARD_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_MEMORY_SELECTED_KEY_BACKWARD_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_MEMORY_SCATTER_ADD_ROWS_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_MEMORY_SPARSE_ADAMW_ROWS_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_MEMORY_SPARSE_ADAMW_COMPACT_ROWS_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_MEMORY_ACCESS_COUNT_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_MEMORY_GATHER_SELECTED_ROWS_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_MEMORY_BOOL_MASK_TO_INDICES_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUDA_MEMORY_SELECTED_TOKENS: AtomicUsize = AtomicUsize::new(0);
static CUDA_MEMORY_SELECTED_ROWS: AtomicUsize = AtomicUsize::new(0);

type CuInit = unsafe extern "C" fn(c_uint) -> CUresult;
type CuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CUresult;
type CuDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult;
type CuDeviceGetPciBusId = unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult;
type CuDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult;
type CuDeviceCanAccessPeer = unsafe extern "C" fn(*mut c_int, CUdevice, CUdevice) -> CUresult;
type CuDevicePrimaryCtxRetain = unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult;
type CuDevicePrimaryCtxRelease = unsafe extern "C" fn(CUdevice) -> CUresult;
type CuCtxCreate = unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult;
type CuCtxSetCurrent = unsafe extern "C" fn(CUcontext) -> CUresult;
type CuCtxDestroy = unsafe extern "C" fn(CUcontext) -> CUresult;
type CuCtxGetCurrent = unsafe extern "C" fn(*mut CUcontext) -> CUresult;
type CuStreamCreate = unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult;
type CuStreamDestroy = unsafe extern "C" fn(CUstream) -> CUresult;
type CuStreamSynchronize = unsafe extern "C" fn(CUstream) -> CUresult;
type CuEventCreate = unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult;
type CuEventRecord = unsafe extern "C" fn(CUevent, CUstream) -> CUresult;
type CuEventQuery = unsafe extern "C" fn(CUevent) -> CUresult;
type CuEventElapsedTime = unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult;
type CuEventDestroy = unsafe extern "C" fn(CUevent) -> CUresult;
type CuMemAlloc = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;
type CuMemFree = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
type CuMemcpyHtoD = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
type CuModuleLoadDataEx = unsafe extern "C" fn(
    *mut CUmodule,
    *const c_void,
    c_uint,
    *mut c_uint,
    *mut *mut c_void,
) -> CUresult;
type CuModuleUnload = unsafe extern "C" fn(CUmodule) -> CUresult;
type CuModuleGetFunction =
    unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
type CuLaunchKernel = unsafe extern "C" fn(
    CUfunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    CUstream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;
type NcclGetVersion = unsafe extern "C" fn(*mut c_int) -> NcclResult;
type NcclGetErrorString = unsafe extern "C" fn(NcclResult) -> *const c_char;
type NcclGetUniqueId = unsafe extern "C" fn(*mut NcclUniqueId) -> NcclResult;
type NcclCommInitRank =
    unsafe extern "C" fn(*mut NcclComm, c_int, NcclUniqueId, c_int) -> NcclResult;
type NcclCommDestroy = unsafe extern "C" fn(NcclComm) -> NcclResult;
type NcclCommAbort = unsafe extern "C" fn(NcclComm) -> NcclResult;
type NcclAllReduce = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    usize,
    c_int,
    c_int,
    NcclComm,
    CUstream,
) -> NcclResult;

#[cfg(unix)]
#[cfg_attr(target_os = "linux", link(name = "dl"))]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

#[derive(Debug, Clone)]
pub struct CudaError {
    message: String,
}

impl CudaError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for CudaError {}

pub type CudaResult<T> = std::result::Result<T, CudaError>;

#[derive(Clone, Debug)]
pub struct CudaDeviceInfo {
    pub ordinal: i32,
    pub name: String,
    pub pci_bus_id: String,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
}

#[derive(Clone, Debug)]
pub struct CudaSystemInfo {
    pub driver_loaded: bool,
    pub device_count: i32,
    pub devices: Vec<CudaDeviceInfo>,
}

#[derive(Clone, Debug)]
pub struct CudaPeerAccess {
    pub from_ordinal: i32,
    pub to_ordinal: i32,
    pub can_access: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct NcclUniqueId {
    internal: [c_char; 128],
}

impl NcclUniqueId {
    pub fn to_hex(self) -> String {
        let mut out = String::with_capacity(self.internal.len() * 2);
        for byte in self.internal {
            out.push_str(&format!("{:02x}", byte as u8));
        }
        out
    }

    pub fn from_hex(value: &str) -> CudaResult<Self> {
        if value.len() != 256 {
            return Err(CudaError::new(format!(
                "NCCL unique id hex must contain 256 hex characters, got {}",
                value.len()
            )));
        }
        let mut internal = [0 as c_char; 128];
        for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
            let hex = std::str::from_utf8(chunk)
                .map_err(|err| CudaError::new(format!("invalid NCCL unique id utf8: {err}")))?;
            let byte = u8::from_str_radix(hex, 16).map_err(|err| {
                CudaError::new(format!(
                    "invalid NCCL unique id hex byte at offset {}: {err}",
                    index * 2
                ))
            })?;
            internal[index] = byte as c_char;
        }
        Ok(Self { internal })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NcclAllReduceStats {
    pub calls: usize,
    pub bytes: usize,
}

#[derive(Clone, Debug)]
pub struct NcclInfo {
    pub loaded: bool,
    pub version: i32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CudaKernelLaunchFamilyCounter {
    pub label: String,
    pub calls: usize,
    pub elements: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct KernelLaunchFamilyStats {
    calls: usize,
    elements: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TensorCoreCounters {
    pub bf16_mma_probe_calls: usize,
    pub bf16_tensor_core_matmul_calls: usize,
    pub bf16_tensor_core_matmul_forward_calls: usize,
    pub bf16_tensor_core_matmul_backward_calls: usize,
    pub bf16_scalar_matmul_fallback_calls: usize,
    pub bf16_tensor_core_attention_forward_calls: usize,
    pub bf16_tensor_core_attention_qk_matmul_calls: usize,
    pub bf16_tensor_core_attention_av_matmul_calls: usize,
    pub bf16_tensor_core_attention_backward_calls: usize,
    pub bf16_tensor_core_attention_score_grad_matmul_calls: usize,
    pub bf16_tensor_core_attention_dq_matmul_calls: usize,
    pub bf16_tensor_core_attention_dk_matmul_calls: usize,
    pub bf16_tensor_core_attention_dv_matmul_calls: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CudaRuntimeCounters {
    pub kernel_launch_calls: usize,
    pub kernel_launch_elements: usize,
    pub kernel_launch_families: Vec<CudaKernelLaunchFamilyCounter>,
    pub host_sync_calls: usize,
    pub stream_create_calls: usize,
    pub stream_sync_calls: usize,
    pub event_create_calls: usize,
    pub event_record_calls: usize,
    pub event_query_calls: usize,
    pub event_elapsed_calls: usize,
    pub h2d_bytes: usize,
    pub d2h_bytes: usize,
    pub d2d_bytes: usize,
    pub allocation_active_bytes: usize,
    pub allocation_reserved_bytes: usize,
    pub allocation_high_water_bytes: usize,
    pub allocation_calls: usize,
    pub allocation_cache_hits: usize,
    pub allocation_frees: usize,
    pub allocation_deferred_frees: usize,
    pub allocation_pending_reclaims: usize,
    pub module_load_calls: usize,
    pub module_cache_hits: usize,
    pub tensor_core_padded_tiles: usize,
    pub tensor_core_remainder_tiles: usize,
    pub tensor_core_cta_gemm_calls: usize,
    pub tensor_core_cta_tiles: usize,
    pub tensor_core_cta_warps_launched: usize,
    pub tensor_core_mma_warp_tiles: usize,
    pub tensor_core_staged_cta_gemm_calls: usize,
    pub tensor_core_staged_cta_gemm_elapsed_us: usize,
    pub tensor_core_shared_stage_tiles: usize,
    pub tensor_core_shared_stage_bytes: usize,
    pub tensor_core_wide_swizzled_cta_gemm_calls: usize,
    pub tensor_core_wide_swizzled_cta_gemm_elapsed_us: usize,
    pub tensor_core_swizzled_stage_tiles: usize,
    pub tensor_core_swizzled_stage_bytes: usize,
    pub tensor_core_ldmatrix_gemm_requested_calls: usize,
    pub tensor_core_ldmatrix_gemm_executed_calls: usize,
    pub tensor_core_ldmatrix_gemm_staged_fallback_calls: usize,
    pub tensor_core_ldmatrix_gemm_hard_require_failures: usize,
    pub tensor_core_ldmatrix_gemm_instructions: usize,
    pub tensor_core_ldmatrix_gemm_elapsed_us: usize,
    pub tensor_core_cp_async_gemm_requested_calls: usize,
    pub tensor_core_cp_async_gemm_executed_calls: usize,
    pub tensor_core_cp_async_gemm_staged_fallback_calls: usize,
    pub tensor_core_cp_async_gemm_hard_require_failures: usize,
    pub tensor_core_cp_async_gemm_instructions: usize,
    pub tensor_core_cp_async_gemm_elapsed_us: usize,
    pub tensor_core_global_cta_gemm_calls: usize,
    pub tensor_core_global_cta_gemm_elapsed_us: usize,
    pub tensor_core_legacy_warp_gemm_calls: usize,
    pub tensor_core_legacy_warp_gemm_elapsed_us: usize,
    pub tensor_core_attention_padded_tiles: usize,
    pub tensor_core_attention_remainder_tiles: usize,
    pub tensor_core_attention_scalar_fallbacks: usize,
    pub tensor_core_attention_ldmatrix_requested_calls: usize,
    pub tensor_core_attention_ldmatrix_global_fallback_calls: usize,
    pub tensor_core_attention_cp_async_requested_calls: usize,
    pub tensor_core_attention_cp_async_global_fallback_calls: usize,
    pub bf16_attention_materialized_reference_calls: usize,
    pub flash_bf16_attention_requested_calls: usize,
    pub flash_bf16_attention_executed_calls: usize,
    pub flash_bf16_attention_fallback_calls: usize,
    pub flash_bf16_attention_scalar_fallback_calls: usize,
    pub flash_bf16_attention_qk_tile_calls: usize,
    pub flash_bf16_attention_av_tile_calls: usize,
    pub flash_bf16_attention_ragged_tile_count: usize,
    pub flash_bf16_attention_causal_masked_tile_count: usize,
    pub flash_bf16_attention_elapsed_us: usize,
    pub flash_bf16_attention_hard_require_failures: usize,
    pub flash_bf16_scalar_streaming_requested_calls: usize,
    pub flash_bf16_scalar_streaming_executed_calls: usize,
    pub flash_bf16_scalar_streaming_qk_tile_calls: usize,
    pub flash_bf16_scalar_streaming_av_tile_calls: usize,
    pub flash_bf16_scalar_streaming_elapsed_us: usize,
    pub flash_bf16_tensor_core_requested_calls: usize,
    pub flash_bf16_tensor_core_executed_calls: usize,
    pub flash_bf16_tensor_core_fallback_calls: usize,
    pub flash_bf16_tensor_core_qk_mma_tile_calls: usize,
    pub flash_bf16_tensor_core_av_mma_tile_calls: usize,
    pub flash_bf16_tensor_core_ragged_tile_count: usize,
    pub flash_bf16_tensor_core_causal_masked_tile_count: usize,
    pub flash_bf16_tensor_core_elapsed_us: usize,
    pub flash_bf16_tensor_core_backward_requested_calls: usize,
    pub flash_bf16_tensor_core_backward_executed_calls: usize,
    pub flash_bf16_tensor_core_backward_fallback_calls: usize,
    pub flash_bf16_tensor_core_backward_row_dot_calls: usize,
    pub flash_bf16_tensor_core_backward_qk_recompute_mma_tile_calls: usize,
    pub flash_bf16_tensor_core_backward_dp_mma_tile_calls: usize,
    pub flash_bf16_tensor_core_backward_dq_mma_tile_calls: usize,
    pub flash_bf16_tensor_core_backward_dk_mma_tile_calls: usize,
    pub flash_bf16_tensor_core_backward_dv_mma_tile_calls: usize,
    pub flash_bf16_tensor_core_backward_scalar_tile_calls: usize,
    pub flash_bf16_tensor_core_backward_ragged_tile_count: usize,
    pub flash_bf16_tensor_core_backward_causal_masked_tile_count: usize,
    pub flash_bf16_tensor_core_backward_elapsed_us: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryKernelCounters {
    pub lookup_rejected_calls: usize,
    pub query_key_score_calls: usize,
    pub topk_calls: usize,
    pub product_key_calls: usize,
    pub softmax_topk_calls: usize,
    pub weighted_value_forward_calls: usize,
    pub weighted_value_backward_calls: usize,
    pub selected_key_backward_calls: usize,
    pub scatter_add_rows_calls: usize,
    pub sparse_adamw_rows_calls: usize,
    pub sparse_adamw_compact_rows_calls: usize,
    pub access_count_calls: usize,
    pub gather_selected_rows_calls: usize,
    pub bool_mask_to_indices_calls: usize,
    pub selected_tokens: usize,
    pub selected_rows: usize,
}

#[derive(Clone, Debug)]
pub struct TensorCoreProbeReport {
    pub device_ordinal: i32,
    pub expected_dot: f32,
    pub max_abs_error: f32,
    pub samples: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct CudaSmokeReport {
    pub device: CudaDeviceInfo,
    pub len: usize,
    pub add_max_abs_error: f32,
    pub relu_max_abs_error: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct CausalAttentionDims {
    pub batch: usize,
    pub time: usize,
    pub channels: usize,
    pub n_heads: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryLookupDims {
    pub tokens: usize,
    pub slots: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub top_k: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryTopkDims {
    pub rows: usize,
    pub cols: usize,
    pub top_k: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryWeightedValueDims {
    pub tokens: usize,
    pub top_k: usize,
    pub slots: usize,
    pub value_dim: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemorySelectedScoreDims {
    pub tokens: usize,
    pub top_k: usize,
    pub slots: usize,
    pub key_dim: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryProductKeyDims {
    pub tokens: usize,
    pub slots: usize,
    pub key_dim: usize,
    pub top_k: usize,
    pub beam: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryProductKeySelectedScoreDims {
    pub tokens: usize,
    pub top_k: usize,
    pub side: usize,
    pub key_dim: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseAdamWRowsDims {
    pub selected_rows: usize,
    pub rows: usize,
    pub row_dim: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectedRowsDims {
    pub selected_rows: usize,
    pub rows: usize,
    pub row_dim: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct MatrixLayout {
    pub rows: usize,
    pub cols: usize,
    pub row_stride: usize,
    pub col_stride: usize,
    pub offset: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct MatmulStridedDims {
    pub left: MatrixLayout,
    pub right: MatrixLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorCoreMatmulCtaPlan {
    pub cta_m: usize,
    pub cta_n: usize,
    pub warps_per_cta: usize,
    pub warp_m: usize,
    pub warp_n: usize,
    pub grid_x: usize,
    pub grid_y: usize,
    pub cta_tiles: usize,
    pub active_mma_warp_tiles: usize,
    pub launched_warps: usize,
    pub k_tiles: usize,
    pub shared_stage_tiles: usize,
    pub shared_stage_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct AdamWParams {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub clip_scale: f32,
    pub bias_correction1: f32,
    pub bias_correction2: f32,
}

#[derive(Clone)]
pub struct CudaBuffer {
    inner: Rc<CudaAllocation>,
    len: usize,
    element_size: usize,
}

impl fmt::Debug for CudaBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaBuffer")
            .field("device_ordinal", &self.inner.device_ordinal)
            .field("len", &self.len)
            .field("element_size", &self.element_size)
            .field("byte_len", &self.byte_len())
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SparseAdamWRowsBuffers<'a> {
    pub param: &'a CudaBuffer,
    pub grad: &'a CudaBuffer,
    pub m: &'a CudaBuffer,
    pub v: &'a CudaBuffer,
    pub selected_rows: &'a CudaBuffer,
    pub row_mask: Option<&'a CudaBuffer>,
}

#[derive(Clone, Copy, Debug)]
pub struct SparseAdamWCompactRowsBuffers<'a> {
    pub param: &'a CudaBuffer,
    pub compact_grad: &'a CudaBuffer,
    pub m: &'a CudaBuffer,
    pub v: &'a CudaBuffer,
    pub selected_rows: &'a CudaBuffer,
    pub row_mask: Option<&'a CudaBuffer>,
}

impl CudaBuffer {
    pub fn uninit_bytes(device_ordinal: i32, len: usize, element_size: usize) -> CudaResult<Self> {
        if element_size == 0 {
            return Err(CudaError::new("CudaBuffer element_size must be non-zero"));
        }
        let bytes = len
            .checked_mul(element_size)
            .ok_or_else(|| CudaError::new("CudaBuffer allocation size overflow"))?;
        let allocation = allocate_cuda_allocation(device_ordinal, bytes)?;
        Ok(Self {
            inner: Rc::new(allocation),
            len,
            element_size,
        })
    }

    pub fn from_f32(device_ordinal: i32, data: &[f32]) -> CudaResult<Self> {
        Self::from_host_slice(device_ordinal, data)
    }

    pub fn to_f32(&self) -> CudaResult<Vec<f32>> {
        self.to_host_vec()
    }

    pub fn from_f64(device_ordinal: i32, data: &[f64]) -> CudaResult<Self> {
        Self::from_host_slice(device_ordinal, data)
    }

    pub fn to_f64(&self) -> CudaResult<Vec<f64>> {
        self.to_host_vec()
    }

    pub fn from_i64(device_ordinal: i32, data: &[i64]) -> CudaResult<Self> {
        Self::from_host_slice(device_ordinal, data)
    }

    pub fn to_i64(&self) -> CudaResult<Vec<i64>> {
        self.to_host_vec()
    }

    pub fn from_u32(device_ordinal: i32, data: &[u32]) -> CudaResult<Self> {
        Self::from_host_slice(device_ordinal, data)
    }

    pub fn to_u32(&self) -> CudaResult<Vec<u32>> {
        self.to_host_vec()
    }

    pub fn from_u64(device_ordinal: i32, data: &[u64]) -> CudaResult<Self> {
        Self::from_host_slice(device_ordinal, data)
    }

    pub fn to_u64(&self) -> CudaResult<Vec<u64>> {
        self.to_host_vec()
    }

    pub fn from_u16(device_ordinal: i32, data: &[u16]) -> CudaResult<Self> {
        Self::from_host_slice(device_ordinal, data)
    }

    pub fn to_u16(&self) -> CudaResult<Vec<u16>> {
        self.to_host_vec()
    }

    pub fn from_u8(device_ordinal: i32, data: &[u8]) -> CudaResult<Self> {
        Self::from_host_slice(device_ordinal, data)
    }

    pub fn to_u8(&self) -> CudaResult<Vec<u8>> {
        self.to_host_vec()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn element_size(&self) -> usize {
        self.element_size
    }

    pub fn byte_len(&self) -> usize {
        self.inner.bytes
    }

    pub fn device_ordinal(&self) -> i32 {
        self.inner.device_ordinal
    }

    fn from_host_slice<T: Copy>(device_ordinal: i32, data: &[T]) -> CudaResult<Self> {
        let buffer = Self::uninit_bytes(device_ordinal, data.len(), std::mem::size_of::<T>())?;
        buffer.copy_from_host_slice(data)?;
        Ok(buffer)
    }

    fn copy_from_host_slice<T: Copy>(&self, data: &[T]) -> CudaResult<()> {
        if data.len() != self.len {
            return Err(CudaError::new(format!(
                "CudaBuffer host copy length mismatch: buffer={} host={}",
                self.len,
                data.len()
            )));
        }
        if std::mem::size_of::<T>() != self.element_size {
            return Err(CudaError::new(format!(
                "CudaBuffer host copy element size mismatch: buffer={} host={}",
                self.element_size,
                std::mem::size_of::<T>()
            )));
        }
        if self.byte_len() == 0 {
            return Ok(());
        }
        let driver = self.inner.driver()?;
        driver.set_current(self.inner.context)?;
        // SAFETY: self.inner.ptr is a live device allocation and data points to byte_len bytes.
        check(
            unsafe {
                (driver.cu_memcpy_htod)(
                    self.inner.ptr,
                    data.as_ptr() as *const c_void,
                    self.byte_len(),
                )
            },
            "cuMemcpyHtoD_v2",
        )?;
        CUDA_H2D_BYTES.fetch_add(self.byte_len(), Ordering::Relaxed);
        Ok(())
    }

    fn to_host_vec<T: Copy + Default>(&self) -> CudaResult<Vec<T>> {
        if std::mem::size_of::<T>() != self.element_size {
            return Err(CudaError::new(format!(
                "CudaBuffer host copy element size mismatch: buffer={} host={}",
                self.element_size,
                std::mem::size_of::<T>()
            )));
        }
        let mut out = vec![T::default(); self.len];
        if self.byte_len() == 0 {
            return Ok(out);
        }
        let driver = self.inner.driver()?;
        driver.set_current(self.inner.context)?;
        synchronize_current_compute_stream(driver, "cuStreamSynchronize before cuMemcpyDtoH_v2")?;
        // SAFETY: out is valid writable host memory and self.inner.ptr is a live device allocation.
        check(
            unsafe {
                (driver.cu_memcpy_dtoh)(
                    out.as_mut_ptr() as *mut c_void,
                    self.inner.ptr,
                    self.byte_len(),
                )
            },
            "cuMemcpyDtoH_v2",
        )?;
        CUDA_D2H_BYTES.fetch_add(self.byte_len(), Ordering::Relaxed);
        Ok(out)
    }
}

pub struct CudaEventTimer {
    driver: CudaDriver,
    device: CUdevice,
    context: CUcontext,
    stream: CUstream,
    start: CUevent,
    stop: CUevent,
    stopped: bool,
}

impl CudaEventTimer {
    pub fn start_current_compute_stream(device_ordinal: i32) -> CudaResult<Self> {
        let (driver, device, context) = create_primary_context(device_ordinal)?;
        let stream = compute_stream_for_current_context(&driver)?;
        Self::start_on_stream(driver, device, context, stream)
    }

    fn start_on_stream(
        driver: CudaDriver,
        device: CUdevice,
        context: CUcontext,
        stream: CUstream,
    ) -> CudaResult<Self> {
        driver.set_current(context)?;
        let start = create_timing_event(&driver)?;
        let stop = match create_timing_event(&driver) {
            Ok(event) => event,
            Err(err) => {
                destroy_event(&driver, context, start);
                return Err(err);
            }
        };
        // SAFETY: start and stream are live CUDA objects for the current context.
        if let Err(err) = check(
            unsafe { (driver.cu_event_record)(start, stream) },
            "cuEventRecord start",
        ) {
            destroy_event(&driver, context, start);
            destroy_event(&driver, context, stop);
            return Err(err);
        }
        CUDA_EVENT_RECORD_CALLS.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            driver,
            device,
            context,
            stream,
            start,
            stop,
            stopped: false,
        })
    }

    pub fn stop_elapsed_ms(&mut self) -> CudaResult<f64> {
        if self.stopped {
            return Err(CudaError::new("CUDA event timer was already stopped"));
        }
        self.driver.set_current(self.context)?;
        // SAFETY: stop and stream are live CUDA objects for the current context.
        check(
            unsafe { (self.driver.cu_event_record)(self.stop, self.stream) },
            "cuEventRecord stop",
        )?;
        CUDA_EVENT_RECORD_CALLS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: stream is live and belongs to the current context.
        check(
            unsafe { (self.driver.cu_stream_synchronize)(self.stream) },
            "cuStreamSynchronize event timer",
        )?;
        CUDA_HOST_SYNC_CALLS.fetch_add(1, Ordering::Relaxed);
        CUDA_STREAM_SYNC_CALLS.fetch_add(1, Ordering::Relaxed);
        let mut elapsed_ms = 0.0f32;
        // SAFETY: start and stop are completed timing events in the same context.
        check(
            unsafe { (self.driver.cu_event_elapsed_time)(&mut elapsed_ms, self.start, self.stop) },
            "cuEventElapsedTime",
        )?;
        CUDA_EVENT_ELAPSED_CALLS.fetch_add(1, Ordering::Relaxed);
        self.stopped = true;
        Ok(elapsed_ms as f64)
    }
}

impl Drop for CudaEventTimer {
    fn drop(&mut self) {
        destroy_event(&self.driver, self.context, self.start);
        destroy_event(&self.driver, self.context, self.stop);
        if !self.context.is_null() {
            let _ = self.driver.set_current(self.context);
            // SAFETY: primary context was retained for this timer and is released once here.
            unsafe {
                let _ = (self.driver.cu_device_primary_ctx_release)(self.device);
            }
            self.context = ptr::null_mut();
        }
    }
}

struct CudaAllocation {
    driver: CudaDriverHandle,
    device: CUdevice,
    context: CUcontext,
    ptr: CUdeviceptr,
    bytes: usize,
    capacity_bytes: usize,
    device_ordinal: i32,
}

struct CudaDriverHandle(Option<CudaDriver>);

impl CudaDriverHandle {
    fn new(driver: CudaDriver) -> Self {
        Self(Some(driver))
    }

    fn take(&mut self) -> Option<CudaDriver> {
        self.0.take()
    }

    fn as_ref(&self) -> Option<&CudaDriver> {
        self.0.as_ref()
    }
}

impl Deref for CudaDriverHandle {
    type Target = CudaDriver;

    fn deref(&self) -> &Self::Target {
        self.0
            .as_ref()
            .expect("CUDA allocation driver has already been released")
    }
}

impl CudaAllocation {
    fn driver(&self) -> CudaResult<&CudaDriver> {
        self.driver
            .as_ref()
            .ok_or_else(|| CudaError::new("CUDA allocation driver has already been released"))
    }
}

impl Drop for CudaAllocation {
    fn drop(&mut self) {
        cache_or_free_allocation(self);
    }
}

pub fn is_available() -> bool {
    system_info()
        .map(|info| info.driver_loaded && info.device_count > 0)
        .unwrap_or(false)
}

pub fn system_info() -> CudaResult<CudaSystemInfo> {
    let driver = CudaDriver::load()?;
    driver.init()?;
    let device_count = driver.device_count()?;
    let mut devices = Vec::new();
    for ordinal in 0..device_count {
        devices.push(driver.device_info(ordinal)?);
    }
    Ok(CudaSystemInfo {
        driver_loaded: true,
        device_count,
        devices,
    })
}

pub fn peer_access_matrix() -> CudaResult<Vec<CudaPeerAccess>> {
    let driver = CudaDriver::load()?;
    driver.init()?;
    let device_count = driver.device_count()?;
    let mut devices = Vec::with_capacity(device_count as usize);
    for ordinal in 0..device_count {
        devices.push(driver.device(ordinal)?);
    }

    let mut matrix = Vec::with_capacity((device_count * device_count) as usize);
    for from_ordinal in 0..device_count {
        for to_ordinal in 0..device_count {
            let can_access = if from_ordinal == to_ordinal {
                true
            } else {
                let mut can_access = 0;
                // SAFETY: can_access is writable and both devices came from cuDeviceGet.
                check(
                    unsafe {
                        (driver.cu_device_can_access_peer)(
                            &mut can_access,
                            devices[from_ordinal as usize],
                            devices[to_ordinal as usize],
                        )
                    },
                    "cuDeviceCanAccessPeer",
                )?;
                can_access != 0
            };
            matrix.push(CudaPeerAccess {
                from_ordinal,
                to_ordinal,
                can_access,
            });
        }
    }
    Ok(matrix)
}

pub fn add_f32(device_ordinal: i32, left: &[f32], right: &[f32]) -> CudaResult<Vec<f32>> {
    if left.len() != right.len() {
        return Err(CudaError::new(format!(
            "add_f32 length mismatch: left={} right={}",
            left.len(),
            right.len()
        )));
    }
    if left.is_empty() {
        return Ok(Vec::new());
    }
    let session = CudaSession::new(device_ordinal)?;
    let module = session.load_module(KERNEL_PTX)?;
    let function = module.function("heirloom_add_f32")?;
    let left_device = DeviceBuffer::from_f32(&session.driver, left)?;
    let right_device = DeviceBuffer::from_f32(&session.driver, right)?;
    let output_device = DeviceBuffer::uninit_f32(&session.driver, left.len())?;
    launch_vector_kernel(
        &session.driver,
        function,
        &[left_device.ptr, right_device.ptr, output_device.ptr],
        left.len(),
    )?;
    output_device.copy_to_f32()
}

pub fn add_f32_buffers(left: &CudaBuffer, right: &CudaBuffer) -> CudaResult<CudaBuffer> {
    binary_f32_buffers(left, right, "heirloom_add_f32", "add_f32_buffers")
}

pub fn sub_f32_buffers(left: &CudaBuffer, right: &CudaBuffer) -> CudaResult<CudaBuffer> {
    binary_f32_buffers(left, right, "heirloom_sub_f32", "sub_f32_buffers")
}

pub fn mul_f32_buffers(left: &CudaBuffer, right: &CudaBuffer) -> CudaResult<CudaBuffer> {
    binary_f32_buffers(left, right, "heirloom_mul_f32", "mul_f32_buffers")
}

pub fn div_f32_buffers(left: &CudaBuffer, right: &CudaBuffer) -> CudaResult<CudaBuffer> {
    binary_f32_buffers(left, right, "heirloom_div_f32", "div_f32_buffers")
}

pub fn bias_add_f32_buffers(
    matrix: &CudaBuffer,
    bias: &CudaBuffer,
    rows: usize,
    cols: usize,
) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(matrix, "bias_add_f32_buffers matrix")?;
    ensure_f32_buffer(bias, "bias_add_f32_buffers bias")?;
    ensure_same_device(matrix, bias, "bias_add_f32_buffers")?;
    let output_len = checked_mul(rows, cols, "bias_add_f32_buffers output")?;
    ensure_buffer_len(matrix, output_len, "bias_add_f32_buffers matrix")?;
    ensure_buffer_len(bias, cols, "bias_add_f32_buffers bias")?;
    let output = CudaBuffer::uninit_bytes(
        matrix.device_ordinal(),
        output_len,
        std::mem::size_of::<f32>(),
    )?;
    if output_len == 0 {
        return Ok(output);
    }

    matrix.inner.driver.set_current(matrix.inner.context)?;
    let module = load_module(&matrix.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_bias_add_f32")?;
    launch_bias_2d_kernel(
        &matrix.inner.driver,
        function,
        &[matrix.inner.ptr, bias.inner.ptr, output.inner.ptr],
        rows,
        cols,
        output_len,
    )?;
    Ok(output)
}

pub fn bias_add_backward_bias_f32_buffer(
    grad_output: &CudaBuffer,
    rows: usize,
    cols: usize,
) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(grad_output, "bias_add_backward_bias_f32_buffer grad_output")?;
    ensure_buffer_len(
        grad_output,
        checked_mul(rows, cols, "bias_add_backward_bias_f32_buffer grad_output")?,
        "bias_add_backward_bias_f32_buffer grad_output",
    )?;
    let output = CudaBuffer::uninit_bytes(
        grad_output.device_ordinal(),
        cols,
        std::mem::size_of::<f32>(),
    )?;
    if cols == 0 {
        return Ok(output);
    }

    grad_output
        .inner
        .driver
        .set_current(grad_output.inner.context)?;
    let module = load_module(&grad_output.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_bias_add_backward_bias_f32")?;
    launch_bias_2d_kernel(
        &grad_output.inner.driver,
        function,
        &[grad_output.inner.ptr, output.inner.ptr],
        rows,
        cols,
        cols,
    )?;
    Ok(output)
}

pub fn neg_f32_buffer(input: &CudaBuffer) -> CudaResult<CudaBuffer> {
    scale_f32_buffer(input, -1.0)
}

pub fn scale_f32_buffer(input: &CudaBuffer, scale: f32) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(input, "scale_f32_buffer input")?;
    let output = CudaBuffer::uninit_bytes(
        input.device_ordinal(),
        input.len(),
        std::mem::size_of::<f32>(),
    )?;
    if input.is_empty() {
        return Ok(output);
    }

    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_scale_f32")?;
    launch_scaled_vector_kernel(
        &input.inner.driver,
        function,
        &[input.inner.ptr, output.inner.ptr],
        scale,
        input.len(),
    )?;
    Ok(output)
}

pub fn scale_in_place_f32_buffer(input: &CudaBuffer, scale: f32) -> CudaResult<()> {
    ensure_f32_buffer(input, "scale_in_place_f32_buffer input")?;
    if !scale.is_finite() {
        return Err(CudaError::new(format!(
            "scale_in_place_f32_buffer expected finite scale, got {scale}"
        )));
    }
    if input.is_empty() {
        return Ok(());
    }

    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_scale_f32")?;
    launch_scaled_vector_kernel(
        &input.inner.driver,
        function,
        &[input.inner.ptr, input.inner.ptr],
        scale,
        input.len(),
    )
}

pub fn f32_to_bf16_buffer(input: &CudaBuffer) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(input, "f32_to_bf16_buffer input")?;
    let output = CudaBuffer::uninit_bytes(
        input.device_ordinal(),
        input.len(),
        std::mem::size_of::<u16>(),
    )?;
    if input.is_empty() {
        return Ok(output);
    }

    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_f32_to_bf16")?;
    launch_vector_kernel(
        &input.inner.driver,
        function,
        &[input.inner.ptr, output.inner.ptr],
        input.len(),
    )?;
    Ok(output)
}

pub fn bf16_to_f32_buffer(input: &CudaBuffer) -> CudaResult<CudaBuffer> {
    ensure_bf16_buffer(input, "bf16_to_f32_buffer input")?;
    let output = CudaBuffer::uninit_bytes(
        input.device_ordinal(),
        input.len(),
        std::mem::size_of::<f32>(),
    )?;
    if input.is_empty() {
        return Ok(output);
    }

    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_bf16_to_f32")?;
    launch_vector_kernel(
        &input.inner.driver,
        function,
        &[input.inner.ptr, output.inner.ptr],
        input.len(),
    )?;
    Ok(output)
}

pub fn f32_bf16_roundtrip_f32_buffer(input: &CudaBuffer) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(input, "f32_bf16_roundtrip_f32_buffer input")?;
    let output = CudaBuffer::uninit_bytes(
        input.device_ordinal(),
        input.len(),
        std::mem::size_of::<f32>(),
    )?;
    if input.is_empty() {
        return Ok(output);
    }

    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_f32_bf16_roundtrip_f32")?;
    launch_vector_kernel_labeled(
        &input.inner.driver,
        function,
        &[input.inner.ptr, output.inner.ptr],
        input.len(),
        "bf16_roundtrip",
    )?;
    Ok(output)
}

fn pad_matrix_bf16_buffer(
    input: &CudaBuffer,
    rows: usize,
    cols: usize,
    padded_rows: usize,
    padded_cols: usize,
) -> CudaResult<CudaBuffer> {
    ensure_bf16_buffer(input, "pad_matrix_bf16_buffer input")?;
    if padded_rows < rows || padded_cols < cols {
        return Err(CudaError::new(format!(
            "pad_matrix_bf16_buffer padded shape [{padded_rows},{padded_cols}] \
             must cover input shape [{rows},{cols}]"
        )));
    }
    ensure_buffer_len(
        input,
        checked_mul(rows, cols, "pad_matrix_bf16_buffer input")?,
        "pad_matrix_bf16_buffer input",
    )?;
    let output_len = checked_mul(padded_rows, padded_cols, "pad_matrix_bf16_buffer output")?;
    let output = CudaBuffer::uninit_bytes(
        input.device_ordinal(),
        output_len,
        std::mem::size_of::<u16>(),
    )?;
    if output_len == 0 {
        return Ok(output);
    }

    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_pad_matrix_bf16")?;
    launch_pad_matrix_bf16_kernel(
        &input.inner.driver,
        function,
        MatrixPadCropLaunch {
            input: input.inner.ptr,
            output: output.inner.ptr,
            rows,
            cols,
            padded_cols,
            total: output_len,
        },
    )?;
    Ok(output)
}

fn crop_matrix_f32_buffer(
    input: &CudaBuffer,
    rows: usize,
    cols: usize,
    padded_cols: usize,
) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(input, "crop_matrix_f32_buffer input")?;
    if padded_cols < cols {
        return Err(CudaError::new(format!(
            "crop_matrix_f32_buffer padded_cols {padded_cols} must be >= cols {cols}"
        )));
    }
    let required_input_len = checked_mul(rows, padded_cols, "crop_matrix_f32_buffer input")?;
    if input.len() < required_input_len {
        return Err(CudaError::new(format!(
            "crop_matrix_f32_buffer input expected at least {required_input_len} elements, got {}",
            input.len()
        )));
    }
    let output_len = checked_mul(rows, cols, "crop_matrix_f32_buffer output")?;
    let output = CudaBuffer::uninit_bytes(
        input.device_ordinal(),
        output_len,
        std::mem::size_of::<f32>(),
    )?;
    if output_len == 0 {
        return Ok(output);
    }

    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_crop_matrix_f32")?;
    launch_crop_matrix_f32_kernel(
        &input.inner.driver,
        function,
        MatrixPadCropLaunch {
            input: input.inner.ptr,
            output: output.inner.ptr,
            rows,
            cols,
            padded_cols,
            total: output_len,
        },
    )?;
    Ok(output)
}

pub fn tensor_core_requirement_enabled() -> bool {
    matches!(
        std::env::var("HEIRLOOM_REQUIRE_TENSOR_CORES")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

pub fn attention_tensor_core_requirement_enabled() -> bool {
    matches!(
        std::env::var("HEIRLOOM_REQUIRE_ATTENTION_TENSOR_CORES")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

pub fn tensor_core_legacy_warp_gemm_enabled() -> bool {
    matches!(
        std::env::var("HEIRLOOM_CUDA_TENSOR_CORE_LEGACY_WARP_GEMM")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

pub fn tensor_core_global_cta_gemm_enabled() -> bool {
    matches!(
        std::env::var("HEIRLOOM_CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

pub fn tensor_core_wide_swizzled_gemm_enabled() -> bool {
    matches!(
        std::env::var("HEIRLOOM_CUDA_TENSOR_CORE_WIDE_SWIZZLED_GEMM")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

pub fn tensor_core_gemm_timing_enabled() -> bool {
    matches!(
        std::env::var("HEIRLOOM_CUDA_TENSOR_CORE_GEMM_TIMING")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

pub fn tensor_core_ldmatrix_gemm_enabled() -> bool {
    matches!(
        std::env::var("HEIRLOOM_CUDA_TENSOR_CORE_LDMATRIX_GEMM")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

pub fn require_tensor_core_ldmatrix_gemm_enabled() -> bool {
    matches!(
        std::env::var("HEIRLOOM_REQUIRE_CUDA_TENSOR_CORE_LDMATRIX_GEMM")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

pub fn tensor_core_cp_async_gemm_enabled() -> bool {
    matches!(
        std::env::var("HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

pub fn tensor_core_normal_rhs_gemm_enabled() -> bool {
    matches!(
        std::env::var("HEIRLOOM_CUDA_TENSOR_CORE_NORMAL_RHS_GEMM")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

pub fn require_tensor_core_cp_async_gemm_enabled() -> bool {
    matches!(
        std::env::var("HEIRLOOM_REQUIRE_CUDA_TENSOR_CORE_CP_ASYNC_GEMM")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

pub fn tensor_core_attention_ldmatrix_enabled() -> bool {
    matches!(
        std::env::var("HEIRLOOM_CUDA_TENSOR_CORE_LDMATRIX_ATTENTION")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

pub fn tensor_core_attention_cp_async_enabled() -> bool {
    matches!(
        std::env::var("HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_ATTENTION")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

pub fn flash_bf16_attention_enabled() -> bool {
    matches!(
        std::env::var("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

pub fn require_flash_bf16_attention_enabled() -> bool {
    matches!(
        std::env::var("HEIRLOOM_REQUIRE_FLASH_BF16_ATTENTION")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

pub fn flash_bf16_attention_timing_enabled() -> bool {
    matches!(
        std::env::var("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TIMING")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

pub fn flash_bf16_tensor_core_attention_enabled() -> bool {
    matches!(
        std::env::var("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_TENSOR_CORE")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

pub fn flash_bf16_tensor_core_attention_backward_enabled() -> bool {
    matches!(
        std::env::var("HEIRLOOM_CUDA_FLASH_BF16_ATTENTION_BACKWARD")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

pub fn tensor_core_gemm_tier_label() -> &'static str {
    if tensor_core_legacy_warp_gemm_enabled() {
        "legacy_warp"
    } else if tensor_core_global_cta_gemm_enabled() {
        "global_cta"
    } else if tensor_core_wide_swizzled_gemm_enabled() {
        "wide_swizzled_cta"
    } else if tensor_core_cp_async_gemm_enabled() {
        "cp_async_double_buffered_ldmatrix_a_mma"
    } else if tensor_core_ldmatrix_gemm_enabled() {
        "ldmatrix_a_shared_mma"
    } else {
        "staged_cta"
    }
}

pub fn tensor_core_attention_tier_label() -> &'static str {
    if tensor_core_attention_cp_async_enabled() {
        "cp_async_requested_global_fallback"
    } else if tensor_core_attention_ldmatrix_enabled() {
        "ldmatrix_requested_global_fallback"
    } else {
        "global_warp"
    }
}

pub fn tensor_core_counters() -> TensorCoreCounters {
    TensorCoreCounters {
        bf16_mma_probe_calls: BF16_MMA_PROBE_CALLS.load(Ordering::Relaxed),
        bf16_tensor_core_matmul_calls: BF16_TENSOR_CORE_MATMUL_CALLS.load(Ordering::Relaxed),
        bf16_tensor_core_matmul_forward_calls: BF16_TENSOR_CORE_MATMUL_FORWARD_CALLS
            .load(Ordering::Relaxed),
        bf16_tensor_core_matmul_backward_calls: BF16_TENSOR_CORE_MATMUL_BACKWARD_CALLS
            .load(Ordering::Relaxed),
        bf16_scalar_matmul_fallback_calls: BF16_SCALAR_MATMUL_FALLBACK_CALLS
            .load(Ordering::Relaxed),
        bf16_tensor_core_attention_forward_calls: BF16_TENSOR_CORE_ATTENTION_FORWARD_CALLS
            .load(Ordering::Relaxed),
        bf16_tensor_core_attention_qk_matmul_calls: BF16_TENSOR_CORE_ATTENTION_QK_MATMUL_CALLS
            .load(Ordering::Relaxed),
        bf16_tensor_core_attention_av_matmul_calls: BF16_TENSOR_CORE_ATTENTION_AV_MATMUL_CALLS
            .load(Ordering::Relaxed),
        bf16_tensor_core_attention_backward_calls: BF16_TENSOR_CORE_ATTENTION_BACKWARD_CALLS
            .load(Ordering::Relaxed),
        bf16_tensor_core_attention_score_grad_matmul_calls:
            BF16_TENSOR_CORE_ATTENTION_SCORE_GRAD_MATMUL_CALLS.load(Ordering::Relaxed),
        bf16_tensor_core_attention_dq_matmul_calls: BF16_TENSOR_CORE_ATTENTION_DQ_MATMUL_CALLS
            .load(Ordering::Relaxed),
        bf16_tensor_core_attention_dk_matmul_calls: BF16_TENSOR_CORE_ATTENTION_DK_MATMUL_CALLS
            .load(Ordering::Relaxed),
        bf16_tensor_core_attention_dv_matmul_calls: BF16_TENSOR_CORE_ATTENTION_DV_MATMUL_CALLS
            .load(Ordering::Relaxed),
    }
}

pub fn reset_tensor_core_counters() {
    BF16_MMA_PROBE_CALLS.store(0, Ordering::Relaxed);
    BF16_TENSOR_CORE_MATMUL_CALLS.store(0, Ordering::Relaxed);
    BF16_TENSOR_CORE_MATMUL_FORWARD_CALLS.store(0, Ordering::Relaxed);
    BF16_TENSOR_CORE_MATMUL_BACKWARD_CALLS.store(0, Ordering::Relaxed);
    BF16_SCALAR_MATMUL_FALLBACK_CALLS.store(0, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_FORWARD_CALLS.store(0, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_QK_MATMUL_CALLS.store(0, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_AV_MATMUL_CALLS.store(0, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_BACKWARD_CALLS.store(0, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_SCORE_GRAD_MATMUL_CALLS.store(0, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_DQ_MATMUL_CALLS.store(0, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_DK_MATMUL_CALLS.store(0, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_DV_MATMUL_CALLS.store(0, Ordering::Relaxed);
}

fn kernel_launch_family_stats() -> &'static Mutex<HashMap<&'static str, KernelLaunchFamilyStats>> {
    CUDA_KERNEL_LAUNCH_FAMILIES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cuda_kernel_launch_families() -> Vec<CudaKernelLaunchFamilyCounter> {
    let stats = kernel_launch_family_stats()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut families = stats
        .iter()
        .map(|(label, stats)| CudaKernelLaunchFamilyCounter {
            label: (*label).to_string(),
            calls: stats.calls,
            elements: stats.elements,
        })
        .collect::<Vec<_>>();
    families.sort_by(|left, right| {
        right
            .calls
            .cmp(&left.calls)
            .then_with(|| right.elements.cmp(&left.elements))
            .then_with(|| left.label.cmp(&right.label))
    });
    families
}

pub fn cuda_runtime_counters() -> CudaRuntimeCounters {
    CudaRuntimeCounters {
        kernel_launch_calls: CUDA_KERNEL_LAUNCH_CALLS.load(Ordering::Relaxed),
        kernel_launch_elements: CUDA_KERNEL_LAUNCH_ELEMENTS.load(Ordering::Relaxed),
        kernel_launch_families: cuda_kernel_launch_families(),
        host_sync_calls: CUDA_HOST_SYNC_CALLS.load(Ordering::Relaxed),
        stream_create_calls: CUDA_STREAM_CREATE_CALLS.load(Ordering::Relaxed),
        stream_sync_calls: CUDA_STREAM_SYNC_CALLS.load(Ordering::Relaxed),
        event_create_calls: CUDA_EVENT_CREATE_CALLS.load(Ordering::Relaxed),
        event_record_calls: CUDA_EVENT_RECORD_CALLS.load(Ordering::Relaxed),
        event_query_calls: CUDA_EVENT_QUERY_CALLS.load(Ordering::Relaxed),
        event_elapsed_calls: CUDA_EVENT_ELAPSED_CALLS.load(Ordering::Relaxed),
        h2d_bytes: CUDA_H2D_BYTES.load(Ordering::Relaxed),
        d2h_bytes: CUDA_D2H_BYTES.load(Ordering::Relaxed),
        d2d_bytes: CUDA_D2D_BYTES.load(Ordering::Relaxed),
        allocation_active_bytes: CUDA_ALLOC_ACTIVE_BYTES.load(Ordering::Relaxed),
        allocation_reserved_bytes: CUDA_ALLOC_RESERVED_BYTES.load(Ordering::Relaxed),
        allocation_high_water_bytes: CUDA_ALLOC_HIGH_WATER_BYTES.load(Ordering::Relaxed),
        allocation_calls: CUDA_ALLOC_CALLS.load(Ordering::Relaxed),
        allocation_cache_hits: CUDA_ALLOC_CACHE_HITS.load(Ordering::Relaxed),
        allocation_frees: CUDA_ALLOC_FREES.load(Ordering::Relaxed),
        allocation_deferred_frees: CUDA_ALLOC_DEFERRED_FREES.load(Ordering::Relaxed),
        allocation_pending_reclaims: CUDA_ALLOC_PENDING_RECLAIMS.load(Ordering::Relaxed),
        module_load_calls: CUDA_MODULE_LOAD_CALLS.load(Ordering::Relaxed),
        module_cache_hits: CUDA_MODULE_CACHE_HITS.load(Ordering::Relaxed),
        tensor_core_padded_tiles: CUDA_TENSOR_CORE_PADDED_TILES.load(Ordering::Relaxed),
        tensor_core_remainder_tiles: CUDA_TENSOR_CORE_REMAINDER_TILES.load(Ordering::Relaxed),
        tensor_core_cta_gemm_calls: CUDA_TENSOR_CORE_CTA_GEMM_CALLS.load(Ordering::Relaxed),
        tensor_core_cta_tiles: CUDA_TENSOR_CORE_CTA_TILES.load(Ordering::Relaxed),
        tensor_core_cta_warps_launched: CUDA_TENSOR_CORE_CTA_WARPS_LAUNCHED.load(Ordering::Relaxed),
        tensor_core_mma_warp_tiles: CUDA_TENSOR_CORE_MMA_WARP_TILES.load(Ordering::Relaxed),
        tensor_core_staged_cta_gemm_calls: CUDA_TENSOR_CORE_STAGED_CTA_GEMM_CALLS
            .load(Ordering::Relaxed),
        tensor_core_staged_cta_gemm_elapsed_us: CUDA_TENSOR_CORE_STAGED_CTA_GEMM_ELAPSED_US
            .load(Ordering::Relaxed),
        tensor_core_shared_stage_tiles: CUDA_TENSOR_CORE_SHARED_STAGE_TILES.load(Ordering::Relaxed),
        tensor_core_shared_stage_bytes: CUDA_TENSOR_CORE_SHARED_STAGE_BYTES.load(Ordering::Relaxed),
        tensor_core_wide_swizzled_cta_gemm_calls: CUDA_TENSOR_CORE_WIDE_SWIZZLED_CTA_GEMM_CALLS
            .load(Ordering::Relaxed),
        tensor_core_wide_swizzled_cta_gemm_elapsed_us:
            CUDA_TENSOR_CORE_WIDE_SWIZZLED_CTA_GEMM_ELAPSED_US.load(Ordering::Relaxed),
        tensor_core_swizzled_stage_tiles: CUDA_TENSOR_CORE_SWIZZLED_STAGE_TILES
            .load(Ordering::Relaxed),
        tensor_core_swizzled_stage_bytes: CUDA_TENSOR_CORE_SWIZZLED_STAGE_BYTES
            .load(Ordering::Relaxed),
        tensor_core_ldmatrix_gemm_requested_calls: CUDA_TENSOR_CORE_LDMATRIX_GEMM_REQUESTED_CALLS
            .load(Ordering::Relaxed),
        tensor_core_ldmatrix_gemm_executed_calls: CUDA_TENSOR_CORE_LDMATRIX_GEMM_EXECUTED_CALLS
            .load(Ordering::Relaxed),
        tensor_core_ldmatrix_gemm_staged_fallback_calls:
            CUDA_TENSOR_CORE_LDMATRIX_GEMM_STAGED_FALLBACK_CALLS.load(Ordering::Relaxed),
        tensor_core_ldmatrix_gemm_hard_require_failures:
            CUDA_TENSOR_CORE_LDMATRIX_GEMM_HARD_REQUIRE_FAILURES.load(Ordering::Relaxed),
        tensor_core_ldmatrix_gemm_instructions: CUDA_TENSOR_CORE_LDMATRIX_GEMM_INSTRUCTIONS
            .load(Ordering::Relaxed),
        tensor_core_ldmatrix_gemm_elapsed_us: CUDA_TENSOR_CORE_LDMATRIX_GEMM_ELAPSED_US
            .load(Ordering::Relaxed),
        tensor_core_cp_async_gemm_requested_calls: CUDA_TENSOR_CORE_CP_ASYNC_GEMM_REQUESTED_CALLS
            .load(Ordering::Relaxed),
        tensor_core_cp_async_gemm_executed_calls: CUDA_TENSOR_CORE_CP_ASYNC_GEMM_EXECUTED_CALLS
            .load(Ordering::Relaxed),
        tensor_core_cp_async_gemm_staged_fallback_calls:
            CUDA_TENSOR_CORE_CP_ASYNC_GEMM_STAGED_FALLBACK_CALLS.load(Ordering::Relaxed),
        tensor_core_cp_async_gemm_hard_require_failures:
            CUDA_TENSOR_CORE_CP_ASYNC_GEMM_HARD_REQUIRE_FAILURES.load(Ordering::Relaxed),
        tensor_core_cp_async_gemm_instructions: CUDA_TENSOR_CORE_CP_ASYNC_GEMM_INSTRUCTIONS
            .load(Ordering::Relaxed),
        tensor_core_cp_async_gemm_elapsed_us: CUDA_TENSOR_CORE_CP_ASYNC_GEMM_ELAPSED_US
            .load(Ordering::Relaxed),
        tensor_core_global_cta_gemm_calls: CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM_CALLS
            .load(Ordering::Relaxed),
        tensor_core_global_cta_gemm_elapsed_us: CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM_ELAPSED_US
            .load(Ordering::Relaxed),
        tensor_core_legacy_warp_gemm_calls: CUDA_TENSOR_CORE_LEGACY_WARP_GEMM_CALLS
            .load(Ordering::Relaxed),
        tensor_core_legacy_warp_gemm_elapsed_us: CUDA_TENSOR_CORE_LEGACY_WARP_GEMM_ELAPSED_US
            .load(Ordering::Relaxed),
        tensor_core_attention_padded_tiles: CUDA_TENSOR_CORE_ATTENTION_PADDED_TILES
            .load(Ordering::Relaxed),
        tensor_core_attention_remainder_tiles: CUDA_TENSOR_CORE_ATTENTION_REMAINDER_TILES
            .load(Ordering::Relaxed),
        tensor_core_attention_scalar_fallbacks: CUDA_TENSOR_CORE_ATTENTION_SCALAR_FALLBACKS
            .load(Ordering::Relaxed),
        tensor_core_attention_ldmatrix_requested_calls:
            CUDA_TENSOR_CORE_ATTENTION_LDMATRIX_REQUESTED_CALLS.load(Ordering::Relaxed),
        tensor_core_attention_ldmatrix_global_fallback_calls:
            CUDA_TENSOR_CORE_ATTENTION_LDMATRIX_GLOBAL_FALLBACK_CALLS.load(Ordering::Relaxed),
        tensor_core_attention_cp_async_requested_calls:
            CUDA_TENSOR_CORE_ATTENTION_CP_ASYNC_REQUESTED_CALLS.load(Ordering::Relaxed),
        tensor_core_attention_cp_async_global_fallback_calls:
            CUDA_TENSOR_CORE_ATTENTION_CP_ASYNC_GLOBAL_FALLBACK_CALLS.load(Ordering::Relaxed),
        bf16_attention_materialized_reference_calls:
            CUDA_BF16_ATTENTION_MATERIALIZED_REFERENCE_CALLS.load(Ordering::Relaxed),
        flash_bf16_attention_requested_calls: CUDA_FLASH_BF16_ATTENTION_REQUESTED_CALLS
            .load(Ordering::Relaxed),
        flash_bf16_attention_executed_calls: CUDA_FLASH_BF16_ATTENTION_EXECUTED_CALLS
            .load(Ordering::Relaxed),
        flash_bf16_attention_fallback_calls: CUDA_FLASH_BF16_ATTENTION_FALLBACK_CALLS
            .load(Ordering::Relaxed),
        flash_bf16_attention_scalar_fallback_calls: CUDA_FLASH_BF16_ATTENTION_SCALAR_FALLBACK_CALLS
            .load(Ordering::Relaxed),
        flash_bf16_attention_qk_tile_calls: CUDA_FLASH_BF16_ATTENTION_QK_TILE_CALLS
            .load(Ordering::Relaxed),
        flash_bf16_attention_av_tile_calls: CUDA_FLASH_BF16_ATTENTION_AV_TILE_CALLS
            .load(Ordering::Relaxed),
        flash_bf16_attention_ragged_tile_count: CUDA_FLASH_BF16_ATTENTION_RAGGED_TILE_COUNT
            .load(Ordering::Relaxed),
        flash_bf16_attention_causal_masked_tile_count:
            CUDA_FLASH_BF16_ATTENTION_CAUSAL_MASKED_TILE_COUNT.load(Ordering::Relaxed),
        flash_bf16_attention_elapsed_us: CUDA_FLASH_BF16_ATTENTION_ELAPSED_US
            .load(Ordering::Relaxed),
        flash_bf16_attention_hard_require_failures: CUDA_FLASH_BF16_ATTENTION_HARD_REQUIRE_FAILURES
            .load(Ordering::Relaxed),
        flash_bf16_scalar_streaming_requested_calls:
            CUDA_FLASH_BF16_SCALAR_STREAMING_REQUESTED_CALLS.load(Ordering::Relaxed),
        flash_bf16_scalar_streaming_executed_calls: CUDA_FLASH_BF16_SCALAR_STREAMING_EXECUTED_CALLS
            .load(Ordering::Relaxed),
        flash_bf16_scalar_streaming_qk_tile_calls: CUDA_FLASH_BF16_SCALAR_STREAMING_QK_TILE_CALLS
            .load(Ordering::Relaxed),
        flash_bf16_scalar_streaming_av_tile_calls: CUDA_FLASH_BF16_SCALAR_STREAMING_AV_TILE_CALLS
            .load(Ordering::Relaxed),
        flash_bf16_scalar_streaming_elapsed_us: CUDA_FLASH_BF16_SCALAR_STREAMING_ELAPSED_US
            .load(Ordering::Relaxed),
        flash_bf16_tensor_core_requested_calls: CUDA_FLASH_BF16_TENSOR_CORE_REQUESTED_CALLS
            .load(Ordering::Relaxed),
        flash_bf16_tensor_core_executed_calls: CUDA_FLASH_BF16_TENSOR_CORE_EXECUTED_CALLS
            .load(Ordering::Relaxed),
        flash_bf16_tensor_core_fallback_calls: CUDA_FLASH_BF16_TENSOR_CORE_FALLBACK_CALLS
            .load(Ordering::Relaxed),
        flash_bf16_tensor_core_qk_mma_tile_calls: CUDA_FLASH_BF16_TENSOR_CORE_QK_MMA_TILE_CALLS
            .load(Ordering::Relaxed),
        flash_bf16_tensor_core_av_mma_tile_calls: CUDA_FLASH_BF16_TENSOR_CORE_AV_MMA_TILE_CALLS
            .load(Ordering::Relaxed),
        flash_bf16_tensor_core_ragged_tile_count: CUDA_FLASH_BF16_TENSOR_CORE_RAGGED_TILE_COUNT
            .load(Ordering::Relaxed),
        flash_bf16_tensor_core_causal_masked_tile_count:
            CUDA_FLASH_BF16_TENSOR_CORE_CAUSAL_MASKED_TILE_COUNT.load(Ordering::Relaxed),
        flash_bf16_tensor_core_elapsed_us: CUDA_FLASH_BF16_TENSOR_CORE_ELAPSED_US
            .load(Ordering::Relaxed),
        flash_bf16_tensor_core_backward_requested_calls:
            CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_REQUESTED_CALLS.load(Ordering::Relaxed),
        flash_bf16_tensor_core_backward_executed_calls:
            CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_EXECUTED_CALLS.load(Ordering::Relaxed),
        flash_bf16_tensor_core_backward_fallback_calls:
            CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_FALLBACK_CALLS.load(Ordering::Relaxed),
        flash_bf16_tensor_core_backward_row_dot_calls:
            CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_ROW_DOT_CALLS.load(Ordering::Relaxed),
        flash_bf16_tensor_core_backward_qk_recompute_mma_tile_calls:
            CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_QK_RECOMPUTE_MMA_TILE_CALLS.load(Ordering::Relaxed),
        flash_bf16_tensor_core_backward_dp_mma_tile_calls:
            CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_DP_MMA_TILE_CALLS.load(Ordering::Relaxed),
        flash_bf16_tensor_core_backward_dq_mma_tile_calls:
            CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_DQ_MMA_TILE_CALLS.load(Ordering::Relaxed),
        flash_bf16_tensor_core_backward_dk_mma_tile_calls:
            CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_DK_MMA_TILE_CALLS.load(Ordering::Relaxed),
        flash_bf16_tensor_core_backward_dv_mma_tile_calls:
            CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_DV_MMA_TILE_CALLS.load(Ordering::Relaxed),
        flash_bf16_tensor_core_backward_scalar_tile_calls:
            CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_SCALAR_TILE_CALLS.load(Ordering::Relaxed),
        flash_bf16_tensor_core_backward_ragged_tile_count:
            CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_RAGGED_TILE_COUNT.load(Ordering::Relaxed),
        flash_bf16_tensor_core_backward_causal_masked_tile_count:
            CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_CAUSAL_MASKED_TILE_COUNT.load(Ordering::Relaxed),
        flash_bf16_tensor_core_backward_elapsed_us: CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_ELAPSED_US
            .load(Ordering::Relaxed),
    }
}

pub fn reset_cuda_runtime_counters() {
    CUDA_KERNEL_LAUNCH_CALLS.store(0, Ordering::Relaxed);
    CUDA_KERNEL_LAUNCH_ELEMENTS.store(0, Ordering::Relaxed);
    kernel_launch_family_stats()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
    CUDA_HOST_SYNC_CALLS.store(0, Ordering::Relaxed);
    CUDA_STREAM_CREATE_CALLS.store(0, Ordering::Relaxed);
    CUDA_STREAM_SYNC_CALLS.store(0, Ordering::Relaxed);
    CUDA_EVENT_CREATE_CALLS.store(0, Ordering::Relaxed);
    CUDA_EVENT_RECORD_CALLS.store(0, Ordering::Relaxed);
    CUDA_EVENT_QUERY_CALLS.store(0, Ordering::Relaxed);
    CUDA_EVENT_ELAPSED_CALLS.store(0, Ordering::Relaxed);
    CUDA_H2D_BYTES.store(0, Ordering::Relaxed);
    CUDA_D2H_BYTES.store(0, Ordering::Relaxed);
    CUDA_D2D_BYTES.store(0, Ordering::Relaxed);
    CUDA_ALLOC_ACTIVE_BYTES.store(0, Ordering::Relaxed);
    CUDA_ALLOC_RESERVED_BYTES.store(0, Ordering::Relaxed);
    CUDA_ALLOC_HIGH_WATER_BYTES.store(0, Ordering::Relaxed);
    CUDA_ALLOC_CALLS.store(0, Ordering::Relaxed);
    CUDA_ALLOC_CACHE_HITS.store(0, Ordering::Relaxed);
    CUDA_ALLOC_FREES.store(0, Ordering::Relaxed);
    CUDA_ALLOC_DEFERRED_FREES.store(0, Ordering::Relaxed);
    CUDA_ALLOC_PENDING_RECLAIMS.store(0, Ordering::Relaxed);
    CUDA_MODULE_LOAD_CALLS.store(0, Ordering::Relaxed);
    CUDA_MODULE_CACHE_HITS.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_PADDED_TILES.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_REMAINDER_TILES.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_CTA_GEMM_CALLS.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_CTA_TILES.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_CTA_WARPS_LAUNCHED.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_MMA_WARP_TILES.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_STAGED_CTA_GEMM_CALLS.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_STAGED_CTA_GEMM_ELAPSED_US.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_SHARED_STAGE_TILES.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_SHARED_STAGE_BYTES.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_WIDE_SWIZZLED_CTA_GEMM_CALLS.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_WIDE_SWIZZLED_CTA_GEMM_ELAPSED_US.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_SWIZZLED_STAGE_TILES.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_SWIZZLED_STAGE_BYTES.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_LDMATRIX_GEMM_REQUESTED_CALLS.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_LDMATRIX_GEMM_EXECUTED_CALLS.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_LDMATRIX_GEMM_STAGED_FALLBACK_CALLS.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_LDMATRIX_GEMM_HARD_REQUIRE_FAILURES.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_LDMATRIX_GEMM_INSTRUCTIONS.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_LDMATRIX_GEMM_ELAPSED_US.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_CP_ASYNC_GEMM_REQUESTED_CALLS.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_CP_ASYNC_GEMM_EXECUTED_CALLS.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_CP_ASYNC_GEMM_STAGED_FALLBACK_CALLS.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_CP_ASYNC_GEMM_HARD_REQUIRE_FAILURES.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_CP_ASYNC_GEMM_INSTRUCTIONS.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_CP_ASYNC_GEMM_ELAPSED_US.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM_CALLS.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM_ELAPSED_US.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_LEGACY_WARP_GEMM_CALLS.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_LEGACY_WARP_GEMM_ELAPSED_US.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_ATTENTION_PADDED_TILES.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_ATTENTION_REMAINDER_TILES.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_ATTENTION_SCALAR_FALLBACKS.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_ATTENTION_LDMATRIX_REQUESTED_CALLS.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_ATTENTION_LDMATRIX_GLOBAL_FALLBACK_CALLS.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_ATTENTION_CP_ASYNC_REQUESTED_CALLS.store(0, Ordering::Relaxed);
    CUDA_TENSOR_CORE_ATTENTION_CP_ASYNC_GLOBAL_FALLBACK_CALLS.store(0, Ordering::Relaxed);
    CUDA_BF16_ATTENTION_MATERIALIZED_REFERENCE_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_ATTENTION_REQUESTED_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_ATTENTION_EXECUTED_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_ATTENTION_FALLBACK_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_ATTENTION_SCALAR_FALLBACK_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_ATTENTION_QK_TILE_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_ATTENTION_AV_TILE_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_ATTENTION_RAGGED_TILE_COUNT.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_ATTENTION_CAUSAL_MASKED_TILE_COUNT.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_ATTENTION_ELAPSED_US.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_ATTENTION_HARD_REQUIRE_FAILURES.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_SCALAR_STREAMING_REQUESTED_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_SCALAR_STREAMING_EXECUTED_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_SCALAR_STREAMING_QK_TILE_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_SCALAR_STREAMING_AV_TILE_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_SCALAR_STREAMING_ELAPSED_US.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_REQUESTED_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_EXECUTED_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_FALLBACK_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_QK_MMA_TILE_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_AV_MMA_TILE_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_RAGGED_TILE_COUNT.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_CAUSAL_MASKED_TILE_COUNT.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_ELAPSED_US.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_REQUESTED_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_EXECUTED_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_FALLBACK_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_ROW_DOT_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_QK_RECOMPUTE_MMA_TILE_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_DP_MMA_TILE_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_DQ_MMA_TILE_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_DK_MMA_TILE_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_DV_MMA_TILE_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_SCALAR_TILE_CALLS.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_RAGGED_TILE_COUNT.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_CAUSAL_MASKED_TILE_COUNT.store(0, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_ELAPSED_US.store(0, Ordering::Relaxed);
}

pub fn memory_kernel_counters() -> MemoryKernelCounters {
    MemoryKernelCounters {
        lookup_rejected_calls: CUDA_MEMORY_LOOKUP_REJECTED_CALLS.load(Ordering::Relaxed),
        query_key_score_calls: CUDA_MEMORY_QUERY_KEY_SCORE_CALLS.load(Ordering::Relaxed),
        topk_calls: CUDA_MEMORY_TOPK_CALLS.load(Ordering::Relaxed),
        product_key_calls: CUDA_MEMORY_PRODUCT_KEY_CALLS.load(Ordering::Relaxed),
        softmax_topk_calls: CUDA_MEMORY_SOFTMAX_TOPK_CALLS.load(Ordering::Relaxed),
        weighted_value_forward_calls: CUDA_MEMORY_WEIGHTED_VALUE_FORWARD_CALLS
            .load(Ordering::Relaxed),
        weighted_value_backward_calls: CUDA_MEMORY_WEIGHTED_VALUE_BACKWARD_CALLS
            .load(Ordering::Relaxed),
        selected_key_backward_calls: CUDA_MEMORY_SELECTED_KEY_BACKWARD_CALLS
            .load(Ordering::Relaxed),
        scatter_add_rows_calls: CUDA_MEMORY_SCATTER_ADD_ROWS_CALLS.load(Ordering::Relaxed),
        sparse_adamw_rows_calls: CUDA_MEMORY_SPARSE_ADAMW_ROWS_CALLS.load(Ordering::Relaxed),
        sparse_adamw_compact_rows_calls: CUDA_MEMORY_SPARSE_ADAMW_COMPACT_ROWS_CALLS
            .load(Ordering::Relaxed),
        access_count_calls: CUDA_MEMORY_ACCESS_COUNT_CALLS.load(Ordering::Relaxed),
        gather_selected_rows_calls: CUDA_MEMORY_GATHER_SELECTED_ROWS_CALLS.load(Ordering::Relaxed),
        bool_mask_to_indices_calls: CUDA_MEMORY_BOOL_MASK_TO_INDICES_CALLS.load(Ordering::Relaxed),
        selected_tokens: CUDA_MEMORY_SELECTED_TOKENS.load(Ordering::Relaxed),
        selected_rows: CUDA_MEMORY_SELECTED_ROWS.load(Ordering::Relaxed),
    }
}

pub fn reset_memory_kernel_counters() {
    CUDA_MEMORY_LOOKUP_REJECTED_CALLS.store(0, Ordering::Relaxed);
    CUDA_MEMORY_QUERY_KEY_SCORE_CALLS.store(0, Ordering::Relaxed);
    CUDA_MEMORY_TOPK_CALLS.store(0, Ordering::Relaxed);
    CUDA_MEMORY_PRODUCT_KEY_CALLS.store(0, Ordering::Relaxed);
    CUDA_MEMORY_SOFTMAX_TOPK_CALLS.store(0, Ordering::Relaxed);
    CUDA_MEMORY_WEIGHTED_VALUE_FORWARD_CALLS.store(0, Ordering::Relaxed);
    CUDA_MEMORY_WEIGHTED_VALUE_BACKWARD_CALLS.store(0, Ordering::Relaxed);
    CUDA_MEMORY_SELECTED_KEY_BACKWARD_CALLS.store(0, Ordering::Relaxed);
    CUDA_MEMORY_SCATTER_ADD_ROWS_CALLS.store(0, Ordering::Relaxed);
    CUDA_MEMORY_SPARSE_ADAMW_ROWS_CALLS.store(0, Ordering::Relaxed);
    CUDA_MEMORY_SPARSE_ADAMW_COMPACT_ROWS_CALLS.store(0, Ordering::Relaxed);
    CUDA_MEMORY_ACCESS_COUNT_CALLS.store(0, Ordering::Relaxed);
    CUDA_MEMORY_GATHER_SELECTED_ROWS_CALLS.store(0, Ordering::Relaxed);
    CUDA_MEMORY_BOOL_MASK_TO_INDICES_CALLS.store(0, Ordering::Relaxed);
    CUDA_MEMORY_SELECTED_TOKENS.store(0, Ordering::Relaxed);
    CUDA_MEMORY_SELECTED_ROWS.store(0, Ordering::Relaxed);
}

pub fn record_memory_cuda_lookup_rejected(dims: MemoryLookupDims) -> CudaResult<()> {
    validate_memory_lookup_dims(dims, "record_memory_cuda_lookup_rejected")?;
    CUDA_MEMORY_LOOKUP_REJECTED_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_MEMORY_SELECTED_TOKENS.fetch_add(dims.tokens, Ordering::Relaxed);
    CUDA_MEMORY_SELECTED_ROWS.fetch_add(
        checked_mul(
            dims.tokens,
            dims.top_k,
            "memory rejected selected row count",
        )?,
        Ordering::Relaxed,
    );
    Ok(())
}

pub fn memory_topk_indices_f32(
    scores: &CudaBuffer,
    dims: MemoryTopkDims,
) -> CudaResult<CudaBuffer> {
    validate_memory_topk_dims(dims, "memory_topk_indices_f32")?;
    ensure_f32_buffer(scores, "memory_topk_indices_f32 scores")?;
    ensure_buffer_len(
        scores,
        checked_mul(dims.rows, dims.cols, "memory_topk_indices_f32 scores")?,
        "memory_topk_indices_f32 scores",
    )?;
    let output_len = checked_mul(dims.rows, dims.top_k, "memory_topk_indices_f32 output")?;
    let output = CudaBuffer::uninit_bytes(
        scores.device_ordinal(),
        output_len,
        std::mem::size_of::<i64>(),
    )?;
    if output_len == 0 {
        return Ok(output);
    }

    scores.inner.driver.set_current(scores.inner.context)?;
    let module = load_module(&scores.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_memory_topk_f32")?;
    launch_memory_topk_kernel(
        &scores.inner.driver,
        function,
        &[scores.inner.ptr, output.inner.ptr],
        dims,
    )?;
    CUDA_MEMORY_TOPK_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_MEMORY_SELECTED_TOKENS.fetch_add(dims.rows, Ordering::Relaxed);
    CUDA_MEMORY_SELECTED_ROWS.fetch_add(output_len, Ordering::Relaxed);
    Ok(output)
}

pub fn memory_access_count_rows_i64_u64(
    selected_rows: &CudaBuffer,
    memory_slots: usize,
) -> CudaResult<CudaBuffer> {
    ensure_i64_buffer(
        selected_rows,
        "memory_access_count_rows_i64_u64 selected_rows",
    )?;
    if memory_slots == 0 {
        return Err(CudaError::new(
            "memory_access_count_rows_i64_u64 requires non-zero memory_slots",
        ));
    }
    u32_kernel_dim(memory_slots, "memory_slots")?;
    let output = CudaBuffer::from_u64(selected_rows.device_ordinal(), &vec![0; memory_slots])?;
    if selected_rows.is_empty() {
        return Ok(output);
    }

    let status = CudaBuffer::from_u32(selected_rows.device_ordinal(), &[0])?;
    selected_rows
        .inner
        .driver
        .set_current(selected_rows.inner.context)?;
    let module = load_module(&selected_rows.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_memory_access_count_rows_i64_u64")?;
    launch_memory_access_count_kernel(
        &selected_rows.inner.driver,
        function,
        &[selected_rows.inner.ptr, output.inner.ptr, status.inner.ptr],
        selected_rows.len(),
        memory_slots,
    )?;
    ensure_status_zero(&status, "memory_access_count_rows_i64_u64")?;
    CUDA_MEMORY_ACCESS_COUNT_CALLS.fetch_add(1, Ordering::Relaxed);
    Ok(output)
}

pub fn i64_arange_buffer(device_ordinal: i32, len: usize) -> CudaResult<CudaBuffer> {
    u32_kernel_dim(len, "i64_arange len")?;
    let output = CudaBuffer::uninit_bytes(device_ordinal, len, std::mem::size_of::<i64>())?;
    if len == 0 {
        return Ok(output);
    }
    output.inner.driver.set_current(output.inner.context)?;
    {
        let module = load_module(&output.inner.driver, KERNEL_PTX)?;
        let function = module.function("heirloom_i64_arange")?;
        launch_i64_arange_kernel(&output.inner.driver, function, &[output.inner.ptr], len)?;
    }
    Ok(output)
}

pub fn memory_selected_rows_to_f32_mask(
    selected_rows: &CudaBuffer,
    rows: usize,
) -> CudaResult<CudaBuffer> {
    ensure_i64_buffer(
        selected_rows,
        "memory_selected_rows_to_f32_mask selected_rows",
    )?;
    if rows == 0 {
        return Err(CudaError::new(
            "memory_selected_rows_to_f32_mask requires non-zero rows",
        ));
    }
    u32_kernel_dim(rows, "rows")?;
    let output = CudaBuffer::from_f32(selected_rows.device_ordinal(), &vec![0.0; rows])?;
    if selected_rows.is_empty() {
        return Ok(output);
    }
    let status = CudaBuffer::from_u32(selected_rows.device_ordinal(), &[0])?;
    selected_rows
        .inner
        .driver
        .set_current(selected_rows.inner.context)?;
    let module = load_module(&selected_rows.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_memory_selected_rows_to_f32_mask")?;
    launch_selected_rows_to_f32_mask_kernel(
        &selected_rows.inner.driver,
        function,
        &[selected_rows.inner.ptr, output.inner.ptr, status.inner.ptr],
        selected_rows.len(),
        rows,
    )?;
    ensure_status_zero(&status, "memory_selected_rows_to_f32_mask")?;
    Ok(output)
}

pub fn memory_gather_selected_rows_f32_i64(
    table: &CudaBuffer,
    selected_rows: &CudaBuffer,
    dims: SelectedRowsDims,
) -> CudaResult<CudaBuffer> {
    validate_selected_rows_gather_buffers(
        table,
        selected_rows,
        dims,
        "memory_gather_selected_rows_f32_i64",
    )?;
    let output_len = checked_mul(
        dims.selected_rows,
        dims.row_dim,
        "memory selected-row gather output",
    )?;
    let output = CudaBuffer::uninit_bytes(
        table.device_ordinal(),
        output_len,
        std::mem::size_of::<f32>(),
    )?;
    if output_len == 0 {
        return Ok(output);
    }
    let status = CudaBuffer::from_u32(table.device_ordinal(), &[0])?;
    table.inner.driver.set_current(table.inner.context)?;
    let module = load_module(&table.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_memory_gather_selected_rows_f32_i64")?;
    launch_selected_rows_gather_kernel(
        &table.inner.driver,
        function,
        &[
            table.inner.ptr,
            selected_rows.inner.ptr,
            output.inner.ptr,
            status.inner.ptr,
        ],
        dims,
        output_len,
    )?;
    ensure_status_zero(&status, "memory_gather_selected_rows_f32_i64")?;
    CUDA_MEMORY_GATHER_SELECTED_ROWS_CALLS.fetch_add(1, Ordering::Relaxed);
    Ok(output)
}

pub fn f32_mask_to_bool_buffer(mask: &CudaBuffer) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(mask, "f32_mask_to_bool_buffer mask")?;
    let output =
        CudaBuffer::uninit_bytes(mask.device_ordinal(), mask.len(), std::mem::size_of::<u8>())?;
    if mask.is_empty() {
        return Ok(output);
    }
    mask.inner.driver.set_current(mask.inner.context)?;
    let module = load_module(&mask.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_f32_mask_to_bool")?;
    launch_f32_mask_to_bool_kernel(
        &mask.inner.driver,
        function,
        &[mask.inner.ptr, output.inner.ptr],
        mask.len(),
    )?;
    Ok(output)
}

pub fn bool_and_buffers(left: &CudaBuffer, right: &CudaBuffer) -> CudaResult<CudaBuffer> {
    ensure_u8_buffer(left, "bool_and_buffers left")?;
    ensure_u8_buffer(right, "bool_and_buffers right")?;
    ensure_same_device_len(left, right, "bool_and_buffers")?;
    let output =
        CudaBuffer::uninit_bytes(left.device_ordinal(), left.len(), std::mem::size_of::<u8>())?;
    if left.is_empty() {
        return Ok(output);
    }
    left.inner.driver.set_current(left.inner.context)?;
    {
        let module = load_module(&left.inner.driver, KERNEL_PTX)?;
        let function = module.function("heirloom_bool_and_u8")?;
        launch_bool_and_u8_kernel(
            &left.inner.driver,
            function,
            &[left.inner.ptr, right.inner.ptr, output.inner.ptr],
            left.len(),
        )?;
    }
    Ok(output)
}

pub fn bool_mask_to_i64_indices(mask: &CudaBuffer) -> CudaResult<(CudaBuffer, usize)> {
    ensure_u8_buffer(mask, "bool_mask_to_i64_indices mask")?;
    u32_kernel_dim(mask.len(), "bool_mask_to_i64_indices len")?;
    let scratch = CudaBuffer::uninit_bytes(
        mask.device_ordinal(),
        mask.len(),
        std::mem::size_of::<i64>(),
    )?;
    if mask.is_empty() {
        return Ok((scratch, 0));
    }
    let count = CudaBuffer::from_u32(mask.device_ordinal(), &[0])?;
    mask.inner.driver.set_current(mask.inner.context)?;
    {
        let module = load_module(&mask.inner.driver, KERNEL_PTX)?;
        let function = module.function("heirloom_bool_mask_to_i64_indices")?;
        launch_bool_mask_to_i64_indices_kernel(
            &mask.inner.driver,
            function,
            &[mask.inner.ptr, scratch.inner.ptr, count.inner.ptr],
            mask.len(),
        )?;
    }
    let count_values = count.to_u32()?;
    let count = count_values.first().copied().unwrap_or(0) as usize;
    if count > mask.len() {
        return Err(CudaError::new(format!(
            "bool_mask_to_i64_indices produced count {count} larger than mask length {}",
            mask.len()
        )));
    }
    let output = i64_prefix_buffer(&scratch, count)?;
    CUDA_MEMORY_BOOL_MASK_TO_INDICES_CALLS.fetch_add(1, Ordering::Relaxed);
    Ok((output, count))
}

pub fn memory_product_key_topk_indices_f32(
    query: &CudaBuffer,
    keys: &CudaBuffer,
    dims: MemoryProductKeyDims,
) -> CudaResult<CudaBuffer> {
    validate_memory_product_key_buffers(query, keys, dims, "memory_product_key_topk_indices_f32")?;
    let side = product_key_side_cuda(dims.slots, "memory_product_key_topk_indices_f32")?;
    let side_len = checked_mul(dims.tokens, side, "memory product-key side scores")?;
    let left_scores =
        CudaBuffer::uninit_bytes(query.device_ordinal(), side_len, std::mem::size_of::<f32>())?;
    let right_scores =
        CudaBuffer::uninit_bytes(query.device_ordinal(), side_len, std::mem::size_of::<f32>())?;

    query.inner.driver.set_current(query.inner.context)?;
    let module = load_module(&query.inner.driver, KERNEL_PTX)?;
    let side_function = module.function("heirloom_memory_product_key_side_scores_f32")?;
    launch_memory_product_key_side_scores_kernel(
        &query.inner.driver,
        side_function,
        &[
            query.inner.ptr,
            keys.inner.ptr,
            left_scores.inner.ptr,
            right_scores.inner.ptr,
        ],
        dims,
        side,
    )?;

    let side_topk_dims = MemoryTopkDims {
        rows: dims.tokens,
        cols: side,
        top_k: dims.beam,
    };
    let left_indices = memory_topk_indices_f32(&left_scores, side_topk_dims)?;
    let right_indices = memory_topk_indices_f32(&right_scores, side_topk_dims)?;
    CUDA_MEMORY_SELECTED_TOKENS.fetch_sub(dims.tokens * 2, Ordering::Relaxed);
    CUDA_MEMORY_SELECTED_ROWS.fetch_sub(dims.tokens * dims.beam * 2, Ordering::Relaxed);

    let output_len = checked_mul(dims.tokens, dims.top_k, "memory product-key output")?;
    let output = CudaBuffer::uninit_bytes(
        query.device_ordinal(),
        output_len,
        std::mem::size_of::<i64>(),
    )?;
    let status = CudaBuffer::from_u32(query.device_ordinal(), &[0])?;
    query.inner.driver.set_current(query.inner.context)?;
    let candidate_function = module.function("heirloom_memory_product_key_candidates_f32_i64")?;
    launch_memory_product_key_candidates_kernel(
        &query.inner.driver,
        candidate_function,
        &[
            query.inner.ptr,
            keys.inner.ptr,
            left_indices.inner.ptr,
            right_indices.inner.ptr,
            output.inner.ptr,
            status.inner.ptr,
        ],
        dims,
        side,
    )?;
    ensure_status_zero(&status, "memory_product_key_topk_indices_f32")?;
    CUDA_MEMORY_PRODUCT_KEY_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_MEMORY_QUERY_KEY_SCORE_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_MEMORY_SELECTED_TOKENS.fetch_add(dims.tokens, Ordering::Relaxed);
    CUDA_MEMORY_SELECTED_ROWS.fetch_add(output_len, Ordering::Relaxed);
    Ok(output)
}

pub fn memory_product_key_topk_indices_from_side_scores_f32(
    left_scores: &CudaBuffer,
    right_scores: &CudaBuffer,
    dims: MemoryProductKeyDims,
) -> CudaResult<CudaBuffer> {
    validate_memory_product_key_side_score_buffers(
        left_scores,
        right_scores,
        dims,
        "memory_product_key_topk_indices_from_side_scores_f32",
    )?;
    let side = product_key_side_cuda(
        dims.slots,
        "memory_product_key_topk_indices_from_side_scores_f32",
    )?;
    let side_topk_dims = MemoryTopkDims {
        rows: dims.tokens,
        cols: side,
        top_k: dims.beam,
    };
    let left_indices = memory_topk_indices_f32(left_scores, side_topk_dims)?;
    let right_indices = memory_topk_indices_f32(right_scores, side_topk_dims)?;
    CUDA_MEMORY_SELECTED_TOKENS.fetch_sub(dims.tokens * 2, Ordering::Relaxed);
    CUDA_MEMORY_SELECTED_ROWS.fetch_sub(dims.tokens * dims.beam * 2, Ordering::Relaxed);

    let output_len = checked_mul(
        dims.tokens,
        dims.top_k,
        "memory product-key side-score output",
    )?;
    let output = CudaBuffer::uninit_bytes(
        left_scores.device_ordinal(),
        output_len,
        std::mem::size_of::<i64>(),
    )?;
    let status = CudaBuffer::from_u32(left_scores.device_ordinal(), &[0])?;
    left_scores
        .inner
        .driver
        .set_current(left_scores.inner.context)?;
    let module = load_module(&left_scores.inner.driver, KERNEL_PTX)?;
    let candidate_function =
        module.function("heirloom_memory_product_key_candidates_from_scores_f32_i64")?;
    launch_memory_product_key_candidates_from_scores_kernel(
        &left_scores.inner.driver,
        candidate_function,
        &[
            left_scores.inner.ptr,
            right_scores.inner.ptr,
            left_indices.inner.ptr,
            right_indices.inner.ptr,
            output.inner.ptr,
            status.inner.ptr,
        ],
        dims,
        side,
    )?;
    ensure_status_zero(
        &status,
        "memory_product_key_topk_indices_from_side_scores_f32",
    )?;
    CUDA_MEMORY_PRODUCT_KEY_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_MEMORY_QUERY_KEY_SCORE_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_MEMORY_SELECTED_TOKENS.fetch_add(dims.tokens, Ordering::Relaxed);
    CUDA_MEMORY_SELECTED_ROWS.fetch_add(output_len, Ordering::Relaxed);
    Ok(output)
}

pub fn memory_product_key_split_rows_i64(
    selected_rows: &CudaBuffer,
    side: usize,
) -> CudaResult<(CudaBuffer, CudaBuffer)> {
    ensure_i64_buffer(
        selected_rows,
        "memory_product_key_split_rows_i64 selected rows",
    )?;
    if side == 0 {
        return Err(CudaError::new(
            "memory_product_key_split_rows_i64 requires non-zero side".to_string(),
        ));
    }
    let len = selected_rows.len();
    let left = CudaBuffer::uninit_bytes(
        selected_rows.device_ordinal(),
        len,
        std::mem::size_of::<i64>(),
    )?;
    let right = CudaBuffer::uninit_bytes(
        selected_rows.device_ordinal(),
        len,
        std::mem::size_of::<i64>(),
    )?;
    if len == 0 {
        return Ok((left, right));
    }
    let status = CudaBuffer::from_u32(selected_rows.device_ordinal(), &[0])?;
    selected_rows
        .inner
        .driver
        .set_current(selected_rows.inner.context)?;
    let module = load_module(&selected_rows.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_memory_product_key_split_rows_i64")?;
    launch_memory_product_key_split_rows_kernel(
        &selected_rows.inner.driver,
        function,
        &[
            selected_rows.inner.ptr,
            left.inner.ptr,
            right.inner.ptr,
            status.inner.ptr,
        ],
        len,
        side,
    )?;
    ensure_status_zero(&status, "memory_product_key_split_rows_i64")?;
    Ok((left, right))
}

pub fn memory_product_key_selected_scores_forward_f32_i64_buffers(
    indices: &CudaBuffer,
    query: &CudaBuffer,
    left_keys: &CudaBuffer,
    right_keys: &CudaBuffer,
    dims: MemoryProductKeySelectedScoreDims,
) -> CudaResult<CudaBuffer> {
    validate_memory_product_key_selected_score_buffers(
        indices,
        query,
        left_keys,
        right_keys,
        dims,
        "memory_product_key_selected_scores_forward_f32_i64_buffers",
    )?;
    let output_len = checked_mul(
        dims.tokens,
        dims.top_k,
        "memory product-key selected-score output",
    )?;
    let output = CudaBuffer::uninit_bytes(
        indices.device_ordinal(),
        output_len,
        std::mem::size_of::<f32>(),
    )?;
    if output_len == 0 {
        return Ok(output);
    }
    let status = CudaBuffer::from_u32(indices.device_ordinal(), &[0])?;
    indices.inner.driver.set_current(indices.inner.context)?;
    let module = load_module(&indices.inner.driver, KERNEL_PTX)?;
    let function =
        module.function("heirloom_memory_product_key_selected_scores_forward_f32_i64")?;
    launch_memory_product_key_selected_score_kernel(
        &indices.inner.driver,
        function,
        &[
            indices.inner.ptr,
            query.inner.ptr,
            left_keys.inner.ptr,
            right_keys.inner.ptr,
            output.inner.ptr,
            status.inner.ptr,
        ],
        dims,
        output_len,
    )?;
    ensure_status_zero(
        &status,
        "memory_product_key_selected_scores_forward_f32_i64_buffers",
    )?;
    Ok(output)
}

pub fn memory_product_key_selected_scores_backward_query_f32_i64_buffers(
    indices: &CudaBuffer,
    left_keys: &CudaBuffer,
    right_keys: &CudaBuffer,
    grad_scores: &CudaBuffer,
    dims: MemoryProductKeySelectedScoreDims,
) -> CudaResult<CudaBuffer> {
    ensure_i64_buffer(
        indices,
        "memory_product_key_selected_scores_backward_query_f32_i64_buffers indices",
    )?;
    ensure_f32_buffer(
        left_keys,
        "memory_product_key_selected_scores_backward_query_f32_i64_buffers left_keys",
    )?;
    ensure_f32_buffer(
        right_keys,
        "memory_product_key_selected_scores_backward_query_f32_i64_buffers right_keys",
    )?;
    ensure_f32_buffer(
        grad_scores,
        "memory_product_key_selected_scores_backward_query_f32_i64_buffers grad_scores",
    )?;
    ensure_same_device(
        indices,
        left_keys,
        "memory_product_key_selected_scores_backward_query_f32_i64_buffers",
    )?;
    ensure_same_device(
        indices,
        right_keys,
        "memory_product_key_selected_scores_backward_query_f32_i64_buffers",
    )?;
    ensure_same_device(
        indices,
        grad_scores,
        "memory_product_key_selected_scores_backward_query_f32_i64_buffers",
    )?;
    validate_memory_product_key_selected_score_dims(
        dims,
        "memory_product_key_selected_scores_backward_query_f32_i64_buffers",
    )?;
    ensure_product_key_selected_score_lengths(
        indices,
        None,
        left_keys,
        right_keys,
        Some(grad_scores),
        dims,
        "memory_product_key_selected_scores_backward_query_f32_i64_buffers",
    )?;
    let output_len = checked_mul(
        dims.tokens,
        dims.key_dim,
        "memory product-key selected-score backward query output",
    )?;
    let output = CudaBuffer::uninit_bytes(
        indices.device_ordinal(),
        output_len,
        std::mem::size_of::<f32>(),
    )?;
    let status = CudaBuffer::from_u32(indices.device_ordinal(), &[0])?;
    indices.inner.driver.set_current(indices.inner.context)?;
    let module = load_module(&indices.inner.driver, KERNEL_PTX)?;
    let function =
        module.function("heirloom_memory_product_key_selected_scores_backward_query_f32_i64")?;
    launch_memory_product_key_selected_score_kernel(
        &indices.inner.driver,
        function,
        &[
            indices.inner.ptr,
            left_keys.inner.ptr,
            right_keys.inner.ptr,
            grad_scores.inner.ptr,
            output.inner.ptr,
            status.inner.ptr,
        ],
        dims,
        output_len,
    )?;
    ensure_status_zero(
        &status,
        "memory_product_key_selected_scores_backward_query_f32_i64_buffers",
    )?;
    Ok(output)
}

pub fn memory_product_key_selected_scores_backward_half_keys_f32_i64_buffers(
    indices: &CudaBuffer,
    query: &CudaBuffer,
    grad_scores: &CudaBuffer,
    dims: MemoryProductKeySelectedScoreDims,
    left: bool,
) -> CudaResult<CudaBuffer> {
    ensure_i64_buffer(
        indices,
        "memory_product_key_selected_scores_backward_half_keys_f32_i64_buffers indices",
    )?;
    ensure_f32_buffer(
        query,
        "memory_product_key_selected_scores_backward_half_keys_f32_i64_buffers query",
    )?;
    ensure_f32_buffer(
        grad_scores,
        "memory_product_key_selected_scores_backward_half_keys_f32_i64_buffers grad_scores",
    )?;
    ensure_same_device(
        indices,
        query,
        "memory_product_key_selected_scores_backward_half_keys_f32_i64_buffers",
    )?;
    ensure_same_device(
        indices,
        grad_scores,
        "memory_product_key_selected_scores_backward_half_keys_f32_i64_buffers",
    )?;
    validate_memory_product_key_selected_score_dims(
        dims,
        "memory_product_key_selected_scores_backward_half_keys_f32_i64_buffers",
    )?;
    ensure_buffer_len(
        indices,
        checked_mul(
            dims.tokens,
            dims.top_k,
            "memory product-key selected-score backward half indices",
        )?,
        "memory_product_key_selected_scores_backward_half_keys_f32_i64_buffers indices",
    )?;
    ensure_buffer_len(
        query,
        checked_mul(
            dims.tokens,
            dims.key_dim,
            "memory product-key selected-score backward half query",
        )?,
        "memory_product_key_selected_scores_backward_half_keys_f32_i64_buffers query",
    )?;
    ensure_buffer_len(
        grad_scores,
        checked_mul(
            dims.tokens,
            dims.top_k,
            "memory product-key selected-score backward half grad_scores",
        )?,
        "memory_product_key_selected_scores_backward_half_keys_f32_i64_buffers grad_scores",
    )?;
    let half_dim = dims.key_dim / 2;
    let output_len = checked_mul(
        dims.side,
        half_dim,
        "memory product-key selected-score backward half-key output",
    )?;
    let output = fill_constant_f32_buffer(indices.device_ordinal(), output_len, 0.0)?;
    let status = CudaBuffer::from_u32(indices.device_ordinal(), &[0])?;
    indices.inner.driver.set_current(indices.inner.context)?;
    let module = load_module(&indices.inner.driver, KERNEL_PTX)?;
    let function = module
        .function("heirloom_memory_product_key_selected_scores_backward_half_keys_f32_i64")?;
    launch_memory_product_key_selected_score_half_key_kernel(
        &indices.inner.driver,
        function,
        &[
            indices.inner.ptr,
            query.inner.ptr,
            grad_scores.inner.ptr,
            output.inner.ptr,
            status.inner.ptr,
        ],
        dims,
        output_len,
        left,
    )?;
    ensure_status_zero(
        &status,
        "memory_product_key_selected_scores_backward_half_keys_f32_i64_buffers",
    )?;
    Ok(output)
}

pub fn softmax_dim1_f32_buffer(
    input: &CudaBuffer,
    rows: usize,
    cols: usize,
) -> CudaResult<CudaBuffer> {
    validate_rank2_buffer(input, rows, cols, "softmax_dim1_f32_buffer input")?;
    let len = checked_mul(rows, cols, "softmax_dim1_f32_buffer output")?;
    let output = CudaBuffer::uninit_bytes(input.device_ordinal(), len, std::mem::size_of::<f32>())?;
    if len == 0 {
        return Ok(output);
    }
    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_softmax_dim1_f32")?;
    launch_rank2_kernel(
        &input.inner.driver,
        function,
        &[input.inner.ptr, output.inner.ptr],
        rows,
        cols,
    )?;
    Ok(output)
}

pub fn softmax_dim1_backward_f32_buffer(
    input: &CudaBuffer,
    grad_output: &CudaBuffer,
    rows: usize,
    cols: usize,
) -> CudaResult<CudaBuffer> {
    validate_rank2_buffer(input, rows, cols, "softmax_dim1_backward_f32_buffer input")?;
    validate_rank2_buffer(
        grad_output,
        rows,
        cols,
        "softmax_dim1_backward_f32_buffer grad_output",
    )?;
    ensure_same_device(input, grad_output, "softmax_dim1_backward_f32_buffer")?;
    let len = checked_mul(rows, cols, "softmax_dim1_backward_f32_buffer output")?;
    let output = CudaBuffer::uninit_bytes(input.device_ordinal(), len, std::mem::size_of::<f32>())?;
    if len == 0 {
        return Ok(output);
    }
    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_softmax_dim1_backward_f32")?;
    launch_rank2_kernel(
        &input.inner.driver,
        function,
        &[input.inner.ptr, grad_output.inner.ptr, output.inner.ptr],
        rows,
        cols,
    )?;
    Ok(output)
}

pub fn memory_weighted_value_forward_f32_i64_buffers(
    indices: &CudaBuffer,
    weights: &CudaBuffer,
    values: &CudaBuffer,
    dims: MemoryWeightedValueDims,
) -> CudaResult<CudaBuffer> {
    validate_memory_weighted_value_buffers(
        indices,
        weights,
        values,
        dims,
        "memory_weighted_value_forward_f32_i64_buffers",
    )?;
    let output_len = checked_mul(
        dims.tokens,
        dims.value_dim,
        "memory weighted value forward output",
    )?;
    let output = CudaBuffer::uninit_bytes(
        indices.device_ordinal(),
        output_len,
        std::mem::size_of::<f32>(),
    )?;
    if output_len == 0 {
        return Ok(output);
    }
    let status = CudaBuffer::from_u32(indices.device_ordinal(), &[0])?;
    indices.inner.driver.set_current(indices.inner.context)?;
    let module = load_module(&indices.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_memory_weighted_value_forward_f32_i64")?;
    launch_memory_weighted_value_kernel(
        &indices.inner.driver,
        function,
        &[
            indices.inner.ptr,
            weights.inner.ptr,
            values.inner.ptr,
            output.inner.ptr,
            status.inner.ptr,
        ],
        dims,
        output_len,
    )?;
    ensure_status_zero(&status, "memory_weighted_value_forward_f32_i64_buffers")?;
    CUDA_MEMORY_WEIGHTED_VALUE_FORWARD_CALLS.fetch_add(1, Ordering::Relaxed);
    Ok(output)
}

pub fn memory_weighted_value_backward_weights_f32_i64_buffers(
    indices: &CudaBuffer,
    values: &CudaBuffer,
    grad_output: &CudaBuffer,
    dims: MemoryWeightedValueDims,
) -> CudaResult<CudaBuffer> {
    ensure_i64_buffer(
        indices,
        "memory_weighted_value_backward_weights_f32_i64_buffers indices",
    )?;
    ensure_f32_buffer(
        values,
        "memory_weighted_value_backward_weights_f32_i64_buffers values",
    )?;
    ensure_f32_buffer(
        grad_output,
        "memory_weighted_value_backward_weights_f32_i64_buffers grad_output",
    )?;
    ensure_same_device(
        indices,
        values,
        "memory_weighted_value_backward_weights_f32_i64_buffers",
    )?;
    ensure_same_device(
        indices,
        grad_output,
        "memory_weighted_value_backward_weights_f32_i64_buffers",
    )?;
    validate_memory_weighted_value_dims(
        dims,
        "memory_weighted_value_backward_weights_f32_i64_buffers",
    )?;
    ensure_buffer_len(
        indices,
        checked_mul(
            dims.tokens,
            dims.top_k,
            "memory weighted backward weights indices",
        )?,
        "memory_weighted_value_backward_weights_f32_i64_buffers indices",
    )?;
    ensure_buffer_len(
        values,
        checked_mul(
            dims.slots,
            dims.value_dim,
            "memory weighted backward weights values",
        )?,
        "memory_weighted_value_backward_weights_f32_i64_buffers values",
    )?;
    ensure_buffer_len(
        grad_output,
        checked_mul(
            dims.tokens,
            dims.value_dim,
            "memory weighted backward weights grad_output",
        )?,
        "memory_weighted_value_backward_weights_f32_i64_buffers grad_output",
    )?;
    let output_len = checked_mul(
        dims.tokens,
        dims.top_k,
        "memory weighted backward weights output",
    )?;
    let output = CudaBuffer::uninit_bytes(
        indices.device_ordinal(),
        output_len,
        std::mem::size_of::<f32>(),
    )?;
    if output_len == 0 {
        return Ok(output);
    }
    let status = CudaBuffer::from_u32(indices.device_ordinal(), &[0])?;
    indices.inner.driver.set_current(indices.inner.context)?;
    let module = load_module(&indices.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_memory_weighted_value_backward_weights_f32_i64")?;
    launch_memory_weighted_value_kernel(
        &indices.inner.driver,
        function,
        &[
            indices.inner.ptr,
            values.inner.ptr,
            grad_output.inner.ptr,
            output.inner.ptr,
            status.inner.ptr,
        ],
        dims,
        output_len,
    )?;
    ensure_status_zero(
        &status,
        "memory_weighted_value_backward_weights_f32_i64_buffers",
    )?;
    CUDA_MEMORY_WEIGHTED_VALUE_BACKWARD_CALLS.fetch_add(1, Ordering::Relaxed);
    Ok(output)
}

pub fn memory_weighted_value_backward_values_f32_i64_buffers(
    indices: &CudaBuffer,
    weights: &CudaBuffer,
    grad_output: &CudaBuffer,
    dims: MemoryWeightedValueDims,
) -> CudaResult<CudaBuffer> {
    ensure_i64_buffer(
        indices,
        "memory_weighted_value_backward_values_f32_i64_buffers indices",
    )?;
    ensure_f32_buffer(
        weights,
        "memory_weighted_value_backward_values_f32_i64_buffers weights",
    )?;
    ensure_f32_buffer(
        grad_output,
        "memory_weighted_value_backward_values_f32_i64_buffers grad_output",
    )?;
    ensure_same_device(
        indices,
        weights,
        "memory_weighted_value_backward_values_f32_i64_buffers",
    )?;
    ensure_same_device(
        indices,
        grad_output,
        "memory_weighted_value_backward_values_f32_i64_buffers",
    )?;
    validate_memory_weighted_value_dims(
        dims,
        "memory_weighted_value_backward_values_f32_i64_buffers",
    )?;
    ensure_buffer_len(
        indices,
        checked_mul(
            dims.tokens,
            dims.top_k,
            "memory weighted backward values indices",
        )?,
        "memory_weighted_value_backward_values_f32_i64_buffers indices",
    )?;
    ensure_buffer_len(
        weights,
        checked_mul(
            dims.tokens,
            dims.top_k,
            "memory weighted backward values weights",
        )?,
        "memory_weighted_value_backward_values_f32_i64_buffers weights",
    )?;
    ensure_buffer_len(
        grad_output,
        checked_mul(
            dims.tokens,
            dims.value_dim,
            "memory weighted backward values grad_output",
        )?,
        "memory_weighted_value_backward_values_f32_i64_buffers grad_output",
    )?;
    let output_len = checked_mul(
        dims.slots,
        dims.value_dim,
        "memory weighted backward values output",
    )?;
    let output = fill_constant_f32_buffer(indices.device_ordinal(), output_len, 0.0)?;
    if output_len == 0 {
        return Ok(output);
    }
    let total = checked_mul(
        checked_mul(
            dims.tokens,
            dims.top_k,
            "memory weighted backward values token top-k",
        )?,
        dims.value_dim,
        "memory weighted backward values total",
    )?;
    let status = CudaBuffer::from_u32(indices.device_ordinal(), &[0])?;
    indices.inner.driver.set_current(indices.inner.context)?;
    let module = load_module(&indices.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_memory_weighted_value_backward_values_f32_i64")?;
    launch_memory_weighted_value_kernel(
        &indices.inner.driver,
        function,
        &[
            indices.inner.ptr,
            weights.inner.ptr,
            grad_output.inner.ptr,
            output.inner.ptr,
            status.inner.ptr,
        ],
        dims,
        total,
    )?;
    ensure_status_zero(
        &status,
        "memory_weighted_value_backward_values_f32_i64_buffers",
    )?;
    CUDA_MEMORY_SCATTER_ADD_ROWS_CALLS.fetch_add(1, Ordering::Relaxed);
    Ok(output)
}

pub fn memory_selected_scores_forward_f32_i64_buffers(
    indices: &CudaBuffer,
    query: &CudaBuffer,
    keys: &CudaBuffer,
    dims: MemorySelectedScoreDims,
) -> CudaResult<CudaBuffer> {
    validate_memory_selected_score_buffers(
        indices,
        query,
        keys,
        dims,
        "memory_selected_scores_forward_f32_i64_buffers",
    )?;
    let output_len = checked_mul(
        dims.tokens,
        dims.top_k,
        "memory selected scores forward output",
    )?;
    let output = CudaBuffer::uninit_bytes(
        indices.device_ordinal(),
        output_len,
        std::mem::size_of::<f32>(),
    )?;
    if output_len == 0 {
        return Ok(output);
    }
    let status = CudaBuffer::from_u32(indices.device_ordinal(), &[0])?;
    indices.inner.driver.set_current(indices.inner.context)?;
    let module = load_module(&indices.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_memory_selected_scores_forward_f32_i64")?;
    launch_memory_selected_score_kernel(
        &indices.inner.driver,
        function,
        &[
            indices.inner.ptr,
            query.inner.ptr,
            keys.inner.ptr,
            output.inner.ptr,
            status.inner.ptr,
        ],
        dims,
        output_len,
    )?;
    ensure_status_zero(&status, "memory_selected_scores_forward_f32_i64_buffers")?;
    CUDA_MEMORY_QUERY_KEY_SCORE_CALLS.fetch_add(1, Ordering::Relaxed);
    Ok(output)
}

pub fn memory_selected_scores_backward_query_f32_i64_buffers(
    indices: &CudaBuffer,
    keys: &CudaBuffer,
    grad_scores: &CudaBuffer,
    dims: MemorySelectedScoreDims,
) -> CudaResult<CudaBuffer> {
    ensure_i64_buffer(
        indices,
        "memory_selected_scores_backward_query_f32_i64_buffers indices",
    )?;
    ensure_f32_buffer(
        keys,
        "memory_selected_scores_backward_query_f32_i64_buffers keys",
    )?;
    ensure_f32_buffer(
        grad_scores,
        "memory_selected_scores_backward_query_f32_i64_buffers grad_scores",
    )?;
    ensure_same_device(
        indices,
        keys,
        "memory_selected_scores_backward_query_f32_i64_buffers",
    )?;
    ensure_same_device(
        indices,
        grad_scores,
        "memory_selected_scores_backward_query_f32_i64_buffers",
    )?;
    validate_memory_selected_score_dims(
        dims,
        "memory_selected_scores_backward_query_f32_i64_buffers",
    )?;
    ensure_buffer_len(
        indices,
        checked_mul(
            dims.tokens,
            dims.top_k,
            "memory selected backward query indices",
        )?,
        "memory_selected_scores_backward_query_f32_i64_buffers indices",
    )?;
    ensure_buffer_len(
        keys,
        checked_mul(
            dims.slots,
            dims.key_dim,
            "memory selected backward query keys",
        )?,
        "memory_selected_scores_backward_query_f32_i64_buffers keys",
    )?;
    ensure_buffer_len(
        grad_scores,
        checked_mul(
            dims.tokens,
            dims.top_k,
            "memory selected backward query grad_scores",
        )?,
        "memory_selected_scores_backward_query_f32_i64_buffers grad_scores",
    )?;
    let output_len = checked_mul(
        dims.tokens,
        dims.key_dim,
        "memory selected backward query output",
    )?;
    let output = CudaBuffer::uninit_bytes(
        indices.device_ordinal(),
        output_len,
        std::mem::size_of::<f32>(),
    )?;
    if output_len == 0 {
        return Ok(output);
    }
    let status = CudaBuffer::from_u32(indices.device_ordinal(), &[0])?;
    indices.inner.driver.set_current(indices.inner.context)?;
    let module = load_module(&indices.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_memory_selected_scores_backward_query_f32_i64")?;
    launch_memory_selected_score_kernel(
        &indices.inner.driver,
        function,
        &[
            indices.inner.ptr,
            keys.inner.ptr,
            grad_scores.inner.ptr,
            output.inner.ptr,
            status.inner.ptr,
        ],
        dims,
        output_len,
    )?;
    ensure_status_zero(
        &status,
        "memory_selected_scores_backward_query_f32_i64_buffers",
    )?;
    Ok(output)
}

pub fn memory_selected_scores_backward_keys_f32_i64_buffers(
    indices: &CudaBuffer,
    query: &CudaBuffer,
    grad_scores: &CudaBuffer,
    dims: MemorySelectedScoreDims,
) -> CudaResult<CudaBuffer> {
    ensure_i64_buffer(
        indices,
        "memory_selected_scores_backward_keys_f32_i64_buffers indices",
    )?;
    ensure_f32_buffer(
        query,
        "memory_selected_scores_backward_keys_f32_i64_buffers query",
    )?;
    ensure_f32_buffer(
        grad_scores,
        "memory_selected_scores_backward_keys_f32_i64_buffers grad_scores",
    )?;
    ensure_same_device(
        indices,
        query,
        "memory_selected_scores_backward_keys_f32_i64_buffers",
    )?;
    ensure_same_device(
        indices,
        grad_scores,
        "memory_selected_scores_backward_keys_f32_i64_buffers",
    )?;
    validate_memory_selected_score_dims(
        dims,
        "memory_selected_scores_backward_keys_f32_i64_buffers",
    )?;
    ensure_buffer_len(
        indices,
        checked_mul(
            dims.tokens,
            dims.top_k,
            "memory selected backward keys indices",
        )?,
        "memory_selected_scores_backward_keys_f32_i64_buffers indices",
    )?;
    ensure_buffer_len(
        query,
        checked_mul(
            dims.tokens,
            dims.key_dim,
            "memory selected backward keys query",
        )?,
        "memory_selected_scores_backward_keys_f32_i64_buffers query",
    )?;
    ensure_buffer_len(
        grad_scores,
        checked_mul(
            dims.tokens,
            dims.top_k,
            "memory selected backward keys grad_scores",
        )?,
        "memory_selected_scores_backward_keys_f32_i64_buffers grad_scores",
    )?;
    let output_len = checked_mul(
        dims.slots,
        dims.key_dim,
        "memory selected backward keys output",
    )?;
    let output = fill_constant_f32_buffer(indices.device_ordinal(), output_len, 0.0)?;
    if output_len == 0 {
        return Ok(output);
    }
    let total = checked_mul(
        checked_mul(
            dims.tokens,
            dims.top_k,
            "memory selected backward keys selected rows",
        )?,
        dims.key_dim,
        "memory selected backward keys total",
    )?;
    let status = CudaBuffer::from_u32(indices.device_ordinal(), &[0])?;
    indices.inner.driver.set_current(indices.inner.context)?;
    let module = load_module(&indices.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_memory_selected_scores_backward_keys_f32_i64")?;
    launch_memory_selected_score_kernel(
        &indices.inner.driver,
        function,
        &[
            indices.inner.ptr,
            query.inner.ptr,
            grad_scores.inner.ptr,
            output.inner.ptr,
            status.inner.ptr,
        ],
        dims,
        total,
    )?;
    ensure_status_zero(
        &status,
        "memory_selected_scores_backward_keys_f32_i64_buffers",
    )?;
    CUDA_MEMORY_SELECTED_KEY_BACKWARD_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_MEMORY_SCATTER_ADD_ROWS_CALLS.fetch_add(1, Ordering::Relaxed);
    Ok(output)
}

pub fn validate_memory_lookup_dims(dims: MemoryLookupDims, op_name: &str) -> CudaResult<()> {
    if dims.tokens == 0
        || dims.slots == 0
        || dims.key_dim == 0
        || dims.value_dim == 0
        || dims.top_k == 0
    {
        return Err(CudaError::new(format!(
            "{op_name} requires non-zero tokens/slots/key_dim/value_dim/top_k, got {:?}",
            dims
        )));
    }
    if dims.top_k > dims.slots {
        return Err(CudaError::new(format!(
            "{op_name} requires top_k <= slots, got top_k={} slots={}",
            dims.top_k, dims.slots
        )));
    }
    checked_mul(dims.tokens, dims.key_dim, &format!("{op_name} query"))?;
    checked_mul(dims.slots, dims.key_dim, &format!("{op_name} key table"))?;
    checked_mul(
        dims.slots,
        dims.value_dim,
        &format!("{op_name} value table"),
    )?;
    checked_mul(dims.tokens, dims.slots, &format!("{op_name} score matrix"))?;
    checked_mul(dims.tokens, dims.top_k, &format!("{op_name} top-k"))?;
    checked_mul(
        checked_mul(
            dims.tokens,
            dims.top_k,
            &format!("{op_name} selected values"),
        )?,
        dims.value_dim,
        &format!("{op_name} selected value elements"),
    )?;
    checked_mul(
        dims.tokens,
        dims.value_dim,
        &format!("{op_name} aggregated output"),
    )?;
    Ok(())
}

pub fn validate_memory_topk_dims(dims: MemoryTopkDims, op_name: &str) -> CudaResult<()> {
    if dims.rows == 0 || dims.cols == 0 || dims.top_k == 0 {
        return Err(CudaError::new(format!(
            "{op_name} requires non-zero rows/cols/top_k, got {:?}",
            dims
        )));
    }
    if dims.top_k > dims.cols {
        return Err(CudaError::new(format!(
            "{op_name} requires top_k <= cols, got top_k={} cols={}",
            dims.top_k, dims.cols
        )));
    }
    checked_mul(dims.rows, dims.cols, &format!("{op_name} scores"))?;
    checked_mul(dims.rows, dims.top_k, &format!("{op_name} output"))?;
    Ok(())
}

pub fn validate_memory_weighted_value_dims(
    dims: MemoryWeightedValueDims,
    op_name: &str,
) -> CudaResult<()> {
    if dims.tokens == 0 || dims.top_k == 0 || dims.slots == 0 || dims.value_dim == 0 {
        return Err(CudaError::new(format!(
            "{op_name} requires non-zero tokens/top_k/slots/value_dim, got {:?}",
            dims
        )));
    }
    if dims.top_k > dims.slots {
        return Err(CudaError::new(format!(
            "{op_name} requires top_k <= slots, got top_k={} slots={}",
            dims.top_k, dims.slots
        )));
    }
    checked_mul(dims.tokens, dims.top_k, &format!("{op_name} selected rows"))?;
    checked_mul(
        dims.slots,
        dims.value_dim,
        &format!("{op_name} value table"),
    )?;
    checked_mul(dims.tokens, dims.value_dim, &format!("{op_name} output"))?;
    checked_mul(
        checked_mul(
            dims.tokens,
            dims.top_k,
            &format!("{op_name} selected values"),
        )?,
        dims.value_dim,
        &format!("{op_name} selected value elements"),
    )?;
    Ok(())
}

pub fn validate_memory_selected_score_dims(
    dims: MemorySelectedScoreDims,
    op_name: &str,
) -> CudaResult<()> {
    if dims.tokens == 0 || dims.top_k == 0 || dims.slots == 0 || dims.key_dim == 0 {
        return Err(CudaError::new(format!(
            "{op_name} requires non-zero tokens/top_k/slots/key_dim, got {:?}",
            dims
        )));
    }
    if dims.top_k > dims.slots {
        return Err(CudaError::new(format!(
            "{op_name} requires top_k <= slots, got top_k={} slots={}",
            dims.top_k, dims.slots
        )));
    }
    checked_mul(dims.tokens, dims.top_k, &format!("{op_name} selected rows"))?;
    checked_mul(dims.tokens, dims.key_dim, &format!("{op_name} query"))?;
    checked_mul(dims.slots, dims.key_dim, &format!("{op_name} key table"))?;
    checked_mul(dims.tokens, dims.top_k, &format!("{op_name} output"))?;
    checked_mul(
        checked_mul(
            dims.tokens,
            dims.top_k,
            &format!("{op_name} selected key rows"),
        )?,
        dims.key_dim,
        &format!("{op_name} selected key elements"),
    )?;
    Ok(())
}

pub fn validate_memory_product_key_dims(
    dims: MemoryProductKeyDims,
    op_name: &str,
) -> CudaResult<()> {
    if dims.tokens == 0 || dims.slots == 0 || dims.key_dim == 0 || dims.top_k == 0 || dims.beam == 0
    {
        return Err(CudaError::new(format!(
            "{op_name} requires non-zero tokens/slots/key_dim/top_k/beam, got {:?}",
            dims
        )));
    }
    if !dims.key_dim.is_multiple_of(2) {
        return Err(CudaError::new(format!(
            "{op_name} requires even key_dim for product-key lookup, got {}",
            dims.key_dim
        )));
    }
    let side = product_key_side_cuda(dims.slots, op_name)?;
    if dims.top_k > dims.slots {
        return Err(CudaError::new(format!(
            "{op_name} requires top_k <= slots, got top_k={} slots={}",
            dims.top_k, dims.slots
        )));
    }
    if dims.beam > side {
        return Err(CudaError::new(format!(
            "{op_name} requires beam <= sqrt(slots), got beam={} side={side}",
            dims.beam
        )));
    }
    let candidate_count = checked_mul(dims.beam, dims.beam, &format!("{op_name} candidates"))?;
    if dims.top_k > candidate_count {
        return Err(CudaError::new(format!(
            "{op_name} requires top_k <= beam^2, got top_k={} beam={}",
            dims.top_k, dims.beam
        )));
    }
    checked_mul(dims.tokens, dims.key_dim, &format!("{op_name} query"))?;
    checked_mul(dims.slots, dims.key_dim, &format!("{op_name} keys"))?;
    checked_mul(dims.tokens, side, &format!("{op_name} side scores"))?;
    checked_mul(dims.tokens, dims.beam, &format!("{op_name} side top-k"))?;
    checked_mul(dims.tokens, dims.top_k, &format!("{op_name} output"))?;
    Ok(())
}

pub fn validate_memory_product_key_selected_score_dims(
    dims: MemoryProductKeySelectedScoreDims,
    op_name: &str,
) -> CudaResult<()> {
    if dims.tokens == 0 || dims.top_k == 0 || dims.side == 0 || dims.key_dim == 0 {
        return Err(CudaError::new(format!(
            "{op_name} requires non-zero tokens/top_k/side/key_dim, got {:?}",
            dims
        )));
    }
    if !dims.key_dim.is_multiple_of(2) {
        return Err(CudaError::new(format!(
            "{op_name} requires even key_dim for product-key selected scores, got {}",
            dims.key_dim
        )));
    }
    checked_mul(dims.side, dims.side, &format!("{op_name} slots"))?;
    checked_mul(dims.tokens, dims.top_k, &format!("{op_name} indices"))?;
    checked_mul(dims.tokens, dims.key_dim, &format!("{op_name} query"))?;
    checked_mul(
        dims.side,
        dims.key_dim / 2,
        &format!("{op_name} half-key table"),
    )?;
    checked_mul(dims.tokens, dims.top_k, &format!("{op_name} output"))?;
    Ok(())
}

fn validate_memory_weighted_value_buffers(
    indices: &CudaBuffer,
    weights: &CudaBuffer,
    values: &CudaBuffer,
    dims: MemoryWeightedValueDims,
    op_name: &str,
) -> CudaResult<()> {
    ensure_i64_buffer(indices, &format!("{op_name} indices"))?;
    ensure_f32_buffer(weights, &format!("{op_name} weights"))?;
    ensure_f32_buffer(values, &format!("{op_name} values"))?;
    ensure_same_device(indices, weights, op_name)?;
    ensure_same_device(indices, values, op_name)?;
    validate_memory_weighted_value_dims(dims, op_name)?;
    ensure_buffer_len(
        indices,
        checked_mul(dims.tokens, dims.top_k, &format!("{op_name} indices"))?,
        &format!("{op_name} indices"),
    )?;
    ensure_buffer_len(
        weights,
        checked_mul(dims.tokens, dims.top_k, &format!("{op_name} weights"))?,
        &format!("{op_name} weights"),
    )?;
    ensure_buffer_len(
        values,
        checked_mul(dims.slots, dims.value_dim, &format!("{op_name} values"))?,
        &format!("{op_name} values"),
    )?;
    Ok(())
}

fn validate_memory_selected_score_buffers(
    indices: &CudaBuffer,
    query: &CudaBuffer,
    keys: &CudaBuffer,
    dims: MemorySelectedScoreDims,
    op_name: &str,
) -> CudaResult<()> {
    ensure_i64_buffer(indices, &format!("{op_name} indices"))?;
    ensure_f32_buffer(query, &format!("{op_name} query"))?;
    ensure_f32_buffer(keys, &format!("{op_name} keys"))?;
    ensure_same_device(indices, query, op_name)?;
    ensure_same_device(indices, keys, op_name)?;
    validate_memory_selected_score_dims(dims, op_name)?;
    ensure_buffer_len(
        indices,
        checked_mul(dims.tokens, dims.top_k, &format!("{op_name} indices"))?,
        &format!("{op_name} indices"),
    )?;
    ensure_buffer_len(
        query,
        checked_mul(dims.tokens, dims.key_dim, &format!("{op_name} query"))?,
        &format!("{op_name} query"),
    )?;
    ensure_buffer_len(
        keys,
        checked_mul(dims.slots, dims.key_dim, &format!("{op_name} keys"))?,
        &format!("{op_name} keys"),
    )?;
    Ok(())
}

fn validate_memory_product_key_buffers(
    query: &CudaBuffer,
    keys: &CudaBuffer,
    dims: MemoryProductKeyDims,
    op_name: &str,
) -> CudaResult<()> {
    ensure_f32_buffer(query, &format!("{op_name} query"))?;
    ensure_f32_buffer(keys, &format!("{op_name} keys"))?;
    ensure_same_device(query, keys, op_name)?;
    validate_memory_product_key_dims(dims, op_name)?;
    ensure_buffer_len(
        query,
        checked_mul(dims.tokens, dims.key_dim, &format!("{op_name} query"))?,
        &format!("{op_name} query"),
    )?;
    ensure_buffer_len(
        keys,
        checked_mul(dims.slots, dims.key_dim, &format!("{op_name} keys"))?,
        &format!("{op_name} keys"),
    )?;
    Ok(())
}

fn validate_memory_product_key_side_score_buffers(
    left_scores: &CudaBuffer,
    right_scores: &CudaBuffer,
    dims: MemoryProductKeyDims,
    op_name: &str,
) -> CudaResult<()> {
    ensure_f32_buffer(left_scores, &format!("{op_name} left_scores"))?;
    ensure_f32_buffer(right_scores, &format!("{op_name} right_scores"))?;
    ensure_same_device(left_scores, right_scores, op_name)?;
    validate_memory_product_key_dims(dims, op_name)?;
    let side = product_key_side_cuda(dims.slots, op_name)?;
    let side_len = checked_mul(dims.tokens, side, &format!("{op_name} side scores"))?;
    ensure_buffer_len(left_scores, side_len, &format!("{op_name} left_scores"))?;
    ensure_buffer_len(right_scores, side_len, &format!("{op_name} right_scores"))?;
    Ok(())
}

fn validate_memory_product_key_selected_score_buffers(
    indices: &CudaBuffer,
    query: &CudaBuffer,
    left_keys: &CudaBuffer,
    right_keys: &CudaBuffer,
    dims: MemoryProductKeySelectedScoreDims,
    op_name: &str,
) -> CudaResult<()> {
    ensure_i64_buffer(indices, &format!("{op_name} indices"))?;
    ensure_f32_buffer(query, &format!("{op_name} query"))?;
    ensure_f32_buffer(left_keys, &format!("{op_name} left_keys"))?;
    ensure_f32_buffer(right_keys, &format!("{op_name} right_keys"))?;
    ensure_same_device(indices, query, op_name)?;
    ensure_same_device(indices, left_keys, op_name)?;
    ensure_same_device(indices, right_keys, op_name)?;
    validate_memory_product_key_selected_score_dims(dims, op_name)?;
    ensure_product_key_selected_score_lengths(
        indices,
        Some(query),
        left_keys,
        right_keys,
        None,
        dims,
        op_name,
    )
}

fn ensure_product_key_selected_score_lengths(
    indices: &CudaBuffer,
    query: Option<&CudaBuffer>,
    left_keys: &CudaBuffer,
    right_keys: &CudaBuffer,
    grad_scores: Option<&CudaBuffer>,
    dims: MemoryProductKeySelectedScoreDims,
    op_name: &str,
) -> CudaResult<()> {
    ensure_buffer_len(
        indices,
        checked_mul(dims.tokens, dims.top_k, &format!("{op_name} indices"))?,
        &format!("{op_name} indices"),
    )?;
    if let Some(query) = query {
        ensure_buffer_len(
            query,
            checked_mul(dims.tokens, dims.key_dim, &format!("{op_name} query"))?,
            &format!("{op_name} query"),
        )?;
    }
    let half_dim = dims.key_dim / 2;
    let half_table_len = checked_mul(dims.side, half_dim, &format!("{op_name} half-key table"))?;
    ensure_buffer_len(left_keys, half_table_len, &format!("{op_name} left_keys"))?;
    ensure_buffer_len(right_keys, half_table_len, &format!("{op_name} right_keys"))?;
    if let Some(grad_scores) = grad_scores {
        ensure_buffer_len(
            grad_scores,
            checked_mul(dims.tokens, dims.top_k, &format!("{op_name} grad_scores"))?,
            &format!("{op_name} grad_scores"),
        )?;
    }
    Ok(())
}

pub fn validate_sparse_adamw_rows_dims(dims: SparseAdamWRowsDims, op_name: &str) -> CudaResult<()> {
    if dims.rows == 0 || dims.row_dim == 0 {
        return Err(CudaError::new(format!(
            "{op_name} requires non-zero rows/row_dim, got {:?}",
            dims
        )));
    }
    checked_mul(dims.rows, dims.row_dim, &format!("{op_name} table"))?;
    checked_mul(
        dims.selected_rows,
        dims.row_dim,
        &format!("{op_name} selected row elements"),
    )?;
    Ok(())
}

fn validate_sparse_adamw_rows_buffers(
    buffers: SparseAdamWRowsBuffers<'_>,
    dims: SparseAdamWRowsDims,
    op_name: &str,
) -> CudaResult<()> {
    ensure_f32_buffer(buffers.param, &format!("{op_name} param"))?;
    ensure_f32_buffer(buffers.grad, &format!("{op_name} grad"))?;
    ensure_f32_buffer(buffers.m, &format!("{op_name} m"))?;
    ensure_f32_buffer(buffers.v, &format!("{op_name} v"))?;
    ensure_i64_buffer(buffers.selected_rows, &format!("{op_name} selected_rows"))?;
    ensure_same_device(buffers.param, buffers.grad, op_name)?;
    ensure_same_device(buffers.param, buffers.m, op_name)?;
    ensure_same_device(buffers.param, buffers.v, op_name)?;
    ensure_same_device(buffers.param, buffers.selected_rows, op_name)?;
    if let Some(row_mask) = buffers.row_mask {
        ensure_u8_buffer(row_mask, &format!("{op_name} row_mask"))?;
        ensure_same_device(buffers.param, row_mask, op_name)?;
        ensure_buffer_len(row_mask, dims.rows, &format!("{op_name} row_mask"))?;
    }
    validate_sparse_adamw_rows_dims(dims, op_name)?;
    let table_len = checked_mul(dims.rows, dims.row_dim, &format!("{op_name} table"))?;
    ensure_buffer_len(buffers.param, table_len, &format!("{op_name} param"))?;
    ensure_buffer_len(buffers.grad, table_len, &format!("{op_name} grad"))?;
    ensure_buffer_len(buffers.m, table_len, &format!("{op_name} m"))?;
    ensure_buffer_len(buffers.v, table_len, &format!("{op_name} v"))?;
    ensure_buffer_len(
        buffers.selected_rows,
        dims.selected_rows,
        &format!("{op_name} selected_rows"),
    )?;
    Ok(())
}

fn validate_sparse_adamw_compact_rows_buffers(
    buffers: SparseAdamWCompactRowsBuffers<'_>,
    dims: SparseAdamWRowsDims,
    op_name: &str,
) -> CudaResult<()> {
    ensure_f32_buffer(buffers.param, &format!("{op_name} param"))?;
    ensure_f32_buffer(buffers.compact_grad, &format!("{op_name} compact_grad"))?;
    ensure_f32_buffer(buffers.m, &format!("{op_name} m"))?;
    ensure_f32_buffer(buffers.v, &format!("{op_name} v"))?;
    ensure_i64_buffer(buffers.selected_rows, &format!("{op_name} selected_rows"))?;
    ensure_same_device(buffers.param, buffers.compact_grad, op_name)?;
    ensure_same_device(buffers.param, buffers.m, op_name)?;
    ensure_same_device(buffers.param, buffers.v, op_name)?;
    ensure_same_device(buffers.param, buffers.selected_rows, op_name)?;
    if let Some(row_mask) = buffers.row_mask {
        ensure_u8_buffer(row_mask, &format!("{op_name} row_mask"))?;
        ensure_same_device(buffers.param, row_mask, op_name)?;
        ensure_buffer_len(row_mask, dims.rows, &format!("{op_name} row_mask"))?;
    }
    validate_sparse_adamw_rows_dims(dims, op_name)?;
    let table_len = checked_mul(dims.rows, dims.row_dim, &format!("{op_name} table"))?;
    let compact_len = checked_mul(
        dims.selected_rows,
        dims.row_dim,
        &format!("{op_name} compact_grad"),
    )?;
    ensure_buffer_len(buffers.param, table_len, &format!("{op_name} param"))?;
    ensure_buffer_len(
        buffers.compact_grad,
        compact_len,
        &format!("{op_name} compact_grad"),
    )?;
    ensure_buffer_len(buffers.m, table_len, &format!("{op_name} m"))?;
    ensure_buffer_len(buffers.v, table_len, &format!("{op_name} v"))?;
    ensure_buffer_len(
        buffers.selected_rows,
        dims.selected_rows,
        &format!("{op_name} selected_rows"),
    )?;
    Ok(())
}

fn validate_selected_rows_gather_buffers(
    table: &CudaBuffer,
    selected_rows: &CudaBuffer,
    dims: SelectedRowsDims,
    op_name: &str,
) -> CudaResult<()> {
    ensure_f32_buffer(table, &format!("{op_name} table"))?;
    ensure_i64_buffer(selected_rows, &format!("{op_name} selected_rows"))?;
    ensure_same_device(table, selected_rows, op_name)?;
    if dims.rows == 0 || dims.row_dim == 0 {
        return Err(CudaError::new(format!(
            "{op_name} requires non-zero rows/row_dim, got {:?}",
            dims
        )));
    }
    let table_len = checked_mul(dims.rows, dims.row_dim, &format!("{op_name} table"))?;
    ensure_buffer_len(table, table_len, &format!("{op_name} table"))?;
    ensure_buffer_len(
        selected_rows,
        dims.selected_rows,
        &format!("{op_name} selected_rows"),
    )?;
    checked_mul(
        dims.selected_rows,
        dims.row_dim,
        &format!("{op_name} output"),
    )?;
    Ok(())
}

fn validate_rank2_buffer(
    buffer: &CudaBuffer,
    rows: usize,
    cols: usize,
    role: &str,
) -> CudaResult<()> {
    ensure_f32_buffer(buffer, role)?;
    ensure_buffer_len(buffer, checked_mul(rows, cols, role)?, role)
}

fn allocator_cache_disabled() -> bool {
    matches!(
        std::env::var("HEIRLOOM_CUDA_ALLOCATOR_DIRECT")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

fn update_allocation_high_water() {
    let active = CUDA_ALLOC_ACTIVE_BYTES.load(Ordering::Relaxed);
    let mut observed = CUDA_ALLOC_HIGH_WATER_BYTES.load(Ordering::Relaxed);
    while active > observed {
        match CUDA_ALLOC_HIGH_WATER_BYTES.compare_exchange_weak(
            observed,
            active,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(next) => observed = next,
        }
    }
}

pub fn bf16_tensor_core_matmul_exact_tile_shape_supported(m: usize, k: usize, n: usize) -> bool {
    m > 0 && k > 0 && n > 0 && m.is_multiple_of(16) && k.is_multiple_of(16) && n.is_multiple_of(8)
}

pub fn bf16_tensor_core_matmul_shape_supported(m: usize, k: usize, n: usize) -> bool {
    m > 0 && k > 0 && n > 0
}

pub fn bf16_tensor_core_matmul_cta_plan(
    m: usize,
    k: usize,
    n: usize,
) -> CudaResult<TensorCoreMatmulCtaPlan> {
    if !bf16_tensor_core_matmul_exact_tile_shape_supported(m, k, n) {
        return Err(CudaError::new(format!(
            "BF16 Tensor Core CTA plan requires padded tile shapes with M%16=0 K%16=0 N%8=0, got M={m} K={k} N={n}"
        )));
    }

    let cta_m = 32;
    let cta_n = 16;
    let warps_per_cta = 4;
    let warp_m = 16;
    let warp_n = 8;
    let grid_x = n.div_ceil(cta_n);
    let grid_y = m.div_ceil(cta_m);
    let cta_tiles = checked_mul(grid_x, grid_y, "Tensor Core CTA tile count")?;
    let k_tiles = k / 16;
    let shared_stage_tiles =
        checked_mul(cta_tiles, k_tiles, "Tensor Core shared stage tile count")?;
    let shared_stage_bytes = checked_mul(
        shared_stage_tiles,
        1536,
        "Tensor Core shared stage byte count",
    )?;
    let active_mma_warp_tiles = checked_mul(
        m / warp_m,
        n / warp_n,
        "Tensor Core active MMA warp tile count",
    )?;
    let launched_warps = checked_mul(
        cta_tiles,
        warps_per_cta,
        "Tensor Core launched CTA warp count",
    )?;
    Ok(TensorCoreMatmulCtaPlan {
        cta_m,
        cta_n,
        warps_per_cta,
        warp_m,
        warp_n,
        grid_x,
        grid_y,
        cta_tiles,
        active_mma_warp_tiles,
        launched_warps,
        k_tiles,
        shared_stage_tiles,
        shared_stage_bytes,
    })
}

pub fn bf16_tensor_core_matmul_wide_swizzled_cta_plan(
    m: usize,
    k: usize,
    n: usize,
) -> CudaResult<TensorCoreMatmulCtaPlan> {
    if !bf16_tensor_core_matmul_exact_tile_shape_supported(m, k, n) {
        return Err(CudaError::new(format!(
            "BF16 Tensor Core wide swizzled CTA plan requires padded tile shapes with M%16=0 K%16=0 N%8=0, got M={m} K={k} N={n}"
        )));
    }

    let cta_m = 32;
    let cta_n = 32;
    let warps_per_cta = 8;
    let warp_m = 16;
    let warp_n = 8;
    let grid_x = n.div_ceil(cta_n);
    let grid_y = m.div_ceil(cta_m);
    let cta_tiles = checked_mul(grid_x, grid_y, "wide swizzled Tensor Core CTA tile count")?;
    let k_tiles = k / 16;
    let shared_stage_tiles = checked_mul(
        cta_tiles,
        k_tiles,
        "wide swizzled Tensor Core shared stage tile count",
    )?;
    let shared_stage_bytes = checked_mul(
        shared_stage_tiles,
        2048,
        "wide swizzled Tensor Core shared stage byte count",
    )?;
    let active_mma_warp_tiles = checked_mul(
        m / warp_m,
        n / warp_n,
        "wide swizzled Tensor Core active MMA warp tile count",
    )?;
    let launched_warps = checked_mul(
        cta_tiles,
        warps_per_cta,
        "wide swizzled Tensor Core launched CTA warp count",
    )?;
    Ok(TensorCoreMatmulCtaPlan {
        cta_m,
        cta_n,
        warps_per_cta,
        warp_m,
        warp_n,
        grid_x,
        grid_y,
        cta_tiles,
        active_mma_warp_tiles,
        launched_warps,
        k_tiles,
        shared_stage_tiles,
        shared_stage_bytes,
    })
}

pub fn causal_attention_bf16_tensor_core_exact_tile_shape_supported(
    dims: CausalAttentionDims,
) -> bool {
    validate_causal_attention_dims(dims, "causal_attention_bf16_tensor_core_shape_supported")
        .is_ok_and(|head_dim| dims.time.is_multiple_of(16) && head_dim.is_multiple_of(16))
}

pub fn causal_attention_bf16_flash_tensor_core_backward_shape_supported(
    dims: CausalAttentionDims,
) -> bool {
    causal_attention_bf16_tensor_core_exact_tile_shape_supported(dims)
}

pub fn causal_attention_bf16_tensor_core_shape_supported(dims: CausalAttentionDims) -> bool {
    validate_causal_attention_dims(dims, "causal_attention_bf16_tensor_core_shape_supported")
        .is_ok()
}

pub fn causal_attention_bf16_flash_shape_supported(dims: CausalAttentionDims) -> bool {
    validate_causal_attention_dims(dims, "causal_attention_bf16_flash_shape_supported").is_ok()
}

pub fn record_flash_bf16_attention_fallback(scalar: bool) {
    CUDA_FLASH_BF16_ATTENTION_FALLBACK_CALLS.fetch_add(1, Ordering::Relaxed);
    if scalar {
        CUDA_FLASH_BF16_ATTENTION_SCALAR_FALLBACK_CALLS.fetch_add(1, Ordering::Relaxed);
        CUDA_TENSOR_CORE_ATTENTION_SCALAR_FALLBACKS.fetch_add(1, Ordering::Relaxed);
    } else {
        CUDA_FLASH_BF16_TENSOR_CORE_FALLBACK_CALLS.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn record_flash_bf16_attention_request() {
    CUDA_FLASH_BF16_ATTENTION_REQUESTED_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_FLASH_BF16_SCALAR_STREAMING_REQUESTED_CALLS.fetch_add(1, Ordering::Relaxed);
}

pub fn record_flash_bf16_tensor_core_attention_request() {
    CUDA_FLASH_BF16_ATTENTION_REQUESTED_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_REQUESTED_CALLS.fetch_add(1, Ordering::Relaxed);
}

pub fn record_flash_bf16_tensor_core_attention_backward_request() {
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_REQUESTED_CALLS.fetch_add(1, Ordering::Relaxed);
}

pub fn record_flash_bf16_tensor_core_attention_backward_fallback() {
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_FALLBACK_CALLS.fetch_add(1, Ordering::Relaxed);
}

pub fn record_flash_bf16_attention_hard_require_failure() {
    CUDA_FLASH_BF16_ATTENTION_HARD_REQUIRE_FAILURES.fetch_add(1, Ordering::Relaxed);
}

pub fn flash_bf16_attention_required_error(reason: &str) -> CudaError {
    CudaError::new(format!(
        "HEIRLOOM_REQUIRE_FLASH_BF16_ATTENTION=1 but flash BF16 causal attention cannot run: {reason}"
    ))
}

fn record_flash_scalar_streaming_tiles(
    dims: CausalAttentionDims,
    head_dim: usize,
) -> CudaResult<()> {
    let batch_heads = checked_mul(
        dims.batch,
        dims.n_heads,
        "flash BF16 attention batch*heads tile count",
    )?;
    let query_blocks = dims.time.div_ceil(16);
    let key_blocks = dims.time.div_ceil(16);
    let output_dim_tiles = head_dim.div_ceil(8);
    let qk_tiles = checked_mul(
        checked_mul(
            batch_heads,
            query_blocks,
            "flash BF16 attention QK query tiles",
        )?,
        key_blocks,
        "flash BF16 attention QK key tiles",
    )?;
    let av_tiles = checked_mul(
        qk_tiles,
        output_dim_tiles,
        "flash BF16 attention AV output-dim tiles",
    )?;
    CUDA_FLASH_BF16_ATTENTION_QK_TILE_CALLS.fetch_add(qk_tiles, Ordering::Relaxed);
    CUDA_FLASH_BF16_ATTENTION_AV_TILE_CALLS.fetch_add(av_tiles, Ordering::Relaxed);
    CUDA_FLASH_BF16_SCALAR_STREAMING_QK_TILE_CALLS.fetch_add(qk_tiles, Ordering::Relaxed);
    CUDA_FLASH_BF16_SCALAR_STREAMING_AV_TILE_CALLS.fetch_add(av_tiles, Ordering::Relaxed);

    let mut causal_tiles = 0usize;
    for query_block in 0..query_blocks {
        let query_start = query_block * 16;
        for key_block in 0..key_blocks {
            let key_end = ((key_block + 1) * 16).min(dims.time);
            if key_end > query_start + 1 {
                causal_tiles += 1;
            }
        }
    }
    CUDA_FLASH_BF16_ATTENTION_CAUSAL_MASKED_TILE_COUNT.fetch_add(
        checked_mul(
            causal_tiles,
            batch_heads,
            "flash BF16 attention causal masked tiles",
        )?,
        Ordering::Relaxed,
    );

    if !dims.time.is_multiple_of(16) || !head_dim.is_multiple_of(8) {
        CUDA_FLASH_BF16_ATTENTION_RAGGED_TILE_COUNT
            .fetch_add(qk_tiles + av_tiles, Ordering::Relaxed);
    }
    Ok(())
}

pub fn bf16_mma_probe(device_ordinal: i32) -> CudaResult<TensorCoreProbeReport> {
    if !device_supports_bf16_tensor_cores(device_ordinal)? {
        return Err(CudaError::new(format!(
            "BF16 Tensor Core probe requires compute capability >= 8.0 on cuda:{device_ordinal}"
        )));
    }

    let output = CudaBuffer::uninit_bytes(device_ordinal, 32, std::mem::size_of::<f32>())?;
    output.inner.driver.set_current(output.inner.context)?;
    let module = load_module(&output.inner.driver, TENSOR_CORE_PTX)?;
    let function = module.function("heirloom_bf16_mma_probe")?;
    launch_tensor_core_probe_kernel(&output.inner.driver, function, output.inner.ptr)?;
    BF16_MMA_PROBE_CALLS.fetch_add(1, Ordering::Relaxed);

    let samples = output.to_f32()?;
    let expected_dot = 16.0;
    let max_abs_error = samples
        .iter()
        .map(|value| (value - expected_dot).abs())
        .fold(0.0_f32, f32::max);
    Ok(TensorCoreProbeReport {
        device_ordinal,
        expected_dot,
        max_abs_error,
        samples,
    })
}

pub fn matmul_bf16_tensor_core_rhs_t_f32_buffers(
    left: &CudaBuffer,
    right: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> CudaResult<CudaBuffer> {
    let output = matmul_bf16_tensor_core_rhs_t_f32_buffers_impl(left, right, m, k, n)?;
    BF16_TENSOR_CORE_MATMUL_CALLS.fetch_add(1, Ordering::Relaxed);
    BF16_TENSOR_CORE_MATMUL_FORWARD_CALLS.fetch_add(1, Ordering::Relaxed);
    Ok(output)
}

pub fn matmul_bf16_tensor_core_rhs_t_bias_f32_buffers(
    left: &CudaBuffer,
    right: &CudaBuffer,
    bias: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> CudaResult<CudaBuffer> {
    ensure_bf16_buffer(left, "matmul_bf16_tensor_core_rhs_t_bias_f32_buffers left")?;
    ensure_bf16_buffer(
        right,
        "matmul_bf16_tensor_core_rhs_t_bias_f32_buffers right",
    )?;
    ensure_f32_buffer(bias, "matmul_bf16_tensor_core_rhs_t_bias_f32_buffers bias")?;
    ensure_same_device(
        left,
        right,
        "matmul_bf16_tensor_core_rhs_t_bias_f32_buffers",
    )?;
    ensure_same_device(left, bias, "matmul_bf16_tensor_core_rhs_t_bias_f32_buffers")?;
    if !device_supports_bf16_tensor_cores(left.device_ordinal())? {
        return Err(CudaError::new(format!(
            "matmul_bf16_tensor_core_rhs_t_bias_f32_buffers requires compute capability >= 8.0 on cuda:{}",
            left.device_ordinal()
        )));
    }
    if !tensor_core_cp_async_gemm_enabled() {
        return Err(CudaError::new(
            "matmul_bf16_tensor_core_rhs_t_bias_f32_buffers requires HEIRLOOM_CUDA_TENSOR_CORE_CP_ASYNC_GEMM=1".to_string(),
        ));
    }
    if !bf16_tensor_core_matmul_exact_tile_shape_supported(m, k, n) {
        return Err(CudaError::new(format!(
            "matmul_bf16_tensor_core_rhs_t_bias_f32_buffers requires exact Tensor Core tile shapes M%16=0 K%16=0 N%8=0, got M={m} K={k} N={n}"
        )));
    }
    ensure_buffer_len(
        left,
        checked_mul(m, k, "matmul_bf16_tensor_core_rhs_t_bias_f32_buffers left")?,
        "matmul_bf16_tensor_core_rhs_t_bias_f32_buffers left",
    )?;
    ensure_buffer_len(
        right,
        checked_mul(n, k, "matmul_bf16_tensor_core_rhs_t_bias_f32_buffers right")?,
        "matmul_bf16_tensor_core_rhs_t_bias_f32_buffers right",
    )?;
    ensure_buffer_len(
        bias,
        n,
        "matmul_bf16_tensor_core_rhs_t_bias_f32_buffers bias",
    )?;

    let output_len = checked_mul(
        m,
        n,
        "matmul_bf16_tensor_core_rhs_t_bias_f32_buffers output",
    )?;
    let output = CudaBuffer::uninit_bytes(
        left.device_ordinal(),
        output_len,
        std::mem::size_of::<f32>(),
    )?;

    left.inner.driver.set_current(left.inner.context)?;
    let module = load_module(&left.inner.driver, TENSOR_CORE_PTX)?;
    let function = module.function("heirloom_matmul_bf16_mma_rhs_t_f32_cta_cp_async_db")?;
    CUDA_TENSOR_CORE_CP_ASYNC_GEMM_REQUESTED_CALLS.fetch_add(1, Ordering::Relaxed);
    let mut timer = start_tensor_core_gemm_timer(left.device_ordinal())?;
    launch_bf16_mma_matmul_cp_async_cta_kernel_with_bias(
        &left.inner.driver,
        function,
        &[left.inner.ptr, right.inner.ptr, output.inner.ptr],
        bias.inner.ptr,
        m,
        k,
        n,
    )?;
    stop_tensor_core_gemm_timer(&mut timer, &CUDA_TENSOR_CORE_CP_ASYNC_GEMM_ELAPSED_US)?;

    BF16_TENSOR_CORE_MATMUL_CALLS.fetch_add(1, Ordering::Relaxed);
    BF16_TENSOR_CORE_MATMUL_FORWARD_CALLS.fetch_add(1, Ordering::Relaxed);
    Ok(output)
}

pub fn matmul_bf16_tensor_core_rhs_t_f32_buffers_backward(
    left: &CudaBuffer,
    right: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> CudaResult<CudaBuffer> {
    let output = matmul_bf16_tensor_core_rhs_t_f32_buffers_impl(left, right, m, k, n)?;
    BF16_TENSOR_CORE_MATMUL_CALLS.fetch_add(1, Ordering::Relaxed);
    BF16_TENSOR_CORE_MATMUL_BACKWARD_CALLS.fetch_add(1, Ordering::Relaxed);
    Ok(output)
}

pub fn matmul_bf16_tensor_core_normal_rhs_f32_buffers_backward(
    left: &CudaBuffer,
    right: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> CudaResult<CudaBuffer> {
    ensure_bf16_buffer(
        left,
        "matmul_bf16_tensor_core_normal_rhs_f32_buffers_backward left",
    )?;
    ensure_bf16_buffer(
        right,
        "matmul_bf16_tensor_core_normal_rhs_f32_buffers_backward right",
    )?;
    ensure_same_device(
        left,
        right,
        "matmul_bf16_tensor_core_normal_rhs_f32_buffers_backward",
    )?;
    if !device_supports_bf16_tensor_cores(left.device_ordinal())? {
        return Err(CudaError::new(format!(
            "matmul_bf16_tensor_core_normal_rhs_f32_buffers_backward requires compute capability >= 8.0 on cuda:{}",
            left.device_ordinal()
        )));
    }
    if !bf16_tensor_core_matmul_exact_tile_shape_supported(m, k, n) {
        return Err(CudaError::new(format!(
            "matmul_bf16_tensor_core_normal_rhs_f32_buffers_backward requires exact Tensor Core tile shapes M%16=0 K%16=0 N%8=0, got M={m} K={k} N={n}"
        )));
    }
    ensure_buffer_len(
        left,
        checked_mul(
            m,
            k,
            "matmul_bf16_tensor_core_normal_rhs_f32_buffers_backward left",
        )?,
        "matmul_bf16_tensor_core_normal_rhs_f32_buffers_backward left",
    )?;
    ensure_buffer_len(
        right,
        checked_mul(
            k,
            n,
            "matmul_bf16_tensor_core_normal_rhs_f32_buffers_backward right",
        )?,
        "matmul_bf16_tensor_core_normal_rhs_f32_buffers_backward right",
    )?;

    let output_len = checked_mul(
        m,
        n,
        "matmul_bf16_tensor_core_normal_rhs_f32_buffers_backward output",
    )?;
    let output = CudaBuffer::uninit_bytes(
        left.device_ordinal(),
        output_len,
        std::mem::size_of::<f32>(),
    )?;

    left.inner.driver.set_current(left.inner.context)?;
    let module = load_module(&left.inner.driver, TENSOR_CORE_PTX)?;
    let function = module.function("heirloom_matmul_bf16_mma_normal_rhs_f32_cta_staged")?;
    let mut timer = start_tensor_core_gemm_timer(left.device_ordinal())?;
    launch_bf16_mma_matmul_normal_rhs_staged_cta_kernel(
        &left.inner.driver,
        function,
        &[left.inner.ptr, right.inner.ptr, output.inner.ptr],
        m,
        k,
        n,
    )?;
    stop_tensor_core_gemm_timer(&mut timer, &CUDA_TENSOR_CORE_STAGED_CTA_GEMM_ELAPSED_US)?;

    BF16_TENSOR_CORE_MATMUL_CALLS.fetch_add(1, Ordering::Relaxed);
    BF16_TENSOR_CORE_MATMUL_BACKWARD_CALLS.fetch_add(1, Ordering::Relaxed);
    Ok(output)
}

fn matmul_bf16_tensor_core_rhs_t_f32_buffers_impl(
    left: &CudaBuffer,
    right: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> CudaResult<CudaBuffer> {
    ensure_bf16_buffer(left, "matmul_bf16_tensor_core_rhs_t_f32_buffers left")?;
    ensure_bf16_buffer(right, "matmul_bf16_tensor_core_rhs_t_f32_buffers right")?;
    ensure_same_device(left, right, "matmul_bf16_tensor_core_rhs_t_f32_buffers")?;
    if !device_supports_bf16_tensor_cores(left.device_ordinal())? {
        return Err(CudaError::new(format!(
            "matmul_bf16_tensor_core_rhs_t_f32_buffers requires compute capability >= 8.0 on cuda:{}",
            left.device_ordinal()
        )));
    }
    if !bf16_tensor_core_matmul_shape_supported(m, k, n) {
        return Err(CudaError::new(format!(
            "matmul_bf16_tensor_core_rhs_t_f32_buffers requires non-zero dimensions, got M={m} K={k} N={n}"
        )));
    }
    ensure_buffer_len(
        left,
        checked_mul(m, k, "matmul_bf16_tensor_core_rhs_t_f32_buffers left")?,
        "matmul_bf16_tensor_core_rhs_t_f32_buffers left",
    )?;
    ensure_buffer_len(
        right,
        checked_mul(n, k, "matmul_bf16_tensor_core_rhs_t_f32_buffers right")?,
        "matmul_bf16_tensor_core_rhs_t_f32_buffers right",
    )?;

    let padded_m = round_up_to_multiple(m, 16, "Tensor Core matmul M")?;
    let padded_k = round_up_to_multiple(k, 16, "Tensor Core matmul K")?;
    let padded_n = round_up_to_multiple(n, 8, "Tensor Core matmul N")?;
    let needs_padding = padded_m != m || padded_k != k || padded_n != n;
    if needs_padding {
        record_tensor_core_padding(m, k, n, padded_m, padded_k, padded_n)?;
    }

    let left_work = if needs_padding {
        pad_matrix_bf16_buffer(left, m, k, padded_m, padded_k)?
    } else {
        left.clone()
    };
    let right_work = if needs_padding {
        pad_matrix_bf16_buffer(right, n, k, padded_n, padded_k)?
    } else {
        right.clone()
    };

    let padded_output_len = checked_mul(
        padded_m,
        padded_n,
        "matmul_bf16_tensor_core_rhs_t_f32_buffers padded output",
    )?;
    let padded_output = CudaBuffer::uninit_bytes(
        left.device_ordinal(),
        padded_output_len,
        std::mem::size_of::<f32>(),
    )?;

    left.inner.driver.set_current(left.inner.context)?;
    let module = load_module(&left.inner.driver, TENSOR_CORE_PTX)?;
    if tensor_core_legacy_warp_gemm_enabled() {
        let function = module.function("heirloom_matmul_bf16_mma_rhs_t_f32")?;
        let mut timer = start_tensor_core_gemm_timer(left.device_ordinal())?;
        launch_bf16_mma_matmul_legacy_warp_kernel(
            &left.inner.driver,
            function,
            &[
                left_work.inner.ptr,
                right_work.inner.ptr,
                padded_output.inner.ptr,
            ],
            padded_m,
            padded_k,
            padded_n,
        )?;
        stop_tensor_core_gemm_timer(&mut timer, &CUDA_TENSOR_CORE_LEGACY_WARP_GEMM_ELAPSED_US)?;
    } else if tensor_core_global_cta_gemm_enabled() {
        let function = module.function("heirloom_matmul_bf16_mma_rhs_t_f32_cta")?;
        let mut timer = start_tensor_core_gemm_timer(left.device_ordinal())?;
        launch_bf16_mma_matmul_global_cta_kernel(
            &left.inner.driver,
            function,
            &[
                left_work.inner.ptr,
                right_work.inner.ptr,
                padded_output.inner.ptr,
            ],
            padded_m,
            padded_k,
            padded_n,
        )?;
        stop_tensor_core_gemm_timer(&mut timer, &CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM_ELAPSED_US)?;
    } else if tensor_core_wide_swizzled_gemm_enabled() {
        let function = module.function("heirloom_matmul_bf16_mma_rhs_t_f32_cta_wide_swizzled")?;
        let mut timer = start_tensor_core_gemm_timer(left.device_ordinal())?;
        launch_bf16_mma_matmul_wide_swizzled_cta_kernel(
            &left.inner.driver,
            function,
            &[
                left_work.inner.ptr,
                right_work.inner.ptr,
                padded_output.inner.ptr,
            ],
            padded_m,
            padded_k,
            padded_n,
        )?;
        stop_tensor_core_gemm_timer(
            &mut timer,
            &CUDA_TENSOR_CORE_WIDE_SWIZZLED_CTA_GEMM_ELAPSED_US,
        )?;
    } else if tensor_core_cp_async_gemm_enabled() {
        CUDA_TENSOR_CORE_CP_ASYNC_GEMM_REQUESTED_CALLS.fetch_add(1, Ordering::Relaxed);
        let function = module.function("heirloom_matmul_bf16_mma_rhs_t_f32_cta_cp_async_db")?;
        let mut timer = start_tensor_core_gemm_timer(left.device_ordinal())?;
        launch_bf16_mma_matmul_cp_async_cta_kernel(
            &left.inner.driver,
            function,
            &[
                left_work.inner.ptr,
                right_work.inner.ptr,
                padded_output.inner.ptr,
            ],
            padded_m,
            padded_k,
            padded_n,
        )?;
        stop_tensor_core_gemm_timer(&mut timer, &CUDA_TENSOR_CORE_CP_ASYNC_GEMM_ELAPSED_US)?;
    } else if tensor_core_ldmatrix_gemm_enabled() {
        CUDA_TENSOR_CORE_LDMATRIX_GEMM_REQUESTED_CALLS.fetch_add(1, Ordering::Relaxed);
        let function = module.function("heirloom_matmul_bf16_mma_rhs_t_f32_cta_ldmatrix_a")?;
        let mut timer = start_tensor_core_gemm_timer(left.device_ordinal())?;
        launch_bf16_mma_matmul_ldmatrix_cta_kernel(
            &left.inner.driver,
            function,
            &[
                left_work.inner.ptr,
                right_work.inner.ptr,
                padded_output.inner.ptr,
            ],
            padded_m,
            padded_k,
            padded_n,
        )?;
        stop_tensor_core_gemm_timer(&mut timer, &CUDA_TENSOR_CORE_LDMATRIX_GEMM_ELAPSED_US)?;
    } else {
        let function = module.function("heirloom_matmul_bf16_mma_rhs_t_f32_cta_staged")?;
        let mut timer = start_tensor_core_gemm_timer(left.device_ordinal())?;
        launch_bf16_mma_matmul_staged_cta_kernel(
            &left.inner.driver,
            function,
            &[
                left_work.inner.ptr,
                right_work.inner.ptr,
                padded_output.inner.ptr,
            ],
            padded_m,
            padded_k,
            padded_n,
        )?;
        stop_tensor_core_gemm_timer(&mut timer, &CUDA_TENSOR_CORE_STAGED_CTA_GEMM_ELAPSED_US)?;
    }
    if needs_padding {
        crop_matrix_f32_buffer(&padded_output, m, n, padded_n)
    } else {
        Ok(padded_output)
    }
}

pub fn div_backward_rhs_f32_buffer(
    left: &CudaBuffer,
    right: &CudaBuffer,
    grad_output: &CudaBuffer,
) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(left, "div_backward_rhs_f32_buffer left")?;
    ensure_f32_buffer(right, "div_backward_rhs_f32_buffer right")?;
    ensure_f32_buffer(grad_output, "div_backward_rhs_f32_buffer grad_output")?;
    ensure_same_device_len(left, right, "div_backward_rhs_f32_buffer")?;
    ensure_same_device_len(left, grad_output, "div_backward_rhs_f32_buffer")?;
    let output = CudaBuffer::uninit_bytes(
        left.device_ordinal(),
        left.len(),
        std::mem::size_of::<f32>(),
    )?;
    if left.is_empty() {
        return Ok(output);
    }

    left.inner.driver.set_current(left.inner.context)?;
    let module = load_module(&left.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_div_backward_rhs_f32")?;
    launch_vector_kernel(
        &left.inner.driver,
        function,
        &[
            left.inner.ptr,
            right.inner.ptr,
            grad_output.inner.ptr,
            output.inner.ptr,
        ],
        left.len(),
    )?;
    Ok(output)
}

pub fn matmul_f32_buffers(
    left: &CudaBuffer,
    right: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(left, "matmul_f32_buffers left")?;
    ensure_f32_buffer(right, "matmul_f32_buffers right")?;
    ensure_same_device(left, right, "matmul_f32_buffers")?;
    ensure_buffer_len(
        left,
        checked_mul(m, k, "matmul_f32_buffers left")?,
        "matmul_f32_buffers left",
    )?;
    ensure_buffer_len(
        right,
        checked_mul(k, n, "matmul_f32_buffers right")?,
        "matmul_f32_buffers right",
    )?;
    let output_len = checked_mul(m, n, "matmul_f32_buffers output")?;
    let output = CudaBuffer::uninit_bytes(
        left.device_ordinal(),
        output_len,
        std::mem::size_of::<f32>(),
    )?;
    if output_len == 0 {
        return Ok(output);
    }

    left.inner.driver.set_current(left.inner.context)?;
    let module = load_module(&left.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_matmul_f32")?;
    launch_matmul_kernel(
        &left.inner.driver,
        function,
        &[left.inner.ptr, right.inner.ptr, output.inner.ptr],
        m,
        k,
        n,
        output_len,
    )?;
    Ok(output)
}

pub fn matmul_strided_f32_buffers(
    left: &CudaBuffer,
    right: &CudaBuffer,
    dims: MatmulStridedDims,
) -> CudaResult<CudaBuffer> {
    validate_matmul_strided_buffers(left, right, dims, "matmul_strided_f32_buffers")?;
    let m = dims.left.rows;
    let n = dims.right.cols;
    let output_len = checked_mul(m, n, "matmul_strided_f32_buffers output")?;
    let output = CudaBuffer::uninit_bytes(
        left.device_ordinal(),
        output_len,
        std::mem::size_of::<f32>(),
    )?;
    if output_len == 0 {
        return Ok(output);
    }

    left.inner.driver.set_current(left.inner.context)?;
    let module = load_module(&left.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_matmul_strided_f32")?;
    launch_matmul_strided_kernel(
        &left.inner.driver,
        function,
        &[left.inner.ptr, right.inner.ptr, output.inner.ptr],
        dims,
        output_len,
    )?;
    Ok(output)
}

pub fn matmul_grad_left_f32_buffers(
    grad_output: &CudaBuffer,
    right: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(grad_output, "matmul_grad_left_f32_buffers grad_output")?;
    ensure_f32_buffer(right, "matmul_grad_left_f32_buffers right")?;
    ensure_same_device(grad_output, right, "matmul_grad_left_f32_buffers")?;
    ensure_buffer_len(
        grad_output,
        checked_mul(m, n, "matmul_grad_left_f32_buffers grad_output")?,
        "matmul_grad_left_f32_buffers grad_output",
    )?;
    ensure_buffer_len(
        right,
        checked_mul(k, n, "matmul_grad_left_f32_buffers right")?,
        "matmul_grad_left_f32_buffers right",
    )?;
    let output_len = checked_mul(m, k, "matmul_grad_left_f32_buffers output")?;
    let output = CudaBuffer::uninit_bytes(
        grad_output.device_ordinal(),
        output_len,
        std::mem::size_of::<f32>(),
    )?;
    if output_len == 0 {
        return Ok(output);
    }

    grad_output
        .inner
        .driver
        .set_current(grad_output.inner.context)?;
    let module = load_module(&grad_output.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_matmul_grad_left_f32")?;
    launch_matmul_kernel(
        &grad_output.inner.driver,
        function,
        &[grad_output.inner.ptr, right.inner.ptr, output.inner.ptr],
        m,
        k,
        n,
        output_len,
    )?;
    Ok(output)
}

pub fn matmul_strided_grad_left_f32_buffers(
    grad_output: &CudaBuffer,
    right: &CudaBuffer,
    dims: MatmulStridedDims,
) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(
        grad_output,
        "matmul_strided_grad_left_f32_buffers grad_output",
    )?;
    validate_matrix_layout_buffer(
        right,
        dims.right,
        "matmul_strided_grad_left_f32_buffers right",
    )?;
    ensure_same_device(grad_output, right, "matmul_strided_grad_left_f32_buffers")?;
    ensure_matmul_dims(dims, "matmul_strided_grad_left_f32_buffers")?;
    let m = dims.left.rows;
    let k = dims.left.cols;
    let n = dims.right.cols;
    ensure_buffer_len(
        grad_output,
        checked_mul(m, n, "matmul_strided_grad_left_f32_buffers grad_output")?,
        "matmul_strided_grad_left_f32_buffers grad_output",
    )?;
    let output_len = checked_mul(m, k, "matmul_strided_grad_left_f32_buffers output")?;
    let output = CudaBuffer::uninit_bytes(
        grad_output.device_ordinal(),
        output_len,
        std::mem::size_of::<f32>(),
    )?;
    if output_len == 0 {
        return Ok(output);
    }

    grad_output
        .inner
        .driver
        .set_current(grad_output.inner.context)?;
    let module = load_module(&grad_output.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_matmul_strided_grad_left_f32")?;
    launch_matmul_strided_kernel(
        &grad_output.inner.driver,
        function,
        &[grad_output.inner.ptr, right.inner.ptr, output.inner.ptr],
        dims,
        output_len,
    )?;
    Ok(output)
}

pub fn matmul_grad_right_f32_buffers(
    left: &CudaBuffer,
    grad_output: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(left, "matmul_grad_right_f32_buffers left")?;
    ensure_f32_buffer(grad_output, "matmul_grad_right_f32_buffers grad_output")?;
    ensure_same_device(left, grad_output, "matmul_grad_right_f32_buffers")?;
    ensure_buffer_len(
        left,
        checked_mul(m, k, "matmul_grad_right_f32_buffers left")?,
        "matmul_grad_right_f32_buffers left",
    )?;
    ensure_buffer_len(
        grad_output,
        checked_mul(m, n, "matmul_grad_right_f32_buffers grad_output")?,
        "matmul_grad_right_f32_buffers grad_output",
    )?;
    let output_len = checked_mul(k, n, "matmul_grad_right_f32_buffers output")?;
    let output = CudaBuffer::uninit_bytes(
        left.device_ordinal(),
        output_len,
        std::mem::size_of::<f32>(),
    )?;
    if output_len == 0 {
        return Ok(output);
    }

    left.inner.driver.set_current(left.inner.context)?;
    let module = load_module(&left.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_matmul_grad_right_f32")?;
    launch_matmul_kernel(
        &left.inner.driver,
        function,
        &[left.inner.ptr, grad_output.inner.ptr, output.inner.ptr],
        m,
        k,
        n,
        output_len,
    )?;
    Ok(output)
}

pub fn matmul_strided_grad_right_f32_buffers(
    left: &CudaBuffer,
    grad_output: &CudaBuffer,
    dims: MatmulStridedDims,
) -> CudaResult<CudaBuffer> {
    validate_matrix_layout_buffer(
        left,
        dims.left,
        "matmul_strided_grad_right_f32_buffers left",
    )?;
    ensure_f32_buffer(
        grad_output,
        "matmul_strided_grad_right_f32_buffers grad_output",
    )?;
    ensure_same_device(left, grad_output, "matmul_strided_grad_right_f32_buffers")?;
    ensure_matmul_dims(dims, "matmul_strided_grad_right_f32_buffers")?;
    let m = dims.left.rows;
    let k = dims.left.cols;
    let n = dims.right.cols;
    ensure_buffer_len(
        grad_output,
        checked_mul(m, n, "matmul_strided_grad_right_f32_buffers grad_output")?,
        "matmul_strided_grad_right_f32_buffers grad_output",
    )?;
    let output_len = checked_mul(k, n, "matmul_strided_grad_right_f32_buffers output")?;
    let output = CudaBuffer::uninit_bytes(
        left.device_ordinal(),
        output_len,
        std::mem::size_of::<f32>(),
    )?;
    if output_len == 0 {
        return Ok(output);
    }

    left.inner.driver.set_current(left.inner.context)?;
    let module = load_module(&left.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_matmul_strided_grad_right_f32")?;
    launch_matmul_strided_kernel(
        &left.inner.driver,
        function,
        &[left.inner.ptr, grad_output.inner.ptr, output.inner.ptr],
        dims,
        output_len,
    )?;
    Ok(output)
}

pub fn embedding_f32_i64_buffers(
    indices: &CudaBuffer,
    weight: &CudaBuffer,
    vocab_size: usize,
    embedding_dim: usize,
) -> CudaResult<CudaBuffer> {
    ensure_i64_buffer(indices, "embedding_f32_i64_buffers indices")?;
    ensure_f32_buffer(weight, "embedding_f32_i64_buffers weight")?;
    ensure_same_device(indices, weight, "embedding_f32_i64_buffers")?;
    if embedding_dim == 0 {
        return Err(CudaError::new(
            "embedding_f32_i64_buffers currently requires embedding_dim > 0",
        ));
    }
    ensure_buffer_len(
        weight,
        checked_mul(
            vocab_size,
            embedding_dim,
            "embedding_f32_i64_buffers weight",
        )?,
        "embedding_f32_i64_buffers weight",
    )?;
    let output_len = checked_mul(
        indices.len(),
        embedding_dim,
        "embedding_f32_i64_buffers output",
    )?;
    let output = CudaBuffer::uninit_bytes(
        indices.device_ordinal(),
        output_len,
        std::mem::size_of::<f32>(),
    )?;
    if output_len == 0 {
        return Ok(output);
    }

    let status = CudaBuffer::from_u32(indices.device_ordinal(), &[0])?;
    indices.inner.driver.set_current(indices.inner.context)?;
    let module = load_module(&indices.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_embedding_f32_i64")?;
    launch_embedding_kernel(
        &indices.inner.driver,
        function,
        &[
            indices.inner.ptr,
            weight.inner.ptr,
            output.inner.ptr,
            status.inner.ptr,
        ],
        indices.len(),
        vocab_size,
        embedding_dim,
        output_len,
    )?;
    ensure_status_zero(&status, "embedding_f32_i64_buffers")?;
    Ok(output)
}

pub fn embedding_backward_f32_i64_buffers(
    indices: &CudaBuffer,
    grad_output: &CudaBuffer,
    vocab_size: usize,
    embedding_dim: usize,
) -> CudaResult<CudaBuffer> {
    ensure_i64_buffer(indices, "embedding_backward_f32_i64_buffers indices")?;
    ensure_f32_buffer(
        grad_output,
        "embedding_backward_f32_i64_buffers grad_output",
    )?;
    ensure_same_device(indices, grad_output, "embedding_backward_f32_i64_buffers")?;
    if embedding_dim == 0 {
        return Err(CudaError::new(
            "embedding_backward_f32_i64_buffers currently requires embedding_dim > 0",
        ));
    }
    ensure_buffer_len(
        grad_output,
        checked_mul(
            indices.len(),
            embedding_dim,
            "embedding_backward_f32_i64_buffers grad_output",
        )?,
        "embedding_backward_f32_i64_buffers grad_output",
    )?;
    let grad_weight_len = checked_mul(
        vocab_size,
        embedding_dim,
        "embedding_backward_f32_i64_buffers grad_weight",
    )?;
    let grad_weight = fill_constant_f32_buffer(indices.device_ordinal(), grad_weight_len, 0.0)?;
    if grad_output.is_empty() {
        return Ok(grad_weight);
    }

    let status = CudaBuffer::from_u32(indices.device_ordinal(), &[0])?;
    indices.inner.driver.set_current(indices.inner.context)?;
    let module = load_module(&indices.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_embedding_backward_f32_i64")?;
    launch_embedding_kernel(
        &indices.inner.driver,
        function,
        &[
            indices.inner.ptr,
            grad_output.inner.ptr,
            grad_weight.inner.ptr,
            status.inner.ptr,
        ],
        indices.len(),
        vocab_size,
        embedding_dim,
        grad_output.len(),
    )?;
    ensure_status_zero(&status, "embedding_backward_f32_i64_buffers")?;
    Ok(grad_weight)
}

pub fn layer_norm_forward_f32_buffers(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    bias: &CudaBuffer,
    rows: usize,
    features: usize,
    eps: f32,
) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(input, "layer_norm_forward_f32_buffers input")?;
    ensure_f32_buffer(weight, "layer_norm_forward_f32_buffers weight")?;
    ensure_f32_buffer(bias, "layer_norm_forward_f32_buffers bias")?;
    ensure_same_device(input, weight, "layer_norm_forward_f32_buffers")?;
    ensure_same_device(input, bias, "layer_norm_forward_f32_buffers")?;
    ensure_positive_eps(eps, "layer_norm_forward_f32_buffers")?;
    ensure_buffer_len(
        input,
        checked_mul(rows, features, "layer_norm_forward_f32_buffers input")?,
        "layer_norm_forward_f32_buffers input",
    )?;
    ensure_buffer_len(weight, features, "layer_norm_forward_f32_buffers weight")?;
    ensure_buffer_len(bias, features, "layer_norm_forward_f32_buffers bias")?;
    let output = CudaBuffer::uninit_bytes(
        input.device_ordinal(),
        input.len(),
        std::mem::size_of::<f32>(),
    )?;
    if input.is_empty() {
        return Ok(output);
    }

    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_layer_norm_forward_f32")?;
    launch_layer_norm_kernel(
        &input.inner.driver,
        function,
        &[
            input.inner.ptr,
            weight.inner.ptr,
            bias.inner.ptr,
            output.inner.ptr,
        ],
        rows,
        features,
        eps,
        input.len(),
    )?;
    Ok(output)
}

pub fn layer_norm_backward_input_f32_buffers(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    grad_output: &CudaBuffer,
    rows: usize,
    features: usize,
    eps: f32,
) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(input, "layer_norm_backward_input_f32_buffers input")?;
    ensure_f32_buffer(weight, "layer_norm_backward_input_f32_buffers weight")?;
    ensure_f32_buffer(
        grad_output,
        "layer_norm_backward_input_f32_buffers grad_output",
    )?;
    ensure_same_device(input, weight, "layer_norm_backward_input_f32_buffers")?;
    ensure_same_device(input, grad_output, "layer_norm_backward_input_f32_buffers")?;
    ensure_positive_eps(eps, "layer_norm_backward_input_f32_buffers")?;
    ensure_buffer_len(
        input,
        checked_mul(
            rows,
            features,
            "layer_norm_backward_input_f32_buffers input",
        )?,
        "layer_norm_backward_input_f32_buffers input",
    )?;
    ensure_buffer_len(
        weight,
        features,
        "layer_norm_backward_input_f32_buffers weight",
    )?;
    ensure_buffer_len(
        grad_output,
        input.len(),
        "layer_norm_backward_input_f32_buffers grad_output",
    )?;
    let output = CudaBuffer::uninit_bytes(
        input.device_ordinal(),
        input.len(),
        std::mem::size_of::<f32>(),
    )?;
    if input.is_empty() {
        return Ok(output);
    }

    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_layer_norm_backward_input_f32")?;
    launch_layer_norm_kernel(
        &input.inner.driver,
        function,
        &[
            input.inner.ptr,
            weight.inner.ptr,
            grad_output.inner.ptr,
            output.inner.ptr,
        ],
        rows,
        features,
        eps,
        input.len(),
    )?;
    Ok(output)
}

pub fn layer_norm_backward_weight_bias_f32_buffers(
    input: &CudaBuffer,
    grad_output: &CudaBuffer,
    rows: usize,
    features: usize,
    eps: f32,
) -> CudaResult<(CudaBuffer, CudaBuffer)> {
    ensure_f32_buffer(input, "layer_norm_backward_weight_bias_f32_buffers input")?;
    ensure_f32_buffer(
        grad_output,
        "layer_norm_backward_weight_bias_f32_buffers grad_output",
    )?;
    ensure_same_device(
        input,
        grad_output,
        "layer_norm_backward_weight_bias_f32_buffers",
    )?;
    ensure_positive_eps(eps, "layer_norm_backward_weight_bias_f32_buffers")?;
    ensure_buffer_len(
        input,
        checked_mul(
            rows,
            features,
            "layer_norm_backward_weight_bias_f32_buffers input",
        )?,
        "layer_norm_backward_weight_bias_f32_buffers input",
    )?;
    ensure_buffer_len(
        grad_output,
        input.len(),
        "layer_norm_backward_weight_bias_f32_buffers grad_output",
    )?;
    let grad_weight =
        CudaBuffer::uninit_bytes(input.device_ordinal(), features, std::mem::size_of::<f32>())?;
    let grad_bias =
        CudaBuffer::uninit_bytes(input.device_ordinal(), features, std::mem::size_of::<f32>())?;
    if features == 0 {
        return Ok((grad_weight, grad_bias));
    }

    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_layer_norm_backward_weight_bias_f32")?;
    launch_layer_norm_kernel(
        &input.inner.driver,
        function,
        &[
            input.inner.ptr,
            grad_output.inner.ptr,
            grad_weight.inner.ptr,
            grad_bias.inner.ptr,
        ],
        rows,
        features,
        eps,
        features,
    )?;
    Ok((grad_weight, grad_bias))
}

pub fn cross_entropy_forward_f32_i64_buffers(
    logits: &CudaBuffer,
    targets: &CudaBuffer,
    batch: usize,
    classes: usize,
) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(logits, "cross_entropy_forward_f32_i64_buffers logits")?;
    ensure_i64_buffer(targets, "cross_entropy_forward_f32_i64_buffers targets")?;
    ensure_same_device(logits, targets, "cross_entropy_forward_f32_i64_buffers")?;
    ensure_buffer_len(
        logits,
        checked_mul(
            batch,
            classes,
            "cross_entropy_forward_f32_i64_buffers logits",
        )?,
        "cross_entropy_forward_f32_i64_buffers logits",
    )?;
    ensure_buffer_len(
        targets,
        batch,
        "cross_entropy_forward_f32_i64_buffers targets",
    )?;
    if batch == 0 || classes == 0 {
        return Err(CudaError::new(
            "cross_entropy_forward_f32_i64_buffers requires non-empty batch/classes",
        ));
    }
    let output = CudaBuffer::uninit_bytes(logits.device_ordinal(), 1, std::mem::size_of::<f32>())?;
    let status = CudaBuffer::from_u32(logits.device_ordinal(), &[0])?;

    logits.inner.driver.set_current(logits.inner.context)?;
    let module = load_module(&logits.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_cross_entropy_forward_f32_i64")?;
    launch_cross_entropy_forward_kernel(
        &logits.inner.driver,
        function,
        &[
            logits.inner.ptr,
            targets.inner.ptr,
            output.inner.ptr,
            status.inner.ptr,
        ],
        batch,
        classes,
    )?;
    ensure_status_zero(&status, "cross_entropy_forward_f32_i64_buffers")?;
    Ok(output)
}

pub fn cross_entropy_backward_f32_i64_buffers(
    logits: &CudaBuffer,
    targets: &CudaBuffer,
    grad_output: &CudaBuffer,
    batch: usize,
    classes: usize,
) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(logits, "cross_entropy_backward_f32_i64_buffers logits")?;
    ensure_i64_buffer(targets, "cross_entropy_backward_f32_i64_buffers targets")?;
    ensure_f32_buffer(
        grad_output,
        "cross_entropy_backward_f32_i64_buffers grad_output",
    )?;
    ensure_same_device(logits, targets, "cross_entropy_backward_f32_i64_buffers")?;
    ensure_same_device(
        logits,
        grad_output,
        "cross_entropy_backward_f32_i64_buffers",
    )?;
    ensure_buffer_len(
        logits,
        checked_mul(
            batch,
            classes,
            "cross_entropy_backward_f32_i64_buffers logits",
        )?,
        "cross_entropy_backward_f32_i64_buffers logits",
    )?;
    ensure_buffer_len(
        targets,
        batch,
        "cross_entropy_backward_f32_i64_buffers targets",
    )?;
    ensure_buffer_len(
        grad_output,
        1,
        "cross_entropy_backward_f32_i64_buffers grad_output",
    )?;
    if batch == 0 || classes == 0 {
        return Err(CudaError::new(
            "cross_entropy_backward_f32_i64_buffers requires non-empty batch/classes",
        ));
    }
    let output = CudaBuffer::uninit_bytes(
        logits.device_ordinal(),
        logits.len(),
        std::mem::size_of::<f32>(),
    )?;
    let status = CudaBuffer::from_u32(logits.device_ordinal(), &[0])?;

    logits.inner.driver.set_current(logits.inner.context)?;
    let module = load_module(&logits.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_cross_entropy_backward_f32_i64")?;
    launch_cross_entropy_backward_kernel(
        &logits.inner.driver,
        function,
        &[
            logits.inner.ptr,
            targets.inner.ptr,
            grad_output.inner.ptr,
            output.inner.ptr,
            status.inner.ptr,
        ],
        batch,
        classes,
        logits.len(),
    )?;
    ensure_status_zero(&status, "cross_entropy_backward_f32_i64_buffers")?;
    Ok(output)
}

pub fn causal_attention_forward_f32_buffers(
    query: &CudaBuffer,
    key: &CudaBuffer,
    value: &CudaBuffer,
    dims: CausalAttentionDims,
) -> CudaResult<(CudaBuffer, CudaBuffer)> {
    let head_dim = validate_causal_attention_buffers(
        query,
        key,
        value,
        dims,
        "causal_attention_forward_f32_buffers",
    )?;
    let qkv_len = checked_mul(
        checked_mul(
            dims.batch,
            dims.time,
            "causal_attention_forward_f32_buffers batch*time",
        )?,
        dims.channels,
        "causal_attention_forward_f32_buffers qkv",
    )?;
    let attention_len = checked_mul(
        checked_mul(
            checked_mul(
                dims.batch,
                dims.n_heads,
                "causal_attention_forward_f32_buffers batch*heads",
            )?,
            dims.time,
            "causal_attention_forward_f32_buffers batch*heads*time",
        )?,
        dims.time,
        "causal_attention_forward_f32_buffers attention",
    )?;
    let output =
        CudaBuffer::uninit_bytes(query.device_ordinal(), qkv_len, std::mem::size_of::<f32>())?;
    let attention = CudaBuffer::uninit_bytes(
        query.device_ordinal(),
        attention_len,
        std::mem::size_of::<f32>(),
    )?;

    query.inner.driver.set_current(query.inner.context)?;
    let module = load_module(&query.inner.driver, KERNEL_PTX)?;
    let weights_function = module.function("heirloom_causal_attention_weights_f32")?;
    launch_causal_attention_kernel(
        &query.inner.driver,
        weights_function,
        &[query.inner.ptr, key.inner.ptr, attention.inner.ptr],
        dims,
        head_dim,
        attention_len,
    )?;
    let output_function = module.function("heirloom_causal_attention_output_f32")?;
    launch_causal_attention_kernel(
        &query.inner.driver,
        output_function,
        &[attention.inner.ptr, value.inner.ptr, output.inner.ptr],
        dims,
        head_dim,
        qkv_len,
    )?;
    Ok((output, attention))
}

pub fn causal_attention_bf16_flash_forward_f32_buffers(
    query: &CudaBuffer,
    key: &CudaBuffer,
    value: &CudaBuffer,
    dims: CausalAttentionDims,
) -> CudaResult<CudaBuffer> {
    CUDA_FLASH_BF16_ATTENTION_REQUESTED_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_FLASH_BF16_SCALAR_STREAMING_REQUESTED_CALLS.fetch_add(1, Ordering::Relaxed);
    let head_dim = validate_causal_attention_buffers(
        query,
        key,
        value,
        dims,
        "causal_attention_bf16_flash_forward_f32_buffers",
    )?;
    if !device_supports_bf16_tensor_cores(query.device_ordinal())? {
        return Err(CudaError::new(format!(
            "causal_attention_bf16_flash_forward_f32_buffers requires compute capability >= 8.0 on cuda:{}",
            query.device_ordinal()
        )));
    }

    let qkv_len = checked_mul(
        checked_mul(
            dims.batch,
            dims.time,
            "causal_attention_bf16_flash_forward_f32_buffers batch*time",
        )?,
        dims.channels,
        "causal_attention_bf16_flash_forward_f32_buffers qkv",
    )?;
    let query_bf16 = f32_to_bf16_buffer(query)?;
    let key_bf16 = f32_to_bf16_buffer(key)?;
    let value_bf16 = f32_to_bf16_buffer(value)?;
    let output =
        CudaBuffer::uninit_bytes(query.device_ordinal(), qkv_len, std::mem::size_of::<f32>())?;
    if qkv_len == 0 {
        return Ok(output);
    }

    query.inner.driver.set_current(query.inner.context)?;
    let mut timer = if flash_bf16_attention_timing_enabled() {
        Some(CudaEventTimer::start_current_compute_stream(
            query.device_ordinal(),
        )?)
    } else {
        None
    };
    let tensor_core_module = load_module(&query.inner.driver, TENSOR_CORE_PTX)?;
    let function = tensor_core_module.function("heirloom_flash_attention_bf16_fwd_f32")?;
    launch_causal_attention_kernel(
        &query.inner.driver,
        function,
        &[
            query_bf16.inner.ptr,
            key_bf16.inner.ptr,
            value_bf16.inner.ptr,
            output.inner.ptr,
        ],
        dims,
        head_dim,
        qkv_len,
    )?;
    if let Some(timer) = timer.as_mut() {
        let elapsed_ms = timer.stop_elapsed_ms()?;
        let elapsed_us = (elapsed_ms * 1000.0).max(0.0) as usize;
        CUDA_FLASH_BF16_ATTENTION_ELAPSED_US.fetch_add(elapsed_us, Ordering::Relaxed);
        CUDA_FLASH_BF16_SCALAR_STREAMING_ELAPSED_US.fetch_add(elapsed_us, Ordering::Relaxed);
    }
    CUDA_FLASH_BF16_ATTENTION_EXECUTED_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_FLASH_BF16_SCALAR_STREAMING_EXECUTED_CALLS.fetch_add(1, Ordering::Relaxed);
    record_flash_scalar_streaming_tiles(dims, head_dim)?;
    Ok(output)
}

pub fn causal_attention_bf16_flash_tensor_core_forward_f32_buffers(
    query: &CudaBuffer,
    key: &CudaBuffer,
    value: &CudaBuffer,
    dims: CausalAttentionDims,
) -> CudaResult<(CudaBuffer, CudaBuffer, CudaBuffer)> {
    let head_dim = validate_causal_attention_buffers(
        query,
        key,
        value,
        dims,
        "causal_attention_bf16_flash_tensor_core_forward_f32_buffers",
    )?;
    if !device_supports_bf16_tensor_cores(query.device_ordinal())? {
        return Err(CudaError::new(format!(
            "causal_attention_bf16_flash_tensor_core_forward_f32_buffers requires compute capability >= 8.0 on cuda:{}",
            query.device_ordinal()
        )));
    }
    let qkv_len = checked_mul(
        checked_mul(
            dims.batch,
            dims.time,
            "causal_attention_bf16_flash_tensor_core_forward_f32_buffers batch*time",
        )?,
        dims.channels,
        "causal_attention_bf16_flash_tensor_core_forward_f32_buffers qkv",
    )?;
    let stats_len = checked_mul(
        checked_mul(
            dims.batch,
            dims.n_heads,
            "causal_attention_bf16_flash_tensor_core_forward_f32_buffers batch*heads",
        )?,
        dims.time,
        "causal_attention_bf16_flash_tensor_core_forward_f32_buffers stats",
    )?;
    CUDA_FLASH_BF16_ATTENTION_REQUESTED_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_REQUESTED_CALLS.fetch_add(1, Ordering::Relaxed);
    let query_bf16 = f32_to_bf16_buffer(query)?;
    let key_bf16 = f32_to_bf16_buffer(key)?;
    let value_bf16 = f32_to_bf16_buffer(value)?;
    let output =
        CudaBuffer::uninit_bytes(query.device_ordinal(), qkv_len, std::mem::size_of::<f32>())?;
    let row_max = CudaBuffer::uninit_bytes(
        query.device_ordinal(),
        stats_len,
        std::mem::size_of::<f32>(),
    )?;
    let row_denom = CudaBuffer::uninit_bytes(
        query.device_ordinal(),
        stats_len,
        std::mem::size_of::<f32>(),
    )?;

    query.inner.driver.set_current(query.inner.context)?;
    let mut timer = if flash_bf16_attention_timing_enabled() {
        Some(CudaEventTimer::start_current_compute_stream(
            query.device_ordinal(),
        )?)
    } else {
        None
    };
    let tensor_core_module = load_module(&query.inner.driver, TENSOR_CORE_PTX)?;
    let function = tensor_core_module.function("heirloom_flash_attention_bf16_tc_fwd_f32")?;
    launch_flash_bf16_tensor_core_attention_kernel(
        &query.inner.driver,
        function,
        &[
            query_bf16.inner.ptr,
            key_bf16.inner.ptr,
            value_bf16.inner.ptr,
            output.inner.ptr,
            row_max.inner.ptr,
            row_denom.inner.ptr,
        ],
        dims,
        head_dim,
    )?;
    if let Some(timer) = timer.as_mut() {
        let elapsed_ms = timer.stop_elapsed_ms()?;
        let elapsed_us = (elapsed_ms * 1000.0).max(0.0) as usize;
        CUDA_FLASH_BF16_ATTENTION_ELAPSED_US.fetch_add(elapsed_us, Ordering::Relaxed);
        CUDA_FLASH_BF16_TENSOR_CORE_ELAPSED_US.fetch_add(elapsed_us, Ordering::Relaxed);
    }
    let batch_heads = checked_mul(
        dims.batch,
        dims.n_heads,
        "flash BF16 Tensor Core attention batch*heads tile count",
    )?;
    let query_blocks = dims.time.div_ceil(16);
    let key_blocks = dims.time.div_ceil(16);
    let output_dim_tiles = head_dim.div_ceil(8);
    let head_dim_chunks = head_dim.div_ceil(16);
    let qk_base_tiles = checked_mul(
        checked_mul(
            checked_mul(
                batch_heads,
                query_blocks,
                "flash BF16 Tensor Core attention QK batch query tiles",
            )?,
            key_blocks,
            "flash BF16 Tensor Core attention QK key tiles",
        )?,
        output_dim_tiles,
        "flash BF16 Tensor Core attention QK output tile replay count",
    )?;
    let qk_tiles = checked_mul(
        checked_mul(
            qk_base_tiles,
            head_dim_chunks,
            "flash BF16 Tensor Core attention QK head-dim chunks",
        )?,
        2,
        "flash BF16 Tensor Core attention QK two n8 MMA tiles per k16 head-dim chunk",
    )?;
    let av_tiles = qk_base_tiles;
    CUDA_FLASH_BF16_ATTENTION_EXECUTED_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_EXECUTED_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_FLASH_BF16_ATTENTION_QK_TILE_CALLS.fetch_add(qk_tiles, Ordering::Relaxed);
    CUDA_FLASH_BF16_ATTENTION_AV_TILE_CALLS.fetch_add(av_tiles, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_QK_MMA_TILE_CALLS.fetch_add(qk_tiles, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_AV_MMA_TILE_CALLS.fetch_add(av_tiles, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_FORWARD_CALLS.fetch_add(1, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_QK_MATMUL_CALLS.fetch_add(qk_tiles, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_AV_MATMUL_CALLS.fetch_add(av_tiles, Ordering::Relaxed);
    BF16_TENSOR_CORE_MATMUL_CALLS.fetch_add(qk_tiles + av_tiles, Ordering::Relaxed);
    BF16_TENSOR_CORE_MATMUL_FORWARD_CALLS.fetch_add(qk_tiles + av_tiles, Ordering::Relaxed);
    let mut causal_tiles = 0usize;
    for query_block in 0..query_blocks {
        let query_start = query_block * 16;
        for key_block in 0..key_blocks {
            let key_end = ((key_block + 1) * 16).min(dims.time);
            if key_end > query_start + 1 {
                causal_tiles += 1;
            }
        }
    }
    let causal_tiles = checked_mul(
        checked_mul(
            causal_tiles,
            batch_heads,
            "flash BF16 Tensor Core attention causal batch tiles",
        )?,
        output_dim_tiles,
        "flash BF16 Tensor Core attention causal output tiles",
    )?;
    CUDA_FLASH_BF16_ATTENTION_CAUSAL_MASKED_TILE_COUNT.fetch_add(causal_tiles, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_CAUSAL_MASKED_TILE_COUNT.fetch_add(causal_tiles, Ordering::Relaxed);
    if !dims.time.is_multiple_of(16) || !dims.time.is_multiple_of(8) || !head_dim.is_multiple_of(8)
    {
        CUDA_FLASH_BF16_ATTENTION_RAGGED_TILE_COUNT
            .fetch_add(qk_tiles + av_tiles, Ordering::Relaxed);
        CUDA_FLASH_BF16_TENSOR_CORE_RAGGED_TILE_COUNT
            .fetch_add(qk_tiles + av_tiles, Ordering::Relaxed);
    }
    Ok((output, row_max, row_denom))
}

#[allow(clippy::too_many_arguments)]
pub fn causal_attention_bf16_flash_tensor_core_backward_f32_buffers(
    query: &CudaBuffer,
    key: &CudaBuffer,
    value: &CudaBuffer,
    output: &CudaBuffer,
    row_max: &CudaBuffer,
    row_denom: &CudaBuffer,
    grad_output: &CudaBuffer,
    dims: CausalAttentionDims,
) -> CudaResult<(CudaBuffer, CudaBuffer, CudaBuffer)> {
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_REQUESTED_CALLS.fetch_add(1, Ordering::Relaxed);
    let head_dim = validate_causal_attention_buffers(
        query,
        key,
        value,
        dims,
        "causal_attention_bf16_flash_tensor_core_backward_f32_buffers",
    )?;
    ensure_f32_buffer(
        output,
        "causal_attention_bf16_flash_tensor_core_backward_f32_buffers output",
    )?;
    ensure_f32_buffer(
        row_max,
        "causal_attention_bf16_flash_tensor_core_backward_f32_buffers row_max",
    )?;
    ensure_f32_buffer(
        row_denom,
        "causal_attention_bf16_flash_tensor_core_backward_f32_buffers row_denom",
    )?;
    ensure_f32_buffer(
        grad_output,
        "causal_attention_bf16_flash_tensor_core_backward_f32_buffers grad_output",
    )?;
    ensure_same_device(
        query,
        output,
        "causal_attention_bf16_flash_tensor_core_backward_f32_buffers query/output",
    )?;
    ensure_same_device(
        query,
        row_max,
        "causal_attention_bf16_flash_tensor_core_backward_f32_buffers query/row_max",
    )?;
    ensure_same_device(
        query,
        row_denom,
        "causal_attention_bf16_flash_tensor_core_backward_f32_buffers query/row_denom",
    )?;
    ensure_same_device(
        query,
        grad_output,
        "causal_attention_bf16_flash_tensor_core_backward_f32_buffers query/grad_output",
    )?;
    if !device_supports_bf16_tensor_cores(query.device_ordinal())? {
        return Err(CudaError::new(format!(
            "causal_attention_bf16_flash_tensor_core_backward_f32_buffers requires compute capability >= 8.0 on cuda:{}",
            query.device_ordinal()
        )));
    }
    if !causal_attention_bf16_flash_tensor_core_backward_shape_supported(dims) {
        return Err(CudaError::new(format!(
            "causal_attention_bf16_flash_tensor_core_backward_f32_buffers requires tile-aligned sm80 shape for V1 Tensor Core backward: time={} head_dim={}",
            dims.time,
            head_dim
        )));
    }

    let qkv_len = checked_mul(
        checked_mul(
            dims.batch,
            dims.time,
            "causal_attention_bf16_flash_tensor_core_backward_f32_buffers batch*time",
        )?,
        dims.channels,
        "causal_attention_bf16_flash_tensor_core_backward_f32_buffers qkv",
    )?;
    let stats_len = checked_mul(
        checked_mul(
            dims.batch,
            dims.n_heads,
            "causal_attention_bf16_flash_tensor_core_backward_f32_buffers batch*heads",
        )?,
        dims.time,
        "causal_attention_bf16_flash_tensor_core_backward_f32_buffers row stats",
    )?;
    ensure_buffer_len(
        output,
        qkv_len,
        "causal_attention_bf16_flash_tensor_core_backward_f32_buffers output",
    )?;
    ensure_buffer_len(
        row_max,
        stats_len,
        "causal_attention_bf16_flash_tensor_core_backward_f32_buffers row_max",
    )?;
    ensure_buffer_len(
        row_denom,
        stats_len,
        "causal_attention_bf16_flash_tensor_core_backward_f32_buffers row_denom",
    )?;
    ensure_buffer_len(
        grad_output,
        qkv_len,
        "causal_attention_bf16_flash_tensor_core_backward_f32_buffers grad_output",
    )?;

    let query_bf16 = f32_to_bf16_buffer(query)?;
    let key_bf16 = f32_to_bf16_buffer(key)?;
    let value_bf16 = f32_to_bf16_buffer(value)?;
    let grad_output_bf16 = f32_to_bf16_buffer(grad_output)?;
    let row_dot = CudaBuffer::uninit_bytes(
        query.device_ordinal(),
        stats_len,
        std::mem::size_of::<f32>(),
    )?;
    let grad_query = fill_constant_f32_buffer(query.device_ordinal(), qkv_len, 0.0)?;
    let grad_key = fill_constant_f32_buffer(query.device_ordinal(), qkv_len, 0.0)?;
    let grad_value = fill_constant_f32_buffer(query.device_ordinal(), qkv_len, 0.0)?;

    query.inner.driver.set_current(query.inner.context)?;
    let mut timer = if flash_bf16_attention_timing_enabled() {
        Some(CudaEventTimer::start_current_compute_stream(
            query.device_ordinal(),
        )?)
    } else {
        None
    };
    let tensor_core_module = load_module(&query.inner.driver, TENSOR_CORE_PTX)?;
    let row_dot_function =
        tensor_core_module.function("heirloom_flash_attention_bf16_bwd_row_dot_f32")?;
    launch_causal_attention_kernel(
        &query.inner.driver,
        row_dot_function,
        &[
            output.inner.ptr,
            grad_output_bf16.inner.ptr,
            row_dot.inner.ptr,
        ],
        dims,
        head_dim,
        stats_len,
    )?;
    let function =
        tensor_core_module.function("heirloom_flash_attention_bf16_tc_bwd_tiled_mma_f32")?;
    launch_flash_bf16_tensor_core_attention_backward_kernel(
        &query.inner.driver,
        function,
        &[
            query_bf16.inner.ptr,
            key_bf16.inner.ptr,
            value_bf16.inner.ptr,
            output.inner.ptr,
            row_max.inner.ptr,
            row_denom.inner.ptr,
            grad_output_bf16.inner.ptr,
            row_dot.inner.ptr,
            grad_query.inner.ptr,
            grad_key.inner.ptr,
            grad_value.inner.ptr,
        ],
        dims,
        head_dim,
    )?;
    if let Some(timer) = timer.as_mut() {
        let elapsed_ms = timer.stop_elapsed_ms()?;
        let elapsed_us = (elapsed_ms * 1000.0).max(0.0) as usize;
        CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_ELAPSED_US.fetch_add(elapsed_us, Ordering::Relaxed);
    }

    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_EXECUTED_CALLS.fetch_add(1, Ordering::Relaxed);
    let batch_heads = checked_mul(
        dims.batch,
        dims.n_heads,
        "flash BF16 Tensor Core backward batch*heads tile count",
    )?;
    let row_dot_calls = checked_mul(
        batch_heads,
        dims.time,
        "flash BF16 Tensor Core backward row-dot calls",
    )?;
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_ROW_DOT_CALLS.fetch_add(row_dot_calls, Ordering::Relaxed);
    let query_blocks = dims.time.div_ceil(16);
    let key_blocks = dims.time.div_ceil(16);
    let output_dim_tiles = head_dim.div_ceil(8);
    let head_dim_chunks = head_dim.div_ceil(16);
    let replay_tiles = checked_mul(
        checked_mul(
            checked_mul(
                batch_heads,
                query_blocks,
                "flash BF16 Tensor Core backward MMA batch query tiles",
            )?,
            key_blocks,
            "flash BF16 Tensor Core backward MMA key tiles",
        )?,
        output_dim_tiles,
        "flash BF16 Tensor Core backward MMA output tiles",
    )?;
    let qk_recompute_mma_tiles = checked_mul(
        checked_mul(
            replay_tiles,
            head_dim_chunks,
            "flash BF16 Tensor Core backward QK recompute head-dim chunks",
        )?,
        2,
        "flash BF16 Tensor Core backward QK two n8 MMA tiles per k16 chunk",
    )?;
    let dp_mma_tiles = checked_mul(
        checked_mul(
            replay_tiles,
            head_dim_chunks,
            "flash BF16 Tensor Core backward dP head-dim chunks",
        )?,
        2,
        "flash BF16 Tensor Core backward dP two n8 MMA tiles per k16 chunk",
    )?;
    let dq_mma_tiles = replay_tiles;
    let dk_mma_tiles = replay_tiles;
    let dv_mma_tiles = replay_tiles;
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_QK_RECOMPUTE_MMA_TILE_CALLS
        .fetch_add(qk_recompute_mma_tiles, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_DP_MMA_TILE_CALLS
        .fetch_add(dp_mma_tiles, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_DQ_MMA_TILE_CALLS
        .fetch_add(dq_mma_tiles, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_DK_MMA_TILE_CALLS
        .fetch_add(dk_mma_tiles, Ordering::Relaxed);
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_DV_MMA_TILE_CALLS
        .fetch_add(dv_mma_tiles, Ordering::Relaxed);
    let backward_mma_tiles =
        qk_recompute_mma_tiles + dp_mma_tiles + dq_mma_tiles + dk_mma_tiles + dv_mma_tiles;
    BF16_TENSOR_CORE_ATTENTION_BACKWARD_CALLS.fetch_add(1, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_SCORE_GRAD_MATMUL_CALLS.fetch_add(dp_mma_tiles, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_DQ_MATMUL_CALLS.fetch_add(dq_mma_tiles, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_DK_MATMUL_CALLS.fetch_add(dk_mma_tiles, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_DV_MATMUL_CALLS.fetch_add(dv_mma_tiles, Ordering::Relaxed);
    BF16_TENSOR_CORE_MATMUL_CALLS.fetch_add(backward_mma_tiles, Ordering::Relaxed);
    BF16_TENSOR_CORE_MATMUL_BACKWARD_CALLS.fetch_add(backward_mma_tiles, Ordering::Relaxed);
    let mut causal_tiles = 0usize;
    for query_block in 0..query_blocks {
        let query_start = query_block * 16;
        for key_block in 0..key_blocks {
            let key_end = ((key_block + 1) * 16).min(dims.time);
            if key_end > query_start + 1 {
                causal_tiles += 1;
            }
        }
    }
    let causal_tiles = checked_mul(
        checked_mul(
            causal_tiles,
            batch_heads,
            "flash BF16 Tensor Core backward causal batch tiles",
        )?,
        output_dim_tiles,
        "flash BF16 Tensor Core backward causal output tiles",
    )?;
    CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_CAUSAL_MASKED_TILE_COUNT
        .fetch_add(causal_tiles, Ordering::Relaxed);
    if !dims.time.is_multiple_of(16) || !dims.time.is_multiple_of(8) || !head_dim.is_multiple_of(8)
    {
        CUDA_FLASH_BF16_TENSOR_CORE_BACKWARD_RAGGED_TILE_COUNT
            .fetch_add(backward_mma_tiles, Ordering::Relaxed);
    }
    Ok((grad_query, grad_key, grad_value))
}

pub fn causal_attention_bf16_tensor_core_forward_f32_buffers(
    query: &CudaBuffer,
    key: &CudaBuffer,
    value: &CudaBuffer,
    dims: CausalAttentionDims,
) -> CudaResult<(CudaBuffer, CudaBuffer)> {
    let head_dim = validate_causal_attention_buffers(
        query,
        key,
        value,
        dims,
        "causal_attention_bf16_tensor_core_forward_f32_buffers",
    )?;
    if !device_supports_bf16_tensor_cores(query.device_ordinal())? {
        return Err(CudaError::new(format!(
            "causal_attention_bf16_tensor_core_forward_f32_buffers requires compute capability >= 8.0 on cuda:{}",
            query.device_ordinal()
        )));
    }

    let qkv_len = checked_mul(
        checked_mul(
            dims.batch,
            dims.time,
            "causal_attention_bf16_tensor_core_forward_f32_buffers batch*time",
        )?,
        dims.channels,
        "causal_attention_bf16_tensor_core_forward_f32_buffers qkv",
    )?;
    let attention_len = checked_mul(
        checked_mul(
            checked_mul(
                dims.batch,
                dims.n_heads,
                "causal_attention_bf16_tensor_core_forward_f32_buffers batch*heads",
            )?,
            dims.time,
            "causal_attention_bf16_tensor_core_forward_f32_buffers batch*heads*time",
        )?,
        dims.time,
        "causal_attention_bf16_tensor_core_forward_f32_buffers attention",
    )?;
    let query_bf16 = f32_to_bf16_buffer(query)?;
    let key_bf16 = f32_to_bf16_buffer(key)?;
    let value_bf16 = f32_to_bf16_buffer(value)?;
    let scores = CudaBuffer::uninit_bytes(
        query.device_ordinal(),
        attention_len,
        std::mem::size_of::<f32>(),
    )?;

    query.inner.driver.set_current(query.inner.context)?;
    let tensor_core_module = load_module(&query.inner.driver, TENSOR_CORE_PTX)?;
    let qk_function = tensor_core_module.function("heirloom_attention_qk_bf16_mma_f32")?;
    launch_bf16_mma_attention_kernel(
        &query.inner.driver,
        qk_function,
        &[query_bf16.inner.ptr, key_bf16.inner.ptr, scores.inner.ptr],
        AttentionMmaLaunch {
            dims,
            head_dim,
            grid_x: dims.time.div_ceil(8),
            grid_y: dims.time.div_ceil(16),
            grid_z: dims.batch * dims.n_heads,
        },
    )?;

    let attention = CudaBuffer::uninit_bytes(
        query.device_ordinal(),
        attention_len,
        std::mem::size_of::<f32>(),
    )?;
    let kernel_module = load_module(&query.inner.driver, KERNEL_PTX)?;
    let softmax_function =
        kernel_module.function("heirloom_causal_attention_softmax_scores_f32")?;
    launch_causal_attention_kernel(
        &query.inner.driver,
        softmax_function,
        &[scores.inner.ptr, attention.inner.ptr],
        dims,
        head_dim,
        attention_len,
    )?;

    let attention_bf16 = f32_to_bf16_buffer(&attention)?;
    let output =
        CudaBuffer::uninit_bytes(query.device_ordinal(), qkv_len, std::mem::size_of::<f32>())?;
    let av_function = tensor_core_module.function("heirloom_attention_av_bf16_mma_f32")?;
    launch_bf16_mma_attention_kernel(
        &query.inner.driver,
        av_function,
        &[
            attention_bf16.inner.ptr,
            value_bf16.inner.ptr,
            output.inner.ptr,
        ],
        AttentionMmaLaunch {
            dims,
            head_dim,
            grid_x: head_dim.div_ceil(8),
            grid_y: dims.time.div_ceil(16),
            grid_z: dims.batch * dims.n_heads,
        },
    )?;

    let matmul_calls = checked_mul(
        dims.batch,
        dims.n_heads,
        "causal_attention_bf16_tensor_core_forward_f32_buffers matmul calls",
    )?;
    let total_matmul_calls = checked_mul(
        matmul_calls,
        2,
        "causal_attention_bf16_tensor_core_forward_f32_buffers total matmul calls",
    )?;
    if tensor_core_attention_cp_async_enabled() {
        CUDA_TENSOR_CORE_ATTENTION_CP_ASYNC_REQUESTED_CALLS
            .fetch_add(total_matmul_calls, Ordering::Relaxed);
        CUDA_TENSOR_CORE_ATTENTION_CP_ASYNC_GLOBAL_FALLBACK_CALLS
            .fetch_add(total_matmul_calls, Ordering::Relaxed);
    } else if tensor_core_attention_ldmatrix_enabled() {
        CUDA_TENSOR_CORE_ATTENTION_LDMATRIX_REQUESTED_CALLS
            .fetch_add(total_matmul_calls, Ordering::Relaxed);
        CUDA_TENSOR_CORE_ATTENTION_LDMATRIX_GLOBAL_FALLBACK_CALLS
            .fetch_add(total_matmul_calls, Ordering::Relaxed);
    }
    BF16_TENSOR_CORE_ATTENTION_FORWARD_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_BF16_ATTENTION_MATERIALIZED_REFERENCE_CALLS.fetch_add(1, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_QK_MATMUL_CALLS.fetch_add(matmul_calls, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_AV_MATMUL_CALLS.fetch_add(matmul_calls, Ordering::Relaxed);
    BF16_TENSOR_CORE_MATMUL_CALLS.fetch_add(total_matmul_calls, Ordering::Relaxed);
    BF16_TENSOR_CORE_MATMUL_FORWARD_CALLS.fetch_add(total_matmul_calls, Ordering::Relaxed);
    record_attention_tensor_core_edge_tiles(
        dims.time,
        head_dim,
        dims.time,
        matmul_calls,
        "Tensor Core attention QK forward",
    )?;
    record_attention_tensor_core_edge_tiles(
        dims.time,
        dims.time,
        head_dim,
        matmul_calls,
        "Tensor Core attention AV forward",
    )?;
    Ok((output, attention))
}

pub fn causal_attention_bf16_tensor_core_backward_f32_buffers(
    query: &CudaBuffer,
    key: &CudaBuffer,
    value: &CudaBuffer,
    attention: &CudaBuffer,
    grad_output: &CudaBuffer,
    dims: CausalAttentionDims,
) -> CudaResult<(CudaBuffer, CudaBuffer, CudaBuffer)> {
    let head_dim = validate_causal_attention_buffers(
        query,
        key,
        value,
        dims,
        "causal_attention_bf16_tensor_core_backward_f32_buffers",
    )?;
    ensure_f32_buffer(
        attention,
        "causal_attention_bf16_tensor_core_backward_f32_buffers attention",
    )?;
    ensure_f32_buffer(
        grad_output,
        "causal_attention_bf16_tensor_core_backward_f32_buffers grad_output",
    )?;
    ensure_same_device(
        query,
        attention,
        "causal_attention_bf16_tensor_core_backward_f32_buffers query/attention",
    )?;
    ensure_same_device(
        query,
        grad_output,
        "causal_attention_bf16_tensor_core_backward_f32_buffers query/grad_output",
    )?;
    if !device_supports_bf16_tensor_cores(query.device_ordinal())? {
        return Err(CudaError::new(format!(
            "causal_attention_bf16_tensor_core_backward_f32_buffers requires compute capability >= 8.0 on cuda:{}",
            query.device_ordinal()
        )));
    }

    let qkv_len = checked_mul(
        checked_mul(
            dims.batch,
            dims.time,
            "causal_attention_bf16_tensor_core_backward_f32_buffers batch*time",
        )?,
        dims.channels,
        "causal_attention_bf16_tensor_core_backward_f32_buffers qkv",
    )?;
    let attention_len = checked_mul(
        checked_mul(
            checked_mul(
                dims.batch,
                dims.n_heads,
                "causal_attention_bf16_tensor_core_backward_f32_buffers batch*heads",
            )?,
            dims.time,
            "causal_attention_bf16_tensor_core_backward_f32_buffers batch*heads*time",
        )?,
        dims.time,
        "causal_attention_bf16_tensor_core_backward_f32_buffers attention",
    )?;
    ensure_buffer_len(
        attention,
        attention_len,
        "causal_attention_bf16_tensor_core_backward_f32_buffers attention",
    )?;
    ensure_buffer_len(
        grad_output,
        qkv_len,
        "causal_attention_bf16_tensor_core_backward_f32_buffers grad_output",
    )?;

    let query_bf16 = f32_to_bf16_buffer(query)?;
    let key_bf16 = f32_to_bf16_buffer(key)?;
    let value_bf16 = f32_to_bf16_buffer(value)?;
    let grad_output_bf16 = f32_to_bf16_buffer(grad_output)?;
    let score_grad = CudaBuffer::uninit_bytes(
        query.device_ordinal(),
        attention_len,
        std::mem::size_of::<f32>(),
    )?;
    let dscore = CudaBuffer::uninit_bytes(
        query.device_ordinal(),
        attention_len,
        std::mem::size_of::<f32>(),
    )?;
    let attention_t = CudaBuffer::uninit_bytes(
        query.device_ordinal(),
        attention_len,
        std::mem::size_of::<f32>(),
    )?;
    let dscore_t = CudaBuffer::uninit_bytes(
        query.device_ordinal(),
        attention_len,
        std::mem::size_of::<f32>(),
    )?;
    let grad_query =
        CudaBuffer::uninit_bytes(query.device_ordinal(), qkv_len, std::mem::size_of::<f32>())?;
    let grad_key =
        CudaBuffer::uninit_bytes(query.device_ordinal(), qkv_len, std::mem::size_of::<f32>())?;
    let grad_value =
        CudaBuffer::uninit_bytes(query.device_ordinal(), qkv_len, std::mem::size_of::<f32>())?;

    query.inner.driver.set_current(query.inner.context)?;
    let tensor_core_module = load_module(&query.inner.driver, TENSOR_CORE_PTX)?;
    let qk_function = tensor_core_module.function("heirloom_attention_qk_bf16_mma_f32")?;
    let av_function = tensor_core_module.function("heirloom_attention_av_bf16_mma_f32")?;
    let kernel_module = load_module(&query.inner.driver, KERNEL_PTX)?;
    let softmax_backward_function =
        kernel_module.function("heirloom_causal_attention_softmax_backward_scores_f32")?;
    let transpose_attention_function =
        kernel_module.function("heirloom_attention_square_transpose_f32")?;

    launch_bf16_mma_attention_kernel(
        &query.inner.driver,
        qk_function,
        &[
            grad_output_bf16.inner.ptr,
            value_bf16.inner.ptr,
            score_grad.inner.ptr,
        ],
        AttentionMmaLaunch {
            dims,
            head_dim,
            grid_x: dims.time.div_ceil(8),
            grid_y: dims.time.div_ceil(16),
            grid_z: dims.batch * dims.n_heads,
        },
    )?;
    launch_causal_attention_kernel(
        &query.inner.driver,
        softmax_backward_function,
        &[attention.inner.ptr, score_grad.inner.ptr, dscore.inner.ptr],
        dims,
        head_dim,
        attention_len,
    )?;
    let dscore_bf16 = f32_to_bf16_buffer(&dscore)?;
    launch_bf16_mma_attention_kernel(
        &query.inner.driver,
        av_function,
        &[
            dscore_bf16.inner.ptr,
            key_bf16.inner.ptr,
            grad_query.inner.ptr,
        ],
        AttentionMmaLaunch {
            dims,
            head_dim,
            grid_x: head_dim.div_ceil(8),
            grid_y: dims.time.div_ceil(16),
            grid_z: dims.batch * dims.n_heads,
        },
    )?;

    launch_causal_attention_kernel(
        &query.inner.driver,
        transpose_attention_function,
        &[attention.inner.ptr, attention_t.inner.ptr],
        dims,
        head_dim,
        attention_len,
    )?;
    let attention_t_bf16 = f32_to_bf16_buffer(&attention_t)?;
    launch_bf16_mma_attention_kernel(
        &query.inner.driver,
        av_function,
        &[
            attention_t_bf16.inner.ptr,
            grad_output_bf16.inner.ptr,
            grad_value.inner.ptr,
        ],
        AttentionMmaLaunch {
            dims,
            head_dim,
            grid_x: head_dim.div_ceil(8),
            grid_y: dims.time.div_ceil(16),
            grid_z: dims.batch * dims.n_heads,
        },
    )?;

    launch_causal_attention_kernel(
        &query.inner.driver,
        transpose_attention_function,
        &[dscore.inner.ptr, dscore_t.inner.ptr],
        dims,
        head_dim,
        attention_len,
    )?;
    let dscore_t_bf16 = f32_to_bf16_buffer(&dscore_t)?;
    launch_bf16_mma_attention_kernel(
        &query.inner.driver,
        av_function,
        &[
            dscore_t_bf16.inner.ptr,
            query_bf16.inner.ptr,
            grad_key.inner.ptr,
        ],
        AttentionMmaLaunch {
            dims,
            head_dim,
            grid_x: head_dim.div_ceil(8),
            grid_y: dims.time.div_ceil(16),
            grid_z: dims.batch * dims.n_heads,
        },
    )?;

    let matmul_calls = checked_mul(
        dims.batch,
        dims.n_heads,
        "causal_attention_bf16_tensor_core_backward_f32_buffers matmul calls",
    )?;
    let total_matmul_calls = checked_mul(
        matmul_calls,
        4,
        "causal_attention_bf16_tensor_core_backward_f32_buffers total matmul calls",
    )?;
    if tensor_core_attention_cp_async_enabled() {
        CUDA_TENSOR_CORE_ATTENTION_CP_ASYNC_REQUESTED_CALLS
            .fetch_add(total_matmul_calls, Ordering::Relaxed);
        CUDA_TENSOR_CORE_ATTENTION_CP_ASYNC_GLOBAL_FALLBACK_CALLS
            .fetch_add(total_matmul_calls, Ordering::Relaxed);
    } else if tensor_core_attention_ldmatrix_enabled() {
        CUDA_TENSOR_CORE_ATTENTION_LDMATRIX_REQUESTED_CALLS
            .fetch_add(total_matmul_calls, Ordering::Relaxed);
        CUDA_TENSOR_CORE_ATTENTION_LDMATRIX_GLOBAL_FALLBACK_CALLS
            .fetch_add(total_matmul_calls, Ordering::Relaxed);
    }
    BF16_TENSOR_CORE_ATTENTION_BACKWARD_CALLS.fetch_add(1, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_SCORE_GRAD_MATMUL_CALLS.fetch_add(matmul_calls, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_DQ_MATMUL_CALLS.fetch_add(matmul_calls, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_DK_MATMUL_CALLS.fetch_add(matmul_calls, Ordering::Relaxed);
    BF16_TENSOR_CORE_ATTENTION_DV_MATMUL_CALLS.fetch_add(matmul_calls, Ordering::Relaxed);
    BF16_TENSOR_CORE_MATMUL_CALLS.fetch_add(total_matmul_calls, Ordering::Relaxed);
    BF16_TENSOR_CORE_MATMUL_BACKWARD_CALLS.fetch_add(total_matmul_calls, Ordering::Relaxed);
    record_attention_tensor_core_edge_tiles(
        dims.time,
        head_dim,
        dims.time,
        matmul_calls,
        "Tensor Core attention score-grad backward",
    )?;
    record_attention_tensor_core_edge_tiles(
        dims.time,
        dims.time,
        head_dim,
        matmul_calls,
        "Tensor Core attention dQ backward",
    )?;
    record_attention_tensor_core_edge_tiles(
        dims.time,
        dims.time,
        head_dim,
        matmul_calls,
        "Tensor Core attention dV backward",
    )?;
    record_attention_tensor_core_edge_tiles(
        dims.time,
        dims.time,
        head_dim,
        matmul_calls,
        "Tensor Core attention dK backward",
    )?;
    Ok((grad_query, grad_key, grad_value))
}

pub fn causal_attention_backward_query_f32_buffers(
    query: &CudaBuffer,
    key: &CudaBuffer,
    value: &CudaBuffer,
    attention: &CudaBuffer,
    grad_output: &CudaBuffer,
    dims: CausalAttentionDims,
) -> CudaResult<CudaBuffer> {
    causal_attention_backward_f32_buffer(
        query,
        key,
        value,
        attention,
        grad_output,
        dims,
        (
            "heirloom_causal_attention_backward_query_f32",
            "causal_attention_backward_query_f32_buffers",
        ),
    )
}

pub fn causal_attention_backward_key_f32_buffers(
    query: &CudaBuffer,
    key: &CudaBuffer,
    value: &CudaBuffer,
    attention: &CudaBuffer,
    grad_output: &CudaBuffer,
    dims: CausalAttentionDims,
) -> CudaResult<CudaBuffer> {
    causal_attention_backward_f32_buffer(
        query,
        key,
        value,
        attention,
        grad_output,
        dims,
        (
            "heirloom_causal_attention_backward_key_f32",
            "causal_attention_backward_key_f32_buffers",
        ),
    )
}

pub fn causal_attention_backward_value_f32_buffers(
    attention: &CudaBuffer,
    grad_output: &CudaBuffer,
    dims: CausalAttentionDims,
) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(
        attention,
        "causal_attention_backward_value_f32_buffers attention",
    )?;
    ensure_f32_buffer(
        grad_output,
        "causal_attention_backward_value_f32_buffers grad_output",
    )?;
    ensure_same_device(
        attention,
        grad_output,
        "causal_attention_backward_value_f32_buffers",
    )?;
    let head_dim =
        validate_causal_attention_dims(dims, "causal_attention_backward_value_f32_buffers")?;
    let qkv_len = checked_mul(
        checked_mul(
            dims.batch,
            dims.time,
            "causal_attention_backward_value_f32_buffers batch*time",
        )?,
        dims.channels,
        "causal_attention_backward_value_f32_buffers qkv",
    )?;
    let attention_len = checked_mul(
        checked_mul(
            checked_mul(
                dims.batch,
                dims.n_heads,
                "causal_attention_backward_value_f32_buffers batch*heads",
            )?,
            dims.time,
            "causal_attention_backward_value_f32_buffers batch*heads*time",
        )?,
        dims.time,
        "causal_attention_backward_value_f32_buffers attention",
    )?;
    ensure_buffer_len(
        attention,
        attention_len,
        "causal_attention_backward_value_f32_buffers attention",
    )?;
    ensure_buffer_len(
        grad_output,
        qkv_len,
        "causal_attention_backward_value_f32_buffers grad_output",
    )?;
    let output = CudaBuffer::uninit_bytes(
        attention.device_ordinal(),
        qkv_len,
        std::mem::size_of::<f32>(),
    )?;

    attention
        .inner
        .driver
        .set_current(attention.inner.context)?;
    let module = load_module(&attention.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_causal_attention_backward_value_f32")?;
    launch_causal_attention_kernel(
        &attention.inner.driver,
        function,
        &[attention.inner.ptr, grad_output.inner.ptr, output.inner.ptr],
        dims,
        head_dim,
        qkv_len,
    )?;
    Ok(output)
}

pub fn sum_f32_buffer(input: &CudaBuffer) -> CudaResult<CudaBuffer> {
    reduce_f32_buffer(input, "heirloom_sum_f32", "sum_f32_buffer")
}

pub fn sum_squares_f32_buffer(input: &CudaBuffer) -> CudaResult<CudaBuffer> {
    reduce_f32_buffer(input, "heirloom_sum_squares_f32", "sum_squares_f32_buffer")
}

pub fn mean_f32_buffer(input: &CudaBuffer) -> CudaResult<CudaBuffer> {
    if input.is_empty() {
        return Err(CudaError::new(
            "mean_f32_buffer cannot reduce an empty buffer",
        ));
    }
    reduce_f32_buffer(input, "heirloom_mean_f32", "mean_f32_buffer")
}

pub fn fill_from_scalar_f32_buffer(
    scalar: &CudaBuffer,
    len: usize,
    scale: f32,
) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(scalar, "fill_from_scalar_f32_buffer scalar")?;
    if scalar.len() != 1 {
        return Err(CudaError::new(format!(
            "fill_from_scalar_f32_buffer expected scalar length 1, got {}",
            scalar.len()
        )));
    }
    let output =
        CudaBuffer::uninit_bytes(scalar.device_ordinal(), len, std::mem::size_of::<f32>())?;
    if len == 0 {
        return Ok(output);
    }

    scalar.inner.driver.set_current(scalar.inner.context)?;
    let module = load_module(&scalar.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_fill_from_scalar_f32")?;
    launch_scaled_vector_kernel(
        &scalar.inner.driver,
        function,
        &[scalar.inner.ptr, output.inner.ptr],
        scale,
        len,
    )?;
    Ok(output)
}

pub fn fill_constant_f32_buffer(
    device_ordinal: i32,
    len: usize,
    value: f32,
) -> CudaResult<CudaBuffer> {
    let output = CudaBuffer::uninit_bytes(device_ordinal, len, std::mem::size_of::<f32>())?;
    if len == 0 {
        return Ok(output);
    }

    output.inner.driver.set_current(output.inner.context)?;
    {
        let module = load_module(&output.inner.driver, KERNEL_PTX)?;
        let function = module.function("heirloom_fill_constant_f32")?;
        launch_scaled_vector_kernel(
            &output.inner.driver,
            function,
            &[output.inner.ptr],
            value,
            len,
        )?;
    }
    Ok(output)
}

pub fn sgd_update_f32_buffer(param: &CudaBuffer, grad: &CudaBuffer, lr: f32) -> CudaResult<()> {
    if lr <= 0.0 {
        return Err(CudaError::new(format!(
            "sgd_update_f32_buffer learning rate must be positive, got {lr}"
        )));
    }
    ensure_f32_buffer(param, "sgd_update_f32_buffer param")?;
    ensure_f32_buffer(grad, "sgd_update_f32_buffer grad")?;
    ensure_same_device_len(param, grad, "sgd_update_f32_buffer")?;
    if param.is_empty() {
        return Ok(());
    }

    param.inner.driver.set_current(param.inner.context)?;
    let module = load_module(&param.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_sgd_update_f32")?;
    launch_scaled_vector_kernel(
        &param.inner.driver,
        function,
        &[param.inner.ptr, grad.inner.ptr],
        lr,
        param.len(),
    )
}

pub fn adamw_update_f32_buffers(
    param: &CudaBuffer,
    grad: &CudaBuffer,
    m: &CudaBuffer,
    v: &CudaBuffer,
    params: AdamWParams,
) -> CudaResult<()> {
    ensure_adamw_params(params)?;
    ensure_f32_buffer(param, "adamw_update_f32_buffers param")?;
    ensure_f32_buffer(grad, "adamw_update_f32_buffers grad")?;
    ensure_f32_buffer(m, "adamw_update_f32_buffers m")?;
    ensure_f32_buffer(v, "adamw_update_f32_buffers v")?;
    ensure_same_device_len(param, grad, "adamw_update_f32_buffers param/grad")?;
    ensure_same_device_len(param, m, "adamw_update_f32_buffers param/m")?;
    ensure_same_device_len(param, v, "adamw_update_f32_buffers param/v")?;
    if param.is_empty() {
        return Ok(());
    }

    param.inner.driver.set_current(param.inner.context)?;
    let module = load_module(&param.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_adamw_update_f32")?;
    launch_adamw_kernel(
        &param.inner.driver,
        function,
        &[param.inner.ptr, grad.inner.ptr, m.inner.ptr, v.inner.ptr],
        params,
        param.len(),
    )
}

pub fn memory_sparse_adamw_rows_f32_i64_buffers(
    buffers: SparseAdamWRowsBuffers<'_>,
    dims: SparseAdamWRowsDims,
    params: AdamWParams,
) -> CudaResult<()> {
    ensure_adamw_params(params)?;
    validate_sparse_adamw_rows_buffers(buffers, dims, "memory_sparse_adamw_rows_f32_i64_buffers")?;
    if buffers.selected_rows.is_empty() {
        return Ok(());
    }

    let total = checked_mul(
        dims.selected_rows,
        dims.row_dim,
        "memory sparse AdamW selected row elements",
    )?;
    let status = CudaBuffer::from_u32(buffers.param.device_ordinal(), &[0])?;
    let row_mask_buffer = buffers.row_mask.unwrap_or(buffers.selected_rows);
    let has_row_mask = buffers.row_mask.is_some();
    buffers
        .param
        .inner
        .driver
        .set_current(buffers.param.inner.context)?;
    let module = load_module(&buffers.param.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_memory_sparse_adamw_rows_f32_i64")?;
    launch_sparse_adamw_rows_kernel(
        &buffers.param.inner.driver,
        function,
        &[
            buffers.param.inner.ptr,
            buffers.grad.inner.ptr,
            buffers.m.inner.ptr,
            buffers.v.inner.ptr,
            buffers.selected_rows.inner.ptr,
            status.inner.ptr,
            row_mask_buffer.inner.ptr,
        ],
        params,
        dims,
        total,
        has_row_mask,
    )?;
    ensure_status_zero(&status, "memory_sparse_adamw_rows_f32_i64_buffers")?;
    CUDA_MEMORY_SPARSE_ADAMW_ROWS_CALLS.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

pub fn memory_sparse_adamw_compact_rows_f32_i64_buffers(
    buffers: SparseAdamWCompactRowsBuffers<'_>,
    dims: SparseAdamWRowsDims,
    params: AdamWParams,
) -> CudaResult<()> {
    ensure_adamw_params(params)?;
    validate_sparse_adamw_compact_rows_buffers(
        buffers,
        dims,
        "memory_sparse_adamw_compact_rows_f32_i64_buffers",
    )?;
    if buffers.selected_rows.is_empty() {
        return Ok(());
    }

    let total = checked_mul(
        dims.selected_rows,
        dims.row_dim,
        "memory sparse AdamW compact selected row elements",
    )?;
    let status = CudaBuffer::from_u32(buffers.param.device_ordinal(), &[0])?;
    let row_mask_buffer = buffers.row_mask.unwrap_or(buffers.selected_rows);
    let has_row_mask = buffers.row_mask.is_some();
    buffers
        .param
        .inner
        .driver
        .set_current(buffers.param.inner.context)?;
    let module = load_module(&buffers.param.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_memory_sparse_adamw_compact_rows_f32_i64")?;
    launch_sparse_adamw_rows_kernel(
        &buffers.param.inner.driver,
        function,
        &[
            buffers.param.inner.ptr,
            buffers.compact_grad.inner.ptr,
            buffers.m.inner.ptr,
            buffers.v.inner.ptr,
            buffers.selected_rows.inner.ptr,
            status.inner.ptr,
            row_mask_buffer.inner.ptr,
        ],
        params,
        dims,
        total,
        has_row_mask,
    )?;
    ensure_status_zero(&status, "memory_sparse_adamw_compact_rows_f32_i64_buffers")?;
    CUDA_MEMORY_SPARSE_ADAMW_ROWS_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_MEMORY_SPARSE_ADAMW_COMPACT_ROWS_CALLS.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

pub fn concat_i64_buffers(inputs: &[CudaBuffer]) -> CudaResult<CudaBuffer> {
    let Some(first) = inputs.first() else {
        return Err(CudaError::new(
            "concat_i64_buffers requires at least one input buffer",
        ));
    };
    ensure_i64_buffer(first, "concat_i64_buffers input[0]")?;
    let device_ordinal = first.device_ordinal();
    let mut total = 0usize;
    for (index, input) in inputs.iter().enumerate() {
        ensure_i64_buffer(input, &format!("concat_i64_buffers input[{index}]"))?;
        if input.device_ordinal() != device_ordinal {
            return Err(CudaError::new(format!(
                "concat_i64_buffers expected all inputs on cuda:{device_ordinal}, got input[{index}] on cuda:{}",
                input.device_ordinal()
            )));
        }
        total = total
            .checked_add(input.len())
            .ok_or_else(|| CudaError::new("concat_i64_buffers total length overflow"))?;
    }
    let output = CudaBuffer::uninit_bytes(device_ordinal, total, std::mem::size_of::<i64>())?;
    if total == 0 {
        return Ok(output);
    }
    first.inner.driver.set_current(first.inner.context)?;
    let module = load_module(&first.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_copy_i64_with_offset")?;
    let mut offset = 0usize;
    for input in inputs {
        if input.is_empty() {
            continue;
        }
        launch_i64_copy_with_offset_kernel(
            &first.inner.driver,
            function,
            &[input.inner.ptr, output.inner.ptr],
            offset,
            input.len(),
        )?;
        offset += input.len();
    }
    Ok(output)
}

fn i64_prefix_buffer(input: &CudaBuffer, len: usize) -> CudaResult<CudaBuffer> {
    ensure_i64_buffer(input, "i64_prefix_buffer input")?;
    if len > input.len() {
        return Err(CudaError::new(format!(
            "i64_prefix_buffer requested len {len} larger than input length {}",
            input.len()
        )));
    }
    let output = CudaBuffer::uninit_bytes(input.device_ordinal(), len, std::mem::size_of::<i64>())?;
    if len == 0 {
        return Ok(output);
    }
    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_copy_i64_with_offset")?;
    launch_i64_copy_with_offset_kernel(
        &input.inner.driver,
        function,
        &[input.inner.ptr, output.inner.ptr],
        0,
        len,
    )?;
    Ok(output)
}

fn binary_f32_buffers(
    left: &CudaBuffer,
    right: &CudaBuffer,
    kernel_name: &str,
    op_name: &str,
) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(left, &format!("{op_name} left"))?;
    ensure_f32_buffer(right, &format!("{op_name} right"))?;
    ensure_same_device_len(left, right, op_name)?;
    let output = CudaBuffer::uninit_bytes(
        left.device_ordinal(),
        left.len(),
        std::mem::size_of::<f32>(),
    )?;
    if left.is_empty() {
        return Ok(output);
    }

    left.inner.driver.set_current(left.inner.context)?;
    let module = load_module(&left.inner.driver, KERNEL_PTX)?;
    let function = module.function(kernel_name)?;
    launch_vector_kernel(
        &left.inner.driver,
        function,
        &[left.inner.ptr, right.inner.ptr, output.inner.ptr],
        left.len(),
    )?;
    Ok(output)
}

fn reduce_f32_buffer(
    input: &CudaBuffer,
    kernel_name: &str,
    op_name: &str,
) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(input, &format!("{op_name} input"))?;
    if input.is_empty() {
        return CudaBuffer::from_f32(input.device_ordinal(), &[0.0]);
    }
    let output = CudaBuffer::uninit_bytes(input.device_ordinal(), 1, std::mem::size_of::<f32>())?;

    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function(kernel_name)?;
    launch_vector_kernel(
        &input.inner.driver,
        function,
        &[input.inner.ptr, output.inner.ptr],
        input.len(),
    )?;
    Ok(output)
}

pub fn relu_f32(device_ordinal: i32, input: &[f32]) -> CudaResult<Vec<f32>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let session = CudaSession::new(device_ordinal)?;
    let module = session.load_module(KERNEL_PTX)?;
    let function = module.function("heirloom_relu_f32")?;
    let input_device = DeviceBuffer::from_f32(&session.driver, input)?;
    let output_device = DeviceBuffer::uninit_f32(&session.driver, input.len())?;
    launch_vector_kernel(
        &session.driver,
        function,
        &[input_device.ptr, output_device.ptr],
        input.len(),
    )?;
    output_device.copy_to_f32()
}

pub fn relu_f32_buffer(input: &CudaBuffer) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(input, "relu_f32_buffer input")?;
    let output = CudaBuffer::uninit_bytes(
        input.device_ordinal(),
        input.len(),
        std::mem::size_of::<f32>(),
    )?;
    if input.is_empty() {
        return Ok(output);
    }

    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_relu_f32")?;
    launch_vector_kernel(
        &input.inner.driver,
        function,
        &[input.inner.ptr, output.inner.ptr],
        input.len(),
    )?;
    Ok(output)
}

pub fn relu_backward_f32_buffer(
    input: &CudaBuffer,
    grad_output: &CudaBuffer,
) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(input, "relu_backward_f32_buffer input")?;
    ensure_f32_buffer(grad_output, "relu_backward_f32_buffer grad_output")?;
    ensure_same_device_len(input, grad_output, "relu_backward_f32_buffer")?;
    let output = CudaBuffer::uninit_bytes(
        input.device_ordinal(),
        input.len(),
        std::mem::size_of::<f32>(),
    )?;
    if input.is_empty() {
        return Ok(output);
    }

    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_relu_backward_f32")?;
    launch_vector_kernel(
        &input.inner.driver,
        function,
        &[input.inner.ptr, grad_output.inner.ptr, output.inner.ptr],
        input.len(),
    )?;
    Ok(output)
}

pub fn gelu_f32_buffer(input: &CudaBuffer) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(input, "gelu_f32_buffer input")?;
    let output = CudaBuffer::uninit_bytes(
        input.device_ordinal(),
        input.len(),
        std::mem::size_of::<f32>(),
    )?;
    if input.is_empty() {
        return Ok(output);
    }

    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_gelu_f32")?;
    launch_vector_kernel(
        &input.inner.driver,
        function,
        &[input.inner.ptr, output.inner.ptr],
        input.len(),
    )?;
    Ok(output)
}

pub fn gelu_backward_f32_buffer(
    input: &CudaBuffer,
    grad_output: &CudaBuffer,
) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(input, "gelu_backward_f32_buffer input")?;
    ensure_f32_buffer(grad_output, "gelu_backward_f32_buffer grad_output")?;
    ensure_same_device_len(input, grad_output, "gelu_backward_f32_buffer")?;
    let output = CudaBuffer::uninit_bytes(
        input.device_ordinal(),
        input.len(),
        std::mem::size_of::<f32>(),
    )?;
    if input.is_empty() {
        return Ok(output);
    }

    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_gelu_backward_f32")?;
    launch_vector_kernel(
        &input.inner.driver,
        function,
        &[input.inner.ptr, grad_output.inner.ptr, output.inner.ptr],
        input.len(),
    )?;
    Ok(output)
}

pub fn transpose2d_f32_buffer(
    input: &CudaBuffer,
    rows: usize,
    cols: usize,
) -> CudaResult<CudaBuffer> {
    ensure_f32_buffer(input, "transpose2d_f32_buffer input")?;
    let len = checked_mul(rows, cols, "transpose2d_f32_buffer input")?;
    ensure_buffer_len(input, len, "transpose2d_f32_buffer input")?;
    let output = CudaBuffer::uninit_bytes(input.device_ordinal(), len, std::mem::size_of::<f32>())?;
    if len == 0 {
        return Ok(output);
    }

    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_transpose2d_f32")?;
    launch_bias_2d_kernel(
        &input.inner.driver,
        function,
        &[input.inner.ptr, output.inner.ptr],
        rows,
        cols,
        len,
    )?;
    Ok(output)
}

pub fn transpose2d_bf16_buffer(
    input: &CudaBuffer,
    rows: usize,
    cols: usize,
) -> CudaResult<CudaBuffer> {
    ensure_bf16_buffer(input, "transpose2d_bf16_buffer input")?;
    let len = checked_mul(rows, cols, "transpose2d_bf16_buffer input")?;
    ensure_buffer_len(input, len, "transpose2d_bf16_buffer input")?;
    let output = CudaBuffer::uninit_bytes(input.device_ordinal(), len, std::mem::size_of::<u16>())?;
    if len == 0 {
        return Ok(output);
    }

    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_transpose2d_bf16")?;
    launch_bias_2d_kernel(
        &input.inner.driver,
        function,
        &[input.inner.ptr, output.inner.ptr],
        rows,
        cols,
        len,
    )?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn transpose2d_pair_bf16_buffers(
    first: &CudaBuffer,
    first_rows: usize,
    first_cols: usize,
    second: &CudaBuffer,
    second_rows: usize,
    second_cols: usize,
) -> CudaResult<(CudaBuffer, CudaBuffer)> {
    ensure_bf16_buffer(first, "transpose2d_pair_bf16_buffers first")?;
    ensure_bf16_buffer(second, "transpose2d_pair_bf16_buffers second")?;
    ensure_same_device(first, second, "transpose2d_pair_bf16_buffers")?;
    let first_len = checked_mul(
        first_rows,
        first_cols,
        "transpose2d_pair_bf16_buffers first",
    )?;
    let second_len = checked_mul(
        second_rows,
        second_cols,
        "transpose2d_pair_bf16_buffers second",
    )?;
    ensure_buffer_len(first, first_len, "transpose2d_pair_bf16_buffers first")?;
    ensure_buffer_len(second, second_len, "transpose2d_pair_bf16_buffers second")?;
    let first_output = CudaBuffer::uninit_bytes(
        first.device_ordinal(),
        first_len,
        std::mem::size_of::<u16>(),
    )?;
    let second_output = CudaBuffer::uninit_bytes(
        second.device_ordinal(),
        second_len,
        std::mem::size_of::<u16>(),
    )?;
    if first_len == 0 && second_len == 0 {
        return Ok((first_output, second_output));
    }

    first.inner.driver.set_current(first.inner.context)?;
    let module = load_module(&first.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_transpose2d_pair_bf16")?;
    launch_transpose2d_pair_bf16_kernel(
        &first.inner.driver,
        function,
        &[
            first.inner.ptr,
            first_output.inner.ptr,
            second.inner.ptr,
            second_output.inner.ptr,
        ],
        first_rows,
        first_cols,
        first_len,
        second_rows,
        second_cols,
        second_len,
    )?;
    Ok((first_output, second_output))
}

pub fn materialize_matrix_layout_bf16_buffer(
    input: &CudaBuffer,
    layout: MatrixLayout,
) -> CudaResult<CudaBuffer> {
    ensure_bf16_buffer(input, "materialize_matrix_layout_bf16_buffer input")?;
    if layout.rows == 0 || layout.cols == 0 {
        return CudaBuffer::uninit_bytes(input.device_ordinal(), 0, std::mem::size_of::<u16>());
    }
    let row_span = checked_mul(
        layout.rows - 1,
        layout.row_stride,
        "materialize_matrix_layout_bf16_buffer row span",
    )?;
    let col_span = checked_mul(
        layout.cols - 1,
        layout.col_stride,
        "materialize_matrix_layout_bf16_buffer col span",
    )?;
    let max_index = layout
        .offset
        .checked_add(row_span)
        .and_then(|value| value.checked_add(col_span))
        .ok_or_else(|| {
            CudaError::new("materialize_matrix_layout_bf16_buffer layout overflow".to_string())
        })?;
    if max_index >= input.len() {
        return Err(CudaError::new(format!(
            "materialize_matrix_layout_bf16_buffer layout {:?} exceeds buffer length {}",
            layout,
            input.len()
        )));
    }
    let len = checked_mul(
        layout.rows,
        layout.cols,
        "materialize_matrix_layout_bf16_buffer output",
    )?;
    let output = CudaBuffer::uninit_bytes(input.device_ordinal(), len, std::mem::size_of::<u16>())?;

    input.inner.driver.set_current(input.inner.context)?;
    let module = load_module(&input.inner.driver, KERNEL_PTX)?;
    let function = module.function("heirloom_materialize_matrix_layout_bf16")?;
    launch_materialize_matrix_layout_kernel(
        &input.inner.driver,
        function,
        input.inner.ptr,
        output.inner.ptr,
        layout,
        len,
    )?;
    Ok(output)
}

pub fn smoke_f32(device_ordinal: i32, len: usize) -> CudaResult<CudaSmokeReport> {
    if len == 0 {
        return Err(CudaError::new("smoke_f32 requires len > 0"));
    }
    let left = (0..len)
        .map(|index| index as f32 * 0.25 - 8.0)
        .collect::<Vec<_>>();
    let right = (0..len)
        .map(|index| 4.0 - index as f32 * 0.125)
        .collect::<Vec<_>>();

    let added = add_f32(device_ordinal, &left, &right)?;
    let add_expected = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    let add_max_abs_error = max_abs_error(&added, &add_expected);

    let relu_output = relu_f32(device_ordinal, &added)?;
    let relu_expected = add_expected
        .iter()
        .map(|value| value.max(0.0))
        .collect::<Vec<_>>();
    let relu_max_abs_error = max_abs_error(&relu_output, &relu_expected);

    let info = system_info()?;
    let device = info
        .devices
        .into_iter()
        .find(|device| device.ordinal == device_ordinal)
        .ok_or_else(|| CudaError::new(format!("device ordinal {device_ordinal} not found")))?;
    Ok(CudaSmokeReport {
        device,
        len,
        add_max_abs_error,
        relu_max_abs_error,
    })
}

fn ensure_f32_buffer(buffer: &CudaBuffer, role: &str) -> CudaResult<()> {
    if buffer.element_size() != std::mem::size_of::<f32>() {
        return Err(CudaError::new(format!(
            "{role} expected f32 element size {}, got {}",
            std::mem::size_of::<f32>(),
            buffer.element_size()
        )));
    }
    Ok(())
}

fn ensure_bf16_buffer(buffer: &CudaBuffer, role: &str) -> CudaResult<()> {
    if buffer.element_size() != std::mem::size_of::<u16>() {
        return Err(CudaError::new(format!(
            "{role} expected bf16/u16 element size {}, got {}",
            std::mem::size_of::<u16>(),
            buffer.element_size()
        )));
    }
    Ok(())
}

fn ensure_i64_buffer(buffer: &CudaBuffer, role: &str) -> CudaResult<()> {
    if buffer.element_size() != std::mem::size_of::<i64>() {
        return Err(CudaError::new(format!(
            "{role} expected i64 element size {}, got {}",
            std::mem::size_of::<i64>(),
            buffer.element_size()
        )));
    }
    Ok(())
}

fn ensure_u8_buffer(buffer: &CudaBuffer, role: &str) -> CudaResult<()> {
    if buffer.element_size() != std::mem::size_of::<u8>() {
        return Err(CudaError::new(format!(
            "{role} expected u8 element size {}, got {}",
            std::mem::size_of::<u8>(),
            buffer.element_size()
        )));
    }
    Ok(())
}

fn ensure_positive_eps(eps: f32, op_name: &str) -> CudaResult<()> {
    if eps <= 0.0 || !eps.is_finite() {
        return Err(CudaError::new(format!(
            "{op_name} expected finite positive eps, got {eps}"
        )));
    }
    Ok(())
}

fn ensure_adamw_params(params: AdamWParams) -> CudaResult<()> {
    if params.lr <= 0.0 || !params.lr.is_finite() {
        return Err(CudaError::new(format!(
            "adamw_update_f32_buffers expected finite positive lr, got {}",
            params.lr
        )));
    }
    if !(0.0..1.0).contains(&params.beta1) || !params.beta1.is_finite() {
        return Err(CudaError::new(format!(
            "adamw_update_f32_buffers expected beta1 in [0, 1), got {}",
            params.beta1
        )));
    }
    if !(0.0..1.0).contains(&params.beta2) || !params.beta2.is_finite() {
        return Err(CudaError::new(format!(
            "adamw_update_f32_buffers expected beta2 in [0, 1), got {}",
            params.beta2
        )));
    }
    if params.eps <= 0.0 || !params.eps.is_finite() {
        return Err(CudaError::new(format!(
            "adamw_update_f32_buffers expected finite positive eps, got {}",
            params.eps
        )));
    }
    if params.weight_decay < 0.0 || !params.weight_decay.is_finite() {
        return Err(CudaError::new(format!(
            "adamw_update_f32_buffers expected finite non-negative weight_decay, got {}",
            params.weight_decay
        )));
    }
    if params.clip_scale < 0.0 || !params.clip_scale.is_finite() {
        return Err(CudaError::new(format!(
            "adamw_update_f32_buffers expected finite non-negative clip_scale, got {}",
            params.clip_scale
        )));
    }
    if params.bias_correction1 <= 0.0 || !params.bias_correction1.is_finite() {
        return Err(CudaError::new(format!(
            "adamw_update_f32_buffers expected finite positive bias_correction1, got {}",
            params.bias_correction1
        )));
    }
    if params.bias_correction2 <= 0.0 || !params.bias_correction2.is_finite() {
        return Err(CudaError::new(format!(
            "adamw_update_f32_buffers expected finite positive bias_correction2, got {}",
            params.bias_correction2
        )));
    }
    Ok(())
}

fn ensure_same_device_len(left: &CudaBuffer, right: &CudaBuffer, op_name: &str) -> CudaResult<()> {
    ensure_same_device(left, right, op_name)?;
    if left.len() != right.len() {
        return Err(CudaError::new(format!(
            "{op_name} expected equal buffer lengths, got {} and {}",
            left.len(),
            right.len()
        )));
    }
    Ok(())
}

fn ensure_same_device(left: &CudaBuffer, right: &CudaBuffer, op_name: &str) -> CudaResult<()> {
    if left.device_ordinal() != right.device_ordinal() {
        return Err(CudaError::new(format!(
            "{op_name} expected buffers on the same CUDA device, got cuda:{} and cuda:{}",
            left.device_ordinal(),
            right.device_ordinal()
        )));
    }
    Ok(())
}

fn ensure_buffer_len(buffer: &CudaBuffer, expected: usize, role: &str) -> CudaResult<()> {
    if buffer.len() != expected {
        return Err(CudaError::new(format!(
            "{role} expected buffer length {expected}, got {}",
            buffer.len()
        )));
    }
    Ok(())
}

fn validate_matrix_layout_buffer(
    buffer: &CudaBuffer,
    layout: MatrixLayout,
    role: &str,
) -> CudaResult<()> {
    ensure_f32_buffer(buffer, role)?;
    if layout.rows == 0 || layout.cols == 0 {
        return Ok(());
    }
    let row_span = checked_mul(
        layout.rows - 1,
        layout.row_stride,
        &format!("{role} row span"),
    )?;
    let col_span = checked_mul(
        layout.cols - 1,
        layout.col_stride,
        &format!("{role} col span"),
    )?;
    let max_index = layout
        .offset
        .checked_add(row_span)
        .and_then(|value| value.checked_add(col_span))
        .ok_or_else(|| CudaError::new(format!("{role} layout offset overflow")))?;
    if max_index >= buffer.len() {
        return Err(CudaError::new(format!(
            "{role} layout {:?} exceeds buffer length {}",
            layout,
            buffer.len()
        )));
    }
    Ok(())
}

fn ensure_matmul_dims(dims: MatmulStridedDims, op_name: &str) -> CudaResult<()> {
    if dims.left.cols != dims.right.rows {
        return Err(CudaError::new(format!(
            "{op_name} shape mismatch: left {:?} right {:?}",
            dims.left, dims.right
        )));
    }
    checked_mul(dims.left.rows, dims.left.cols, &format!("{op_name} left"))?;
    checked_mul(
        dims.right.rows,
        dims.right.cols,
        &format!("{op_name} right"),
    )?;
    checked_mul(
        dims.left.rows,
        dims.right.cols,
        &format!("{op_name} output"),
    )?;
    Ok(())
}

fn validate_matmul_strided_buffers(
    left: &CudaBuffer,
    right: &CudaBuffer,
    dims: MatmulStridedDims,
    op_name: &str,
) -> CudaResult<()> {
    validate_matrix_layout_buffer(left, dims.left, &format!("{op_name} left"))?;
    validate_matrix_layout_buffer(right, dims.right, &format!("{op_name} right"))?;
    ensure_same_device(left, right, op_name)?;
    ensure_matmul_dims(dims, op_name)
}

fn checked_mul(left: usize, right: usize, role: &str) -> CudaResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| CudaError::new(format!("{role} shape product overflow: {left} * {right}")))
}

fn product_key_side_cuda(slots: usize, op_name: &str) -> CudaResult<usize> {
    let side = (slots as f64).sqrt() as usize;
    if side == 0 || side * side != slots {
        return Err(CudaError::new(format!(
            "{op_name} requires square slots for product-key lookup, got {slots}"
        )));
    }
    Ok(side)
}

fn round_up_to_multiple(value: usize, multiple: usize, role: &str) -> CudaResult<usize> {
    if value == 0 || multiple == 0 {
        return Err(CudaError::new(format!(
            "{role} requires non-zero value and multiple, got value={value} multiple={multiple}"
        )));
    }
    value
        .checked_add(multiple - 1)
        .map(|rounded| (rounded / multiple) * multiple)
        .ok_or_else(|| {
            CudaError::new(format!(
                "{role} overflow while rounding {value} up to multiple {multiple}"
            ))
        })
}

fn record_tensor_core_padding(
    m: usize,
    k: usize,
    n: usize,
    padded_m: usize,
    padded_k: usize,
    padded_n: usize,
) -> CudaResult<()> {
    let padded_tiles = checked_mul(padded_m / 16, padded_n / 8, "Tensor Core padded tile count")?;
    let fully_interior_tiles = checked_mul(m / 16, n / 8, "Tensor Core fully interior tile count")?;
    let edge_tiles = if padded_k != k {
        padded_tiles
    } else {
        padded_tiles.saturating_sub(fully_interior_tiles)
    };
    CUDA_TENSOR_CORE_PADDED_TILES.fetch_add(padded_tiles, Ordering::Relaxed);
    CUDA_TENSOR_CORE_REMAINDER_TILES.fetch_add(edge_tiles, Ordering::Relaxed);
    Ok(())
}

fn record_attention_tensor_core_edge_tiles(
    m: usize,
    k: usize,
    n: usize,
    calls: usize,
    role: &str,
) -> CudaResult<()> {
    let padded_m = round_up_to_multiple(m, 16, role)?;
    let padded_k = round_up_to_multiple(k, 16, role)?;
    let padded_n = round_up_to_multiple(n, 8, role)?;
    if padded_m == m && padded_k == k && padded_n == n {
        return Ok(());
    }
    let padded_tiles = checked_mul(
        checked_mul(
            padded_m / 16,
            padded_n / 8,
            &format!("{role} padded attention tile count"),
        )?,
        calls,
        &format!("{role} padded attention tile call count"),
    )?;
    let fully_interior_tiles = checked_mul(
        checked_mul(
            m / 16,
            n / 8,
            &format!("{role} fully interior attention tile count"),
        )?,
        calls,
        &format!("{role} fully interior attention tile call count"),
    )?;
    let edge_tiles = if padded_k != k {
        padded_tiles
    } else {
        padded_tiles.saturating_sub(fully_interior_tiles)
    };
    CUDA_TENSOR_CORE_ATTENTION_PADDED_TILES.fetch_add(padded_tiles, Ordering::Relaxed);
    CUDA_TENSOR_CORE_ATTENTION_REMAINDER_TILES.fetch_add(edge_tiles, Ordering::Relaxed);
    Ok(())
}

fn ensure_status_zero(status: &CudaBuffer, op_name: &str) -> CudaResult<()> {
    let status = status.to_u32()?;
    if status.first().copied().unwrap_or(1) == 0 {
        Ok(())
    } else {
        Err(CudaError::new(format!(
            "{op_name} found an out-of-range embedding index on device"
        )))
    }
}

fn validate_causal_attention_dims(dims: CausalAttentionDims, op_name: &str) -> CudaResult<usize> {
    if dims.batch == 0 || dims.time == 0 || dims.channels == 0 {
        return Err(CudaError::new(format!(
            "{op_name} requires non-empty batch/time/channels, got batch={} time={} channels={}",
            dims.batch, dims.time, dims.channels
        )));
    }
    if dims.n_heads == 0 {
        return Err(CudaError::new(format!("{op_name} requires n_heads > 0")));
    }
    if !dims.channels.is_multiple_of(dims.n_heads) {
        return Err(CudaError::new(format!(
            "{op_name} requires channels divisible by n_heads, got channels={} n_heads={}",
            dims.channels, dims.n_heads
        )));
    }
    Ok(dims.channels / dims.n_heads)
}

fn validate_causal_attention_buffers(
    query: &CudaBuffer,
    key: &CudaBuffer,
    value: &CudaBuffer,
    dims: CausalAttentionDims,
    op_name: &str,
) -> CudaResult<usize> {
    ensure_f32_buffer(query, &format!("{op_name} query"))?;
    ensure_f32_buffer(key, &format!("{op_name} key"))?;
    ensure_f32_buffer(value, &format!("{op_name} value"))?;
    ensure_same_device(query, key, op_name)?;
    ensure_same_device(query, value, op_name)?;
    let head_dim = validate_causal_attention_dims(dims, op_name)?;
    let expected = checked_mul(
        checked_mul(dims.batch, dims.time, &format!("{op_name} batch*time"))?,
        dims.channels,
        &format!("{op_name} qkv"),
    )?;
    ensure_buffer_len(query, expected, &format!("{op_name} query"))?;
    ensure_buffer_len(key, expected, &format!("{op_name} key"))?;
    ensure_buffer_len(value, expected, &format!("{op_name} value"))?;
    Ok(head_dim)
}

fn causal_attention_backward_f32_buffer(
    query: &CudaBuffer,
    key: &CudaBuffer,
    value: &CudaBuffer,
    attention: &CudaBuffer,
    grad_output: &CudaBuffer,
    dims: CausalAttentionDims,
    kernel: (&str, &str),
) -> CudaResult<CudaBuffer> {
    let (kernel_name, op_name) = kernel;
    let head_dim = validate_causal_attention_buffers(query, key, value, dims, op_name)?;
    ensure_f32_buffer(attention, &format!("{op_name} attention"))?;
    ensure_f32_buffer(grad_output, &format!("{op_name} grad_output"))?;
    ensure_same_device(query, attention, op_name)?;
    ensure_same_device(query, grad_output, op_name)?;
    let qkv_len = checked_mul(
        checked_mul(dims.batch, dims.time, &format!("{op_name} batch*time"))?,
        dims.channels,
        &format!("{op_name} qkv"),
    )?;
    let attention_len = checked_mul(
        checked_mul(
            checked_mul(dims.batch, dims.n_heads, &format!("{op_name} batch*heads"))?,
            dims.time,
            &format!("{op_name} batch*heads*time"),
        )?,
        dims.time,
        &format!("{op_name} attention"),
    )?;
    ensure_buffer_len(attention, attention_len, &format!("{op_name} attention"))?;
    ensure_buffer_len(grad_output, qkv_len, &format!("{op_name} grad_output"))?;
    let output =
        CudaBuffer::uninit_bytes(query.device_ordinal(), qkv_len, std::mem::size_of::<f32>())?;

    query.inner.driver.set_current(query.inner.context)?;
    let module = load_module(&query.inner.driver, KERNEL_PTX)?;
    let function = module.function(kernel_name)?;
    launch_causal_attention_kernel(
        &query.inner.driver,
        function,
        &[
            query.inner.ptr,
            key.inner.ptr,
            value.inner.ptr,
            attention.inner.ptr,
            grad_output.inner.ptr,
            output.inner.ptr,
        ],
        dims,
        head_dim,
        qkv_len,
    )?;
    Ok(output)
}

fn max_abs_error(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f32::max)
}

struct CudaDriver {
    handle: *mut c_void,
    cu_init: CuInit,
    cu_device_get_count: CuDeviceGetCount,
    cu_device_get: CuDeviceGet,
    cu_device_get_name: CuDeviceGetName,
    cu_device_get_pci_bus_id: CuDeviceGetPciBusId,
    cu_device_get_attribute: CuDeviceGetAttribute,
    cu_device_can_access_peer: CuDeviceCanAccessPeer,
    cu_device_primary_ctx_retain: CuDevicePrimaryCtxRetain,
    cu_device_primary_ctx_release: CuDevicePrimaryCtxRelease,
    cu_ctx_create: CuCtxCreate,
    cu_ctx_set_current: CuCtxSetCurrent,
    cu_ctx_destroy: CuCtxDestroy,
    cu_ctx_get_current: CuCtxGetCurrent,
    cu_stream_create: CuStreamCreate,
    cu_stream_destroy: CuStreamDestroy,
    cu_stream_synchronize: CuStreamSynchronize,
    cu_event_create: CuEventCreate,
    cu_event_record: CuEventRecord,
    cu_event_query: CuEventQuery,
    cu_event_elapsed_time: CuEventElapsedTime,
    cu_event_destroy: CuEventDestroy,
    cu_mem_alloc: CuMemAlloc,
    cu_mem_free: CuMemFree,
    cu_memcpy_htod: CuMemcpyHtoD,
    cu_memcpy_dtoh: CuMemcpyDtoH,
    cu_module_load_data_ex: CuModuleLoadDataEx,
    cu_module_unload: CuModuleUnload,
    cu_module_get_function: CuModuleGetFunction,
    cu_launch_kernel: CuLaunchKernel,
}

impl CudaDriver {
    fn load() -> CudaResult<Self> {
        #[cfg(unix)]
        {
            let mut last_error = None;
            for library in ["libcuda.so.1", "libcuda.so"] {
                let library = CString::new(library).expect("static library name");
                // SAFETY: dlopen is called with a valid C string and RTLD_NOW.
                let handle = unsafe { dlopen(library.as_ptr(), RTLD_NOW) };
                if handle.is_null() {
                    last_error = Some(dl_error_string());
                    continue;
                }
                // SAFETY: symbol lookups are checked and immediately stored as typed function pointers.
                return unsafe { Self::from_handle(handle) };
            }
            Err(CudaError::new(format!(
                "CUDA driver library not found: {}",
                last_error.unwrap_or_else(|| "no dlopen error".to_string())
            )))
        }
        #[cfg(not(unix))]
        {
            Err(CudaError::new(
                "CUDA dynamic loading is currently implemented for Unix targets only",
            ))
        }
    }

    unsafe fn from_handle(handle: *mut c_void) -> CudaResult<Self> {
        Ok(Self {
            handle,
            cu_init: load_symbol(handle, "cuInit")?,
            cu_device_get_count: load_symbol(handle, "cuDeviceGetCount")?,
            cu_device_get: load_symbol(handle, "cuDeviceGet")?,
            cu_device_get_name: load_symbol(handle, "cuDeviceGetName")?,
            cu_device_get_pci_bus_id: load_symbol(handle, "cuDeviceGetPCIBusId")?,
            cu_device_get_attribute: load_symbol(handle, "cuDeviceGetAttribute")?,
            cu_device_can_access_peer: load_symbol(handle, "cuDeviceCanAccessPeer")?,
            cu_device_primary_ctx_retain: load_symbol(handle, "cuDevicePrimaryCtxRetain")?,
            cu_device_primary_ctx_release: load_symbol(handle, "cuDevicePrimaryCtxRelease")?,
            cu_ctx_create: load_symbol(handle, "cuCtxCreate_v2")?,
            cu_ctx_set_current: load_symbol(handle, "cuCtxSetCurrent")?,
            cu_ctx_destroy: load_symbol(handle, "cuCtxDestroy_v2")?,
            cu_ctx_get_current: load_symbol(handle, "cuCtxGetCurrent")?,
            cu_stream_create: load_symbol(handle, "cuStreamCreate")?,
            cu_stream_destroy: load_symbol(handle, "cuStreamDestroy_v2")?,
            cu_stream_synchronize: load_symbol(handle, "cuStreamSynchronize")?,
            cu_event_create: load_symbol(handle, "cuEventCreate")?,
            cu_event_record: load_symbol(handle, "cuEventRecord")?,
            cu_event_query: load_symbol(handle, "cuEventQuery")?,
            cu_event_elapsed_time: load_symbol(handle, "cuEventElapsedTime")?,
            cu_event_destroy: load_symbol(handle, "cuEventDestroy_v2")?,
            cu_mem_alloc: load_symbol(handle, "cuMemAlloc_v2")?,
            cu_mem_free: load_symbol(handle, "cuMemFree_v2")?,
            cu_memcpy_htod: load_symbol(handle, "cuMemcpyHtoD_v2")?,
            cu_memcpy_dtoh: load_symbol(handle, "cuMemcpyDtoH_v2")?,
            cu_module_load_data_ex: load_symbol(handle, "cuModuleLoadDataEx")?,
            cu_module_unload: load_symbol(handle, "cuModuleUnload")?,
            cu_module_get_function: load_symbol(handle, "cuModuleGetFunction")?,
            cu_launch_kernel: load_symbol(handle, "cuLaunchKernel")?,
        })
    }

    fn init(&self) -> CudaResult<()> {
        // SAFETY: function pointer is loaded from libcuda and accepts flag 0.
        check(unsafe { (self.cu_init)(0) }, "cuInit")
    }

    fn device_count(&self) -> CudaResult<i32> {
        let mut count = 0;
        // SAFETY: count points to valid writable memory.
        check(
            unsafe { (self.cu_device_get_count)(&mut count) },
            "cuDeviceGetCount",
        )?;
        Ok(count)
    }

    fn device_info(&self, ordinal: i32) -> CudaResult<CudaDeviceInfo> {
        let device = self.device(ordinal)?;
        let mut name = [0i8; 128];
        // SAFETY: name is a valid writable buffer and device is returned by cuDeviceGet.
        check(
            unsafe { (self.cu_device_get_name)(name.as_mut_ptr(), name.len() as c_int, device) },
            "cuDeviceGetName",
        )?;
        // SAFETY: CUDA writes a null-terminated device name into the provided buffer.
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let mut pci_bus_id = [0i8; 32];
        // SAFETY: pci_bus_id is a valid writable buffer and device is returned by cuDeviceGet.
        check(
            unsafe {
                (self.cu_device_get_pci_bus_id)(
                    pci_bus_id.as_mut_ptr(),
                    pci_bus_id.len() as c_int,
                    device,
                )
            },
            "cuDeviceGetPCIBusId",
        )?;
        // SAFETY: CUDA writes a null-terminated PCI bus id into the provided buffer.
        let pci_bus_id = unsafe { CStr::from_ptr(pci_bus_id.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let compute_capability_major =
            self.device_attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let compute_capability_minor =
            self.device_attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        Ok(CudaDeviceInfo {
            ordinal,
            name,
            pci_bus_id,
            compute_capability_major,
            compute_capability_minor,
        })
    }

    fn device(&self, ordinal: i32) -> CudaResult<CUdevice> {
        let mut device = 0;
        // SAFETY: device points to valid writable memory.
        check(
            unsafe { (self.cu_device_get)(&mut device, ordinal) },
            "cuDeviceGet",
        )?;
        Ok(device)
    }

    fn set_current(&self, context: CUcontext) -> CudaResult<()> {
        // SAFETY: context is either null or a context returned by cuCtxCreate_v2.
        check(
            unsafe { (self.cu_ctx_set_current)(context) },
            "cuCtxSetCurrent",
        )
    }

    fn current_context(&self) -> CudaResult<CUcontext> {
        let mut context = ptr::null_mut();
        // SAFETY: context points to valid writable memory.
        check(
            unsafe { (self.cu_ctx_get_current)(&mut context) },
            "cuCtxGetCurrent",
        )?;
        if context.is_null() {
            return Err(CudaError::new(
                "no current CUDA context is active for stream selection",
            ));
        }
        Ok(context)
    }

    fn device_attribute(&self, device: CUdevice, attribute: c_int) -> CudaResult<i32> {
        let mut value = 0;
        // SAFETY: value is writable and device was returned by cuDeviceGet.
        check(
            unsafe { (self.cu_device_get_attribute)(&mut value, attribute, device) },
            "cuDeviceGetAttribute",
        )?;
        Ok(value)
    }
}

impl Drop for CudaDriver {
    fn drop(&mut self) {
        #[cfg(unix)]
        if !self.handle.is_null() {
            // SAFETY: handle was returned by dlopen and is closed at most once here.
            unsafe {
                let _ = dlclose(self.handle);
            }
        }
    }
}

thread_local! {
    static CUDA_STREAM_CACHE: RefCell<HashMap<usize, CachedStream>> = RefCell::new(HashMap::new());
    static CUDA_ALLOCATION_CACHE: RefCell<CudaAllocatorCache> = RefCell::new(CudaAllocatorCache::default());
    static CUDA_MODULE_CACHE: RefCell<HashMap<(usize, &'static str), CUmodule>> = RefCell::new(HashMap::new());
}

struct CachedStream {
    driver: CudaDriver,
    context: CUcontext,
    stream: CUstream,
}

impl Drop for CachedStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            let _ = self.driver.set_current(self.context);
            // SAFETY: stream was created by cuStreamCreate and is destroyed at most once here.
            unsafe {
                let _ = (self.driver.cu_stream_destroy)(self.stream);
            }
            self.stream = ptr::null_mut();
        }
    }
}

#[derive(Default)]
struct CudaAllocatorCache {
    free: HashMap<i32, Vec<CachedBlock>>,
    pending: HashMap<i32, Vec<CachedBlock>>,
}

struct CachedBlock {
    driver: Option<CudaDriver>,
    device: CUdevice,
    context: CUcontext,
    ptr: CUdeviceptr,
    capacity_bytes: usize,
    device_ordinal: i32,
    event: Option<CUevent>,
}

impl CachedBlock {
    fn from_allocation(allocation: &mut CudaAllocation, event: Option<CUevent>) -> Option<Self> {
        let driver = allocation.driver.take()?;
        let block = Self {
            driver: Some(driver),
            device: allocation.device,
            context: allocation.context,
            ptr: allocation.ptr,
            capacity_bytes: allocation.capacity_bytes,
            device_ordinal: allocation.device_ordinal,
            event,
        };
        allocation.ptr = 0;
        allocation.context = ptr::null_mut();
        Some(block)
    }

    fn into_allocation(mut self, requested_bytes: usize) -> CudaAllocation {
        let driver = self
            .driver
            .take()
            .expect("cached CUDA block driver missing during reuse");
        let allocation = CudaAllocation {
            driver: CudaDriverHandle::new(driver),
            device: self.device,
            context: self.context,
            ptr: self.ptr,
            bytes: requested_bytes,
            capacity_bytes: self.capacity_bytes,
            device_ordinal: self.device_ordinal,
        };
        self.ptr = 0;
        self.context = ptr::null_mut();
        allocation
    }

    fn event_ready(&self) -> bool {
        let Some(driver) = &self.driver else {
            return false;
        };
        let Some(event) = self.event else {
            return true;
        };
        let _ = driver.set_current(self.context);
        CUDA_EVENT_QUERY_CALLS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: event was created by cuEventCreate and recorded on this context.
        match unsafe { (driver.cu_event_query)(event) } {
            CUDA_SUCCESS => true,
            CUDA_ERROR_NOT_READY => false,
            _ => false,
        }
    }

    fn destroy_event(&mut self) {
        let Some(event) = self.event.take() else {
            return;
        };
        if let Some(driver) = &self.driver {
            let _ = driver.set_current(self.context);
            // SAFETY: event was created by cuEventCreate and is destroyed at most once here.
            unsafe {
                let _ = (driver.cu_event_destroy)(event);
            }
        }
    }
}

impl Drop for CachedBlock {
    fn drop(&mut self) {
        self.destroy_event();
        let Some(driver) = &self.driver else {
            return;
        };
        if !self.context.is_null() {
            let _ = driver.set_current(self.context);
        }
        if self.ptr != 0 {
            // SAFETY: ptr was allocated by cuMemAlloc_v2 and is freed once here.
            unsafe {
                let _ = (driver.cu_mem_free)(self.ptr);
            }
            CUDA_ALLOC_FREES.fetch_add(1, Ordering::Relaxed);
            CUDA_ALLOC_RESERVED_BYTES.fetch_sub(self.capacity_bytes, Ordering::Relaxed);
            self.ptr = 0;
        }
        if !self.context.is_null() {
            // SAFETY: primary context was retained for this block and is released once here.
            unsafe {
                let _ = (driver.cu_device_primary_ctx_release)(self.device);
            }
            self.context = ptr::null_mut();
        }
    }
}

fn compute_stream_for_current_context(driver: &CudaDriver) -> CudaResult<CUstream> {
    let context = driver.current_context()?;
    let key = context as usize;
    CUDA_STREAM_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(stream) = cache.get(&key) {
            return Ok(stream.stream);
        }
        let stream_driver = CudaDriver::load()?;
        stream_driver.init()?;
        stream_driver.set_current(context)?;
        let mut stream = ptr::null_mut();
        // SAFETY: stream points to writable memory and the context is current.
        check(
            unsafe { (stream_driver.cu_stream_create)(&mut stream, 0) },
            "cuStreamCreate",
        )?;
        CUDA_STREAM_CREATE_CALLS.fetch_add(1, Ordering::Relaxed);
        cache.insert(
            key,
            CachedStream {
                driver: stream_driver,
                context,
                stream,
            },
        );
        Ok(stream)
    })
}

fn create_timing_event(driver: &CudaDriver) -> CudaResult<CUevent> {
    let mut event = ptr::null_mut();
    // SAFETY: event points to writable memory and timing events use flags=0.
    check(
        unsafe { (driver.cu_event_create)(&mut event, 0) },
        "cuEventCreate",
    )?;
    CUDA_EVENT_CREATE_CALLS.fetch_add(1, Ordering::Relaxed);
    Ok(event)
}

fn destroy_event(driver: &CudaDriver, context: CUcontext, event: CUevent) {
    if event.is_null() {
        return;
    }
    if !context.is_null() {
        let _ = driver.set_current(context);
    }
    // SAFETY: event was created by cuEventCreate and is destroyed at most once here.
    unsafe {
        let _ = (driver.cu_event_destroy)(event);
    }
}

fn synchronize_current_compute_stream(driver: &CudaDriver, reason: &str) -> CudaResult<()> {
    let stream = compute_stream_for_current_context(driver)?;
    CUDA_HOST_SYNC_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_STREAM_SYNC_CALLS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: stream belongs to the current context and is a live cached stream.
    check(unsafe { (driver.cu_stream_synchronize)(stream) }, reason)
}

fn evict_stream_for_context(context: CUcontext) {
    let key = context as usize;
    CUDA_STREAM_CACHE.with(|cache| {
        cache.borrow_mut().remove(&key);
    });
}

fn record_kernel_launch(label: &'static str, elements: usize) {
    CUDA_KERNEL_LAUNCH_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_KERNEL_LAUNCH_ELEMENTS.fetch_add(elements, Ordering::Relaxed);
    let mut stats = kernel_launch_family_stats()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let entry = stats.entry(label).or_default();
    entry.calls = entry.calls.saturating_add(1);
    entry.elements = entry.elements.saturating_add(elements);
}

fn allocation_from_cache(device_ordinal: i32, bytes: usize) -> Option<CudaAllocation> {
    if allocator_cache_disabled() {
        return None;
    }
    CUDA_ALLOCATION_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        drain_pending_blocks(&mut cache, device_ordinal);
        let free = cache.free.get_mut(&device_ordinal)?;
        let index = free
            .iter()
            .position(|block| block.capacity_bytes >= bytes)?;
        let mut block = free.swap_remove(index);
        block.destroy_event();
        CUDA_ALLOC_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
        CUDA_ALLOC_ACTIVE_BYTES.fetch_add(bytes, Ordering::Relaxed);
        update_allocation_high_water();
        Some(block.into_allocation(bytes))
    })
}

fn drain_pending_blocks(cache: &mut CudaAllocatorCache, device_ordinal: i32) {
    let Some(pending) = cache.pending.get_mut(&device_ordinal) else {
        return;
    };
    let mut index = 0;
    while index < pending.len() {
        if pending[index].event_ready() {
            let mut block = pending.swap_remove(index);
            block.destroy_event();
            CUDA_ALLOC_PENDING_RECLAIMS.fetch_add(1, Ordering::Relaxed);
            cache.free.entry(device_ordinal).or_default().push(block);
        } else {
            index += 1;
        }
    }
}

fn allocate_cuda_allocation(device_ordinal: i32, bytes: usize) -> CudaResult<CudaAllocation> {
    if bytes > 0 {
        if let Some(allocation) = allocation_from_cache(device_ordinal, bytes) {
            return Ok(allocation);
        }
    }
    let (driver, device, context) = create_primary_context(device_ordinal)?;
    if bytes == 0 {
        return Ok(CudaAllocation {
            driver: CudaDriverHandle::new(driver),
            device,
            context,
            ptr: 0,
            bytes,
            capacity_bytes: 0,
            device_ordinal,
        });
    }
    let mut ptr = 0;
    // SAFETY: ptr points to valid writable memory for the allocated device pointer.
    check(
        unsafe { (driver.cu_mem_alloc)(&mut ptr, bytes) },
        "cuMemAlloc_v2",
    )?;
    CUDA_ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_ALLOC_ACTIVE_BYTES.fetch_add(bytes, Ordering::Relaxed);
    CUDA_ALLOC_RESERVED_BYTES.fetch_add(bytes, Ordering::Relaxed);
    update_allocation_high_water();
    Ok(CudaAllocation {
        driver: CudaDriverHandle::new(driver),
        device,
        context,
        ptr,
        bytes,
        capacity_bytes: bytes,
        device_ordinal,
    })
}

fn cache_or_free_allocation(allocation: &mut CudaAllocation) {
    let Some(driver) = allocation.driver.as_ref() else {
        return;
    };
    if !allocation.context.is_null() {
        let _ = driver.set_current(allocation.context);
    }
    if allocation.ptr == 0 {
        release_allocation_context(allocation);
        return;
    }
    CUDA_ALLOC_ACTIVE_BYTES.fetch_sub(allocation.bytes, Ordering::Relaxed);
    if allocator_cache_disabled() {
        free_allocation_now(allocation);
        return;
    }
    let event = match compute_stream_for_current_context(driver) {
        Ok(stream) => {
            let mut event = ptr::null_mut();
            // SAFETY: event points to writable memory and the context is current.
            let created = unsafe { (driver.cu_event_create)(&mut event, CU_EVENT_DISABLE_TIMING) };
            if created == CUDA_SUCCESS {
                CUDA_EVENT_CREATE_CALLS.fetch_add(1, Ordering::Relaxed);
                // SAFETY: event and stream are live objects in the current context.
                let recorded = unsafe { (driver.cu_event_record)(event, stream) };
                if recorded == CUDA_SUCCESS {
                    CUDA_EVENT_RECORD_CALLS.fetch_add(1, Ordering::Relaxed);
                    Some(event)
                } else {
                    // SAFETY: event was created above and has not been destroyed.
                    unsafe {
                        let _ = (driver.cu_event_destroy)(event);
                    }
                    None
                }
            } else {
                None
            }
        }
        Err(_) => None,
    };
    if let Some(block) = CachedBlock::from_allocation(allocation, event) {
        CUDA_ALLOC_DEFERRED_FREES.fetch_add(1, Ordering::Relaxed);
        CUDA_ALLOCATION_CACHE.with(|cache| {
            cache
                .borrow_mut()
                .pending
                .entry(block.device_ordinal)
                .or_default()
                .push(block);
        });
    }
}

fn free_allocation_now(allocation: &mut CudaAllocation) {
    let Some(driver) = allocation.driver.as_ref() else {
        return;
    };
    if allocation.ptr != 0 {
        // SAFETY: ptr was allocated by cuMemAlloc_v2 and is freed once here.
        unsafe {
            let _ = (driver.cu_mem_free)(allocation.ptr);
        }
        CUDA_ALLOC_FREES.fetch_add(1, Ordering::Relaxed);
        CUDA_ALLOC_RESERVED_BYTES.fetch_sub(allocation.capacity_bytes, Ordering::Relaxed);
        allocation.ptr = 0;
    }
    release_allocation_context(allocation);
}

fn release_allocation_context(allocation: &mut CudaAllocation) {
    let Some(driver) = allocation.driver.as_ref() else {
        return;
    };
    if !allocation.context.is_null() {
        // SAFETY: primary context was retained for this allocation and is released once here.
        unsafe {
            let _ = (driver.cu_device_primary_ctx_release)(allocation.device);
        }
        allocation.context = ptr::null_mut();
    }
    let _ = allocation.driver.take();
}

struct NcclLibrary {
    handle: *mut c_void,
    nccl_get_version: NcclGetVersion,
    nccl_get_error_string: NcclGetErrorString,
    nccl_get_unique_id: NcclGetUniqueId,
    nccl_comm_init_rank: NcclCommInitRank,
    nccl_comm_destroy: NcclCommDestroy,
    nccl_comm_abort: NcclCommAbort,
    nccl_all_reduce: NcclAllReduce,
}

impl NcclLibrary {
    fn load() -> CudaResult<Self> {
        #[cfg(unix)]
        {
            let mut last_error = None;
            for library in ["libnccl.so.2", "libnccl.so"] {
                let library = CString::new(library).expect("static library name");
                // SAFETY: dlopen is called with a valid C string. NCCL is loaded globally on
                // Linux so NCCL network/coll plugins can resolve NCCL symbols during init.
                let handle = unsafe { dlopen(library.as_ptr(), NCCL_DLOPEN_FLAGS) };
                if handle.is_null() {
                    last_error = Some(dl_error_string());
                    continue;
                }
                // SAFETY: symbol lookups are checked and immediately stored as typed function pointers.
                return unsafe { Self::from_handle(handle) };
            }
            Err(CudaError::new(format!(
                "NCCL library not found: {}",
                last_error.unwrap_or_else(|| "no dlopen error".to_string())
            )))
        }
        #[cfg(not(unix))]
        {
            Err(CudaError::new(
                "NCCL dynamic loading is currently implemented for Unix targets only",
            ))
        }
    }

    unsafe fn from_handle(handle: *mut c_void) -> CudaResult<Self> {
        Ok(Self {
            handle,
            nccl_get_version: load_symbol(handle, "ncclGetVersion")?,
            nccl_get_error_string: load_symbol(handle, "ncclGetErrorString")?,
            nccl_get_unique_id: load_symbol(handle, "ncclGetUniqueId")?,
            nccl_comm_init_rank: load_symbol(handle, "ncclCommInitRank")?,
            nccl_comm_destroy: load_symbol(handle, "ncclCommDestroy")?,
            nccl_comm_abort: load_symbol(handle, "ncclCommAbort")?,
            nccl_all_reduce: load_symbol(handle, "ncclAllReduce")?,
        })
    }

    fn version(&self) -> CudaResult<i32> {
        let mut version = 0;
        // SAFETY: version points to valid writable memory.
        check_nccl(
            unsafe { (self.nccl_get_version)(&mut version) },
            self,
            "ncclGetVersion",
        )?;
        Ok(version)
    }

    fn error_string(&self, result: NcclResult) -> String {
        // SAFETY: NCCL returns a static null-terminated error string for the result code.
        let ptr = unsafe { (self.nccl_get_error_string)(result) };
        if ptr.is_null() {
            return format!("NCCL error code {result}");
        }
        // SAFETY: ptr is a valid C string per NCCL contract.
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

impl Drop for NcclLibrary {
    fn drop(&mut self) {
        #[cfg(unix)]
        if !self.handle.is_null() {
            // SAFETY: handle was returned by dlopen and is closed at most once here.
            unsafe {
                let _ = dlclose(self.handle);
            }
        }
    }
}

pub struct NcclCommunicator {
    library: NcclLibrary,
    driver: CudaDriver,
    device: CUdevice,
    context: CUcontext,
    stream: CUstream,
    comm: NcclComm,
    rank: i32,
    world_size: i32,
    device_ordinal: i32,
}

impl NcclCommunicator {
    pub fn init_rank(
        device_ordinal: i32,
        rank: i32,
        world_size: i32,
        unique_id: NcclUniqueId,
    ) -> CudaResult<Self> {
        if rank < 0 || rank >= world_size {
            return Err(CudaError::new(format!(
                "NCCL rank must be in [0, {world_size}), got {rank}"
            )));
        }
        if world_size <= 0 {
            return Err(CudaError::new(format!(
                "NCCL world_size must be positive, got {world_size}"
            )));
        }
        nccl_trace(format_args!(
            "rank={rank} device=cuda:{device_ordinal} loading NCCL library"
        ));
        let library = NcclLibrary::load()?;
        nccl_trace(format_args!(
            "rank={rank} device=cuda:{device_ordinal} retaining CUDA primary context"
        ));
        let (driver, device, context) = create_primary_context(device_ordinal)?;
        driver.set_current(context)?;
        nccl_trace(format_args!(
            "rank={rank} device=cuda:{device_ordinal} creating NCCL stream"
        ));
        let mut stream = ptr::null_mut();
        // SAFETY: stream points to valid writable memory and current context is set.
        check(
            unsafe { (driver.cu_stream_create)(&mut stream, 0) },
            "cuStreamCreate",
        )?;
        CUDA_STREAM_CREATE_CALLS.fetch_add(1, Ordering::Relaxed);
        let mut comm = ptr::null_mut();
        nccl_trace(format_args!(
            "rank={rank} device=cuda:{device_ordinal} calling ncclCommInitRank"
        ));
        // SAFETY: comm points to valid writable memory; unique_id came from NCCL or validated hex.
        check_nccl(
            unsafe { (library.nccl_comm_init_rank)(&mut comm, world_size, unique_id, rank) },
            &library,
            "ncclCommInitRank",
        )?;
        nccl_trace(format_args!(
            "rank={rank} device=cuda:{device_ordinal} ncclCommInitRank returned success"
        ));
        Ok(Self {
            library,
            driver,
            device,
            context,
            stream,
            comm,
            rank,
            world_size,
            device_ordinal,
        })
    }

    pub fn rank(&self) -> i32 {
        self.rank
    }

    pub fn world_size(&self) -> i32 {
        self.world_size
    }

    pub fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    pub fn version(&self) -> CudaResult<i32> {
        self.library.version()
    }

    pub fn start_event_timer(&self) -> CudaResult<CudaEventTimer> {
        let (driver, device, context) = create_primary_context(self.device_ordinal)?;
        CudaEventTimer::start_on_stream(driver, device, context, self.stream)
    }

    pub fn abort(&mut self) -> CudaResult<()> {
        if self.comm.is_null() {
            return Ok(());
        }
        self.driver.set_current(self.context)?;
        // SAFETY: comm is a live NCCL communicator; abort is used only once before clearing it.
        check_nccl(
            unsafe { (self.library.nccl_comm_abort)(self.comm) },
            &self.library,
            "ncclCommAbort",
        )?;
        self.comm = ptr::null_mut();
        Ok(())
    }

    pub fn all_reduce_sum_in_place_f32(
        &mut self,
        buffer: &CudaBuffer,
    ) -> CudaResult<NcclAllReduceStats> {
        ensure_f32_buffer(
            buffer,
            "NcclCommunicator::all_reduce_sum_in_place_f32 buffer",
        )?;
        if buffer.device_ordinal() != self.device_ordinal {
            return Err(CudaError::new(format!(
                "NCCL communicator is on cuda:{}, but buffer is on cuda:{}",
                self.device_ordinal,
                buffer.device_ordinal()
            )));
        }
        if buffer.is_empty() {
            return Ok(NcclAllReduceStats { calls: 0, bytes: 0 });
        }
        self.driver.set_current(self.context)?;
        let ptr = buffer.inner.ptr as *mut c_void;
        // SAFETY: ptr is a live f32 CUDA allocation on the communicator device; NCCL supports in-place all-reduce with send == recv.
        check_nccl(
            unsafe {
                (self.library.nccl_all_reduce)(
                    ptr as *const c_void,
                    ptr,
                    buffer.len(),
                    NCCL_FLOAT32,
                    NCCL_SUM,
                    self.comm,
                    self.stream,
                )
            },
            &self.library,
            "ncclAllReduce",
        )?;
        // SAFETY: stream was created for the current context and the NCCL operation was queued on it.
        check(
            unsafe { (self.driver.cu_stream_synchronize)(self.stream) },
            "cuStreamSynchronize",
        )?;
        CUDA_HOST_SYNC_CALLS.fetch_add(1, Ordering::Relaxed);
        CUDA_STREAM_SYNC_CALLS.fetch_add(1, Ordering::Relaxed);
        Ok(NcclAllReduceStats {
            calls: 1,
            bytes: buffer.byte_len(),
        })
    }
}

impl Drop for NcclCommunicator {
    fn drop(&mut self) {
        let _ = self.driver.set_current(self.context);
        if !self.comm.is_null() {
            // SAFETY: comm was initialized by ncclCommInitRank and is destroyed at most once here.
            unsafe {
                let _ = (self.library.nccl_comm_destroy)(self.comm);
            }
            self.comm = ptr::null_mut();
        }
        if !self.stream.is_null() {
            // SAFETY: stream was created by cuStreamCreate and is destroyed at most once here.
            unsafe {
                let _ = (self.driver.cu_stream_destroy)(self.stream);
            }
            self.stream = ptr::null_mut();
        }
        if !self.context.is_null() {
            // SAFETY: primary context was retained for this communicator and is released once here.
            unsafe {
                let _ = (self.driver.cu_device_primary_ctx_release)(self.device);
            }
        }
    }
}

fn nccl_trace(args: fmt::Arguments<'_>) {
    if std::env::var("HEIRLOOM_NCCL_TRACE")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
    {
        eprintln!("heirloom-kernels NCCL trace: {args}");
    }
}

pub fn nccl_info() -> CudaResult<NcclInfo> {
    let library = NcclLibrary::load()?;
    Ok(NcclInfo {
        loaded: true,
        version: library.version()?,
    })
}

pub struct NcclUniqueIdRoot {
    _library: NcclLibrary,
    unique_id: NcclUniqueId,
}

impl NcclUniqueIdRoot {
    pub fn new() -> CudaResult<Self> {
        let library = NcclLibrary::load()?;
        let unique_id = nccl_unique_id_with_library(&library)?;
        Ok(Self {
            _library: library,
            unique_id,
        })
    }

    pub fn unique_id(&self) -> NcclUniqueId {
        self.unique_id
    }
}

pub fn nccl_unique_id() -> CudaResult<NcclUniqueId> {
    let library = NcclLibrary::load()?;
    nccl_unique_id_with_library(&library)
}

fn nccl_unique_id_with_library(library: &NcclLibrary) -> CudaResult<NcclUniqueId> {
    let mut unique_id = NcclUniqueId {
        internal: [0 as c_char; 128],
    };
    // SAFETY: unique_id points to valid writable storage for NCCL's 128-byte id.
    check_nccl(
        unsafe { (library.nccl_get_unique_id)(&mut unique_id) },
        library,
        "ncclGetUniqueId",
    )?;
    Ok(unique_id)
}

pub fn device_supports_bf16_tensor_cores(device_ordinal: i32) -> CudaResult<bool> {
    let driver = CudaDriver::load()?;
    driver.init()?;
    let device = driver.device(device_ordinal)?;
    let major = driver.device_attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    Ok(major >= 8)
}

fn check_nccl(result: NcclResult, library: &NcclLibrary, context: &str) -> CudaResult<()> {
    if result == NCCL_SUCCESS {
        Ok(())
    } else {
        Err(CudaError::new(format!(
            "{context} failed: {}",
            library.error_string(result)
        )))
    }
}

struct CudaSession {
    driver: CudaDriver,
    context: CUcontext,
}

impl CudaSession {
    fn new(device_ordinal: i32) -> CudaResult<Self> {
        let (driver, context) = create_context(device_ordinal)?;
        Ok(Self { driver, context })
    }

    fn load_module<'a>(&'a self, ptx: &str) -> CudaResult<CudaModule<'a>> {
        load_module(&self.driver, ptx)
    }
}

impl Drop for CudaSession {
    fn drop(&mut self) {
        if !self.context.is_null() {
            let _ = self.driver.set_current(self.context);
            evict_stream_for_context(self.context);
            evict_modules_for_context(&self.driver, self.context);
            // SAFETY: context was created by cuCtxCreate_v2 and is destroyed once here.
            unsafe {
                let _ = (self.driver.cu_ctx_destroy)(self.context);
            }
        }
    }
}

fn create_context(device_ordinal: i32) -> CudaResult<(CudaDriver, CUcontext)> {
    let driver = CudaDriver::load()?;
    driver.init()?;
    let device_count = driver.device_count()?;
    if device_ordinal < 0 || device_ordinal >= device_count {
        return Err(CudaError::new(format!(
            "device ordinal {device_ordinal} out of range for {device_count} CUDA devices"
        )));
    }
    let device = driver.device(device_ordinal)?;
    let mut context = ptr::null_mut();
    // SAFETY: context points to valid writable memory and device was returned by cuDeviceGet.
    check(
        unsafe { (driver.cu_ctx_create)(&mut context, 0, device) },
        "cuCtxCreate_v2",
    )?;
    Ok((driver, context))
}

fn create_primary_context(device_ordinal: i32) -> CudaResult<(CudaDriver, CUdevice, CUcontext)> {
    let driver = CudaDriver::load()?;
    driver.init()?;
    let device_count = driver.device_count()?;
    if device_ordinal < 0 || device_ordinal >= device_count {
        return Err(CudaError::new(format!(
            "device ordinal {device_ordinal} out of range for {device_count} CUDA devices"
        )));
    }
    let device = driver.device(device_ordinal)?;
    let mut context = ptr::null_mut();
    // SAFETY: context points to writable memory and device was returned by cuDeviceGet.
    check(
        unsafe { (driver.cu_device_primary_ctx_retain)(&mut context, device) },
        "cuDevicePrimaryCtxRetain",
    )?;
    driver.set_current(context)?;
    Ok((driver, device, context))
}

fn load_module<'a>(driver: &'a CudaDriver, ptx: &str) -> CudaResult<CudaModule<'a>> {
    let context = driver.current_context()?;
    let cache_key = (context as usize, ptx_cache_label(ptx));
    if let Some(module) = CUDA_MODULE_CACHE.with(|cache| cache.borrow().get(&cache_key).copied()) {
        CUDA_MODULE_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
        return Ok(CudaModule { driver, module });
    }
    let ptx = CString::new(ptx).map_err(|err| CudaError::new(err.to_string()))?;
    let mut module = ptr::null_mut();
    let mut info_log = vec![0 as c_char; 8192];
    let mut error_log = vec![0 as c_char; 8192];
    let mut options = [
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    ];
    let mut option_values = [
        info_log.as_mut_ptr() as *mut c_void,
        info_log.len() as *mut c_void,
        error_log.as_mut_ptr() as *mut c_void,
        error_log.len() as *mut c_void,
    ];
    // SAFETY: ptx is a valid null-terminated PTX string for CUDA JIT loading.
    let result = unsafe {
        (driver.cu_module_load_data_ex)(
            &mut module,
            ptx.as_ptr() as *const c_void,
            options.len() as c_uint,
            options.as_mut_ptr(),
            option_values.as_mut_ptr(),
        )
    };
    if result != CUDA_SUCCESS {
        let error_log = jit_log_string(&error_log);
        let info_log = jit_log_string(&info_log);
        let mut message = format!("cuModuleLoadDataEx returned CUDA error {result}");
        if !error_log.is_empty() {
            message.push_str(&format!("; JIT error log: {error_log}"));
        }
        if !info_log.is_empty() {
            message.push_str(&format!("; JIT info log: {info_log}"));
        }
        return Err(CudaError::new(message));
    }
    CUDA_MODULE_LOAD_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_MODULE_CACHE.with(|cache| {
        cache.borrow_mut().insert(cache_key, module);
    });
    Ok(CudaModule { driver, module })
}

fn ptx_cache_label(ptx: &str) -> &'static str {
    if ptx.as_ptr() == KERNEL_PTX.as_ptr() && ptx.len() == KERNEL_PTX.len() {
        "kernel"
    } else if ptx.as_ptr() == TENSOR_CORE_PTX.as_ptr() && ptx.len() == TENSOR_CORE_PTX.len() {
        "tensor_core"
    } else {
        "custom"
    }
}

fn evict_modules_for_context(driver: &CudaDriver, context: CUcontext) {
    let context_key = context as usize;
    let modules = CUDA_MODULE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let keys = cache
            .keys()
            .copied()
            .filter(|(stored_context, _)| *stored_context == context_key)
            .collect::<Vec<_>>();
        keys.into_iter()
            .filter_map(|key| cache.remove(&key))
            .collect::<Vec<_>>()
    });
    for module in modules {
        // SAFETY: module was loaded by cuModuleLoadDataEx for this context and is unloaded once here.
        unsafe {
            let _ = (driver.cu_module_unload)(module);
        }
    }
}

fn jit_log_string(log: &[c_char]) -> String {
    let bytes = log
        .iter()
        .copied()
        .take_while(|&byte| byte != 0)
        .map(|byte| byte as u8)
        .collect::<Vec<_>>();
    String::from_utf8_lossy(&bytes).trim().to_string()
}

struct CudaModule<'a> {
    driver: &'a CudaDriver,
    module: CUmodule,
}

impl CudaModule<'_> {
    fn function(&self, name: &str) -> CudaResult<CUfunction> {
        let name = CString::new(name).map_err(|err| CudaError::new(err.to_string()))?;
        let mut function = ptr::null_mut();
        // SAFETY: module is valid and function points to valid writable memory.
        check(
            unsafe {
                (self.driver.cu_module_get_function)(&mut function, self.module, name.as_ptr())
            },
            "cuModuleGetFunction",
        )?;
        Ok(function)
    }
}

impl Drop for CudaModule<'_> {
    fn drop(&mut self) {
        // Modules are cached per context and unloaded when an owned throwaway context is destroyed.
        // Primary-context modules intentionally remain hot for the lifetime of the thread.
    }
}

struct DeviceBuffer<'a> {
    driver: &'a CudaDriver,
    ptr: CUdeviceptr,
    len: usize,
}

impl<'a> DeviceBuffer<'a> {
    fn uninit_f32(driver: &'a CudaDriver, len: usize) -> CudaResult<Self> {
        let mut ptr = 0;
        let bytes = len * std::mem::size_of::<f32>();
        // SAFETY: ptr points to valid writable memory for the allocated device pointer.
        check(
            unsafe { (driver.cu_mem_alloc)(&mut ptr, bytes) },
            "cuMemAlloc_v2",
        )?;
        Ok(Self { driver, ptr, len })
    }

    fn from_f32(driver: &'a CudaDriver, data: &[f32]) -> CudaResult<Self> {
        let buffer = Self::uninit_f32(driver, data.len())?;
        let bytes = std::mem::size_of_val(data);
        // SAFETY: buffer.ptr is a valid allocation and data.as_ptr is valid for bytes.
        check(
            unsafe { (driver.cu_memcpy_htod)(buffer.ptr, data.as_ptr() as *const c_void, bytes) },
            "cuMemcpyHtoD_v2",
        )?;
        CUDA_H2D_BYTES.fetch_add(bytes, Ordering::Relaxed);
        Ok(buffer)
    }

    fn copy_to_f32(&self) -> CudaResult<Vec<f32>> {
        let mut out = vec![0.0; self.len];
        let bytes = out.len() * std::mem::size_of::<f32>();
        synchronize_current_compute_stream(
            self.driver,
            "cuStreamSynchronize before cuMemcpyDtoH_v2",
        )?;
        // SAFETY: out is valid writable host memory and self.ptr is a valid device allocation.
        check(
            unsafe {
                (self.driver.cu_memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, self.ptr, bytes)
            },
            "cuMemcpyDtoH_v2",
        )?;
        CUDA_D2H_BYTES.fetch_add(bytes, Ordering::Relaxed);
        Ok(out)
    }
}

impl Drop for DeviceBuffer<'_> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: ptr was allocated by cuMemAlloc_v2 and is freed once here.
            unsafe {
                let _ = (self.driver.cu_mem_free)(self.ptr);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct KernelLaunchConfig {
    label: &'static str,
    grid: (c_uint, c_uint, c_uint),
    block: (c_uint, c_uint, c_uint),
    shared_mem_bytes: c_uint,
    elements: usize,
}

fn launch_kernel_on_current_stream(
    driver: &CudaDriver,
    function: CUfunction,
    config: KernelLaunchConfig,
    params: &mut [*mut c_void],
) -> CudaResult<()> {
    let stream = compute_stream_for_current_context(driver)?;
    // SAFETY: function is a valid CUDA kernel, params point to live kernel argument values, and stream belongs to the current context.
    check(
        unsafe {
            (driver.cu_launch_kernel)(
                function,
                config.grid.0,
                config.grid.1,
                config.grid.2,
                config.block.0,
                config.block.1,
                config.block.2,
                config.shared_mem_bytes,
                stream,
                params.as_mut_ptr(),
                ptr::null_mut(),
            )
        },
        "cuLaunchKernel",
    )?;
    record_kernel_launch(config.label, config.elements);
    Ok(())
}

fn launch_vector_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    len: usize,
) -> CudaResult<()> {
    launch_vector_kernel_labeled(driver, function, device_ptrs, len, "vector_elementwise")
}

fn launch_vector_kernel_labeled(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    len: usize,
    label: &'static str,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut n = len as u32;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut n as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = (len as u32).div_ceil(block);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label,
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: len,
        },
        &mut params,
    )
}

fn launch_scaled_vector_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    scale: f32,
    len: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut scale = scale;
    let mut n = len as u32;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut scale as *mut f32 as *mut c_void);
    params.push(&mut n as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = (len as u32).div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "scaled_vector_elementwise",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: len,
        },
        &mut params,
    )
}

fn launch_tensor_core_probe_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    output: CUdeviceptr,
) -> CudaResult<()> {
    let mut output = output;
    let mut params = vec![&mut output as *mut CUdeviceptr as *mut c_void];
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "tensor_core_probe",
            grid: (1, 1, 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
            elements: 32,
        },
        &mut params,
    )
}

fn record_tensor_core_cta_plan(plan: TensorCoreMatmulCtaPlan) {
    CUDA_TENSOR_CORE_CTA_GEMM_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_TENSOR_CORE_CTA_TILES.fetch_add(plan.cta_tiles, Ordering::Relaxed);
    CUDA_TENSOR_CORE_CTA_WARPS_LAUNCHED.fetch_add(plan.launched_warps, Ordering::Relaxed);
    CUDA_TENSOR_CORE_MMA_WARP_TILES.fetch_add(plan.active_mma_warp_tiles, Ordering::Relaxed);
}

fn start_tensor_core_gemm_timer(device_ordinal: i32) -> CudaResult<Option<CudaEventTimer>> {
    if tensor_core_gemm_timing_enabled() {
        Ok(Some(CudaEventTimer::start_current_compute_stream(
            device_ordinal,
        )?))
    } else {
        Ok(None)
    }
}

fn stop_tensor_core_gemm_timer(
    timer: &mut Option<CudaEventTimer>,
    elapsed_counter: &AtomicUsize,
) -> CudaResult<()> {
    if let Some(timer) = timer.as_mut() {
        let elapsed_ms = timer.stop_elapsed_ms()?;
        let elapsed_us = (elapsed_ms * 1000.0).max(0.0) as usize;
        elapsed_counter.fetch_add(elapsed_us, Ordering::Relaxed);
    }
    Ok(())
}

fn launch_bf16_mma_matmul_staged_cta_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    m: usize,
    k: usize,
    n: usize,
) -> CudaResult<()> {
    let plan = bf16_tensor_core_matmul_cta_plan(m, k, n)?;
    let mut ptr_args = device_ptrs.to_vec();
    let mut m = u32_kernel_dim(m, "m")?;
    let mut k = u32_kernel_dim(k, "k")?;
    let mut n = u32_kernel_dim(n, "n")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut m as *mut u32 as *mut c_void);
    params.push(&mut k as *mut u32 as *mut c_void);
    params.push(&mut n as *mut u32 as *mut c_void);
    let grid_x = u32_kernel_dim(plan.grid_x, "Tensor Core CTA grid_x")?;
    let grid_y = u32_kernel_dim(plan.grid_y, "Tensor Core CTA grid_y")?;
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "tensor_core_gemm_staged_cta",
            grid: (grid_x, grid_y, 1),
            block: (128, 1, 1),
            shared_mem_bytes: 0,
            elements: (m as usize) * (n as usize),
        },
        &mut params,
    )?;
    record_tensor_core_cta_plan(plan);
    CUDA_TENSOR_CORE_STAGED_CTA_GEMM_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_TENSOR_CORE_SHARED_STAGE_TILES.fetch_add(plan.shared_stage_tiles, Ordering::Relaxed);
    CUDA_TENSOR_CORE_SHARED_STAGE_BYTES.fetch_add(plan.shared_stage_bytes, Ordering::Relaxed);
    Ok(())
}

fn launch_bf16_mma_matmul_normal_rhs_staged_cta_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    m: usize,
    k: usize,
    n: usize,
) -> CudaResult<()> {
    let plan = bf16_tensor_core_matmul_cta_plan(m, k, n)?;
    let mut ptr_args = device_ptrs.to_vec();
    let mut m = u32_kernel_dim(m, "normal RHS m")?;
    let mut k = u32_kernel_dim(k, "normal RHS k")?;
    let mut n = u32_kernel_dim(n, "normal RHS n")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut m as *mut u32 as *mut c_void);
    params.push(&mut k as *mut u32 as *mut c_void);
    params.push(&mut n as *mut u32 as *mut c_void);
    let grid_x = u32_kernel_dim(plan.grid_x, "normal RHS Tensor Core CTA grid_x")?;
    let grid_y = u32_kernel_dim(plan.grid_y, "normal RHS Tensor Core CTA grid_y")?;
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "tensor_core_gemm_normal_rhs_staged",
            grid: (grid_x, grid_y, 1),
            block: (128, 1, 1),
            shared_mem_bytes: 0,
            elements: (m as usize) * (n as usize),
        },
        &mut params,
    )?;
    record_tensor_core_cta_plan(plan);
    CUDA_TENSOR_CORE_SHARED_STAGE_TILES.fetch_add(plan.shared_stage_tiles, Ordering::Relaxed);
    CUDA_TENSOR_CORE_SHARED_STAGE_BYTES.fetch_add(plan.shared_stage_bytes, Ordering::Relaxed);
    Ok(())
}

fn launch_bf16_mma_matmul_ldmatrix_cta_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    m: usize,
    k: usize,
    n: usize,
) -> CudaResult<()> {
    let plan = bf16_tensor_core_matmul_cta_plan(m, k, n)?;
    let mut ptr_args = device_ptrs.to_vec();
    let mut m = u32_kernel_dim(m, "m")?;
    let mut k = u32_kernel_dim(k, "k")?;
    let mut n = u32_kernel_dim(n, "n")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut m as *mut u32 as *mut c_void);
    params.push(&mut k as *mut u32 as *mut c_void);
    params.push(&mut n as *mut u32 as *mut c_void);
    let grid_x = u32_kernel_dim(plan.grid_x, "ldmatrix Tensor Core CTA grid_x")?;
    let grid_y = u32_kernel_dim(plan.grid_y, "ldmatrix Tensor Core CTA grid_y")?;
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "tensor_core_gemm_ldmatrix",
            grid: (grid_x, grid_y, 1),
            block: (128, 1, 1),
            shared_mem_bytes: 0,
            elements: (m as usize) * (n as usize),
        },
        &mut params,
    )?;
    record_tensor_core_cta_plan(plan);
    CUDA_TENSOR_CORE_LDMATRIX_GEMM_EXECUTED_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_TENSOR_CORE_LDMATRIX_GEMM_INSTRUCTIONS
        .fetch_add(plan.active_mma_warp_tiles, Ordering::Relaxed);
    CUDA_TENSOR_CORE_SHARED_STAGE_TILES.fetch_add(plan.shared_stage_tiles, Ordering::Relaxed);
    CUDA_TENSOR_CORE_SHARED_STAGE_BYTES.fetch_add(plan.shared_stage_bytes, Ordering::Relaxed);
    Ok(())
}

fn launch_bf16_mma_matmul_cp_async_cta_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    m: usize,
    k: usize,
    n: usize,
) -> CudaResult<()> {
    launch_bf16_mma_matmul_cp_async_cta_kernel_optional_bias(
        driver,
        function,
        device_ptrs,
        None,
        m,
        k,
        n,
    )
}

fn launch_bf16_mma_matmul_cp_async_cta_kernel_with_bias(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    bias: CUdeviceptr,
    m: usize,
    k: usize,
    n: usize,
) -> CudaResult<()> {
    launch_bf16_mma_matmul_cp_async_cta_kernel_optional_bias(
        driver,
        function,
        device_ptrs,
        Some(bias),
        m,
        k,
        n,
    )
}

fn launch_bf16_mma_matmul_cp_async_cta_kernel_optional_bias(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    bias: Option<CUdeviceptr>,
    m: usize,
    k: usize,
    n: usize,
) -> CudaResult<()> {
    let plan = bf16_tensor_core_matmul_cta_plan(m, k, n)?;
    let mut ptr_args = device_ptrs.to_vec();
    let mut bias_ptr = bias.unwrap_or(0);
    let mut add_bias = u32::from(bias.is_some());
    let mut m = u32_kernel_dim(m, "m")?;
    let mut k = u32_kernel_dim(k, "k")?;
    let mut n = u32_kernel_dim(n, "n")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut m as *mut u32 as *mut c_void);
    params.push(&mut k as *mut u32 as *mut c_void);
    params.push(&mut n as *mut u32 as *mut c_void);
    params.push(&mut bias_ptr as *mut CUdeviceptr as *mut c_void);
    params.push(&mut add_bias as *mut u32 as *mut c_void);
    let grid_x = u32_kernel_dim(plan.grid_x, "cp.async Tensor Core CTA grid_x")?;
    let grid_y = u32_kernel_dim(plan.grid_y, "cp.async Tensor Core CTA grid_y")?;
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "tensor_core_gemm_cp_async",
            grid: (grid_x, grid_y, 1),
            block: (128, 1, 1),
            shared_mem_bytes: 0,
            elements: (m as usize) * (n as usize),
        },
        &mut params,
    )?;
    record_tensor_core_cta_plan(plan);
    CUDA_TENSOR_CORE_CP_ASYNC_GEMM_EXECUTED_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_TENSOR_CORE_CP_ASYNC_GEMM_INSTRUCTIONS.fetch_add(
        checked_mul(
            plan.shared_stage_tiles,
            96,
            "cp.async Tensor Core GEMM instruction count",
        )?,
        Ordering::Relaxed,
    );
    CUDA_TENSOR_CORE_LDMATRIX_GEMM_INSTRUCTIONS
        .fetch_add(plan.active_mma_warp_tiles, Ordering::Relaxed);
    CUDA_TENSOR_CORE_SHARED_STAGE_TILES.fetch_add(plan.shared_stage_tiles, Ordering::Relaxed);
    CUDA_TENSOR_CORE_SHARED_STAGE_BYTES.fetch_add(plan.shared_stage_bytes, Ordering::Relaxed);
    Ok(())
}

fn launch_bf16_mma_matmul_wide_swizzled_cta_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    m: usize,
    k: usize,
    n: usize,
) -> CudaResult<()> {
    let plan = bf16_tensor_core_matmul_wide_swizzled_cta_plan(m, k, n)?;
    let mut ptr_args = device_ptrs.to_vec();
    let mut m = u32_kernel_dim(m, "m")?;
    let mut k = u32_kernel_dim(k, "k")?;
    let mut n = u32_kernel_dim(n, "n")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut m as *mut u32 as *mut c_void);
    params.push(&mut k as *mut u32 as *mut c_void);
    params.push(&mut n as *mut u32 as *mut c_void);
    let grid_x = u32_kernel_dim(plan.grid_x, "wide swizzled Tensor Core CTA grid_x")?;
    let grid_y = u32_kernel_dim(plan.grid_y, "wide swizzled Tensor Core CTA grid_y")?;
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "tensor_core_gemm_wide_swizzled",
            grid: (grid_x, grid_y, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
            elements: (m as usize) * (n as usize),
        },
        &mut params,
    )?;
    record_tensor_core_cta_plan(plan);
    CUDA_TENSOR_CORE_WIDE_SWIZZLED_CTA_GEMM_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_TENSOR_CORE_SWIZZLED_STAGE_TILES.fetch_add(plan.shared_stage_tiles, Ordering::Relaxed);
    CUDA_TENSOR_CORE_SWIZZLED_STAGE_BYTES.fetch_add(plan.shared_stage_bytes, Ordering::Relaxed);
    Ok(())
}

fn launch_bf16_mma_matmul_global_cta_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    m: usize,
    k: usize,
    n: usize,
) -> CudaResult<()> {
    let plan = bf16_tensor_core_matmul_cta_plan(m, k, n)?;
    let mut ptr_args = device_ptrs.to_vec();
    let mut m = u32_kernel_dim(m, "m")?;
    let mut k = u32_kernel_dim(k, "k")?;
    let mut n = u32_kernel_dim(n, "n")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut m as *mut u32 as *mut c_void);
    params.push(&mut k as *mut u32 as *mut c_void);
    params.push(&mut n as *mut u32 as *mut c_void);
    let grid_x = u32_kernel_dim(plan.grid_x, "Tensor Core CTA grid_x")?;
    let grid_y = u32_kernel_dim(plan.grid_y, "Tensor Core CTA grid_y")?;
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "tensor_core_gemm_global_cta",
            grid: (grid_x, grid_y, 1),
            block: (128, 1, 1),
            shared_mem_bytes: 0,
            elements: (m as usize) * (n as usize),
        },
        &mut params,
    )?;
    record_tensor_core_cta_plan(plan);
    CUDA_TENSOR_CORE_GLOBAL_CTA_GEMM_CALLS.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

fn launch_bf16_mma_matmul_legacy_warp_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    m: usize,
    k: usize,
    n: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut m = u32_kernel_dim(m, "m")?;
    let mut k = u32_kernel_dim(k, "k")?;
    let mut n = u32_kernel_dim(n, "n")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut m as *mut u32 as *mut c_void);
    params.push(&mut k as *mut u32 as *mut c_void);
    params.push(&mut n as *mut u32 as *mut c_void);
    let grid_x = (n / 8).max(1);
    let grid_y = (m / 16).max(1);
    let warp_tiles = (grid_x as usize) * (grid_y as usize);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "tensor_core_gemm_legacy_warp",
            grid: (grid_x, grid_y, 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
            elements: (m as usize) * (n as usize),
        },
        &mut params,
    )?;
    CUDA_TENSOR_CORE_LEGACY_WARP_GEMM_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_TENSOR_CORE_MMA_WARP_TILES.fetch_add(warp_tiles, Ordering::Relaxed);
    Ok(())
}

#[derive(Clone, Copy)]
struct MatrixPadCropLaunch {
    input: CUdeviceptr,
    output: CUdeviceptr,
    rows: usize,
    cols: usize,
    padded_cols: usize,
    total: usize,
}

fn launch_pad_matrix_bf16_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    launch: MatrixPadCropLaunch,
) -> CudaResult<()> {
    let mut input = launch.input;
    let mut output = launch.output;
    let mut rows = u32_kernel_dim(launch.rows, "rows")?;
    let mut cols = u32_kernel_dim(launch.cols, "cols")?;
    let mut padded_cols = u32_kernel_dim(launch.padded_cols, "padded_cols")?;
    let mut total = u32_kernel_dim(launch.total, "total")?;
    let mut params = vec![
        &mut input as *mut CUdeviceptr as *mut c_void,
        &mut output as *mut CUdeviceptr as *mut c_void,
        &mut rows as *mut u32 as *mut c_void,
        &mut cols as *mut u32 as *mut c_void,
        &mut padded_cols as *mut u32 as *mut c_void,
        &mut total as *mut u32 as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total).div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "pad_matrix_bf16",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: total as usize,
        },
        &mut params,
    )
}

fn launch_crop_matrix_f32_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    launch: MatrixPadCropLaunch,
) -> CudaResult<()> {
    let mut input = launch.input;
    let mut output = launch.output;
    let mut rows = u32_kernel_dim(launch.rows, "rows")?;
    let mut cols = u32_kernel_dim(launch.cols, "cols")?;
    let mut padded_cols = u32_kernel_dim(launch.padded_cols, "padded_cols")?;
    let mut total = u32_kernel_dim(launch.total, "total")?;
    let mut params = vec![
        &mut input as *mut CUdeviceptr as *mut c_void,
        &mut output as *mut CUdeviceptr as *mut c_void,
        &mut rows as *mut u32 as *mut c_void,
        &mut cols as *mut u32 as *mut c_void,
        &mut padded_cols as *mut u32 as *mut c_void,
        &mut total as *mut u32 as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total).div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "crop_matrix_f32",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: total as usize,
        },
        &mut params,
    )
}

#[derive(Clone, Copy)]
struct AttentionMmaLaunch {
    dims: CausalAttentionDims,
    head_dim: usize,
    grid_x: usize,
    grid_y: usize,
    grid_z: usize,
}

fn launch_bf16_mma_attention_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    launch: AttentionMmaLaunch,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let dims = launch.dims;
    let mut batch = u32_kernel_dim(dims.batch, "batch")?;
    let mut time = u32_kernel_dim(dims.time, "time")?;
    let mut channels = u32_kernel_dim(dims.channels, "channels")?;
    let mut n_heads = u32_kernel_dim(dims.n_heads, "n_heads")?;
    let mut head_dim = u32_kernel_dim(launch.head_dim, "head_dim")?;
    let grid_x = u32_kernel_dim(launch.grid_x, "grid_x")?.max(1);
    let grid_y = u32_kernel_dim(launch.grid_y, "grid_y")?.max(1);
    let grid_z = u32_kernel_dim(launch.grid_z, "grid_z")?.max(1);
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut batch as *mut u32 as *mut c_void);
    params.push(&mut time as *mut u32 as *mut c_void);
    params.push(&mut channels as *mut u32 as *mut c_void);
    params.push(&mut n_heads as *mut u32 as *mut c_void);
    params.push(&mut head_dim as *mut u32 as *mut c_void);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "attention_bf16_mma_materialized",
            grid: (grid_x, grid_y, grid_z),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
            elements: dims.batch * dims.n_heads * dims.time * dims.time,
        },
        &mut params,
    )
}

fn launch_flash_bf16_tensor_core_attention_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    dims: CausalAttentionDims,
    head_dim: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut batch = u32_kernel_dim(dims.batch, "batch")?;
    let mut time = u32_kernel_dim(dims.time, "time")?;
    let mut channels = u32_kernel_dim(dims.channels, "channels")?;
    let mut n_heads = u32_kernel_dim(dims.n_heads, "n_heads")?;
    let mut head_dim_param = u32_kernel_dim(head_dim, "head_dim")?;
    let grid_x = u32_kernel_dim(head_dim.div_ceil(8), "flash attention output dim tiles")?;
    let grid_y = u32_kernel_dim(dims.time.div_ceil(16), "flash attention query blocks")?;
    let grid_z = u32_kernel_dim(
        checked_mul(dims.batch, dims.n_heads, "flash attention batch-head grid")?,
        "flash attention batch-head grid",
    )?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut batch as *mut u32 as *mut c_void);
    params.push(&mut time as *mut u32 as *mut c_void);
    params.push(&mut channels as *mut u32 as *mut c_void);
    params.push(&mut n_heads as *mut u32 as *mut c_void);
    params.push(&mut head_dim_param as *mut u32 as *mut c_void);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "flash_attention_bf16_tensor_core_forward",
            grid: (grid_x.max(1), grid_y.max(1), grid_z.max(1)),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
            elements: dims.batch * dims.n_heads * dims.time * head_dim,
        },
        &mut params,
    )
}

fn launch_flash_bf16_tensor_core_attention_backward_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    dims: CausalAttentionDims,
    head_dim: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut batch = u32_kernel_dim(dims.batch, "batch")?;
    let mut time = u32_kernel_dim(dims.time, "time")?;
    let mut channels = u32_kernel_dim(dims.channels, "channels")?;
    let mut n_heads = u32_kernel_dim(dims.n_heads, "n_heads")?;
    let mut head_dim_param = u32_kernel_dim(head_dim, "head_dim")?;
    let output_dim_tiles = head_dim.div_ceil(8);
    let key_blocks = dims.time.div_ceil(16);
    let grid_x = u32_kernel_dim(
        checked_mul(
            key_blocks,
            output_dim_tiles,
            "flash attention backward key/output grid",
        )?,
        "flash attention backward key/output grid",
    )?;
    let grid_y = u32_kernel_dim(
        dims.time.div_ceil(16),
        "flash attention backward query blocks",
    )?;
    let grid_z = u32_kernel_dim(
        checked_mul(
            dims.batch,
            dims.n_heads,
            "flash attention backward batch-head grid",
        )?,
        "flash attention backward batch-head grid",
    )?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut batch as *mut u32 as *mut c_void);
    params.push(&mut time as *mut u32 as *mut c_void);
    params.push(&mut channels as *mut u32 as *mut c_void);
    params.push(&mut n_heads as *mut u32 as *mut c_void);
    params.push(&mut head_dim_param as *mut u32 as *mut c_void);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "flash_attention_bf16_tensor_core_backward",
            grid: (grid_x.max(1), grid_y.max(1), grid_z.max(1)),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
            elements: dims.batch * dims.n_heads * dims.time * head_dim,
        },
        &mut params,
    )
}

fn launch_materialize_matrix_layout_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    input: CUdeviceptr,
    output: CUdeviceptr,
    layout: MatrixLayout,
    total: usize,
) -> CudaResult<()> {
    let mut input = input;
    let mut output = output;
    let mut rows = u32_kernel_dim(layout.rows, "rows")?;
    let mut cols = u32_kernel_dim(layout.cols, "cols")?;
    let mut row_stride = u32_kernel_dim(layout.row_stride, "row_stride")?;
    let mut col_stride = u32_kernel_dim(layout.col_stride, "col_stride")?;
    let mut offset = u32_kernel_dim(layout.offset, "offset")?;
    let mut total = u32_kernel_dim(total, "total")?;
    let mut params = vec![
        &mut input as *mut CUdeviceptr as *mut c_void,
        &mut output as *mut CUdeviceptr as *mut c_void,
        &mut rows as *mut u32 as *mut c_void,
        &mut cols as *mut u32 as *mut c_void,
        &mut row_stride as *mut u32 as *mut c_void,
        &mut col_stride as *mut u32 as *mut c_void,
        &mut offset as *mut u32 as *mut c_void,
        &mut total as *mut u32 as *mut c_void,
    ];
    let block = 256u32;
    let grid = total.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "materialize_matrix_layout",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: total as usize,
        },
        &mut params,
    )
}

fn launch_adamw_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    adamw: AdamWParams,
    len: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut lr = adamw.lr;
    let mut beta1 = adamw.beta1;
    let mut beta2 = adamw.beta2;
    let mut eps = adamw.eps;
    let mut weight_decay = adamw.weight_decay;
    let mut clip_scale = adamw.clip_scale;
    let mut bias_correction1 = adamw.bias_correction1;
    let mut bias_correction2 = adamw.bias_correction2;
    let mut n = u32_kernel_dim(len, "n")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut lr as *mut f32 as *mut c_void);
    params.push(&mut beta1 as *mut f32 as *mut c_void);
    params.push(&mut beta2 as *mut f32 as *mut c_void);
    params.push(&mut eps as *mut f32 as *mut c_void);
    params.push(&mut weight_decay as *mut f32 as *mut c_void);
    params.push(&mut clip_scale as *mut f32 as *mut c_void);
    params.push(&mut bias_correction1 as *mut f32 as *mut c_void);
    params.push(&mut bias_correction2 as *mut f32 as *mut c_void);
    params.push(&mut n as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = n.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "adamw_dense",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: n as usize,
        },
        &mut params,
    )
}

fn launch_matmul_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    m: usize,
    k: usize,
    n: usize,
    total: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut m = u32_kernel_dim(m, "m")?;
    let mut k = u32_kernel_dim(k, "k")?;
    let mut n = u32_kernel_dim(n, "n")?;
    let mut total = u32_kernel_dim(total, "total")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut m as *mut u32 as *mut c_void);
    params.push(&mut k as *mut u32 as *mut c_void);
    params.push(&mut n as *mut u32 as *mut c_void);
    params.push(&mut total as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = total.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "matmul_f32_reference",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: total as usize,
        },
        &mut params,
    )
}

fn launch_sparse_adamw_rows_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    adamw: AdamWParams,
    dims: SparseAdamWRowsDims,
    elements: usize,
    has_row_mask: bool,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut lr = adamw.lr;
    let mut beta1 = adamw.beta1;
    let mut beta2 = adamw.beta2;
    let mut eps = adamw.eps;
    let mut weight_decay = adamw.weight_decay;
    let mut clip_scale = adamw.clip_scale;
    let mut bias_correction1 = adamw.bias_correction1;
    let mut bias_correction2 = adamw.bias_correction2;
    let mut selected_rows = u32_kernel_dim(dims.selected_rows, "selected_rows")?;
    let mut rows = u32_kernel_dim(dims.rows, "rows")?;
    let mut row_dim = u32_kernel_dim(dims.row_dim, "row_dim")?;
    let mut total = u32_kernel_dim(elements, "total")?;
    let mut has_row_mask = u32::from(has_row_mask);
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut lr as *mut f32 as *mut c_void);
    params.push(&mut beta1 as *mut f32 as *mut c_void);
    params.push(&mut beta2 as *mut f32 as *mut c_void);
    params.push(&mut eps as *mut f32 as *mut c_void);
    params.push(&mut weight_decay as *mut f32 as *mut c_void);
    params.push(&mut clip_scale as *mut f32 as *mut c_void);
    params.push(&mut bias_correction1 as *mut f32 as *mut c_void);
    params.push(&mut bias_correction2 as *mut f32 as *mut c_void);
    params.push(&mut selected_rows as *mut u32 as *mut c_void);
    params.push(&mut rows as *mut u32 as *mut c_void);
    params.push(&mut row_dim as *mut u32 as *mut c_void);
    params.push(&mut total as *mut u32 as *mut c_void);
    params.push(&mut has_row_mask as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = total.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "sparse_adamw_rows",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements,
        },
        &mut params,
    )
}

fn launch_selected_rows_gather_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    dims: SelectedRowsDims,
    elements: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut selected_rows = u32_kernel_dim(dims.selected_rows, "selected_rows")?;
    let mut rows = u32_kernel_dim(dims.rows, "rows")?;
    let mut row_dim = u32_kernel_dim(dims.row_dim, "row_dim")?;
    let mut total = u32_kernel_dim(elements, "total")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut selected_rows as *mut u32 as *mut c_void);
    params.push(&mut rows as *mut u32 as *mut c_void);
    params.push(&mut row_dim as *mut u32 as *mut c_void);
    params.push(&mut total as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = total.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "selected_rows_gather",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements,
        },
        &mut params,
    )
}

fn launch_i64_copy_with_offset_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    output_offset: usize,
    len: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut output_offset = u32_kernel_dim(output_offset, "output_offset")?;
    let mut n = u32_kernel_dim(len, "n")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut output_offset as *mut u32 as *mut c_void);
    params.push(&mut n as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = n.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "i64_copy_with_offset",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: n as usize,
        },
        &mut params,
    )
}

fn launch_memory_access_count_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    selected_count: usize,
    memory_slots: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut selected_count = u32_kernel_dim(selected_count, "selected_count")?;
    let mut memory_slots = u32_kernel_dim(memory_slots, "memory_slots")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut selected_count as *mut u32 as *mut c_void);
    params.push(&mut memory_slots as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = selected_count.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "memory_access_count",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: selected_count as usize,
        },
        &mut params,
    )
}

fn launch_i64_arange_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    len: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut len = u32_kernel_dim(len, "i64 arange len")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut len as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = len.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "i64_arange",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: len as usize,
        },
        &mut params,
    )
}

fn launch_selected_rows_to_f32_mask_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    selected_count: usize,
    rows: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut selected_count = u32_kernel_dim(selected_count, "selected_count")?;
    let mut rows = u32_kernel_dim(rows, "rows")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut selected_count as *mut u32 as *mut c_void);
    params.push(&mut rows as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = selected_count.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "selected_rows_to_f32_mask",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: selected_count as usize,
        },
        &mut params,
    )
}

fn launch_f32_mask_to_bool_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    len: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut len = u32_kernel_dim(len, "f32 mask len")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut len as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = len.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "f32_mask_to_bool",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: len as usize,
        },
        &mut params,
    )
}

fn launch_bool_and_u8_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    len: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut len = u32_kernel_dim(len, "bool and len")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut len as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = len.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "bool_and_u8",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: len as usize,
        },
        &mut params,
    )
}

fn launch_bool_mask_to_i64_indices_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    len: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut len = u32_kernel_dim(len, "bool mask to indices len")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut len as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = len.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "bool_mask_to_i64_indices",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: len as usize,
        },
        &mut params,
    )
}

fn launch_memory_topk_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    dims: MemoryTopkDims,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut rows = dims.rows as u32;
    let mut cols = dims.cols as u32;
    let mut top_k = dims.top_k as u32;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut rows as *mut u32 as *mut c_void);
    params.push(&mut cols as *mut u32 as *mut c_void);
    params.push(&mut top_k as *mut u32 as *mut c_void);
    let block = 128u32;
    let grid = rows.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "memory_topk",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: dims.rows * dims.top_k,
        },
        &mut params,
    )
}

fn launch_memory_product_key_side_scores_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    dims: MemoryProductKeyDims,
    side: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut tokens = u32_kernel_dim(dims.tokens, "tokens")?;
    let mut side = u32_kernel_dim(side, "side")?;
    let mut key_dim = u32_kernel_dim(dims.key_dim, "key_dim")?;
    let total = checked_mul(
        checked_mul(
            dims.tokens,
            side as usize,
            "product-key side score elements",
        )?,
        2,
        "product-key side score outputs",
    )?;
    let mut total_u32 = u32_kernel_dim(total, "total")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut tokens as *mut u32 as *mut c_void);
    params.push(&mut side as *mut u32 as *mut c_void);
    params.push(&mut key_dim as *mut u32 as *mut c_void);
    params.push(&mut total_u32 as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = total_u32.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "memory_product_key_side_scores",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: total,
        },
        &mut params,
    )
}

fn launch_memory_product_key_candidates_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    dims: MemoryProductKeyDims,
    side: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut tokens = u32_kernel_dim(dims.tokens, "tokens")?;
    let mut side = u32_kernel_dim(side, "side")?;
    let mut key_dim = u32_kernel_dim(dims.key_dim, "key_dim")?;
    let mut top_k = u32_kernel_dim(dims.top_k, "top_k")?;
    let mut beam = u32_kernel_dim(dims.beam, "beam")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut tokens as *mut u32 as *mut c_void);
    params.push(&mut side as *mut u32 as *mut c_void);
    params.push(&mut key_dim as *mut u32 as *mut c_void);
    params.push(&mut top_k as *mut u32 as *mut c_void);
    params.push(&mut beam as *mut u32 as *mut c_void);
    let block = 128u32;
    let grid = tokens.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "memory_product_key_candidates",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: dims.tokens * dims.top_k,
        },
        &mut params,
    )
}

fn launch_memory_product_key_candidates_from_scores_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    dims: MemoryProductKeyDims,
    side: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut tokens = u32_kernel_dim(dims.tokens, "tokens")?;
    let mut side = u32_kernel_dim(side, "side")?;
    let mut top_k = u32_kernel_dim(dims.top_k, "top_k")?;
    let mut beam = u32_kernel_dim(dims.beam, "beam")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut tokens as *mut u32 as *mut c_void);
    params.push(&mut side as *mut u32 as *mut c_void);
    params.push(&mut top_k as *mut u32 as *mut c_void);
    params.push(&mut beam as *mut u32 as *mut c_void);
    let block = 128u32;
    let grid = tokens.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "memory_product_key_candidates_from_scores",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: dims.tokens * dims.top_k,
        },
        &mut params,
    )
}

fn launch_memory_product_key_split_rows_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    len: usize,
    side: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut side = u32_kernel_dim(side, "side")?;
    let mut len_u32 = u32_kernel_dim(len, "len")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut side as *mut u32 as *mut c_void);
    params.push(&mut len_u32 as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = len_u32.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "memory_product_key_split_rows",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: len,
        },
        &mut params,
    )
}

fn launch_memory_product_key_selected_score_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    dims: MemoryProductKeySelectedScoreDims,
    elements: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut tokens = u32_kernel_dim(dims.tokens, "tokens")?;
    let mut top_k = u32_kernel_dim(dims.top_k, "top_k")?;
    let mut side = u32_kernel_dim(dims.side, "side")?;
    let mut key_dim = u32_kernel_dim(dims.key_dim, "key_dim")?;
    let mut total = u32_kernel_dim(elements, "total")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut tokens as *mut u32 as *mut c_void);
    params.push(&mut top_k as *mut u32 as *mut c_void);
    params.push(&mut side as *mut u32 as *mut c_void);
    params.push(&mut key_dim as *mut u32 as *mut c_void);
    params.push(&mut total as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = total.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "memory_product_key_selected_score",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements,
        },
        &mut params,
    )
}

fn launch_memory_product_key_selected_score_half_key_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    dims: MemoryProductKeySelectedScoreDims,
    elements: usize,
    left: bool,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut tokens = u32_kernel_dim(dims.tokens, "tokens")?;
    let mut top_k = u32_kernel_dim(dims.top_k, "top_k")?;
    let mut side = u32_kernel_dim(dims.side, "side")?;
    let mut key_dim = u32_kernel_dim(dims.key_dim, "key_dim")?;
    let mut is_left = u32::from(left);
    let mut total = u32_kernel_dim(elements, "total")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut tokens as *mut u32 as *mut c_void);
    params.push(&mut top_k as *mut u32 as *mut c_void);
    params.push(&mut side as *mut u32 as *mut c_void);
    params.push(&mut key_dim as *mut u32 as *mut c_void);
    params.push(&mut is_left as *mut u32 as *mut c_void);
    params.push(&mut total as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = total.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "memory_product_key_selected_score_half_key",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements,
        },
        &mut params,
    )
}

fn launch_rank2_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    rows: usize,
    cols: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut rows = rows as u32;
    let mut cols = cols as u32;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut rows as *mut u32 as *mut c_void);
    params.push(&mut cols as *mut u32 as *mut c_void);
    let len = rows as usize * cols as usize;
    let block = 256u32;
    let grid = (len as u32).div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "rank2_elementwise",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: len,
        },
        &mut params,
    )
}

fn launch_memory_weighted_value_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    dims: MemoryWeightedValueDims,
    elements: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut tokens = dims.tokens as u32;
    let mut top_k = dims.top_k as u32;
    let mut slots = dims.slots as u32;
    let mut value_dim = dims.value_dim as u32;
    let mut total = elements as u32;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut tokens as *mut u32 as *mut c_void);
    params.push(&mut top_k as *mut u32 as *mut c_void);
    params.push(&mut slots as *mut u32 as *mut c_void);
    params.push(&mut value_dim as *mut u32 as *mut c_void);
    params.push(&mut total as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = (elements as u32).div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "memory_weighted_value",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements,
        },
        &mut params,
    )
}

fn launch_memory_selected_score_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    dims: MemorySelectedScoreDims,
    elements: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut tokens = dims.tokens as u32;
    let mut top_k = dims.top_k as u32;
    let mut slots = dims.slots as u32;
    let mut key_dim = dims.key_dim as u32;
    let mut total = elements as u32;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut tokens as *mut u32 as *mut c_void);
    params.push(&mut top_k as *mut u32 as *mut c_void);
    params.push(&mut slots as *mut u32 as *mut c_void);
    params.push(&mut key_dim as *mut u32 as *mut c_void);
    params.push(&mut total as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = (elements as u32).div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "memory_selected_score",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements,
        },
        &mut params,
    )
}

fn launch_bias_2d_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    rows: usize,
    cols: usize,
    total: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut rows = u32_kernel_dim(rows, "rows")?;
    let mut cols = u32_kernel_dim(cols, "cols")?;
    let mut total = u32_kernel_dim(total, "total")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut rows as *mut u32 as *mut c_void);
    params.push(&mut cols as *mut u32 as *mut c_void);
    params.push(&mut total as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = total.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "bias_2d",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: total as usize,
        },
        &mut params,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_transpose2d_pair_bf16_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    first_rows: usize,
    first_cols: usize,
    first_total: usize,
    second_rows: usize,
    second_cols: usize,
    second_total: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut first_rows = u32_kernel_dim(first_rows, "first_rows")?;
    let mut first_cols = u32_kernel_dim(first_cols, "first_cols")?;
    let mut first_total = u32_kernel_dim(first_total, "first_total")?;
    let mut second_rows = u32_kernel_dim(second_rows, "second_rows")?;
    let mut second_cols = u32_kernel_dim(second_cols, "second_cols")?;
    let mut second_total = u32_kernel_dim(second_total, "second_total")?;
    let total = (first_total as usize)
        .checked_add(second_total as usize)
        .ok_or_else(|| CudaError::new("transpose2d_pair_bf16 element count overflow"))?;
    let mut launch_total = first_total.max(second_total);
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut first_rows as *mut u32 as *mut c_void);
    params.push(&mut first_cols as *mut u32 as *mut c_void);
    params.push(&mut first_total as *mut u32 as *mut c_void);
    params.push(&mut second_rows as *mut u32 as *mut c_void);
    params.push(&mut second_cols as *mut u32 as *mut c_void);
    params.push(&mut second_total as *mut u32 as *mut c_void);
    params.push(&mut launch_total as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = launch_total.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "transpose2d_pair_bf16",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: total,
        },
        &mut params,
    )
}

fn launch_matmul_strided_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    dims: MatmulStridedDims,
    total: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut m = u32_kernel_dim(dims.left.rows, "m")?;
    let mut k = u32_kernel_dim(dims.left.cols, "k")?;
    let mut n = u32_kernel_dim(dims.right.cols, "n")?;
    let mut left_row_stride = u32_kernel_dim(dims.left.row_stride, "left_row_stride")?;
    let mut left_col_stride = u32_kernel_dim(dims.left.col_stride, "left_col_stride")?;
    let mut left_offset = u32_kernel_dim(dims.left.offset, "left_offset")?;
    let mut right_row_stride = u32_kernel_dim(dims.right.row_stride, "right_row_stride")?;
    let mut right_col_stride = u32_kernel_dim(dims.right.col_stride, "right_col_stride")?;
    let mut right_offset = u32_kernel_dim(dims.right.offset, "right_offset")?;
    let mut total = u32_kernel_dim(total, "total")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut m as *mut u32 as *mut c_void);
    params.push(&mut k as *mut u32 as *mut c_void);
    params.push(&mut n as *mut u32 as *mut c_void);
    params.push(&mut left_row_stride as *mut u32 as *mut c_void);
    params.push(&mut left_col_stride as *mut u32 as *mut c_void);
    params.push(&mut left_offset as *mut u32 as *mut c_void);
    params.push(&mut right_row_stride as *mut u32 as *mut c_void);
    params.push(&mut right_col_stride as *mut u32 as *mut c_void);
    params.push(&mut right_offset as *mut u32 as *mut c_void);
    params.push(&mut total as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = total.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "matmul_strided_f32_reference",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: total as usize,
        },
        &mut params,
    )
}

fn launch_embedding_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    index_count: usize,
    vocab_size: usize,
    embedding_dim: usize,
    total: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut index_count = u32_kernel_dim(index_count, "index_count")?;
    let mut vocab_size = u32_kernel_dim(vocab_size, "vocab_size")?;
    let mut embedding_dim = u32_kernel_dim(embedding_dim, "embedding_dim")?;
    let mut total = u32_kernel_dim(total, "total")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut index_count as *mut u32 as *mut c_void);
    params.push(&mut vocab_size as *mut u32 as *mut c_void);
    params.push(&mut embedding_dim as *mut u32 as *mut c_void);
    params.push(&mut total as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = total.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "embedding",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: total as usize,
        },
        &mut params,
    )
}

fn launch_layer_norm_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    rows: usize,
    features: usize,
    eps: f32,
    total: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut rows = u32_kernel_dim(rows, "rows")?;
    let mut features = u32_kernel_dim(features, "features")?;
    let mut eps = eps;
    let mut total = u32_kernel_dim(total, "total")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut rows as *mut u32 as *mut c_void);
    params.push(&mut features as *mut u32 as *mut c_void);
    params.push(&mut eps as *mut f32 as *mut c_void);
    params.push(&mut total as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = total.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "layer_norm",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: total as usize,
        },
        &mut params,
    )
}

fn launch_cross_entropy_forward_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    batch: usize,
    classes: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut batch = u32_kernel_dim(batch, "batch")?;
    let mut classes = u32_kernel_dim(classes, "classes")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut batch as *mut u32 as *mut c_void);
    params.push(&mut classes as *mut u32 as *mut c_void);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "cross_entropy_forward",
            grid: (1, 1, 1),
            block: (1, 1, 1),
            shared_mem_bytes: 0,
            elements: batch as usize * classes as usize,
        },
        &mut params,
    )
}

fn launch_cross_entropy_backward_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    batch: usize,
    classes: usize,
    total: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut batch = u32_kernel_dim(batch, "batch")?;
    let mut classes = u32_kernel_dim(classes, "classes")?;
    let mut total = u32_kernel_dim(total, "total")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut batch as *mut u32 as *mut c_void);
    params.push(&mut classes as *mut u32 as *mut c_void);
    params.push(&mut total as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = total.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "cross_entropy_backward",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: total as usize,
        },
        &mut params,
    )
}

fn launch_causal_attention_kernel(
    driver: &CudaDriver,
    function: CUfunction,
    device_ptrs: &[CUdeviceptr],
    dims: CausalAttentionDims,
    head_dim: usize,
    total: usize,
) -> CudaResult<()> {
    let mut ptr_args = device_ptrs.to_vec();
    let mut batch = u32_kernel_dim(dims.batch, "batch")?;
    let mut time = u32_kernel_dim(dims.time, "time")?;
    let mut channels = u32_kernel_dim(dims.channels, "channels")?;
    let mut n_heads = u32_kernel_dim(dims.n_heads, "n_heads")?;
    let mut head_dim = u32_kernel_dim(head_dim, "head_dim")?;
    let mut total = u32_kernel_dim(total, "total")?;
    let mut params = ptr_args
        .iter_mut()
        .map(|ptr| ptr as *mut CUdeviceptr as *mut c_void)
        .collect::<Vec<_>>();
    params.push(&mut batch as *mut u32 as *mut c_void);
    params.push(&mut time as *mut u32 as *mut c_void);
    params.push(&mut channels as *mut u32 as *mut c_void);
    params.push(&mut n_heads as *mut u32 as *mut c_void);
    params.push(&mut head_dim as *mut u32 as *mut c_void);
    params.push(&mut total as *mut u32 as *mut c_void);
    let block = 256u32;
    let grid = total.div_ceil(block).max(1);
    launch_kernel_on_current_stream(
        driver,
        function,
        KernelLaunchConfig {
            label: "causal_attention_reference",
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
            elements: total as usize,
        },
        &mut params,
    )
}

fn u32_kernel_dim(value: usize, name: &str) -> CudaResult<u32> {
    u32::try_from(value).map_err(|_| {
        CudaError::new(format!(
            "CUDA kernel dimension {name}={value} exceeds u32 limit"
        ))
    })
}

fn check(result: CUresult, name: &str) -> CudaResult<()> {
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CudaError::new(format!(
            "{name} returned CUDA error {result}"
        )))
    }
}

#[cfg(unix)]
unsafe fn load_symbol<T: Copy>(handle: *mut c_void, name: &str) -> CudaResult<T> {
    let symbol_name = CString::new(name).expect("static symbol name");
    let symbol = dlsym(handle, symbol_name.as_ptr());
    if symbol.is_null() {
        return Err(CudaError::new(format!(
            "missing CUDA driver symbol {name}: {}",
            dl_error_string()
        )));
    }
    Ok(std::mem::transmute_copy::<*mut c_void, T>(&symbol))
}

#[cfg(unix)]
fn dl_error_string() -> String {
    // SAFETY: dlerror returns either null or a valid thread-local C string.
    let error = unsafe { dlerror() };
    if error.is_null() {
        "unknown dlerror".to_string()
    } else {
        // SAFETY: non-null dlerror pointer is a valid null-terminated C string.
        unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned()
    }
}

const TENSOR_CORE_PTX: &str = r#"
.version 7.8
.target sm_80
.address_size 64

.visible .entry heirloom_bf16_mma_probe(
    .param .u64 output
)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    .reg .f32 %f<9>;

    ld.param.u64 %rd1, [output];
    mov.u32 %r1, %tid.x;

    // Packed BF16 1.0 values: high 16 bits = 0x3f80, low 16 bits = 0x3f80.
    mov.u32 %r2, 1065369472;
    mov.u32 %r3, 1065369472;
    mov.u32 %r4, 1065369472;
    mov.u32 %r5, 1065369472;
    mov.u32 %r6, 1065369472;
    mov.u32 %r7, 1065369472;

    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.f32 %f4, 0f00000000;

    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f5, %f6, %f7, %f8},
        {%r2, %r3, %r4, %r5},
        {%r6, %r7},
        {%f1, %f2, %f3, %f4};

    mul.wide.u32 %rd2, %r1, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.f32 [%rd3], %f5;
    ret;
}

.visible .entry heirloom_matmul_bf16_mma_rhs_t_f32(
    .param .u64 left,
    .param .u64 right,
    .param .u64 output,
    .param .u32 m,
    .param .u32 k,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<80>;
    .reg .b64 %rd<80>;
    .reg .f32 %f<9>;

    ld.param.u64 %rd1, [left];
    ld.param.u64 %rd2, [right];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [m];
    ld.param.u32 %r5, [k];
    ld.param.u32 %r6, [n];

    mov.u32 %r10, %tid.x;
    mov.u32 %r11, %ctaid.x;
    mov.u32 %r12, %ctaid.y;
    shr.u32 %r13, %r10, 2;
    and.b32 %r14, %r10, 3;
    shl.b32 %r15, %r12, 4;
    shl.b32 %r16, %r11, 3;
    shl.b32 %r18, %r14, 1;

    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.f32 %f4, 0f00000000;
    mov.u32 %r17, 0;

MMA_RHS_T_K_LOOP:
    setp.ge.u32 %p1, %r17, %r5;
    @%p1 bra MMA_RHS_T_STORE;

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r17, %r18;
    mad.lo.u32 %r21, %r19, %r5, %r20;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u16 %r22, [%rd5];
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r19, %r5, %r23;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u16 %r24, [%rd5];
    shl.b32 %r25, %r24, 16;
    or.b32 %r30, %r22, %r25;

    add.u32 %r19, %r19, 8;
    mad.lo.u32 %r21, %r19, %r5, %r20;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u16 %r22, [%rd5];
    mad.lo.u32 %r21, %r19, %r5, %r23;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u16 %r24, [%rd5];
    shl.b32 %r25, %r24, 16;
    or.b32 %r31, %r22, %r25;

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r19, %r5, %r20;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u16 %r22, [%rd5];
    mad.lo.u32 %r21, %r19, %r5, %r23;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u16 %r24, [%rd5];
    shl.b32 %r25, %r24, 16;
    or.b32 %r32, %r22, %r25;

    add.u32 %r19, %r19, 8;
    mad.lo.u32 %r21, %r19, %r5, %r20;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u16 %r22, [%rd5];
    mad.lo.u32 %r21, %r19, %r5, %r23;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u16 %r24, [%rd5];
    shl.b32 %r25, %r24, 16;
    or.b32 %r33, %r22, %r25;

    add.u32 %r26, %r16, %r13;
    add.u32 %r20, %r17, %r18;
    mad.lo.u32 %r21, %r26, %r5, %r20;
    mul.wide.u32 %rd6, %r21, 2;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.u16 %r22, [%rd7];
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r26, %r5, %r23;
    mul.wide.u32 %rd6, %r21, 2;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.u16 %r24, [%rd7];
    shl.b32 %r25, %r24, 16;
    or.b32 %r34, %r22, %r25;

    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r26, %r5, %r20;
    mul.wide.u32 %rd6, %r21, 2;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.u16 %r22, [%rd7];
    mad.lo.u32 %r21, %r26, %r5, %r23;
    mul.wide.u32 %rd6, %r21, 2;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.u16 %r24, [%rd7];
    shl.b32 %r25, %r24, 16;
    or.b32 %r35, %r22, %r25;

    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f5, %f6, %f7, %f8},
        {%r30, %r31, %r32, %r33},
        {%r34, %r35},
        {%f1, %f2, %f3, %f4};
    mov.f32 %f1, %f5;
    mov.f32 %f2, %f6;
    mov.f32 %f3, %f7;
    mov.f32 %f4, %f8;

    add.u32 %r17, %r17, 16;
    bra MMA_RHS_T_K_LOOP;

MMA_RHS_T_STORE:
    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r16, %r18;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd8, %r21, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f1;

    add.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd8, %r21, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f2;

    add.u32 %r19, %r19, 8;
    sub.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd8, %r21, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f3;

    add.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd8, %r21, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f4;
    ret;
}

.visible .entry heirloom_matmul_bf16_mma_rhs_t_f32_cta(
    .param .u64 left,
    .param .u64 right,
    .param .u64 output,
    .param .u32 m,
    .param .u32 k,
    .param .u32 n
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<90>;
    .reg .b64 %rd<80>;
    .reg .f32 %f<9>;

    ld.param.u64 %rd1, [left];
    ld.param.u64 %rd2, [right];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [m];
    ld.param.u32 %r5, [k];
    ld.param.u32 %r6, [n];

    mov.u32 %r10, %tid.x;
    mov.u32 %r11, %ctaid.x;
    mov.u32 %r12, %ctaid.y;
    shr.u32 %r70, %r10, 5;
    and.b32 %r71, %r10, 31;
    shr.u32 %r72, %r70, 1;
    and.b32 %r73, %r70, 1;
    shr.u32 %r13, %r71, 2;
    and.b32 %r14, %r71, 3;
    shl.b32 %r15, %r12, 5;
    shl.b32 %r74, %r72, 4;
    add.u32 %r15, %r15, %r74;
    shl.b32 %r16, %r11, 4;
    shl.b32 %r75, %r73, 3;
    add.u32 %r16, %r16, %r75;
    setp.ge.u32 %p2, %r15, %r4;
    @%p2 bra MMA_RHS_T_CTA_DONE;
    setp.ge.u32 %p3, %r16, %r6;
    @%p3 bra MMA_RHS_T_CTA_DONE;
    shl.b32 %r18, %r14, 1;

    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.f32 %f4, 0f00000000;
    mov.u32 %r17, 0;

MMA_RHS_T_CTA_K_LOOP:
    setp.ge.u32 %p1, %r17, %r5;
    @%p1 bra MMA_RHS_T_CTA_STORE;

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r17, %r18;
    mad.lo.u32 %r21, %r19, %r5, %r20;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u16 %r22, [%rd5];
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r19, %r5, %r23;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u16 %r24, [%rd5];
    shl.b32 %r25, %r24, 16;
    or.b32 %r30, %r22, %r25;

    add.u32 %r19, %r19, 8;
    mad.lo.u32 %r21, %r19, %r5, %r20;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u16 %r22, [%rd5];
    mad.lo.u32 %r21, %r19, %r5, %r23;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u16 %r24, [%rd5];
    shl.b32 %r25, %r24, 16;
    or.b32 %r31, %r22, %r25;

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r19, %r5, %r20;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u16 %r22, [%rd5];
    mad.lo.u32 %r21, %r19, %r5, %r23;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u16 %r24, [%rd5];
    shl.b32 %r25, %r24, 16;
    or.b32 %r32, %r22, %r25;

    add.u32 %r19, %r19, 8;
    mad.lo.u32 %r21, %r19, %r5, %r20;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u16 %r22, [%rd5];
    mad.lo.u32 %r21, %r19, %r5, %r23;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u16 %r24, [%rd5];
    shl.b32 %r25, %r24, 16;
    or.b32 %r33, %r22, %r25;

    add.u32 %r26, %r16, %r13;
    add.u32 %r20, %r17, %r18;
    mad.lo.u32 %r21, %r26, %r5, %r20;
    mul.wide.u32 %rd6, %r21, 2;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.u16 %r22, [%rd7];
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r26, %r5, %r23;
    mul.wide.u32 %rd6, %r21, 2;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.u16 %r24, [%rd7];
    shl.b32 %r25, %r24, 16;
    or.b32 %r34, %r22, %r25;

    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r26, %r5, %r20;
    mul.wide.u32 %rd6, %r21, 2;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.u16 %r22, [%rd7];
    mad.lo.u32 %r21, %r26, %r5, %r23;
    mul.wide.u32 %rd6, %r21, 2;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.u16 %r24, [%rd7];
    shl.b32 %r25, %r24, 16;
    or.b32 %r35, %r22, %r25;

    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f5, %f6, %f7, %f8},
        {%r30, %r31, %r32, %r33},
        {%r34, %r35},
        {%f1, %f2, %f3, %f4};
    mov.f32 %f1, %f5;
    mov.f32 %f2, %f6;
    mov.f32 %f3, %f7;
    mov.f32 %f4, %f8;

    add.u32 %r17, %r17, 16;
    bra MMA_RHS_T_CTA_K_LOOP;

MMA_RHS_T_CTA_STORE:
    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r16, %r18;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd8, %r21, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f1;

    add.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd8, %r21, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f2;

    add.u32 %r19, %r19, 8;
    sub.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd8, %r21, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f3;

    add.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd8, %r21, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f4;
MMA_RHS_T_CTA_DONE:
    ret;
}

.visible .entry heirloom_matmul_bf16_mma_rhs_t_f32_cta_staged(
    .param .u64 left,
    .param .u64 right,
    .param .u64 output,
    .param .u32 m,
    .param .u32 k,
    .param .u32 n
)
{
    .reg .pred %p<10>;
    .reg .b32 %r<130>;
    .reg .b64 %rd<100>;
    .reg .f32 %f<9>;
    .shared .align 4 .b8 heirloom_matmul_cta_smem_a[1024];
    .shared .align 4 .b8 heirloom_matmul_cta_smem_b[512];

    ld.param.u64 %rd1, [left];
    ld.param.u64 %rd2, [right];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [m];
    ld.param.u32 %r5, [k];
    ld.param.u32 %r6, [n];

    mov.u32 %r10, %tid.x;
    mov.u32 %r11, %ctaid.x;
    mov.u32 %r12, %ctaid.y;
    shr.u32 %r70, %r10, 5;
    and.b32 %r71, %r10, 31;
    shr.u32 %r72, %r70, 1;
    and.b32 %r73, %r70, 1;
    shr.u32 %r13, %r71, 2;
    and.b32 %r14, %r71, 3;
    shl.b32 %r80, %r12, 5;
    shl.b32 %r81, %r11, 4;
    shl.b32 %r74, %r72, 4;
    shl.b32 %r75, %r73, 3;
    add.u32 %r15, %r80, %r74;
    add.u32 %r16, %r81, %r75;
    shl.b32 %r18, %r14, 1;
    mov.u32 %r89, 16;

    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.f32 %f4, 0f00000000;
    mov.u32 %r17, 0;

MMA_RHS_T_CTA_STAGED_K_LOOP:
    setp.ge.u32 %p1, %r17, %r5;
    @%p1 bra MMA_RHS_T_CTA_STAGED_STORE;

    mov.u32 %r90, %r10;
MMA_RHS_T_CTA_STAGE_A_LOOP:
    setp.ge.u32 %p4, %r90, 512;
    @%p4 bra MMA_RHS_T_CTA_STAGE_A_DONE;
    div.u32 %r91, %r90, %r89;
    rem.u32 %r92, %r90, %r89;
    add.u32 %r93, %r80, %r91;
    add.u32 %r94, %r17, %r92;
    setp.ge.u32 %p5, %r93, %r4;
    setp.ge.u32 %p6, %r94, %r5;
    or.pred %p7, %p5, %p6;
    mov.u32 %r95, 0;
    @%p7 bra MMA_RHS_T_CTA_STAGE_A_STORE;
    mad.lo.u32 %r96, %r93, %r5, %r94;
    mul.wide.u32 %rd20, %r96, 2;
    add.s64 %rd21, %rd1, %rd20;
    ld.global.u16 %r95, [%rd21];
MMA_RHS_T_CTA_STAGE_A_STORE:
    mul.wide.u32 %rd22, %r90, 2;
    mov.u64 %rd23, heirloom_matmul_cta_smem_a;
    add.s64 %rd24, %rd23, %rd22;
    st.shared.u16 [%rd24], %r95;
    add.u32 %r90, %r90, 128;
    bra MMA_RHS_T_CTA_STAGE_A_LOOP;
MMA_RHS_T_CTA_STAGE_A_DONE:

    mov.u32 %r90, %r10;
MMA_RHS_T_CTA_STAGE_B_LOOP:
    setp.ge.u32 %p4, %r90, 256;
    @%p4 bra MMA_RHS_T_CTA_STAGE_B_DONE;
    div.u32 %r91, %r90, %r89;
    rem.u32 %r92, %r90, %r89;
    add.u32 %r93, %r81, %r91;
    add.u32 %r94, %r17, %r92;
    setp.ge.u32 %p5, %r93, %r6;
    setp.ge.u32 %p6, %r94, %r5;
    or.pred %p7, %p5, %p6;
    mov.u32 %r95, 0;
    @%p7 bra MMA_RHS_T_CTA_STAGE_B_STORE;
    mad.lo.u32 %r96, %r93, %r5, %r94;
    mul.wide.u32 %rd20, %r96, 2;
    add.s64 %rd21, %rd2, %rd20;
    ld.global.u16 %r95, [%rd21];
MMA_RHS_T_CTA_STAGE_B_STORE:
    mul.wide.u32 %rd22, %r90, 2;
    mov.u64 %rd23, heirloom_matmul_cta_smem_b;
    add.s64 %rd24, %rd23, %rd22;
    st.shared.u16 [%rd24], %r95;
    add.u32 %r90, %r90, 128;
    bra MMA_RHS_T_CTA_STAGE_B_LOOP;
MMA_RHS_T_CTA_STAGE_B_DONE:

    bar.sync 0;

    add.u32 %r19, %r74, %r13;
    add.u32 %r20, %r18, 0;
    mad.lo.u32 %r21, %r19, %r89, %r20;
    mul.wide.u32 %rd4, %r21, 2;
    mov.u64 %rd5, heirloom_matmul_cta_smem_a;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r22, [%rd6];
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r19, %r89, %r23;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r24, [%rd6];
    shl.b32 %r25, %r24, 16;
    or.b32 %r30, %r22, %r25;

    add.u32 %r19, %r19, 8;
    mad.lo.u32 %r21, %r19, %r89, %r20;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r22, [%rd6];
    mad.lo.u32 %r21, %r19, %r89, %r23;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r24, [%rd6];
    shl.b32 %r25, %r24, 16;
    or.b32 %r31, %r22, %r25;

    add.u32 %r19, %r74, %r13;
    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r19, %r89, %r20;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r22, [%rd6];
    mad.lo.u32 %r21, %r19, %r89, %r23;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r24, [%rd6];
    shl.b32 %r25, %r24, 16;
    or.b32 %r32, %r22, %r25;

    add.u32 %r19, %r19, 8;
    mad.lo.u32 %r21, %r19, %r89, %r20;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r22, [%rd6];
    mad.lo.u32 %r21, %r19, %r89, %r23;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r24, [%rd6];
    shl.b32 %r25, %r24, 16;
    or.b32 %r33, %r22, %r25;

    add.u32 %r26, %r75, %r13;
    add.u32 %r20, %r18, 0;
    mad.lo.u32 %r21, %r26, %r89, %r20;
    mul.wide.u32 %rd7, %r21, 2;
    mov.u64 %rd8, heirloom_matmul_cta_smem_b;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r22, [%rd9];
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r26, %r89, %r23;
    mul.wide.u32 %rd7, %r21, 2;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r24, [%rd9];
    shl.b32 %r25, %r24, 16;
    or.b32 %r34, %r22, %r25;

    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r26, %r89, %r20;
    mul.wide.u32 %rd7, %r21, 2;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r22, [%rd9];
    mad.lo.u32 %r21, %r26, %r89, %r23;
    mul.wide.u32 %rd7, %r21, 2;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r24, [%rd9];
    shl.b32 %r25, %r24, 16;
    or.b32 %r35, %r22, %r25;

    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f5, %f6, %f7, %f8},
        {%r30, %r31, %r32, %r33},
        {%r34, %r35},
        {%f1, %f2, %f3, %f4};
    mov.f32 %f1, %f5;
    mov.f32 %f2, %f6;
    mov.f32 %f3, %f7;
    mov.f32 %f4, %f8;

    bar.sync 0;
    add.u32 %r17, %r17, 16;
    bra MMA_RHS_T_CTA_STAGED_K_LOOP;

MMA_RHS_T_CTA_STAGED_STORE:
    setp.ge.u32 %p2, %r15, %r4;
    setp.ge.u32 %p3, %r16, %r6;
    or.pred %p4, %p2, %p3;
    @%p4 bra MMA_RHS_T_CTA_STAGED_DONE;

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r16, %r18;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f1;

    add.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f2;

    add.u32 %r19, %r19, 8;
    sub.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f3;

    add.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f4;
MMA_RHS_T_CTA_STAGED_DONE:
    ret;
}

.visible .entry heirloom_matmul_bf16_mma_normal_rhs_f32_cta_staged(
    .param .u64 left,
    .param .u64 right,
    .param .u64 output,
    .param .u32 m,
    .param .u32 k,
    .param .u32 n
)
{
    .reg .pred %p<10>;
    .reg .b32 %r<130>;
    .reg .b64 %rd<100>;
    .reg .f32 %f<9>;
    .shared .align 4 .b8 heirloom_matmul_normal_rhs_smem_a[1024];
    .shared .align 4 .b8 heirloom_matmul_normal_rhs_smem_b[512];

    ld.param.u64 %rd1, [left];
    ld.param.u64 %rd2, [right];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [m];
    ld.param.u32 %r5, [k];
    ld.param.u32 %r6, [n];

    mov.u32 %r10, %tid.x;
    mov.u32 %r11, %ctaid.x;
    mov.u32 %r12, %ctaid.y;
    shr.u32 %r70, %r10, 5;
    and.b32 %r71, %r10, 31;
    shr.u32 %r72, %r70, 1;
    and.b32 %r73, %r70, 1;
    shr.u32 %r13, %r71, 2;
    and.b32 %r14, %r71, 3;
    shl.b32 %r80, %r12, 5;
    shl.b32 %r81, %r11, 4;
    shl.b32 %r74, %r72, 4;
    shl.b32 %r75, %r73, 3;
    add.u32 %r15, %r80, %r74;
    add.u32 %r16, %r81, %r75;
    shl.b32 %r18, %r14, 1;
    mov.u32 %r89, 16;

    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.f32 %f4, 0f00000000;
    mov.u32 %r17, 0;

MMA_NORMAL_RHS_CTA_STAGED_K_LOOP:
    setp.ge.u32 %p1, %r17, %r5;
    @%p1 bra MMA_NORMAL_RHS_CTA_STAGED_STORE;

    mov.u32 %r90, %r10;
MMA_NORMAL_RHS_CTA_STAGE_A_LOOP:
    setp.ge.u32 %p4, %r90, 512;
    @%p4 bra MMA_NORMAL_RHS_CTA_STAGE_A_DONE;
    div.u32 %r91, %r90, %r89;
    rem.u32 %r92, %r90, %r89;
    add.u32 %r93, %r80, %r91;
    add.u32 %r94, %r17, %r92;
    setp.ge.u32 %p5, %r93, %r4;
    setp.ge.u32 %p6, %r94, %r5;
    or.pred %p7, %p5, %p6;
    mov.u32 %r95, 0;
    @%p7 bra MMA_NORMAL_RHS_CTA_STAGE_A_STORE;
    mad.lo.u32 %r96, %r93, %r5, %r94;
    mul.wide.u32 %rd20, %r96, 2;
    add.s64 %rd21, %rd1, %rd20;
    ld.global.u16 %r95, [%rd21];
MMA_NORMAL_RHS_CTA_STAGE_A_STORE:
    mul.wide.u32 %rd22, %r90, 2;
    mov.u64 %rd23, heirloom_matmul_normal_rhs_smem_a;
    add.s64 %rd24, %rd23, %rd22;
    st.shared.u16 [%rd24], %r95;
    add.u32 %r90, %r90, 128;
    bra MMA_NORMAL_RHS_CTA_STAGE_A_LOOP;
MMA_NORMAL_RHS_CTA_STAGE_A_DONE:

    mov.u32 %r90, %r10;
MMA_NORMAL_RHS_CTA_STAGE_B_LOOP:
    setp.ge.u32 %p4, %r90, 256;
    @%p4 bra MMA_NORMAL_RHS_CTA_STAGE_B_DONE;
    div.u32 %r91, %r90, %r89;
    rem.u32 %r92, %r90, %r89;
    add.u32 %r93, %r81, %r91;
    add.u32 %r94, %r17, %r92;
    setp.ge.u32 %p5, %r93, %r6;
    setp.ge.u32 %p6, %r94, %r5;
    or.pred %p7, %p5, %p6;
    mov.u32 %r95, 0;
    @%p7 bra MMA_NORMAL_RHS_CTA_STAGE_B_STORE;
    mad.lo.u32 %r96, %r94, %r6, %r93;
    mul.wide.u32 %rd20, %r96, 2;
    add.s64 %rd21, %rd2, %rd20;
    ld.global.u16 %r95, [%rd21];
MMA_NORMAL_RHS_CTA_STAGE_B_STORE:
    mul.wide.u32 %rd22, %r90, 2;
    mov.u64 %rd23, heirloom_matmul_normal_rhs_smem_b;
    add.s64 %rd24, %rd23, %rd22;
    st.shared.u16 [%rd24], %r95;
    add.u32 %r90, %r90, 128;
    bra MMA_NORMAL_RHS_CTA_STAGE_B_LOOP;
MMA_NORMAL_RHS_CTA_STAGE_B_DONE:

    bar.sync 0;

    add.u32 %r19, %r74, %r13;
    add.u32 %r20, %r18, 0;
    mad.lo.u32 %r21, %r19, %r89, %r20;
    mul.wide.u32 %rd4, %r21, 2;
    mov.u64 %rd5, heirloom_matmul_normal_rhs_smem_a;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r22, [%rd6];
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r19, %r89, %r23;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r24, [%rd6];
    shl.b32 %r25, %r24, 16;
    or.b32 %r30, %r22, %r25;

    add.u32 %r19, %r19, 8;
    mad.lo.u32 %r21, %r19, %r89, %r20;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r22, [%rd6];
    mad.lo.u32 %r21, %r19, %r89, %r23;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r24, [%rd6];
    shl.b32 %r25, %r24, 16;
    or.b32 %r31, %r22, %r25;

    add.u32 %r19, %r74, %r13;
    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r19, %r89, %r20;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r22, [%rd6];
    mad.lo.u32 %r21, %r19, %r89, %r23;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r24, [%rd6];
    shl.b32 %r25, %r24, 16;
    or.b32 %r32, %r22, %r25;

    add.u32 %r19, %r19, 8;
    mad.lo.u32 %r21, %r19, %r89, %r20;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r22, [%rd6];
    mad.lo.u32 %r21, %r19, %r89, %r23;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r24, [%rd6];
    shl.b32 %r25, %r24, 16;
    or.b32 %r33, %r22, %r25;

    add.u32 %r26, %r75, %r13;
    add.u32 %r20, %r18, 0;
    mad.lo.u32 %r21, %r26, %r89, %r20;
    mul.wide.u32 %rd7, %r21, 2;
    mov.u64 %rd8, heirloom_matmul_normal_rhs_smem_b;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r22, [%rd9];
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r26, %r89, %r23;
    mul.wide.u32 %rd7, %r21, 2;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r24, [%rd9];
    shl.b32 %r25, %r24, 16;
    or.b32 %r34, %r22, %r25;

    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r26, %r89, %r20;
    mul.wide.u32 %rd7, %r21, 2;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r22, [%rd9];
    mad.lo.u32 %r21, %r26, %r89, %r23;
    mul.wide.u32 %rd7, %r21, 2;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r24, [%rd9];
    shl.b32 %r25, %r24, 16;
    or.b32 %r35, %r22, %r25;

    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f5, %f6, %f7, %f8},
        {%r30, %r31, %r32, %r33},
        {%r34, %r35},
        {%f1, %f2, %f3, %f4};
    mov.f32 %f1, %f5;
    mov.f32 %f2, %f6;
    mov.f32 %f3, %f7;
    mov.f32 %f4, %f8;

    bar.sync 0;
    add.u32 %r17, %r17, 16;
    bra MMA_NORMAL_RHS_CTA_STAGED_K_LOOP;

MMA_NORMAL_RHS_CTA_STAGED_STORE:
    setp.ge.u32 %p2, %r15, %r4;
    setp.ge.u32 %p3, %r16, %r6;
    or.pred %p4, %p2, %p3;
    @%p4 bra MMA_NORMAL_RHS_CTA_STAGED_DONE;

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r16, %r18;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f1;

    add.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f2;

    add.u32 %r19, %r19, 8;
    sub.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f3;

    add.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f4;
MMA_NORMAL_RHS_CTA_STAGED_DONE:
    ret;
}

.visible .entry heirloom_matmul_bf16_mma_rhs_t_f32_cta_ldmatrix_a(
    .param .u64 left,
    .param .u64 right,
    .param .u64 output,
    .param .u32 m,
    .param .u32 k,
    .param .u32 n
)
{
    .reg .pred %p<10>;
    .reg .b32 %r<130>;
    .reg .b64 %rd<100>;
    .reg .f32 %f<9>;
    .shared .align 16 .b8 heirloom_matmul_ldmatrix_smem_a[1024];
    .shared .align 4 .b8 heirloom_matmul_ldmatrix_smem_b[512];

    ld.param.u64 %rd1, [left];
    ld.param.u64 %rd2, [right];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [m];
    ld.param.u32 %r5, [k];
    ld.param.u32 %r6, [n];

    mov.u32 %r10, %tid.x;
    mov.u32 %r11, %ctaid.x;
    mov.u32 %r12, %ctaid.y;
    shr.u32 %r70, %r10, 5;
    and.b32 %r71, %r10, 31;
    shr.u32 %r72, %r70, 1;
    and.b32 %r73, %r70, 1;
    shr.u32 %r13, %r71, 2;
    and.b32 %r14, %r71, 3;
    shl.b32 %r80, %r12, 5;
    shl.b32 %r81, %r11, 4;
    shl.b32 %r74, %r72, 4;
    shl.b32 %r75, %r73, 3;
    add.u32 %r15, %r80, %r74;
    add.u32 %r16, %r81, %r75;
    shl.b32 %r18, %r14, 1;
    mov.u32 %r89, 16;

    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.f32 %f4, 0f00000000;
    mov.u32 %r17, 0;

MMA_RHS_T_CTA_LDMATRIX_A_K_LOOP:
    setp.ge.u32 %p1, %r17, %r5;
    @%p1 bra MMA_RHS_T_CTA_LDMATRIX_A_STORE;

    mov.u32 %r90, %r10;
MMA_RHS_T_CTA_LDMATRIX_A_STAGE_A_LOOP:
    setp.ge.u32 %p4, %r90, 512;
    @%p4 bra MMA_RHS_T_CTA_LDMATRIX_A_STAGE_A_DONE;
    div.u32 %r91, %r90, %r89;
    rem.u32 %r92, %r90, %r89;
    add.u32 %r93, %r80, %r91;
    add.u32 %r94, %r17, %r92;
    setp.ge.u32 %p5, %r93, %r4;
    setp.ge.u32 %p6, %r94, %r5;
    or.pred %p7, %p5, %p6;
    mov.u32 %r95, 0;
    @%p7 bra MMA_RHS_T_CTA_LDMATRIX_A_STAGE_A_STORE;
    mad.lo.u32 %r96, %r93, %r5, %r94;
    mul.wide.u32 %rd20, %r96, 2;
    add.s64 %rd21, %rd1, %rd20;
    ld.global.u16 %r95, [%rd21];
MMA_RHS_T_CTA_LDMATRIX_A_STAGE_A_STORE:
    mul.wide.u32 %rd22, %r90, 2;
    mov.u64 %rd23, heirloom_matmul_ldmatrix_smem_a;
    add.s64 %rd24, %rd23, %rd22;
    st.shared.u16 [%rd24], %r95;
    add.u32 %r90, %r90, 128;
    bra MMA_RHS_T_CTA_LDMATRIX_A_STAGE_A_LOOP;
MMA_RHS_T_CTA_LDMATRIX_A_STAGE_A_DONE:

    mov.u32 %r90, %r10;
MMA_RHS_T_CTA_LDMATRIX_A_STAGE_B_LOOP:
    setp.ge.u32 %p4, %r90, 256;
    @%p4 bra MMA_RHS_T_CTA_LDMATRIX_A_STAGE_B_DONE;
    div.u32 %r91, %r90, %r89;
    rem.u32 %r92, %r90, %r89;
    add.u32 %r93, %r81, %r91;
    add.u32 %r94, %r17, %r92;
    setp.ge.u32 %p5, %r93, %r6;
    setp.ge.u32 %p6, %r94, %r5;
    or.pred %p7, %p5, %p6;
    mov.u32 %r95, 0;
    @%p7 bra MMA_RHS_T_CTA_LDMATRIX_A_STAGE_B_STORE;
    mad.lo.u32 %r96, %r93, %r5, %r94;
    mul.wide.u32 %rd20, %r96, 2;
    add.s64 %rd21, %rd2, %rd20;
    ld.global.u16 %r95, [%rd21];
MMA_RHS_T_CTA_LDMATRIX_A_STAGE_B_STORE:
    mul.wide.u32 %rd22, %r90, 2;
    mov.u64 %rd23, heirloom_matmul_ldmatrix_smem_b;
    add.s64 %rd24, %rd23, %rd22;
    st.shared.u16 [%rd24], %r95;
    add.u32 %r90, %r90, 128;
    bra MMA_RHS_T_CTA_LDMATRIX_A_STAGE_B_LOOP;
MMA_RHS_T_CTA_LDMATRIX_A_STAGE_B_DONE:

    bar.sync 0;

    and.b32 %r100, %r71, 7;
    and.b32 %r101, %r71, 15;
    shr.u32 %r102, %r71, 4;
    setp.ge.u32 %p8, %r101, 8;
    selp.u32 %r103, 8, 0, %p8;
    add.u32 %r104, %r74, %r103;
    add.u32 %r104, %r104, %r100;
    shl.b32 %r105, %r102, 3;
    mad.lo.u32 %r106, %r104, %r89, %r105;
    mul.wide.u32 %rd30, %r106, 2;
    mov.u64 %rd31, heirloom_matmul_ldmatrix_smem_a;
    add.s64 %rd32, %rd31, %rd30;
    cvt.u32.u64 %r107, %rd32;
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r30, %r31, %r32, %r33}, [%r107];

    add.u32 %r26, %r75, %r13;
    add.u32 %r20, %r18, 0;
    mad.lo.u32 %r21, %r26, %r89, %r20;
    mul.wide.u32 %rd7, %r21, 2;
    mov.u64 %rd8, heirloom_matmul_ldmatrix_smem_b;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r22, [%rd9];
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r26, %r89, %r23;
    mul.wide.u32 %rd7, %r21, 2;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r24, [%rd9];
    shl.b32 %r25, %r24, 16;
    or.b32 %r34, %r22, %r25;

    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r26, %r89, %r20;
    mul.wide.u32 %rd7, %r21, 2;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r22, [%rd9];
    mad.lo.u32 %r21, %r26, %r89, %r23;
    mul.wide.u32 %rd7, %r21, 2;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r24, [%rd9];
    shl.b32 %r25, %r24, 16;
    or.b32 %r35, %r22, %r25;

    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f5, %f6, %f7, %f8},
        {%r30, %r31, %r32, %r33},
        {%r34, %r35},
        {%f1, %f2, %f3, %f4};
    mov.f32 %f1, %f5;
    mov.f32 %f2, %f6;
    mov.f32 %f3, %f7;
    mov.f32 %f4, %f8;

    bar.sync 0;
    add.u32 %r17, %r17, 16;
    bra MMA_RHS_T_CTA_LDMATRIX_A_K_LOOP;

MMA_RHS_T_CTA_LDMATRIX_A_STORE:
    setp.ge.u32 %p2, %r15, %r4;
    setp.ge.u32 %p3, %r16, %r6;
    or.pred %p4, %p2, %p3;
    @%p4 bra MMA_RHS_T_CTA_LDMATRIX_A_DONE;

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r16, %r18;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f1;

    add.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f2;

    add.u32 %r19, %r19, 8;
    sub.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f3;

    add.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f4;
MMA_RHS_T_CTA_LDMATRIX_A_DONE:
    ret;
}

.visible .entry heirloom_matmul_bf16_mma_rhs_t_f32_cta_cp_async(
    .param .u64 left,
    .param .u64 right,
    .param .u64 output,
    .param .u32 m,
    .param .u32 k,
    .param .u32 n
)
{
    .reg .pred %p<10>;
    .reg .b32 %r<140>;
    .reg .b64 %rd<110>;
    .reg .f32 %f<9>;
    .shared .align 16 .b8 heirloom_matmul_cp_async_smem_a[1024];
    .shared .align 16 .b8 heirloom_matmul_cp_async_smem_b[512];

    ld.param.u64 %rd1, [left];
    ld.param.u64 %rd2, [right];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [m];
    ld.param.u32 %r5, [k];
    ld.param.u32 %r6, [n];

    mov.u32 %r10, %tid.x;
    mov.u32 %r11, %ctaid.x;
    mov.u32 %r12, %ctaid.y;
    shr.u32 %r70, %r10, 5;
    and.b32 %r71, %r10, 31;
    shr.u32 %r72, %r70, 1;
    and.b32 %r73, %r70, 1;
    shr.u32 %r13, %r71, 2;
    and.b32 %r14, %r71, 3;
    shl.b32 %r80, %r12, 5;
    shl.b32 %r81, %r11, 4;
    shl.b32 %r74, %r72, 4;
    shl.b32 %r75, %r73, 3;
    add.u32 %r15, %r80, %r74;
    add.u32 %r16, %r81, %r75;
    shl.b32 %r18, %r14, 1;
    mov.u32 %r89, 16;

    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.f32 %f4, 0f00000000;
    mov.u32 %r17, 0;

MMA_RHS_T_CTA_CP_ASYNC_K_LOOP:
    setp.ge.u32 %p1, %r17, %r5;
    @%p1 bra MMA_RHS_T_CTA_CP_ASYNC_STORE;

    setp.ge.u32 %p4, %r10, 64;
    @%p4 bra MMA_RHS_T_CTA_CP_ASYNC_A_DONE;
    shr.u32 %r91, %r10, 1;
    and.b32 %r92, %r10, 1;
    shl.b32 %r92, %r92, 4;
    add.u32 %r93, %r80, %r91;
    mad.lo.u32 %r96, %r93, %r5, %r17;
    mul.wide.u32 %rd20, %r96, 2;
    cvt.u64.u32 %rd26, %r92;
    add.s64 %rd21, %rd1, %rd20;
    add.s64 %rd21, %rd21, %rd26;
    mul.wide.u32 %rd22, %r91, 32;
    add.s64 %rd22, %rd22, %rd26;
    mov.u64 %rd23, heirloom_matmul_cp_async_smem_a;
    add.s64 %rd24, %rd23, %rd22;
    setp.ge.u32 %p5, %r93, %r4;
    @%p5 bra MMA_RHS_T_CTA_CP_ASYNC_A_ZERO;
    cvt.u32.u64 %r107, %rd24;
    cp.async.ca.shared.global [%r107], [%rd21], 16;
    bra MMA_RHS_T_CTA_CP_ASYNC_A_DONE;
MMA_RHS_T_CTA_CP_ASYNC_A_ZERO:
    mov.u64 %rd27, 0;
    st.shared.u64 [%rd24], %rd27;
    add.s64 %rd28, %rd24, 8;
    st.shared.u64 [%rd28], %rd27;
MMA_RHS_T_CTA_CP_ASYNC_A_DONE:

    setp.ge.u32 %p4, %r10, 32;
    @%p4 bra MMA_RHS_T_CTA_CP_ASYNC_B_DONE;
    shr.u32 %r91, %r10, 1;
    and.b32 %r92, %r10, 1;
    shl.b32 %r92, %r92, 4;
    add.u32 %r93, %r81, %r91;
    mad.lo.u32 %r96, %r93, %r5, %r17;
    mul.wide.u32 %rd20, %r96, 2;
    cvt.u64.u32 %rd26, %r92;
    add.s64 %rd21, %rd2, %rd20;
    add.s64 %rd21, %rd21, %rd26;
    mul.wide.u32 %rd22, %r91, 32;
    add.s64 %rd22, %rd22, %rd26;
    mov.u64 %rd23, heirloom_matmul_cp_async_smem_b;
    add.s64 %rd24, %rd23, %rd22;
    setp.ge.u32 %p5, %r93, %r6;
    @%p5 bra MMA_RHS_T_CTA_CP_ASYNC_B_ZERO;
    cvt.u32.u64 %r107, %rd24;
    cp.async.ca.shared.global [%r107], [%rd21], 16;
    bra MMA_RHS_T_CTA_CP_ASYNC_B_DONE;
MMA_RHS_T_CTA_CP_ASYNC_B_ZERO:
    mov.u64 %rd27, 0;
    st.shared.u64 [%rd24], %rd27;
    add.s64 %rd28, %rd24, 8;
    st.shared.u64 [%rd28], %rd27;
MMA_RHS_T_CTA_CP_ASYNC_B_DONE:

    cp.async.commit_group;
    cp.async.wait_group 0;
    bar.sync 0;

    and.b32 %r100, %r71, 7;
    and.b32 %r101, %r71, 15;
    shr.u32 %r102, %r71, 4;
    setp.ge.u32 %p8, %r101, 8;
    selp.u32 %r103, 8, 0, %p8;
    add.u32 %r104, %r74, %r103;
    add.u32 %r104, %r104, %r100;
    shl.b32 %r105, %r102, 3;
    mad.lo.u32 %r106, %r104, %r89, %r105;
    mul.wide.u32 %rd30, %r106, 2;
    mov.u64 %rd31, heirloom_matmul_cp_async_smem_a;
    add.s64 %rd32, %rd31, %rd30;
    cvt.u32.u64 %r107, %rd32;
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r30, %r31, %r32, %r33}, [%r107];

    add.u32 %r26, %r75, %r13;
    add.u32 %r20, %r18, 0;
    mad.lo.u32 %r21, %r26, %r89, %r20;
    mul.wide.u32 %rd7, %r21, 2;
    mov.u64 %rd8, heirloom_matmul_cp_async_smem_b;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r22, [%rd9];
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r26, %r89, %r23;
    mul.wide.u32 %rd7, %r21, 2;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r24, [%rd9];
    shl.b32 %r25, %r24, 16;
    or.b32 %r34, %r22, %r25;

    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r26, %r89, %r20;
    mul.wide.u32 %rd7, %r21, 2;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r22, [%rd9];
    mad.lo.u32 %r21, %r26, %r89, %r23;
    mul.wide.u32 %rd7, %r21, 2;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r24, [%rd9];
    shl.b32 %r25, %r24, 16;
    or.b32 %r35, %r22, %r25;

    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f5, %f6, %f7, %f8},
        {%r30, %r31, %r32, %r33},
        {%r34, %r35},
        {%f1, %f2, %f3, %f4};
    mov.f32 %f1, %f5;
    mov.f32 %f2, %f6;
    mov.f32 %f3, %f7;
    mov.f32 %f4, %f8;

    bar.sync 0;
    add.u32 %r17, %r17, 16;
    bra MMA_RHS_T_CTA_CP_ASYNC_K_LOOP;

MMA_RHS_T_CTA_CP_ASYNC_STORE:
    setp.ge.u32 %p2, %r15, %r4;
    setp.ge.u32 %p3, %r16, %r6;
    or.pred %p4, %p2, %p3;
    @%p4 bra MMA_RHS_T_CTA_CP_ASYNC_DONE;

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r16, %r18;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f1;

    add.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f2;

    add.u32 %r19, %r19, 8;
    sub.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f3;

    add.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f4;
MMA_RHS_T_CTA_CP_ASYNC_DONE:
    ret;
}

.visible .entry heirloom_matmul_bf16_mma_rhs_t_f32_cta_cp_async_db(
    .param .u64 left,
    .param .u64 right,
    .param .u64 output,
    .param .u32 m,
    .param .u32 k,
    .param .u32 n,
    .param .u64 bias,
    .param .u32 add_bias
)
{
    .reg .pred %p<12>;
    .reg .b32 %r<150>;
    .reg .b64 %rd<120>;
    .reg .f32 %f<9>;
    .shared .align 16 .b8 heirloom_matmul_cp_async_db_smem_a[2048];
    .shared .align 16 .b8 heirloom_matmul_cp_async_db_smem_b[1024];

    ld.param.u64 %rd1, [left];
    ld.param.u64 %rd2, [right];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [m];
    ld.param.u32 %r5, [k];
    ld.param.u32 %r6, [n];
    ld.param.u64 %rd40, [bias];
    ld.param.u32 %r7, [add_bias];

    mov.u32 %r10, %tid.x;
    mov.u32 %r11, %ctaid.x;
    mov.u32 %r12, %ctaid.y;
    shr.u32 %r70, %r10, 5;
    and.b32 %r71, %r10, 31;
    shr.u32 %r72, %r70, 1;
    and.b32 %r73, %r70, 1;
    shr.u32 %r13, %r71, 2;
    and.b32 %r14, %r71, 3;
    shl.b32 %r80, %r12, 5;
    shl.b32 %r81, %r11, 4;
    shl.b32 %r74, %r72, 4;
    shl.b32 %r75, %r73, 3;
    add.u32 %r15, %r80, %r74;
    add.u32 %r16, %r81, %r75;
    shl.b32 %r18, %r14, 1;
    mov.u32 %r89, 16;

    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.f32 %f4, 0f00000000;
    mov.u32 %r17, 0;
    mov.u32 %r120, 0;

    // Prefetch K tile 0 into stage 0.
    setp.ge.u32 %p4, %r10, 64;
    @%p4 bra MMA_RHS_T_CTA_CP_ASYNC_DB_PRE_A_DONE;
    shr.u32 %r91, %r10, 1;
    and.b32 %r92, %r10, 1;
    shl.b32 %r92, %r92, 3;
    add.u32 %r93, %r80, %r91;
    add.u32 %r94, %r17, %r92;
    mad.lo.u32 %r96, %r93, %r5, %r94;
    mul.wide.u32 %rd20, %r96, 2;
    add.s64 %rd21, %rd1, %rd20;
    shl.b32 %r95, %r91, 4;
    shl.b32 %r97, %r91, 5;
    shl.b32 %r98, %r92, 1;
    add.u32 %r97, %r97, %r98;
    mov.u64 %rd23, heirloom_matmul_cp_async_db_smem_a;
    cvt.u64.u32 %rd22, %r97;
    add.s64 %rd24, %rd23, %rd22;
    cvt.u32.u64 %r107, %rd24;
    cp.async.ca.shared.global [%r107], [%rd21], 16;
MMA_RHS_T_CTA_CP_ASYNC_DB_PRE_A_DONE:

    setp.ge.u32 %p4, %r10, 32;
    @%p4 bra MMA_RHS_T_CTA_CP_ASYNC_DB_PRE_B_DONE;
    shr.u32 %r91, %r10, 1;
    and.b32 %r92, %r10, 1;
    shl.b32 %r92, %r92, 3;
    add.u32 %r93, %r81, %r91;
    add.u32 %r94, %r17, %r92;
    mad.lo.u32 %r96, %r93, %r5, %r94;
    mul.wide.u32 %rd20, %r96, 2;
    add.s64 %rd21, %rd2, %rd20;
    shl.b32 %r97, %r91, 5;
    shl.b32 %r98, %r92, 1;
    add.u32 %r97, %r97, %r98;
    mov.u64 %rd23, heirloom_matmul_cp_async_db_smem_b;
    cvt.u64.u32 %rd22, %r97;
    add.s64 %rd24, %rd23, %rd22;
    cvt.u32.u64 %r107, %rd24;
    cp.async.ca.shared.global [%r107], [%rd21], 16;
MMA_RHS_T_CTA_CP_ASYNC_DB_PRE_B_DONE:
    cp.async.commit_group;
    cp.async.wait_group 0;
    bar.sync 0;

MMA_RHS_T_CTA_CP_ASYNC_DB_K_LOOP:
    setp.ge.u32 %p1, %r17, %r5;
    @%p1 bra MMA_RHS_T_CTA_CP_ASYNC_DB_STORE;

    add.u32 %r121, %r17, 16;
    xor.b32 %r122, %r120, 1;
    setp.ge.u32 %p9, %r121, %r5;
    @%p9 bra MMA_RHS_T_CTA_CP_ASYNC_DB_PREFETCH_DONE;
    shl.b32 %r123, %r122, 10;
    shl.b32 %r124, %r122, 9;

    setp.ge.u32 %p4, %r10, 64;
    @%p4 bra MMA_RHS_T_CTA_CP_ASYNC_DB_STAGE_A_DONE;
    shr.u32 %r91, %r10, 1;
    and.b32 %r92, %r10, 1;
    shl.b32 %r92, %r92, 3;
    add.u32 %r93, %r80, %r91;
    add.u32 %r94, %r121, %r92;
    mad.lo.u32 %r96, %r93, %r5, %r94;
    mul.wide.u32 %rd20, %r96, 2;
    add.s64 %rd21, %rd1, %rd20;
    shl.b32 %r97, %r91, 5;
    shl.b32 %r98, %r92, 1;
    add.u32 %r97, %r97, %r98;
    add.u32 %r97, %r97, %r123;
    mov.u64 %rd23, heirloom_matmul_cp_async_db_smem_a;
    cvt.u64.u32 %rd22, %r97;
    add.s64 %rd24, %rd23, %rd22;
    cvt.u32.u64 %r107, %rd24;
    cp.async.ca.shared.global [%r107], [%rd21], 16;
MMA_RHS_T_CTA_CP_ASYNC_DB_STAGE_A_DONE:

    setp.ge.u32 %p4, %r10, 32;
    @%p4 bra MMA_RHS_T_CTA_CP_ASYNC_DB_STAGE_B_DONE;
    shr.u32 %r91, %r10, 1;
    and.b32 %r92, %r10, 1;
    shl.b32 %r92, %r92, 3;
    add.u32 %r93, %r81, %r91;
    add.u32 %r94, %r121, %r92;
    mad.lo.u32 %r96, %r93, %r5, %r94;
    mul.wide.u32 %rd20, %r96, 2;
    add.s64 %rd21, %rd2, %rd20;
    shl.b32 %r97, %r91, 5;
    shl.b32 %r98, %r92, 1;
    add.u32 %r97, %r97, %r98;
    add.u32 %r97, %r97, %r124;
    mov.u64 %rd23, heirloom_matmul_cp_async_db_smem_b;
    cvt.u64.u32 %rd22, %r97;
    add.s64 %rd24, %rd23, %rd22;
    cvt.u32.u64 %r107, %rd24;
    cp.async.ca.shared.global [%r107], [%rd21], 16;
MMA_RHS_T_CTA_CP_ASYNC_DB_STAGE_B_DONE:
    cp.async.commit_group;
MMA_RHS_T_CTA_CP_ASYNC_DB_PREFETCH_DONE:

    shl.b32 %r125, %r120, 10;
    shl.b32 %r126, %r120, 9;
    and.b32 %r100, %r71, 7;
    and.b32 %r101, %r71, 15;
    shr.u32 %r102, %r71, 4;
    setp.ge.u32 %p8, %r101, 8;
    selp.u32 %r103, 8, 0, %p8;
    add.u32 %r104, %r74, %r103;
    add.u32 %r104, %r104, %r100;
    shl.b32 %r105, %r102, 3;
    mad.lo.u32 %r106, %r104, %r89, %r105;
    mul.wide.u32 %rd30, %r106, 2;
    cvt.u64.u32 %rd34, %r125;
    add.s64 %rd30, %rd30, %rd34;
    mov.u64 %rd31, heirloom_matmul_cp_async_db_smem_a;
    add.s64 %rd32, %rd31, %rd30;
    cvt.u32.u64 %r107, %rd32;
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r30, %r31, %r32, %r33}, [%r107];

    add.u32 %r26, %r75, %r13;
    add.u32 %r20, %r18, 0;
    mad.lo.u32 %r21, %r26, %r89, %r20;
    mul.wide.u32 %rd7, %r21, 2;
    cvt.u64.u32 %rd35, %r126;
    add.s64 %rd7, %rd7, %rd35;
    mov.u64 %rd8, heirloom_matmul_cp_async_db_smem_b;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r22, [%rd9];
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r26, %r89, %r23;
    mul.wide.u32 %rd7, %r21, 2;
    add.s64 %rd7, %rd7, %rd35;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r24, [%rd9];
    shl.b32 %r25, %r24, 16;
    or.b32 %r34, %r22, %r25;

    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r26, %r89, %r20;
    mul.wide.u32 %rd7, %r21, 2;
    add.s64 %rd7, %rd7, %rd35;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r22, [%rd9];
    mad.lo.u32 %r21, %r26, %r89, %r23;
    mul.wide.u32 %rd7, %r21, 2;
    add.s64 %rd7, %rd7, %rd35;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r24, [%rd9];
    shl.b32 %r25, %r24, 16;
    or.b32 %r35, %r22, %r25;

    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f5, %f6, %f7, %f8},
        {%r30, %r31, %r32, %r33},
        {%r34, %r35},
        {%f1, %f2, %f3, %f4};
    mov.f32 %f1, %f5;
    mov.f32 %f2, %f6;
    mov.f32 %f3, %f7;
    mov.f32 %f4, %f8;

    @%p9 bra MMA_RHS_T_CTA_CP_ASYNC_DB_ADVANCE;
    cp.async.wait_group 0;
    bar.sync 0;
MMA_RHS_T_CTA_CP_ASYNC_DB_ADVANCE:
    mov.u32 %r17, %r121;
    mov.u32 %r120, %r122;
    bra MMA_RHS_T_CTA_CP_ASYNC_DB_K_LOOP;

MMA_RHS_T_CTA_CP_ASYNC_DB_STORE:
    setp.ge.u32 %p2, %r15, %r4;
    setp.ge.u32 %p3, %r16, %r6;
    or.pred %p4, %p2, %p3;
    @%p4 bra MMA_RHS_T_CTA_CP_ASYNC_DB_DONE;
    setp.eq.u32 %p10, %r7, 0;

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r16, %r18;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    @%p10 bra MMA_RHS_T_CTA_CP_ASYNC_DB_STORE_0_NO_BIAS;
    mul.wide.u32 %rd41, %r20, 4;
    add.s64 %rd42, %rd40, %rd41;
    ld.global.f32 %f5, [%rd42];
    add.f32 %f1, %f1, %f5;
MMA_RHS_T_CTA_CP_ASYNC_DB_STORE_0_NO_BIAS:
    st.global.f32 [%rd11], %f1;

    add.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    @%p10 bra MMA_RHS_T_CTA_CP_ASYNC_DB_STORE_1_NO_BIAS;
    mul.wide.u32 %rd41, %r20, 4;
    add.s64 %rd42, %rd40, %rd41;
    ld.global.f32 %f5, [%rd42];
    add.f32 %f2, %f2, %f5;
MMA_RHS_T_CTA_CP_ASYNC_DB_STORE_1_NO_BIAS:
    st.global.f32 [%rd11], %f2;

    add.u32 %r19, %r19, 8;
    sub.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    @%p10 bra MMA_RHS_T_CTA_CP_ASYNC_DB_STORE_2_NO_BIAS;
    mul.wide.u32 %rd41, %r20, 4;
    add.s64 %rd42, %rd40, %rd41;
    ld.global.f32 %f5, [%rd42];
    add.f32 %f3, %f3, %f5;
MMA_RHS_T_CTA_CP_ASYNC_DB_STORE_2_NO_BIAS:
    st.global.f32 [%rd11], %f3;

    add.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    @%p10 bra MMA_RHS_T_CTA_CP_ASYNC_DB_STORE_3_NO_BIAS;
    mul.wide.u32 %rd41, %r20, 4;
    add.s64 %rd42, %rd40, %rd41;
    ld.global.f32 %f5, [%rd42];
    add.f32 %f4, %f4, %f5;
MMA_RHS_T_CTA_CP_ASYNC_DB_STORE_3_NO_BIAS:
    st.global.f32 [%rd11], %f4;
MMA_RHS_T_CTA_CP_ASYNC_DB_DONE:
    ret;
}

.visible .entry heirloom_matmul_bf16_mma_rhs_t_f32_cta_wide_swizzled(
    .param .u64 left,
    .param .u64 right,
    .param .u64 output,
    .param .u32 m,
    .param .u32 k,
    .param .u32 n
)
{
    .reg .pred %p<10>;
    .reg .b32 %r<150>;
    .reg .b64 %rd<100>;
    .reg .f32 %f<9>;
    .shared .align 4 .b8 heirloom_matmul_wide_smem_a[1024];
    .shared .align 4 .b8 heirloom_matmul_wide_smem_b[1024];

    ld.param.u64 %rd1, [left];
    ld.param.u64 %rd2, [right];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [m];
    ld.param.u32 %r5, [k];
    ld.param.u32 %r6, [n];

    mov.u32 %r10, %tid.x;
    mov.u32 %r11, %ctaid.x;
    mov.u32 %r12, %ctaid.y;
    shr.u32 %r70, %r10, 5;
    and.b32 %r71, %r10, 31;
    shr.u32 %r72, %r70, 2;
    and.b32 %r73, %r70, 3;
    shr.u32 %r13, %r71, 2;
    and.b32 %r14, %r71, 3;
    shl.b32 %r80, %r12, 5;
    shl.b32 %r81, %r11, 5;
    shl.b32 %r74, %r72, 4;
    shl.b32 %r75, %r73, 3;
    add.u32 %r15, %r80, %r74;
    add.u32 %r16, %r81, %r75;
    shl.b32 %r18, %r14, 1;
    mov.u32 %r89, 16;

    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.f32 %f4, 0f00000000;
    mov.u32 %r17, 0;

MMA_RHS_T_CTA_WIDE_SWIZZLED_K_LOOP:
    setp.ge.u32 %p1, %r17, %r5;
    @%p1 bra MMA_RHS_T_CTA_WIDE_SWIZZLED_STORE;

    mov.u32 %r90, %r10;
MMA_RHS_T_CTA_WIDE_SWIZZLED_STAGE_A_LOOP:
    setp.ge.u32 %p4, %r90, 512;
    @%p4 bra MMA_RHS_T_CTA_WIDE_SWIZZLED_STAGE_A_DONE;
    div.u32 %r91, %r90, %r89;
    rem.u32 %r92, %r90, %r89;
    add.u32 %r93, %r80, %r91;
    add.u32 %r94, %r17, %r92;
    setp.ge.u32 %p5, %r93, %r4;
    setp.ge.u32 %p6, %r94, %r5;
    or.pred %p7, %p5, %p6;
    mov.u32 %r95, 0;
    @%p7 bra MMA_RHS_T_CTA_WIDE_SWIZZLED_STAGE_A_STORE;
    mad.lo.u32 %r96, %r93, %r5, %r94;
    mul.wide.u32 %rd20, %r96, 2;
    add.s64 %rd21, %rd1, %rd20;
    ld.global.u16 %r95, [%rd21];
MMA_RHS_T_CTA_WIDE_SWIZZLED_STAGE_A_STORE:
    and.b32 %r97, %r91, 7;
    xor.b32 %r98, %r92, %r97;
    mad.lo.u32 %r99, %r91, %r89, %r98;
    mul.wide.u32 %rd22, %r99, 2;
    mov.u64 %rd23, heirloom_matmul_wide_smem_a;
    add.s64 %rd24, %rd23, %rd22;
    st.shared.u16 [%rd24], %r95;
    add.u32 %r90, %r90, 256;
    bra MMA_RHS_T_CTA_WIDE_SWIZZLED_STAGE_A_LOOP;
MMA_RHS_T_CTA_WIDE_SWIZZLED_STAGE_A_DONE:

    mov.u32 %r90, %r10;
MMA_RHS_T_CTA_WIDE_SWIZZLED_STAGE_B_LOOP:
    setp.ge.u32 %p4, %r90, 512;
    @%p4 bra MMA_RHS_T_CTA_WIDE_SWIZZLED_STAGE_B_DONE;
    div.u32 %r91, %r90, %r89;
    rem.u32 %r92, %r90, %r89;
    add.u32 %r93, %r81, %r91;
    add.u32 %r94, %r17, %r92;
    setp.ge.u32 %p5, %r93, %r6;
    setp.ge.u32 %p6, %r94, %r5;
    or.pred %p7, %p5, %p6;
    mov.u32 %r95, 0;
    @%p7 bra MMA_RHS_T_CTA_WIDE_SWIZZLED_STAGE_B_STORE;
    mad.lo.u32 %r96, %r93, %r5, %r94;
    mul.wide.u32 %rd20, %r96, 2;
    add.s64 %rd21, %rd2, %rd20;
    ld.global.u16 %r95, [%rd21];
MMA_RHS_T_CTA_WIDE_SWIZZLED_STAGE_B_STORE:
    and.b32 %r97, %r91, 7;
    xor.b32 %r98, %r92, %r97;
    mad.lo.u32 %r99, %r91, %r89, %r98;
    mul.wide.u32 %rd22, %r99, 2;
    mov.u64 %rd23, heirloom_matmul_wide_smem_b;
    add.s64 %rd24, %rd23, %rd22;
    st.shared.u16 [%rd24], %r95;
    add.u32 %r90, %r90, 256;
    bra MMA_RHS_T_CTA_WIDE_SWIZZLED_STAGE_B_LOOP;
MMA_RHS_T_CTA_WIDE_SWIZZLED_STAGE_B_DONE:

    bar.sync 0;

    add.u32 %r19, %r74, %r13;
    add.u32 %r20, %r18, 0;
    and.b32 %r97, %r19, 7;
    xor.b32 %r98, %r20, %r97;
    mad.lo.u32 %r21, %r19, %r89, %r98;
    mul.wide.u32 %rd4, %r21, 2;
    mov.u64 %rd5, heirloom_matmul_wide_smem_a;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r22, [%rd6];
    add.u32 %r23, %r20, 1;
    xor.b32 %r98, %r23, %r97;
    mad.lo.u32 %r21, %r19, %r89, %r98;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r24, [%rd6];
    shl.b32 %r25, %r24, 16;
    or.b32 %r30, %r22, %r25;

    add.u32 %r19, %r19, 8;
    and.b32 %r97, %r19, 7;
    xor.b32 %r98, %r20, %r97;
    mad.lo.u32 %r21, %r19, %r89, %r98;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r22, [%rd6];
    xor.b32 %r98, %r23, %r97;
    mad.lo.u32 %r21, %r19, %r89, %r98;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r24, [%rd6];
    shl.b32 %r25, %r24, 16;
    or.b32 %r31, %r22, %r25;

    add.u32 %r19, %r74, %r13;
    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    and.b32 %r97, %r19, 7;
    xor.b32 %r98, %r20, %r97;
    mad.lo.u32 %r21, %r19, %r89, %r98;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r22, [%rd6];
    xor.b32 %r98, %r23, %r97;
    mad.lo.u32 %r21, %r19, %r89, %r98;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r24, [%rd6];
    shl.b32 %r25, %r24, 16;
    or.b32 %r32, %r22, %r25;

    add.u32 %r19, %r19, 8;
    and.b32 %r97, %r19, 7;
    xor.b32 %r98, %r20, %r97;
    mad.lo.u32 %r21, %r19, %r89, %r98;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r22, [%rd6];
    xor.b32 %r98, %r23, %r97;
    mad.lo.u32 %r21, %r19, %r89, %r98;
    mul.wide.u32 %rd4, %r21, 2;
    add.s64 %rd6, %rd5, %rd4;
    ld.shared.u16 %r24, [%rd6];
    shl.b32 %r25, %r24, 16;
    or.b32 %r33, %r22, %r25;

    add.u32 %r26, %r75, %r13;
    add.u32 %r20, %r18, 0;
    and.b32 %r97, %r26, 7;
    xor.b32 %r98, %r20, %r97;
    mad.lo.u32 %r21, %r26, %r89, %r98;
    mul.wide.u32 %rd7, %r21, 2;
    mov.u64 %rd8, heirloom_matmul_wide_smem_b;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r22, [%rd9];
    add.u32 %r23, %r20, 1;
    xor.b32 %r98, %r23, %r97;
    mad.lo.u32 %r21, %r26, %r89, %r98;
    mul.wide.u32 %rd7, %r21, 2;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r24, [%rd9];
    shl.b32 %r25, %r24, 16;
    or.b32 %r34, %r22, %r25;

    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    xor.b32 %r98, %r20, %r97;
    mad.lo.u32 %r21, %r26, %r89, %r98;
    mul.wide.u32 %rd7, %r21, 2;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r22, [%rd9];
    xor.b32 %r98, %r23, %r97;
    mad.lo.u32 %r21, %r26, %r89, %r98;
    mul.wide.u32 %rd7, %r21, 2;
    add.s64 %rd9, %rd8, %rd7;
    ld.shared.u16 %r24, [%rd9];
    shl.b32 %r25, %r24, 16;
    or.b32 %r35, %r22, %r25;

    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f5, %f6, %f7, %f8},
        {%r30, %r31, %r32, %r33},
        {%r34, %r35},
        {%f1, %f2, %f3, %f4};
    mov.f32 %f1, %f5;
    mov.f32 %f2, %f6;
    mov.f32 %f3, %f7;
    mov.f32 %f4, %f8;

    bar.sync 0;
    add.u32 %r17, %r17, 16;
    bra MMA_RHS_T_CTA_WIDE_SWIZZLED_K_LOOP;

MMA_RHS_T_CTA_WIDE_SWIZZLED_STORE:
    setp.ge.u32 %p2, %r15, %r4;
    setp.ge.u32 %p3, %r16, %r6;
    or.pred %p4, %p2, %p3;
    @%p4 bra MMA_RHS_T_CTA_WIDE_SWIZZLED_DONE;

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r16, %r18;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f1;

    add.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f2;

    add.u32 %r19, %r19, 8;
    sub.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f3;

    add.u32 %r20, %r20, 1;
    mad.lo.u32 %r21, %r19, %r6, %r20;
    mul.wide.u32 %rd10, %r21, 4;
    add.s64 %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f4;
MMA_RHS_T_CTA_WIDE_SWIZZLED_DONE:
    ret;
}

.visible .entry heirloom_attention_qk_bf16_mma_f32(
    .param .u64 query,
    .param .u64 key,
    .param .u64 scores,
    .param .u32 batch,
    .param .u32 time,
    .param .u32 channels,
    .param .u32 n_heads,
    .param .u32 head_dim
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<100>;
    .reg .b64 %rd<80>;
    .reg .f32 %f<9>;

    ld.param.u64 %rd1, [query];
    ld.param.u64 %rd2, [key];
    ld.param.u64 %rd3, [scores];
    ld.param.u32 %r4, [batch];
    ld.param.u32 %r5, [time];
    ld.param.u32 %r6, [channels];
    ld.param.u32 %r7, [n_heads];
    ld.param.u32 %r8, [head_dim];

    mov.u32 %r10, %tid.x;
    mov.u32 %r11, %ctaid.x;
    mov.u32 %r12, %ctaid.y;
    mov.u32 %r27, %ctaid.z;
    shr.u32 %r13, %r10, 2;
    and.b32 %r14, %r10, 3;
    shl.b32 %r15, %r12, 4;
    shl.b32 %r16, %r11, 3;
    shl.b32 %r18, %r14, 1;
    rem.u32 %r28, %r27, %r7;
    div.u32 %r29, %r27, %r7;
    mul.lo.u32 %r40, %r28, %r8;

    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.f32 %f4, 0f00000000;
    mov.u32 %r17, 0;

ATTN_QK_K_LOOP:
    setp.ge.u32 %p1, %r17, %r8;
    @%p1 bra ATTN_QK_STORE;

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r17, %r18;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r41, %r29, %r5;
    add.u32 %r41, %r41, %r19;
    mul.lo.u32 %r41, %r41, %r6;
    add.u32 %r41, %r41, %r40;
    add.u32 %r42, %r41, %r20;
    mul.wide.u32 %rd4, %r42, 2;
    add.s64 %rd5, %rd1, %rd4;
    mov.u32 %r22, 0;
    setp.lt.u32 %p2, %r19, %r5;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r22, [%rd5];
    add.u32 %r42, %r41, %r23;
    mul.wide.u32 %rd4, %r42, 2;
    add.s64 %rd5, %rd1, %rd4;
    mov.u32 %r24, 0;
    setp.lt.u32 %p3, %r23, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r24, [%rd5];
    shl.b32 %r25, %r24, 16;
    or.b32 %r30, %r22, %r25;

    add.u32 %r19, %r19, 8;
    mul.lo.u32 %r41, %r29, %r5;
    add.u32 %r41, %r41, %r19;
    mul.lo.u32 %r41, %r41, %r6;
    add.u32 %r41, %r41, %r40;
    add.u32 %r42, %r41, %r20;
    mul.wide.u32 %rd4, %r42, 2;
    add.s64 %rd5, %rd1, %rd4;
    mov.u32 %r22, 0;
    setp.lt.u32 %p2, %r19, %r5;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r22, [%rd5];
    add.u32 %r42, %r41, %r23;
    mul.wide.u32 %rd4, %r42, 2;
    add.s64 %rd5, %rd1, %rd4;
    mov.u32 %r24, 0;
    setp.lt.u32 %p3, %r23, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r24, [%rd5];
    shl.b32 %r25, %r24, 16;
    or.b32 %r31, %r22, %r25;

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r41, %r29, %r5;
    add.u32 %r41, %r41, %r19;
    mul.lo.u32 %r41, %r41, %r6;
    add.u32 %r41, %r41, %r40;
    add.u32 %r42, %r41, %r20;
    mul.wide.u32 %rd4, %r42, 2;
    add.s64 %rd5, %rd1, %rd4;
    mov.u32 %r22, 0;
    setp.lt.u32 %p2, %r19, %r5;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r22, [%rd5];
    add.u32 %r42, %r41, %r23;
    mul.wide.u32 %rd4, %r42, 2;
    add.s64 %rd5, %rd1, %rd4;
    mov.u32 %r24, 0;
    setp.lt.u32 %p3, %r23, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r24, [%rd5];
    shl.b32 %r25, %r24, 16;
    or.b32 %r32, %r22, %r25;

    add.u32 %r19, %r19, 8;
    mul.lo.u32 %r41, %r29, %r5;
    add.u32 %r41, %r41, %r19;
    mul.lo.u32 %r41, %r41, %r6;
    add.u32 %r41, %r41, %r40;
    add.u32 %r42, %r41, %r20;
    mul.wide.u32 %rd4, %r42, 2;
    add.s64 %rd5, %rd1, %rd4;
    mov.u32 %r22, 0;
    setp.lt.u32 %p2, %r19, %r5;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r22, [%rd5];
    add.u32 %r42, %r41, %r23;
    mul.wide.u32 %rd4, %r42, 2;
    add.s64 %rd5, %rd1, %rd4;
    mov.u32 %r24, 0;
    setp.lt.u32 %p3, %r23, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r24, [%rd5];
    shl.b32 %r25, %r24, 16;
    or.b32 %r33, %r22, %r25;

    add.u32 %r26, %r16, %r13;
    add.u32 %r20, %r17, %r18;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r43, %r29, %r5;
    add.u32 %r43, %r43, %r26;
    mul.lo.u32 %r43, %r43, %r6;
    add.u32 %r43, %r43, %r40;
    add.u32 %r44, %r43, %r20;
    mul.wide.u32 %rd6, %r44, 2;
    add.s64 %rd7, %rd2, %rd6;
    mov.u32 %r22, 0;
    setp.lt.u32 %p2, %r26, %r5;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r22, [%rd7];
    add.u32 %r44, %r43, %r23;
    mul.wide.u32 %rd6, %r44, 2;
    add.s64 %rd7, %rd2, %rd6;
    mov.u32 %r24, 0;
    setp.lt.u32 %p3, %r23, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r24, [%rd7];
    shl.b32 %r25, %r24, 16;
    or.b32 %r34, %r22, %r25;

    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    add.u32 %r44, %r43, %r20;
    mul.wide.u32 %rd6, %r44, 2;
    add.s64 %rd7, %rd2, %rd6;
    mov.u32 %r22, 0;
    setp.lt.u32 %p2, %r26, %r5;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r22, [%rd7];
    add.u32 %r44, %r43, %r23;
    mul.wide.u32 %rd6, %r44, 2;
    add.s64 %rd7, %rd2, %rd6;
    mov.u32 %r24, 0;
    setp.lt.u32 %p3, %r23, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r24, [%rd7];
    shl.b32 %r25, %r24, 16;
    or.b32 %r35, %r22, %r25;

    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f5, %f6, %f7, %f8},
        {%r30, %r31, %r32, %r33},
        {%r34, %r35},
        {%f1, %f2, %f3, %f4};
    mov.f32 %f1, %f5;
    mov.f32 %f2, %f6;
    mov.f32 %f3, %f7;
    mov.f32 %f4, %f8;

    add.u32 %r17, %r17, 16;
    bra ATTN_QK_K_LOOP;

ATTN_QK_STORE:
    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r16, %r18;
    mul.lo.u32 %r45, %r27, %r5;
    add.u32 %r45, %r45, %r19;
    mul.lo.u32 %r45, %r45, %r5;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd8, %r46, 4;
    add.s64 %rd9, %rd3, %rd8;
    setp.lt.u32 %p2, %r19, %r5;
    setp.lt.u32 %p3, %r20, %r5;
    and.pred %p4, %p2, %p3;
    @%p4 st.global.f32 [%rd9], %f1;

    add.u32 %r20, %r20, 1;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd8, %r46, 4;
    add.s64 %rd9, %rd3, %rd8;
    setp.lt.u32 %p3, %r20, %r5;
    and.pred %p4, %p2, %p3;
    @%p4 st.global.f32 [%rd9], %f2;

    add.u32 %r19, %r19, 8;
    sub.u32 %r20, %r20, 1;
    mul.lo.u32 %r45, %r27, %r5;
    add.u32 %r45, %r45, %r19;
    mul.lo.u32 %r45, %r45, %r5;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd8, %r46, 4;
    add.s64 %rd9, %rd3, %rd8;
    setp.lt.u32 %p2, %r19, %r5;
    setp.lt.u32 %p3, %r20, %r5;
    and.pred %p4, %p2, %p3;
    @%p4 st.global.f32 [%rd9], %f3;

    add.u32 %r20, %r20, 1;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd8, %r46, 4;
    add.s64 %rd9, %rd3, %rd8;
    setp.lt.u32 %p3, %r20, %r5;
    and.pred %p4, %p2, %p3;
    @%p4 st.global.f32 [%rd9], %f4;
    ret;
}

.visible .entry heirloom_attention_av_bf16_mma_f32(
    .param .u64 attention,
    .param .u64 value,
    .param .u64 output,
    .param .u32 batch,
    .param .u32 time,
    .param .u32 channels,
    .param .u32 n_heads,
    .param .u32 head_dim
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<100>;
    .reg .b64 %rd<80>;
    .reg .f32 %f<9>;

    ld.param.u64 %rd1, [attention];
    ld.param.u64 %rd2, [value];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [batch];
    ld.param.u32 %r5, [time];
    ld.param.u32 %r6, [channels];
    ld.param.u32 %r7, [n_heads];
    ld.param.u32 %r8, [head_dim];

    mov.u32 %r10, %tid.x;
    mov.u32 %r11, %ctaid.x;
    mov.u32 %r12, %ctaid.y;
    mov.u32 %r27, %ctaid.z;
    shr.u32 %r13, %r10, 2;
    and.b32 %r14, %r10, 3;
    shl.b32 %r15, %r12, 4;
    shl.b32 %r16, %r11, 3;
    shl.b32 %r18, %r14, 1;
    rem.u32 %r28, %r27, %r7;
    div.u32 %r29, %r27, %r7;
    mul.lo.u32 %r40, %r28, %r8;

    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.f32 %f4, 0f00000000;
    mov.u32 %r17, 0;

ATTN_AV_K_LOOP:
    setp.ge.u32 %p1, %r17, %r5;
    @%p1 bra ATTN_AV_STORE;

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r17, %r18;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r41, %r27, %r5;
    add.u32 %r41, %r41, %r19;
    mul.lo.u32 %r41, %r41, %r5;
    add.u32 %r42, %r41, %r20;
    mul.wide.u32 %rd4, %r42, 2;
    add.s64 %rd5, %rd1, %rd4;
    mov.u32 %r22, 0;
    setp.lt.u32 %p2, %r19, %r5;
    setp.lt.u32 %p3, %r20, %r5;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r22, [%rd5];
    add.u32 %r42, %r41, %r23;
    mul.wide.u32 %rd4, %r42, 2;
    add.s64 %rd5, %rd1, %rd4;
    mov.u32 %r24, 0;
    setp.lt.u32 %p3, %r23, %r5;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r24, [%rd5];
    shl.b32 %r25, %r24, 16;
    or.b32 %r30, %r22, %r25;

    add.u32 %r19, %r19, 8;
    mul.lo.u32 %r41, %r27, %r5;
    add.u32 %r41, %r41, %r19;
    mul.lo.u32 %r41, %r41, %r5;
    add.u32 %r42, %r41, %r20;
    mul.wide.u32 %rd4, %r42, 2;
    add.s64 %rd5, %rd1, %rd4;
    mov.u32 %r22, 0;
    setp.lt.u32 %p2, %r19, %r5;
    setp.lt.u32 %p3, %r20, %r5;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r22, [%rd5];
    add.u32 %r42, %r41, %r23;
    mul.wide.u32 %rd4, %r42, 2;
    add.s64 %rd5, %rd1, %rd4;
    mov.u32 %r24, 0;
    setp.lt.u32 %p3, %r23, %r5;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r24, [%rd5];
    shl.b32 %r25, %r24, 16;
    or.b32 %r31, %r22, %r25;

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r41, %r27, %r5;
    add.u32 %r41, %r41, %r19;
    mul.lo.u32 %r41, %r41, %r5;
    add.u32 %r42, %r41, %r20;
    mul.wide.u32 %rd4, %r42, 2;
    add.s64 %rd5, %rd1, %rd4;
    mov.u32 %r22, 0;
    setp.lt.u32 %p2, %r19, %r5;
    setp.lt.u32 %p3, %r20, %r5;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r22, [%rd5];
    add.u32 %r42, %r41, %r23;
    mul.wide.u32 %rd4, %r42, 2;
    add.s64 %rd5, %rd1, %rd4;
    mov.u32 %r24, 0;
    setp.lt.u32 %p3, %r23, %r5;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r24, [%rd5];
    shl.b32 %r25, %r24, 16;
    or.b32 %r32, %r22, %r25;

    add.u32 %r19, %r19, 8;
    mul.lo.u32 %r41, %r27, %r5;
    add.u32 %r41, %r41, %r19;
    mul.lo.u32 %r41, %r41, %r5;
    add.u32 %r42, %r41, %r20;
    mul.wide.u32 %rd4, %r42, 2;
    add.s64 %rd5, %rd1, %rd4;
    mov.u32 %r22, 0;
    setp.lt.u32 %p2, %r19, %r5;
    setp.lt.u32 %p3, %r20, %r5;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r22, [%rd5];
    add.u32 %r42, %r41, %r23;
    mul.wide.u32 %rd4, %r42, 2;
    add.s64 %rd5, %rd1, %rd4;
    mov.u32 %r24, 0;
    setp.lt.u32 %p3, %r23, %r5;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r24, [%rd5];
    shl.b32 %r25, %r24, 16;
    or.b32 %r33, %r22, %r25;

    add.u32 %r26, %r16, %r13;
    add.u32 %r20, %r17, %r18;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r43, %r29, %r5;
    add.u32 %r43, %r43, %r20;
    mul.lo.u32 %r43, %r43, %r6;
    add.u32 %r43, %r43, %r40;
    add.u32 %r43, %r43, %r26;
    mul.wide.u32 %rd6, %r43, 2;
    add.s64 %rd7, %rd2, %rd6;
    mov.u32 %r22, 0;
    setp.lt.u32 %p2, %r20, %r5;
    setp.lt.u32 %p3, %r26, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r22, [%rd7];
    add.u32 %r44, %r43, %r6;
    mul.wide.u32 %rd6, %r44, 2;
    add.s64 %rd7, %rd2, %rd6;
    mov.u32 %r24, 0;
    add.u32 %r44, %r20, 1;
    setp.lt.u32 %p2, %r44, %r5;
    setp.lt.u32 %p3, %r26, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r24, [%rd7];
    shl.b32 %r25, %r24, 16;
    or.b32 %r34, %r22, %r25;

    add.u32 %r20, %r20, 8;
    mul.lo.u32 %r43, %r29, %r5;
    add.u32 %r43, %r43, %r20;
    mul.lo.u32 %r43, %r43, %r6;
    add.u32 %r43, %r43, %r40;
    add.u32 %r43, %r43, %r26;
    mul.wide.u32 %rd6, %r43, 2;
    add.s64 %rd7, %rd2, %rd6;
    mov.u32 %r22, 0;
    setp.lt.u32 %p2, %r20, %r5;
    setp.lt.u32 %p3, %r26, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r22, [%rd7];
    add.u32 %r44, %r43, %r6;
    mul.wide.u32 %rd6, %r44, 2;
    add.s64 %rd7, %rd2, %rd6;
    mov.u32 %r24, 0;
    add.u32 %r44, %r20, 1;
    setp.lt.u32 %p2, %r44, %r5;
    setp.lt.u32 %p3, %r26, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 ld.global.u16 %r24, [%rd7];
    shl.b32 %r25, %r24, 16;
    or.b32 %r35, %r22, %r25;

    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f5, %f6, %f7, %f8},
        {%r30, %r31, %r32, %r33},
        {%r34, %r35},
        {%f1, %f2, %f3, %f4};
    mov.f32 %f1, %f5;
    mov.f32 %f2, %f6;
    mov.f32 %f3, %f7;
    mov.f32 %f4, %f8;

    add.u32 %r17, %r17, 16;
    bra ATTN_AV_K_LOOP;

ATTN_AV_STORE:
    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r16, %r18;
    mul.lo.u32 %r45, %r29, %r5;
    add.u32 %r45, %r45, %r19;
    mul.lo.u32 %r45, %r45, %r6;
    add.u32 %r45, %r45, %r40;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd8, %r46, 4;
    add.s64 %rd9, %rd3, %rd8;
    setp.lt.u32 %p2, %r19, %r5;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 st.global.f32 [%rd9], %f1;

    add.u32 %r20, %r20, 1;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd8, %r46, 4;
    add.s64 %rd9, %rd3, %rd8;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 st.global.f32 [%rd9], %f2;

    add.u32 %r19, %r19, 8;
    sub.u32 %r20, %r20, 1;
    mul.lo.u32 %r45, %r29, %r5;
    add.u32 %r45, %r45, %r19;
    mul.lo.u32 %r45, %r45, %r6;
    add.u32 %r45, %r45, %r40;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd8, %r46, 4;
    add.s64 %rd9, %rd3, %rd8;
    setp.lt.u32 %p2, %r19, %r5;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 st.global.f32 [%rd9], %f3;

    add.u32 %r20, %r20, 1;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd8, %r46, 4;
    add.s64 %rd9, %rd3, %rd8;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 st.global.f32 [%rd9], %f4;
    ret;
}

.visible .entry heirloom_flash_attention_bf16_tc_fwd_f32(
    .param .u64 query,
    .param .u64 key,
    .param .u64 value,
    .param .u64 output,
    .param .u64 row_max_out,
    .param .u64 row_denom_out,
    .param .u32 batch,
    .param .u32 time,
    .param .u32 channels,
    .param .u32 n_heads,
    .param .u32 head_dim
)
{
    .reg .pred %p<24>;
    .reg .b32 %r<190>;
    .reg .b64 %rd<130>;
    .reg .f32 %f<100>;
    .shared .align 16 .b8 heirloom_flash_tc_scores[1024];
    .shared .align 16 .b8 heirloom_flash_tc_prob[512];
    .shared .align 4 .b8 heirloom_flash_tc_row_max[64];
    .shared .align 4 .b8 heirloom_flash_tc_row_denom[64];
    .shared .align 4 .b8 heirloom_flash_tc_alpha[64];

    ld.param.u64 %rd1, [query];
    ld.param.u64 %rd2, [key];
    ld.param.u64 %rd3, [value];
    ld.param.u64 %rd4, [output];
    ld.param.u64 %rd5, [row_max_out];
    ld.param.u64 %rd6, [row_denom_out];
    ld.param.u32 %r4, [batch];
    ld.param.u32 %r5, [time];
    ld.param.u32 %r6, [channels];
    ld.param.u32 %r7, [n_heads];
    ld.param.u32 %r8, [head_dim];

    mov.u32 %r10, %tid.x;
    mov.u32 %r11, %ctaid.x;
    mov.u32 %r12, %ctaid.y;
    mov.u32 %r27, %ctaid.z;
    shr.u32 %r13, %r10, 2;
    and.b32 %r14, %r10, 3;
    shl.b32 %r15, %r12, 4;
    shl.b32 %r16, %r11, 3;
    shl.b32 %r18, %r14, 1;
    rem.u32 %r28, %r27, %r7;
    div.u32 %r29, %r27, %r7;
    mul.lo.u32 %r40, %r28, %r8;

    cvt.rn.f32.u32 %f70, %r8;
    sqrt.rn.f32 %f71, %f70;
    mov.f32 %f72, 0f3f800000;
    div.rn.f32 %f73, %f72, %f71;
    mov.f32 %f74, 0f3fb8aa3b;
    mov.f32 %f75, 0fff800000;
    mov.f32 %f76, 0f00000000;
    mov.f32 %f77, 0f3f800000;

    setp.ge.u32 %p1, %r10, 16;
    @%p1 bra FLASH_TC_INIT_DONE;
    mul.wide.u32 %rd20, %r10, 4;
    mov.u64 %rd21, heirloom_flash_tc_row_max;
    add.s64 %rd22, %rd21, %rd20;
    st.shared.f32 [%rd22], %f75;
    mov.u64 %rd23, heirloom_flash_tc_row_denom;
    add.s64 %rd24, %rd23, %rd20;
    st.shared.f32 [%rd24], %f76;
    mov.u64 %rd25, heirloom_flash_tc_alpha;
    add.s64 %rd26, %rd25, %rd20;
    st.shared.f32 [%rd26], %f77;
FLASH_TC_INIT_DONE:
    bar.sync 0;

    mov.f32 %f60, 0f00000000;
    mov.f32 %f61, 0f00000000;
    mov.f32 %f62, 0f00000000;
    mov.f32 %f63, 0f00000000;
    mov.u32 %r50, 0;

FLASH_TC_KEY_BLOCK_LOOP:
    setp.ge.u32 %p1, %r50, %r5;
    @%p1 bra FLASH_TC_STORE_OUTPUT;

    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.f32 %f4, 0f00000000;
    mov.f32 %f5, 0f00000000;
    mov.f32 %f6, 0f00000000;
    mov.f32 %f7, 0f00000000;
    mov.f32 %f8, 0f00000000;
    mov.u32 %r17, 0;

FLASH_TC_QK_HEAD_DIM_LOOP:
    setp.ge.u32 %p2, %r17, %r8;
    @%p2 bra FLASH_TC_QK_DONE;

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r17, %r18;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r41, %r29, %r5;
    add.u32 %r41, %r41, %r19;
    mul.lo.u32 %r41, %r41, %r6;
    add.u32 %r41, %r41, %r40;
    add.u32 %r42, %r41, %r20;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd1, %rd30;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r19, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd31];
    add.u32 %r42, %r41, %r23;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd1, %rd30;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd31];
    shl.b32 %r25, %r24, 16;
    or.b32 %r30, %r22, %r25;

    add.u32 %r19, %r19, 8;
    mul.lo.u32 %r41, %r29, %r5;
    add.u32 %r41, %r41, %r19;
    mul.lo.u32 %r41, %r41, %r6;
    add.u32 %r41, %r41, %r40;
    add.u32 %r42, %r41, %r20;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd1, %rd30;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r19, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd31];
    add.u32 %r42, %r41, %r23;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd1, %rd30;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd31];
    shl.b32 %r25, %r24, 16;
    or.b32 %r31, %r22, %r25;

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r41, %r29, %r5;
    add.u32 %r41, %r41, %r19;
    mul.lo.u32 %r41, %r41, %r6;
    add.u32 %r41, %r41, %r40;
    add.u32 %r42, %r41, %r20;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd1, %rd30;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r19, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd31];
    add.u32 %r42, %r41, %r23;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd1, %rd30;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd31];
    shl.b32 %r25, %r24, 16;
    or.b32 %r32, %r22, %r25;

    add.u32 %r19, %r19, 8;
    mul.lo.u32 %r41, %r29, %r5;
    add.u32 %r41, %r41, %r19;
    mul.lo.u32 %r41, %r41, %r6;
    add.u32 %r41, %r41, %r40;
    add.u32 %r42, %r41, %r20;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd1, %rd30;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r19, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd31];
    add.u32 %r42, %r41, %r23;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd1, %rd30;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd31];
    shl.b32 %r25, %r24, 16;
    or.b32 %r33, %r22, %r25;

    add.u32 %r26, %r50, %r13;
    add.u32 %r20, %r17, %r18;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r43, %r29, %r5;
    add.u32 %r43, %r43, %r26;
    mul.lo.u32 %r43, %r43, %r6;
    add.u32 %r43, %r43, %r40;
    add.u32 %r44, %r43, %r20;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd2, %rd32;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r26, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd33];
    add.u32 %r44, %r43, %r23;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd2, %rd32;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd33];
    shl.b32 %r25, %r24, 16;
    or.b32 %r34, %r22, %r25;

    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    add.u32 %r44, %r43, %r20;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd2, %rd32;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r26, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd33];
    add.u32 %r44, %r43, %r23;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd2, %rd32;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd33];
    shl.b32 %r25, %r24, 16;
    or.b32 %r35, %r22, %r25;

    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f21, %f22, %f23, %f24},
        {%r30, %r31, %r32, %r33},
        {%r34, %r35},
        {%f1, %f2, %f3, %f4};
    mov.f32 %f1, %f21;
    mov.f32 %f2, %f22;
    mov.f32 %f3, %f23;
    mov.f32 %f4, %f24;

    add.u32 %r26, %r50, %r13;
    add.u32 %r26, %r26, 8;
    add.u32 %r20, %r17, %r18;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r43, %r29, %r5;
    add.u32 %r43, %r43, %r26;
    mul.lo.u32 %r43, %r43, %r6;
    add.u32 %r43, %r43, %r40;
    add.u32 %r44, %r43, %r20;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd2, %rd32;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r26, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd33];
    add.u32 %r44, %r43, %r23;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd2, %rd32;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd33];
    shl.b32 %r25, %r24, 16;
    or.b32 %r36, %r22, %r25;

    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    add.u32 %r44, %r43, %r20;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd2, %rd32;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r26, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd33];
    add.u32 %r44, %r43, %r23;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd2, %rd32;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd33];
    shl.b32 %r25, %r24, 16;
    or.b32 %r37, %r22, %r25;

    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f21, %f22, %f23, %f24},
        {%r30, %r31, %r32, %r33},
        {%r36, %r37},
        {%f5, %f6, %f7, %f8};
    mov.f32 %f5, %f21;
    mov.f32 %f6, %f22;
    mov.f32 %f7, %f23;
    mov.f32 %f8, %f24;

    add.u32 %r17, %r17, 16;
    bra FLASH_TC_QK_HEAD_DIM_LOOP;

FLASH_TC_QK_DONE:
    mul.f32 %f1, %f1, %f73;
    mul.f32 %f2, %f2, %f73;
    mul.f32 %f3, %f3, %f73;
    mul.f32 %f4, %f4, %f73;
    mul.f32 %f5, %f5, %f73;
    mul.f32 %f6, %f6, %f73;
    mul.f32 %f7, %f7, %f73;
    mul.f32 %f8, %f8, %f73;

    add.u32 %r55, %r15, %r13;
    add.u32 %r56, %r55, 8;
    add.u32 %r57, %r50, %r18;
    add.u32 %r58, %r57, 1;
    mov.u32 %r59, 16;
    mad.lo.u32 %r60, %r13, %r59, %r18;
    mul.wide.u32 %rd40, %r60, 4;
    mov.u64 %rd41, heirloom_flash_tc_scores;
    add.s64 %rd42, %rd41, %rd40;

    setp.lt.u32 %p2, %r55, %r5;
    setp.lt.u32 %p3, %r57, %r5;
    setp.le.u32 %p4, %r57, %r55;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f1, %f75, %p5;
    st.shared.f32 [%rd42], %f20;
    add.s64 %rd43, %rd42, 4;
    setp.lt.u32 %p3, %r58, %r5;
    setp.le.u32 %p4, %r58, %r55;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f2, %f75, %p5;
    st.shared.f32 [%rd43], %f20;

    add.u32 %r60, %r13, 8;
    mad.lo.u32 %r60, %r60, %r59, %r18;
    mul.wide.u32 %rd40, %r60, 4;
    add.s64 %rd42, %rd41, %rd40;
    setp.lt.u32 %p2, %r56, %r5;
    setp.lt.u32 %p3, %r57, %r5;
    setp.le.u32 %p4, %r57, %r56;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f3, %f75, %p5;
    st.shared.f32 [%rd42], %f20;
    add.s64 %rd43, %rd42, 4;
    setp.lt.u32 %p3, %r58, %r5;
    setp.le.u32 %p4, %r58, %r56;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f4, %f75, %p5;
    st.shared.f32 [%rd43], %f20;

    add.u32 %r57, %r50, %r18;
    add.u32 %r57, %r57, 8;
    add.u32 %r58, %r57, 1;
    mad.lo.u32 %r60, %r13, %r59, %r18;
    add.u32 %r60, %r60, 8;
    mul.wide.u32 %rd40, %r60, 4;
    add.s64 %rd42, %rd41, %rd40;
    setp.lt.u32 %p2, %r55, %r5;
    setp.lt.u32 %p3, %r57, %r5;
    setp.le.u32 %p4, %r57, %r55;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f5, %f75, %p5;
    st.shared.f32 [%rd42], %f20;
    add.s64 %rd43, %rd42, 4;
    setp.lt.u32 %p3, %r58, %r5;
    setp.le.u32 %p4, %r58, %r55;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f6, %f75, %p5;
    st.shared.f32 [%rd43], %f20;

    add.u32 %r60, %r13, 8;
    mad.lo.u32 %r60, %r60, %r59, %r18;
    add.u32 %r60, %r60, 8;
    mul.wide.u32 %rd40, %r60, 4;
    add.s64 %rd42, %rd41, %rd40;
    setp.lt.u32 %p2, %r56, %r5;
    setp.lt.u32 %p3, %r57, %r5;
    setp.le.u32 %p4, %r57, %r56;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f7, %f75, %p5;
    st.shared.f32 [%rd42], %f20;
    add.s64 %rd43, %rd42, 4;
    setp.lt.u32 %p3, %r58, %r5;
    setp.le.u32 %p4, %r58, %r56;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f8, %f75, %p5;
    st.shared.f32 [%rd43], %f20;
    bar.sync 0;

    setp.ge.u32 %p1, %r10, 16;
    @%p1 bra FLASH_TC_SOFTMAX_DONE;
    add.u32 %r61, %r15, %r10;
    setp.ge.u32 %p2, %r61, %r5;
    @%p2 bra FLASH_TC_SOFTMAX_INVALID_ROW;
    mul.wide.u32 %rd44, %r10, 4;
    mov.u64 %rd45, heirloom_flash_tc_row_max;
    add.s64 %rd46, %rd45, %rd44;
    ld.shared.f32 %f30, [%rd46];
    mov.u64 %rd47, heirloom_flash_tc_row_denom;
    add.s64 %rd48, %rd47, %rd44;
    ld.shared.f32 %f31, [%rd48];
    mov.f32 %f32, 0fff800000;
    mov.u32 %r62, 0;
FLASH_TC_BLOCK_MAX_LOOP:
    setp.ge.u32 %p3, %r62, 16;
    @%p3 bra FLASH_TC_BLOCK_MAX_DONE;
    mad.lo.u32 %r63, %r10, %r59, %r62;
    mul.wide.u32 %rd49, %r63, 4;
    mov.u64 %rd50, heirloom_flash_tc_scores;
    add.s64 %rd51, %rd50, %rd49;
    ld.shared.f32 %f33, [%rd51];
    max.f32 %f32, %f32, %f33;
    add.u32 %r62, %r62, 1;
    bra FLASH_TC_BLOCK_MAX_LOOP;
FLASH_TC_BLOCK_MAX_DONE:
    max.f32 %f34, %f30, %f32;
    sub.f32 %f35, %f30, %f34;
    mul.f32 %f35, %f35, %f74;
    ex2.approx.ftz.f32 %f36, %f35;
    mov.f32 %f37, 0f00000000;
    mov.u32 %r62, 0;
FLASH_TC_PROB_LOOP:
    setp.ge.u32 %p3, %r62, 16;
    @%p3 bra FLASH_TC_PROB_DONE;
    mad.lo.u32 %r63, %r10, %r59, %r62;
    mul.wide.u32 %rd49, %r63, 4;
    mov.u64 %rd50, heirloom_flash_tc_scores;
    add.s64 %rd51, %rd50, %rd49;
    ld.shared.f32 %f33, [%rd51];
    sub.f32 %f38, %f33, %f34;
    mul.f32 %f38, %f38, %f74;
    ex2.approx.ftz.f32 %f39, %f38;
    add.f32 %f37, %f37, %f39;
    mul.wide.u32 %rd52, %r63, 2;
    mov.u64 %rd53, heirloom_flash_tc_prob;
    add.s64 %rd54, %rd53, %rd52;
    mov.b32 %r64, %f39;
    shr.u32 %r64, %r64, 16;
    st.shared.u16 [%rd54], %r64;
    add.u32 %r62, %r62, 1;
    bra FLASH_TC_PROB_LOOP;
FLASH_TC_PROB_DONE:
    fma.rn.f32 %f40, %f31, %f36, %f37;
    st.shared.f32 [%rd46], %f34;
    st.shared.f32 [%rd48], %f40;
    mov.u64 %rd55, heirloom_flash_tc_alpha;
    add.s64 %rd56, %rd55, %rd44;
    st.shared.f32 [%rd56], %f36;
    bra FLASH_TC_SOFTMAX_DONE;

FLASH_TC_SOFTMAX_INVALID_ROW:
    mov.u32 %r62, 0;
FLASH_TC_ZERO_PROB_LOOP:
    setp.ge.u32 %p3, %r62, 16;
    @%p3 bra FLASH_TC_ZERO_PROB_DONE;
    mad.lo.u32 %r63, %r10, %r59, %r62;
    mul.wide.u32 %rd52, %r63, 2;
    mov.u64 %rd53, heirloom_flash_tc_prob;
    add.s64 %rd54, %rd53, %rd52;
    mov.u32 %r64, 0;
    st.shared.u16 [%rd54], %r64;
    add.u32 %r62, %r62, 1;
    bra FLASH_TC_ZERO_PROB_LOOP;
FLASH_TC_ZERO_PROB_DONE:
    mul.wide.u32 %rd44, %r10, 4;
    mov.u64 %rd55, heirloom_flash_tc_alpha;
    add.s64 %rd56, %rd55, %rd44;
    st.shared.f32 [%rd56], %f77;
FLASH_TC_SOFTMAX_DONE:
    bar.sync 0;

    add.u32 %r19, %r13, 0;
    add.u32 %r20, %r18, 0;
    mad.lo.u32 %r21, %r19, %r59, %r20;
    mul.wide.u32 %rd60, %r21, 2;
    mov.u64 %rd61, heirloom_flash_tc_prob;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r22, [%rd62];
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r19, %r59, %r23;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r24, [%rd62];
    shl.b32 %r25, %r24, 16;
    or.b32 %r30, %r22, %r25;

    add.u32 %r19, %r19, 8;
    mad.lo.u32 %r21, %r19, %r59, %r20;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r22, [%rd62];
    mad.lo.u32 %r21, %r19, %r59, %r23;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r24, [%rd62];
    shl.b32 %r25, %r24, 16;
    or.b32 %r31, %r22, %r25;

    add.u32 %r19, %r13, 0;
    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r19, %r59, %r20;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r22, [%rd62];
    mad.lo.u32 %r21, %r19, %r59, %r23;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r24, [%rd62];
    shl.b32 %r25, %r24, 16;
    or.b32 %r32, %r22, %r25;

    add.u32 %r19, %r19, 8;
    mad.lo.u32 %r21, %r19, %r59, %r20;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r22, [%rd62];
    mad.lo.u32 %r21, %r19, %r59, %r23;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r24, [%rd62];
    shl.b32 %r25, %r24, 16;
    or.b32 %r33, %r22, %r25;

    add.u32 %r26, %r16, %r13;
    add.u32 %r20, %r50, %r18;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r43, %r29, %r5;
    add.u32 %r43, %r43, %r20;
    mul.lo.u32 %r43, %r43, %r6;
    add.u32 %r43, %r43, %r40;
    add.u32 %r43, %r43, %r26;
    mul.wide.u32 %rd63, %r43, 2;
    add.s64 %rd64, %rd3, %rd63;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r20, %r5;
    setp.lt.u32 %p4, %r26, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd64];
    add.u32 %r44, %r43, %r6;
    mul.wide.u32 %rd63, %r44, 2;
    add.s64 %rd64, %rd3, %rd63;
    mov.u32 %r24, 0;
    setp.lt.u32 %p3, %r23, %r5;
    setp.lt.u32 %p4, %r26, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd64];
    shl.b32 %r25, %r24, 16;
    or.b32 %r34, %r22, %r25;

    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r43, %r29, %r5;
    add.u32 %r43, %r43, %r20;
    mul.lo.u32 %r43, %r43, %r6;
    add.u32 %r43, %r43, %r40;
    add.u32 %r43, %r43, %r26;
    mul.wide.u32 %rd63, %r43, 2;
    add.s64 %rd64, %rd3, %rd63;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r20, %r5;
    setp.lt.u32 %p4, %r26, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd64];
    add.u32 %r44, %r43, %r6;
    mul.wide.u32 %rd63, %r44, 2;
    add.s64 %rd64, %rd3, %rd63;
    mov.u32 %r24, 0;
    setp.lt.u32 %p3, %r23, %r5;
    setp.lt.u32 %p4, %r26, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd64];
    shl.b32 %r25, %r24, 16;
    or.b32 %r35, %r22, %r25;

    mov.f32 %f41, 0f00000000;
    mov.f32 %f42, 0f00000000;
    mov.f32 %f43, 0f00000000;
    mov.f32 %f44, 0f00000000;
    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f45, %f46, %f47, %f48},
        {%r30, %r31, %r32, %r33},
        {%r34, %r35},
        {%f41, %f42, %f43, %f44};

    mul.wide.u32 %rd65, %r13, 4;
    mov.u64 %rd66, heirloom_flash_tc_alpha;
    add.s64 %rd67, %rd66, %rd65;
    ld.shared.f32 %f49, [%rd67];
    add.u32 %r65, %r13, 8;
    mul.wide.u32 %rd65, %r65, 4;
    add.s64 %rd67, %rd66, %rd65;
    ld.shared.f32 %f50, [%rd67];

    fma.rn.f32 %f60, %f60, %f49, %f45;
    fma.rn.f32 %f61, %f61, %f49, %f46;
    fma.rn.f32 %f62, %f62, %f50, %f47;
    fma.rn.f32 %f63, %f63, %f50, %f48;

    bar.sync 0;
    add.u32 %r50, %r50, 16;
    bra FLASH_TC_KEY_BLOCK_LOOP;

FLASH_TC_STORE_OUTPUT:
    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r16, %r18;
    mul.wide.u32 %rd70, %r13, 4;
    mov.u64 %rd71, heirloom_flash_tc_row_denom;
    add.s64 %rd72, %rd71, %rd70;
    ld.shared.f32 %f51, [%rd72];
    div.rn.f32 %f60, %f60, %f51;
    div.rn.f32 %f61, %f61, %f51;
    mul.lo.u32 %r45, %r29, %r5;
    add.u32 %r45, %r45, %r19;
    mul.lo.u32 %r45, %r45, %r6;
    add.u32 %r45, %r45, %r40;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd73, %r46, 4;
    add.s64 %rd74, %rd4, %rd73;
    setp.lt.u32 %p2, %r19, %r5;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 st.global.f32 [%rd74], %f60;
    add.u32 %r20, %r20, 1;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd73, %r46, 4;
    add.s64 %rd74, %rd4, %rd73;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 st.global.f32 [%rd74], %f61;

    add.u32 %r19, %r19, 8;
    sub.u32 %r20, %r20, 1;
    add.u32 %r65, %r13, 8;
    mul.wide.u32 %rd70, %r65, 4;
    add.s64 %rd72, %rd71, %rd70;
    ld.shared.f32 %f52, [%rd72];
    div.rn.f32 %f62, %f62, %f52;
    div.rn.f32 %f63, %f63, %f52;
    mul.lo.u32 %r45, %r29, %r5;
    add.u32 %r45, %r45, %r19;
    mul.lo.u32 %r45, %r45, %r6;
    add.u32 %r45, %r45, %r40;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd73, %r46, 4;
    add.s64 %rd74, %rd4, %rd73;
    setp.lt.u32 %p2, %r19, %r5;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 st.global.f32 [%rd74], %f62;
    add.u32 %r20, %r20, 1;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd73, %r46, 4;
    add.s64 %rd74, %rd4, %rd73;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 st.global.f32 [%rd74], %f63;

    setp.ne.u32 %p1, %r11, 0;
    @%p1 bra FLASH_TC_DONE;
    setp.ge.u32 %p1, %r10, 16;
    @%p1 bra FLASH_TC_DONE;
    add.u32 %r19, %r15, %r10;
    setp.ge.u32 %p1, %r19, %r5;
    @%p1 bra FLASH_TC_DONE;
    mul.lo.u32 %r66, %r27, %r5;
    add.u32 %r66, %r66, %r19;
    mul.wide.u32 %rd75, %r66, 4;
    mov.u64 %rd76, heirloom_flash_tc_row_max;
    mul.wide.u32 %rd77, %r10, 4;
    add.s64 %rd78, %rd76, %rd77;
    ld.shared.f32 %f53, [%rd78];
    add.s64 %rd79, %rd5, %rd75;
    st.global.f32 [%rd79], %f53;
    mov.u64 %rd80, heirloom_flash_tc_row_denom;
    add.s64 %rd81, %rd80, %rd77;
    ld.shared.f32 %f54, [%rd81];
    add.s64 %rd82, %rd6, %rd75;
    st.global.f32 [%rd82], %f54;
FLASH_TC_DONE:
    ret;
}

.visible .entry heirloom_flash_attention_bf16_fwd_f32(
    .param .u64 query,
    .param .u64 key,
    .param .u64 value,
    .param .u64 output,
    .param .u32 batch,
    .param .u32 time,
    .param .u32 channels,
    .param .u32 n_heads,
    .param .u32 head_dim,
    .param .u32 total
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<70>;
    .reg .b64 %rd<44>;
    .reg .f32 %f<40>;

    ld.param.u64 %rd1, [query];
    ld.param.u64 %rd2, [key];
    ld.param.u64 %rd3, [value];
    ld.param.u64 %rd4, [output];
    ld.param.u32 %r4, [batch];
    ld.param.u32 %r5, [time];
    ld.param.u32 %r6, [channels];
    ld.param.u32 %r7, [n_heads];
    ld.param.u32 %r8, [head_dim];
    ld.param.u32 %r9, [total];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r10, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r10, %r9;
    @%p1 bra DONE_FLASH_ATTN_BF16_FWD;

    rem.u32 %r11, %r10, %r6;
    div.u32 %r12, %r10, %r6;
    rem.u32 %r13, %r12, %r5;
    div.u32 %r14, %r12, %r5;
    div.u32 %r15, %r11, %r8;
    rem.u32 %r16, %r11, %r8;

    cvt.rn.f32.u32 %f1, %r8;
    sqrt.rn.f32 %f2, %f1;
    mov.f32 %f3, 0f3f800000;
    div.rn.f32 %f4, %f3, %f2;
    mov.f32 %f5, 0fff800000;
    mov.f32 %f6, 0f00000000;
    mov.f32 %f7, 0f00000000;
    mov.f32 %f20, 0f3fb8aa3b;
    mov.u32 %r17, 0;

FLASH_ATTN_TOKEN_LOOP:
    setp.gt.u32 %p2, %r17, %r13;
    @%p2 bra FLASH_ATTN_TOKEN_DONE;

    mov.f32 %f8, 0f00000000;
    mov.u32 %r18, 0;

FLASH_ATTN_HEAD_DIM_LOOP:
    setp.ge.u32 %p3, %r18, %r8;
    @%p3 bra FLASH_ATTN_DOT_DONE;

    mul.lo.u32 %r20, %r14, %r5;
    add.u32 %r20, %r20, %r13;
    mul.lo.u32 %r20, %r20, %r6;
    mul.lo.u32 %r21, %r15, %r8;
    add.u32 %r22, %r21, %r18;
    add.u32 %r23, %r20, %r22;
    mul.wide.u32 %rd10, %r23, 2;
    add.s64 %rd11, %rd1, %rd10;
    ld.global.u16 %r24, [%rd11];
    shl.b32 %r25, %r24, 16;
    mov.b32 %f9, %r25;

    mul.lo.u32 %r26, %r14, %r5;
    add.u32 %r26, %r26, %r17;
    mul.lo.u32 %r26, %r26, %r6;
    add.u32 %r27, %r26, %r22;
    mul.wide.u32 %rd12, %r27, 2;
    add.s64 %rd13, %rd2, %rd12;
    ld.global.u16 %r28, [%rd13];
    shl.b32 %r29, %r28, 16;
    mov.b32 %f10, %r29;

    fma.rn.f32 %f8, %f9, %f10, %f8;
    add.u32 %r18, %r18, 1;
    bra FLASH_ATTN_HEAD_DIM_LOOP;

FLASH_ATTN_DOT_DONE:
    mul.f32 %f11, %f8, %f4;
    max.f32 %f12, %f5, %f11;
    sub.f32 %f13, %f5, %f12;
    mul.f32 %f14, %f13, %f20;
    ex2.approx.ftz.f32 %f15, %f14;
    sub.f32 %f16, %f11, %f12;
    mul.f32 %f17, %f16, %f20;
    ex2.approx.ftz.f32 %f18, %f17;

    mul.lo.u32 %r30, %r14, %r5;
    add.u32 %r30, %r30, %r17;
    mul.lo.u32 %r30, %r30, %r6;
    mul.lo.u32 %r31, %r15, %r8;
    add.u32 %r31, %r31, %r16;
    add.u32 %r32, %r30, %r31;
    mul.wide.u32 %rd14, %r32, 2;
    add.s64 %rd15, %rd3, %rd14;
    ld.global.u16 %r33, [%rd15];
    shl.b32 %r34, %r33, 16;
    mov.b32 %f19, %r34;

    mul.f32 %f21, %f7, %f15;
    fma.rn.f32 %f7, %f18, %f19, %f21;
    fma.rn.f32 %f6, %f6, %f15, %f18;
    mov.f32 %f5, %f12;
    add.u32 %r17, %r17, 1;
    bra FLASH_ATTN_TOKEN_LOOP;

FLASH_ATTN_TOKEN_DONE:
    div.rn.f32 %f22, %f7, %f6;
    mul.wide.u32 %rd16, %r10, 4;
    add.s64 %rd17, %rd4, %rd16;
    st.global.f32 [%rd17], %f22;

DONE_FLASH_ATTN_BF16_FWD:
    ret;
}

.visible .entry heirloom_flash_attention_bf16_bwd_row_dot_f32(
    .param .u64 output,
    .param .u64 grad_output,
    .param .u64 row_dot,
    .param .u32 batch,
    .param .u32 time,
    .param .u32 channels,
    .param .u32 n_heads,
    .param .u32 head_dim,
    .param .u32 total
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<20>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [output];
    ld.param.u64 %rd2, [grad_output];
    ld.param.u64 %rd3, [row_dot];
    ld.param.u32 %r4, [batch];
    ld.param.u32 %r5, [time];
    ld.param.u32 %r6, [channels];
    ld.param.u32 %r7, [n_heads];
    ld.param.u32 %r8, [head_dim];
    ld.param.u32 %r9, [total];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r10, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r10, %r9;
    @%p1 bra DONE_FLASH_ATTN_BF16_ROW_DOT;

    rem.u32 %r11, %r10, %r5;
    div.u32 %r12, %r10, %r5;
    rem.u32 %r13, %r12, %r7;
    div.u32 %r14, %r12, %r7;
    mul.lo.u32 %r15, %r13, %r8;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r16, 0;

FLASH_ATTN_BWD_ROW_DOT_LOOP:
    setp.ge.u32 %p2, %r16, %r8;
    @%p2 bra FLASH_ATTN_BWD_ROW_DOT_STORE;
    mul.lo.u32 %r17, %r14, %r5;
    add.u32 %r17, %r17, %r11;
    mul.lo.u32 %r17, %r17, %r6;
    add.u32 %r18, %r15, %r16;
    add.u32 %r17, %r17, %r18;
    mul.wide.u32 %rd4, %r17, 4;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];
    mul.wide.u32 %rd6, %r17, 2;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.u16 %r19, [%rd7];
    shl.b32 %r20, %r19, 16;
    mov.b32 %f3, %r20;
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r16, %r16, 1;
    bra FLASH_ATTN_BWD_ROW_DOT_LOOP;

FLASH_ATTN_BWD_ROW_DOT_STORE:
    mul.wide.u32 %rd8, %r10, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f1;

DONE_FLASH_ATTN_BF16_ROW_DOT:
    ret;
}

.visible .entry heirloom_flash_attention_bf16_tc_bwd_tiled_mma_f32(
    .param .u64 query,
    .param .u64 key,
    .param .u64 value,
    .param .u64 output,
    .param .u64 row_max,
    .param .u64 row_denom,
    .param .u64 grad_output,
    .param .u64 row_dot,
    .param .u64 grad_query,
    .param .u64 grad_key,
    .param .u64 grad_value,
    .param .u32 batch,
    .param .u32 time,
    .param .u32 channels,
    .param .u32 n_heads,
    .param .u32 head_dim
)
{
    .reg .pred %p<48>;
    .reg .b32 %r<280>;
    .reg .b64 %rd<200>;
    .reg .f32 %f<190>;
    .shared .align 16 .b8 heirloom_flash_bwd_scores[1024];
    .shared .align 16 .b8 heirloom_flash_bwd_dp[1024];
    .shared .align 16 .b8 heirloom_flash_bwd_p[512];
    .shared .align 16 .b8 heirloom_flash_bwd_pt[512];
    .shared .align 16 .b8 heirloom_flash_bwd_ds[512];
    .shared .align 16 .b8 heirloom_flash_bwd_dst[512];

    ld.param.u64 %rd1, [query];
    ld.param.u64 %rd2, [key];
    ld.param.u64 %rd3, [value];
    ld.param.u64 %rd4, [output];
    ld.param.u64 %rd5, [row_max];
    ld.param.u64 %rd6, [row_denom];
    ld.param.u64 %rd7, [grad_output];
    ld.param.u64 %rd8, [row_dot];
    ld.param.u64 %rd9, [grad_query];
    ld.param.u64 %rd10, [grad_key];
    ld.param.u64 %rd11, [grad_value];
    ld.param.u32 %r4, [batch];
    ld.param.u32 %r5, [time];
    ld.param.u32 %r6, [channels];
    ld.param.u32 %r7, [n_heads];
    ld.param.u32 %r8, [head_dim];

    mov.u32 %r10, %tid.x;
    mov.u32 %r11, %ctaid.x;
    mov.u32 %r12, %ctaid.y;
    mov.u32 %r27, %ctaid.z;
    shr.u32 %r13, %r10, 2;
    and.b32 %r14, %r10, 3;
    shl.b32 %r18, %r14, 1;
    div.u32 %r90, %r8, 8;
    rem.u32 %r91, %r11, %r90;
    div.u32 %r92, %r11, %r90;
    shl.b32 %r15, %r12, 4;
    shl.b32 %r50, %r92, 4;
    shl.b32 %r16, %r91, 3;
    rem.u32 %r28, %r27, %r7;
    div.u32 %r29, %r27, %r7;
    mul.lo.u32 %r40, %r28, %r8;
    mov.u32 %r59, 16;

    cvt.rn.f32.u32 %f70, %r8;
    sqrt.rn.f32 %f71, %f70;
    mov.f32 %f72, 0f3f800000;
    div.rn.f32 %f73, %f72, %f71;
    mov.f32 %f74, 0f3fb8aa3b;
    mov.f32 %f75, 0fff800000;
    mov.f32 %f76, 0f00000000;

    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.f32 %f4, 0f00000000;
    mov.f32 %f5, 0f00000000;
    mov.f32 %f6, 0f00000000;
    mov.f32 %f7, 0f00000000;
    mov.f32 %f8, 0f00000000;
    mov.u32 %r17, 0;

FLASH_BWD_QK_HEAD_DIM_LOOP:
    setp.ge.u32 %p1, %r17, %r8;
    @%p1 bra FLASH_BWD_QK_DONE;

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r17, %r18;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r41, %r29, %r5;
    add.u32 %r41, %r41, %r19;
    mul.lo.u32 %r41, %r41, %r6;
    add.u32 %r41, %r41, %r40;
    add.u32 %r42, %r41, %r20;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd1, %rd30;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r19, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd31];
    add.u32 %r42, %r41, %r23;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd1, %rd30;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd31];
    shl.b32 %r25, %r24, 16;
    or.b32 %r30, %r22, %r25;

    add.u32 %r19, %r19, 8;
    mul.lo.u32 %r41, %r29, %r5;
    add.u32 %r41, %r41, %r19;
    mul.lo.u32 %r41, %r41, %r6;
    add.u32 %r41, %r41, %r40;
    add.u32 %r42, %r41, %r20;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd1, %rd30;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r19, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd31];
    add.u32 %r42, %r41, %r23;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd1, %rd30;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd31];
    shl.b32 %r25, %r24, 16;
    or.b32 %r31, %r22, %r25;

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r41, %r29, %r5;
    add.u32 %r41, %r41, %r19;
    mul.lo.u32 %r41, %r41, %r6;
    add.u32 %r41, %r41, %r40;
    add.u32 %r42, %r41, %r20;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd1, %rd30;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r19, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd31];
    add.u32 %r42, %r41, %r23;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd1, %rd30;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd31];
    shl.b32 %r25, %r24, 16;
    or.b32 %r32, %r22, %r25;

    add.u32 %r19, %r19, 8;
    mul.lo.u32 %r41, %r29, %r5;
    add.u32 %r41, %r41, %r19;
    mul.lo.u32 %r41, %r41, %r6;
    add.u32 %r41, %r41, %r40;
    add.u32 %r42, %r41, %r20;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd1, %rd30;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r19, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd31];
    add.u32 %r42, %r41, %r23;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd1, %rd30;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd31];
    shl.b32 %r25, %r24, 16;
    or.b32 %r33, %r22, %r25;

    add.u32 %r26, %r50, %r13;
    add.u32 %r20, %r17, %r18;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r43, %r29, %r5;
    add.u32 %r43, %r43, %r26;
    mul.lo.u32 %r43, %r43, %r6;
    add.u32 %r43, %r43, %r40;
    add.u32 %r44, %r43, %r20;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd2, %rd32;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r26, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd33];
    add.u32 %r44, %r43, %r23;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd2, %rd32;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd33];
    shl.b32 %r25, %r24, 16;
    or.b32 %r34, %r22, %r25;

    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    add.u32 %r44, %r43, %r20;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd2, %rd32;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r26, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd33];
    add.u32 %r44, %r43, %r23;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd2, %rd32;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd33];
    shl.b32 %r25, %r24, 16;
    or.b32 %r35, %r22, %r25;

    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f21, %f22, %f23, %f24},
        {%r30, %r31, %r32, %r33},
        {%r34, %r35},
        {%f1, %f2, %f3, %f4};
    mov.f32 %f1, %f21;
    mov.f32 %f2, %f22;
    mov.f32 %f3, %f23;
    mov.f32 %f4, %f24;

    add.u32 %r26, %r50, %r13;
    add.u32 %r26, %r26, 8;
    add.u32 %r20, %r17, %r18;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r43, %r29, %r5;
    add.u32 %r43, %r43, %r26;
    mul.lo.u32 %r43, %r43, %r6;
    add.u32 %r43, %r43, %r40;
    add.u32 %r44, %r43, %r20;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd2, %rd32;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r26, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd33];
    add.u32 %r44, %r43, %r23;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd2, %rd32;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd33];
    shl.b32 %r25, %r24, 16;
    or.b32 %r36, %r22, %r25;

    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    add.u32 %r44, %r43, %r20;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd2, %rd32;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r26, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd33];
    add.u32 %r44, %r43, %r23;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd2, %rd32;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd33];
    shl.b32 %r25, %r24, 16;
    or.b32 %r37, %r22, %r25;

    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f21, %f22, %f23, %f24},
        {%r30, %r31, %r32, %r33},
        {%r36, %r37},
        {%f5, %f6, %f7, %f8};
    mov.f32 %f5, %f21;
    mov.f32 %f6, %f22;
    mov.f32 %f7, %f23;
    mov.f32 %f8, %f24;

    add.u32 %r17, %r17, 16;
    bra FLASH_BWD_QK_HEAD_DIM_LOOP;

FLASH_BWD_QK_DONE:
    mul.f32 %f1, %f1, %f73;
    mul.f32 %f2, %f2, %f73;
    mul.f32 %f3, %f3, %f73;
    mul.f32 %f4, %f4, %f73;
    mul.f32 %f5, %f5, %f73;
    mul.f32 %f6, %f6, %f73;
    mul.f32 %f7, %f7, %f73;
    mul.f32 %f8, %f8, %f73;

    add.u32 %r55, %r15, %r13;
    add.u32 %r56, %r55, 8;
    add.u32 %r57, %r50, %r18;
    add.u32 %r58, %r57, 1;
    mad.lo.u32 %r60, %r13, %r59, %r18;
    mul.wide.u32 %rd40, %r60, 4;
    mov.u64 %rd41, heirloom_flash_bwd_scores;
    add.s64 %rd42, %rd41, %rd40;
    setp.lt.u32 %p2, %r55, %r5;
    setp.lt.u32 %p3, %r57, %r5;
    setp.le.u32 %p4, %r57, %r55;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f1, %f75, %p5;
    st.shared.f32 [%rd42], %f20;
    add.s64 %rd43, %rd42, 4;
    setp.lt.u32 %p3, %r58, %r5;
    setp.le.u32 %p4, %r58, %r55;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f2, %f75, %p5;
    st.shared.f32 [%rd43], %f20;

    add.u32 %r60, %r13, 8;
    mad.lo.u32 %r60, %r60, %r59, %r18;
    mul.wide.u32 %rd40, %r60, 4;
    add.s64 %rd42, %rd41, %rd40;
    setp.lt.u32 %p2, %r56, %r5;
    setp.lt.u32 %p3, %r57, %r5;
    setp.le.u32 %p4, %r57, %r56;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f3, %f75, %p5;
    st.shared.f32 [%rd42], %f20;
    add.s64 %rd43, %rd42, 4;
    setp.lt.u32 %p3, %r58, %r5;
    setp.le.u32 %p4, %r58, %r56;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f4, %f75, %p5;
    st.shared.f32 [%rd43], %f20;

    add.u32 %r57, %r50, %r18;
    add.u32 %r57, %r57, 8;
    add.u32 %r58, %r57, 1;
    mad.lo.u32 %r60, %r13, %r59, %r18;
    add.u32 %r60, %r60, 8;
    mul.wide.u32 %rd40, %r60, 4;
    add.s64 %rd42, %rd41, %rd40;
    setp.lt.u32 %p2, %r55, %r5;
    setp.lt.u32 %p3, %r57, %r5;
    setp.le.u32 %p4, %r57, %r55;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f5, %f75, %p5;
    st.shared.f32 [%rd42], %f20;
    add.s64 %rd43, %rd42, 4;
    setp.lt.u32 %p3, %r58, %r5;
    setp.le.u32 %p4, %r58, %r55;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f6, %f75, %p5;
    st.shared.f32 [%rd43], %f20;

    add.u32 %r60, %r13, 8;
    mad.lo.u32 %r60, %r60, %r59, %r18;
    add.u32 %r60, %r60, 8;
    mul.wide.u32 %rd40, %r60, 4;
    add.s64 %rd42, %rd41, %rd40;
    setp.lt.u32 %p2, %r56, %r5;
    setp.lt.u32 %p3, %r57, %r5;
    setp.le.u32 %p4, %r57, %r56;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f7, %f75, %p5;
    st.shared.f32 [%rd42], %f20;
    add.s64 %rd43, %rd42, 4;
    setp.lt.u32 %p3, %r58, %r5;
    setp.le.u32 %p4, %r58, %r56;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f8, %f75, %p5;
    st.shared.f32 [%rd43], %f20;

    mov.f32 %f9, 0f00000000;
    mov.f32 %f10, 0f00000000;
    mov.f32 %f11, 0f00000000;
    mov.f32 %f12, 0f00000000;
    mov.f32 %f13, 0f00000000;
    mov.f32 %f14, 0f00000000;
    mov.f32 %f15, 0f00000000;
    mov.f32 %f16, 0f00000000;
    mov.u32 %r17, 0;

FLASH_BWD_DP_HEAD_DIM_LOOP:
    setp.ge.u32 %p1, %r17, %r8;
    @%p1 bra FLASH_BWD_DP_DONE;

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r17, %r18;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r41, %r29, %r5;
    add.u32 %r41, %r41, %r19;
    mul.lo.u32 %r41, %r41, %r6;
    add.u32 %r41, %r41, %r40;
    add.u32 %r42, %r41, %r20;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd7, %rd30;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r19, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd31];
    add.u32 %r42, %r41, %r23;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd7, %rd30;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd31];
    shl.b32 %r25, %r24, 16;
    or.b32 %r30, %r22, %r25;

    add.u32 %r19, %r19, 8;
    mul.lo.u32 %r41, %r29, %r5;
    add.u32 %r41, %r41, %r19;
    mul.lo.u32 %r41, %r41, %r6;
    add.u32 %r41, %r41, %r40;
    add.u32 %r42, %r41, %r20;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd7, %rd30;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r19, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd31];
    add.u32 %r42, %r41, %r23;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd7, %rd30;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd31];
    shl.b32 %r25, %r24, 16;
    or.b32 %r31, %r22, %r25;

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r41, %r29, %r5;
    add.u32 %r41, %r41, %r19;
    mul.lo.u32 %r41, %r41, %r6;
    add.u32 %r41, %r41, %r40;
    add.u32 %r42, %r41, %r20;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd7, %rd30;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r19, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd31];
    add.u32 %r42, %r41, %r23;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd7, %rd30;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd31];
    shl.b32 %r25, %r24, 16;
    or.b32 %r32, %r22, %r25;

    add.u32 %r19, %r19, 8;
    mul.lo.u32 %r41, %r29, %r5;
    add.u32 %r41, %r41, %r19;
    mul.lo.u32 %r41, %r41, %r6;
    add.u32 %r41, %r41, %r40;
    add.u32 %r42, %r41, %r20;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd7, %rd30;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r19, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd31];
    add.u32 %r42, %r41, %r23;
    mul.wide.u32 %rd30, %r42, 2;
    add.s64 %rd31, %rd7, %rd30;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd31];
    shl.b32 %r25, %r24, 16;
    or.b32 %r33, %r22, %r25;

    add.u32 %r26, %r50, %r13;
    add.u32 %r20, %r17, %r18;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r43, %r29, %r5;
    add.u32 %r43, %r43, %r26;
    mul.lo.u32 %r43, %r43, %r6;
    add.u32 %r43, %r43, %r40;
    add.u32 %r44, %r43, %r20;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd3, %rd32;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r26, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd33];
    add.u32 %r44, %r43, %r23;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd3, %rd32;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd33];
    shl.b32 %r25, %r24, 16;
    or.b32 %r34, %r22, %r25;

    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    add.u32 %r44, %r43, %r20;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd3, %rd32;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r26, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd33];
    add.u32 %r44, %r43, %r23;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd3, %rd32;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd33];
    shl.b32 %r25, %r24, 16;
    or.b32 %r35, %r22, %r25;

    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f21, %f22, %f23, %f24},
        {%r30, %r31, %r32, %r33},
        {%r34, %r35},
        {%f9, %f10, %f11, %f12};
    mov.f32 %f9, %f21;
    mov.f32 %f10, %f22;
    mov.f32 %f11, %f23;
    mov.f32 %f12, %f24;

    add.u32 %r26, %r50, %r13;
    add.u32 %r26, %r26, 8;
    add.u32 %r20, %r17, %r18;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r43, %r29, %r5;
    add.u32 %r43, %r43, %r26;
    mul.lo.u32 %r43, %r43, %r6;
    add.u32 %r43, %r43, %r40;
    add.u32 %r44, %r43, %r20;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd3, %rd32;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r26, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd33];
    add.u32 %r44, %r43, %r23;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd3, %rd32;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd33];
    shl.b32 %r25, %r24, 16;
    or.b32 %r36, %r22, %r25;

    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    add.u32 %r44, %r43, %r20;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd3, %rd32;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r26, %r5;
    setp.lt.u32 %p4, %r20, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd33];
    add.u32 %r44, %r43, %r23;
    mul.wide.u32 %rd32, %r44, 2;
    add.s64 %rd33, %rd3, %rd32;
    mov.u32 %r24, 0;
    setp.lt.u32 %p4, %r23, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd33];
    shl.b32 %r25, %r24, 16;
    or.b32 %r37, %r22, %r25;

    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f21, %f22, %f23, %f24},
        {%r30, %r31, %r32, %r33},
        {%r36, %r37},
        {%f13, %f14, %f15, %f16};
    mov.f32 %f13, %f21;
    mov.f32 %f14, %f22;
    mov.f32 %f15, %f23;
    mov.f32 %f16, %f24;

    add.u32 %r17, %r17, 16;
    bra FLASH_BWD_DP_HEAD_DIM_LOOP;

FLASH_BWD_DP_DONE:
    mov.u64 %rd41, heirloom_flash_bwd_dp;
    add.u32 %r55, %r15, %r13;
    add.u32 %r56, %r55, 8;
    add.u32 %r57, %r50, %r18;
    add.u32 %r58, %r57, 1;
    mad.lo.u32 %r60, %r13, %r59, %r18;
    mul.wide.u32 %rd40, %r60, 4;
    add.s64 %rd42, %rd41, %rd40;
    setp.lt.u32 %p2, %r55, %r5;
    setp.lt.u32 %p3, %r57, %r5;
    setp.le.u32 %p4, %r57, %r55;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f9, %f76, %p5;
    st.shared.f32 [%rd42], %f20;
    add.s64 %rd43, %rd42, 4;
    setp.lt.u32 %p3, %r58, %r5;
    setp.le.u32 %p4, %r58, %r55;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f10, %f76, %p5;
    st.shared.f32 [%rd43], %f20;

    add.u32 %r60, %r13, 8;
    mad.lo.u32 %r60, %r60, %r59, %r18;
    mul.wide.u32 %rd40, %r60, 4;
    add.s64 %rd42, %rd41, %rd40;
    setp.lt.u32 %p2, %r56, %r5;
    setp.lt.u32 %p3, %r57, %r5;
    setp.le.u32 %p4, %r57, %r56;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f11, %f76, %p5;
    st.shared.f32 [%rd42], %f20;
    add.s64 %rd43, %rd42, 4;
    setp.lt.u32 %p3, %r58, %r5;
    setp.le.u32 %p4, %r58, %r56;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f12, %f76, %p5;
    st.shared.f32 [%rd43], %f20;

    add.u32 %r57, %r50, %r18;
    add.u32 %r57, %r57, 8;
    add.u32 %r58, %r57, 1;
    mad.lo.u32 %r60, %r13, %r59, %r18;
    add.u32 %r60, %r60, 8;
    mul.wide.u32 %rd40, %r60, 4;
    add.s64 %rd42, %rd41, %rd40;
    setp.lt.u32 %p2, %r55, %r5;
    setp.lt.u32 %p3, %r57, %r5;
    setp.le.u32 %p4, %r57, %r55;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f13, %f76, %p5;
    st.shared.f32 [%rd42], %f20;
    add.s64 %rd43, %rd42, 4;
    setp.lt.u32 %p3, %r58, %r5;
    setp.le.u32 %p4, %r58, %r55;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f14, %f76, %p5;
    st.shared.f32 [%rd43], %f20;

    add.u32 %r60, %r13, 8;
    mad.lo.u32 %r60, %r60, %r59, %r18;
    add.u32 %r60, %r60, 8;
    mul.wide.u32 %rd40, %r60, 4;
    add.s64 %rd42, %rd41, %rd40;
    setp.lt.u32 %p2, %r56, %r5;
    setp.lt.u32 %p3, %r57, %r5;
    setp.le.u32 %p4, %r57, %r56;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f15, %f76, %p5;
    st.shared.f32 [%rd42], %f20;
    add.s64 %rd43, %rd42, 4;
    setp.lt.u32 %p3, %r58, %r5;
    setp.le.u32 %p4, %r58, %r56;
    and.pred %p5, %p2, %p3;
    and.pred %p5, %p5, %p4;
    selp.f32 %f20, %f16, %f76, %p5;
    st.shared.f32 [%rd43], %f20;
    bar.sync 0;

    setp.ge.u32 %p1, %r10, 16;
    @%p1 bra FLASH_BWD_SOFTMAX_TILE_DONE;
    add.u32 %r61, %r15, %r10;
    setp.ge.u32 %p2, %r61, %r5;
    mul.lo.u32 %r62, %r27, %r5;
    add.u32 %r62, %r62, %r61;
    mul.wide.u32 %rd44, %r62, 4;
    add.s64 %rd45, %rd5, %rd44;
    ld.global.f32 %f30, [%rd45];
    add.s64 %rd46, %rd6, %rd44;
    ld.global.f32 %f31, [%rd46];
    add.s64 %rd47, %rd8, %rd44;
    ld.global.f32 %f32, [%rd47];
    mov.u32 %r63, 0;
FLASH_BWD_SOFTMAX_COL_LOOP:
    setp.ge.u32 %p3, %r63, 16;
    @%p3 bra FLASH_BWD_SOFTMAX_TILE_DONE;
    add.u32 %r64, %r50, %r63;
    mad.lo.u32 %r65, %r10, %r59, %r63;
    mul.wide.u32 %rd48, %r65, 4;
    mov.u64 %rd49, heirloom_flash_bwd_scores;
    add.s64 %rd50, %rd49, %rd48;
    ld.shared.f32 %f33, [%rd50];
    mov.u64 %rd51, heirloom_flash_bwd_dp;
    add.s64 %rd52, %rd51, %rd48;
    ld.shared.f32 %f34, [%rd52];
    setp.lt.u32 %p4, %r64, %r5;
    setp.le.u32 %p5, %r64, %r61;
    and.pred %p6, %p4, %p5;
    not.pred %p7, %p2;
    and.pred %p6, %p6, %p7;
    sub.f32 %f35, %f33, %f30;
    mul.f32 %f35, %f35, %f74;
    ex2.approx.ftz.f32 %f36, %f35;
    div.rn.f32 %f36, %f36, %f31;
    sub.f32 %f37, %f34, %f32;
    mul.f32 %f38, %f36, %f37;
    mul.f32 %f38, %f38, %f73;
    selp.f32 %f36, %f36, %f76, %p6;
    selp.f32 %f38, %f38, %f76, %p6;

    mul.wide.u32 %rd53, %r65, 2;
    mov.u64 %rd54, heirloom_flash_bwd_p;
    add.s64 %rd55, %rd54, %rd53;
    mov.b32 %r66, %f36;
    shr.u32 %r66, %r66, 16;
    st.shared.u16 [%rd55], %r66;
    mov.u64 %rd56, heirloom_flash_bwd_ds;
    add.s64 %rd57, %rd56, %rd53;
    mov.b32 %r67, %f38;
    shr.u32 %r67, %r67, 16;
    st.shared.u16 [%rd57], %r67;

    mad.lo.u32 %r68, %r63, %r59, %r10;
    mul.wide.u32 %rd58, %r68, 2;
    mov.u64 %rd59, heirloom_flash_bwd_pt;
    add.s64 %rd60, %rd59, %rd58;
    st.shared.u16 [%rd60], %r66;
    mov.u64 %rd61, heirloom_flash_bwd_dst;
    add.s64 %rd62, %rd61, %rd58;
    st.shared.u16 [%rd62], %r67;

    add.u32 %r63, %r63, 1;
    bra FLASH_BWD_SOFTMAX_COL_LOOP;

FLASH_BWD_SOFTMAX_TILE_DONE:
    bar.sync 0;

    // dQ = dS * K
    mov.u64 %rd61, heirloom_flash_bwd_ds;
    add.u32 %r19, %r13, 0;
    add.u32 %r20, %r18, 0;
    mad.lo.u32 %r21, %r19, %r59, %r20;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r22, [%rd62];
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r19, %r59, %r23;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r24, [%rd62];
    shl.b32 %r25, %r24, 16;
    or.b32 %r30, %r22, %r25;
    add.u32 %r19, %r19, 8;
    mad.lo.u32 %r21, %r19, %r59, %r20;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r22, [%rd62];
    mad.lo.u32 %r21, %r19, %r59, %r23;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r24, [%rd62];
    shl.b32 %r25, %r24, 16;
    or.b32 %r31, %r22, %r25;
    add.u32 %r19, %r13, 0;
    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r19, %r59, %r20;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r22, [%rd62];
    mad.lo.u32 %r21, %r19, %r59, %r23;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r24, [%rd62];
    shl.b32 %r25, %r24, 16;
    or.b32 %r32, %r22, %r25;
    add.u32 %r19, %r19, 8;
    mad.lo.u32 %r21, %r19, %r59, %r20;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r22, [%rd62];
    mad.lo.u32 %r21, %r19, %r59, %r23;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r24, [%rd62];
    shl.b32 %r25, %r24, 16;
    or.b32 %r33, %r22, %r25;

    add.u32 %r26, %r16, %r13;
    add.u32 %r20, %r50, %r18;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r43, %r29, %r5;
    add.u32 %r43, %r43, %r20;
    mul.lo.u32 %r43, %r43, %r6;
    add.u32 %r43, %r43, %r40;
    add.u32 %r43, %r43, %r26;
    mul.wide.u32 %rd63, %r43, 2;
    add.s64 %rd64, %rd2, %rd63;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r20, %r5;
    setp.lt.u32 %p4, %r26, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd64];
    add.u32 %r44, %r43, %r6;
    mul.wide.u32 %rd63, %r44, 2;
    add.s64 %rd64, %rd2, %rd63;
    mov.u32 %r24, 0;
    setp.lt.u32 %p3, %r23, %r5;
    setp.lt.u32 %p4, %r26, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd64];
    shl.b32 %r25, %r24, 16;
    or.b32 %r34, %r22, %r25;
    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r43, %r29, %r5;
    add.u32 %r43, %r43, %r20;
    mul.lo.u32 %r43, %r43, %r6;
    add.u32 %r43, %r43, %r40;
    add.u32 %r43, %r43, %r26;
    mul.wide.u32 %rd63, %r43, 2;
    add.s64 %rd64, %rd2, %rd63;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r20, %r5;
    setp.lt.u32 %p4, %r26, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd64];
    add.u32 %r44, %r43, %r6;
    mul.wide.u32 %rd63, %r44, 2;
    add.s64 %rd64, %rd2, %rd63;
    mov.u32 %r24, 0;
    setp.lt.u32 %p3, %r23, %r5;
    setp.lt.u32 %p4, %r26, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd64];
    shl.b32 %r25, %r24, 16;
    or.b32 %r35, %r22, %r25;
    mov.f32 %f41, 0f00000000;
    mov.f32 %f42, 0f00000000;
    mov.f32 %f43, 0f00000000;
    mov.f32 %f44, 0f00000000;
    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f45, %f46, %f47, %f48},
        {%r30, %r31, %r32, %r33},
        {%r34, %r35},
        {%f41, %f42, %f43, %f44};

    add.u32 %r19, %r15, %r13;
    add.u32 %r20, %r16, %r18;
    mul.lo.u32 %r45, %r29, %r5;
    add.u32 %r45, %r45, %r19;
    mul.lo.u32 %r45, %r45, %r6;
    add.u32 %r45, %r45, %r40;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd73, %r46, 4;
    add.s64 %rd74, %rd9, %rd73;
    setp.lt.u32 %p2, %r19, %r5;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 atom.global.add.f32 %f80, [%rd74], %f45;
    add.u32 %r20, %r20, 1;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd73, %r46, 4;
    add.s64 %rd74, %rd9, %rd73;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 atom.global.add.f32 %f80, [%rd74], %f46;
    add.u32 %r19, %r19, 8;
    sub.u32 %r20, %r20, 1;
    mul.lo.u32 %r45, %r29, %r5;
    add.u32 %r45, %r45, %r19;
    mul.lo.u32 %r45, %r45, %r6;
    add.u32 %r45, %r45, %r40;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd73, %r46, 4;
    add.s64 %rd74, %rd9, %rd73;
    setp.lt.u32 %p2, %r19, %r5;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 atom.global.add.f32 %f80, [%rd74], %f47;
    add.u32 %r20, %r20, 1;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd73, %r46, 4;
    add.s64 %rd74, %rd9, %rd73;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 atom.global.add.f32 %f80, [%rd74], %f48;

    // dV = P^T * dO
    mov.u64 %rd61, heirloom_flash_bwd_pt;
    add.u32 %r19, %r13, 0;
    add.u32 %r20, %r18, 0;
    mad.lo.u32 %r21, %r19, %r59, %r20;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r22, [%rd62];
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r19, %r59, %r23;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r24, [%rd62];
    shl.b32 %r25, %r24, 16;
    or.b32 %r30, %r22, %r25;
    add.u32 %r19, %r19, 8;
    mad.lo.u32 %r21, %r19, %r59, %r20;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r22, [%rd62];
    mad.lo.u32 %r21, %r19, %r59, %r23;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r24, [%rd62];
    shl.b32 %r25, %r24, 16;
    or.b32 %r31, %r22, %r25;
    add.u32 %r19, %r13, 0;
    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r19, %r59, %r20;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r22, [%rd62];
    mad.lo.u32 %r21, %r19, %r59, %r23;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r24, [%rd62];
    shl.b32 %r25, %r24, 16;
    or.b32 %r32, %r22, %r25;
    add.u32 %r19, %r19, 8;
    mad.lo.u32 %r21, %r19, %r59, %r20;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r22, [%rd62];
    mad.lo.u32 %r21, %r19, %r59, %r23;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r24, [%rd62];
    shl.b32 %r25, %r24, 16;
    or.b32 %r33, %r22, %r25;

    add.u32 %r26, %r16, %r13;
    add.u32 %r20, %r15, %r18;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r43, %r29, %r5;
    add.u32 %r43, %r43, %r20;
    mul.lo.u32 %r43, %r43, %r6;
    add.u32 %r43, %r43, %r40;
    add.u32 %r43, %r43, %r26;
    mul.wide.u32 %rd63, %r43, 2;
    add.s64 %rd64, %rd7, %rd63;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r20, %r5;
    setp.lt.u32 %p4, %r26, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd64];
    add.u32 %r44, %r43, %r6;
    mul.wide.u32 %rd63, %r44, 2;
    add.s64 %rd64, %rd7, %rd63;
    mov.u32 %r24, 0;
    setp.lt.u32 %p3, %r23, %r5;
    setp.lt.u32 %p4, %r26, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd64];
    shl.b32 %r25, %r24, 16;
    or.b32 %r34, %r22, %r25;
    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r43, %r29, %r5;
    add.u32 %r43, %r43, %r20;
    mul.lo.u32 %r43, %r43, %r6;
    add.u32 %r43, %r43, %r40;
    add.u32 %r43, %r43, %r26;
    mul.wide.u32 %rd63, %r43, 2;
    add.s64 %rd64, %rd7, %rd63;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r20, %r5;
    setp.lt.u32 %p4, %r26, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd64];
    add.u32 %r44, %r43, %r6;
    mul.wide.u32 %rd63, %r44, 2;
    add.s64 %rd64, %rd7, %rd63;
    mov.u32 %r24, 0;
    setp.lt.u32 %p3, %r23, %r5;
    setp.lt.u32 %p4, %r26, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd64];
    shl.b32 %r25, %r24, 16;
    or.b32 %r35, %r22, %r25;
    mov.f32 %f41, 0f00000000;
    mov.f32 %f42, 0f00000000;
    mov.f32 %f43, 0f00000000;
    mov.f32 %f44, 0f00000000;
    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f45, %f46, %f47, %f48},
        {%r30, %r31, %r32, %r33},
        {%r34, %r35},
        {%f41, %f42, %f43, %f44};

    add.u32 %r19, %r50, %r13;
    add.u32 %r20, %r16, %r18;
    mul.lo.u32 %r45, %r29, %r5;
    add.u32 %r45, %r45, %r19;
    mul.lo.u32 %r45, %r45, %r6;
    add.u32 %r45, %r45, %r40;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd73, %r46, 4;
    add.s64 %rd74, %rd11, %rd73;
    setp.lt.u32 %p2, %r19, %r5;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 atom.global.add.f32 %f80, [%rd74], %f45;
    add.u32 %r20, %r20, 1;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd73, %r46, 4;
    add.s64 %rd74, %rd11, %rd73;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 atom.global.add.f32 %f80, [%rd74], %f46;
    add.u32 %r19, %r19, 8;
    sub.u32 %r20, %r20, 1;
    mul.lo.u32 %r45, %r29, %r5;
    add.u32 %r45, %r45, %r19;
    mul.lo.u32 %r45, %r45, %r6;
    add.u32 %r45, %r45, %r40;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd73, %r46, 4;
    add.s64 %rd74, %rd11, %rd73;
    setp.lt.u32 %p2, %r19, %r5;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 atom.global.add.f32 %f80, [%rd74], %f47;
    add.u32 %r20, %r20, 1;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd73, %r46, 4;
    add.s64 %rd74, %rd11, %rd73;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 atom.global.add.f32 %f80, [%rd74], %f48;

    // dK = dS^T * Q
    mov.u64 %rd61, heirloom_flash_bwd_dst;
    add.u32 %r19, %r13, 0;
    add.u32 %r20, %r18, 0;
    mad.lo.u32 %r21, %r19, %r59, %r20;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r22, [%rd62];
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r19, %r59, %r23;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r24, [%rd62];
    shl.b32 %r25, %r24, 16;
    or.b32 %r30, %r22, %r25;
    add.u32 %r19, %r19, 8;
    mad.lo.u32 %r21, %r19, %r59, %r20;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r22, [%rd62];
    mad.lo.u32 %r21, %r19, %r59, %r23;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r24, [%rd62];
    shl.b32 %r25, %r24, 16;
    or.b32 %r31, %r22, %r25;
    add.u32 %r19, %r13, 0;
    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mad.lo.u32 %r21, %r19, %r59, %r20;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r22, [%rd62];
    mad.lo.u32 %r21, %r19, %r59, %r23;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r24, [%rd62];
    shl.b32 %r25, %r24, 16;
    or.b32 %r32, %r22, %r25;
    add.u32 %r19, %r19, 8;
    mad.lo.u32 %r21, %r19, %r59, %r20;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r22, [%rd62];
    mad.lo.u32 %r21, %r19, %r59, %r23;
    mul.wide.u32 %rd60, %r21, 2;
    add.s64 %rd62, %rd61, %rd60;
    ld.shared.u16 %r24, [%rd62];
    shl.b32 %r25, %r24, 16;
    or.b32 %r33, %r22, %r25;

    add.u32 %r26, %r16, %r13;
    add.u32 %r20, %r15, %r18;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r43, %r29, %r5;
    add.u32 %r43, %r43, %r20;
    mul.lo.u32 %r43, %r43, %r6;
    add.u32 %r43, %r43, %r40;
    add.u32 %r43, %r43, %r26;
    mul.wide.u32 %rd63, %r43, 2;
    add.s64 %rd64, %rd1, %rd63;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r20, %r5;
    setp.lt.u32 %p4, %r26, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd64];
    add.u32 %r44, %r43, %r6;
    mul.wide.u32 %rd63, %r44, 2;
    add.s64 %rd64, %rd1, %rd63;
    mov.u32 %r24, 0;
    setp.lt.u32 %p3, %r23, %r5;
    setp.lt.u32 %p4, %r26, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd64];
    shl.b32 %r25, %r24, 16;
    or.b32 %r34, %r22, %r25;
    add.u32 %r20, %r20, 8;
    add.u32 %r23, %r20, 1;
    mul.lo.u32 %r43, %r29, %r5;
    add.u32 %r43, %r43, %r20;
    mul.lo.u32 %r43, %r43, %r6;
    add.u32 %r43, %r43, %r40;
    add.u32 %r43, %r43, %r26;
    mul.wide.u32 %rd63, %r43, 2;
    add.s64 %rd64, %rd1, %rd63;
    mov.u32 %r22, 0;
    setp.lt.u32 %p3, %r20, %r5;
    setp.lt.u32 %p4, %r26, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r22, [%rd64];
    add.u32 %r44, %r43, %r6;
    mul.wide.u32 %rd63, %r44, 2;
    add.s64 %rd64, %rd1, %rd63;
    mov.u32 %r24, 0;
    setp.lt.u32 %p3, %r23, %r5;
    setp.lt.u32 %p4, %r26, %r8;
    and.pred %p5, %p3, %p4;
    @%p5 ld.global.u16 %r24, [%rd64];
    shl.b32 %r25, %r24, 16;
    or.b32 %r35, %r22, %r25;
    mov.f32 %f41, 0f00000000;
    mov.f32 %f42, 0f00000000;
    mov.f32 %f43, 0f00000000;
    mov.f32 %f44, 0f00000000;
    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f45, %f46, %f47, %f48},
        {%r30, %r31, %r32, %r33},
        {%r34, %r35},
        {%f41, %f42, %f43, %f44};

    add.u32 %r19, %r50, %r13;
    add.u32 %r20, %r16, %r18;
    mul.lo.u32 %r45, %r29, %r5;
    add.u32 %r45, %r45, %r19;
    mul.lo.u32 %r45, %r45, %r6;
    add.u32 %r45, %r45, %r40;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd73, %r46, 4;
    add.s64 %rd74, %rd10, %rd73;
    setp.lt.u32 %p2, %r19, %r5;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 atom.global.add.f32 %f80, [%rd74], %f45;
    add.u32 %r20, %r20, 1;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd73, %r46, 4;
    add.s64 %rd74, %rd10, %rd73;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 atom.global.add.f32 %f80, [%rd74], %f46;
    add.u32 %r19, %r19, 8;
    sub.u32 %r20, %r20, 1;
    mul.lo.u32 %r45, %r29, %r5;
    add.u32 %r45, %r45, %r19;
    mul.lo.u32 %r45, %r45, %r6;
    add.u32 %r45, %r45, %r40;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd73, %r46, 4;
    add.s64 %rd74, %rd10, %rd73;
    setp.lt.u32 %p2, %r19, %r5;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 atom.global.add.f32 %f80, [%rd74], %f47;
    add.u32 %r20, %r20, 1;
    add.u32 %r46, %r45, %r20;
    mul.wide.u32 %rd73, %r46, 4;
    add.s64 %rd74, %rd10, %rd73;
    setp.lt.u32 %p3, %r20, %r8;
    and.pred %p4, %p2, %p3;
    @%p4 atom.global.add.f32 %f80, [%rd74], %f48;

DONE_FLASH_ATTN_BF16_BWD_TILED_MMA:
    ret;
}

.visible .entry heirloom_flash_attention_bf16_tc_bwd_scalar_recompute_f32(
    .param .u64 query,
    .param .u64 key,
    .param .u64 value,
    .param .u64 output,
    .param .u64 row_max,
    .param .u64 row_denom,
    .param .u64 grad_output,
    .param .u64 row_dot,
    .param .u64 grad_query,
    .param .u64 grad_key,
    .param .u64 grad_value,
    .param .u32 batch,
    .param .u32 time,
    .param .u32 channels,
    .param .u32 n_heads,
    .param .u32 head_dim,
    .param .u32 total
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<120>;
    .reg .b64 %rd<80>;
    .reg .f32 %f<64>;

    ld.param.u64 %rd1, [query];
    ld.param.u64 %rd2, [key];
    ld.param.u64 %rd3, [value];
    ld.param.u64 %rd4, [output];
    ld.param.u64 %rd5, [row_max];
    ld.param.u64 %rd6, [row_denom];
    ld.param.u64 %rd7, [grad_output];
    ld.param.u64 %rd8, [row_dot];
    ld.param.u64 %rd9, [grad_query];
    ld.param.u64 %rd10, [grad_key];
    ld.param.u64 %rd11, [grad_value];
    ld.param.u32 %r4, [batch];
    ld.param.u32 %r5, [time];
    ld.param.u32 %r6, [channels];
    ld.param.u32 %r7, [n_heads];
    ld.param.u32 %r8, [head_dim];
    ld.param.u32 %r9, [total];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r10, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r10, %r9;
    @%p1 bra DONE_FLASH_ATTN_BF16_BWD_SCALAR;

    rem.u32 %r11, %r10, %r6;
    div.u32 %r12, %r10, %r6;
    rem.u32 %r13, %r12, %r5;
    div.u32 %r14, %r12, %r5;
    div.u32 %r15, %r11, %r8;
    rem.u32 %r16, %r11, %r8;
    mul.lo.u32 %r17, %r15, %r8;
    mad.lo.u32 %r18, %r14, %r7, %r15;

    cvt.rn.f32.u32 %f1, %r8;
    sqrt.rn.f32 %f2, %f1;
    mov.f32 %f3, 0f3f800000;
    div.rn.f32 %f4, %f3, %f2;
    mov.f32 %f5, 0f3fb8aa3b;

    mad.lo.u32 %r19, %r18, %r5, %r13;
    mul.wide.u32 %rd12, %r19, 4;
    add.s64 %rd13, %rd5, %rd12;
    ld.global.f32 %f6, [%rd13];
    add.s64 %rd14, %rd6, %rd12;
    ld.global.f32 %f7, [%rd14];
    add.s64 %rd15, %rd8, %rd12;
    ld.global.f32 %f8, [%rd15];

    mov.f32 %f30, 0f00000000;
    mov.u32 %r20, 0;

FLASH_ATTN_BWD_GQ_TOKEN_LOOP:
    setp.gt.u32 %p2, %r20, %r13;
    @%p2 bra FLASH_ATTN_BWD_GQ_DONE;
    mov.f32 %f9, 0f00000000;
    mov.f32 %f10, 0f00000000;
    mov.u32 %r21, 0;

FLASH_ATTN_BWD_GQ_DOT_LOOP:
    setp.ge.u32 %p3, %r21, %r8;
    @%p3 bra FLASH_ATTN_BWD_GQ_DOT_DONE;
    mul.lo.u32 %r22, %r14, %r5;
    add.u32 %r22, %r22, %r13;
    mul.lo.u32 %r22, %r22, %r6;
    add.u32 %r23, %r17, %r21;
    add.u32 %r24, %r22, %r23;
    mul.wide.u32 %rd16, %r24, 2;
    add.s64 %rd17, %rd1, %rd16;
    ld.global.u16 %r25, [%rd17];
    shl.b32 %r26, %r25, 16;
    mov.b32 %f11, %r26;
    mul.lo.u32 %r27, %r14, %r5;
    add.u32 %r27, %r27, %r20;
    mul.lo.u32 %r27, %r27, %r6;
    add.u32 %r28, %r27, %r23;
    mul.wide.u32 %rd18, %r28, 2;
    add.s64 %rd19, %rd2, %rd18;
    ld.global.u16 %r29, [%rd19];
    shl.b32 %r30, %r29, 16;
    mov.b32 %f12, %r30;
    fma.rn.f32 %f9, %f11, %f12, %f9;
    mul.wide.u32 %rd20, %r24, 2;
    add.s64 %rd21, %rd7, %rd20;
    ld.global.u16 %r31, [%rd21];
    shl.b32 %r32, %r31, 16;
    mov.b32 %f13, %r32;
    mul.wide.u32 %rd22, %r28, 2;
    add.s64 %rd23, %rd3, %rd22;
    ld.global.u16 %r33, [%rd23];
    shl.b32 %r34, %r33, 16;
    mov.b32 %f14, %r34;
    fma.rn.f32 %f10, %f13, %f14, %f10;
    add.u32 %r21, %r21, 1;
    bra FLASH_ATTN_BWD_GQ_DOT_LOOP;

FLASH_ATTN_BWD_GQ_DOT_DONE:
    mul.f32 %f15, %f9, %f4;
    sub.f32 %f15, %f15, %f6;
    mul.f32 %f15, %f15, %f5;
    ex2.approx.ftz.f32 %f16, %f15;
    div.rn.f32 %f16, %f16, %f7;
    sub.f32 %f17, %f10, %f8;
    mul.f32 %f17, %f17, %f16;
    mul.f32 %f17, %f17, %f4;
    mul.lo.u32 %r35, %r14, %r5;
    add.u32 %r35, %r35, %r20;
    mul.lo.u32 %r35, %r35, %r6;
    add.u32 %r36, %r17, %r16;
    add.u32 %r35, %r35, %r36;
    mul.wide.u32 %rd24, %r35, 2;
    add.s64 %rd25, %rd2, %rd24;
    ld.global.u16 %r37, [%rd25];
    shl.b32 %r38, %r37, 16;
    mov.b32 %f18, %r38;
    fma.rn.f32 %f30, %f17, %f18, %f30;
    add.u32 %r20, %r20, 1;
    bra FLASH_ATTN_BWD_GQ_TOKEN_LOOP;

FLASH_ATTN_BWD_GQ_DONE:
    mov.f32 %f31, 0f00000000;
    mov.f32 %f32, 0f00000000;
    mov.u32 %r40, %r13;

FLASH_ATTN_BWD_GKGV_TOKEN_LOOP:
    setp.ge.u32 %p2, %r40, %r5;
    @%p2 bra FLASH_ATTN_BWD_GKGV_DONE;
    mad.lo.u32 %r41, %r18, %r5, %r40;
    mul.wide.u32 %rd26, %r41, 4;
    add.s64 %rd27, %rd5, %rd26;
    ld.global.f32 %f19, [%rd27];
    add.s64 %rd28, %rd6, %rd26;
    ld.global.f32 %f20, [%rd28];
    add.s64 %rd29, %rd8, %rd26;
    ld.global.f32 %f21, [%rd29];
    mov.f32 %f22, 0f00000000;
    mov.f32 %f23, 0f00000000;
    mov.u32 %r42, 0;

FLASH_ATTN_BWD_GKGV_DOT_LOOP:
    setp.ge.u32 %p3, %r42, %r8;
    @%p3 bra FLASH_ATTN_BWD_GKGV_DOT_DONE;
    mul.lo.u32 %r43, %r14, %r5;
    add.u32 %r43, %r43, %r40;
    mul.lo.u32 %r43, %r43, %r6;
    add.u32 %r44, %r17, %r42;
    add.u32 %r45, %r43, %r44;
    mul.wide.u32 %rd30, %r45, 2;
    add.s64 %rd31, %rd1, %rd30;
    ld.global.u16 %r46, [%rd31];
    shl.b32 %r47, %r46, 16;
    mov.b32 %f24, %r47;
    mul.lo.u32 %r48, %r14, %r5;
    add.u32 %r48, %r48, %r13;
    mul.lo.u32 %r48, %r48, %r6;
    add.u32 %r49, %r48, %r44;
    mul.wide.u32 %rd32, %r49, 2;
    add.s64 %rd33, %rd2, %rd32;
    ld.global.u16 %r50, [%rd33];
    shl.b32 %r51, %r50, 16;
    mov.b32 %f25, %r51;
    fma.rn.f32 %f22, %f24, %f25, %f22;
    mul.wide.u32 %rd34, %r45, 2;
    add.s64 %rd35, %rd7, %rd34;
    ld.global.u16 %r52, [%rd35];
    shl.b32 %r53, %r52, 16;
    mov.b32 %f26, %r53;
    mul.wide.u32 %rd36, %r49, 2;
    add.s64 %rd37, %rd3, %rd36;
    ld.global.u16 %r54, [%rd37];
    shl.b32 %r55, %r54, 16;
    mov.b32 %f27, %r55;
    fma.rn.f32 %f23, %f26, %f27, %f23;
    add.u32 %r42, %r42, 1;
    bra FLASH_ATTN_BWD_GKGV_DOT_LOOP;

FLASH_ATTN_BWD_GKGV_DOT_DONE:
    mul.f32 %f28, %f22, %f4;
    sub.f32 %f28, %f28, %f19;
    mul.f32 %f28, %f28, %f5;
    ex2.approx.ftz.f32 %f29, %f28;
    div.rn.f32 %f29, %f29, %f20;
    mul.lo.u32 %r56, %r14, %r5;
    add.u32 %r56, %r56, %r40;
    mul.lo.u32 %r56, %r56, %r6;
    add.u32 %r57, %r17, %r16;
    add.u32 %r58, %r56, %r57;
    mul.wide.u32 %rd38, %r58, 2;
    add.s64 %rd39, %rd7, %rd38;
    ld.global.u16 %r59, [%rd39];
    shl.b32 %r60, %r59, 16;
    mov.b32 %f33, %r60;
    fma.rn.f32 %f32, %f29, %f33, %f32;
    sub.f32 %f34, %f23, %f21;
    mul.f32 %f34, %f34, %f29;
    mul.f32 %f34, %f34, %f4;
    mul.wide.u32 %rd40, %r58, 2;
    add.s64 %rd41, %rd1, %rd40;
    ld.global.u16 %r61, [%rd41];
    shl.b32 %r62, %r61, 16;
    mov.b32 %f35, %r62;
    fma.rn.f32 %f31, %f34, %f35, %f31;
    add.u32 %r40, %r40, 1;
    bra FLASH_ATTN_BWD_GKGV_TOKEN_LOOP;

FLASH_ATTN_BWD_GKGV_DONE:
    mul.wide.u32 %rd42, %r10, 4;
    add.s64 %rd43, %rd9, %rd42;
    st.global.f32 [%rd43], %f30;
    add.s64 %rd44, %rd10, %rd42;
    st.global.f32 [%rd44], %f31;
    add.s64 %rd45, %rd11, %rd42;
    st.global.f32 [%rd45], %f32;

DONE_FLASH_ATTN_BF16_BWD_SCALAR:
    ret;
}
"#;

const KERNEL_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry heirloom_add_f32(
    .param .u64 left,
    .param .u64 right,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [left];
    ld.param.u64 %rd2, [right];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_ADD;
    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;
DONE_ADD:
    ret;
}

.visible .entry heirloom_sub_f32(
    .param .u64 left,
    .param .u64 right,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [left];
    ld.param.u64 %rd2, [right];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_SUB;
    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    sub.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;
DONE_SUB:
    ret;
}

.visible .entry heirloom_mul_f32(
    .param .u64 left,
    .param .u64 right,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [left];
    ld.param.u64 %rd2, [right];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_MUL;
    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    mul.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;
DONE_MUL:
    ret;
}

.visible .entry heirloom_div_f32(
    .param .u64 left,
    .param .u64 right,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [left];
    ld.param.u64 %rd2, [right];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_DIV;
    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    div.rn.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;
DONE_DIV:
    ret;
}

.visible .entry heirloom_scale_f32(
    .param .u64 input,
    .param .u64 output,
    .param .f32 scale,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.f32 %f2, [scale];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_SCALE;
    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];
    mul.f32 %f3, %f1, %f2;
    st.global.f32 [%rd5], %f3;
DONE_SCALE:
    ret;
}

.visible .entry heirloom_f32_to_bf16(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_F32_TO_BF16;
    mul.wide.u32 %rd3, %r5, 4;
    mul.wide.u32 %rd4, %r5, 2;
    add.s64 %rd5, %rd1, %rd3;
    add.s64 %rd6, %rd2, %rd4;
    ld.global.u32 %r6, [%rd5];
    shr.u32 %r7, %r6, 16;
    and.b32 %r8, %r7, 1;
    add.u32 %r9, %r8, 32767;
    add.u32 %r10, %r6, %r9;
    shr.u32 %r11, %r10, 16;
    st.global.u16 [%rd6], %r11;
DONE_F32_TO_BF16:
    ret;
}

.visible .entry heirloom_bf16_to_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_BF16_TO_F32;
    mul.wide.u32 %rd3, %r5, 2;
    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd3;
    add.s64 %rd6, %rd2, %rd4;
    ld.global.u16 %r6, [%rd5];
    shl.b32 %r7, %r6, 16;
    st.global.u32 [%rd6], %r7;
DONE_BF16_TO_F32:
    ret;
}

.visible .entry heirloom_f32_bf16_roundtrip_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<14>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_F32_BF16_ROUNDTRIP_F32;
    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.u32 %r6, [%rd4];
    shr.u32 %r7, %r6, 16;
    and.b32 %r8, %r7, 1;
    add.u32 %r9, %r8, 32767;
    add.u32 %r10, %r6, %r9;
    shr.u32 %r11, %r10, 16;
    shl.b32 %r12, %r11, 16;
    st.global.u32 [%rd5], %r12;
DONE_F32_BF16_ROUNDTRIP_F32:
    ret;
}

.visible .entry heirloom_bias_add_f32(
    .param .u64 matrix,
    .param .u64 bias,
    .param .u64 output,
    .param .u32 rows,
    .param .u32 cols,
    .param .u32 total
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [matrix];
    ld.param.u64 %rd2, [bias];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [rows];
    ld.param.u32 %r5, [cols];
    ld.param.u32 %r6, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r7, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r7, %r6;
    @%p1 bra DONE_BIAS_ADD;
    rem.u32 %r4, %r7, %r5;
    mul.wide.u32 %rd4, %r7, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd3, %rd4;
    mul.wide.u32 %rd7, %r4, 4;
    add.s64 %rd8, %rd2, %rd7;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd8];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd6], %f3;
DONE_BIAS_ADD:
    ret;
}

.visible .entry heirloom_bias_add_backward_bias_f32(
    .param .u64 grad_output,
    .param .u64 output,
    .param .u32 rows,
    .param .u32 cols,
    .param .u32 total
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<9>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<3>;

    ld.param.u64 %rd1, [grad_output];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r4, [rows];
    ld.param.u32 %r5, [cols];
    ld.param.u32 %r6, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r7, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r7, %r6;
    @%p1 bra DONE_BIAS_BACKWARD;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r8, 0;
BIAS_BACKWARD_LOOP:
    setp.ge.u32 %p2, %r8, %r4;
    @%p2 bra BIAS_BACKWARD_WRITE;
    mad.lo.s32 %r6, %r8, %r5, %r7;
    mul.wide.u32 %rd3, %r6, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f2, [%rd4];
    add.f32 %f1, %f1, %f2;
    add.u32 %r8, %r8, 1;
    bra BIAS_BACKWARD_LOOP;
BIAS_BACKWARD_WRITE:
    mul.wide.u32 %rd5, %r7, 4;
    add.s64 %rd6, %rd2, %rd5;
    st.global.f32 [%rd6], %f1;
DONE_BIAS_BACKWARD:
    ret;
}

.visible .entry heirloom_sgd_update_f32(
    .param .u64 param,
    .param .u64 grad,
    .param .f32 lr,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<6>;

    ld.param.u64 %rd1, [param];
    ld.param.u64 %rd2, [grad];
    ld.param.f32 %f3, [lr];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_SGD_UPDATE;
    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];
    ld.global.f32 %f2, [%rd5];
    mul.f32 %f4, %f3, %f2;
    sub.f32 %f5, %f1, %f4;
    st.global.f32 [%rd4], %f5;
DONE_SGD_UPDATE:
    ret;
}

.visible .entry heirloom_adamw_update_f32(
    .param .u64 param,
    .param .u64 grad,
    .param .u64 m,
    .param .u64 v,
    .param .f32 lr,
    .param .f32 beta1,
    .param .f32 beta2,
    .param .f32 eps,
    .param .f32 weight_decay,
    .param .f32 clip_scale,
    .param .f32 bias_correction1,
    .param .f32 bias_correction2,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<14>;
    .reg .f32 %f<33>;

    ld.param.u64 %rd1, [param];
    ld.param.u64 %rd2, [grad];
    ld.param.u64 %rd3, [m];
    ld.param.u64 %rd4, [v];
    ld.param.f32 %f1, [lr];
    ld.param.f32 %f2, [beta1];
    ld.param.f32 %f3, [beta2];
    ld.param.f32 %f4, [eps];
    ld.param.f32 %f5, [weight_decay];
    ld.param.f32 %f6, [clip_scale];
    ld.param.f32 %f7, [bias_correction1];
    ld.param.f32 %f8, [bias_correction2];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_ADAMW_UPDATE;

    mul.wide.u32 %rd5, %r5, 4;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd7, %rd2, %rd5;
    add.s64 %rd8, %rd3, %rd5;
    add.s64 %rd9, %rd4, %rd5;
    ld.global.f32 %f9, [%rd6];
    ld.global.f32 %f10, [%rd7];
    ld.global.f32 %f11, [%rd8];
    ld.global.f32 %f12, [%rd9];

    mul.f32 %f13, %f10, %f6;
    mul.f32 %f14, %f9, %f5;
    add.f32 %f15, %f13, %f14;
    mov.f32 %f16, 0f3F800000;
    sub.f32 %f17, %f16, %f2;
    mul.f32 %f18, %f2, %f11;
    mul.f32 %f19, %f17, %f15;
    add.f32 %f20, %f18, %f19;
    sub.f32 %f21, %f16, %f3;
    mul.f32 %f22, %f3, %f12;
    mul.f32 %f23, %f15, %f15;
    mul.f32 %f24, %f21, %f23;
    add.f32 %f25, %f22, %f24;

    st.global.f32 [%rd8], %f20;
    st.global.f32 [%rd9], %f25;

    div.rn.f32 %f26, %f20, %f7;
    div.rn.f32 %f27, %f25, %f8;
    sqrt.rn.f32 %f28, %f27;
    add.f32 %f29, %f28, %f4;
    div.rn.f32 %f30, %f26, %f29;
    mul.f32 %f31, %f1, %f30;
    sub.f32 %f32, %f9, %f31;
    st.global.f32 [%rd6], %f32;
DONE_ADAMW_UPDATE:
    ret;
}

.visible .entry heirloom_memory_gather_selected_rows_f32_i64(
    .param .u64 table,
    .param .u64 selected_rows,
    .param .u64 output,
    .param .u64 status,
    .param .u32 selected_count,
    .param .u32 rows,
    .param .u32 row_dim,
    .param .u32 total
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<14>;
    .reg .b64 %rd<18>;
    .reg .f32 %f<2>;

    ld.param.u64 %rd1, [table];
    ld.param.u64 %rd2, [selected_rows];
    ld.param.u64 %rd3, [output];
    ld.param.u64 %rd4, [status];
    ld.param.u32 %r4, [selected_count];
    ld.param.u32 %r5, [rows];
    ld.param.u32 %r6, [row_dim];
    ld.param.u32 %r7, [total];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r8, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r8, %r7;
    @%p1 bra DONE_GATHER_SELECTED_ROWS;

    div.u32 %r9, %r8, %r6;
    rem.u32 %r10, %r8, %r6;
    setp.ge.u32 %p2, %r9, %r4;
    @%p2 bra GATHER_SELECTED_ROWS_BAD_INDEX;

    mul.wide.u32 %rd5, %r9, 8;
    add.s64 %rd6, %rd2, %rd5;
    ld.global.s64 %rd7, [%rd6];
    mov.s64 %rd8, 0;
    setp.lt.s64 %p3, %rd7, %rd8;
    cvt.s64.u32 %rd9, %r5;
    setp.ge.s64 %p4, %rd7, %rd9;
    or.pred %p3, %p3, %p4;
    @%p3 bra GATHER_SELECTED_ROWS_BAD_INDEX;

    cvt.u32.u64 %r11, %rd7;
    mul.lo.u32 %r12, %r11, %r6;
    add.u32 %r13, %r12, %r10;
    mul.wide.u32 %rd10, %r13, 4;
    add.s64 %rd11, %rd1, %rd10;
    ld.global.f32 %f1, [%rd11];

    mul.wide.u32 %rd12, %r8, 4;
    add.s64 %rd13, %rd3, %rd12;
    st.global.f32 [%rd13], %f1;
    bra DONE_GATHER_SELECTED_ROWS;

GATHER_SELECTED_ROWS_BAD_INDEX:
    mov.u32 %r11, 1;
    st.global.u32 [%rd4], %r11;

DONE_GATHER_SELECTED_ROWS:
    ret;
}

.visible .entry heirloom_memory_sparse_adamw_rows_f32_i64(
    .param .u64 param,
    .param .u64 grad,
    .param .u64 m,
    .param .u64 v,
    .param .u64 selected_rows,
    .param .u64 status,
    .param .u64 row_mask,
    .param .f32 lr,
    .param .f32 beta1,
    .param .f32 beta2,
    .param .f32 eps,
    .param .f32 weight_decay,
    .param .f32 clip_scale,
    .param .f32 bias_correction1,
    .param .f32 bias_correction2,
    .param .u32 selected_count,
    .param .u32 rows,
    .param .u32 row_dim,
    .param .u32 total,
    .param .u32 has_row_mask
)
{
    .reg .pred %p<10>;
    .reg .b32 %r<28>;
    .reg .b64 %rd<30>;
    .reg .f32 %f<33>;

    ld.param.u64 %rd1, [param];
    ld.param.u64 %rd2, [grad];
    ld.param.u64 %rd3, [m];
    ld.param.u64 %rd4, [v];
    ld.param.u64 %rd5, [selected_rows];
    ld.param.u64 %rd6, [status];
    ld.param.u64 %rd27, [row_mask];
    ld.param.f32 %f1, [lr];
    ld.param.f32 %f2, [beta1];
    ld.param.f32 %f3, [beta2];
    ld.param.f32 %f4, [eps];
    ld.param.f32 %f5, [weight_decay];
    ld.param.f32 %f6, [clip_scale];
    ld.param.f32 %f7, [bias_correction1];
    ld.param.f32 %f8, [bias_correction2];
    ld.param.u32 %r4, [selected_count];
    ld.param.u32 %r5, [rows];
    ld.param.u32 %r6, [row_dim];
    ld.param.u32 %r7, [total];
    ld.param.u32 %r23, [has_row_mask];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r8, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r8, %r7;
    @%p1 bra DONE_MEMORY_SPARSE_ADAMW_ROWS;

    div.u32 %r9, %r8, %r6;
    rem.u32 %r10, %r8, %r6;
    mul.wide.u32 %rd7, %r9, 8;
    add.s64 %rd8, %rd5, %rd7;
    ld.global.s64 %rd9, [%rd8];
    mov.s64 %rd10, 0;
    setp.lt.s64 %p2, %rd9, %rd10;
    cvt.s64.u32 %rd11, %r5;
    setp.ge.s64 %p3, %rd9, %rd11;
    or.pred %p4, %p2, %p3;
    @%p4 bra MEMORY_SPARSE_ADAMW_BAD_INDEX;

    setp.eq.u32 %p7, %r23, 0;
    @%p7 bra MEMORY_SPARSE_ADAMW_DEDUP_START;
    cvt.u32.u64 %r24, %rd9;
    cvt.u64.u32 %rd28, %r24;
    add.s64 %rd29, %rd27, %rd28;
    ld.global.u8 %r25, [%rd29];
    setp.eq.u32 %p8, %r25, 0;
    @%p8 bra DONE_MEMORY_SPARSE_ADAMW_ROWS;

MEMORY_SPARSE_ADAMW_DEDUP_START:
    mov.u32 %r11, 0;
MEMORY_SPARSE_ADAMW_DEDUP_LOOP:
    setp.ge.u32 %p5, %r11, %r9;
    @%p5 bra MEMORY_SPARSE_ADAMW_APPLY;
    mul.wide.u32 %rd12, %r11, 8;
    add.s64 %rd13, %rd5, %rd12;
    ld.global.s64 %rd14, [%rd13];
    setp.eq.s64 %p6, %rd14, %rd9;
    @%p6 bra DONE_MEMORY_SPARSE_ADAMW_ROWS;
    add.u32 %r11, %r11, 1;
    bra MEMORY_SPARSE_ADAMW_DEDUP_LOOP;

MEMORY_SPARSE_ADAMW_APPLY:
    cvt.u32.u64 %r12, %rd9;
    mad.lo.s32 %r13, %r12, %r6, %r10;
    mul.wide.u32 %rd15, %r13, 4;
    add.s64 %rd16, %rd1, %rd15;
    add.s64 %rd17, %rd2, %rd15;
    add.s64 %rd18, %rd3, %rd15;
    add.s64 %rd19, %rd4, %rd15;
    ld.global.f32 %f9, [%rd16];
    ld.global.f32 %f10, [%rd17];
    ld.global.f32 %f11, [%rd18];
    ld.global.f32 %f12, [%rd19];

    mul.f32 %f13, %f10, %f6;
    mul.f32 %f14, %f9, %f5;
    add.f32 %f15, %f13, %f14;
    mov.f32 %f16, 0f3F800000;
    sub.f32 %f17, %f16, %f2;
    mul.f32 %f18, %f2, %f11;
    mul.f32 %f19, %f17, %f15;
    add.f32 %f20, %f18, %f19;
    sub.f32 %f21, %f16, %f3;
    mul.f32 %f22, %f3, %f12;
    mul.f32 %f23, %f15, %f15;
    mul.f32 %f24, %f21, %f23;
    add.f32 %f25, %f22, %f24;

    st.global.f32 [%rd18], %f20;
    st.global.f32 [%rd19], %f25;

    div.rn.f32 %f26, %f20, %f7;
    div.rn.f32 %f27, %f25, %f8;
    sqrt.rn.f32 %f28, %f27;
    add.f32 %f29, %f28, %f4;
    div.rn.f32 %f30, %f26, %f29;
    mul.f32 %f31, %f1, %f30;
    sub.f32 %f32, %f9, %f31;
    st.global.f32 [%rd16], %f32;
    bra DONE_MEMORY_SPARSE_ADAMW_ROWS;

MEMORY_SPARSE_ADAMW_BAD_INDEX:
    mov.u32 %r14, 1;
    atom.global.exch.b32 %r15, [%rd6], %r14;
DONE_MEMORY_SPARSE_ADAMW_ROWS:
    ret;
}

.visible .entry heirloom_memory_sparse_adamw_compact_rows_f32_i64(
    .param .u64 param,
    .param .u64 compact_grad,
    .param .u64 m,
    .param .u64 v,
    .param .u64 selected_rows,
    .param .u64 status,
    .param .u64 row_mask,
    .param .f32 lr,
    .param .f32 beta1,
    .param .f32 beta2,
    .param .f32 eps,
    .param .f32 weight_decay,
    .param .f32 clip_scale,
    .param .f32 bias_correction1,
    .param .f32 bias_correction2,
    .param .u32 selected_count,
    .param .u32 rows,
    .param .u32 row_dim,
    .param .u32 total,
    .param .u32 has_row_mask
)
{
    .reg .pred %p<10>;
    .reg .b32 %r<29>;
    .reg .b64 %rd<31>;
    .reg .f32 %f<33>;

    ld.param.u64 %rd1, [param];
    ld.param.u64 %rd2, [compact_grad];
    ld.param.u64 %rd3, [m];
    ld.param.u64 %rd4, [v];
    ld.param.u64 %rd5, [selected_rows];
    ld.param.u64 %rd6, [status];
    ld.param.u64 %rd27, [row_mask];
    ld.param.f32 %f1, [lr];
    ld.param.f32 %f2, [beta1];
    ld.param.f32 %f3, [beta2];
    ld.param.f32 %f4, [eps];
    ld.param.f32 %f5, [weight_decay];
    ld.param.f32 %f6, [clip_scale];
    ld.param.f32 %f7, [bias_correction1];
    ld.param.f32 %f8, [bias_correction2];
    ld.param.u32 %r4, [selected_count];
    ld.param.u32 %r5, [rows];
    ld.param.u32 %r6, [row_dim];
    ld.param.u32 %r7, [total];
    ld.param.u32 %r23, [has_row_mask];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r8, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r8, %r7;
    @%p1 bra DONE_MEMORY_SPARSE_ADAMW_COMPACT_ROWS;

    div.u32 %r9, %r8, %r6;
    rem.u32 %r10, %r8, %r6;
    mul.wide.u32 %rd7, %r9, 8;
    add.s64 %rd8, %rd5, %rd7;
    ld.global.s64 %rd9, [%rd8];
    mov.s64 %rd10, 0;
    setp.lt.s64 %p2, %rd9, %rd10;
    cvt.s64.u32 %rd11, %r5;
    setp.ge.s64 %p3, %rd9, %rd11;
    or.pred %p4, %p2, %p3;
    @%p4 bra MEMORY_SPARSE_ADAMW_COMPACT_BAD_INDEX;

    setp.eq.u32 %p7, %r23, 0;
    @%p7 bra MEMORY_SPARSE_ADAMW_COMPACT_DEDUP_START;
    cvt.u32.u64 %r24, %rd9;
    cvt.u64.u32 %rd28, %r24;
    add.s64 %rd29, %rd27, %rd28;
    ld.global.u8 %r25, [%rd29];
    setp.eq.u32 %p8, %r25, 0;
    @%p8 bra DONE_MEMORY_SPARSE_ADAMW_COMPACT_ROWS;

MEMORY_SPARSE_ADAMW_COMPACT_DEDUP_START:
    mov.u32 %r11, 0;
MEMORY_SPARSE_ADAMW_COMPACT_DEDUP_LOOP:
    setp.ge.u32 %p5, %r11, %r9;
    @%p5 bra MEMORY_SPARSE_ADAMW_COMPACT_APPLY;
    mul.wide.u32 %rd12, %r11, 8;
    add.s64 %rd13, %rd5, %rd12;
    ld.global.s64 %rd14, [%rd13];
    setp.eq.s64 %p6, %rd14, %rd9;
    @%p6 bra DONE_MEMORY_SPARSE_ADAMW_COMPACT_ROWS;
    add.u32 %r11, %r11, 1;
    bra MEMORY_SPARSE_ADAMW_COMPACT_DEDUP_LOOP;

MEMORY_SPARSE_ADAMW_COMPACT_APPLY:
    cvt.u32.u64 %r12, %rd9;
    mad.lo.s32 %r13, %r12, %r6, %r10;
    mul.wide.u32 %rd15, %r13, 4;
    add.s64 %rd16, %rd1, %rd15;
    add.s64 %rd18, %rd3, %rd15;
    add.s64 %rd19, %rd4, %rd15;

    mad.lo.s32 %r26, %r9, %r6, %r10;
    mul.wide.u32 %rd17, %r26, 4;
    add.s64 %rd20, %rd2, %rd17;

    ld.global.f32 %f9, [%rd16];
    ld.global.f32 %f10, [%rd20];
    ld.global.f32 %f11, [%rd18];
    ld.global.f32 %f12, [%rd19];

    mul.f32 %f13, %f10, %f6;
    mul.f32 %f14, %f9, %f5;
    add.f32 %f15, %f13, %f14;
    mov.f32 %f16, 0f3F800000;
    sub.f32 %f17, %f16, %f2;
    mul.f32 %f18, %f2, %f11;
    mul.f32 %f19, %f17, %f15;
    add.f32 %f20, %f18, %f19;
    sub.f32 %f21, %f16, %f3;
    mul.f32 %f22, %f3, %f12;
    mul.f32 %f23, %f15, %f15;
    mul.f32 %f24, %f21, %f23;
    add.f32 %f25, %f22, %f24;

    st.global.f32 [%rd18], %f20;
    st.global.f32 [%rd19], %f25;

    div.rn.f32 %f26, %f20, %f7;
    div.rn.f32 %f27, %f25, %f8;
    sqrt.rn.f32 %f28, %f27;
    add.f32 %f29, %f28, %f4;
    div.rn.f32 %f30, %f26, %f29;
    mul.f32 %f31, %f1, %f30;
    sub.f32 %f32, %f9, %f31;
    st.global.f32 [%rd16], %f32;
    bra DONE_MEMORY_SPARSE_ADAMW_COMPACT_ROWS;

MEMORY_SPARSE_ADAMW_COMPACT_BAD_INDEX:
    mov.u32 %r14, 1;
    atom.global.exch.b32 %r15, [%rd6], %r14;
DONE_MEMORY_SPARSE_ADAMW_COMPACT_ROWS:
    ret;
}

.visible .entry heirloom_memory_access_count_rows_i64_u64(
    .param .u64 selected_rows,
    .param .u64 output_counts,
    .param .u64 status,
    .param .u32 selected_count,
    .param .u32 memory_slots
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<14>;

    ld.param.u64 %rd1, [selected_rows];
    ld.param.u64 %rd2, [output_counts];
    ld.param.u64 %rd3, [status];
    ld.param.u32 %r4, [selected_count];
    ld.param.u32 %r5, [memory_slots];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r6, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r6, %r4;
    @%p1 bra DONE_MEMORY_ACCESS_COUNT;

    mul.wide.u32 %rd4, %r6, 8;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.s64 %rd6, [%rd5];
    mov.s64 %rd7, 0;
    setp.lt.s64 %p2, %rd6, %rd7;
    cvt.s64.u32 %rd8, %r5;
    setp.ge.s64 %p3, %rd6, %rd8;
    or.pred %p4, %p2, %p3;
    @%p4 bra MEMORY_ACCESS_COUNT_BAD_INDEX;

    mul.lo.u64 %rd9, %rd6, 8;
    add.s64 %rd10, %rd2, %rd9;
    mov.u64 %rd11, 1;
    atom.global.add.u64 %rd12, [%rd10], %rd11;
    bra DONE_MEMORY_ACCESS_COUNT;

MEMORY_ACCESS_COUNT_BAD_INDEX:
    mov.u32 %r7, 1;
    st.global.u32 [%rd3], %r7;

DONE_MEMORY_ACCESS_COUNT:
    ret;
}

.visible .entry heirloom_i64_arange(
    .param .u64 output,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [output];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_I64_ARANGE;

    mul.wide.u32 %rd2, %r5, 8;
    add.s64 %rd3, %rd1, %rd2;
    cvt.s64.u32 %rd4, %r5;
    st.global.s64 [%rd3], %rd4;

DONE_I64_ARANGE:
    ret;
}

.visible .entry heirloom_memory_selected_rows_to_f32_mask(
    .param .u64 selected_rows,
    .param .u64 output_mask,
    .param .u64 status,
    .param .u32 selected_count,
    .param .u32 rows
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<3>;

    ld.param.u64 %rd1, [selected_rows];
    ld.param.u64 %rd2, [output_mask];
    ld.param.u64 %rd3, [status];
    ld.param.u32 %r4, [selected_count];
    ld.param.u32 %r5, [rows];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r6, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r6, %r4;
    @%p1 bra DONE_SELECTED_ROWS_TO_MASK;

    mul.wide.u32 %rd4, %r6, 8;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.s64 %rd6, [%rd5];
    mov.s64 %rd7, 0;
    setp.lt.s64 %p2, %rd6, %rd7;
    cvt.s64.u32 %rd8, %r5;
    setp.ge.s64 %p3, %rd6, %rd8;
    or.pred %p4, %p2, %p3;
    @%p4 bra SELECTED_ROWS_TO_MASK_BAD_INDEX;

    cvt.u32.u64 %r7, %rd6;
    mul.wide.u32 %rd9, %r7, 4;
    add.s64 %rd10, %rd2, %rd9;
    mov.f32 %f1, 0f3F800000;
    atom.global.add.f32 %f2, [%rd10], %f1;
    bra DONE_SELECTED_ROWS_TO_MASK;

SELECTED_ROWS_TO_MASK_BAD_INDEX:
    mov.u32 %r7, 1;
    st.global.u32 [%rd3], %r7;

DONE_SELECTED_ROWS_TO_MASK:
    ret;
}

.visible .entry heirloom_f32_mask_to_bool(
    .param .u64 input_mask,
    .param .u64 output_mask,
    .param .u32 n
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<3>;

    ld.param.u64 %rd1, [input_mask];
    ld.param.u64 %rd2, [output_mask];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_F32_MASK_TO_BOOL;

    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    mov.f32 %f2, 0f00000000;
    setp.gt.f32 %p2, %f1, %f2;
    selp.u32 %r6, 1, 0, %p2;
    cvt.u64.u32 %rd5, %r5;
    add.s64 %rd6, %rd2, %rd5;
    st.global.u8 [%rd6], %r6;

DONE_F32_MASK_TO_BOOL:
    ret;
}

.visible .entry heirloom_bool_and_u8(
    .param .u64 left_mask,
    .param .u64 right_mask,
    .param .u64 output_mask,
    .param .u32 n
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [left_mask];
    ld.param.u64 %rd2, [right_mask];
    ld.param.u64 %rd3, [output_mask];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_BOOL_AND_U8;

    cvt.u64.u32 %rd4, %r5;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.u8 %r6, [%rd5];
    ld.global.u8 %r7, [%rd6];
    setp.ne.u32 %p2, %r6, 0;
    setp.ne.u32 %p3, %r7, 0;
    and.pred %p4, %p2, %p3;
    selp.u32 %r8, 1, 0, %p4;
    st.global.u8 [%rd7], %r8;

DONE_BOOL_AND_U8:
    ret;
}

.visible .entry heirloom_bool_mask_to_i64_indices(
    .param .u64 mask,
    .param .u64 output_indices,
    .param .u64 output_count,
    .param .u32 n
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<14>;
    .reg .b64 %rd<12>;

    ld.param.u64 %rd1, [mask];
    ld.param.u64 %rd2, [output_indices];
    ld.param.u64 %rd3, [output_count];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_BOOL_MASK_TO_INDICES;

    cvt.u64.u32 %rd4, %r5;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u8 %r6, [%rd5];
    setp.eq.u32 %p2, %r6, 0;
    @%p2 bra DONE_BOOL_MASK_TO_INDICES;

    mov.u32 %r7, 0;
    mov.u32 %r8, 0;
BOOL_MASK_TO_INDICES_PREFIX_LOOP:
    setp.ge.u32 %p3, %r8, %r5;
    @%p3 bra BOOL_MASK_TO_INDICES_WRITE;
    cvt.u64.u32 %rd6, %r8;
    add.s64 %rd7, %rd1, %rd6;
    ld.global.u8 %r9, [%rd7];
    setp.eq.u32 %p4, %r9, 0;
    @%p4 bra BOOL_MASK_TO_INDICES_PREFIX_NEXT;
    add.u32 %r7, %r7, 1;
BOOL_MASK_TO_INDICES_PREFIX_NEXT:
    add.u32 %r8, %r8, 1;
    bra BOOL_MASK_TO_INDICES_PREFIX_LOOP;

BOOL_MASK_TO_INDICES_WRITE:
    mul.wide.u32 %rd8, %r7, 8;
    add.s64 %rd9, %rd2, %rd8;
    cvt.s64.u32 %rd10, %r5;
    st.global.s64 [%rd9], %rd10;
    add.u32 %r10, %r7, 1;
    atom.global.max.u32 %r11, [%rd3], %r10;

DONE_BOOL_MASK_TO_INDICES:
    ret;
}

.visible .entry heirloom_copy_i64_with_offset(
    .param .u64 input,
    .param .u64 output,
    .param .u32 output_offset,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r4, [output_offset];
    ld.param.u32 %r5, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r6, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r6, %r5;
    @%p1 bra DONE_COPY_I64_WITH_OFFSET;

    mul.wide.u32 %rd3, %r6, 8;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.u64 %rd5, [%rd4];
    add.u32 %r7, %r6, %r4;
    mul.wide.u32 %rd6, %r7, 8;
    add.s64 %rd7, %rd2, %rd6;
    st.global.u64 [%rd7], %rd5;

DONE_COPY_I64_WITH_OFFSET:
    ret;
}

.visible .entry heirloom_fill_from_scalar_f32(
    .param .u64 scalar,
    .param .u64 output,
    .param .f32 scale,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [scalar];
    ld.param.u64 %rd2, [output];
    ld.param.f32 %f2, [scale];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_FILL_SCALAR;
    ld.global.f32 %f1, [%rd1];
    mul.f32 %f3, %f1, %f2;
    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f3;
DONE_FILL_SCALAR:
    ret;
}

.visible .entry heirloom_fill_constant_f32(
    .param .u64 output,
    .param .f32 value,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<6>;
    .reg .f32 %f<2>;

    ld.param.u64 %rd1, [output];
    ld.param.f32 %f1, [value];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_FILL_CONSTANT;
    mul.wide.u32 %rd2, %r5, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.f32 [%rd3], %f1;
DONE_FILL_CONSTANT:
    ret;
}

.visible .entry heirloom_pad_matrix_bf16(
    .param .u64 input,
    .param .u64 output,
    .param .u32 rows,
    .param .u32 cols,
    .param .u32 padded_cols,
    .param .u32 total
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r4, [rows];
    ld.param.u32 %r5, [cols];
    ld.param.u32 %r6, [padded_cols];
    ld.param.u32 %r7, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r8, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r8, %r7;
    @%p1 bra DONE_PAD_MATRIX_BF16;

    div.u32 %r9, %r8, %r6;
    rem.u32 %r10, %r8, %r6;
    mul.wide.u32 %rd3, %r8, 2;
    add.s64 %rd4, %rd2, %rd3;

    setp.ge.u32 %p2, %r9, %r4;
    @%p2 bra PAD_MATRIX_BF16_ZERO;
    setp.ge.u32 %p3, %r10, %r5;
    @%p3 bra PAD_MATRIX_BF16_ZERO;

    mad.lo.u32 %r11, %r9, %r5, %r10;
    mul.wide.u32 %rd5, %r11, 2;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.u16 %r12, [%rd6];
    bra PAD_MATRIX_BF16_STORE;

PAD_MATRIX_BF16_ZERO:
    mov.u32 %r12, 0;

PAD_MATRIX_BF16_STORE:
    st.global.u16 [%rd4], %r12;
DONE_PAD_MATRIX_BF16:
    ret;
}

.visible .entry heirloom_crop_matrix_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 rows,
    .param .u32 cols,
    .param .u32 padded_cols,
    .param .u32 total
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<14>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<2>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r4, [rows];
    ld.param.u32 %r5, [cols];
    ld.param.u32 %r6, [padded_cols];
    ld.param.u32 %r7, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r8, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r8, %r7;
    @%p1 bra DONE_CROP_MATRIX_F32;

    div.u32 %r9, %r8, %r5;
    rem.u32 %r10, %r8, %r5;
    mad.lo.u32 %r11, %r9, %r6, %r10;
    mul.wide.u32 %rd3, %r11, 4;
    add.s64 %rd4, %rd1, %rd3;
    mul.wide.u32 %rd5, %r8, 4;
    add.s64 %rd6, %rd2, %rd5;
    ld.global.f32 %f1, [%rd4];
    st.global.f32 [%rd6], %f1;
DONE_CROP_MATRIX_F32:
    ret;
}

.visible .entry heirloom_div_backward_rhs_f32(
    .param .u64 left,
    .param .u64 right,
    .param .u64 grad_output,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [left];
    ld.param.u64 %rd2, [right];
    ld.param.u64 %rd3, [grad_output];
    ld.param.u64 %rd4, [output];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_DIV_RHS_BACKWARD;
    mul.wide.u32 %rd5, %r5, 4;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd7, %rd2, %rd5;
    add.s64 %rd8, %rd3, %rd5;
    add.s64 %rd9, %rd4, %rd5;
    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    ld.global.f32 %f3, [%rd8];
    mul.f32 %f4, %f2, %f2;
    mul.f32 %f5, %f3, %f1;
    neg.f32 %f6, %f5;
    div.rn.f32 %f7, %f6, %f4;
    st.global.f32 [%rd9], %f7;
DONE_DIV_RHS_BACKWARD:
    ret;
}

.visible .entry heirloom_matmul_f32(
    .param .u64 left,
    .param .u64 right,
    .param .u64 output,
    .param .u32 m,
    .param .u32 k,
    .param .u32 n,
    .param .u32 total
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<14>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<5>;

    ld.param.u64 %rd1, [left];
    ld.param.u64 %rd2, [right];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [m];
    ld.param.u32 %r5, [k];
    ld.param.u32 %r6, [n];
    ld.param.u32 %r7, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r8, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r8, %r7;
    @%p1 bra DONE_MATMUL;
    div.u32 %r9, %r8, %r6;
    rem.u32 %r10, %r8, %r6;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r11, 0;
MATMUL_LOOP:
    setp.ge.u32 %p2, %r11, %r5;
    @%p2 bra MATMUL_WRITE;
    mad.lo.s32 %r12, %r9, %r5, %r11;
    mul.wide.u32 %rd4, %r12, 4;
    add.s64 %rd5, %rd1, %rd4;
    mad.lo.s32 %r13, %r11, %r6, %r10;
    mul.wide.u32 %rd6, %r13, 4;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.f32 %f2, [%rd5];
    ld.global.f32 %f3, [%rd7];
    mul.f32 %f4, %f2, %f3;
    add.f32 %f1, %f1, %f4;
    add.u32 %r11, %r11, 1;
    bra MATMUL_LOOP;
MATMUL_WRITE:
    mul.wide.u32 %rd8, %r8, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f1;
DONE_MATMUL:
    ret;
}

.visible .entry heirloom_matmul_grad_left_f32(
    .param .u64 grad_output,
    .param .u64 right,
    .param .u64 output,
    .param .u32 m,
    .param .u32 k,
    .param .u32 n,
    .param .u32 total
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<14>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<5>;

    ld.param.u64 %rd1, [grad_output];
    ld.param.u64 %rd2, [right];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [m];
    ld.param.u32 %r5, [k];
    ld.param.u32 %r6, [n];
    ld.param.u32 %r7, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r8, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r8, %r7;
    @%p1 bra DONE_MATMUL_GRAD_LEFT;
    div.u32 %r9, %r8, %r5;
    rem.u32 %r10, %r8, %r5;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r11, 0;
MATMUL_GRAD_LEFT_LOOP:
    setp.ge.u32 %p2, %r11, %r6;
    @%p2 bra MATMUL_GRAD_LEFT_WRITE;
    mad.lo.s32 %r12, %r9, %r6, %r11;
    mul.wide.u32 %rd4, %r12, 4;
    add.s64 %rd5, %rd1, %rd4;
    mad.lo.s32 %r13, %r10, %r6, %r11;
    mul.wide.u32 %rd6, %r13, 4;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.f32 %f2, [%rd5];
    ld.global.f32 %f3, [%rd7];
    mul.f32 %f4, %f2, %f3;
    add.f32 %f1, %f1, %f4;
    add.u32 %r11, %r11, 1;
    bra MATMUL_GRAD_LEFT_LOOP;
MATMUL_GRAD_LEFT_WRITE:
    mul.wide.u32 %rd8, %r8, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f1;
DONE_MATMUL_GRAD_LEFT:
    ret;
}

.visible .entry heirloom_matmul_grad_right_f32(
    .param .u64 left,
    .param .u64 grad_output,
    .param .u64 output,
    .param .u32 m,
    .param .u32 k,
    .param .u32 n,
    .param .u32 total
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<14>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<5>;

    ld.param.u64 %rd1, [left];
    ld.param.u64 %rd2, [grad_output];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [m];
    ld.param.u32 %r5, [k];
    ld.param.u32 %r6, [n];
    ld.param.u32 %r7, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r8, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r8, %r7;
    @%p1 bra DONE_MATMUL_GRAD_RIGHT;
    div.u32 %r9, %r8, %r6;
    rem.u32 %r10, %r8, %r6;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r11, 0;
MATMUL_GRAD_RIGHT_LOOP:
    setp.ge.u32 %p2, %r11, %r4;
    @%p2 bra MATMUL_GRAD_RIGHT_WRITE;
    mad.lo.s32 %r12, %r11, %r5, %r9;
    mul.wide.u32 %rd4, %r12, 4;
    add.s64 %rd5, %rd1, %rd4;
    mad.lo.s32 %r13, %r11, %r6, %r10;
    mul.wide.u32 %rd6, %r13, 4;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.f32 %f2, [%rd5];
    ld.global.f32 %f3, [%rd7];
    mul.f32 %f4, %f2, %f3;
    add.f32 %f1, %f1, %f4;
    add.u32 %r11, %r11, 1;
    bra MATMUL_GRAD_RIGHT_LOOP;
MATMUL_GRAD_RIGHT_WRITE:
    mul.wide.u32 %rd8, %r8, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f1;
DONE_MATMUL_GRAD_RIGHT:
    ret;
}

.visible .entry heirloom_matmul_strided_f32(
    .param .u64 left,
    .param .u64 right,
    .param .u64 output,
    .param .u32 m,
    .param .u32 k,
    .param .u32 n,
    .param .u32 left_row_stride,
    .param .u32 left_col_stride,
    .param .u32 left_offset,
    .param .u32 right_row_stride,
    .param .u32 right_col_stride,
    .param .u32 right_offset,
    .param .u32 total
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<5>;

    ld.param.u64 %rd1, [left];
    ld.param.u64 %rd2, [right];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [m];
    ld.param.u32 %r5, [k];
    ld.param.u32 %r6, [n];
    ld.param.u32 %r7, [left_row_stride];
    ld.param.u32 %r8, [left_col_stride];
    ld.param.u32 %r9, [left_offset];
    ld.param.u32 %r10, [right_row_stride];
    ld.param.u32 %r11, [right_col_stride];
    ld.param.u32 %r12, [right_offset];
    ld.param.u32 %r13, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r14, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r14, %r13;
    @%p1 bra DONE_MATMUL_STRIDED;
    div.u32 %r15, %r14, %r6;
    rem.u32 %r16, %r14, %r6;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r17, 0;
MATMUL_STRIDED_LOOP:
    setp.ge.u32 %p2, %r17, %r5;
    @%p2 bra MATMUL_STRIDED_WRITE;
    mad.lo.s32 %r18, %r15, %r7, %r9;
    mad.lo.s32 %r18, %r17, %r8, %r18;
    mul.wide.u32 %rd4, %r18, 4;
    add.s64 %rd5, %rd1, %rd4;
    mad.lo.s32 %r19, %r17, %r10, %r12;
    mad.lo.s32 %r19, %r16, %r11, %r19;
    mul.wide.u32 %rd6, %r19, 4;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.f32 %f2, [%rd5];
    ld.global.f32 %f3, [%rd7];
    mul.f32 %f4, %f2, %f3;
    add.f32 %f1, %f1, %f4;
    add.u32 %r17, %r17, 1;
    bra MATMUL_STRIDED_LOOP;
MATMUL_STRIDED_WRITE:
    mul.wide.u32 %rd8, %r14, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f1;
DONE_MATMUL_STRIDED:
    ret;
}

.visible .entry heirloom_matmul_strided_grad_left_f32(
    .param .u64 grad_output,
    .param .u64 right,
    .param .u64 output,
    .param .u32 m,
    .param .u32 k,
    .param .u32 n,
    .param .u32 left_row_stride,
    .param .u32 left_col_stride,
    .param .u32 left_offset,
    .param .u32 right_row_stride,
    .param .u32 right_col_stride,
    .param .u32 right_offset,
    .param .u32 total
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<5>;

    ld.param.u64 %rd1, [grad_output];
    ld.param.u64 %rd2, [right];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [m];
    ld.param.u32 %r5, [k];
    ld.param.u32 %r6, [n];
    ld.param.u32 %r7, [left_row_stride];
    ld.param.u32 %r8, [left_col_stride];
    ld.param.u32 %r9, [left_offset];
    ld.param.u32 %r10, [right_row_stride];
    ld.param.u32 %r11, [right_col_stride];
    ld.param.u32 %r12, [right_offset];
    ld.param.u32 %r13, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r14, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r14, %r13;
    @%p1 bra DONE_MATMUL_STRIDED_GRAD_LEFT;
    div.u32 %r15, %r14, %r5;
    rem.u32 %r16, %r14, %r5;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r17, 0;
MATMUL_STRIDED_GRAD_LEFT_LOOP:
    setp.ge.u32 %p2, %r17, %r6;
    @%p2 bra MATMUL_STRIDED_GRAD_LEFT_WRITE;
    mad.lo.s32 %r18, %r15, %r6, %r17;
    mul.wide.u32 %rd4, %r18, 4;
    add.s64 %rd5, %rd1, %rd4;
    mad.lo.s32 %r19, %r16, %r10, %r12;
    mad.lo.s32 %r19, %r17, %r11, %r19;
    mul.wide.u32 %rd6, %r19, 4;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.f32 %f2, [%rd5];
    ld.global.f32 %f3, [%rd7];
    mul.f32 %f4, %f2, %f3;
    add.f32 %f1, %f1, %f4;
    add.u32 %r17, %r17, 1;
    bra MATMUL_STRIDED_GRAD_LEFT_LOOP;
MATMUL_STRIDED_GRAD_LEFT_WRITE:
    mul.wide.u32 %rd8, %r14, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f1;
DONE_MATMUL_STRIDED_GRAD_LEFT:
    ret;
}

.visible .entry heirloom_matmul_strided_grad_right_f32(
    .param .u64 left,
    .param .u64 grad_output,
    .param .u64 output,
    .param .u32 m,
    .param .u32 k,
    .param .u32 n,
    .param .u32 left_row_stride,
    .param .u32 left_col_stride,
    .param .u32 left_offset,
    .param .u32 right_row_stride,
    .param .u32 right_col_stride,
    .param .u32 right_offset,
    .param .u32 total
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<5>;

    ld.param.u64 %rd1, [left];
    ld.param.u64 %rd2, [grad_output];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [m];
    ld.param.u32 %r5, [k];
    ld.param.u32 %r6, [n];
    ld.param.u32 %r7, [left_row_stride];
    ld.param.u32 %r8, [left_col_stride];
    ld.param.u32 %r9, [left_offset];
    ld.param.u32 %r10, [right_row_stride];
    ld.param.u32 %r11, [right_col_stride];
    ld.param.u32 %r12, [right_offset];
    ld.param.u32 %r13, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r14, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r14, %r13;
    @%p1 bra DONE_MATMUL_STRIDED_GRAD_RIGHT;
    div.u32 %r15, %r14, %r6;
    rem.u32 %r16, %r14, %r6;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r17, 0;
MATMUL_STRIDED_GRAD_RIGHT_LOOP:
    setp.ge.u32 %p2, %r17, %r4;
    @%p2 bra MATMUL_STRIDED_GRAD_RIGHT_WRITE;
    mad.lo.s32 %r18, %r17, %r7, %r9;
    mad.lo.s32 %r18, %r15, %r8, %r18;
    mul.wide.u32 %rd4, %r18, 4;
    add.s64 %rd5, %rd1, %rd4;
    mad.lo.s32 %r19, %r17, %r6, %r16;
    mul.wide.u32 %rd6, %r19, 4;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.f32 %f2, [%rd5];
    ld.global.f32 %f3, [%rd7];
    mul.f32 %f4, %f2, %f3;
    add.f32 %f1, %f1, %f4;
    add.u32 %r17, %r17, 1;
    bra MATMUL_STRIDED_GRAD_RIGHT_LOOP;
MATMUL_STRIDED_GRAD_RIGHT_WRITE:
    mul.wide.u32 %rd8, %r14, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f1;
DONE_MATMUL_STRIDED_GRAD_RIGHT:
    ret;
}

.visible .entry heirloom_embedding_f32_i64(
    .param .u64 indices,
    .param .u64 weight,
    .param .u64 output,
    .param .u64 status,
    .param .u32 index_count,
    .param .u32 vocab_size,
    .param .u32 embedding_dim,
    .param .u32 total
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<18>;
    .reg .f32 %f<2>;

    ld.param.u64 %rd1, [indices];
    ld.param.u64 %rd2, [weight];
    ld.param.u64 %rd3, [output];
    ld.param.u64 %rd4, [status];
    ld.param.u32 %r4, [index_count];
    ld.param.u32 %r5, [vocab_size];
    ld.param.u32 %r6, [embedding_dim];
    ld.param.u32 %r7, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r8, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r8, %r7;
    @%p1 bra DONE_EMBEDDING;

    div.u32 %r9, %r8, %r6;
    rem.u32 %r10, %r8, %r6;
    mul.wide.u32 %rd5, %r9, 8;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.s64 %rd7, [%rd6];
    mov.s64 %rd8, 0;
    setp.lt.s64 %p2, %rd7, %rd8;
    cvt.s64.u32 %rd9, %r5;
    setp.ge.s64 %p3, %rd7, %rd9;
    or.pred %p4, %p2, %p3;
    @%p4 bra EMBEDDING_BAD_INDEX;

    cvt.u32.u64 %r11, %rd7;
    mad.lo.s32 %r12, %r11, %r6, %r10;
    mul.wide.u32 %rd10, %r12, 4;
    add.s64 %rd11, %rd2, %rd10;
    ld.global.f32 %f1, [%rd11];
    mul.wide.u32 %rd12, %r8, 4;
    add.s64 %rd13, %rd3, %rd12;
    st.global.f32 [%rd13], %f1;
    bra DONE_EMBEDDING;

EMBEDDING_BAD_INDEX:
    mov.u32 %r13, 1;
    atom.global.exch.b32 %r14, [%rd4], %r13;
DONE_EMBEDDING:
    ret;
}

.visible .entry heirloom_embedding_backward_f32_i64(
    .param .u64 indices,
    .param .u64 grad_output,
    .param .u64 grad_weight,
    .param .u64 status,
    .param .u32 index_count,
    .param .u32 vocab_size,
    .param .u32 embedding_dim,
    .param .u32 total
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<18>;
    .reg .f32 %f<3>;

    ld.param.u64 %rd1, [indices];
    ld.param.u64 %rd2, [grad_output];
    ld.param.u64 %rd3, [grad_weight];
    ld.param.u64 %rd4, [status];
    ld.param.u32 %r4, [index_count];
    ld.param.u32 %r5, [vocab_size];
    ld.param.u32 %r6, [embedding_dim];
    ld.param.u32 %r7, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r8, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r8, %r7;
    @%p1 bra DONE_EMBEDDING_BACKWARD;

    div.u32 %r9, %r8, %r6;
    rem.u32 %r10, %r8, %r6;
    mul.wide.u32 %rd5, %r9, 8;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.s64 %rd7, [%rd6];
    mov.s64 %rd8, 0;
    setp.lt.s64 %p2, %rd7, %rd8;
    cvt.s64.u32 %rd9, %r5;
    setp.ge.s64 %p3, %rd7, %rd9;
    or.pred %p4, %p2, %p3;
    @%p4 bra EMBEDDING_BACKWARD_BAD_INDEX;

    mul.wide.u32 %rd10, %r8, 4;
    add.s64 %rd11, %rd2, %rd10;
    ld.global.f32 %f1, [%rd11];
    cvt.u32.u64 %r11, %rd7;
    mad.lo.s32 %r12, %r11, %r6, %r10;
    mul.wide.u32 %rd12, %r12, 4;
    add.s64 %rd13, %rd3, %rd12;
    atom.global.add.f32 %f2, [%rd13], %f1;
    bra DONE_EMBEDDING_BACKWARD;

EMBEDDING_BACKWARD_BAD_INDEX:
    mov.u32 %r13, 1;
    atom.global.exch.b32 %r14, [%rd4], %r13;
DONE_EMBEDDING_BACKWARD:
    ret;
}

.visible .entry heirloom_memory_topk_f32(
    .param .u64 scores,
    .param .u64 indices,
    .param .u32 rows,
    .param .u32 cols,
    .param .u32 top_k
)
{
    .reg .pred %p<12>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<20>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [scores];
    ld.param.u64 %rd2, [indices];
    ld.param.u32 %r4, [rows];
    ld.param.u32 %r5, [cols];
    ld.param.u32 %r6, [top_k];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r7, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r7, %r4;
    @%p1 bra DONE_MEMORY_TOPK;

    mov.u32 %r8, 0;
MEMORY_TOPK_RANK_LOOP:
    setp.ge.u32 %p2, %r8, %r6;
    @%p2 bra DONE_MEMORY_TOPK;
    mov.u32 %r9, %r5;
    mov.f32 %f1, 0fFF800000;
    mov.u32 %r10, 0;

MEMORY_TOPK_COL_LOOP:
    setp.ge.u32 %p3, %r10, %r5;
    @%p3 bra MEMORY_TOPK_COL_DONE;
    mov.u32 %r11, 0;
    mov.u32 %r12, 0;

MEMORY_TOPK_PRIOR_LOOP:
    setp.ge.u32 %p4, %r11, %r8;
    @%p4 bra MEMORY_TOPK_PRIOR_DONE;
    mad.lo.s32 %r13, %r7, %r6, %r11;
    mul.wide.u32 %rd3, %r13, 8;
    add.s64 %rd4, %rd2, %rd3;
    ld.global.u64 %rd5, [%rd4];
    cvt.u64.u32 %rd6, %r10;
    setp.eq.u64 %p5, %rd5, %rd6;
    @%p5 bra MEMORY_TOPK_MARK_SKIP;
    add.u32 %r11, %r11, 1;
    bra MEMORY_TOPK_PRIOR_LOOP;

MEMORY_TOPK_MARK_SKIP:
    mov.u32 %r12, 1;

MEMORY_TOPK_PRIOR_DONE:
    setp.ne.u32 %p6, %r12, 0;
    @%p6 bra MEMORY_TOPK_NEXT_COL;
    mad.lo.s32 %r14, %r7, %r5, %r10;
    mul.wide.u32 %rd7, %r14, 4;
    add.s64 %rd8, %rd1, %rd7;
    ld.global.f32 %f2, [%rd8];
    setp.eq.u32 %p7, %r9, %r5;
    @%p7 bra MEMORY_TOPK_UPDATE_BEST;
    setp.gt.f32 %p8, %f2, %f1;
    @%p8 bra MEMORY_TOPK_UPDATE_BEST;
    setp.eq.f32 %p9, %f2, %f1;
    setp.lt.u32 %p10, %r10, %r9;
    and.pred %p11, %p9, %p10;
    @%p11 bra MEMORY_TOPK_UPDATE_BEST;
    bra MEMORY_TOPK_NEXT_COL;

MEMORY_TOPK_UPDATE_BEST:
    mov.u32 %r9, %r10;
    mov.f32 %f1, %f2;

MEMORY_TOPK_NEXT_COL:
    add.u32 %r10, %r10, 1;
    bra MEMORY_TOPK_COL_LOOP;

MEMORY_TOPK_COL_DONE:
    mad.lo.s32 %r15, %r7, %r6, %r8;
    mul.wide.u32 %rd9, %r15, 8;
    add.s64 %rd10, %rd2, %rd9;
    cvt.u64.u32 %rd11, %r9;
    st.global.u64 [%rd10], %rd11;
    add.u32 %r8, %r8, 1;
    bra MEMORY_TOPK_RANK_LOOP;

DONE_MEMORY_TOPK:
    ret;
}

.visible .entry heirloom_memory_product_key_side_scores_f32(
    .param .u64 query,
    .param .u64 keys,
    .param .u64 left_scores,
    .param .u64 right_scores,
    .param .u32 tokens,
    .param .u32 side,
    .param .u32 key_dim,
    .param .u32 total
)
{
    .reg .pred %p<7>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<28>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [query];
    ld.param.u64 %rd2, [keys];
    ld.param.u64 %rd3, [left_scores];
    ld.param.u64 %rd4, [right_scores];
    ld.param.u32 %r4, [tokens];
    ld.param.u32 %r5, [side];
    ld.param.u32 %r6, [key_dim];
    ld.param.u32 %r7, [total];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r8, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r8, %r7;
    @%p1 bra DONE_PRODUCT_KEY_SIDE_SCORES;

    mul.lo.u32 %r9, %r4, %r5;
    div.u32 %r10, %r8, %r9;
    rem.u32 %r11, %r8, %r9;
    div.u32 %r12, %r11, %r5;
    rem.u32 %r13, %r11, %r5;
    shr.u32 %r14, %r6, 1;
    mov.f32 %f1, 0fFF800000;
    mov.u32 %r15, 0;

PRODUCT_KEY_SIDE_OTHER_LOOP:
    setp.ge.u32 %p2, %r15, %r5;
    @%p2 bra PRODUCT_KEY_SIDE_WRITE;
    setp.eq.u32 %p3, %r10, 0;
    @%p3 bra PRODUCT_KEY_LEFT_SLOT;

    mad.lo.s32 %r16, %r15, %r5, %r13;
    mov.u32 %r17, %r14;
    bra PRODUCT_KEY_SIDE_DOT_START;

PRODUCT_KEY_LEFT_SLOT:
    mad.lo.s32 %r16, %r13, %r5, %r15;
    mov.u32 %r17, 0;

PRODUCT_KEY_SIDE_DOT_START:
    mov.f32 %f2, 0f00000000;
    mov.u32 %r18, 0;
PRODUCT_KEY_SIDE_DIM_LOOP:
    setp.ge.u32 %p4, %r18, %r14;
    @%p4 bra PRODUCT_KEY_SIDE_DIM_DONE;
    add.u32 %r19, %r17, %r18;
    mad.lo.s32 %r20, %r12, %r6, %r19;
    mad.lo.s32 %r21, %r16, %r6, %r19;
    mul.wide.u32 %rd5, %r20, 4;
    mul.wide.u32 %rd6, %r21, 4;
    add.s64 %rd7, %rd1, %rd5;
    add.s64 %rd8, %rd2, %rd6;
    ld.global.f32 %f3, [%rd7];
    ld.global.f32 %f4, [%rd8];
    fma.rn.f32 %f2, %f3, %f4, %f2;
    add.u32 %r18, %r18, 1;
    bra PRODUCT_KEY_SIDE_DIM_LOOP;

PRODUCT_KEY_SIDE_DIM_DONE:
    max.f32 %f1, %f1, %f2;
    add.u32 %r15, %r15, 1;
    bra PRODUCT_KEY_SIDE_OTHER_LOOP;

PRODUCT_KEY_SIDE_WRITE:
    mad.lo.s32 %r22, %r12, %r5, %r13;
    mul.wide.u32 %rd9, %r22, 4;
    setp.eq.u32 %p5, %r10, 0;
    @%p5 bra PRODUCT_KEY_WRITE_LEFT;
    add.s64 %rd10, %rd4, %rd9;
    st.global.f32 [%rd10], %f1;
    bra DONE_PRODUCT_KEY_SIDE_SCORES;

PRODUCT_KEY_WRITE_LEFT:
    add.s64 %rd11, %rd3, %rd9;
    st.global.f32 [%rd11], %f1;

DONE_PRODUCT_KEY_SIDE_SCORES:
    ret;
}

.visible .entry heirloom_memory_product_key_candidates_f32_i64(
    .param .u64 query,
    .param .u64 keys,
    .param .u64 left_indices,
    .param .u64 right_indices,
    .param .u64 output,
    .param .u64 status,
    .param .u32 tokens,
    .param .u32 side,
    .param .u32 key_dim,
    .param .u32 top_k,
    .param .u32 beam
)
{
    .reg .pred %p<16>;
    .reg .b32 %r<36>;
    .reg .b64 %rd<40>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [query];
    ld.param.u64 %rd2, [keys];
    ld.param.u64 %rd3, [left_indices];
    ld.param.u64 %rd4, [right_indices];
    ld.param.u64 %rd5, [output];
    ld.param.u64 %rd6, [status];
    ld.param.u32 %r4, [tokens];
    ld.param.u32 %r5, [side];
    ld.param.u32 %r6, [key_dim];
    ld.param.u32 %r7, [top_k];
    ld.param.u32 %r8, [beam];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r9, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r9, %r4;
    @%p1 bra DONE_PRODUCT_KEY_CANDIDATES;

    mul.lo.u32 %r10, %r5, %r5;
    mov.u32 %r11, 0;

PRODUCT_KEY_CANDIDATE_RANK_LOOP:
    setp.ge.u32 %p2, %r11, %r7;
    @%p2 bra DONE_PRODUCT_KEY_CANDIDATES;
    mov.u32 %r12, %r10;
    mov.f32 %f1, 0fFF800000;
    mov.u32 %r13, 0;

PRODUCT_KEY_CANDIDATE_LEFT_LOOP:
    setp.ge.u32 %p3, %r13, %r8;
    @%p3 bra PRODUCT_KEY_CANDIDATE_WRITE;
    mad.lo.s32 %r14, %r9, %r8, %r13;
    mul.wide.u32 %rd7, %r14, 8;
    add.s64 %rd8, %rd3, %rd7;
    ld.global.s64 %rd9, [%rd8];
    mov.s64 %rd10, 0;
    setp.lt.s64 %p4, %rd9, %rd10;
    cvt.s64.u32 %rd11, %r5;
    setp.ge.s64 %p5, %rd9, %rd11;
    or.pred %p6, %p4, %p5;
    @%p6 bra PRODUCT_KEY_CANDIDATE_BAD_INDEX;
    cvt.u32.u64 %r15, %rd9;
    mov.u32 %r16, 0;

PRODUCT_KEY_CANDIDATE_RIGHT_LOOP:
    setp.ge.u32 %p7, %r16, %r8;
    @%p7 bra PRODUCT_KEY_CANDIDATE_NEXT_LEFT;
    mad.lo.s32 %r17, %r9, %r8, %r16;
    mul.wide.u32 %rd12, %r17, 8;
    add.s64 %rd13, %rd4, %rd12;
    ld.global.s64 %rd14, [%rd13];
    setp.lt.s64 %p8, %rd14, %rd10;
    setp.ge.s64 %p9, %rd14, %rd11;
    or.pred %p10, %p8, %p9;
    @%p10 bra PRODUCT_KEY_CANDIDATE_BAD_INDEX;
    cvt.u32.u64 %r18, %rd14;
    mad.lo.s32 %r19, %r15, %r5, %r18;

    mov.u32 %r20, 0;
PRODUCT_KEY_CANDIDATE_PRIOR_LOOP:
    setp.ge.u32 %p11, %r20, %r11;
    @%p11 bra PRODUCT_KEY_CANDIDATE_SCORE;
    mad.lo.s32 %r21, %r9, %r7, %r20;
    mul.wide.u32 %rd15, %r21, 8;
    add.s64 %rd16, %rd5, %rd15;
    ld.global.u64 %rd17, [%rd16];
    cvt.u64.u32 %rd18, %r19;
    setp.eq.u64 %p12, %rd17, %rd18;
    @%p12 bra PRODUCT_KEY_CANDIDATE_NEXT_RIGHT;
    add.u32 %r20, %r20, 1;
    bra PRODUCT_KEY_CANDIDATE_PRIOR_LOOP;

PRODUCT_KEY_CANDIDATE_SCORE:
    mov.f32 %f2, 0f00000000;
    mov.u32 %r22, 0;
PRODUCT_KEY_CANDIDATE_DIM_LOOP:
    setp.ge.u32 %p13, %r22, %r6;
    @%p13 bra PRODUCT_KEY_CANDIDATE_SCORE_DONE;
    mad.lo.s32 %r23, %r9, %r6, %r22;
    mad.lo.s32 %r24, %r19, %r6, %r22;
    mul.wide.u32 %rd19, %r23, 4;
    mul.wide.u32 %rd20, %r24, 4;
    add.s64 %rd21, %rd1, %rd19;
    add.s64 %rd22, %rd2, %rd20;
    ld.global.f32 %f3, [%rd21];
    ld.global.f32 %f4, [%rd22];
    fma.rn.f32 %f2, %f3, %f4, %f2;
    add.u32 %r22, %r22, 1;
    bra PRODUCT_KEY_CANDIDATE_DIM_LOOP;

PRODUCT_KEY_CANDIDATE_SCORE_DONE:
    setp.eq.u32 %p14, %r12, %r10;
    @%p14 bra PRODUCT_KEY_CANDIDATE_UPDATE_BEST;
    setp.gt.f32 %p15, %f2, %f1;
    @%p15 bra PRODUCT_KEY_CANDIDATE_UPDATE_BEST;
    setp.eq.f32 %p14, %f2, %f1;
    setp.lt.u32 %p15, %r19, %r12;
    and.pred %p14, %p14, %p15;
    @%p14 bra PRODUCT_KEY_CANDIDATE_UPDATE_BEST;
    bra PRODUCT_KEY_CANDIDATE_NEXT_RIGHT;

PRODUCT_KEY_CANDIDATE_UPDATE_BEST:
    mov.u32 %r12, %r19;
    mov.f32 %f1, %f2;

PRODUCT_KEY_CANDIDATE_NEXT_RIGHT:
    add.u32 %r16, %r16, 1;
    bra PRODUCT_KEY_CANDIDATE_RIGHT_LOOP;

PRODUCT_KEY_CANDIDATE_NEXT_LEFT:
    add.u32 %r13, %r13, 1;
    bra PRODUCT_KEY_CANDIDATE_LEFT_LOOP;

PRODUCT_KEY_CANDIDATE_WRITE:
    setp.eq.u32 %p14, %r12, %r10;
    @%p14 bra PRODUCT_KEY_CANDIDATE_BAD_INDEX;
    mad.lo.s32 %r25, %r9, %r7, %r11;
    mul.wide.u32 %rd23, %r25, 8;
    add.s64 %rd24, %rd5, %rd23;
    cvt.u64.u32 %rd25, %r12;
    st.global.u64 [%rd24], %rd25;
    add.u32 %r11, %r11, 1;
    bra PRODUCT_KEY_CANDIDATE_RANK_LOOP;

PRODUCT_KEY_CANDIDATE_BAD_INDEX:
    mov.u32 %r26, 1;
    st.global.u32 [%rd6], %r26;

DONE_PRODUCT_KEY_CANDIDATES:
    ret;
}

.visible .entry heirloom_memory_product_key_candidates_from_scores_f32_i64(
    .param .u64 left_scores,
    .param .u64 right_scores,
    .param .u64 left_indices,
    .param .u64 right_indices,
    .param .u64 output,
    .param .u64 status,
    .param .u32 tokens,
    .param .u32 side,
    .param .u32 top_k,
    .param .u32 beam
)
{
    .reg .pred %p<16>;
    .reg .b32 %r<34>;
    .reg .b64 %rd<34>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [left_scores];
    ld.param.u64 %rd2, [right_scores];
    ld.param.u64 %rd3, [left_indices];
    ld.param.u64 %rd4, [right_indices];
    ld.param.u64 %rd5, [output];
    ld.param.u64 %rd6, [status];
    ld.param.u32 %r4, [tokens];
    ld.param.u32 %r5, [side];
    ld.param.u32 %r6, [top_k];
    ld.param.u32 %r7, [beam];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r8, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r8, %r4;
    @%p1 bra DONE_PRODUCT_KEY_CANDIDATES_FROM_SCORES;

    mul.lo.u32 %r9, %r5, %r5;
    mov.u32 %r10, 0;

PRODUCT_KEY_CANDIDATES_FROM_SCORES_RANK_LOOP:
    setp.ge.u32 %p2, %r10, %r6;
    @%p2 bra DONE_PRODUCT_KEY_CANDIDATES_FROM_SCORES;
    mov.u32 %r11, %r9;
    mov.f32 %f1, 0fFF800000;
    mov.u32 %r12, 0;

PRODUCT_KEY_CANDIDATES_FROM_SCORES_LEFT_LOOP:
    setp.ge.u32 %p3, %r12, %r7;
    @%p3 bra PRODUCT_KEY_CANDIDATES_FROM_SCORES_WRITE;
    mad.lo.s32 %r13, %r8, %r7, %r12;
    mul.wide.u32 %rd7, %r13, 8;
    add.s64 %rd8, %rd3, %rd7;
    ld.global.s64 %rd9, [%rd8];
    mov.s64 %rd10, 0;
    setp.lt.s64 %p4, %rd9, %rd10;
    cvt.s64.u32 %rd11, %r5;
    setp.ge.s64 %p5, %rd9, %rd11;
    or.pred %p6, %p4, %p5;
    @%p6 bra PRODUCT_KEY_CANDIDATES_FROM_SCORES_BAD_INDEX;
    cvt.u32.u64 %r14, %rd9;
    mov.u32 %r15, 0;

PRODUCT_KEY_CANDIDATES_FROM_SCORES_RIGHT_LOOP:
    setp.ge.u32 %p7, %r15, %r7;
    @%p7 bra PRODUCT_KEY_CANDIDATES_FROM_SCORES_NEXT_LEFT;
    mad.lo.s32 %r16, %r8, %r7, %r15;
    mul.wide.u32 %rd12, %r16, 8;
    add.s64 %rd13, %rd4, %rd12;
    ld.global.s64 %rd14, [%rd13];
    setp.lt.s64 %p8, %rd14, %rd10;
    setp.ge.s64 %p9, %rd14, %rd11;
    or.pred %p10, %p8, %p9;
    @%p10 bra PRODUCT_KEY_CANDIDATES_FROM_SCORES_BAD_INDEX;
    cvt.u32.u64 %r17, %rd14;
    mad.lo.s32 %r18, %r14, %r5, %r17;

    mov.u32 %r19, 0;
PRODUCT_KEY_CANDIDATES_FROM_SCORES_PRIOR_LOOP:
    setp.ge.u32 %p11, %r19, %r10;
    @%p11 bra PRODUCT_KEY_CANDIDATES_FROM_SCORES_SCORE;
    mad.lo.s32 %r20, %r8, %r6, %r19;
    mul.wide.u32 %rd15, %r20, 8;
    add.s64 %rd16, %rd5, %rd15;
    ld.global.u64 %rd17, [%rd16];
    cvt.u64.u32 %rd18, %r18;
    setp.eq.u64 %p12, %rd17, %rd18;
    @%p12 bra PRODUCT_KEY_CANDIDATES_FROM_SCORES_NEXT_RIGHT;
    add.u32 %r19, %r19, 1;
    bra PRODUCT_KEY_CANDIDATES_FROM_SCORES_PRIOR_LOOP;

PRODUCT_KEY_CANDIDATES_FROM_SCORES_SCORE:
    mad.lo.s32 %r21, %r8, %r5, %r14;
    mul.wide.u32 %rd19, %r21, 4;
    add.s64 %rd20, %rd1, %rd19;
    ld.global.f32 %f2, [%rd20];
    mad.lo.s32 %r22, %r8, %r5, %r17;
    mul.wide.u32 %rd21, %r22, 4;
    add.s64 %rd22, %rd2, %rd21;
    ld.global.f32 %f3, [%rd22];
    add.f32 %f4, %f2, %f3;

    setp.eq.u32 %p13, %r11, %r9;
    @%p13 bra PRODUCT_KEY_CANDIDATES_FROM_SCORES_UPDATE_BEST;
    setp.gt.f32 %p14, %f4, %f1;
    @%p14 bra PRODUCT_KEY_CANDIDATES_FROM_SCORES_UPDATE_BEST;
    setp.eq.f32 %p13, %f4, %f1;
    setp.lt.u32 %p14, %r18, %r11;
    and.pred %p13, %p13, %p14;
    @%p13 bra PRODUCT_KEY_CANDIDATES_FROM_SCORES_UPDATE_BEST;
    bra PRODUCT_KEY_CANDIDATES_FROM_SCORES_NEXT_RIGHT;

PRODUCT_KEY_CANDIDATES_FROM_SCORES_UPDATE_BEST:
    mov.u32 %r11, %r18;
    mov.f32 %f1, %f4;

PRODUCT_KEY_CANDIDATES_FROM_SCORES_NEXT_RIGHT:
    add.u32 %r15, %r15, 1;
    bra PRODUCT_KEY_CANDIDATES_FROM_SCORES_RIGHT_LOOP;

PRODUCT_KEY_CANDIDATES_FROM_SCORES_NEXT_LEFT:
    add.u32 %r12, %r12, 1;
    bra PRODUCT_KEY_CANDIDATES_FROM_SCORES_LEFT_LOOP;

PRODUCT_KEY_CANDIDATES_FROM_SCORES_WRITE:
    setp.eq.u32 %p15, %r11, %r9;
    @%p15 bra PRODUCT_KEY_CANDIDATES_FROM_SCORES_BAD_INDEX;
    mad.lo.s32 %r23, %r8, %r6, %r10;
    mul.wide.u32 %rd23, %r23, 8;
    add.s64 %rd24, %rd5, %rd23;
    cvt.u64.u32 %rd25, %r11;
    st.global.u64 [%rd24], %rd25;
    add.u32 %r10, %r10, 1;
    bra PRODUCT_KEY_CANDIDATES_FROM_SCORES_RANK_LOOP;

PRODUCT_KEY_CANDIDATES_FROM_SCORES_BAD_INDEX:
    mov.u32 %r24, 1;
    st.global.u32 [%rd6], %r24;

DONE_PRODUCT_KEY_CANDIDATES_FROM_SCORES:
    ret;
}

.visible .entry heirloom_memory_product_key_split_rows_i64(
    .param .u64 selected_rows,
    .param .u64 left_rows,
    .param .u64 right_rows,
    .param .u64 status,
    .param .u32 side,
    .param .u32 total
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<18>;

    ld.param.u64 %rd1, [selected_rows];
    ld.param.u64 %rd2, [left_rows];
    ld.param.u64 %rd3, [right_rows];
    ld.param.u64 %rd4, [status];
    ld.param.u32 %r4, [side];
    ld.param.u32 %r5, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r6, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r6, %r5;
    @%p1 bra DONE_PRODUCT_KEY_SPLIT_ROWS;

    mul.wide.u32 %rd5, %r6, 8;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.s64 %rd7, [%rd6];
    mov.s64 %rd8, 0;
    setp.lt.s64 %p2, %rd7, %rd8;
    @%p2 bra PRODUCT_KEY_SPLIT_ROWS_BAD_INDEX;
    cvt.u32.u64 %r7, %rd7;
    mul.lo.u32 %r8, %r4, %r4;
    setp.ge.u32 %p3, %r7, %r8;
    @%p3 bra PRODUCT_KEY_SPLIT_ROWS_BAD_INDEX;
    div.u32 %r9, %r7, %r4;
    rem.u32 %r10, %r7, %r4;
    add.s64 %rd9, %rd2, %rd5;
    add.s64 %rd10, %rd3, %rd5;
    cvt.u64.u32 %rd11, %r9;
    cvt.u64.u32 %rd12, %r10;
    st.global.u64 [%rd9], %rd11;
    st.global.u64 [%rd10], %rd12;
    bra DONE_PRODUCT_KEY_SPLIT_ROWS;

PRODUCT_KEY_SPLIT_ROWS_BAD_INDEX:
    mov.u32 %r11, 1;
    st.global.u32 [%rd4], %r11;

DONE_PRODUCT_KEY_SPLIT_ROWS:
    ret;
}

.visible .entry heirloom_memory_product_key_selected_scores_forward_f32_i64(
    .param .u64 indices,
    .param .u64 query,
    .param .u64 left_keys,
    .param .u64 right_keys,
    .param .u64 output,
    .param .u64 status,
    .param .u32 tokens,
    .param .u32 top_k,
    .param .u32 side,
    .param .u32 key_dim,
    .param .u32 total
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<30>;
    .reg .b64 %rd<34>;
    .reg .f32 %f<10>;

    ld.param.u64 %rd1, [indices];
    ld.param.u64 %rd2, [query];
    ld.param.u64 %rd3, [left_keys];
    ld.param.u64 %rd4, [right_keys];
    ld.param.u64 %rd5, [output];
    ld.param.u64 %rd6, [status];
    ld.param.u32 %r4, [tokens];
    ld.param.u32 %r5, [top_k];
    ld.param.u32 %r6, [side];
    ld.param.u32 %r7, [key_dim];
    ld.param.u32 %r8, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r9, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r9, %r8;
    @%p1 bra DONE_PRODUCT_KEY_SELECTED_SCORES_FWD;

    div.u32 %r10, %r9, %r5;
    mul.wide.u32 %rd7, %r9, 8;
    add.s64 %rd8, %rd1, %rd7;
    ld.global.s64 %rd9, [%rd8];
    mov.s64 %rd10, 0;
    setp.lt.s64 %p2, %rd9, %rd10;
    cvt.s64.u32 %rd11, %r6;
    mul.wide.u32 %rd12, %r6, %r6;
    cvt.s64.u64 %rd13, %rd12;
    setp.ge.s64 %p3, %rd9, %rd13;
    or.pred %p4, %p2, %p3;
    @%p4 bra PRODUCT_KEY_SELECTED_SCORES_FWD_BAD_INDEX;
    cvt.u32.u64 %r11, %rd9;
    div.u32 %r12, %r11, %r6;
    rem.u32 %r13, %r11, %r6;
    shr.u32 %r14, %r7, 1;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r15, 0;

PRODUCT_KEY_SELECTED_SCORES_FWD_LOOP:
    setp.ge.u32 %p5, %r15, %r14;
    @%p5 bra PRODUCT_KEY_SELECTED_SCORES_FWD_WRITE;
    mad.lo.s32 %r16, %r10, %r7, %r15;
    mul.wide.u32 %rd14, %r16, 4;
    add.s64 %rd15, %rd2, %rd14;
    ld.global.f32 %f2, [%rd15];
    mad.lo.s32 %r17, %r12, %r14, %r15;
    mul.wide.u32 %rd16, %r17, 4;
    add.s64 %rd17, %rd3, %rd16;
    ld.global.f32 %f3, [%rd17];
    fma.rn.f32 %f1, %f2, %f3, %f1;

    add.u32 %r18, %r14, %r15;
    mad.lo.s32 %r19, %r10, %r7, %r18;
    mul.wide.u32 %rd18, %r19, 4;
    add.s64 %rd19, %rd2, %rd18;
    ld.global.f32 %f4, [%rd19];
    mad.lo.s32 %r20, %r13, %r14, %r15;
    mul.wide.u32 %rd20, %r20, 4;
    add.s64 %rd21, %rd4, %rd20;
    ld.global.f32 %f5, [%rd21];
    fma.rn.f32 %f1, %f4, %f5, %f1;
    add.u32 %r15, %r15, 1;
    bra PRODUCT_KEY_SELECTED_SCORES_FWD_LOOP;

PRODUCT_KEY_SELECTED_SCORES_FWD_WRITE:
    mul.wide.u32 %rd22, %r9, 4;
    add.s64 %rd23, %rd5, %rd22;
    st.global.f32 [%rd23], %f1;
    bra DONE_PRODUCT_KEY_SELECTED_SCORES_FWD;

PRODUCT_KEY_SELECTED_SCORES_FWD_BAD_INDEX:
    mov.u32 %r21, 1;
    st.global.u32 [%rd6], %r21;
    mul.wide.u32 %rd22, %r9, 4;
    add.s64 %rd23, %rd5, %rd22;
    mov.f32 %f1, 0f00000000;
    st.global.f32 [%rd23], %f1;

DONE_PRODUCT_KEY_SELECTED_SCORES_FWD:
    ret;
}

.visible .entry heirloom_memory_product_key_selected_scores_backward_query_f32_i64(
    .param .u64 indices,
    .param .u64 left_keys,
    .param .u64 right_keys,
    .param .u64 grad_scores,
    .param .u64 grad_query,
    .param .u64 status,
    .param .u32 tokens,
    .param .u32 top_k,
    .param .u32 side,
    .param .u32 key_dim,
    .param .u32 total
)
{
    .reg .pred %p<9>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<34>;
    .reg .f32 %f<10>;

    ld.param.u64 %rd1, [indices];
    ld.param.u64 %rd2, [left_keys];
    ld.param.u64 %rd3, [right_keys];
    ld.param.u64 %rd4, [grad_scores];
    ld.param.u64 %rd5, [grad_query];
    ld.param.u64 %rd6, [status];
    ld.param.u32 %r4, [tokens];
    ld.param.u32 %r5, [top_k];
    ld.param.u32 %r6, [side];
    ld.param.u32 %r7, [key_dim];
    ld.param.u32 %r8, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r9, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r9, %r8;
    @%p1 bra DONE_PRODUCT_KEY_SELECTED_SCORES_BACK_QUERY;

    div.u32 %r10, %r9, %r7;
    rem.u32 %r11, %r9, %r7;
    shr.u32 %r12, %r7, 1;
    setp.lt.u32 %p2, %r11, %r12;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r13, 0;

PRODUCT_KEY_SELECTED_SCORES_BACK_QUERY_LOOP:
    setp.ge.u32 %p3, %r13, %r5;
    @%p3 bra PRODUCT_KEY_SELECTED_SCORES_BACK_QUERY_WRITE;
    mad.lo.s32 %r14, %r10, %r5, %r13;
    mul.wide.u32 %rd7, %r14, 8;
    add.s64 %rd8, %rd1, %rd7;
    ld.global.s64 %rd9, [%rd8];
    mov.s64 %rd10, 0;
    setp.lt.s64 %p4, %rd9, %rd10;
    mul.wide.u32 %rd11, %r6, %r6;
    cvt.s64.u64 %rd12, %rd11;
    setp.ge.s64 %p5, %rd9, %rd12;
    or.pred %p6, %p4, %p5;
    @%p6 bra PRODUCT_KEY_SELECTED_SCORES_BACK_QUERY_BAD_INDEX;
    cvt.u32.u64 %r15, %rd9;
    div.u32 %r16, %r15, %r6;
    rem.u32 %r17, %r15, %r6;
    mul.wide.u32 %rd13, %r14, 4;
    add.s64 %rd14, %rd4, %rd13;
    ld.global.f32 %f2, [%rd14];
    @%p2 bra PRODUCT_KEY_SELECTED_SCORES_BACK_QUERY_LEFT;
    sub.u32 %r18, %r11, %r12;
    mad.lo.s32 %r19, %r17, %r12, %r18;
    mul.wide.u32 %rd15, %r19, 4;
    add.s64 %rd16, %rd3, %rd15;
    ld.global.f32 %f3, [%rd16];
    bra PRODUCT_KEY_SELECTED_SCORES_BACK_QUERY_ACCUM;

PRODUCT_KEY_SELECTED_SCORES_BACK_QUERY_LEFT:
    mad.lo.s32 %r19, %r16, %r12, %r11;
    mul.wide.u32 %rd15, %r19, 4;
    add.s64 %rd16, %rd2, %rd15;
    ld.global.f32 %f3, [%rd16];

PRODUCT_KEY_SELECTED_SCORES_BACK_QUERY_ACCUM:
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r13, %r13, 1;
    bra PRODUCT_KEY_SELECTED_SCORES_BACK_QUERY_LOOP;

PRODUCT_KEY_SELECTED_SCORES_BACK_QUERY_BAD_INDEX:
    mov.u32 %r20, 1;
    st.global.u32 [%rd6], %r20;
    add.u32 %r13, %r13, 1;
    bra PRODUCT_KEY_SELECTED_SCORES_BACK_QUERY_LOOP;

PRODUCT_KEY_SELECTED_SCORES_BACK_QUERY_WRITE:
    mul.wide.u32 %rd17, %r9, 4;
    add.s64 %rd18, %rd5, %rd17;
    st.global.f32 [%rd18], %f1;

DONE_PRODUCT_KEY_SELECTED_SCORES_BACK_QUERY:
    ret;
}

.visible .entry heirloom_memory_product_key_selected_scores_backward_half_keys_f32_i64(
    .param .u64 indices,
    .param .u64 query,
    .param .u64 grad_scores,
    .param .u64 grad_half_keys,
    .param .u64 status,
    .param .u32 tokens,
    .param .u32 top_k,
    .param .u32 side,
    .param .u32 key_dim,
    .param .u32 is_left,
    .param .u32 total
)
{
    .reg .pred %p<12>;
    .reg .b32 %r<36>;
    .reg .b64 %rd<34>;
    .reg .f32 %f<10>;

    ld.param.u64 %rd1, [indices];
    ld.param.u64 %rd2, [query];
    ld.param.u64 %rd3, [grad_scores];
    ld.param.u64 %rd4, [grad_half_keys];
    ld.param.u64 %rd5, [status];
    ld.param.u32 %r4, [tokens];
    ld.param.u32 %r5, [top_k];
    ld.param.u32 %r6, [side];
    ld.param.u32 %r7, [key_dim];
    ld.param.u32 %r8, [is_left];
    ld.param.u32 %r9, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r10, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r10, %r9;
    @%p1 bra DONE_PRODUCT_KEY_SELECTED_SCORES_BACK_HALF;

    shr.u32 %r11, %r7, 1;
    div.u32 %r12, %r10, %r11;
    rem.u32 %r13, %r10, %r11;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r14, 0;

PRODUCT_KEY_SELECTED_SCORES_BACK_HALF_TOKEN_LOOP:
    setp.ge.u32 %p2, %r14, %r4;
    @%p2 bra PRODUCT_KEY_SELECTED_SCORES_BACK_HALF_WRITE;
    mov.u32 %r15, 0;

PRODUCT_KEY_SELECTED_SCORES_BACK_HALF_TOPK_LOOP:
    setp.ge.u32 %p3, %r15, %r5;
    @%p3 bra PRODUCT_KEY_SELECTED_SCORES_BACK_HALF_NEXT_TOKEN;
    mad.lo.s32 %r16, %r14, %r5, %r15;
    mul.wide.u32 %rd6, %r16, 8;
    add.s64 %rd7, %rd1, %rd6;
    ld.global.s64 %rd8, [%rd7];
    mov.s64 %rd9, 0;
    setp.lt.s64 %p4, %rd8, %rd9;
    mul.wide.u32 %rd10, %r6, %r6;
    cvt.s64.u64 %rd11, %rd10;
    setp.ge.s64 %p5, %rd8, %rd11;
    or.pred %p6, %p4, %p5;
    @%p6 bra PRODUCT_KEY_SELECTED_SCORES_BACK_HALF_BAD_INDEX;
    cvt.u32.u64 %r17, %rd8;
    div.u32 %r18, %r17, %r6;
    rem.u32 %r19, %r17, %r6;
    setp.ne.u32 %p7, %r8, 0;
    @%p7 bra PRODUCT_KEY_SELECTED_SCORES_BACK_HALF_CHECK_LEFT;
    setp.ne.u32 %p8, %r19, %r12;
    @%p8 bra PRODUCT_KEY_SELECTED_SCORES_BACK_HALF_NEXT_SELECTED;
    add.u32 %r20, %r11, %r13;
    bra PRODUCT_KEY_SELECTED_SCORES_BACK_HALF_ACCUM;

PRODUCT_KEY_SELECTED_SCORES_BACK_HALF_CHECK_LEFT:
    setp.ne.u32 %p9, %r18, %r12;
    @%p9 bra PRODUCT_KEY_SELECTED_SCORES_BACK_HALF_NEXT_SELECTED;
    mov.u32 %r20, %r13;

PRODUCT_KEY_SELECTED_SCORES_BACK_HALF_ACCUM:
    mul.wide.u32 %rd12, %r16, 4;
    add.s64 %rd13, %rd3, %rd12;
    ld.global.f32 %f2, [%rd13];
    mad.lo.s32 %r21, %r14, %r7, %r20;
    mul.wide.u32 %rd14, %r21, 4;
    add.s64 %rd15, %rd2, %rd14;
    ld.global.f32 %f3, [%rd15];
    fma.rn.f32 %f1, %f2, %f3, %f1;

PRODUCT_KEY_SELECTED_SCORES_BACK_HALF_NEXT_SELECTED:
    add.u32 %r15, %r15, 1;
    bra PRODUCT_KEY_SELECTED_SCORES_BACK_HALF_TOPK_LOOP;

PRODUCT_KEY_SELECTED_SCORES_BACK_HALF_BAD_INDEX:
    mov.u32 %r22, 1;
    st.global.u32 [%rd5], %r22;
    add.u32 %r15, %r15, 1;
    bra PRODUCT_KEY_SELECTED_SCORES_BACK_HALF_TOPK_LOOP;

PRODUCT_KEY_SELECTED_SCORES_BACK_HALF_NEXT_TOKEN:
    add.u32 %r14, %r14, 1;
    bra PRODUCT_KEY_SELECTED_SCORES_BACK_HALF_TOKEN_LOOP;

PRODUCT_KEY_SELECTED_SCORES_BACK_HALF_WRITE:
    mul.wide.u32 %rd16, %r10, 4;
    add.s64 %rd17, %rd4, %rd16;
    st.global.f32 [%rd17], %f1;

DONE_PRODUCT_KEY_SELECTED_SCORES_BACK_HALF:
    ret;
}

.visible .entry heirloom_softmax_dim1_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 rows,
    .param .u32 cols
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<12>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r4, [rows];
    ld.param.u32 %r5, [cols];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r6, %r2, %r3, %r1;
    mul.lo.u32 %r7, %r4, %r5;
    setp.ge.u32 %p1, %r6, %r7;
    @%p1 bra DONE_SOFTMAX_DIM1;

    div.u32 %r8, %r6, %r5;
    rem.u32 %r9, %r6, %r5;
    mov.f32 %f1, 0fFF800000;
    mov.u32 %r10, 0;
SOFTMAX_DIM1_MAX_LOOP:
    setp.ge.u32 %p2, %r10, %r5;
    @%p2 bra SOFTMAX_DIM1_MAX_DONE;
    mad.lo.s32 %r11, %r8, %r5, %r10;
    mul.wide.u32 %rd3, %r11, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f2, [%rd4];
    max.f32 %f1, %f1, %f2;
    add.u32 %r10, %r10, 1;
    bra SOFTMAX_DIM1_MAX_LOOP;

SOFTMAX_DIM1_MAX_DONE:
    mov.f32 %f3, 0f00000000;
    mov.u32 %r10, 0;
SOFTMAX_DIM1_SUM_LOOP:
    setp.ge.u32 %p3, %r10, %r5;
    @%p3 bra SOFTMAX_DIM1_SUM_DONE;
    mad.lo.s32 %r11, %r8, %r5, %r10;
    mul.wide.u32 %rd3, %r11, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f2, [%rd4];
    sub.f32 %f4, %f2, %f1;
    mul.f32 %f4, %f4, 0f3FB8AA3B;
    ex2.approx.ftz.f32 %f5, %f4;
    add.f32 %f3, %f3, %f5;
    add.u32 %r10, %r10, 1;
    bra SOFTMAX_DIM1_SUM_LOOP;

SOFTMAX_DIM1_SUM_DONE:
    mad.lo.s32 %r12, %r8, %r5, %r9;
    mul.wide.u32 %rd5, %r12, 4;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd7, %rd2, %rd5;
    ld.global.f32 %f6, [%rd6];
    sub.f32 %f7, %f6, %f1;
    mul.f32 %f7, %f7, 0f3FB8AA3B;
    ex2.approx.ftz.f32 %f8, %f7;
    div.rn.f32 %f9, %f8, %f3;
    st.global.f32 [%rd7], %f9;

DONE_SOFTMAX_DIM1:
    ret;
}

.visible .entry heirloom_softmax_dim1_backward_f32(
    .param .u64 input,
    .param .u64 grad_output,
    .param .u64 grad_input,
    .param .u32 rows,
    .param .u32 cols
)
{
    .reg .pred %p<6>;
    .reg .b32 %r<18>;
    .reg .b64 %rd<18>;
    .reg .f32 %f<20>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [grad_output];
    ld.param.u64 %rd3, [grad_input];
    ld.param.u32 %r4, [rows];
    ld.param.u32 %r5, [cols];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r6, %r2, %r3, %r1;
    mul.lo.u32 %r7, %r4, %r5;
    setp.ge.u32 %p1, %r6, %r7;
    @%p1 bra DONE_SOFTMAX_DIM1_BACKWARD;

    div.u32 %r8, %r6, %r5;
    rem.u32 %r9, %r6, %r5;
    mov.f32 %f1, 0fFF800000;
    mov.u32 %r10, 0;
SOFTMAX_DIM1_BACK_MAX_LOOP:
    setp.ge.u32 %p2, %r10, %r5;
    @%p2 bra SOFTMAX_DIM1_BACK_MAX_DONE;
    mad.lo.s32 %r11, %r8, %r5, %r10;
    mul.wide.u32 %rd4, %r11, 4;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];
    max.f32 %f1, %f1, %f2;
    add.u32 %r10, %r10, 1;
    bra SOFTMAX_DIM1_BACK_MAX_LOOP;

SOFTMAX_DIM1_BACK_MAX_DONE:
    mov.f32 %f3, 0f00000000;
    mov.u32 %r10, 0;
SOFTMAX_DIM1_BACK_SUM_LOOP:
    setp.ge.u32 %p3, %r10, %r5;
    @%p3 bra SOFTMAX_DIM1_BACK_SUM_DONE;
    mad.lo.s32 %r11, %r8, %r5, %r10;
    mul.wide.u32 %rd4, %r11, 4;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];
    sub.f32 %f4, %f2, %f1;
    mul.f32 %f4, %f4, 0f3FB8AA3B;
    ex2.approx.ftz.f32 %f5, %f4;
    add.f32 %f3, %f3, %f5;
    add.u32 %r10, %r10, 1;
    bra SOFTMAX_DIM1_BACK_SUM_LOOP;

SOFTMAX_DIM1_BACK_SUM_DONE:
    mov.f32 %f6, 0f00000000;
    mov.u32 %r10, 0;
SOFTMAX_DIM1_BACK_DOT_LOOP:
    setp.ge.u32 %p4, %r10, %r5;
    @%p4 bra SOFTMAX_DIM1_BACK_DOT_DONE;
    mad.lo.s32 %r11, %r8, %r5, %r10;
    mul.wide.u32 %rd4, %r11, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    ld.global.f32 %f2, [%rd5];
    ld.global.f32 %f7, [%rd6];
    sub.f32 %f4, %f2, %f1;
    mul.f32 %f4, %f4, 0f3FB8AA3B;
    ex2.approx.ftz.f32 %f5, %f4;
    div.rn.f32 %f8, %f5, %f3;
    mul.f32 %f9, %f8, %f7;
    add.f32 %f6, %f6, %f9;
    add.u32 %r10, %r10, 1;
    bra SOFTMAX_DIM1_BACK_DOT_LOOP;

SOFTMAX_DIM1_BACK_DOT_DONE:
    mad.lo.s32 %r12, %r8, %r5, %r9;
    mul.wide.u32 %rd7, %r12, 4;
    add.s64 %rd8, %rd1, %rd7;
    add.s64 %rd9, %rd2, %rd7;
    add.s64 %rd10, %rd3, %rd7;
    ld.global.f32 %f10, [%rd8];
    ld.global.f32 %f11, [%rd9];
    sub.f32 %f12, %f10, %f1;
    mul.f32 %f12, %f12, 0f3FB8AA3B;
    ex2.approx.ftz.f32 %f13, %f12;
    div.rn.f32 %f14, %f13, %f3;
    sub.f32 %f15, %f11, %f6;
    mul.f32 %f16, %f14, %f15;
    st.global.f32 [%rd10], %f16;

DONE_SOFTMAX_DIM1_BACKWARD:
    ret;
}

.visible .entry heirloom_memory_weighted_value_forward_f32_i64(
    .param .u64 indices,
    .param .u64 weights,
    .param .u64 values,
    .param .u64 output,
    .param .u64 status,
    .param .u32 tokens,
    .param .u32 top_k,
    .param .u32 slots,
    .param .u32 value_dim,
    .param .u32 total
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<24>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [indices];
    ld.param.u64 %rd2, [weights];
    ld.param.u64 %rd3, [values];
    ld.param.u64 %rd4, [output];
    ld.param.u64 %rd5, [status];
    ld.param.u32 %r4, [tokens];
    ld.param.u32 %r5, [top_k];
    ld.param.u32 %r6, [slots];
    ld.param.u32 %r7, [value_dim];
    ld.param.u32 %r8, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r9, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r9, %r8;
    @%p1 bra DONE_MEMORY_WEIGHTED_VALUE_FWD;

    div.u32 %r10, %r9, %r7;
    rem.u32 %r11, %r9, %r7;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r12, 0;
MEMORY_WEIGHTED_VALUE_FWD_LOOP:
    setp.ge.u32 %p2, %r12, %r5;
    @%p2 bra MEMORY_WEIGHTED_VALUE_FWD_WRITE;
    mad.lo.s32 %r13, %r10, %r5, %r12;
    mul.wide.u32 %rd6, %r13, 8;
    add.s64 %rd7, %rd1, %rd6;
    ld.global.s64 %rd8, [%rd7];
    mov.s64 %rd9, 0;
    setp.lt.s64 %p3, %rd8, %rd9;
    cvt.s64.u32 %rd10, %r6;
    setp.ge.s64 %p4, %rd8, %rd10;
    or.pred %p3, %p3, %p4;
    @%p3 bra MEMORY_WEIGHTED_VALUE_FWD_BAD_INDEX;
    cvt.u32.u64 %r14, %rd8;
    mul.wide.u32 %rd11, %r13, 4;
    add.s64 %rd12, %rd2, %rd11;
    ld.global.f32 %f2, [%rd12];
    mad.lo.s32 %r15, %r14, %r7, %r11;
    mul.wide.u32 %rd13, %r15, 4;
    add.s64 %rd14, %rd3, %rd13;
    ld.global.f32 %f3, [%rd14];
    mul.f32 %f4, %f2, %f3;
    add.f32 %f1, %f1, %f4;
    add.u32 %r12, %r12, 1;
    bra MEMORY_WEIGHTED_VALUE_FWD_LOOP;

MEMORY_WEIGHTED_VALUE_FWD_BAD_INDEX:
    mov.u32 %r16, 1;
    atom.global.exch.b32 %r17, [%rd5], %r16;
    add.u32 %r12, %r12, 1;
    bra MEMORY_WEIGHTED_VALUE_FWD_LOOP;

MEMORY_WEIGHTED_VALUE_FWD_WRITE:
    mul.wide.u32 %rd15, %r9, 4;
    add.s64 %rd16, %rd4, %rd15;
    st.global.f32 [%rd16], %f1;
DONE_MEMORY_WEIGHTED_VALUE_FWD:
    ret;
}

.visible .entry heirloom_memory_weighted_value_backward_weights_f32_i64(
    .param .u64 indices,
    .param .u64 values,
    .param .u64 grad_output,
    .param .u64 grad_weights,
    .param .u64 status,
    .param .u32 tokens,
    .param .u32 top_k,
    .param .u32 slots,
    .param .u32 value_dim,
    .param .u32 total
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<24>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [indices];
    ld.param.u64 %rd2, [values];
    ld.param.u64 %rd3, [grad_output];
    ld.param.u64 %rd4, [grad_weights];
    ld.param.u64 %rd5, [status];
    ld.param.u32 %r4, [tokens];
    ld.param.u32 %r5, [top_k];
    ld.param.u32 %r6, [slots];
    ld.param.u32 %r7, [value_dim];
    ld.param.u32 %r8, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r9, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r9, %r8;
    @%p1 bra DONE_MEMORY_WEIGHTED_VALUE_BACK_WEIGHTS;

    div.u32 %r10, %r9, %r5;
    rem.u32 %r11, %r9, %r5;
    mul.wide.u32 %rd6, %r9, 8;
    add.s64 %rd7, %rd1, %rd6;
    ld.global.s64 %rd8, [%rd7];
    mov.s64 %rd9, 0;
    setp.lt.s64 %p2, %rd8, %rd9;
    cvt.s64.u32 %rd10, %r6;
    setp.ge.s64 %p3, %rd8, %rd10;
    or.pred %p4, %p2, %p3;
    @%p4 bra MEMORY_WEIGHTED_VALUE_BACK_WEIGHTS_BAD_INDEX;
    cvt.u32.u64 %r12, %rd8;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r13, 0;
MEMORY_WEIGHTED_VALUE_BACK_WEIGHTS_LOOP:
    setp.ge.u32 %p2, %r13, %r7;
    @%p2 bra MEMORY_WEIGHTED_VALUE_BACK_WEIGHTS_WRITE;
    mad.lo.s32 %r14, %r10, %r7, %r13;
    mul.wide.u32 %rd11, %r14, 4;
    add.s64 %rd12, %rd3, %rd11;
    ld.global.f32 %f2, [%rd12];
    mad.lo.s32 %r15, %r12, %r7, %r13;
    mul.wide.u32 %rd13, %r15, 4;
    add.s64 %rd14, %rd2, %rd13;
    ld.global.f32 %f3, [%rd14];
    mul.f32 %f4, %f2, %f3;
    add.f32 %f1, %f1, %f4;
    add.u32 %r13, %r13, 1;
    bra MEMORY_WEIGHTED_VALUE_BACK_WEIGHTS_LOOP;

MEMORY_WEIGHTED_VALUE_BACK_WEIGHTS_WRITE:
    mul.wide.u32 %rd15, %r9, 4;
    add.s64 %rd16, %rd4, %rd15;
    st.global.f32 [%rd16], %f1;
    bra DONE_MEMORY_WEIGHTED_VALUE_BACK_WEIGHTS;

MEMORY_WEIGHTED_VALUE_BACK_WEIGHTS_BAD_INDEX:
    mov.u32 %r16, 1;
    atom.global.exch.b32 %r17, [%rd5], %r16;
    mul.wide.u32 %rd15, %r9, 4;
    add.s64 %rd16, %rd4, %rd15;
    mov.f32 %f1, 0f00000000;
    st.global.f32 [%rd16], %f1;
DONE_MEMORY_WEIGHTED_VALUE_BACK_WEIGHTS:
    ret;
}

.visible .entry heirloom_memory_weighted_value_backward_values_f32_i64(
    .param .u64 indices,
    .param .u64 weights,
    .param .u64 grad_output,
    .param .u64 grad_values,
    .param .u64 status,
    .param .u32 tokens,
    .param .u32 top_k,
    .param .u32 slots,
    .param .u32 value_dim,
    .param .u32 total
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<26>;
    .reg .b64 %rd<24>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [indices];
    ld.param.u64 %rd2, [weights];
    ld.param.u64 %rd3, [grad_output];
    ld.param.u64 %rd4, [grad_values];
    ld.param.u64 %rd5, [status];
    ld.param.u32 %r4, [tokens];
    ld.param.u32 %r5, [top_k];
    ld.param.u32 %r6, [slots];
    ld.param.u32 %r7, [value_dim];
    ld.param.u32 %r8, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r9, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r9, %r8;
    @%p1 bra DONE_MEMORY_WEIGHTED_VALUE_BACK_VALUES;

    rem.u32 %r10, %r9, %r7;
    div.u32 %r11, %r9, %r7;
    rem.u32 %r12, %r11, %r5;
    div.u32 %r13, %r11, %r5;
    mad.lo.s32 %r14, %r13, %r5, %r12;
    mul.wide.u32 %rd6, %r14, 8;
    add.s64 %rd7, %rd1, %rd6;
    ld.global.s64 %rd8, [%rd7];
    mov.s64 %rd9, 0;
    setp.lt.s64 %p2, %rd8, %rd9;
    cvt.s64.u32 %rd10, %r6;
    setp.ge.s64 %p3, %rd8, %rd10;
    or.pred %p4, %p2, %p3;
    @%p4 bra MEMORY_WEIGHTED_VALUE_BACK_VALUES_BAD_INDEX;
    cvt.u32.u64 %r15, %rd8;
    mul.wide.u32 %rd11, %r14, 4;
    add.s64 %rd12, %rd2, %rd11;
    ld.global.f32 %f1, [%rd12];
    mad.lo.s32 %r16, %r13, %r7, %r10;
    mul.wide.u32 %rd13, %r16, 4;
    add.s64 %rd14, %rd3, %rd13;
    ld.global.f32 %f2, [%rd14];
    mul.f32 %f3, %f1, %f2;
    mad.lo.s32 %r17, %r15, %r7, %r10;
    mul.wide.u32 %rd15, %r17, 4;
    add.s64 %rd16, %rd4, %rd15;
    atom.global.add.f32 %f4, [%rd16], %f3;
    bra DONE_MEMORY_WEIGHTED_VALUE_BACK_VALUES;

MEMORY_WEIGHTED_VALUE_BACK_VALUES_BAD_INDEX:
    mov.u32 %r18, 1;
    atom.global.exch.b32 %r19, [%rd5], %r18;
DONE_MEMORY_WEIGHTED_VALUE_BACK_VALUES:
    ret;
}

.visible .entry heirloom_memory_selected_scores_forward_f32_i64(
    .param .u64 indices,
    .param .u64 query,
    .param .u64 keys,
    .param .u64 output,
    .param .u64 status,
    .param .u32 tokens,
    .param .u32 top_k,
    .param .u32 slots,
    .param .u32 key_dim,
    .param .u32 total
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<24>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [indices];
    ld.param.u64 %rd2, [query];
    ld.param.u64 %rd3, [keys];
    ld.param.u64 %rd4, [output];
    ld.param.u64 %rd5, [status];
    ld.param.u32 %r4, [tokens];
    ld.param.u32 %r5, [top_k];
    ld.param.u32 %r6, [slots];
    ld.param.u32 %r7, [key_dim];
    ld.param.u32 %r8, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r9, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r9, %r8;
    @%p1 bra DONE_MEMORY_SELECTED_SCORES_FWD;

    div.u32 %r10, %r9, %r5;
    rem.u32 %r11, %r9, %r5;
    mul.wide.u32 %rd6, %r9, 8;
    add.s64 %rd7, %rd1, %rd6;
    ld.global.s64 %rd8, [%rd7];
    mov.s64 %rd9, 0;
    setp.lt.s64 %p2, %rd8, %rd9;
    cvt.s64.u32 %rd10, %r6;
    setp.ge.s64 %p3, %rd8, %rd10;
    or.pred %p4, %p2, %p3;
    @%p4 bra MEMORY_SELECTED_SCORES_FWD_BAD_INDEX;
    cvt.u32.u64 %r12, %rd8;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r13, 0;
MEMORY_SELECTED_SCORES_FWD_LOOP:
    setp.ge.u32 %p2, %r13, %r7;
    @%p2 bra MEMORY_SELECTED_SCORES_FWD_WRITE;
    mad.lo.s32 %r14, %r10, %r7, %r13;
    mul.wide.u32 %rd11, %r14, 4;
    add.s64 %rd12, %rd2, %rd11;
    ld.global.f32 %f2, [%rd12];
    mad.lo.s32 %r15, %r12, %r7, %r13;
    mul.wide.u32 %rd13, %r15, 4;
    add.s64 %rd14, %rd3, %rd13;
    ld.global.f32 %f3, [%rd14];
    mul.f32 %f4, %f2, %f3;
    add.f32 %f1, %f1, %f4;
    add.u32 %r13, %r13, 1;
    bra MEMORY_SELECTED_SCORES_FWD_LOOP;

MEMORY_SELECTED_SCORES_FWD_WRITE:
    mul.wide.u32 %rd15, %r9, 4;
    add.s64 %rd16, %rd4, %rd15;
    st.global.f32 [%rd16], %f1;
    bra DONE_MEMORY_SELECTED_SCORES_FWD;

MEMORY_SELECTED_SCORES_FWD_BAD_INDEX:
    mov.u32 %r16, 1;
    atom.global.exch.b32 %r17, [%rd5], %r16;
    mul.wide.u32 %rd15, %r9, 4;
    add.s64 %rd16, %rd4, %rd15;
    mov.f32 %f1, 0f00000000;
    st.global.f32 [%rd16], %f1;
DONE_MEMORY_SELECTED_SCORES_FWD:
    ret;
}

.visible .entry heirloom_memory_selected_scores_backward_query_f32_i64(
    .param .u64 indices,
    .param .u64 keys,
    .param .u64 grad_scores,
    .param .u64 grad_query,
    .param .u64 status,
    .param .u32 tokens,
    .param .u32 top_k,
    .param .u32 slots,
    .param .u32 key_dim,
    .param .u32 total
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<26>;
    .reg .b64 %rd<24>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [indices];
    ld.param.u64 %rd2, [keys];
    ld.param.u64 %rd3, [grad_scores];
    ld.param.u64 %rd4, [grad_query];
    ld.param.u64 %rd5, [status];
    ld.param.u32 %r4, [tokens];
    ld.param.u32 %r5, [top_k];
    ld.param.u32 %r6, [slots];
    ld.param.u32 %r7, [key_dim];
    ld.param.u32 %r8, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r9, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r9, %r8;
    @%p1 bra DONE_MEMORY_SELECTED_SCORES_BACK_QUERY;

    div.u32 %r10, %r9, %r7;
    rem.u32 %r11, %r9, %r7;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r12, 0;
MEMORY_SELECTED_SCORES_BACK_QUERY_LOOP:
    setp.ge.u32 %p2, %r12, %r5;
    @%p2 bra MEMORY_SELECTED_SCORES_BACK_QUERY_WRITE;
    mad.lo.s32 %r13, %r10, %r5, %r12;
    mul.wide.u32 %rd6, %r13, 8;
    add.s64 %rd7, %rd1, %rd6;
    ld.global.s64 %rd8, [%rd7];
    mov.s64 %rd9, 0;
    setp.lt.s64 %p3, %rd8, %rd9;
    cvt.s64.u32 %rd10, %r6;
    setp.ge.s64 %p4, %rd8, %rd10;
    or.pred %p3, %p3, %p4;
    @%p3 bra MEMORY_SELECTED_SCORES_BACK_QUERY_BAD_INDEX;
    cvt.u32.u64 %r14, %rd8;
    mul.wide.u32 %rd11, %r13, 4;
    add.s64 %rd12, %rd3, %rd11;
    ld.global.f32 %f2, [%rd12];
    mad.lo.s32 %r15, %r14, %r7, %r11;
    mul.wide.u32 %rd13, %r15, 4;
    add.s64 %rd14, %rd2, %rd13;
    ld.global.f32 %f3, [%rd14];
    mul.f32 %f4, %f2, %f3;
    add.f32 %f1, %f1, %f4;
    add.u32 %r12, %r12, 1;
    bra MEMORY_SELECTED_SCORES_BACK_QUERY_LOOP;

MEMORY_SELECTED_SCORES_BACK_QUERY_BAD_INDEX:
    mov.u32 %r16, 1;
    atom.global.exch.b32 %r17, [%rd5], %r16;
    add.u32 %r12, %r12, 1;
    bra MEMORY_SELECTED_SCORES_BACK_QUERY_LOOP;

MEMORY_SELECTED_SCORES_BACK_QUERY_WRITE:
    mul.wide.u32 %rd15, %r9, 4;
    add.s64 %rd16, %rd4, %rd15;
    st.global.f32 [%rd16], %f1;
DONE_MEMORY_SELECTED_SCORES_BACK_QUERY:
    ret;
}

.visible .entry heirloom_memory_selected_scores_backward_keys_f32_i64(
    .param .u64 indices,
    .param .u64 query,
    .param .u64 grad_scores,
    .param .u64 grad_keys,
    .param .u64 status,
    .param .u32 tokens,
    .param .u32 top_k,
    .param .u32 slots,
    .param .u32 key_dim,
    .param .u32 total
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<28>;
    .reg .b64 %rd<24>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [indices];
    ld.param.u64 %rd2, [query];
    ld.param.u64 %rd3, [grad_scores];
    ld.param.u64 %rd4, [grad_keys];
    ld.param.u64 %rd5, [status];
    ld.param.u32 %r4, [tokens];
    ld.param.u32 %r5, [top_k];
    ld.param.u32 %r6, [slots];
    ld.param.u32 %r7, [key_dim];
    ld.param.u32 %r8, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r9, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r9, %r8;
    @%p1 bra DONE_MEMORY_SELECTED_SCORES_BACK_KEYS;

    rem.u32 %r10, %r9, %r7;
    div.u32 %r11, %r9, %r7;
    rem.u32 %r12, %r11, %r5;
    div.u32 %r13, %r11, %r5;
    mad.lo.s32 %r14, %r13, %r5, %r12;
    mul.wide.u32 %rd6, %r14, 8;
    add.s64 %rd7, %rd1, %rd6;
    ld.global.s64 %rd8, [%rd7];
    mov.s64 %rd9, 0;
    setp.lt.s64 %p2, %rd8, %rd9;
    cvt.s64.u32 %rd10, %r6;
    setp.ge.s64 %p3, %rd8, %rd10;
    or.pred %p4, %p2, %p3;
    @%p4 bra MEMORY_SELECTED_SCORES_BACK_KEYS_BAD_INDEX;
    cvt.u32.u64 %r15, %rd8;
    mul.wide.u32 %rd11, %r14, 4;
    add.s64 %rd12, %rd3, %rd11;
    ld.global.f32 %f1, [%rd12];
    mad.lo.s32 %r16, %r13, %r7, %r10;
    mul.wide.u32 %rd13, %r16, 4;
    add.s64 %rd14, %rd2, %rd13;
    ld.global.f32 %f2, [%rd14];
    mul.f32 %f3, %f1, %f2;
    mad.lo.s32 %r17, %r15, %r7, %r10;
    mul.wide.u32 %rd15, %r17, 4;
    add.s64 %rd16, %rd4, %rd15;
    atom.global.add.f32 %f4, [%rd16], %f3;
    bra DONE_MEMORY_SELECTED_SCORES_BACK_KEYS;

MEMORY_SELECTED_SCORES_BACK_KEYS_BAD_INDEX:
    mov.u32 %r18, 1;
    atom.global.exch.b32 %r19, [%rd5], %r18;
DONE_MEMORY_SELECTED_SCORES_BACK_KEYS:
    ret;
}

.visible .entry heirloom_layer_norm_forward_f32(
    .param .u64 input,
    .param .u64 weight,
    .param .u64 bias,
    .param .u64 output,
    .param .u32 rows,
    .param .u32 features,
    .param .f32 eps,
    .param .u32 total
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<32>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [weight];
    ld.param.u64 %rd3, [bias];
    ld.param.u64 %rd4, [output];
    ld.param.u32 %r4, [rows];
    ld.param.u32 %r5, [features];
    ld.param.f32 %f10, [eps];
    ld.param.u32 %r6, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r8, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r8, %r6;
    @%p1 bra DONE_LN_FWD;

    div.u32 %r9, %r8, %r5;
    rem.u32 %r10, %r8, %r5;
    cvt.rn.f32.u32 %f3, %r5;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r11, 0;
LN_FWD_MEAN_LOOP:
    setp.ge.u32 %p2, %r11, %r5;
    @%p2 bra LN_FWD_MEAN_DONE;
    mad.lo.s32 %r12, %r9, %r5, %r11;
    mul.wide.u32 %rd5, %r12, 4;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.f32 %f2, [%rd6];
    add.f32 %f1, %f1, %f2;
    add.u32 %r11, %r11, 1;
    bra LN_FWD_MEAN_LOOP;
LN_FWD_MEAN_DONE:
    div.rn.f32 %f4, %f1, %f3;
    mov.f32 %f5, 0f00000000;
    mov.u32 %r11, 0;
LN_FWD_VAR_LOOP:
    setp.ge.u32 %p3, %r11, %r5;
    @%p3 bra LN_FWD_VAR_DONE;
    mad.lo.s32 %r12, %r9, %r5, %r11;
    mul.wide.u32 %rd5, %r12, 4;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.f32 %f2, [%rd6];
    sub.f32 %f6, %f2, %f4;
    mul.f32 %f7, %f6, %f6;
    add.f32 %f5, %f5, %f7;
    add.u32 %r11, %r11, 1;
    bra LN_FWD_VAR_LOOP;
LN_FWD_VAR_DONE:
    div.rn.f32 %f5, %f5, %f3;
    add.f32 %f8, %f5, %f10;
    sqrt.rn.f32 %f9, %f8;
    mov.f32 %f11, 0f3f800000;
    div.rn.f32 %f12, %f11, %f9;
    mad.lo.s32 %r13, %r9, %r5, %r10;
    mul.wide.u32 %rd7, %r13, 4;
    add.s64 %rd8, %rd1, %rd7;
    ld.global.f32 %f13, [%rd8];
    sub.f32 %f14, %f13, %f4;
    mul.f32 %f15, %f14, %f12;
    mul.wide.u32 %rd9, %r10, 4;
    add.s64 %rd10, %rd2, %rd9;
    add.s64 %rd11, %rd3, %rd9;
    ld.global.f32 %f16, [%rd10];
    ld.global.f32 %f17, [%rd11];
    mul.f32 %f18, %f15, %f16;
    add.f32 %f19, %f18, %f17;
    add.s64 %rd12, %rd4, %rd7;
    st.global.f32 [%rd12], %f19;
DONE_LN_FWD:
    ret;
}

.visible .entry heirloom_layer_norm_backward_input_f32(
    .param .u64 input,
    .param .u64 weight,
    .param .u64 grad_output,
    .param .u64 output,
    .param .u32 rows,
    .param .u32 features,
    .param .f32 eps,
    .param .u32 total
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<32>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [weight];
    ld.param.u64 %rd3, [grad_output];
    ld.param.u64 %rd4, [output];
    ld.param.u32 %r4, [rows];
    ld.param.u32 %r5, [features];
    ld.param.f32 %f10, [eps];
    ld.param.u32 %r6, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r8, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r8, %r6;
    @%p1 bra DONE_LN_BWD_INPUT;

    div.u32 %r9, %r8, %r5;
    rem.u32 %r10, %r8, %r5;
    cvt.rn.f32.u32 %f3, %r5;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r11, 0;
LN_BWD_INPUT_MEAN_LOOP:
    setp.ge.u32 %p2, %r11, %r5;
    @%p2 bra LN_BWD_INPUT_MEAN_DONE;
    mad.lo.s32 %r12, %r9, %r5, %r11;
    mul.wide.u32 %rd5, %r12, 4;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.f32 %f2, [%rd6];
    add.f32 %f1, %f1, %f2;
    add.u32 %r11, %r11, 1;
    bra LN_BWD_INPUT_MEAN_LOOP;
LN_BWD_INPUT_MEAN_DONE:
    div.rn.f32 %f4, %f1, %f3;
    mov.f32 %f5, 0f00000000;
    mov.u32 %r11, 0;
LN_BWD_INPUT_VAR_LOOP:
    setp.ge.u32 %p3, %r11, %r5;
    @%p3 bra LN_BWD_INPUT_VAR_DONE;
    mad.lo.s32 %r12, %r9, %r5, %r11;
    mul.wide.u32 %rd5, %r12, 4;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.f32 %f2, [%rd6];
    sub.f32 %f6, %f2, %f4;
    mul.f32 %f7, %f6, %f6;
    add.f32 %f5, %f5, %f7;
    add.u32 %r11, %r11, 1;
    bra LN_BWD_INPUT_VAR_LOOP;
LN_BWD_INPUT_VAR_DONE:
    div.rn.f32 %f5, %f5, %f3;
    add.f32 %f8, %f5, %f10;
    sqrt.rn.f32 %f9, %f8;
    mov.f32 %f11, 0f3f800000;
    div.rn.f32 %f12, %f11, %f9;
    mov.f32 %f13, 0f00000000;
    mov.f32 %f14, 0f00000000;
    mov.u32 %r11, 0;
LN_BWD_INPUT_SUM_LOOP:
    setp.ge.u32 %p4, %r11, %r5;
    @%p4 bra LN_BWD_INPUT_SUM_DONE;
    mad.lo.s32 %r12, %r9, %r5, %r11;
    mul.wide.u32 %rd5, %r12, 4;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd7, %rd3, %rd5;
    mul.wide.u32 %rd8, %r11, 4;
    add.s64 %rd9, %rd2, %rd8;
    ld.global.f32 %f15, [%rd6];
    ld.global.f32 %f16, [%rd7];
    ld.global.f32 %f17, [%rd9];
    sub.f32 %f18, %f15, %f4;
    mul.f32 %f19, %f18, %f12;
    mul.f32 %f20, %f16, %f17;
    add.f32 %f13, %f13, %f20;
    mul.f32 %f21, %f20, %f19;
    add.f32 %f14, %f14, %f21;
    add.u32 %r11, %r11, 1;
    bra LN_BWD_INPUT_SUM_LOOP;
LN_BWD_INPUT_SUM_DONE:
    mad.lo.s32 %r13, %r9, %r5, %r10;
    mul.wide.u32 %rd10, %r13, 4;
    add.s64 %rd11, %rd1, %rd10;
    add.s64 %rd12, %rd3, %rd10;
    mul.wide.u32 %rd13, %r10, 4;
    add.s64 %rd14, %rd2, %rd13;
    ld.global.f32 %f15, [%rd11];
    ld.global.f32 %f16, [%rd12];
    ld.global.f32 %f17, [%rd14];
    sub.f32 %f18, %f15, %f4;
    mul.f32 %f19, %f18, %f12;
    mul.f32 %f20, %f16, %f17;
    mul.f32 %f21, %f3, %f20;
    sub.f32 %f22, %f21, %f13;
    mul.f32 %f23, %f19, %f14;
    sub.f32 %f24, %f22, %f23;
    mul.f32 %f25, %f12, %f24;
    div.rn.f32 %f26, %f25, %f3;
    add.s64 %rd15, %rd4, %rd10;
    st.global.f32 [%rd15], %f26;
DONE_LN_BWD_INPUT:
    ret;
}

.visible .entry heirloom_layer_norm_backward_weight_bias_f32(
    .param .u64 input,
    .param .u64 grad_output,
    .param .u64 grad_weight,
    .param .u64 grad_bias,
    .param .u32 rows,
    .param .u32 features,
    .param .f32 eps,
    .param .u32 total
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<32>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [grad_output];
    ld.param.u64 %rd3, [grad_weight];
    ld.param.u64 %rd4, [grad_bias];
    ld.param.u32 %r4, [rows];
    ld.param.u32 %r5, [features];
    ld.param.f32 %f10, [eps];
    ld.param.u32 %r6, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r8, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r8, %r6;
    @%p1 bra DONE_LN_BWD_WB;

    cvt.rn.f32.u32 %f3, %r5;
    mov.f32 %f27, 0f00000000;
    mov.f32 %f28, 0f00000000;
    mov.u32 %r9, 0;
LN_BWD_WB_ROW_LOOP:
    setp.ge.u32 %p2, %r9, %r4;
    @%p2 bra LN_BWD_WB_WRITE;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r10, 0;
LN_BWD_WB_MEAN_LOOP:
    setp.ge.u32 %p3, %r10, %r5;
    @%p3 bra LN_BWD_WB_MEAN_DONE;
    mad.lo.s32 %r11, %r9, %r5, %r10;
    mul.wide.u32 %rd5, %r11, 4;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.f32 %f2, [%rd6];
    add.f32 %f1, %f1, %f2;
    add.u32 %r10, %r10, 1;
    bra LN_BWD_WB_MEAN_LOOP;
LN_BWD_WB_MEAN_DONE:
    div.rn.f32 %f4, %f1, %f3;
    mov.f32 %f5, 0f00000000;
    mov.u32 %r10, 0;
LN_BWD_WB_VAR_LOOP:
    setp.ge.u32 %p4, %r10, %r5;
    @%p4 bra LN_BWD_WB_VAR_DONE;
    mad.lo.s32 %r11, %r9, %r5, %r10;
    mul.wide.u32 %rd5, %r11, 4;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.f32 %f2, [%rd6];
    sub.f32 %f6, %f2, %f4;
    mul.f32 %f7, %f6, %f6;
    add.f32 %f5, %f5, %f7;
    add.u32 %r10, %r10, 1;
    bra LN_BWD_WB_VAR_LOOP;
LN_BWD_WB_VAR_DONE:
    div.rn.f32 %f5, %f5, %f3;
    add.f32 %f8, %f5, %f10;
    sqrt.rn.f32 %f9, %f8;
    mov.f32 %f11, 0f3f800000;
    div.rn.f32 %f12, %f11, %f9;
    mad.lo.s32 %r12, %r9, %r5, %r8;
    mul.wide.u32 %rd7, %r12, 4;
    add.s64 %rd8, %rd1, %rd7;
    add.s64 %rd9, %rd2, %rd7;
    ld.global.f32 %f13, [%rd8];
    ld.global.f32 %f14, [%rd9];
    sub.f32 %f15, %f13, %f4;
    mul.f32 %f16, %f15, %f12;
    mul.f32 %f17, %f14, %f16;
    add.f32 %f27, %f27, %f17;
    add.f32 %f28, %f28, %f14;
    add.u32 %r9, %r9, 1;
    bra LN_BWD_WB_ROW_LOOP;
LN_BWD_WB_WRITE:
    mul.wide.u32 %rd10, %r8, 4;
    add.s64 %rd11, %rd3, %rd10;
    add.s64 %rd12, %rd4, %rd10;
    st.global.f32 [%rd11], %f27;
    st.global.f32 [%rd12], %f28;
DONE_LN_BWD_WB:
    ret;
}

.visible .entry heirloom_cross_entropy_forward_f32_i64(
    .param .u64 logits,
    .param .u64 targets,
    .param .u64 output,
    .param .u64 status,
    .param .u32 batch,
    .param .u32 classes
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<32>;

    ld.param.u64 %rd1, [logits];
    ld.param.u64 %rd2, [targets];
    ld.param.u64 %rd3, [output];
    ld.param.u64 %rd4, [status];
    ld.param.u32 %r4, [batch];
    ld.param.u32 %r5, [classes];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r6, %r2, %r3, %r1;
    setp.ne.u32 %p1, %r6, 0;
    @%p1 bra DONE_CE_FWD;

    mov.f32 %f20, 0f3fb8aa3b;
    mov.f32 %f21, 0f3f317218;
    mov.f32 %f22, 0f3f800000;
    mov.f32 %f23, 0f00000000;
    mov.u32 %r7, 0;
CE_FWD_ROW_LOOP:
    setp.ge.u32 %p2, %r7, %r4;
    @%p2 bra CE_FWD_WRITE;
    mul.wide.u32 %rd5, %r7, 8;
    add.s64 %rd6, %rd2, %rd5;
    ld.global.s64 %rd7, [%rd6];
    mov.s64 %rd8, 0;
    setp.lt.s64 %p3, %rd7, %rd8;
    cvt.s64.u32 %rd9, %r5;
    setp.ge.s64 %p4, %rd7, %rd9;
    or.pred %p5, %p3, %p4;
    @%p5 bra CE_FWD_BAD_INDEX;
    cvt.u32.u64 %r8, %rd7;
    mul.lo.u32 %r9, %r7, %r5;
    mul.wide.u32 %rd10, %r9, 4;
    add.s64 %rd11, %rd1, %rd10;
    ld.global.f32 %f1, [%rd11];
    mov.u32 %r10, 1;
CE_FWD_MAX_LOOP:
    setp.ge.u32 %p6, %r10, %r5;
    @%p6 bra CE_FWD_MAX_DONE;
    add.u32 %r11, %r9, %r10;
    mul.wide.u32 %rd12, %r11, 4;
    add.s64 %rd13, %rd1, %rd12;
    ld.global.f32 %f2, [%rd13];
    max.f32 %f1, %f1, %f2;
    add.u32 %r10, %r10, 1;
    bra CE_FWD_MAX_LOOP;
CE_FWD_MAX_DONE:
    mov.f32 %f3, 0f00000000;
    mov.u32 %r10, 0;
CE_FWD_SUM_LOOP:
    setp.ge.u32 %p6, %r10, %r5;
    @%p6 bra CE_FWD_SUM_DONE;
    add.u32 %r11, %r9, %r10;
    mul.wide.u32 %rd12, %r11, 4;
    add.s64 %rd13, %rd1, %rd12;
    ld.global.f32 %f2, [%rd13];
    sub.f32 %f4, %f2, %f1;
    mul.f32 %f5, %f4, %f20;
    ex2.approx.ftz.f32 %f6, %f5;
    add.f32 %f3, %f3, %f6;
    add.u32 %r10, %r10, 1;
    bra CE_FWD_SUM_LOOP;
CE_FWD_SUM_DONE:
    add.u32 %r12, %r9, %r8;
    mul.wide.u32 %rd14, %r12, 4;
    add.s64 %rd15, %rd1, %rd14;
    ld.global.f32 %f7, [%rd15];
    lg2.approx.ftz.f32 %f8, %f3;
    mul.f32 %f9, %f8, %f21;
    sub.f32 %f10, %f1, %f7;
    add.f32 %f11, %f10, %f9;
    add.f32 %f23, %f23, %f11;
    add.u32 %r7, %r7, 1;
    bra CE_FWD_ROW_LOOP;
CE_FWD_WRITE:
    cvt.rn.f32.u32 %f12, %r4;
    div.rn.f32 %f13, %f23, %f12;
    st.global.f32 [%rd3], %f13;
    bra DONE_CE_FWD;
CE_FWD_BAD_INDEX:
    mov.u32 %r13, 1;
    atom.global.exch.b32 %r14, [%rd4], %r13;
DONE_CE_FWD:
    ret;
}

.visible .entry heirloom_cross_entropy_backward_f32_i64(
    .param .u64 logits,
    .param .u64 targets,
    .param .u64 grad_output,
    .param .u64 output,
    .param .u64 status,
    .param .u32 batch,
    .param .u32 classes,
    .param .u32 total
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<32>;

    ld.param.u64 %rd1, [logits];
    ld.param.u64 %rd2, [targets];
    ld.param.u64 %rd3, [grad_output];
    ld.param.u64 %rd4, [output];
    ld.param.u64 %rd5, [status];
    ld.param.u32 %r4, [batch];
    ld.param.u32 %r5, [classes];
    ld.param.u32 %r6, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r8, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r8, %r6;
    @%p1 bra DONE_CE_BWD;

    div.u32 %r9, %r8, %r5;
    rem.u32 %r10, %r8, %r5;
    mul.wide.u32 %rd6, %r9, 8;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.s64 %rd8, [%rd7];
    mov.s64 %rd9, 0;
    setp.lt.s64 %p2, %rd8, %rd9;
    cvt.s64.u32 %rd10, %r5;
    setp.ge.s64 %p3, %rd8, %rd10;
    or.pred %p4, %p2, %p3;
    @%p4 bra CE_BWD_BAD_INDEX;
    cvt.u32.u64 %r11, %rd8;

    mul.lo.u32 %r12, %r9, %r5;
    mul.wide.u32 %rd11, %r12, 4;
    add.s64 %rd12, %rd1, %rd11;
    ld.global.f32 %f1, [%rd12];
    mov.u32 %r13, 1;
CE_BWD_MAX_LOOP:
    setp.ge.u32 %p5, %r13, %r5;
    @%p5 bra CE_BWD_MAX_DONE;
    add.u32 %r14, %r12, %r13;
    mul.wide.u32 %rd13, %r14, 4;
    add.s64 %rd14, %rd1, %rd13;
    ld.global.f32 %f2, [%rd14];
    max.f32 %f1, %f1, %f2;
    add.u32 %r13, %r13, 1;
    bra CE_BWD_MAX_LOOP;
CE_BWD_MAX_DONE:
    mov.f32 %f20, 0f3fb8aa3b;
    mov.f32 %f21, 0f3f800000;
    mov.f32 %f22, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.u32 %r13, 0;
CE_BWD_SUM_LOOP:
    setp.ge.u32 %p5, %r13, %r5;
    @%p5 bra CE_BWD_SUM_DONE;
    add.u32 %r14, %r12, %r13;
    mul.wide.u32 %rd13, %r14, 4;
    add.s64 %rd14, %rd1, %rd13;
    ld.global.f32 %f2, [%rd14];
    sub.f32 %f4, %f2, %f1;
    mul.f32 %f5, %f4, %f20;
    ex2.approx.ftz.f32 %f6, %f5;
    add.f32 %f3, %f3, %f6;
    add.u32 %r13, %r13, 1;
    bra CE_BWD_SUM_LOOP;
CE_BWD_SUM_DONE:
    mul.wide.u32 %rd15, %r8, 4;
    add.s64 %rd16, %rd1, %rd15;
    ld.global.f32 %f7, [%rd16];
    sub.f32 %f8, %f7, %f1;
    mul.f32 %f9, %f8, %f20;
    ex2.approx.ftz.f32 %f10, %f9;
    div.rn.f32 %f11, %f10, %f3;
    setp.eq.u32 %p6, %r10, %r11;
    @%p6 sub.f32 %f11, %f11, %f21;
    ld.global.f32 %f12, [%rd3];
    cvt.rn.f32.u32 %f13, %r4;
    div.rn.f32 %f14, %f12, %f13;
    mul.f32 %f15, %f11, %f14;
    add.s64 %rd17, %rd4, %rd15;
    st.global.f32 [%rd17], %f15;
    bra DONE_CE_BWD;
CE_BWD_BAD_INDEX:
    mov.u32 %r15, 1;
    atom.global.exch.b32 %r16, [%rd5], %r15;
DONE_CE_BWD:
    ret;
}

.visible .entry heirloom_causal_attention_softmax_scores_f32(
    .param .u64 scores,
    .param .u64 attention,
    .param .u32 batch,
    .param .u32 time,
    .param .u32 channels,
    .param .u32 n_heads,
    .param .u32 head_dim,
    .param .u32 total
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<32>;

    ld.param.u64 %rd1, [scores];
    ld.param.u64 %rd2, [attention];
    ld.param.u32 %r4, [batch];
    ld.param.u32 %r5, [time];
    ld.param.u32 %r6, [channels];
    ld.param.u32 %r7, [n_heads];
    ld.param.u32 %r8, [head_dim];
    ld.param.u32 %r9, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r10, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r10, %r9;
    @%p1 bra DONE_CA_SOFTMAX_SCORES;

    rem.u32 %r11, %r10, %r5;
    div.u32 %r12, %r10, %r5;
    rem.u32 %r13, %r12, %r5;
    div.u32 %r14, %r12, %r5;
    rem.u32 %r15, %r14, %r7;
    div.u32 %r16, %r14, %r7;
    mul.wide.u32 %rd3, %r10, 4;
    add.s64 %rd4, %rd2, %rd3;
    setp.gt.u32 %p2, %r11, %r13;
    @%p2 bra CA_SOFTMAX_SCORES_MASKED;

    cvt.rn.f32.u32 %f1, %r8;
    sqrt.rn.f32 %f2, %f1;
    mov.f32 %f3, 0f3f800000;
    div.rn.f32 %f4, %f3, %f2;
    mul.lo.u32 %r17, %r16, %r7;
    add.u32 %r17, %r17, %r15;
    mul.lo.u32 %r17, %r17, %r5;
    add.u32 %r17, %r17, %r13;
    mul.lo.u32 %r17, %r17, %r5;

    mov.f32 %f5, 0fff800000;
    mov.u32 %r18, 0;
CA_SOFTMAX_SCORES_MAX_LOOP:
    setp.gt.u32 %p3, %r18, %r13;
    @%p3 bra CA_SOFTMAX_SCORES_MAX_DONE;
    add.u32 %r19, %r17, %r18;
    mul.wide.u32 %rd5, %r19, 4;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.f32 %f6, [%rd6];
    mul.f32 %f6, %f6, %f4;
    max.f32 %f5, %f5, %f6;
    add.u32 %r18, %r18, 1;
    bra CA_SOFTMAX_SCORES_MAX_LOOP;

CA_SOFTMAX_SCORES_MAX_DONE:
    mov.f32 %f10, 0f3fb8aa3b;
    mov.f32 %f11, 0f00000000;
    mov.f32 %f12, 0f00000000;
    mov.u32 %r18, 0;
CA_SOFTMAX_SCORES_SUM_LOOP:
    setp.gt.u32 %p3, %r18, %r13;
    @%p3 bra CA_SOFTMAX_SCORES_SUM_DONE;
    add.u32 %r19, %r17, %r18;
    mul.wide.u32 %rd5, %r19, 4;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.f32 %f6, [%rd6];
    mul.f32 %f6, %f6, %f4;
    sub.f32 %f7, %f6, %f5;
    mul.f32 %f8, %f7, %f10;
    ex2.approx.ftz.f32 %f9, %f8;
    add.f32 %f11, %f11, %f9;
    setp.eq.u32 %p4, %r18, %r11;
    @%p4 mov.f32 %f12, %f9;
    add.u32 %r18, %r18, 1;
    bra CA_SOFTMAX_SCORES_SUM_LOOP;

CA_SOFTMAX_SCORES_SUM_DONE:
    div.rn.f32 %f13, %f12, %f11;
    st.global.f32 [%rd4], %f13;
    bra DONE_CA_SOFTMAX_SCORES;
CA_SOFTMAX_SCORES_MASKED:
    mov.f32 %f14, 0f00000000;
    st.global.f32 [%rd4], %f14;
DONE_CA_SOFTMAX_SCORES:
    ret;
}

.visible .entry heirloom_causal_attention_softmax_backward_scores_f32(
    .param .u64 attention,
    .param .u64 score_grad,
    .param .u64 dscore,
    .param .u32 batch,
    .param .u32 time,
    .param .u32 channels,
    .param .u32 n_heads,
    .param .u32 head_dim,
    .param .u32 total
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<32>;

    ld.param.u64 %rd1, [attention];
    ld.param.u64 %rd2, [score_grad];
    ld.param.u64 %rd3, [dscore];
    ld.param.u32 %r4, [batch];
    ld.param.u32 %r5, [time];
    ld.param.u32 %r6, [channels];
    ld.param.u32 %r7, [n_heads];
    ld.param.u32 %r8, [head_dim];
    ld.param.u32 %r9, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r10, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r10, %r9;
    @%p1 bra DONE_CA_SOFTMAX_BWD_SCORES;

    rem.u32 %r11, %r10, %r5;
    div.u32 %r12, %r10, %r5;
    rem.u32 %r13, %r12, %r5;
    div.u32 %r14, %r12, %r5;
    rem.u32 %r15, %r14, %r7;
    div.u32 %r16, %r14, %r7;
    mul.wide.u32 %rd4, %r10, 4;
    add.s64 %rd5, %rd3, %rd4;
    setp.gt.u32 %p2, %r11, %r13;
    @%p2 bra CA_SOFTMAX_BWD_SCORES_MASKED;

    mul.lo.u32 %r17, %r16, %r7;
    add.u32 %r17, %r17, %r15;
    mul.lo.u32 %r17, %r17, %r5;
    add.u32 %r17, %r17, %r13;
    mul.lo.u32 %r17, %r17, %r5;

    mov.f32 %f1, 0f00000000;
    mov.u32 %r18, 0;
CA_SOFTMAX_BWD_SCORES_DOT_LOOP:
    setp.gt.u32 %p3, %r18, %r13;
    @%p3 bra CA_SOFTMAX_BWD_SCORES_DOT_DONE;
    add.u32 %r19, %r17, %r18;
    mul.wide.u32 %rd6, %r19, 4;
    add.s64 %rd7, %rd1, %rd6;
    add.s64 %rd8, %rd2, %rd6;
    ld.global.f32 %f2, [%rd7];
    ld.global.f32 %f3, [%rd8];
    mul.f32 %f4, %f2, %f3;
    add.f32 %f1, %f1, %f4;
    add.u32 %r18, %r18, 1;
    bra CA_SOFTMAX_BWD_SCORES_DOT_LOOP;

CA_SOFTMAX_BWD_SCORES_DOT_DONE:
    add.s64 %rd9, %rd1, %rd4;
    add.s64 %rd10, %rd2, %rd4;
    ld.global.f32 %f5, [%rd9];
    ld.global.f32 %f6, [%rd10];
    sub.f32 %f7, %f6, %f1;
    mul.f32 %f8, %f5, %f7;
    cvt.rn.f32.u32 %f9, %r8;
    sqrt.rn.f32 %f10, %f9;
    mov.f32 %f11, 0f3f800000;
    div.rn.f32 %f12, %f11, %f10;
    mul.f32 %f13, %f8, %f12;
    st.global.f32 [%rd5], %f13;
    bra DONE_CA_SOFTMAX_BWD_SCORES;

CA_SOFTMAX_BWD_SCORES_MASKED:
    mov.f32 %f14, 0f00000000;
    st.global.f32 [%rd5], %f14;
DONE_CA_SOFTMAX_BWD_SCORES:
    ret;
}

.visible .entry heirloom_attention_square_transpose_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 batch,
    .param .u32 time,
    .param .u32 channels,
    .param .u32 n_heads,
    .param .u32 head_dim,
    .param .u32 total
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<2>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r4, [batch];
    ld.param.u32 %r5, [time];
    ld.param.u32 %r6, [channels];
    ld.param.u32 %r7, [n_heads];
    ld.param.u32 %r8, [head_dim];
    ld.param.u32 %r9, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r10, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r10, %r9;
    @%p1 bra DONE_ATTENTION_SQUARE_TRANSPOSE;

    rem.u32 %r11, %r10, %r5;
    div.u32 %r12, %r10, %r5;
    rem.u32 %r13, %r12, %r5;
    div.u32 %r14, %r12, %r5;
    rem.u32 %r15, %r14, %r7;
    div.u32 %r16, %r14, %r7;

    mul.lo.u32 %r17, %r16, %r7;
    add.u32 %r17, %r17, %r15;
    mul.lo.u32 %r17, %r17, %r5;
    add.u32 %r17, %r17, %r11;
    mul.lo.u32 %r17, %r17, %r5;
    add.u32 %r17, %r17, %r13;

    mul.wide.u32 %rd3, %r17, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    mul.wide.u32 %rd5, %r10, 4;
    add.s64 %rd6, %rd2, %rd5;
    st.global.f32 [%rd6], %f1;
DONE_ATTENTION_SQUARE_TRANSPOSE:
    ret;
}

.visible .entry heirloom_causal_attention_weights_f32(
    .param .u64 query,
    .param .u64 key,
    .param .u64 attention,
    .param .u32 batch,
    .param .u32 time,
    .param .u32 channels,
    .param .u32 n_heads,
    .param .u32 head_dim,
    .param .u32 total
)
{
    .reg .pred %p<16>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<64>;
    .reg .f32 %f<64>;

    ld.param.u64 %rd1, [query];
    ld.param.u64 %rd2, [key];
    ld.param.u64 %rd3, [attention];
    ld.param.u32 %r4, [batch];
    ld.param.u32 %r5, [time];
    ld.param.u32 %r6, [channels];
    ld.param.u32 %r7, [n_heads];
    ld.param.u32 %r8, [head_dim];
    ld.param.u32 %r9, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r10, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r10, %r9;
    @%p1 bra DONE_CA_ATTN_WEIGHTS;

    rem.u32 %r11, %r10, %r5;
    div.u32 %r12, %r10, %r5;
    rem.u32 %r13, %r12, %r5;
    div.u32 %r14, %r12, %r5;
    rem.u32 %r15, %r14, %r7;
    div.u32 %r16, %r14, %r7;
    mul.wide.u32 %rd4, %r10, 4;
    add.s64 %rd5, %rd3, %rd4;
    setp.gt.u32 %p2, %r11, %r13;
    @%p2 bra CA_ATTN_WEIGHTS_MASKED;

    cvt.rn.f32.u32 %f1, %r8;
    sqrt.rn.f32 %f2, %f1;
    mov.f32 %f3, 0f3f800000;
    div.rn.f32 %f4, %f3, %f2;
    mov.f32 %f5, 0fff800000;
    mov.u32 %r17, 0;
CA_ATTN_WEIGHTS_MAX_U_LOOP:
    setp.gt.u32 %p3, %r17, %r13;
    @%p3 bra CA_ATTN_WEIGHTS_MAX_DONE;
    mov.f32 %f6, 0f00000000;
    mov.u32 %r18, 0;
CA_ATTN_WEIGHTS_MAX_D_LOOP:
    setp.ge.u32 %p4, %r18, %r8;
    @%p4 bra CA_ATTN_WEIGHTS_MAX_D_DONE;
    mul.lo.u32 %r19, %r16, %r5;
    add.u32 %r20, %r19, %r13;
    mul.lo.u32 %r20, %r20, %r6;
    mul.lo.u32 %r21, %r15, %r8;
    add.u32 %r20, %r20, %r21;
    add.u32 %r20, %r20, %r18;
    mul.wide.u32 %rd6, %r20, 4;
    add.s64 %rd7, %rd1, %rd6;
    mul.lo.u32 %r22, %r16, %r5;
    add.u32 %r23, %r22, %r17;
    mul.lo.u32 %r23, %r23, %r6;
    add.u32 %r23, %r23, %r21;
    add.u32 %r23, %r23, %r18;
    mul.wide.u32 %rd8, %r23, 4;
    add.s64 %rd9, %rd2, %rd8;
    ld.global.f32 %f7, [%rd7];
    ld.global.f32 %f8, [%rd9];
    mul.f32 %f9, %f7, %f8;
    add.f32 %f6, %f6, %f9;
    add.u32 %r18, %r18, 1;
    bra CA_ATTN_WEIGHTS_MAX_D_LOOP;
CA_ATTN_WEIGHTS_MAX_D_DONE:
    mul.f32 %f6, %f6, %f4;
    max.f32 %f5, %f5, %f6;
    add.u32 %r17, %r17, 1;
    bra CA_ATTN_WEIGHTS_MAX_U_LOOP;
CA_ATTN_WEIGHTS_MAX_DONE:
    mov.f32 %f10, 0f3fb8aa3b;
    mov.f32 %f11, 0f00000000;
    mov.f32 %f12, 0f00000000;
    mov.u32 %r17, 0;
CA_ATTN_WEIGHTS_SUM_U_LOOP:
    setp.gt.u32 %p3, %r17, %r13;
    @%p3 bra CA_ATTN_WEIGHTS_SUM_DONE;
    mov.f32 %f6, 0f00000000;
    mov.u32 %r18, 0;
CA_ATTN_WEIGHTS_SUM_D_LOOP:
    setp.ge.u32 %p4, %r18, %r8;
    @%p4 bra CA_ATTN_WEIGHTS_SUM_D_DONE;
    mul.lo.u32 %r19, %r16, %r5;
    add.u32 %r20, %r19, %r13;
    mul.lo.u32 %r20, %r20, %r6;
    mul.lo.u32 %r21, %r15, %r8;
    add.u32 %r20, %r20, %r21;
    add.u32 %r20, %r20, %r18;
    mul.wide.u32 %rd6, %r20, 4;
    add.s64 %rd7, %rd1, %rd6;
    mul.lo.u32 %r22, %r16, %r5;
    add.u32 %r23, %r22, %r17;
    mul.lo.u32 %r23, %r23, %r6;
    add.u32 %r23, %r23, %r21;
    add.u32 %r23, %r23, %r18;
    mul.wide.u32 %rd8, %r23, 4;
    add.s64 %rd9, %rd2, %rd8;
    ld.global.f32 %f7, [%rd7];
    ld.global.f32 %f8, [%rd9];
    mul.f32 %f9, %f7, %f8;
    add.f32 %f6, %f6, %f9;
    add.u32 %r18, %r18, 1;
    bra CA_ATTN_WEIGHTS_SUM_D_LOOP;
CA_ATTN_WEIGHTS_SUM_D_DONE:
    mul.f32 %f6, %f6, %f4;
    sub.f32 %f13, %f6, %f5;
    mul.f32 %f14, %f13, %f10;
    ex2.approx.ftz.f32 %f15, %f14;
    add.f32 %f11, %f11, %f15;
    setp.eq.u32 %p5, %r17, %r11;
    @%p5 mov.f32 %f12, %f15;
    add.u32 %r17, %r17, 1;
    bra CA_ATTN_WEIGHTS_SUM_U_LOOP;
CA_ATTN_WEIGHTS_SUM_DONE:
    div.rn.f32 %f16, %f12, %f11;
    st.global.f32 [%rd5], %f16;
    bra DONE_CA_ATTN_WEIGHTS;
CA_ATTN_WEIGHTS_MASKED:
    mov.f32 %f17, 0f00000000;
    st.global.f32 [%rd5], %f17;
DONE_CA_ATTN_WEIGHTS:
    ret;
}

.visible .entry heirloom_causal_attention_output_f32(
    .param .u64 attention,
    .param .u64 value,
    .param .u64 output,
    .param .u32 batch,
    .param .u32 time,
    .param .u32 channels,
    .param .u32 n_heads,
    .param .u32 head_dim,
    .param .u32 total
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<64>;
    .reg .f32 %f<16>;

    ld.param.u64 %rd1, [attention];
    ld.param.u64 %rd2, [value];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [batch];
    ld.param.u32 %r5, [time];
    ld.param.u32 %r6, [channels];
    ld.param.u32 %r7, [n_heads];
    ld.param.u32 %r8, [head_dim];
    ld.param.u32 %r9, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r10, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r10, %r9;
    @%p1 bra DONE_CA_ATTN_OUTPUT;

    rem.u32 %r11, %r10, %r6;
    div.u32 %r12, %r10, %r6;
    rem.u32 %r13, %r12, %r5;
    div.u32 %r16, %r12, %r5;
    div.u32 %r15, %r11, %r8;
    rem.u32 %r18, %r11, %r8;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r17, 0;
CA_ATTN_OUTPUT_S_LOOP:
    setp.gt.u32 %p2, %r17, %r13;
    @%p2 bra CA_ATTN_OUTPUT_WRITE;
    mul.lo.u32 %r19, %r16, %r7;
    add.u32 %r19, %r19, %r15;
    mul.lo.u32 %r19, %r19, %r5;
    add.u32 %r19, %r19, %r13;
    mul.lo.u32 %r19, %r19, %r5;
    add.u32 %r19, %r19, %r17;
    mul.wide.u32 %rd4, %r19, 4;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];
    mul.lo.u32 %r20, %r16, %r5;
    add.u32 %r20, %r20, %r17;
    mul.lo.u32 %r20, %r20, %r6;
    mul.lo.u32 %r21, %r15, %r8;
    add.u32 %r20, %r20, %r21;
    add.u32 %r20, %r20, %r18;
    mul.wide.u32 %rd6, %r20, 4;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.f32 %f3, [%rd7];
    mul.f32 %f4, %f2, %f3;
    add.f32 %f1, %f1, %f4;
    add.u32 %r17, %r17, 1;
    bra CA_ATTN_OUTPUT_S_LOOP;
CA_ATTN_OUTPUT_WRITE:
    mul.wide.u32 %rd8, %r10, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f1;
DONE_CA_ATTN_OUTPUT:
    ret;
}

.visible .entry heirloom_causal_attention_backward_value_f32(
    .param .u64 attention,
    .param .u64 grad_output,
    .param .u64 output,
    .param .u32 batch,
    .param .u32 time,
    .param .u32 channels,
    .param .u32 n_heads,
    .param .u32 head_dim,
    .param .u32 total
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<64>;
    .reg .f32 %f<16>;

    ld.param.u64 %rd1, [attention];
    ld.param.u64 %rd2, [grad_output];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [batch];
    ld.param.u32 %r5, [time];
    ld.param.u32 %r6, [channels];
    ld.param.u32 %r7, [n_heads];
    ld.param.u32 %r8, [head_dim];
    ld.param.u32 %r9, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r10, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r10, %r9;
    @%p1 bra DONE_CA_ATTN_BWD_VALUE;

    rem.u32 %r11, %r10, %r6;
    div.u32 %r12, %r10, %r6;
    rem.u32 %r13, %r12, %r5;
    div.u32 %r16, %r12, %r5;
    div.u32 %r15, %r11, %r8;
    rem.u32 %r18, %r11, %r8;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r17, %r13;
CA_ATTN_BWD_VALUE_T_LOOP:
    setp.ge.u32 %p2, %r17, %r5;
    @%p2 bra CA_ATTN_BWD_VALUE_WRITE;
    mul.lo.u32 %r19, %r16, %r7;
    add.u32 %r19, %r19, %r15;
    mul.lo.u32 %r19, %r19, %r5;
    add.u32 %r19, %r19, %r17;
    mul.lo.u32 %r19, %r19, %r5;
    add.u32 %r19, %r19, %r13;
    mul.wide.u32 %rd4, %r19, 4;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];
    mul.lo.u32 %r20, %r16, %r5;
    add.u32 %r20, %r20, %r17;
    mul.lo.u32 %r20, %r20, %r6;
    mul.lo.u32 %r21, %r15, %r8;
    add.u32 %r20, %r20, %r21;
    add.u32 %r20, %r20, %r18;
    mul.wide.u32 %rd6, %r20, 4;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.f32 %f3, [%rd7];
    mul.f32 %f4, %f2, %f3;
    add.f32 %f1, %f1, %f4;
    add.u32 %r17, %r17, 1;
    bra CA_ATTN_BWD_VALUE_T_LOOP;
CA_ATTN_BWD_VALUE_WRITE:
    mul.wide.u32 %rd8, %r10, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f1;
DONE_CA_ATTN_BWD_VALUE:
    ret;
}

.visible .entry heirloom_causal_attention_backward_query_f32(
    .param .u64 query,
    .param .u64 key,
    .param .u64 value,
    .param .u64 attention,
    .param .u64 grad_output,
    .param .u64 output,
    .param .u32 batch,
    .param .u32 time,
    .param .u32 channels,
    .param .u32 n_heads,
    .param .u32 head_dim,
    .param .u32 total
)
{
    .reg .pred %p<16>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<64>;
    .reg .f32 %f<64>;

    ld.param.u64 %rd1, [query];
    ld.param.u64 %rd2, [key];
    ld.param.u64 %rd3, [value];
    ld.param.u64 %rd4, [attention];
    ld.param.u64 %rd5, [grad_output];
    ld.param.u64 %rd6, [output];
    ld.param.u32 %r4, [batch];
    ld.param.u32 %r5, [time];
    ld.param.u32 %r6, [channels];
    ld.param.u32 %r7, [n_heads];
    ld.param.u32 %r8, [head_dim];
    ld.param.u32 %r9, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r10, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r10, %r9;
    @%p1 bra DONE_CA_ATTN_BWD_QUERY;

    rem.u32 %r11, %r10, %r6;
    div.u32 %r12, %r10, %r6;
    rem.u32 %r13, %r12, %r5;
    div.u32 %r16, %r12, %r5;
    div.u32 %r15, %r11, %r8;
    rem.u32 %r18, %r11, %r8;
    cvt.rn.f32.u32 %f1, %r8;
    sqrt.rn.f32 %f2, %f1;
    mov.f32 %f3, 0f3f800000;
    div.rn.f32 %f4, %f3, %f2;

    mov.f32 %f20, 0f00000000;
    mov.u32 %r17, 0;
CA_ATTN_BWD_QUERY_DOT_U_LOOP:
    setp.gt.u32 %p2, %r17, %r13;
    @%p2 bra CA_ATTN_BWD_QUERY_DOT_DONE;
    mov.f32 %f21, 0f00000000;
    mov.u32 %r19, 0;
CA_ATTN_BWD_QUERY_DOT_D_LOOP:
    setp.ge.u32 %p3, %r19, %r8;
    @%p3 bra CA_ATTN_BWD_QUERY_DOT_D_DONE;
    mul.lo.u32 %r20, %r16, %r5;
    add.u32 %r20, %r20, %r13;
    mul.lo.u32 %r20, %r20, %r6;
    mul.lo.u32 %r21, %r15, %r8;
    add.u32 %r20, %r20, %r21;
    add.u32 %r20, %r20, %r19;
    mul.wide.u32 %rd7, %r20, 4;
    add.s64 %rd8, %rd5, %rd7;
    mul.lo.u32 %r22, %r16, %r5;
    add.u32 %r22, %r22, %r17;
    mul.lo.u32 %r22, %r22, %r6;
    add.u32 %r22, %r22, %r21;
    add.u32 %r22, %r22, %r19;
    mul.wide.u32 %rd9, %r22, 4;
    add.s64 %rd10, %rd3, %rd9;
    ld.global.f32 %f22, [%rd8];
    ld.global.f32 %f23, [%rd10];
    mul.f32 %f24, %f22, %f23;
    add.f32 %f21, %f21, %f24;
    add.u32 %r19, %r19, 1;
    bra CA_ATTN_BWD_QUERY_DOT_D_LOOP;
CA_ATTN_BWD_QUERY_DOT_D_DONE:
    mul.lo.u32 %r23, %r16, %r7;
    add.u32 %r23, %r23, %r15;
    mul.lo.u32 %r23, %r23, %r5;
    add.u32 %r23, %r23, %r13;
    mul.lo.u32 %r23, %r23, %r5;
    add.u32 %r23, %r23, %r17;
    mul.wide.u32 %rd11, %r23, 4;
    add.s64 %rd12, %rd4, %rd11;
    ld.global.f32 %f25, [%rd12];
    mul.f32 %f26, %f21, %f25;
    add.f32 %f20, %f20, %f26;
    add.u32 %r17, %r17, 1;
    bra CA_ATTN_BWD_QUERY_DOT_U_LOOP;
CA_ATTN_BWD_QUERY_DOT_DONE:
    mov.f32 %f30, 0f00000000;
    mov.u32 %r17, 0;
CA_ATTN_BWD_QUERY_S_LOOP:
    setp.gt.u32 %p4, %r17, %r13;
    @%p4 bra CA_ATTN_BWD_QUERY_WRITE;
    mov.f32 %f21, 0f00000000;
    mov.u32 %r19, 0;
CA_ATTN_BWD_QUERY_S_D_LOOP:
    setp.ge.u32 %p5, %r19, %r8;
    @%p5 bra CA_ATTN_BWD_QUERY_S_D_DONE;
    mul.lo.u32 %r20, %r16, %r5;
    add.u32 %r20, %r20, %r13;
    mul.lo.u32 %r20, %r20, %r6;
    mul.lo.u32 %r21, %r15, %r8;
    add.u32 %r20, %r20, %r21;
    add.u32 %r20, %r20, %r19;
    mul.wide.u32 %rd7, %r20, 4;
    add.s64 %rd8, %rd5, %rd7;
    mul.lo.u32 %r22, %r16, %r5;
    add.u32 %r22, %r22, %r17;
    mul.lo.u32 %r22, %r22, %r6;
    add.u32 %r22, %r22, %r21;
    add.u32 %r22, %r22, %r19;
    mul.wide.u32 %rd9, %r22, 4;
    add.s64 %rd10, %rd3, %rd9;
    ld.global.f32 %f22, [%rd8];
    ld.global.f32 %f23, [%rd10];
    mul.f32 %f24, %f22, %f23;
    add.f32 %f21, %f21, %f24;
    add.u32 %r19, %r19, 1;
    bra CA_ATTN_BWD_QUERY_S_D_LOOP;
CA_ATTN_BWD_QUERY_S_D_DONE:
    mul.lo.u32 %r23, %r16, %r7;
    add.u32 %r23, %r23, %r15;
    mul.lo.u32 %r23, %r23, %r5;
    add.u32 %r23, %r23, %r13;
    mul.lo.u32 %r23, %r23, %r5;
    add.u32 %r23, %r23, %r17;
    mul.wide.u32 %rd11, %r23, 4;
    add.s64 %rd12, %rd4, %rd11;
    ld.global.f32 %f25, [%rd12];
    sub.f32 %f26, %f21, %f20;
    mul.f32 %f27, %f25, %f26;
    mul.lo.u32 %r24, %r16, %r5;
    add.u32 %r24, %r24, %r17;
    mul.lo.u32 %r24, %r24, %r6;
    mul.lo.u32 %r25, %r15, %r8;
    add.u32 %r24, %r24, %r25;
    add.u32 %r24, %r24, %r18;
    mul.wide.u32 %rd13, %r24, 4;
    add.s64 %rd14, %rd2, %rd13;
    ld.global.f32 %f28, [%rd14];
    mul.f32 %f29, %f27, %f28;
    mul.f32 %f29, %f29, %f4;
    add.f32 %f30, %f30, %f29;
    add.u32 %r17, %r17, 1;
    bra CA_ATTN_BWD_QUERY_S_LOOP;
CA_ATTN_BWD_QUERY_WRITE:
    mul.wide.u32 %rd15, %r10, 4;
    add.s64 %rd16, %rd6, %rd15;
    st.global.f32 [%rd16], %f30;
DONE_CA_ATTN_BWD_QUERY:
    ret;
}

.visible .entry heirloom_causal_attention_backward_key_f32(
    .param .u64 query,
    .param .u64 key,
    .param .u64 value,
    .param .u64 attention,
    .param .u64 grad_output,
    .param .u64 output,
    .param .u32 batch,
    .param .u32 time,
    .param .u32 channels,
    .param .u32 n_heads,
    .param .u32 head_dim,
    .param .u32 total
)
{
    .reg .pred %p<16>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<64>;
    .reg .f32 %f<64>;

    ld.param.u64 %rd1, [query];
    ld.param.u64 %rd2, [key];
    ld.param.u64 %rd3, [value];
    ld.param.u64 %rd4, [attention];
    ld.param.u64 %rd5, [grad_output];
    ld.param.u64 %rd6, [output];
    ld.param.u32 %r4, [batch];
    ld.param.u32 %r5, [time];
    ld.param.u32 %r6, [channels];
    ld.param.u32 %r7, [n_heads];
    ld.param.u32 %r8, [head_dim];
    ld.param.u32 %r9, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r10, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r10, %r9;
    @%p1 bra DONE_CA_ATTN_BWD_KEY;

    rem.u32 %r11, %r10, %r6;
    div.u32 %r12, %r10, %r6;
    rem.u32 %r13, %r12, %r5;
    div.u32 %r16, %r12, %r5;
    div.u32 %r15, %r11, %r8;
    rem.u32 %r18, %r11, %r8;
    cvt.rn.f32.u32 %f1, %r8;
    sqrt.rn.f32 %f2, %f1;
    mov.f32 %f3, 0f3f800000;
    div.rn.f32 %f4, %f3, %f2;
    mov.f32 %f30, 0f00000000;
    mov.u32 %r17, %r13;
CA_ATTN_BWD_KEY_T_LOOP:
    setp.ge.u32 %p2, %r17, %r5;
    @%p2 bra CA_ATTN_BWD_KEY_WRITE;
    mov.f32 %f20, 0f00000000;
    mov.u32 %r19, 0;
CA_ATTN_BWD_KEY_DOT_U_LOOP:
    setp.gt.u32 %p3, %r19, %r17;
    @%p3 bra CA_ATTN_BWD_KEY_DOT_DONE;
    mov.f32 %f21, 0f00000000;
    mov.u32 %r20, 0;
CA_ATTN_BWD_KEY_DOT_D_LOOP:
    setp.ge.u32 %p4, %r20, %r8;
    @%p4 bra CA_ATTN_BWD_KEY_DOT_D_DONE;
    mul.lo.u32 %r21, %r16, %r5;
    add.u32 %r21, %r21, %r17;
    mul.lo.u32 %r21, %r21, %r6;
    mul.lo.u32 %r22, %r15, %r8;
    add.u32 %r21, %r21, %r22;
    add.u32 %r21, %r21, %r20;
    mul.wide.u32 %rd7, %r21, 4;
    add.s64 %rd8, %rd5, %rd7;
    mul.lo.u32 %r23, %r16, %r5;
    add.u32 %r23, %r23, %r19;
    mul.lo.u32 %r23, %r23, %r6;
    add.u32 %r23, %r23, %r22;
    add.u32 %r23, %r23, %r20;
    mul.wide.u32 %rd9, %r23, 4;
    add.s64 %rd10, %rd3, %rd9;
    ld.global.f32 %f22, [%rd8];
    ld.global.f32 %f23, [%rd10];
    mul.f32 %f24, %f22, %f23;
    add.f32 %f21, %f21, %f24;
    add.u32 %r20, %r20, 1;
    bra CA_ATTN_BWD_KEY_DOT_D_LOOP;
CA_ATTN_BWD_KEY_DOT_D_DONE:
    mul.lo.u32 %r24, %r16, %r7;
    add.u32 %r24, %r24, %r15;
    mul.lo.u32 %r24, %r24, %r5;
    add.u32 %r24, %r24, %r17;
    mul.lo.u32 %r24, %r24, %r5;
    add.u32 %r24, %r24, %r19;
    mul.wide.u32 %rd11, %r24, 4;
    add.s64 %rd12, %rd4, %rd11;
    ld.global.f32 %f25, [%rd12];
    mul.f32 %f26, %f21, %f25;
    add.f32 %f20, %f20, %f26;
    add.u32 %r19, %r19, 1;
    bra CA_ATTN_BWD_KEY_DOT_U_LOOP;
CA_ATTN_BWD_KEY_DOT_DONE:
    mov.f32 %f21, 0f00000000;
    mov.u32 %r20, 0;
CA_ATTN_BWD_KEY_S_D_LOOP:
    setp.ge.u32 %p5, %r20, %r8;
    @%p5 bra CA_ATTN_BWD_KEY_S_D_DONE;
    mul.lo.u32 %r21, %r16, %r5;
    add.u32 %r21, %r21, %r17;
    mul.lo.u32 %r21, %r21, %r6;
    mul.lo.u32 %r22, %r15, %r8;
    add.u32 %r21, %r21, %r22;
    add.u32 %r21, %r21, %r20;
    mul.wide.u32 %rd7, %r21, 4;
    add.s64 %rd8, %rd5, %rd7;
    mul.lo.u32 %r23, %r16, %r5;
    add.u32 %r23, %r23, %r13;
    mul.lo.u32 %r23, %r23, %r6;
    add.u32 %r23, %r23, %r22;
    add.u32 %r23, %r23, %r20;
    mul.wide.u32 %rd9, %r23, 4;
    add.s64 %rd10, %rd3, %rd9;
    ld.global.f32 %f22, [%rd8];
    ld.global.f32 %f23, [%rd10];
    mul.f32 %f24, %f22, %f23;
    add.f32 %f21, %f21, %f24;
    add.u32 %r20, %r20, 1;
    bra CA_ATTN_BWD_KEY_S_D_LOOP;
CA_ATTN_BWD_KEY_S_D_DONE:
    mul.lo.u32 %r24, %r16, %r7;
    add.u32 %r24, %r24, %r15;
    mul.lo.u32 %r24, %r24, %r5;
    add.u32 %r24, %r24, %r17;
    mul.lo.u32 %r24, %r24, %r5;
    add.u32 %r24, %r24, %r13;
    mul.wide.u32 %rd11, %r24, 4;
    add.s64 %rd12, %rd4, %rd11;
    ld.global.f32 %f25, [%rd12];
    sub.f32 %f26, %f21, %f20;
    mul.f32 %f27, %f25, %f26;
    mul.lo.u32 %r25, %r16, %r5;
    add.u32 %r25, %r25, %r17;
    mul.lo.u32 %r25, %r25, %r6;
    mul.lo.u32 %r26, %r15, %r8;
    add.u32 %r25, %r25, %r26;
    add.u32 %r25, %r25, %r18;
    mul.wide.u32 %rd13, %r25, 4;
    add.s64 %rd14, %rd1, %rd13;
    ld.global.f32 %f28, [%rd14];
    mul.f32 %f29, %f27, %f28;
    mul.f32 %f29, %f29, %f4;
    add.f32 %f30, %f30, %f29;
    add.u32 %r17, %r17, 1;
    bra CA_ATTN_BWD_KEY_T_LOOP;
CA_ATTN_BWD_KEY_WRITE:
    mul.wide.u32 %rd15, %r10, 4;
    add.s64 %rd16, %rd6, %rd15;
    st.global.f32 [%rd16], %f30;
DONE_CA_ATTN_BWD_KEY:
    ret;
}

.visible .entry heirloom_sum_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<7>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ne.u32 %p1, %r5, 0;
    @%p1 bra DONE_SUM;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r6, 0;
SUM_LOOP:
    setp.ge.u32 %p2, %r6, %r4;
    @%p2 bra SUM_WRITE;
    mul.wide.u32 %rd3, %r6, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f2, [%rd4];
    add.f32 %f1, %f1, %f2;
    add.u32 %r6, %r6, 1;
    bra SUM_LOOP;
SUM_WRITE:
    st.global.f32 [%rd2], %f1;
DONE_SUM:
    ret;
}

.visible .entry heirloom_sum_squares_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<7>;
    .reg .f32 %f<5>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ne.u32 %p1, %r5, 0;
    @%p1 bra DONE_SUM_SQUARES;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r6, 0;
SUM_SQUARES_LOOP:
    setp.ge.u32 %p2, %r6, %r4;
    @%p2 bra SUM_SQUARES_WRITE;
    mul.wide.u32 %rd3, %r6, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f2, [%rd4];
    mul.f32 %f3, %f2, %f2;
    add.f32 %f1, %f1, %f3;
    add.u32 %r6, %r6, 1;
    bra SUM_SQUARES_LOOP;
SUM_SQUARES_WRITE:
    st.global.f32 [%rd2], %f1;
DONE_SUM_SQUARES:
    ret;
}

.visible .entry heirloom_mean_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<7>;
    .reg .f32 %f<5>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ne.u32 %p1, %r5, 0;
    @%p1 bra DONE_MEAN;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r6, 0;
MEAN_LOOP:
    setp.ge.u32 %p2, %r6, %r4;
    @%p2 bra MEAN_WRITE;
    mul.wide.u32 %rd3, %r6, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f2, [%rd4];
    add.f32 %f1, %f1, %f2;
    add.u32 %r6, %r6, 1;
    bra MEAN_LOOP;
MEAN_WRITE:
    cvt.rn.f32.u32 %f3, %r4;
    div.rn.f32 %f4, %f1, %f3;
    st.global.f32 [%rd2], %f4;
DONE_MEAN:
    ret;
}

.visible .entry heirloom_relu_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_RELU;
    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];
    mov.f32 %f2, 0f00000000;
    max.f32 %f3, %f1, %f2;
    st.global.f32 [%rd5], %f3;
DONE_RELU:
    ret;
}

.visible .entry heirloom_relu_backward_f32(
    .param .u64 input,
    .param .u64 grad_output,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<5>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [grad_output];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_RELU_BACKWARD;
    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    mov.f32 %f3, 0f00000000;
    setp.gt.f32 %p2, %f1, %f3;
    selp.f32 %f4, %f2, %f3, %p2;
    st.global.f32 [%rd7], %f4;
DONE_RELU_BACKWARD:
    ret;
}

.visible .entry heirloom_gelu_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<20>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_GELU;
    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];

    mul.f32 %f2, %f1, %f1;
    mul.f32 %f3, %f2, %f1;
    mov.f32 %f4, 0f3D372713;
    mul.f32 %f5, %f4, %f3;
    add.f32 %f6, %f1, %f5;
    mov.f32 %f7, 0f3F4C422A;
    mul.f32 %f8, %f7, %f6;

    mov.f32 %f9, 0fC038AA3B;
    mul.f32 %f10, %f8, %f9;
    ex2.approx.ftz.f32 %f11, %f10;
    mov.f32 %f12, 0f3F800000;
    add.f32 %f13, %f12, %f11;
    mov.f32 %f14, 0f40000000;
    div.rn.f32 %f15, %f14, %f13;
    sub.f32 %f16, %f15, %f12;

    add.f32 %f17, %f12, %f16;
    mul.f32 %f18, %f1, %f17;
    mov.f32 %f19, 0f3F000000;
    mul.f32 %f18, %f18, %f19;
    st.global.f32 [%rd5], %f18;
DONE_GELU:
    ret;
}

.visible .entry heirloom_gelu_backward_f32(
    .param .u64 input,
    .param .u64 grad_output,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<26>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [grad_output];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra DONE_GELU_BACKWARD;
    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];

    mul.f32 %f3, %f1, %f1;
    mul.f32 %f4, %f3, %f1;
    mov.f32 %f5, 0f3D372713;
    mul.f32 %f6, %f5, %f4;
    add.f32 %f7, %f1, %f6;
    mov.f32 %f8, 0f3F4C422A;
    mul.f32 %f9, %f8, %f7;

    mov.f32 %f10, 0fC038AA3B;
    mul.f32 %f11, %f9, %f10;
    ex2.approx.ftz.f32 %f12, %f11;
    mov.f32 %f13, 0f3F800000;
    add.f32 %f14, %f13, %f12;
    mov.f32 %f15, 0f40000000;
    div.rn.f32 %f16, %f15, %f14;
    sub.f32 %f17, %f16, %f13;

    add.f32 %f18, %f13, %f17;
    mov.f32 %f19, 0f3F000000;
    mul.f32 %f20, %f19, %f18;

    mul.f32 %f21, %f17, %f17;
    sub.f32 %f22, %f13, %f21;
    mov.f32 %f23, 0f3E095D4F;
    mul.f32 %f24, %f23, %f3;
    add.f32 %f24, %f13, %f24;
    mul.f32 %f25, %f19, %f1;
    mul.f32 %f25, %f25, %f22;
    mul.f32 %f25, %f25, %f8;
    mul.f32 %f25, %f25, %f24;
    add.f32 %f25, %f20, %f25;
    mul.f32 %f25, %f2, %f25;
    st.global.f32 [%rd7], %f25;
DONE_GELU_BACKWARD:
    ret;
}

.visible .entry heirloom_transpose2d_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 rows,
    .param .u32 cols,
    .param .u32 total
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<9>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<2>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r4, [rows];
    ld.param.u32 %r5, [cols];
    ld.param.u32 %r6, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r7, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r7, %r6;
    @%p1 bra DONE_TRANSPOSE2D;
    div.u32 %r8, %r7, %r5;
    rem.u32 %r6, %r7, %r5;
    mul.wide.u32 %rd3, %r7, 4;
    add.s64 %rd4, %rd1, %rd3;
    mad.lo.s32 %r6, %r6, %r4, %r8;
    mul.wide.u32 %rd5, %r6, 4;
    add.s64 %rd6, %rd2, %rd5;
    ld.global.f32 %f1, [%rd4];
    st.global.f32 [%rd6], %f1;
DONE_TRANSPOSE2D:
    ret;
}

.visible .entry heirloom_transpose2d_bf16(
    .param .u64 input,
    .param .u64 output,
    .param .u32 rows,
    .param .u32 cols,
    .param .u32 total
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<9>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r4, [rows];
    ld.param.u32 %r5, [cols];
    ld.param.u32 %r6, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r7, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r7, %r6;
    @%p1 bra DONE_TRANSPOSE2D_BF16;
    div.u32 %r8, %r7, %r5;
    rem.u32 %r6, %r7, %r5;
    mul.wide.u32 %rd3, %r7, 2;
    add.s64 %rd4, %rd1, %rd3;
    mad.lo.s32 %r6, %r6, %r4, %r8;
    mul.wide.u32 %rd5, %r6, 2;
    add.s64 %rd6, %rd2, %rd5;
    ld.global.u16 %r1, [%rd4];
    st.global.u16 [%rd6], %r1;
DONE_TRANSPOSE2D_BF16:
    ret;
}

.visible .entry heirloom_transpose2d_pair_bf16(
    .param .u64 first_input,
    .param .u64 first_output,
    .param .u64 second_input,
    .param .u64 second_output,
    .param .u32 first_rows,
    .param .u32 first_cols,
    .param .u32 first_total,
    .param .u32 second_rows,
    .param .u32 second_cols,
    .param .u32 second_total,
    .param .u32 launch_total
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<18>;

    ld.param.u64 %rd1, [first_input];
    ld.param.u64 %rd2, [first_output];
    ld.param.u64 %rd3, [second_input];
    ld.param.u64 %rd4, [second_output];
    ld.param.u32 %r4, [first_rows];
    ld.param.u32 %r5, [first_cols];
    ld.param.u32 %r6, [first_total];
    ld.param.u32 %r7, [second_rows];
    ld.param.u32 %r8, [second_cols];
    ld.param.u32 %r9, [second_total];
    ld.param.u32 %r10, [launch_total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r11, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r11, %r10;
    @%p1 bra DONE_TRANSPOSE2D_PAIR_BF16;

    setp.ge.u32 %p2, %r11, %r6;
    @%p2 bra SKIP_FIRST_TRANSPOSE2D_PAIR_BF16;
    div.u32 %r12, %r11, %r5;
    rem.u32 %r13, %r11, %r5;
    mul.wide.u32 %rd5, %r11, 2;
    add.s64 %rd6, %rd1, %rd5;
    mad.lo.s32 %r14, %r13, %r4, %r12;
    mul.wide.u32 %rd7, %r14, 2;
    add.s64 %rd8, %rd2, %rd7;
    ld.global.u16 %r15, [%rd6];
    st.global.u16 [%rd8], %r15;
SKIP_FIRST_TRANSPOSE2D_PAIR_BF16:

    setp.ge.u32 %p3, %r11, %r9;
    @%p3 bra DONE_TRANSPOSE2D_PAIR_BF16;
    div.u32 %r16, %r11, %r8;
    rem.u32 %r17, %r11, %r8;
    mul.wide.u32 %rd9, %r11, 2;
    add.s64 %rd10, %rd3, %rd9;
    mad.lo.s32 %r18, %r17, %r7, %r16;
    mul.wide.u32 %rd11, %r18, 2;
    add.s64 %rd12, %rd4, %rd11;
    ld.global.u16 %r19, [%rd10];
    st.global.u16 [%rd12], %r19;
DONE_TRANSPOSE2D_PAIR_BF16:
    ret;
}

.visible .entry heirloom_materialize_matrix_layout_bf16(
    .param .u64 input,
    .param .u64 output,
    .param .u32 rows,
    .param .u32 cols,
    .param .u32 row_stride,
    .param .u32 col_stride,
    .param .u32 offset,
    .param .u32 total
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r4, [rows];
    ld.param.u32 %r5, [cols];
    ld.param.u32 %r6, [row_stride];
    ld.param.u32 %r7, [col_stride];
    ld.param.u32 %r8, [offset];
    ld.param.u32 %r9, [total];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r10, %r2, %r3, %r1;
    setp.ge.u32 %p1, %r10, %r9;
    @%p1 bra DONE_MATERIALIZE_MATRIX_LAYOUT_BF16;
    div.u32 %r11, %r10, %r5;
    rem.u32 %r12, %r10, %r5;
    mad.lo.u32 %r13, %r11, %r6, %r8;
    mad.lo.u32 %r13, %r12, %r7, %r13;
    mul.wide.u32 %rd3, %r13, 2;
    add.s64 %rd4, %rd1, %rd3;
    mul.wide.u32 %rd5, %r10, 2;
    add.s64 %rd6, %rd2, %rd5;
    ld.global.u16 %r14, [%rd4];
    st.global.u16 [%rd6], %r14;
DONE_MATERIALIZE_MATRIX_LAYOUT_BF16:
    ret;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::mem::{align_of, size_of};

    #[test]
    fn cuda_smoke_is_skipped_when_driver_is_unavailable() {
        if is_available() {
            let report = smoke_f32(0, 64).unwrap();
            assert_eq!(report.len, 64);
            assert!(report.add_max_abs_error <= 1e-6);
            assert!(report.relu_max_abs_error <= 1e-6);
        } else if let Ok(info) = system_info() {
            assert_eq!(info.device_count, 0);
        }
    }

    #[test]
    fn nccl_unique_id_layout_matches_nccl_header_contract() {
        assert_eq!(size_of::<NcclUniqueId>(), 128);
        assert_eq!(align_of::<NcclUniqueId>(), align_of::<c_char>());
    }

    #[test]
    fn nccl_unique_id_hex_round_trip_preserves_all_bytes() {
        let mut internal = [0 as c_char; 128];
        for (index, byte) in internal.iter_mut().enumerate() {
            *byte = index as u8 as c_char;
        }
        let id = NcclUniqueId { internal };
        let decoded = NcclUniqueId::from_hex(&id.to_hex()).unwrap();
        assert_eq!(decoded, id);
    }

    #[test]
    fn embedded_ptx_predicate_declarations_cover_uses() {
        for ptx in [KERNEL_PTX, TENSOR_CORE_PTX] {
            for entry in ptx.split(".visible .entry ").skip(1) {
                let name = entry.split('(').next().unwrap_or("<unknown>").trim();
                let Some(declared) = declared_predicate_count(entry) else {
                    continue;
                };
                let used = used_predicate_registers(entry);
                if let Some(max_used) = used.iter().next_back() {
                    assert!(
                        *max_used < declared,
                        "{name} declares %p<{declared}> but uses %p{max_used}; used={used:?}"
                    );
                }
            }
        }
    }

    fn declared_predicate_count(entry: &str) -> Option<usize> {
        let marker = ".reg .pred %p<";
        let start = entry.find(marker)? + marker.len();
        let end = entry[start..].find('>')? + start;
        entry[start..end].parse().ok()
    }

    fn used_predicate_registers(entry: &str) -> BTreeSet<usize> {
        let mut used = BTreeSet::new();
        for segment in entry.split("%p").skip(1) {
            let digits: String = segment
                .chars()
                .take_while(|character| character.is_ascii_digit())
                .collect();
            if let Ok(index) = digits.parse() {
                used.insert(index);
            }
        }
        used
    }
}
