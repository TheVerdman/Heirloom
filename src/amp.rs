use crate::{DType, Device, Result, TensorError};
use serde::{Deserialize, Serialize};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AmpBf16Policy {
    pub precision: String,
    pub parameter_dtype: String,
    pub gradient_dtype: String,
    pub optimizer_state_dtype: String,
    pub matmul_operand_dtype: String,
    pub matmul_accumulation_dtype: String,
    pub reduction_dtype: String,
    pub loss_dtype: String,
    pub layer_norm_dtype: String,
    pub softmax_dtype: String,
    pub finite_checks: bool,
    pub strict_no_cpu_staging: bool,
    pub dynamic_loss_scaling: bool,
}

impl AmpBf16Policy {
    pub fn cuda_training() -> Self {
        Self {
            precision: "amp-bf16".to_string(),
            parameter_dtype: "f32".to_string(),
            gradient_dtype: "f32".to_string(),
            optimizer_state_dtype: "f32".to_string(),
            matmul_operand_dtype: "bf16".to_string(),
            matmul_accumulation_dtype: "f32".to_string(),
            reduction_dtype: "f32".to_string(),
            loss_dtype: "f32".to_string(),
            layer_norm_dtype: "f32".to_string(),
            softmax_dtype: "f32".to_string(),
            finite_checks: true,
            strict_no_cpu_staging: true,
            dynamic_loss_scaling: false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AmpBf16OpDecision {
    pub op: String,
    pub input_dtype: String,
    pub input_device: String,
    pub compute_dtype: String,
    pub accumulation_dtype: String,
    pub output_dtype: String,
    pub kernel_path: String,
    pub tensor_core: bool,
    pub fallback_reason: Option<String>,
    pub finite_check: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AmpBf16FiniteCheckEvent {
    pub name: String,
    pub target: String,
    pub dtype: String,
    pub device: String,
    pub value: String,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CudaHostStagingEvent {
    pub reason: String,
    pub dtype: String,
    pub device: String,
    pub len: usize,
    pub allowed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AmpBf16ValidationReport {
    pub schema_version: u32,
    pub enabled: bool,
    pub status: String,
    pub strict_no_cpu_staging: bool,
    pub dynamic_loss_scaling: bool,
    pub op_decision_count: usize,
    pub tensor_core_decision_count: usize,
    pub fallback_decision_count: usize,
    pub fallback_reasons: BTreeMap<String, usize>,
    pub finite_check_count: usize,
    pub finite_check_failure_count: usize,
    pub unexpected_host_staging_count: usize,
    pub allowed_host_staging_count: usize,
    pub notes: Vec<String>,
}

thread_local! {
    static AMP_BF16_TRAINING_DEPTH: Cell<usize> = const { Cell::new(0) };
    static CUDA_HOST_STAGING_ALLOW_DEPTH: Cell<usize> = const { Cell::new(0) };
    static CUDA_HOST_STAGING_EVENTS: RefCell<Vec<CudaHostStagingEvent>> = const { RefCell::new(Vec::new()) };
    static AMP_BF16_OP_DECISIONS: RefCell<Vec<AmpBf16OpDecision>> = const { RefCell::new(Vec::new()) };
    static AMP_BF16_FINITE_CHECKS: RefCell<Vec<AmpBf16FiniteCheckEvent>> = const { RefCell::new(Vec::new()) };
}

pub struct AmpBf16TrainingGuard;

impl Drop for AmpBf16TrainingGuard {
    fn drop(&mut self) {
        AMP_BF16_TRAINING_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

pub struct CudaHostStagingGuard;

impl Drop for CudaHostStagingGuard {
    fn drop(&mut self) {
        CUDA_HOST_STAGING_ALLOW_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

pub fn enter_amp_bf16_training() -> AmpBf16TrainingGuard {
    AMP_BF16_TRAINING_DEPTH.with(|depth| depth.set(depth.get() + 1));
    AmpBf16TrainingGuard
}

pub fn is_amp_bf16_training() -> bool {
    AMP_BF16_TRAINING_DEPTH.with(|depth| depth.get() > 0)
}

pub fn allow_cuda_host_staging(_reason: &str) -> CudaHostStagingGuard {
    CUDA_HOST_STAGING_ALLOW_DEPTH.with(|depth| depth.set(depth.get() + 1));
    CudaHostStagingGuard
}

pub fn with_cuda_host_staging_allowed<T>(reason: &str, f: impl FnOnce() -> T) -> T {
    let _guard = allow_cuda_host_staging(reason);
    f()
}

pub(crate) fn record_cuda_host_staging(
    reason: &str,
    dtype: DType,
    device: Device,
    len: usize,
) -> Result<()> {
    let in_strict_training = AMP_BF16_TRAINING_DEPTH.with(|depth| depth.get() > 0);
    let allowed = CUDA_HOST_STAGING_ALLOW_DEPTH.with(|depth| depth.get() > 0);
    CUDA_HOST_STAGING_EVENTS.with(|events| {
        events.borrow_mut().push(CudaHostStagingEvent {
            reason: reason.to_string(),
            dtype: dtype_label(dtype),
            device: device_label(device),
            len,
            allowed,
        });
    });
    if in_strict_training && !allowed {
        return Err(TensorError::Device(format!(
            "strict amp-bf16 training forbids CUDA host staging for {reason}; \
             dtype={} device={} len={len}. Wrap intentional scalar/report/checkpoint reads in \
             allow_cuda_host_staging.",
            dtype_label(dtype),
            device_label(device)
        )));
    }
    Ok(())
}

pub fn reset_amp_bf16_runtime_reports() {
    CUDA_HOST_STAGING_EVENTS.with(|events| events.borrow_mut().clear());
    AMP_BF16_OP_DECISIONS.with(|decisions| decisions.borrow_mut().clear());
    AMP_BF16_FINITE_CHECKS.with(|checks| checks.borrow_mut().clear());
}

pub fn cuda_host_staging_events() -> Vec<CudaHostStagingEvent> {
    CUDA_HOST_STAGING_EVENTS.with(|events| events.borrow().clone())
}

pub fn record_amp_bf16_op_decision(decision: AmpBf16OpDecision) {
    AMP_BF16_OP_DECISIONS.with(|decisions| decisions.borrow_mut().push(decision));
}

pub fn amp_bf16_op_decisions() -> Vec<AmpBf16OpDecision> {
    AMP_BF16_OP_DECISIONS.with(|decisions| decisions.borrow().clone())
}

pub fn record_amp_bf16_finite_check(
    name: &str,
    target: &str,
    dtype: DType,
    device: Device,
    value: f64,
) -> Result<()> {
    if !is_amp_bf16_training() {
        return Ok(());
    }
    let passed = value.is_finite();
    AMP_BF16_FINITE_CHECKS.with(|checks| {
        checks.borrow_mut().push(AmpBf16FiniteCheckEvent {
            name: name.to_string(),
            target: target.to_string(),
            dtype: dtype_label(dtype),
            device: device_label(device),
            value: if passed {
                format!("{value:.9e}")
            } else {
                value.to_string()
            },
            passed,
        });
    });
    if !passed {
        return Err(TensorError::Autograd(format!(
            "amp-bf16 finite check failed for {name} on {target}; \
             dtype={} device={} value={value}",
            dtype_label(dtype),
            device_label(device)
        )));
    }
    Ok(())
}

pub fn amp_bf16_finite_check_events() -> Vec<AmpBf16FiniteCheckEvent> {
    AMP_BF16_FINITE_CHECKS.with(|checks| checks.borrow().clone())
}

pub fn amp_bf16_validation_report(policy: &AmpBf16Policy) -> AmpBf16ValidationReport {
    let enabled = is_amp_bf16_training() || !amp_bf16_op_decisions().is_empty();
    let decisions = amp_bf16_op_decisions();
    let host_staging_events = cuda_host_staging_events();
    let finite_checks = amp_bf16_finite_check_events();
    let unexpected_host_staging_count = host_staging_events
        .iter()
        .filter(|event| !event.allowed)
        .count();
    let allowed_host_staging_count = host_staging_events
        .iter()
        .filter(|event| event.allowed)
        .count();
    let tensor_core_decision_count = decisions
        .iter()
        .filter(|decision| decision.tensor_core)
        .count();
    let mut fallback_reasons = BTreeMap::new();
    for decision in decisions.iter().filter(|decision| !decision.tensor_core) {
        if let Some(reason) = &decision.fallback_reason {
            *fallback_reasons.entry(reason.clone()).or_insert(0) += 1;
        }
    }
    let fallback_decision_count = fallback_reasons.values().copied().sum();
    let finite_check_failure_count = finite_checks.iter().filter(|check| !check.passed).count();

    let mut notes = Vec::new();
    if !policy.dynamic_loss_scaling {
        notes.push(
            "bf16 path intentionally does not use fp16-style dynamic loss scaling".to_string(),
        );
    }
    if policy.strict_no_cpu_staging {
        notes.push("unexpected CUDA tensor host materialization is rejected in strict amp-bf16 training scopes".to_string());
    }

    let status = if !enabled {
        "not_applicable"
    } else if unexpected_host_staging_count == 0 && finite_check_failure_count == 0 {
        "passed"
    } else {
        "failed"
    }
    .to_string();

    AmpBf16ValidationReport {
        schema_version: 1,
        enabled,
        status,
        strict_no_cpu_staging: policy.strict_no_cpu_staging,
        dynamic_loss_scaling: policy.dynamic_loss_scaling,
        op_decision_count: decisions.len(),
        tensor_core_decision_count,
        fallback_decision_count,
        fallback_reasons,
        finite_check_count: finite_checks.len(),
        finite_check_failure_count,
        unexpected_host_staging_count,
        allowed_host_staging_count,
        notes,
    }
}

pub(crate) fn dtype_label(dtype: DType) -> String {
    match dtype {
        DType::F32 => "f32".to_string(),
        DType::BFloat16 => "bf16".to_string(),
        DType::F64 => "f64".to_string(),
        DType::I64 => "i64".to_string(),
        DType::Bool => "bool".to_string(),
    }
}

pub(crate) fn device_label(device: Device) -> String {
    match device {
        Device::Cpu => "cpu".to_string(),
        Device::Cuda(device_id) => format!("cuda:{device_id}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_amp_guard_rejects_unallowed_host_staging() {
        reset_amp_bf16_runtime_reports();
        let _guard = enter_amp_bf16_training();
        let err =
            record_cuda_host_staging("unit-test", DType::F32, Device::Cuda(0), 4).unwrap_err();
        assert!(err.to_string().contains("strict amp-bf16 training forbids"));
        let events = cuda_host_staging_events();
        assert_eq!(events.len(), 1);
        assert!(!events[0].allowed);
    }

    #[test]
    fn strict_amp_guard_allows_scoped_reporting_reads() {
        reset_amp_bf16_runtime_reports();
        let _guard = enter_amp_bf16_training();
        with_cuda_host_staging_allowed("loss scalar", || {
            record_cuda_host_staging("loss scalar", DType::F32, Device::Cuda(0), 1).unwrap();
        });
        let events = cuda_host_staging_events();
        assert_eq!(events.len(), 1);
        assert!(events[0].allowed);
    }

    #[test]
    fn amp_validation_report_summarizes_failures() {
        reset_amp_bf16_runtime_reports();
        let _guard = enter_amp_bf16_training();
        record_amp_bf16_op_decision(AmpBf16OpDecision {
            op: "linear:test".to_string(),
            input_dtype: "f32".to_string(),
            input_device: "cuda:0".to_string(),
            compute_dtype: "bf16".to_string(),
            accumulation_dtype: "f32".to_string(),
            output_dtype: "f32".to_string(),
            kernel_path: "cuda_tensor_core".to_string(),
            tensor_core: true,
            fallback_reason: None,
            finite_check: "loss_is_finite".to_string(),
        });
        record_cuda_host_staging("unexpected", DType::F32, Device::Cuda(0), 8).unwrap_err();

        let report = amp_bf16_validation_report(&AmpBf16Policy::cuda_training());

        assert_eq!(report.status, "failed");
        assert_eq!(report.op_decision_count, 1);
        assert_eq!(report.tensor_core_decision_count, 1);
        assert_eq!(report.unexpected_host_staging_count, 1);
    }

    #[test]
    fn finite_check_records_and_rejects_non_finite_values() {
        reset_amp_bf16_runtime_reports();
        let _guard = enter_amp_bf16_training();

        let err = record_amp_bf16_finite_check(
            "loss",
            "train_loss",
            DType::F32,
            Device::Cuda(0),
            f64::NAN,
        )
        .unwrap_err();

        assert!(err.to_string().contains("finite check failed"));
        let checks = amp_bf16_finite_check_events();
        assert_eq!(checks.len(), 1);
        assert!(!checks[0].passed);
    }
}
