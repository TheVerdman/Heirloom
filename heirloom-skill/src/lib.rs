#![forbid(unsafe_code)]

pub mod compiler;
pub mod embeddings;
pub mod error;
pub mod eval;
pub mod ir;
pub mod lint;
pub mod pack;
pub mod parser;
pub mod policy;
pub mod registry;
pub mod router;
pub mod runtime_context;
pub mod sft_export;
pub mod synthetic_tasks;
pub mod trace;
pub mod trace_hygiene;
pub mod util;
pub mod validators;

pub use compiler::{compile_skill, compile_skill_with_options, CompileOptions};
pub use embeddings::{Embedder, HashEmbedder};
pub use error::{SkillError, SkillResult};
pub use eval::{run_eval, EvalMetrics, EvalReport};
pub use ir::{SkillEvalCase, SkillExample, SkillIr};
pub use lint::{lint_skill, LintIssue, LintLevel, LintReport};
pub use pack::{
    load_compiled_skill, CompiledSkill, CompiledSkillManifest, EmbeddingMetadata, SkillTextSection,
};
pub use parser::{parse_skill_markdown, parse_skill_markdown_str};
pub use policy::{check_tool_policy, PolicyResult, RuntimePlan, ToolCall};
pub use registry::{
    lint_registry, AuditStatus, RegistryLintReport, SkillProvenance, SkillRegistry,
    SkillRegistryEntry, TrustTier,
};
pub use router::{SkillMatch, SkillRouter};
pub use runtime_context::{build_runtime_context, RuntimeContext};
pub use sft_export::{export_sft_records, SftExportSummary};
pub use synthetic_tasks::{generate_synthetic_tasks, write_synthetic_tasks, SyntheticTask};
pub use trace::{read_traces, write_trace, SkillTrace};
pub use trace_hygiene::{
    audit_trace_hygiene, audit_traces, trace_export_allowed, write_trace_audit, HygieneKind,
    HygieneSeverity, TraceAuditSummary, TraceHygieneIssue, TraceHygieneReport,
};
pub use validators::{
    validate_artifact_link_present, validate_expected_skills, validate_file_exists,
    validate_file_extension, validate_hard_constraints_in_context, validate_json_valid,
    ValidationResult,
};
