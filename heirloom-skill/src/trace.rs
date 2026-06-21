use crate::error::{SkillError, SkillResult};
use crate::policy::ToolCall;
use crate::trace_hygiene::{audit_trace_hygiene, hygiene_metadata_value, HYGIENE_METADATA_KEY};
use crate::util::{append_jsonl, read_jsonl};
use crate::validators::ValidationResult;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillTrace {
    pub trace_id: String,
    pub timestamp: String,
    pub user_request: String,
    pub activated_skills: Vec<String>,
    pub retrieved_sections: Vec<String>,
    pub hard_constraints: Vec<String>,
    pub planned_tool_calls: Vec<ToolCall>,
    pub validator_results: Vec<ValidationResult>,
    pub artifact_paths: Vec<PathBuf>,
    pub final_response: String,
    pub success: bool,
    pub metadata: Value,
}

pub fn write_trace(trace: &SkillTrace, path: &Path) -> SkillResult<()> {
    validate_trace_shape(trace)?;
    let report = audit_trace_hygiene(trace);
    let mut trace = trace.clone();
    let Some(metadata) = trace.metadata.as_object_mut() else {
        return Err(SkillError::Invalid(
            "trace metadata must be a JSON object".to_string(),
        ));
    };
    metadata.insert(
        HYGIENE_METADATA_KEY.to_string(),
        hygiene_metadata_value(&report)?,
    );
    append_jsonl(path, &trace)
}

pub fn read_traces(path: &Path) -> SkillResult<Vec<SkillTrace>> {
    read_jsonl(path)?
        .into_iter()
        .map(|value| serde_json::from_value(value).map_err(SkillError::from))
        .collect()
}

fn validate_trace_shape(trace: &SkillTrace) -> SkillResult<()> {
    if trace.trace_id.trim().is_empty() {
        return Err(SkillError::Invalid(
            "trace_id must be non-empty".to_string(),
        ));
    }
    if !trace
        .trace_id
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
    {
        return Err(SkillError::Invalid(format!(
            "trace_id contains unsupported characters: {}",
            trace.trace_id
        )));
    }
    if trace.timestamp.trim().is_empty() {
        return Err(SkillError::Invalid(
            "timestamp must be non-empty".to_string(),
        ));
    }
    if !trace.timestamp.contains('T') {
        return Err(SkillError::Invalid(format!(
            "timestamp must be RFC3339-like and contain T: {}",
            trace.timestamp
        )));
    }
    if !trace.metadata.is_object() {
        return Err(SkillError::Invalid(
            "trace metadata must be a JSON object".to_string(),
        ));
    }
    Ok(())
}
