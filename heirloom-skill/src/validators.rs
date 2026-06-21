use crate::pack::CompiledSkill;
use crate::runtime_context::RuntimeContext;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationResult {
    pub passed: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub metadata: Value,
}

impl ValidationResult {
    pub fn passed(metadata: Value) -> Self {
        Self {
            passed: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            metadata,
        }
    }

    pub fn failed(error: impl Into<String>, metadata: Value) -> Self {
        Self {
            passed: false,
            errors: vec![error.into()],
            warnings: Vec::new(),
            metadata,
        }
    }
}

pub fn validate_file_exists(path: &Path) -> ValidationResult {
    if path.exists() {
        ValidationResult::passed(serde_json::json!({ "path": path.to_string_lossy() }))
    } else {
        ValidationResult::failed(
            format!("file does not exist: {}", path.display()),
            serde_json::json!({ "path": path.to_string_lossy() }),
        )
    }
}

pub fn validate_file_extension(path: &Path, expected_extension: &str) -> ValidationResult {
    let expected = expected_extension.trim_start_matches('.');
    let actual = path
        .extension()
        .and_then(|item| item.to_str())
        .unwrap_or("");
    if actual == expected {
        ValidationResult::passed(serde_json::json!({
            "path": path.to_string_lossy(),
            "expected_extension": expected,
            "actual_extension": actual
        }))
    } else {
        ValidationResult::failed(
            format!(
                "file extension mismatch for {}: expected {expected}, got {actual}",
                path.display()
            ),
            serde_json::json!({
                "path": path.to_string_lossy(),
                "expected_extension": expected,
                "actual_extension": actual
            }),
        )
    }
}

pub fn validate_json_valid(path: &Path) -> ValidationResult {
    match std::fs::read_to_string(path) {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(_) => {
                ValidationResult::passed(serde_json::json!({ "path": path.to_string_lossy() }))
            }
            Err(err) => ValidationResult::failed(
                format!("invalid JSON in {}: {err}", path.display()),
                serde_json::json!({ "path": path.to_string_lossy() }),
            ),
        },
        Err(err) => ValidationResult::failed(
            format!("failed to read {}: {err}", path.display()),
            serde_json::json!({ "path": path.to_string_lossy() }),
        ),
    }
}

pub fn validate_artifact_link_present(
    final_response: &str,
    artifact_paths: &[PathBuf],
) -> ValidationResult {
    if artifact_paths.is_empty() {
        return ValidationResult::passed(serde_json::json!({ "artifact_count": 0 }));
    }
    let contains_known_path = artifact_paths
        .iter()
        .any(|path| final_response.contains(&path.to_string_lossy().to_string()));
    let contains_link_syntax = final_response.contains("](")
        || final_response.contains("artifact://")
        || final_response.contains("sandbox:");
    if contains_known_path || contains_link_syntax {
        ValidationResult::passed(serde_json::json!({ "artifact_count": artifact_paths.len() }))
    } else {
        ValidationResult::failed(
            "final response does not contain a file/sandbox/artifact link",
            serde_json::json!({ "artifact_count": artifact_paths.len() }),
        )
    }
}

pub fn validate_hard_constraints_in_context(
    context: &RuntimeContext,
    skills: &[CompiledSkill],
) -> ValidationResult {
    let mut missing = Vec::new();
    for skill in skills {
        if !context.activated_skills.contains(&skill.manifest.name) {
            continue;
        }
        for constraint in &skill.ir.hard_constraints {
            if !context.hard_constraints.contains(constraint)
                && !context.rendered.contains(constraint)
            {
                missing.push(format!("{}: {constraint}", skill.manifest.name));
            }
        }
    }
    if missing.is_empty() {
        ValidationResult::passed(serde_json::json!({
            "activated_skills": context.activated_skills
        }))
    } else {
        ValidationResult::failed(
            format!(
                "runtime context is missing hard constraints: {}",
                missing.join("; ")
            ),
            serde_json::json!({ "missing": missing }),
        )
    }
}

pub fn validate_expected_skills(
    activated_skills: &[String],
    expected_skills: &[String],
) -> ValidationResult {
    let missing = expected_skills
        .iter()
        .filter(|expected| !activated_skills.iter().any(|item| item == *expected))
        .cloned()
        .collect::<Vec<_>>();
    if missing.is_empty() {
        ValidationResult::passed(serde_json::json!({
            "activated_skills": activated_skills,
            "expected_skills": expected_skills
        }))
    } else {
        ValidationResult::failed(
            format!("missing expected activated skills: {}", missing.join(", ")),
            serde_json::json!({ "missing": missing }),
        )
    }
}
