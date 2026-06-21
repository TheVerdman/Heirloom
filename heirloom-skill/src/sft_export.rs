use crate::error::SkillResult;
use crate::trace::read_traces;
use crate::trace::SkillTrace;
use crate::trace_hygiene::audit_trace_hygiene;
use crate::util::{canonical_json_string, write_string};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SftExportSummary {
    pub read: usize,
    pub written: usize,
    pub skipped_failures: usize,
    pub skipped_ineligible: usize,
    pub skipped_hygiene: usize,
    pub hygiene_blocking_issues: usize,
    pub hygiene_warning_issues: usize,
}

pub fn export_sft_records(
    traces_path: &Path,
    out_path: &Path,
    include_failures: bool,
) -> SkillResult<SftExportSummary> {
    let traces = read_traces(traces_path)?;
    let mut lines = Vec::new();
    let mut skipped_failures = 0;
    let mut skipped_ineligible = 0;
    let mut skipped_hygiene = 0;
    let mut hygiene_blocking_issues = 0;
    let mut hygiene_warning_issues = 0;

    for trace in &traces {
        if !include_failures && !trace.success {
            skipped_failures += 1;
            continue;
        }
        if trace
            .metadata
            .get("export_eligible")
            .and_then(Value::as_bool)
            == Some(false)
        {
            skipped_ineligible += 1;
            continue;
        }
        let normalized_final_response = normalize_artifact_paths_for_export(trace);
        let mut hygiene_trace = trace.clone();
        hygiene_trace.final_response = normalized_final_response.clone();
        let hygiene = audit_trace_hygiene(&hygiene_trace);
        hygiene_blocking_issues += hygiene.blocking_issues;
        hygiene_warning_issues += hygiene.warning_issues;
        if !hygiene.export_allowed {
            skipped_hygiene += 1;
            continue;
        }
        let assistant = format!(
            "Activated skills: {}\nHard constraints: {}\n\n{}",
            trace.activated_skills.join(", "),
            trace.hard_constraints.join("; "),
            normalized_final_response
        );
        let record = serde_json::json!({
            "messages": [
                {
                    "role": "system",
                    "content": "You are operating with compiled Heirloom skill context. Follow symbolic hard constraints, tool policies, and validators."
                },
                {
                    "role": "user",
                    "content": trace.user_request
                },
                {
                    "role": "assistant",
                    "content": assistant
                }
            ],
            "metadata": {
                "activated_skills": trace.activated_skills,
                "source_trace_id": trace.trace_id,
                "success": trace.success
            }
        });
        lines.push(canonical_json_string(&record)?);
    }

    let mut text = lines.join("\n");
    if !text.is_empty() {
        text.push('\n');
    }
    write_string(out_path, &text)?;
    Ok(SftExportSummary {
        read: traces.len(),
        written: lines.len(),
        skipped_failures,
        skipped_ineligible,
        skipped_hygiene,
        hygiene_blocking_issues,
        hygiene_warning_issues,
    })
}

fn normalize_artifact_paths_for_export(trace: &SkillTrace) -> String {
    let mut text = trace.final_response.clone();
    for (index, path) in trace.artifact_paths.iter().enumerate() {
        let path_text = path.to_string_lossy();
        if path_text.is_empty() {
            continue;
        }
        let placeholder = format!("artifact://trace/{}/{}", trace.trace_id, index);
        text = text.replace(path_text.as_ref(), &placeholder);
    }
    text
}
