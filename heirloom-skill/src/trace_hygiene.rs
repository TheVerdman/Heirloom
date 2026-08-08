use crate::error::SkillResult;
use crate::trace::{read_traces, SkillTrace};
use crate::util::{sha256_prefixed_hex, write_json_pretty};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

pub const HYGIENE_FORMAT: &str = "heirloom.skill_trace_hygiene";
pub const HYGIENE_VERSION: u32 = 1;
pub const HYGIENE_METADATA_KEY: &str = "hygiene_v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HygieneSeverity {
    Blocking,
    Warning,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HygieneKind {
    ApiKey,
    GitHubToken,
    OpenAiToken,
    AwsAccessKey,
    PrivateKeyBlock,
    EnvFileReference,
    SecretAssignment,
    EmailAddress,
    PhoneNumber,
    LocalAbsolutePath,
    UserHomePath,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceHygieneIssue {
    pub field_path: String,
    pub kind: HygieneKind,
    pub severity: HygieneSeverity,
    pub message: String,
    pub fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceHygieneReport {
    pub format: String,
    pub version: u32,
    pub trace_id: String,
    pub export_allowed: bool,
    pub blocking_issues: usize,
    pub warning_issues: usize,
    pub issues: Vec<TraceHygieneIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceAuditSummary {
    pub format: String,
    pub version: u32,
    pub traces: usize,
    pub export_allowed: usize,
    pub blocking_issues: usize,
    pub warning_issues: usize,
    pub reports: Vec<TraceHygieneReport>,
}

pub fn audit_trace_hygiene(trace: &SkillTrace) -> TraceHygieneReport {
    let mut issues = Vec::new();
    scan_text(&trace.user_request, "user_request", &mut issues);
    for (index, skill) in trace.activated_skills.iter().enumerate() {
        scan_text(skill, &format!("activated_skills[{index}]"), &mut issues);
    }
    for (index, section) in trace.retrieved_sections.iter().enumerate() {
        scan_text(
            section,
            &format!("retrieved_sections[{index}]"),
            &mut issues,
        );
    }
    for (index, constraint) in trace.hard_constraints.iter().enumerate() {
        scan_text(
            constraint,
            &format!("hard_constraints[{index}]"),
            &mut issues,
        );
    }
    for (index, call) in trace.planned_tool_calls.iter().enumerate() {
        scan_text(
            &call.tool_name,
            &format!("planned_tool_calls[{index}].tool_name"),
            &mut issues,
        );
        scan_json_value(
            &call.args,
            &format!("planned_tool_calls[{index}].args"),
            &mut issues,
        );
    }
    for (index, result) in trace.validator_results.iter().enumerate() {
        for (error_index, error) in result.errors.iter().enumerate() {
            scan_text(
                error,
                &format!("validator_results[{index}].errors[{error_index}]"),
                &mut issues,
            );
        }
        for (warning_index, warning) in result.warnings.iter().enumerate() {
            scan_text(
                warning,
                &format!("validator_results[{index}].warnings[{warning_index}]"),
                &mut issues,
            );
        }
        scan_json_value(
            &result.metadata,
            &format!("validator_results[{index}].metadata"),
            &mut issues,
        );
    }
    for (index, path) in trace.artifact_paths.iter().enumerate() {
        scan_text(
            &path.to_string_lossy(),
            &format!("artifact_paths[{index}]"),
            &mut issues,
        );
    }
    scan_text(&trace.final_response, "final_response", &mut issues);
    scan_trace_metadata(&trace.metadata, &mut issues);

    issues.sort_by(|left, right| {
        left.field_path
            .cmp(&right.field_path)
            .then_with(|| format!("{:?}", left.kind).cmp(&format!("{:?}", right.kind)))
            .then_with(|| left.fingerprint.cmp(&right.fingerprint))
    });
    issues.dedup();

    let blocking_issues = issues
        .iter()
        .filter(|issue| issue.severity == HygieneSeverity::Blocking)
        .count();
    let warning_issues = issues
        .iter()
        .filter(|issue| issue.severity == HygieneSeverity::Warning)
        .count();
    TraceHygieneReport {
        format: HYGIENE_FORMAT.to_string(),
        version: HYGIENE_VERSION,
        trace_id: trace.trace_id.clone(),
        export_allowed: blocking_issues == 0,
        blocking_issues,
        warning_issues,
        issues,
    }
}

pub fn audit_traces(path: &Path) -> SkillResult<TraceAuditSummary> {
    let traces = read_traces(path)?;
    let reports = traces.iter().map(audit_trace_hygiene).collect::<Vec<_>>();
    Ok(summarize_reports(reports))
}

pub fn write_trace_audit(path: &Path, out: &Path) -> SkillResult<TraceAuditSummary> {
    let summary = audit_traces(path)?;
    write_json_pretty(out, &summary)?;
    Ok(summary)
}

pub fn trace_export_allowed(trace: &SkillTrace) -> bool {
    audit_trace_hygiene(trace).export_allowed
}

pub fn hygiene_metadata_value(report: &TraceHygieneReport) -> SkillResult<Value> {
    serde_json::to_value(report).map_err(crate::SkillError::from)
}

fn summarize_reports(reports: Vec<TraceHygieneReport>) -> TraceAuditSummary {
    TraceAuditSummary {
        format: HYGIENE_FORMAT.to_string(),
        version: HYGIENE_VERSION,
        traces: reports.len(),
        export_allowed: reports
            .iter()
            .filter(|report| report.export_allowed)
            .count(),
        blocking_issues: reports.iter().map(|report| report.blocking_issues).sum(),
        warning_issues: reports.iter().map(|report| report.warning_issues).sum(),
        reports,
    }
}

fn scan_trace_metadata(metadata: &Value, issues: &mut Vec<TraceHygieneIssue>) {
    match metadata {
        Value::Object(map) => {
            for (key, value) in map {
                if key == HYGIENE_METADATA_KEY {
                    continue;
                }
                scan_json_value(value, &format!("metadata.{key}"), issues);
            }
        }
        other => scan_json_value(other, "metadata", issues),
    }
}

fn scan_json_value(value: &Value, field_path: &str, issues: &mut Vec<TraceHygieneIssue>) {
    match value {
        Value::String(text) => scan_text(text, field_path, issues),
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                scan_json_value(item, &format!("{field_path}[{index}]"), issues);
            }
        }
        Value::Object(map) => {
            for (key, item) in map {
                scan_json_value(item, &format!("{field_path}.{key}"), issues);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn scan_text(text: &str, field_path: &str, issues: &mut Vec<TraceHygieneIssue>) {
    if text.trim().is_empty() {
        return;
    }
    scan_private_key(text, field_path, issues);
    scan_dot_env(text, field_path, issues);
    scan_secret_assignments(text, field_path, issues);
    scan_tokens(text, field_path, issues);
    scan_warning_tokens(text, field_path, issues);
}

fn scan_private_key(text: &str, field_path: &str, issues: &mut Vec<TraceHygieneIssue>) {
    if text.contains("-----BEGIN ") && text.contains("PRIVATE KEY-----") {
        add_issue(
            issues,
            field_path,
            HygieneKind::PrivateKeyBlock,
            HygieneSeverity::Blocking,
            "private key block detected",
            "private-key-block",
        );
    }
}

fn scan_dot_env(text: &str, field_path: &str, issues: &mut Vec<TraceHygieneIssue>) {
    for token in split_scan_tokens(text) {
        let lower = token.to_ascii_lowercase();
        if lower == ".env"
            || lower.ends_with("/.env")
            || lower.contains(".env.")
            || lower.contains("/.env/")
        {
            add_issue(
                issues,
                field_path,
                HygieneKind::EnvFileReference,
                HygieneSeverity::Blocking,
                ".env reference detected",
                token,
            );
        }
    }
}

fn scan_secret_assignments(text: &str, field_path: &str, issues: &mut Vec<TraceHygieneIssue>) {
    for key in ["password", "token", "secret", "api_key"] {
        for marker in [format!("{key}="), format!("{key}:")] {
            let lower = text.to_ascii_lowercase();
            let mut offset = 0;
            while let Some(index) = lower[offset..].find(&marker) {
                let start = offset + index;
                let end = assignment_end(text, start + marker.len());
                let matched = &text[start..end];
                if matched.len() > marker.len() {
                    add_issue(
                        issues,
                        field_path,
                        HygieneKind::SecretAssignment,
                        HygieneSeverity::Blocking,
                        "secret assignment detected",
                        matched,
                    );
                }
                offset = (start + marker.len()).min(text.len());
            }
        }
    }
}

fn scan_tokens(text: &str, field_path: &str, issues: &mut Vec<TraceHygieneIssue>) {
    for token in split_scan_tokens(text) {
        let trimmed = token.trim_matches(|ch: char| {
            matches!(
                ch,
                '"' | '\'' | '`' | ',' | ';' | ')' | '(' | '[' | ']' | '{' | '}'
            )
        });
        if looks_like_openai_token(trimmed) {
            add_issue(
                issues,
                field_path,
                HygieneKind::OpenAiToken,
                HygieneSeverity::Blocking,
                "OpenAI-style token detected",
                trimmed,
            );
        } else if looks_like_github_token(trimmed) {
            add_issue(
                issues,
                field_path,
                HygieneKind::GitHubToken,
                HygieneSeverity::Blocking,
                "GitHub-style token detected",
                trimmed,
            );
        } else if looks_like_aws_access_key(trimmed) {
            add_issue(
                issues,
                field_path,
                HygieneKind::AwsAccessKey,
                HygieneSeverity::Blocking,
                "AWS access key detected",
                trimmed,
            );
        } else if looks_like_generic_api_key(trimmed) {
            add_issue(
                issues,
                field_path,
                HygieneKind::ApiKey,
                HygieneSeverity::Blocking,
                "API key-like token detected",
                trimmed,
            );
        }
    }
}

fn scan_warning_tokens(text: &str, field_path: &str, issues: &mut Vec<TraceHygieneIssue>) {
    for token in split_scan_tokens(text) {
        let trimmed = token.trim_matches(|ch: char| {
            matches!(
                ch,
                '"' | '\'' | '`' | ',' | ';' | ')' | '(' | '[' | ']' | '{' | '}'
            )
        });
        if looks_like_email(trimmed) {
            add_issue(
                issues,
                field_path,
                HygieneKind::EmailAddress,
                HygieneSeverity::Warning,
                "email-like string detected",
                trimmed,
            );
        }
        if looks_like_phone(trimmed) {
            add_issue(
                issues,
                field_path,
                HygieneKind::PhoneNumber,
                HygieneSeverity::Warning,
                "phone-like digit sequence detected",
                trimmed,
            );
        }
        if looks_like_user_home_path(trimmed) {
            add_issue(
                issues,
                field_path,
                HygieneKind::UserHomePath,
                HygieneSeverity::Warning,
                "user-home path detected",
                trimmed,
            );
        } else if looks_like_local_absolute_path(trimmed) {
            add_issue(
                issues,
                field_path,
                HygieneKind::LocalAbsolutePath,
                HygieneSeverity::Warning,
                "local absolute path detected",
                trimmed,
            );
        }
    }
}

fn split_scan_tokens(text: &str) -> Vec<&str> {
    text.split(|ch: char| ch.is_whitespace() || ch == '<' || ch == '>')
        .filter(|token| !token.trim().is_empty())
        .collect()
}

fn assignment_end(text: &str, value_start: usize) -> usize {
    let mut end = value_start;
    for (offset, ch) in text[value_start..].char_indices() {
        if ch.is_whitespace() || matches!(ch, '&' | ',' | ';' | '"' | '\'' | '`') {
            break;
        }
        end = value_start + offset + ch.len_utf8();
    }
    end
}

fn looks_like_openai_token(token: &str) -> bool {
    token.starts_with("sk-")
        && token.len() >= 18
        && token
            .chars()
            .skip(3)
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
}

fn looks_like_github_token(token: &str) -> bool {
    (token.starts_with("ghp_")
        || token.starts_with("gho_")
        || token.starts_with("ghu_")
        || token.starts_with("ghs_")
        || token.starts_with("ghr_"))
        && token.len() >= 20
        && token
            .chars()
            .skip(4)
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        || token.starts_with("github_pat_") && token.len() >= 30
}

fn looks_like_aws_access_key(token: &str) -> bool {
    token.len() == 20
        && (token.starts_with("AKIA") || token.starts_with("ASIA"))
        && token
            .chars()
            .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit())
}

fn looks_like_generic_api_key(token: &str) -> bool {
    token.len() >= 32
        && token.chars().any(|ch| ch.is_ascii_alphabetic())
        && token.chars().any(|ch| ch.is_ascii_digit())
        && token
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
        && (token.to_ascii_lowercase().contains("key")
            || token.to_ascii_lowercase().contains("api"))
}

fn looks_like_email(token: &str) -> bool {
    let Some((left, right)) = token.split_once('@') else {
        return false;
    };
    !left.is_empty()
        && right.contains('.')
        && right.len() >= 3
        && token
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '@' | '.' | '_' | '-' | '+'))
}

fn looks_like_phone(token: &str) -> bool {
    let digits = token.chars().filter(|ch| ch.is_ascii_digit()).count();
    (10..=16).contains(&digits)
        && token
            .chars()
            .all(|ch| ch.is_ascii_digit() || matches!(ch, '+' | '-' | '(' | ')' | '.'))
}

fn looks_like_user_home_path(token: &str) -> bool {
    token.starts_with("~/") || token.starts_with("/Users/")
}

fn looks_like_local_absolute_path(token: &str) -> bool {
    token.starts_with('/')
        && !token.starts_with("http://")
        && !token.starts_with("https://")
        && token.len() > 1
}

fn add_issue(
    issues: &mut Vec<TraceHygieneIssue>,
    field_path: &str,
    kind: HygieneKind,
    severity: HygieneSeverity,
    message: &str,
    matched: &str,
) {
    issues.push(TraceHygieneIssue {
        field_path: field_path.to_string(),
        kind,
        severity,
        message: message.to_string(),
        fingerprint: sha256_prefixed_hex(matched.as_bytes()),
    });
}
