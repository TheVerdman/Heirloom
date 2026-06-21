use crate::ir::SkillIr;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LintLevel {
    Error,
    Warning,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LintIssue {
    pub level: LintLevel,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LintReport {
    pub issues: Vec<LintIssue>,
}

impl LintReport {
    pub fn errors(&self) -> Vec<String> {
        self.issues
            .iter()
            .filter(|issue| issue.level == LintLevel::Error)
            .map(|issue| issue.message.clone())
            .collect()
    }

    pub fn warnings(&self) -> Vec<String> {
        self.issues
            .iter()
            .filter(|issue| issue.level == LintLevel::Warning)
            .map(|issue| issue.message.clone())
            .collect()
    }

    pub fn passed(&self) -> bool {
        self.errors().is_empty()
    }
}

pub fn lint_skill(ir: &SkillIr) -> LintReport {
    let mut issues = Vec::new();
    push_if_empty(&mut issues, LintLevel::Error, &ir.name, "Missing name");
    push_if_empty(
        &mut issues,
        LintLevel::Error,
        &ir.version,
        "Missing version",
    );
    push_if_vec_empty(
        &mut issues,
        LintLevel::Error,
        &ir.triggers,
        "Missing triggers",
    );
    push_if_vec_empty(
        &mut issues,
        LintLevel::Error,
        &ir.validation_steps,
        "Missing validation steps",
    );
    push_if_vec_empty(
        &mut issues,
        LintLevel::Error,
        &ir.hard_constraints,
        "Missing hard constraints",
    );
    push_if_empty(
        &mut issues,
        LintLevel::Error,
        &ir.source_hash,
        "Missing source hash",
    );

    let allowed: BTreeSet<_> = normalized_set(&ir.allowed_tools);
    let required: BTreeSet<_> = normalized_set(&ir.required_tools);
    let forbidden: BTreeSet<_> = normalized_set(&ir.forbidden_tools);
    for tool in allowed.intersection(&forbidden) {
        issues.push(error(format!(
            "Conflicting tool policy: {tool} is both allowed and forbidden"
        )));
    }
    for tool in required.intersection(&forbidden) {
        issues.push(error(format!(
            "Conflicting tool policy: {tool} is both required and forbidden"
        )));
    }

    for constraint in &ir.hard_constraints {
        if is_vague_constraint(constraint) {
            issues.push(warning(format!("Vague hard constraint: {constraint}")));
        }
    }
    if ir.examples.is_empty() {
        issues.push(warning("Empty examples".to_string()));
    }
    for (index, example) in ir.examples.iter().enumerate() {
        if example.user_request.trim().is_empty() || example.expected_behavior.trim().is_empty() {
            issues.push(warning(format!("Example {index} is incomplete")));
        }
    }
    for (index, case) in ir.eval_cases.iter().enumerate() {
        if case.name.trim().is_empty()
            || case.user_request.trim().is_empty()
            || case.expected_skills.is_empty()
        {
            issues.push(error(format!("Invalid eval case shape at index {index}")));
        }
    }

    LintReport { issues }
}

fn push_if_empty(issues: &mut Vec<LintIssue>, level: LintLevel, value: &str, message: &str) {
    if value.trim().is_empty() {
        issues.push(LintIssue {
            level,
            message: message.to_string(),
        });
    }
}

fn push_if_vec_empty(
    issues: &mut Vec<LintIssue>,
    level: LintLevel,
    value: &[String],
    message: &str,
) {
    if value.is_empty() {
        issues.push(LintIssue {
            level,
            message: message.to_string(),
        });
    }
}

fn normalized_set(items: &[String]) -> BTreeSet<String> {
    items
        .iter()
        .map(|item| item.trim().to_ascii_lowercase())
        .filter(|item| !item.is_empty())
        .collect()
}

fn is_vague_constraint(constraint: &str) -> bool {
    let lower = constraint.to_ascii_lowercase();
    let words = lower.split_whitespace().count();
    words < 4
        || lower.contains("be careful")
        || lower.contains("do well")
        || lower.contains("as appropriate")
        || lower.contains("try to")
        || lower == "validate"
}

fn error(message: String) -> LintIssue {
    LintIssue {
        level: LintLevel::Error,
        message,
    }
}

fn warning(message: String) -> LintIssue {
    LintIssue {
        level: LintLevel::Warning,
        message,
    }
}
