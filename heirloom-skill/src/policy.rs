use crate::pack::CompiledSkill;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub tool_name: String,
    pub args: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimePlan {
    pub user_request: String,
    pub activated_skills: Vec<String>,
    pub planned_tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyResult {
    pub passed: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub metadata: Value,
}

pub fn check_tool_policy(plan: &RuntimePlan, skills: &[CompiledSkill]) -> PolicyResult {
    let active = skills
        .iter()
        .filter(|skill| plan.activated_skills.contains(&skill.manifest.name))
        .collect::<Vec<_>>();
    let planned_tools = plan
        .planned_tool_calls
        .iter()
        .map(|call| normalize_tool(&call.tool_name))
        .collect::<BTreeSet<_>>();

    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut required = BTreeSet::new();
    let mut forbidden = BTreeSet::new();

    for skill in &active {
        for tool in &skill.ir.required_tools {
            required.insert(normalize_tool(tool));
        }
        for tool in &skill.ir.forbidden_tools {
            forbidden.insert(normalize_tool(tool));
        }
    }

    for tool in required.intersection(&forbidden) {
        errors.push(format!(
            "conflicting policy across activated skills for tool {tool}"
        ));
    }
    for tool in &planned_tools {
        if forbidden.contains(tool) {
            errors.push(format!("forbidden tool use planned: {tool}"));
        }
    }
    for tool in &required {
        if !planned_tools.contains(tool) {
            warnings.push(format!("required tool not planned: {tool}"));
        }
    }

    PolicyResult {
        passed: errors.is_empty(),
        errors,
        warnings,
        metadata: serde_json::json!({
            "activated_skill_count": active.len(),
            "planned_tool_count": plan.planned_tool_calls.len(),
            "required_tool_count": required.len(),
            "forbidden_tool_count": forbidden.len()
        }),
    }
}

fn normalize_tool(tool: &str) -> String {
    tool.trim().to_ascii_lowercase()
}
