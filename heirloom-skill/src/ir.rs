use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillIr {
    pub name: String,
    pub version: String,
    pub source_path: PathBuf,
    pub source_hash: String,
    pub description: Option<String>,
    pub triggers: Vec<String>,
    pub negative_triggers: Vec<String>,
    pub required_tools: Vec<String>,
    pub allowed_tools: Vec<String>,
    pub forbidden_tools: Vec<String>,
    pub preflight_steps: Vec<String>,
    pub execution_steps: Vec<String>,
    pub validation_steps: Vec<String>,
    pub failure_modes: Vec<String>,
    pub hard_constraints: Vec<String>,
    pub soft_guidelines: Vec<String>,
    pub examples: Vec<SkillExample>,
    pub eval_cases: Vec<SkillEvalCase>,
    pub metadata: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillExample {
    pub user_request: String,
    pub expected_behavior: String,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillEvalCase {
    pub name: String,
    pub user_request: String,
    pub expected_skills: Vec<String>,
    pub required_constraints: Vec<String>,
}
