use crate::error::SkillResult;
use crate::pack::CompiledSkill;
use crate::util::{canonical_json_string, normalize_text, sha256_hex, write_string};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyntheticTask {
    pub task_id: String,
    pub user_request: String,
    pub expected_skills: Vec<String>,
    pub source_skill: String,
    pub difficulty: String,
}

pub fn generate_synthetic_tasks(skill: &CompiledSkill) -> Vec<SyntheticTask> {
    let mut tasks = Vec::new();
    let mut seen = BTreeSet::new();
    for trigger in &skill.ir.triggers {
        let request = if trigger.ends_with('.') || trigger.ends_with('?') {
            trigger.clone()
        } else {
            format!("Please {trigger}.")
        };
        push_task(skill, &mut tasks, &mut seen, request, "simple", None);
    }
    for example in &skill.ir.examples {
        push_task(
            skill,
            &mut tasks,
            &mut seen,
            example.user_request.clone(),
            "example",
            None,
        );
    }
    for eval in &skill.ir.eval_cases {
        push_task(
            skill,
            &mut tasks,
            &mut seen,
            eval.user_request.clone(),
            "eval",
            Some(eval.expected_skills.clone()),
        );
    }
    tasks
}

pub fn write_synthetic_tasks(skill: &CompiledSkill, out: &Path) -> SkillResult<Vec<SyntheticTask>> {
    let tasks = generate_synthetic_tasks(skill);
    let mut text = String::new();
    for task in &tasks {
        text.push_str(&canonical_json_string(task)?);
        text.push('\n');
    }
    write_string(out, &text)?;
    Ok(tasks)
}

fn push_task(
    skill: &CompiledSkill,
    tasks: &mut Vec<SyntheticTask>,
    seen: &mut BTreeSet<String>,
    user_request: String,
    difficulty: &str,
    expected_skills: Option<Vec<String>>,
) {
    let normalized = normalize_text(&user_request);
    if normalized.is_empty() || !seen.insert(normalized.clone()) {
        return;
    }
    let hash = sha256_hex(format!("{}:{normalized}", skill.manifest.name).as_bytes());
    tasks.push(SyntheticTask {
        task_id: format!("synthetic_{}", &hash[..16]),
        user_request,
        expected_skills: expected_skills.unwrap_or_else(|| vec![skill.manifest.name.clone()]),
        source_skill: skill.manifest.name.clone(),
        difficulty: difficulty.to_string(),
    });
}
