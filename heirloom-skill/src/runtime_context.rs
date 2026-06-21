use crate::embeddings::{cosine, Embedder, HashEmbedder};
use crate::error::{SkillError, SkillResult};
use crate::pack::CompiledSkill;
use crate::router::SkillMatch;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeContext {
    pub user_request: String,
    pub activated_skills: Vec<String>,
    pub relevant_triggers: Vec<String>,
    pub hard_constraints: Vec<String>,
    pub required_tools: Vec<String>,
    pub forbidden_tools: Vec<String>,
    pub preflight_steps: Vec<String>,
    pub validation_steps: Vec<String>,
    pub nearest_examples: Vec<String>,
    pub soft_guidelines: Vec<String>,
    pub rendered: String,
}

pub fn build_runtime_context(
    request: &str,
    matches: &[SkillMatch],
    skills: &[CompiledSkill],
    budget_chars: usize,
) -> SkillResult<RuntimeContext> {
    let mut selected = Vec::new();
    for item in matches {
        let skill = skills
            .iter()
            .find(|skill| skill.manifest.name == item.name)
            .ok_or_else(|| {
                SkillError::Invalid(format!("matched skill {} was not loaded", item.name))
            })?;
        selected.push(skill);
    }

    let activated_skills = selected
        .iter()
        .map(|skill| skill.manifest.name.clone())
        .collect::<Vec<_>>();
    let relevant_triggers = unique_flat_map(&selected, |skill| skill.ir.triggers.clone());
    let hard_constraints = unique_flat_map(&selected, |skill| skill.ir.hard_constraints.clone());
    let required_tools = unique_flat_map(&selected, |skill| skill.ir.required_tools.clone());
    let forbidden_tools = unique_flat_map(&selected, |skill| skill.ir.forbidden_tools.clone());
    let preflight_steps = unique_flat_map(&selected, |skill| skill.ir.preflight_steps.clone());
    let validation_steps = unique_flat_map(&selected, |skill| skill.ir.validation_steps.clone());
    let soft_guidelines = unique_flat_map(&selected, |skill| skill.ir.soft_guidelines.clone());
    let nearest_examples = nearest_examples(request, &selected, 3);

    let mut rendered = String::new();
    add_line(
        &mut rendered,
        budget_chars,
        "Compiled Heirloom skill context",
    );
    add_line(
        &mut rendered,
        budget_chars,
        &format!("User request: {request}"),
    );
    add_list(
        &mut rendered,
        budget_chars,
        "Activated skills",
        &activated_skills,
    );
    add_list(
        &mut rendered,
        budget_chars,
        "Hard constraints",
        &hard_constraints,
    );
    add_list(
        &mut rendered,
        budget_chars,
        "Required tools",
        &required_tools,
    );
    add_list(
        &mut rendered,
        budget_chars,
        "Forbidden tools",
        &forbidden_tools,
    );
    add_list(
        &mut rendered,
        budget_chars,
        "Validation steps",
        &validation_steps,
    );
    add_list(
        &mut rendered,
        budget_chars,
        "Preflight steps",
        &preflight_steps,
    );
    add_list(
        &mut rendered,
        budget_chars,
        "Relevant triggers",
        &relevant_triggers,
    );
    add_list(
        &mut rendered,
        budget_chars,
        "Nearest examples",
        &nearest_examples,
    );
    add_list(
        &mut rendered,
        budget_chars,
        "Soft guidelines",
        &soft_guidelines,
    );

    Ok(RuntimeContext {
        user_request: request.to_string(),
        activated_skills,
        relevant_triggers,
        hard_constraints,
        required_tools,
        forbidden_tools,
        preflight_steps,
        validation_steps,
        nearest_examples,
        soft_guidelines,
        rendered,
    })
}

fn unique_flat_map<F>(skills: &[&CompiledSkill], mut mapper: F) -> Vec<String>
where
    F: FnMut(&CompiledSkill) -> Vec<String>,
{
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for skill in skills {
        for item in mapper(skill) {
            let key = item.to_ascii_lowercase();
            if seen.insert(key) {
                out.push(item);
            }
        }
    }
    out
}

fn nearest_examples(request: &str, skills: &[&CompiledSkill], limit: usize) -> Vec<String> {
    let embedder = HashEmbedder::default();
    let request_embedding = embedder.embed(request);
    let mut scored = Vec::new();
    for skill in skills {
        for (index, example) in skill.ir.examples.iter().enumerate() {
            let Some(embedding) = skill.example_embeddings.get(index) else {
                continue;
            };
            scored.push((
                cosine(&request_embedding, embedding),
                format!(
                    "{}: user_request=\"{}\" expected_behavior=\"{}\"",
                    skill.manifest.name, example.user_request, example.expected_behavior
                ),
            ));
        }
    }
    scored.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored
        .into_iter()
        .take(limit)
        .map(|(_, example)| example)
        .collect()
}

fn add_list(rendered: &mut String, budget_chars: usize, title: &str, items: &[String]) {
    if items.is_empty() {
        return;
    }
    add_line(rendered, budget_chars, &format!("{title}:"));
    for item in items {
        add_line(rendered, budget_chars, &format!("- {item}"));
    }
}

fn add_line(rendered: &mut String, budget_chars: usize, line: &str) {
    if budget_chars == 0 {
        return;
    }
    let needed = line.len() + 1;
    if rendered.len() + needed <= budget_chars {
        rendered.push_str(line);
        rendered.push('\n');
    }
}
