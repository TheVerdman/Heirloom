use crate::error::SkillResult;
use crate::policy::{check_tool_policy, RuntimePlan};
use crate::router::SkillRouter;
use crate::runtime_context::build_runtime_context;
use crate::util::write_json_pretty;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalMetrics {
    pub cases: usize,
    pub top_1_routing_accuracy: f64,
    pub top_k_routing_accuracy: f64,
    pub constraint_recall: f64,
    pub average_context_size: f64,
    pub missing_validation_rate: f64,
    pub policy_conflict_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    pub format: String,
    pub version: u32,
    pub top_k: usize,
    pub metrics: EvalMetrics,
    pub cases: Vec<EvalCaseReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalCaseReport {
    pub skill: String,
    pub case_name: String,
    pub user_request: String,
    pub expected_skills: Vec<String>,
    pub activated_skills: Vec<String>,
    pub top_1_passed: bool,
    pub top_k_passed: bool,
    pub constraint_recall: f64,
    pub context_size: usize,
    pub missing_validation_steps: bool,
    pub policy_conflicts: usize,
}

pub fn run_eval(skills_dir: &Path, out: Option<&Path>) -> SkillResult<EvalReport> {
    let router = SkillRouter::from_dir(skills_dir)?;
    let skills = router.skills.clone();
    let top_k = 3;
    let mut cases = Vec::new();

    for skill in &skills {
        for eval_case in &skill.ir.eval_cases {
            let matches = router.route(&eval_case.user_request, top_k)?;
            let activated_skills = matches
                .iter()
                .map(|item| item.name.clone())
                .collect::<Vec<_>>();
            let context = build_runtime_context(&eval_case.user_request, &matches, &skills, 6000)?;
            let expected = if eval_case.expected_skills.is_empty() {
                vec![skill.manifest.name.clone()]
            } else {
                eval_case.expected_skills.clone()
            };
            let top_1_passed = matches
                .first()
                .map(|item| expected.contains(&item.name))
                .unwrap_or(false);
            let top_k_passed = matches.iter().any(|item| expected.contains(&item.name));
            let required_constraints = if eval_case.required_constraints.is_empty() {
                skill.ir.hard_constraints.clone()
            } else {
                eval_case.required_constraints.clone()
            };
            let recalled = required_constraints
                .iter()
                .filter(|constraint| context.rendered.contains(*constraint))
                .count();
            let constraint_recall = if required_constraints.is_empty() {
                1.0
            } else {
                recalled as f64 / required_constraints.len() as f64
            };
            let plan = RuntimePlan {
                user_request: eval_case.user_request.clone(),
                activated_skills: activated_skills.clone(),
                planned_tool_calls: Vec::new(),
            };
            let policy = check_tool_policy(&plan, &skills);
            let missing_validation_steps = context.validation_steps.is_empty();
            cases.push(EvalCaseReport {
                skill: skill.manifest.name.clone(),
                case_name: eval_case.name.clone(),
                user_request: eval_case.user_request.clone(),
                expected_skills: expected,
                activated_skills,
                top_1_passed,
                top_k_passed,
                constraint_recall,
                context_size: context.rendered.len(),
                missing_validation_steps,
                policy_conflicts: policy.errors.len(),
            });
        }
    }

    let count = cases.len();
    let metrics = if count == 0 {
        EvalMetrics {
            cases: 0,
            top_1_routing_accuracy: 0.0,
            top_k_routing_accuracy: 0.0,
            constraint_recall: 0.0,
            average_context_size: 0.0,
            missing_validation_rate: 0.0,
            policy_conflict_count: 0,
        }
    } else {
        EvalMetrics {
            cases: count,
            top_1_routing_accuracy: ratio(&cases, |case| case.top_1_passed),
            top_k_routing_accuracy: ratio(&cases, |case| case.top_k_passed),
            constraint_recall: cases.iter().map(|case| case.constraint_recall).sum::<f64>()
                / count as f64,
            average_context_size: cases
                .iter()
                .map(|case| case.context_size as f64)
                .sum::<f64>()
                / count as f64,
            missing_validation_rate: ratio(&cases, |case| case.missing_validation_steps),
            policy_conflict_count: cases.iter().map(|case| case.policy_conflicts).sum(),
        }
    };

    let report = EvalReport {
        format: "heirloom.skill_eval".to_string(),
        version: 1,
        top_k,
        metrics,
        cases,
    };
    if let Some(out) = out {
        write_json_pretty(out, &report)?;
    }
    Ok(report)
}

fn ratio(cases: &[EvalCaseReport], predicate: impl Fn(&EvalCaseReport) -> bool) -> f64 {
    cases.iter().filter(|case| predicate(case)).count() as f64 / cases.len() as f64
}
