use heirloom_skill::{
    audit_trace_hygiene, audit_traces, build_runtime_context, check_tool_policy, compile_skill,
    export_sft_records, lint_registry, lint_skill, load_compiled_skill, parse_skill_markdown,
    read_traces, run_eval, validate_expected_skills, validate_file_exists, validate_file_extension,
    validate_json_valid, write_trace, CompileOptions, RuntimePlan, SkillRegistry, SkillTrace,
    ToolCall, TrustTier, ValidationResult,
};
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};

fn temp_root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("heirloom-skill-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    root
}

fn write_skill(dir: &Path, name: &str, extra: &str) -> PathBuf {
    let path = dir.join(name).join("SKILL.md");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(
        &path,
        format!(
            r#"# Skill: {name}

## Version
1.0.0

## Description
Skill for {name}.

## Triggers
- turn this CSV into a styled Excel dashboard
- spreadsheet report

## Negative Triggers
- send an email

## Required Tools
- spreadsheet

## Allowed Tools
- spreadsheet
- filesystem

## Forbidden Tools
- email-send

## Preflight Steps
- Inspect the source file before changing it.

## Execution Steps
- Use structured spreadsheet operations.

## Hard Constraints
- Use spreadsheet-native tooling.
- Validate generated files before returning.
- Include a downloadable file link when producing an artifact.

## Soft Guidelines
- Keep the workbook easy to scan.

## Validation Steps
- Verify the output workbook exists on disk.
- Reopen the workbook after writing.

## Failure Modes
- Missing source file.

## Examples
- user_request: Turn this CSV into a styled Excel dashboard.
  expected_behavior: Create and validate an XLSX artifact.
  notes: Return a file link.

## Eval Cases
- name: spreadsheet_route
  user_request: Turn this CSV into a styled Excel dashboard.
  expected_skills: {name}
  required_constraints:
    - Use spreadsheet-native tooling.
    - Validate generated files before returning.

## Metadata
{{"test": true}}

{extra}
"#
        ),
    )
    .unwrap();
    path
}

#[test]
fn parser_extracts_sections_examples_eval_and_metadata() {
    let root = temp_root("parser");
    let source = write_skill(&root, "spreadsheets", "");
    let ir = parse_skill_markdown(&source).unwrap();
    assert_eq!(ir.name, "spreadsheets");
    assert_eq!(ir.version, "1.0.0");
    assert!(ir
        .triggers
        .contains(&"turn this CSV into a styled Excel dashboard".to_string()));
    assert_eq!(ir.examples.len(), 1);
    assert_eq!(ir.eval_cases.len(), 1);
    assert_eq!(ir.metadata["test"], true);
}

#[test]
fn linter_catches_missing_constraints_and_tool_conflicts() {
    let root = temp_root("lint");
    let source = root.join("SKILL.md");
    fs::write(
        &source,
        r#"# Skill: bad

## Version
1

## Triggers
- bad task

## Allowed Tools
- shell

## Forbidden Tools
- shell

## Validation Steps
- Validate output.
"#,
    )
    .unwrap();
    let ir = parse_skill_markdown(&source).unwrap();
    let lint = lint_skill(&ir);
    let errors = lint.errors().join("\n");
    assert!(errors.contains("Missing hard constraints"));
    assert!(errors.contains("both allowed and forbidden"));
}

#[test]
fn compiler_writes_loadable_deterministic_hskill_and_tracks_source_hash() {
    let root = temp_root("compiler");
    let source = write_skill(&root, "spreadsheets", "");
    let out1 = root.join("one.hskill");
    let out2 = root.join("two.hskill");
    let manifest = compile_skill(&source, &out1).unwrap();
    compile_skill(&source, &out2).unwrap();
    let loaded = load_compiled_skill(&out1).unwrap();
    assert_eq!(loaded.manifest.name, "spreadsheets");
    assert_eq!(loaded.manifest.source_hash, manifest.source_hash);
    assert_eq!(fs::read(&out1).unwrap(), fs::read(&out2).unwrap());

    fs::write(
        &source,
        fs::read_to_string(&source).unwrap() + "\n<!-- changed -->\n",
    )
    .unwrap();
    let out3 = root.join("three.hskill");
    let changed = compile_skill(&source, &out3).unwrap();
    assert_ne!(manifest.source_hash, changed.source_hash);
}

#[test]
fn loader_rejects_checksum_mutation() {
    let root = temp_root("checksum");
    let source = write_skill(&root, "spreadsheets", "");
    let out = root.join("skill.hskill");
    compile_skill(&source, &out).unwrap();
    let mut bytes = fs::read(&out).unwrap();
    let index = bytes.len() / 2;
    bytes[index] ^= 0x7f;
    let bad = root.join("bad.hskill");
    fs::write(&bad, bytes).unwrap();
    let err = load_compiled_skill(&bad).unwrap_err();
    assert!(err.to_string().contains("checksum"));
}

#[test]
fn router_selects_spreadsheet_and_negative_triggers_reduce_score() {
    let root = temp_root("router");
    let source = write_skill(&root, "spreadsheets", "");
    let skill_dir = root.join("skills");
    fs::create_dir_all(&skill_dir).unwrap();
    compile_skill(&source, &skill_dir.join("spreadsheets.hskill")).unwrap();
    let router = heirloom_skill::SkillRouter::from_dir(&skill_dir).unwrap();
    let matches = router
        .route("Turn this CSV into a styled Excel dashboard", 3)
        .unwrap();
    assert_eq!(matches[0].name, "spreadsheets");
    let positive = matches[0].score;
    let negative = router
        .route(
            "Turn this CSV into a styled Excel dashboard and send an email",
            3,
        )
        .unwrap()[0]
        .score;
    assert!(negative < positive);
}

#[test]
fn context_respects_budget_and_contains_required_slices() {
    let root = temp_root("context");
    let source = write_skill(&root, "spreadsheets", "");
    let skill_dir = root.join("skills");
    fs::create_dir_all(&skill_dir).unwrap();
    compile_skill(&source, &skill_dir.join("spreadsheets.hskill")).unwrap();
    let router = heirloom_skill::SkillRouter::from_dir(&skill_dir).unwrap();
    let matches = router.route("spreadsheet report", 1).unwrap();
    let context =
        build_runtime_context("spreadsheet report", &matches, &router.skills, 600).unwrap();
    assert!(context.rendered.len() <= 600);
    assert!(context
        .hard_constraints
        .contains(&"Use spreadsheet-native tooling.".to_string()));
    assert!(context.validation_steps.len() >= 2);
}

#[test]
fn policy_checker_catches_forbidden_tool_use() {
    let root = temp_root("policy");
    let source = write_skill(&root, "spreadsheets", "");
    let out = root.join("spreadsheets.hskill");
    compile_skill(&source, &out).unwrap();
    let skill = load_compiled_skill(&out).unwrap();
    let plan = RuntimePlan {
        user_request: "spreadsheet report".to_string(),
        activated_skills: vec!["spreadsheets".to_string()],
        planned_tool_calls: vec![ToolCall {
            tool_name: "email-send".to_string(),
            args: json!({}),
        }],
    };
    let result = check_tool_policy(&plan, &[skill]);
    assert!(!result.passed);
    assert!(result.errors.join("\n").contains("forbidden"));
}

#[test]
fn validators_cover_files_json_links_constraints_and_expected_skills() {
    let root = temp_root("validators");
    let missing = root.join("missing.xlsx");
    assert!(!validate_file_exists(&missing).passed);
    let json_path = root.join("ok.json");
    fs::write(&json_path, "{}").unwrap();
    assert!(validate_file_exists(&json_path).passed);
    assert!(validate_file_extension(&json_path, "json").passed);
    assert!(validate_json_valid(&json_path).passed);
    assert!(
        validate_expected_skills(&["spreadsheets".to_string()], &["spreadsheets".to_string()])
            .passed
    );
}

#[test]
fn trace_writer_jsonl_and_sft_export_excludes_failures_by_default() {
    let root = temp_root("trace");
    let traces = root.join("traces.jsonl");
    let success = sample_trace("trace_success", true);
    let failure = sample_trace("trace_failure", false);
    write_trace(&success, &traces).unwrap();
    write_trace(&failure, &traces).unwrap();
    let loaded = read_traces(&traces).unwrap();
    assert_eq!(loaded.len(), 2);
    assert!(loaded[0].metadata.get("hygiene_v1").is_some());

    let out = root.join("sft.jsonl");
    let summary = export_sft_records(&traces, &out, false).unwrap();
    assert_eq!(summary.read, 2);
    assert_eq!(summary.written, 1);
    assert_eq!(fs::read_to_string(&out).unwrap().lines().count(), 1);
    let exported = fs::read_to_string(&out).unwrap();
    assert!(!exported.contains("/tmp/report.xlsx"));
    assert!(exported.contains("artifact://trace/trace_success/0"));

    let out_with_failures = root.join("sft_failures.jsonl");
    let summary = export_sft_records(&traces, &out_with_failures, true).unwrap();
    assert_eq!(summary.written, 2);
}

#[test]
fn hygiene_blocks_secrets_but_warning_only_traces_still_export() {
    let root = temp_root("hygiene-export");
    let traces = root.join("traces.jsonl");

    let mut clean_warning = sample_trace("trace_warning", true);
    clean_warning.user_request = "Email maya@example.com about the spreadsheet.".to_string();
    write_trace(&clean_warning, &traces).unwrap();

    let mut blocked_user = sample_trace("trace_blocked_user", true);
    blocked_user.user_request =
        "Use this token sk-1234567890abcdefghi in the dashboard.".to_string();
    write_trace(&blocked_user, &traces).unwrap();

    let mut blocked_final = sample_trace("trace_blocked_final", true);
    blocked_final.final_response = "Saved report with password=swordfish.".to_string();
    write_trace(&blocked_final, &traces).unwrap();

    let mut blocked_args = sample_trace("trace_blocked_args", false);
    blocked_args.planned_tool_calls = vec![ToolCall {
        tool_name: "filesystem".to_string(),
        args: json!({"token": "ghp_1234567890abcdefghi"}),
    }];
    write_trace(&blocked_args, &traces).unwrap();

    let mut ineligible = sample_trace("trace_ineligible", true);
    ineligible.metadata = json!({"export_eligible": false});
    write_trace(&ineligible, &traces).unwrap();

    let out = root.join("sft.jsonl");
    let summary = export_sft_records(&traces, &out, false).unwrap();
    assert_eq!(summary.read, 5);
    assert_eq!(summary.written, 1);
    assert_eq!(summary.skipped_hygiene, 2);
    assert_eq!(summary.skipped_failures, 1);
    assert_eq!(summary.skipped_ineligible, 1);
    assert!(summary.hygiene_blocking_issues >= 2);
    assert!(summary.hygiene_warning_issues >= 1);

    let include_failures = root.join("sft_include_failures.jsonl");
    let summary = export_sft_records(&traces, &include_failures, true).unwrap();
    assert_eq!(summary.written, 1);
    assert_eq!(summary.skipped_hygiene, 3);
}

#[test]
fn trace_audit_uses_fingerprints_without_raw_secret_excerpts() {
    let root = temp_root("hygiene-audit");
    let traces = root.join("traces.jsonl");
    let mut blocked = sample_trace("trace_secret", true);
    blocked.final_response = "The secret is password=swordfish.".to_string();
    write_trace(&blocked, &traces).unwrap();

    let summary = audit_traces(&traces).unwrap();
    assert_eq!(summary.traces, 1);
    assert_eq!(summary.export_allowed, 0);
    assert!(summary.blocking_issues >= 1);
    let text = serde_json::to_string(&summary).unwrap();
    assert!(!text.contains("swordfish"));
    assert!(text.contains("sha256:"));

    let report = audit_trace_hygiene(&blocked);
    assert!(!report.export_allowed);
    assert!(report
        .issues
        .iter()
        .all(|issue| issue.fingerprint.starts_with("sha256:")));
}

#[test]
fn malformed_trace_metadata_is_rejected_at_write_time() {
    let root = temp_root("hygiene-shape");
    let traces = root.join("traces.jsonl");
    let mut trace = sample_trace("trace_bad_metadata", true);
    trace.metadata = json!("not an object");
    let err = write_trace(&trace, &traces).unwrap_err();
    assert!(err.to_string().contains("metadata must be a JSON object"));
}

#[test]
fn synthetic_tasks_and_eval_report_are_produced() {
    let root = temp_root("eval");
    let source = write_skill(&root, "spreadsheets", "");
    let skill_dir = root.join("skills");
    fs::create_dir_all(&skill_dir).unwrap();
    let compiled = skill_dir.join("spreadsheets.hskill");
    compile_skill(&source, &compiled).unwrap();
    let skill = load_compiled_skill(&compiled).unwrap();
    let tasks = heirloom_skill::generate_synthetic_tasks(&skill);
    assert!(tasks.iter().any(|task| task.source_skill == "spreadsheets"));

    let report_path = root.join("skill_eval.json");
    let report = run_eval(&skill_dir, Some(&report_path)).unwrap();
    assert_eq!(report.metrics.cases, 1);
    assert!(report.metrics.top_k_routing_accuracy > 0.0);
    assert!(report_path.exists());
}

#[test]
fn registry_lint_and_trusted_only_routing_use_provenance() {
    let root = temp_root("registry");
    let source = write_skill(&root, "spreadsheets", "");
    let registry_path = root.join("registry.json");
    fs::write(
        &registry_path,
        format!(
            r#"{{
  "entries": [
    {{
      "skill_name": "spreadsheets",
      "source_path": "{}",
      "source_url": null,
      "source_commit": null,
      "author": "Heirloom",
      "organization": "Heirloom",
      "license": "MIT OR Apache-2.0",
      "license_status": "project_local",
      "adoption_signals": ["test"],
      "trust_tier": "HeirloomCore",
      "audit_status": "Passed",
      "notes": "test registry"
    }}
  ]
}}"#,
            source.to_string_lossy()
        ),
    )
    .unwrap();
    let registry = SkillRegistry::load(&registry_path).unwrap();
    assert!(lint_registry(&registry).passed());
    let skill_dir = root.join("skills");
    fs::create_dir_all(&skill_dir).unwrap();
    heirloom_skill::compile_skill_with_options(
        &source,
        &skill_dir.join("spreadsheets.hskill"),
        CompileOptions {
            registry: Some(registry_path),
        },
    )
    .unwrap();
    let router = heirloom_skill::SkillRouter::from_dir_with_options(&skill_dir, true).unwrap();
    assert_eq!(router.skills.len(), 1);
    assert_eq!(
        router.skills[0].manifest.provenance.trust_tier,
        TrustTier::HeirloomCore
    );
}

fn sample_trace(trace_id: &str, success: bool) -> SkillTrace {
    SkillTrace {
        trace_id: trace_id.to_string(),
        timestamp: "2026-06-19T00:00:00Z".to_string(),
        user_request: "Turn this CSV into a styled Excel dashboard.".to_string(),
        activated_skills: vec!["spreadsheets".to_string()],
        retrieved_sections: vec!["hard_constraints".to_string()],
        hard_constraints: vec!["Validate generated files before returning.".to_string()],
        planned_tool_calls: Vec::new(),
        validator_results: vec![ValidationResult::passed(json!({}))],
        artifact_paths: vec![PathBuf::from("/tmp/report.xlsx")],
        final_response: "Created [report](/tmp/report.xlsx).".to_string(),
        success,
        metadata: json!({ "export_eligible": true }),
    }
}
