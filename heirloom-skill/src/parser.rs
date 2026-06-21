use crate::error::{SkillError, SkillResult};
use crate::ir::{SkillEvalCase, SkillExample, SkillIr};
use crate::util::{read_bytes, sha256_prefixed_hex};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::path::Path;

pub fn parse_skill_markdown(path: &Path) -> SkillResult<SkillIr> {
    let bytes = read_bytes(path)?;
    let source_hash = sha256_prefixed_hex(&bytes);
    let text = String::from_utf8(bytes)
        .map_err(|err| SkillError::Parse(format!("{} is not UTF-8: {err}", path.display())))?;
    parse_skill_markdown_str(&text, path, source_hash)
}

pub fn parse_skill_markdown_str(
    text: &str,
    source_path: &Path,
    source_hash: String,
) -> SkillResult<SkillIr> {
    let mut name = String::new();
    let mut current = String::new();
    let mut sections: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for raw_line in text.lines() {
        let line = raw_line.trim_end();
        if let Some(heading) = line.strip_prefix("# ") {
            if name.is_empty() {
                name = heading
                    .trim()
                    .strip_prefix("Skill:")
                    .or_else(|| heading.trim().strip_prefix("Skill"))
                    .unwrap_or(heading.trim())
                    .trim_matches(':')
                    .trim()
                    .to_string();
            }
            continue;
        }
        if let Some(heading) = line.strip_prefix("## ") {
            current = normalize_heading(heading);
            sections.entry(current.clone()).or_default();
            continue;
        }
        if current.is_empty() {
            continue;
        }
        sections
            .entry(current.clone())
            .or_default()
            .push(line.to_string());
    }

    if name.is_empty() {
        name = first_nonempty(&sections, "name").unwrap_or_default();
    }

    let version = first_nonempty(&sections, "version").unwrap_or_default();
    let description = joined_optional(&sections, "description");
    let metadata = parse_metadata(sections.get("metadata").cloned().unwrap_or_default())?;

    Ok(SkillIr {
        name,
        version,
        source_path: source_path.to_path_buf(),
        source_hash,
        description,
        triggers: parse_list(section(&sections, "triggers")),
        negative_triggers: parse_list(section(&sections, "negativetriggers")),
        required_tools: parse_list(section(&sections, "requiredtools")),
        allowed_tools: parse_list(section(&sections, "allowedtools")),
        forbidden_tools: parse_list(section(&sections, "forbiddentools")),
        preflight_steps: parse_list(section(&sections, "preflightsteps")),
        execution_steps: parse_list(section(&sections, "executionsteps")),
        validation_steps: parse_list(section(&sections, "validationsteps")),
        failure_modes: parse_list(section(&sections, "failuremodes")),
        hard_constraints: parse_list(section(&sections, "hardconstraints")),
        soft_guidelines: parse_list(section(&sections, "softguidelines")),
        examples: parse_examples(section(&sections, "examples")),
        eval_cases: parse_eval_cases(section(&sections, "evalcases")),
        metadata,
    })
}

fn normalize_heading(heading: &str) -> String {
    heading
        .trim()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase()
}

fn section<'a>(sections: &'a BTreeMap<String, Vec<String>>, key: &str) -> &'a [String] {
    sections.get(key).map(Vec::as_slice).unwrap_or(&[])
}

fn first_nonempty(sections: &BTreeMap<String, Vec<String>>, key: &str) -> Option<String> {
    section(sections, key)
        .iter()
        .map(|line| line.trim())
        .find(|line| !line.is_empty())
        .map(ToString::to_string)
}

fn joined_optional(sections: &BTreeMap<String, Vec<String>>, key: &str) -> Option<String> {
    let text = section(sections, key)
        .iter()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn parse_list(lines: &[String]) -> Vec<String> {
    let bullet_items: Vec<String> = lines
        .iter()
        .filter_map(|line| {
            let trimmed = line.trim();
            strip_list_marker(trimmed).and_then(|item| {
                let item = item.trim();
                (!item.is_empty()).then(|| item.to_string())
            })
        })
        .collect();
    if !bullet_items.is_empty() {
        dedupe_preserve_order(bullet_items)
    } else {
        dedupe_preserve_order(
            lines
                .iter()
                .map(|line| line.trim())
                .filter(|line| !line.is_empty())
                .map(ToString::to_string)
                .collect(),
        )
    }
}

fn strip_list_marker(line: &str) -> Option<&str> {
    if let Some(rest) = line.strip_prefix("- ") {
        return Some(rest);
    }
    if let Some(rest) = line.strip_prefix("* ") {
        return Some(rest);
    }
    let (number, rest) = line.split_once(". ")?;
    if !number.is_empty() && number.chars().all(|ch| ch.is_ascii_digit()) {
        Some(rest)
    } else {
        None
    }
}

fn dedupe_preserve_order(items: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for item in items {
        let key = item.to_ascii_lowercase();
        if seen.insert(key) {
            out.push(item);
        }
    }
    out
}

fn parse_metadata(lines: Vec<String>) -> SkillResult<Value> {
    let text = lines
        .iter()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() {
        return Ok(Value::Object(Map::new()));
    }
    match serde_json::from_str::<Value>(&text) {
        Ok(Value::Object(map)) => Ok(Value::Object(map)),
        Ok(other) => {
            let mut map = Map::new();
            map.insert(
                "_raw_metadata".to_string(),
                Value::String(other.to_string()),
            );
            Ok(Value::Object(map))
        }
        Err(_) => {
            let mut map = Map::new();
            map.insert("_raw_metadata".to_string(), Value::String(text));
            Ok(Value::Object(map))
        }
    }
}

fn parse_examples(lines: &[String]) -> Vec<SkillExample> {
    parse_keyed_records(lines, &["user_request", "user"])
        .into_iter()
        .filter_map(|record| {
            let user_request = take_first(&record, &["user_request", "user"])?;
            let expected_behavior = take_first(&record, &["expected_behavior", "expected"])
                .unwrap_or_else(|| "Follow the skill constraints.".to_string());
            let notes = take_first(&record, &["notes", "note"]);
            Some(SkillExample {
                user_request,
                expected_behavior,
                notes,
            })
        })
        .collect()
}

fn parse_eval_cases(lines: &[String]) -> Vec<SkillEvalCase> {
    parse_keyed_records(lines, &["name"])
        .into_iter()
        .filter_map(|record| {
            let name = take_first(&record, &["name"])?;
            let user_request = take_first(&record, &["user_request", "user"])?;
            let expected_skills = take_list(&record, &["expected_skills", "expectedskills"]);
            let required_constraints =
                take_list(&record, &["required_constraints", "requiredconstraints"]);
            Some(SkillEvalCase {
                name,
                user_request,
                expected_skills,
                required_constraints,
            })
        })
        .collect()
}

fn parse_keyed_records(
    lines: &[String],
    record_start_keys: &[&str],
) -> Vec<BTreeMap<String, Vec<String>>> {
    let mut records = Vec::new();
    let mut current: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut current_key: Option<String> = None;

    for raw in lines {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let had_marker = strip_list_marker(trimmed).is_some();
        let line = strip_list_marker(trimmed).unwrap_or(trimmed).trim();
        if let Some((key, value)) = split_key_value(line) {
            let normalized = normalize_record_key(key);
            if had_marker && record_start_keys.contains(&normalized.as_str()) && !current.is_empty()
            {
                records.push(current);
                current = BTreeMap::new();
            }
            current
                .entry(normalized.clone())
                .or_default()
                .push(value.trim().trim_matches('"').to_string());
            current_key = Some(normalized);
        } else if let Some(key) = current_key.as_ref() {
            let value = line.trim().trim_matches('"');
            if !value.is_empty() {
                current
                    .entry(key.clone())
                    .or_default()
                    .push(value.to_string());
            }
        }
    }
    if !current.is_empty() {
        records.push(current);
    }
    records
}

fn split_key_value(line: &str) -> Option<(&str, &str)> {
    let (key, value) = line.split_once(':')?;
    if key.trim().is_empty() {
        None
    } else {
        Some((key.trim(), value.trim()))
    }
}

fn normalize_record_key(key: &str) -> String {
    key.chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
        .collect::<String>()
        .to_ascii_lowercase()
}

fn take_first(record: &BTreeMap<String, Vec<String>>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        record.get(*key).and_then(|values| {
            values
                .iter()
                .map(|value| value.trim())
                .find(|value| !value.is_empty())
                .map(ToString::to_string)
        })
    })
}

fn take_list(record: &BTreeMap<String, Vec<String>>, keys: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for key in keys {
        if let Some(values) = record.get(*key) {
            for value in values {
                for item in value.split(',') {
                    let item = item.trim().trim_matches('"');
                    if !item.is_empty() {
                        out.push(item.to_string());
                    }
                }
            }
        }
    }
    dedupe_preserve_order(out)
}
