use crate::embeddings::{Embedder, HashEmbedder};
use crate::error::{SkillError, SkillResult};
use crate::ir::SkillIr;
use crate::lint::lint_skill;
use crate::pack::{
    build_payload_sections, payload_section_infos, write_compiled_skill, CompiledSkillManifest,
    CompiledSkillPayload, EmbeddingMetadata, SkillTextSection, HSKILL_FORMAT,
    HSKILL_FORMAT_VERSION,
};
use crate::parser::parse_skill_markdown;
use crate::registry::{SkillProvenance, SkillRegistry};
use std::path::{Path, PathBuf};

pub const COMPILER_VERSION: &str = "heirloom-skill/0.1.0";

#[derive(Debug, Clone, Default)]
pub struct CompileOptions {
    pub registry: Option<PathBuf>,
}

pub fn compile_skill(source: &Path, out: &Path) -> SkillResult<CompiledSkillManifest> {
    compile_skill_with_options(source, out, CompileOptions::default())
}

pub fn compile_skill_with_options(
    source: &Path,
    out: &Path,
    options: CompileOptions,
) -> SkillResult<CompiledSkillManifest> {
    let ir = parse_skill_markdown(source)?;
    let lint = lint_skill(&ir);
    if !lint.passed() {
        return Err(SkillError::Lint(lint.errors()));
    }
    let provenance = provenance_for(&ir, options.registry.as_deref())?;
    let embedder = HashEmbedder::default();
    let section_texts = build_section_texts(&ir);
    let trigger_embeddings = ir
        .triggers
        .iter()
        .map(|text| embedder.embed(text))
        .collect::<Vec<_>>();
    let negative_trigger_embeddings = ir
        .negative_triggers
        .iter()
        .map(|text| embedder.embed(text))
        .collect::<Vec<_>>();
    let section_embeddings = section_texts
        .iter()
        .map(|section| embedder.embed(&section.text))
        .collect::<Vec<_>>();
    let example_embeddings = ir
        .examples
        .iter()
        .map(|example| {
            embedder.embed(&format!(
                "{}\n{}",
                example.user_request, example.expected_behavior
            ))
        })
        .collect::<Vec<_>>();

    let payload_sections = build_payload_sections(
        &ir,
        &section_texts,
        &trigger_embeddings,
        &negative_trigger_embeddings,
        &section_embeddings,
        &example_embeddings,
        embedder.dim,
    )?;
    let manifest = CompiledSkillManifest {
        format: HSKILL_FORMAT.to_string(),
        format_version: HSKILL_FORMAT_VERSION,
        compiler_version: COMPILER_VERSION.to_string(),
        name: ir.name.clone(),
        version: ir.version.clone(),
        source_path: ir.source_path.to_string_lossy().to_string(),
        source_hash: ir.source_hash.clone(),
        provenance,
        embedding: EmbeddingMetadata {
            backend: "hash-bag-of-words-fnv1a64".to_string(),
            dim: embedder.dim,
        },
        trigger_count: ir.triggers.len(),
        negative_trigger_count: ir.negative_triggers.len(),
        section_count: section_texts.len(),
        example_count: ir.examples.len(),
        eval_case_count: ir.eval_cases.len(),
        payload_sections: payload_section_infos(&payload_sections),
    };

    write_compiled_skill(
        out,
        CompiledSkillPayload {
            manifest: &manifest,
            ir: &ir,
            section_texts: &section_texts,
            trigger_embeddings: &trigger_embeddings,
            negative_trigger_embeddings: &negative_trigger_embeddings,
            section_embeddings: &section_embeddings,
            example_embeddings: &example_embeddings,
        },
    )?;
    Ok(manifest)
}

pub fn build_section_texts(ir: &SkillIr) -> Vec<SkillTextSection> {
    let mut sections = Vec::new();
    push_optional(&mut sections, "description", ir.description.as_deref());
    push_list(&mut sections, "triggers", &ir.triggers);
    push_list(&mut sections, "negative_triggers", &ir.negative_triggers);
    push_list(&mut sections, "preflight_steps", &ir.preflight_steps);
    push_list(&mut sections, "execution_steps", &ir.execution_steps);
    push_list(&mut sections, "validation_steps", &ir.validation_steps);
    push_list(&mut sections, "failure_modes", &ir.failure_modes);
    push_list(&mut sections, "hard_constraints", &ir.hard_constraints);
    push_list(&mut sections, "soft_guidelines", &ir.soft_guidelines);
    if !ir.examples.is_empty() {
        sections.push(SkillTextSection {
            name: "examples".to_string(),
            text: ir
                .examples
                .iter()
                .map(|example| format!("{}\n{}", example.user_request, example.expected_behavior))
                .collect::<Vec<_>>()
                .join("\n\n"),
        });
    }
    if !ir.eval_cases.is_empty() {
        sections.push(SkillTextSection {
            name: "eval_cases".to_string(),
            text: ir
                .eval_cases
                .iter()
                .map(|case| {
                    format!(
                        "{}\n{}\n{}",
                        case.name,
                        case.user_request,
                        case.required_constraints.join("\n")
                    )
                })
                .collect::<Vec<_>>()
                .join("\n\n"),
        });
    }
    sections
}

fn push_optional(sections: &mut Vec<SkillTextSection>, name: &str, text: Option<&str>) {
    if let Some(text) = text {
        if !text.trim().is_empty() {
            sections.push(SkillTextSection {
                name: name.to_string(),
                text: text.trim().to_string(),
            });
        }
    }
}

fn push_list(sections: &mut Vec<SkillTextSection>, name: &str, items: &[String]) {
    if !items.is_empty() {
        sections.push(SkillTextSection {
            name: name.to_string(),
            text: items.join("\n"),
        });
    }
}

fn provenance_for(ir: &SkillIr, registry_path: Option<&Path>) -> SkillResult<SkillProvenance> {
    let Some(registry_path) = registry_path else {
        return Ok(SkillProvenance::default());
    };
    let registry = SkillRegistry::load(registry_path)?;
    Ok(registry
        .find_entry(&ir.name, &ir.source_path)
        .map(|entry| entry.provenance)
        .unwrap_or_default())
}
