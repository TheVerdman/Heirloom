use crate::embeddings::{cosine, mean_embedding, Embedder, HashEmbedder};
use crate::error::SkillResult;
use crate::pack::{load_compiled_skill, CompiledSkill};
use crate::registry::TrustTier;
use crate::util::{collect_files_with_extension, normalize_text};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillMatch {
    pub name: String,
    pub version: String,
    pub score: f32,
    pub path: String,
    pub trust_tier: TrustTier,
}

#[derive(Debug, Clone)]
pub struct SkillRouter {
    pub skills: Vec<CompiledSkill>,
    embedder: HashEmbedder,
}

impl SkillRouter {
    pub fn from_dir(path: impl AsRef<Path>) -> SkillResult<Self> {
        Self::from_dir_with_options(path, false)
    }

    pub fn from_dir_with_options(path: impl AsRef<Path>, trusted_only: bool) -> SkillResult<Self> {
        let mut skills = Vec::new();
        for file in collect_files_with_extension(path.as_ref(), "hskill")? {
            let skill = load_compiled_skill(&file)?;
            let tier = &skill.manifest.provenance.trust_tier;
            if !tier.is_routable() {
                continue;
            }
            if trusted_only && !tier.is_trusted() {
                continue;
            }
            skills.push(skill);
        }
        skills.sort_by(|left, right| left.manifest.name.cmp(&right.manifest.name));
        Ok(Self {
            skills,
            embedder: HashEmbedder::default(),
        })
    }

    pub fn route(&self, request: &str, top_k: usize) -> SkillResult<Vec<SkillMatch>> {
        let request_embedding = self.embedder.embed(request);
        let normalized_request = normalize_text(request);
        let mut matches = self
            .skills
            .iter()
            .map(|skill| {
                let score = score_skill(skill, &normalized_request, &request_embedding);
                SkillMatch {
                    name: skill.manifest.name.clone(),
                    version: skill.manifest.version.clone(),
                    score,
                    path: skill.path.to_string_lossy().to_string(),
                    trust_tier: skill.manifest.provenance.trust_tier.clone(),
                }
            })
            .collect::<Vec<_>>();
        matches.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.name.cmp(&right.name))
        });
        matches.truncate(top_k);
        Ok(matches)
    }
}

fn score_skill(skill: &CompiledSkill, normalized_request: &str, request_embedding: &[f32]) -> f32 {
    let phrase_boost = skill
        .ir
        .triggers
        .iter()
        .filter(|trigger| normalized_request.contains(&normalize_text(trigger)))
        .count() as f32
        * 0.45;
    let phrase_boost = phrase_boost.min(0.9);
    let negative_phrase_penalty = skill
        .ir
        .negative_triggers
        .iter()
        .filter(|trigger| normalized_request.contains(&normalize_text(trigger)))
        .count() as f32
        * 0.8;

    let dim = skill.manifest.embedding.dim;
    let trigger_score = max_cosine(request_embedding, &skill.trigger_embeddings);
    let example_score = max_cosine(request_embedding, &skill.example_embeddings);
    let section_score = skill
        .section_embeddings
        .iter()
        .map(|embedding| cosine(request_embedding, embedding))
        .fold(0.0f32, f32::max);
    let negative_vector = mean_embedding(&skill.negative_trigger_embeddings, dim);
    let negative_embedding_penalty = cosine(request_embedding, &negative_vector).max(0.0) * 0.35;

    let score = phrase_boost
        + trigger_score.max(0.0) * 0.5
        + example_score.max(0.0) * 0.3
        + section_score.max(0.0) * 0.2
        - negative_phrase_penalty
        - negative_embedding_penalty;
    score.clamp(-1.0, 1.5)
}

fn max_cosine(request_embedding: &[f32], candidates: &[Vec<f32>]) -> f32 {
    candidates
        .iter()
        .map(|embedding| cosine(request_embedding, embedding))
        .fold(0.0f32, f32::max)
}
