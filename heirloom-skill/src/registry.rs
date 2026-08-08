use crate::error::{SkillError, SkillResult};
use crate::util::{read_to_string, write_json_pretty};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustTier {
    Untrusted,
    #[default]
    LocalDev,
    CommunityReviewed,
    HeirloomReviewed,
    HeirloomCore,
    Quarantined,
    Rejected,
}

impl TrustTier {
    pub fn is_routable(&self) -> bool {
        matches!(
            self,
            Self::LocalDev | Self::CommunityReviewed | Self::HeirloomReviewed | Self::HeirloomCore
        )
    }

    pub fn is_trusted(&self) -> bool {
        matches!(
            self,
            Self::CommunityReviewed | Self::HeirloomReviewed | Self::HeirloomCore
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditStatus {
    Unknown,
    #[default]
    Pending,
    Passed,
    Failed,
    NeedsReview,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillProvenance {
    pub source_url: Option<String>,
    pub source_commit: Option<String>,
    pub author: Option<String>,
    pub organization: Option<String>,
    pub license: Option<String>,
    pub license_status: Option<String>,
    pub adoption_signals: Vec<String>,
    pub trust_tier: TrustTier,
    pub audit_status: AuditStatus,
    pub notes: Option<String>,
}

impl Default for SkillProvenance {
    fn default() -> Self {
        Self {
            source_url: None,
            source_commit: None,
            author: None,
            organization: None,
            license: None,
            license_status: None,
            adoption_signals: Vec::new(),
            trust_tier: TrustTier::LocalDev,
            audit_status: AuditStatus::Pending,
            notes: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillRegistryEntry {
    pub skill_name: String,
    pub source_path: Option<String>,
    #[serde(flatten)]
    pub provenance: SkillProvenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillRegistry {
    pub entries: Vec<SkillRegistryEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistryLintReport {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl RegistryLintReport {
    pub fn passed(&self) -> bool {
        self.errors.is_empty()
    }
}

impl SkillRegistry {
    pub fn load(path: &Path) -> SkillResult<Self> {
        let text = read_to_string(path)?;
        serde_json::from_str(&text)
            .map_err(|err| SkillError::Json(format!("failed to parse {}: {err}", path.display())))
    }

    pub fn write(&self, path: &Path) -> SkillResult<()> {
        write_json_pretty(path, self)
    }

    pub fn find_entry(&self, skill_name: &str, source_path: &Path) -> Option<SkillRegistryEntry> {
        let source = source_path.to_string_lossy();
        self.entries
            .iter()
            .find(|entry| {
                entry.skill_name == skill_name
                    || entry
                        .source_path
                        .as_deref()
                        .map(|candidate| candidate == source)
                        .unwrap_or(false)
            })
            .cloned()
    }
}

pub fn lint_registry(registry: &SkillRegistry) -> RegistryLintReport {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for entry in &registry.entries {
        if entry.skill_name.trim().is_empty() {
            errors.push("registry entry has empty skill_name".to_string());
        }
        if !seen.insert(entry.skill_name.clone()) {
            errors.push(format!(
                "duplicate registry skill_name: {}",
                entry.skill_name
            ));
        }
        if entry.provenance.source_url.is_none()
            && entry.source_path.is_none()
            && entry.provenance.source_path_is_external()
        {
            warnings.push(format!(
                "{} has no source_url or source_path for reviewed provenance",
                entry.skill_name
            ));
        }
        if entry.provenance.license.is_none() && entry.provenance.trust_tier != TrustTier::LocalDev
        {
            warnings.push(format!("{} has no recorded license", entry.skill_name));
        }
        if entry.provenance.trust_tier.is_trusted()
            && entry.provenance.audit_status != AuditStatus::Passed
        {
            errors.push(format!(
                "{} is trusted but audit_status is not Passed",
                entry.skill_name
            ));
        }
        if matches!(
            entry.provenance.trust_tier,
            TrustTier::Rejected | TrustTier::Quarantined
        ) && entry.provenance.audit_status == AuditStatus::Passed
        {
            warnings.push(format!(
                "{} has audit_status Passed but trust tier is not routable",
                entry.skill_name
            ));
        }
    }
    RegistryLintReport { errors, warnings }
}

impl SkillProvenance {
    fn source_path_is_external(&self) -> bool {
        !matches!(self.trust_tier, TrustTier::LocalDev | TrustTier::Untrusted)
    }
}
