use crate::error::{SkillError, SkillResult};
use crate::ir::SkillIr;
use crate::registry::SkillProvenance;
use crate::util::{
    canonical_json_bytes, hex_bytes, read_bytes, sha256_digest, sha256_prefixed_hex, write_bytes,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const HSKILL_MAGIC: &[u8; 8] = b"HSKILL01";
pub const HSKILL_FORMAT: &str = "heirloom.hskill";
pub const HSKILL_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledSkillManifest {
    pub format: String,
    pub format_version: u32,
    pub compiler_version: String,
    pub name: String,
    pub version: String,
    pub source_path: String,
    pub source_hash: String,
    pub provenance: SkillProvenance,
    pub embedding: EmbeddingMetadata,
    pub trigger_count: usize,
    pub negative_trigger_count: usize,
    pub section_count: usize,
    pub example_count: usize,
    pub eval_case_count: usize,
    pub payload_sections: Vec<CompiledSectionInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledSectionInfo {
    pub name: String,
    pub bytes: usize,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingMetadata {
    pub backend: String,
    pub dim: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillTextSection {
    pub name: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledSkill {
    pub path: PathBuf,
    pub manifest: CompiledSkillManifest,
    pub ir: SkillIr,
    pub section_texts: Vec<SkillTextSection>,
    pub trigger_embeddings: Vec<Vec<f32>>,
    pub negative_trigger_embeddings: Vec<Vec<f32>>,
    pub section_embeddings: Vec<Vec<f32>>,
    pub example_embeddings: Vec<Vec<f32>>,
    pub artifact_checksum: String,
}

pub struct CompiledSkillPayload<'a> {
    pub manifest: &'a CompiledSkillManifest,
    pub ir: &'a SkillIr,
    pub section_texts: &'a [SkillTextSection],
    pub trigger_embeddings: &'a [Vec<f32>],
    pub negative_trigger_embeddings: &'a [Vec<f32>],
    pub section_embeddings: &'a [Vec<f32>],
    pub example_embeddings: &'a [Vec<f32>],
}

pub fn payload_section_infos(sections: &BTreeMap<String, Vec<u8>>) -> Vec<CompiledSectionInfo> {
    sections
        .iter()
        .map(|(name, bytes)| CompiledSectionInfo {
            name: name.clone(),
            bytes: bytes.len(),
            sha256: sha256_prefixed_hex(bytes),
        })
        .collect()
}

pub fn build_payload_sections(
    ir: &SkillIr,
    section_texts: &[SkillTextSection],
    trigger_embeddings: &[Vec<f32>],
    negative_trigger_embeddings: &[Vec<f32>],
    section_embeddings: &[Vec<f32>],
    example_embeddings: &[Vec<f32>],
    dim: usize,
) -> SkillResult<BTreeMap<String, Vec<u8>>> {
    let mut sections = BTreeMap::new();
    sections.insert("ir.json".to_string(), canonical_json_bytes(ir)?);
    sections.insert(
        "sections.json".to_string(),
        canonical_json_bytes(&section_texts)?,
    );
    sections.insert(
        "trigger_embeddings.f32".to_string(),
        encode_embedding_matrix(trigger_embeddings, dim)?,
    );
    sections.insert(
        "negative_trigger_embeddings.f32".to_string(),
        encode_embedding_matrix(negative_trigger_embeddings, dim)?,
    );
    sections.insert(
        "section_embeddings.f32".to_string(),
        encode_embedding_matrix(section_embeddings, dim)?,
    );
    sections.insert(
        "example_embeddings.f32".to_string(),
        encode_embedding_matrix(example_embeddings, dim)?,
    );
    Ok(sections)
}

pub fn write_compiled_skill(path: &Path, payload: CompiledSkillPayload<'_>) -> SkillResult<()> {
    let dim = payload.manifest.embedding.dim;
    let mut sections = build_payload_sections(
        payload.ir,
        payload.section_texts,
        payload.trigger_embeddings,
        payload.negative_trigger_embeddings,
        payload.section_embeddings,
        payload.example_embeddings,
        dim,
    )?;
    sections.insert(
        "manifest.json".to_string(),
        canonical_json_bytes(payload.manifest)?,
    );
    write_section_container(path, &sections)
}

pub fn load_compiled_skill(path: &Path) -> SkillResult<CompiledSkill> {
    let bytes = read_bytes(path)?;
    if bytes.len() < HSKILL_MAGIC.len() + 4 + 4 + 32 {
        return Err(SkillError::Format(format!(
            "{} is too short to be an .hskill file",
            path.display()
        )));
    }
    let payload_len = bytes.len() - 32;
    let expected = &bytes[payload_len..];
    let actual = sha256_digest(&bytes[..payload_len]);
    if actual.as_slice() != expected {
        return Err(SkillError::Checksum(format!(
            "{} checksum mismatch",
            path.display()
        )));
    }
    let artifact_checksum = format!("sha256:{}", hex_bytes(expected));
    let sections = read_section_container(&bytes[..payload_len])?;
    let manifest: CompiledSkillManifest = read_section_json(&sections, "manifest.json")?;
    if manifest.format != HSKILL_FORMAT {
        return Err(SkillError::Format(format!(
            "manifest format must be {HSKILL_FORMAT}, got {}",
            manifest.format
        )));
    }
    if manifest.format_version != HSKILL_FORMAT_VERSION {
        return Err(SkillError::Format(format!(
            "unsupported .hskill version {}",
            manifest.format_version
        )));
    }
    for info in &manifest.payload_sections {
        let bytes = sections.get(&info.name).ok_or_else(|| {
            SkillError::Format(format!("manifest references missing section {}", info.name))
        })?;
        if bytes.len() != info.bytes {
            return Err(SkillError::Format(format!(
                "section {} byte length mismatch",
                info.name
            )));
        }
        let actual = sha256_prefixed_hex(bytes);
        if actual != info.sha256 {
            return Err(SkillError::Checksum(format!(
                "section {} checksum mismatch",
                info.name
            )));
        }
    }

    let ir: SkillIr = read_section_json(&sections, "ir.json")?;
    let section_texts: Vec<SkillTextSection> = read_section_json(&sections, "sections.json")?;
    let dim = manifest.embedding.dim;
    Ok(CompiledSkill {
        path: path.to_path_buf(),
        manifest,
        ir,
        section_texts,
        trigger_embeddings: decode_embedding_matrix(
            section_bytes(&sections, "trigger_embeddings.f32")?,
            dim,
        )?,
        negative_trigger_embeddings: decode_embedding_matrix(
            section_bytes(&sections, "negative_trigger_embeddings.f32")?,
            dim,
        )?,
        section_embeddings: decode_embedding_matrix(
            section_bytes(&sections, "section_embeddings.f32")?,
            dim,
        )?,
        example_embeddings: decode_embedding_matrix(
            section_bytes(&sections, "example_embeddings.f32")?,
            dim,
        )?,
        artifact_checksum,
    })
}

fn write_section_container(path: &Path, sections: &BTreeMap<String, Vec<u8>>) -> SkillResult<()> {
    let mut out = Vec::new();
    out.extend_from_slice(HSKILL_MAGIC);
    out.extend_from_slice(&HSKILL_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(sections.len() as u32).to_le_bytes());
    for (name, bytes) in sections {
        let name_bytes = name.as_bytes();
        if name_bytes.len() > u16::MAX as usize {
            return Err(SkillError::Format(format!("section name too long: {name}")));
        }
        out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    let checksum = sha256_digest(&out);
    out.extend_from_slice(&checksum);
    write_bytes(path, &out)
}

fn read_section_container(bytes: &[u8]) -> SkillResult<BTreeMap<String, Vec<u8>>> {
    let mut cursor = Cursor::new(bytes);
    let magic = cursor.read_exact(HSKILL_MAGIC.len())?;
    if magic != HSKILL_MAGIC {
        return Err(SkillError::Format(
            "invalid .hskill magic header".to_string(),
        ));
    }
    let version = cursor.read_u32()?;
    if version != HSKILL_FORMAT_VERSION {
        return Err(SkillError::Format(format!(
            "unsupported .hskill container version {version}"
        )));
    }
    let count = cursor.read_u32()? as usize;
    let mut sections = BTreeMap::new();
    for _ in 0..count {
        let name_len = cursor.read_u16()? as usize;
        let name = String::from_utf8(cursor.read_exact(name_len)?.to_vec())
            .map_err(|err| SkillError::Format(format!("section name is not UTF-8: {err}")))?;
        let data_len = cursor.read_u64()? as usize;
        let data = cursor.read_exact(data_len)?.to_vec();
        if sections.insert(name.clone(), data).is_some() {
            return Err(SkillError::Format(format!("duplicate section {name}")));
        }
    }
    if !cursor.is_done() {
        return Err(SkillError::Format(
            "trailing bytes before .hskill checksum".to_string(),
        ));
    }
    Ok(sections)
}

fn read_section_json<T: for<'de> Deserialize<'de>>(
    sections: &BTreeMap<String, Vec<u8>>,
    name: &str,
) -> SkillResult<T> {
    let bytes = section_bytes(sections, name)?;
    serde_json::from_slice(bytes)
        .map_err(|err| SkillError::Json(format!("failed to parse section {name}: {err}")))
}

fn section_bytes<'a>(sections: &'a BTreeMap<String, Vec<u8>>, name: &str) -> SkillResult<&'a [u8]> {
    sections
        .get(name)
        .map(Vec::as_slice)
        .ok_or_else(|| SkillError::Format(format!("missing section {name}")))
}

pub fn encode_embedding_matrix(vectors: &[Vec<f32>], dim: usize) -> SkillResult<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(dim as u32).to_le_bytes());
    bytes.extend_from_slice(&(vectors.len() as u32).to_le_bytes());
    for (index, vector) in vectors.iter().enumerate() {
        if vector.len() != dim {
            return Err(SkillError::Format(format!(
                "embedding row {index} has len {}, expected {dim}",
                vector.len()
            )));
        }
        for value in vector {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(bytes)
}

pub fn decode_embedding_matrix(bytes: &[u8], expected_dim: usize) -> SkillResult<Vec<Vec<f32>>> {
    let mut cursor = Cursor::new(bytes);
    let dim = cursor.read_u32()? as usize;
    if dim != expected_dim {
        return Err(SkillError::Format(format!(
            "embedding dim mismatch: {dim} != {expected_dim}"
        )));
    }
    let count = cursor.read_u32()? as usize;
    let mut rows = Vec::with_capacity(count);
    for _ in 0..count {
        let mut row = Vec::with_capacity(dim);
        for _ in 0..dim {
            row.push(f32::from_le_bytes(cursor.read_array::<4>()?));
        }
        rows.push(row);
    }
    if !cursor.is_done() {
        return Err(SkillError::Format(
            "trailing bytes in embedding matrix".to_string(),
        ));
    }
    Ok(rows)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_exact(&mut self, len: usize) -> SkillResult<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| SkillError::Format("section length overflow".to_string()))?;
        if end > self.bytes.len() {
            return Err(SkillError::Format("unexpected end of .hskill".to_string()));
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn read_array<const N: usize>(&mut self) -> SkillResult<[u8; N]> {
        let bytes = self.read_exact(N)?;
        let mut out = [0u8; N];
        out.copy_from_slice(bytes);
        Ok(out)
    }

    fn read_u16(&mut self) -> SkillResult<u16> {
        Ok(u16::from_le_bytes(self.read_array()?))
    }

    fn read_u32(&mut self) -> SkillResult<u32> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    fn read_u64(&mut self) -> SkillResult<u64> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    fn is_done(&self) -> bool {
        self.offset == self.bytes.len()
    }
}
