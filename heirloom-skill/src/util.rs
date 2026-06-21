use crate::error::{SkillError, SkillResult};
use serde::Serialize;
use serde_json::{Map, Value};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

pub fn read_to_string(path: &Path) -> SkillResult<String> {
    fs::read_to_string(path)
        .map_err(|err| SkillError::Io(format!("failed to read {}: {err}", path.display())))
}

pub fn read_bytes(path: &Path) -> SkillResult<Vec<u8>> {
    fs::read(path)
        .map_err(|err| SkillError::Io(format!("failed to read {}: {err}", path.display())))
}

pub fn write_bytes(path: &Path, bytes: &[u8]) -> SkillResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            SkillError::Io(format!("failed to create {}: {err}", parent.display()))
        })?;
    }
    fs::write(path, bytes)
        .map_err(|err| SkillError::Io(format!("failed to write {}: {err}", path.display())))
}

pub fn write_string(path: &Path, text: &str) -> SkillResult<()> {
    write_bytes(path, text.as_bytes())
}

pub fn append_jsonl<T: Serialize>(path: &Path, value: &T) -> SkillResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            SkillError::Io(format!("failed to create {}: {err}", parent.display()))
        })?;
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| SkillError::Io(format!("failed to open {}: {err}", path.display())))?;
    let line = canonical_json_string(value)?;
    file.write_all(line.as_bytes())
        .and_then(|_| file.write_all(b"\n"))
        .map_err(|err| SkillError::Io(format!("failed to write {}: {err}", path.display())))
}

pub fn read_jsonl(path: &Path) -> SkillResult<Vec<Value>> {
    let file = fs::File::open(path)
        .map_err(|err| SkillError::Io(format!("failed to open {}: {err}", path.display())))?;
    let mut records = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line
            .map_err(|err| SkillError::Io(format!("failed to read {}: {err}", path.display())))?;
        if line.trim().is_empty() {
            continue;
        }
        let value = serde_json::from_str::<Value>(&line).map_err(|err| {
            SkillError::Json(format!(
                "{}:{} invalid JSONL: {err}",
                path.display(),
                index + 1
            ))
        })?;
        records.push(value);
    }
    Ok(records)
}

pub fn canonical_json_value(value: Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.into_iter().map(canonical_json_value).collect()),
        Value::Object(map) => {
            let mut sorted = Map::new();
            let mut keys: Vec<_> = map.keys().cloned().collect();
            keys.sort();
            for key in keys {
                if let Some(value) = map.get(&key) {
                    sorted.insert(key, canonical_json_value(value.clone()));
                }
            }
            Value::Object(sorted)
        }
        other => other,
    }
}

pub fn canonical_json_bytes<T: Serialize>(value: &T) -> SkillResult<Vec<u8>> {
    let value = serde_json::to_value(value)?;
    let value = canonical_json_value(value);
    serde_json::to_vec(&value).map_err(SkillError::from)
}

pub fn canonical_json_string<T: Serialize>(value: &T) -> SkillResult<String> {
    let bytes = canonical_json_bytes(value)?;
    String::from_utf8(bytes).map_err(|err| SkillError::Json(err.to_string()))
}

pub fn pretty_json_string<T: Serialize>(value: &T) -> SkillResult<String> {
    let value = serde_json::to_value(value)?;
    let value = canonical_json_value(value);
    serde_json::to_string_pretty(&value).map_err(SkillError::from)
}

pub fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> SkillResult<()> {
    let mut text = pretty_json_string(value)?;
    text.push('\n');
    write_string(path, &text)
}

pub fn collect_files_with_extension(root: &Path, extension: &str) -> SkillResult<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_files_with_extension_inner(root, extension.trim_start_matches('.'), &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_files_with_extension_inner(
    root: &Path,
    extension: &str,
    files: &mut Vec<PathBuf>,
) -> SkillResult<()> {
    if root.is_file() {
        if root.extension().and_then(|item| item.to_str()) == Some(extension) {
            files.push(root.to_path_buf());
        }
        return Ok(());
    }
    let entries = fs::read_dir(root).map_err(|err| {
        SkillError::Io(format!(
            "failed to read directory {}: {err}",
            root.display()
        ))
    })?;
    for entry in entries {
        let entry = entry.map_err(|err| {
            SkillError::Io(format!(
                "failed to read directory entry in {}: {err}",
                root.display()
            ))
        })?;
        let path = entry.path();
        if path.is_dir() {
            collect_files_with_extension_inner(&path, extension, files)?;
        } else if path.extension().and_then(|item| item.to_str()) == Some(extension) {
            files.push(path);
        }
    }
    Ok(())
}

pub fn normalize_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub fn sha256_prefixed_hex(bytes: &[u8]) -> String {
    format!("sha256:{}", hex_bytes(&sha256_digest(bytes)))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex_bytes(&sha256_digest(bytes))
}

pub fn hex_bytes(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}

pub fn sha256_digest(bytes: &[u8]) -> [u8; 32] {
    const H0: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4f, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut message = bytes.to_vec();
    let bit_len = (message.len() as u64) * 8;
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    let mut h = H0;
    for chunk in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (index, word) in w.iter_mut().take(16).enumerate() {
            let offset = index * 4;
            *word = u32::from_be_bytes([
                chunk[offset],
                chunk[offset + 1],
                chunk[offset + 2],
                chunk[offset + 3],
            ]);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }
        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut digest = [0u8; 32];
    for (index, word) in h.iter().enumerate() {
        digest[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}
