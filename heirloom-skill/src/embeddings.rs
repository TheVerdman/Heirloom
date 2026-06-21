use crate::util::fnv1a64;

pub trait Embedder {
    fn embed(&self, text: &str) -> Vec<f32>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HashEmbedder {
    pub dim: usize,
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self { dim: 128 }
    }
}

impl Embedder for HashEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let dim = self.dim.max(1);
        let mut vector = vec![0.0f32; dim];
        for token in tokenize(text) {
            let hash = fnv1a64(token.as_bytes());
            let index = (hash as usize) % dim;
            let sign = if (hash >> 63) == 0 { 1.0 } else { -1.0 };
            vector[index] += sign;
        }
        normalize(vector)
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b).map(|(left, right)| left * right).sum()
}

pub fn mean_embedding(vectors: &[Vec<f32>], dim: usize) -> Vec<f32> {
    if vectors.is_empty() {
        return vec![0.0; dim];
    }
    let mut out = vec![0.0f32; dim];
    for vector in vectors {
        for (index, value) in vector.iter().enumerate().take(dim) {
            out[index] += *value;
        }
    }
    let scale = 1.0 / vectors.len() as f32;
    for value in &mut out {
        *value *= scale;
    }
    normalize(out)
}

fn normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}
