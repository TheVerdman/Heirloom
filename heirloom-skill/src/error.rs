use std::error::Error;
use std::fmt;

pub type SkillResult<T> = std::result::Result<T, SkillError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillError {
    Io(String),
    Json(String),
    Parse(String),
    Lint(Vec<String>),
    Format(String),
    Checksum(String),
    Invalid(String),
}

impl fmt::Display for SkillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(f, "io error: {message}"),
            Self::Json(message) => write!(f, "json error: {message}"),
            Self::Parse(message) => write!(f, "parse error: {message}"),
            Self::Lint(messages) => write!(f, "skill lint failed: {}", messages.join("; ")),
            Self::Format(message) => write!(f, "format error: {message}"),
            Self::Checksum(message) => write!(f, "checksum error: {message}"),
            Self::Invalid(message) => write!(f, "invalid skill operation: {message}"),
        }
    }
}

impl Error for SkillError {}

impl From<std::io::Error> for SkillError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err.to_string())
    }
}

impl From<serde_json::Error> for SkillError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err.to_string())
    }
}
