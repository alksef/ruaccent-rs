//! ruaccent-rs — native Rust port of the RUAccent text pipeline.

pub mod dict;
pub mod models;
pub mod pipeline;
pub mod text;
pub mod tokenization;

pub use pipeline::RuAccent;
