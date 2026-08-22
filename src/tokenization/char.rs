//! Порт `CharTokenizer` из `ruaccent/char_tokenizer.py`.
//!
//! Vocab — `vocab.txt` (один токен на строку, индекс = позиция). Токенизация
//! посимвольная с lowercase, неизвестные символы → unk, обрамление bos/eos.
//! `token_type_ids` — нули (модель nn_accent их требует, см. golden_dump.py).

use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug)]
pub enum TokenizerError {
    /// Файл vocab не найден (в сообщении только имя, без путей).
    NotFound { file: &'static str },
    /// Пустой или некорректный vocab (нет обязательных special tokens).
    BadVocab { file: &'static str, reason: String },
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::NotFound { file } => write!(f, "vocab file not found: {file}"),
            TokenizerError::BadVocab { file, reason } => {
                write!(f, "bad vocab file {file}: {reason}")
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

#[derive(Debug, Clone)]
pub struct EncodedWord {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub token_type_ids: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct CharTokenizer {
    vocab: HashMap<String, i64>,
    ids_to_tokens: Vec<String>,
    bos_id: i64,
    eos_id: i64,
    unk_id: i64,
}

const VOCAB_FILE: &str = "vocab.txt";

impl CharTokenizer {
    pub fn from_vocab_file(path: &Path) -> Result<Self, TokenizerError> {
        let text = fs::read_to_string(path).map_err(|_| TokenizerError::NotFound {
            file: VOCAB_FILE,
        })?;
        let mut vocab = HashMap::new();
        let mut ids_to_tokens = Vec::new();
        for (idx, line) in text.lines().enumerate() {
            let token = line.strip_suffix('\r').unwrap_or(line);
            vocab.insert(token.to_string(), idx as i64);
            ids_to_tokens.push(token.to_string());
        }
        let special = |name: &str| {
            vocab
                .get(&format!("[{name}]"))
                .copied()
                .ok_or_else(|| TokenizerError::BadVocab {
                    file: VOCAB_FILE,
                    reason: format!("missing special token [{name}]"),
                })
        };
        Ok(CharTokenizer {
            bos_id: special("bos")?,
            eos_id: special("eos")?,
            unk_id: special("unk")?,
            vocab,
            ids_to_tokens,
        })
    }

    /// Токенизировать слово: `[bos] + char ids + [eos]`, все маски единичные/
    /// нулевые соответственно. Регистр входа приводится к нижнему.
    pub fn encode_word(&self, word: &str) -> EncodedWord {
        let lower = word.to_lowercase();
        let mut ids = Vec::with_capacity(lower.chars().count() + 2);
        ids.push(self.bos_id);
        for ch in lower.chars() {
            let key = ch.to_string();
            ids.push(*self.vocab.get(&key).unwrap_or(&self.unk_id));
        }
        ids.push(self.eos_id);
        let n = ids.len() as i64;
        EncodedWord {
            input_ids: ids,
            attention_mask: vec![1; n as usize],
            token_type_ids: vec![0; n as usize],
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.ids_to_tokens.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_workdir() -> Option<std::path::PathBuf> {
        std::env::var("RUACCENT_TINY_WORKDIR").ok().map(Into::into)
    }

    fn write_mini_vocab(dir: &Path) -> std::path::PathBuf {
        let path = dir.join("vocab.txt");
        let mini = "[pad]\n[unk]\n[bos]\n[eos]\nа\nб\nв\nd\n";
        fs::write(&path, mini).unwrap();
        path
    }

    #[test]
    fn golden_kvardelyambu() {
        let Some(dir) = tiny_workdir() else {
            eprintln!("RUACCENT_TINY_WORKDIR not set; skipping");
            return;
        };
        let tok = CharTokenizer::from_vocab_file(&dir.join("nn/nn_accent/vocab.txt")).unwrap();
        let enc = tok.encode_word("кварделямбу");
        assert_eq!(enc.input_ids, vec![2, 22, 14, 12, 28, 16, 17, 23, 43, 24, 13, 31, 3]);
        assert_eq!(enc.attention_mask, vec![1; 13]);
        assert_eq!(enc.token_type_ids, vec![0; 13]);
    }

    #[test]
    fn mini_vocab_known_unknown_and_case() {
        let tmp = std::env::temp_dir().join("ruaccent_rs_char_test");
        fs::create_dir_all(&tmp).unwrap();
        let path = write_mini_vocab(&tmp);
        let tok = CharTokenizer::from_vocab_file(&path).unwrap();
        // [bos]=2, а=4, б=5, в=6, d=7, [eos]=3; 'А' приводится к 'а'
        let enc = tok.encode_word("Абвd");
        assert_eq!(enc.input_ids, vec![2, 4, 5, 6, 7, 3]);
        // неизвестный символ → unk=1
        let unk_enc = tok.encode_word("я");
        assert_eq!(unk_enc.input_ids, vec![2, 1, 3]);
        // пустая строка → только special tokens
        let empty = tok.encode_word("");
        assert_eq!(empty.input_ids, vec![2, 3]);
        assert_eq!(tok.vocab_size(), 8);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn missing_vocab_is_not_found() {
        let err = CharTokenizer::from_vocab_file(Path::new("Z:/no/such/vocab.txt")).unwrap_err();
        assert!(matches!(err, TokenizerError::NotFound { .. }), "got: {err}");
    }

    #[test]
    fn vocab_without_special_tokens_is_bad() {
        let tmp = std::env::temp_dir().join("ruaccent_rs_char_bad");
        fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("vocab.txt");
        fs::write(&path, "а\nб\n").unwrap();
        let err = CharTokenizer::from_vocab_file(&path).unwrap_err();
        assert!(matches!(err, TokenizerError::BadVocab { .. }), "got: {err}");
        let _ = fs::remove_dir_all(&tmp);
    }
}
