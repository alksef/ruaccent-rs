//! Загрузчик словарей ruaccent (gzip + JSON) из `<workdir>/dictionary/`.
//!
//! Порт чтения словарей из `ruaccent/ruaccent.py::load`: `omographs.json.gz`,
//! `accents.json.gz` (use_dictionary=True), `yo_words.json.gz`,
//! `yo_homographs.json.gz`. `accents_nn.json.gz` (tiny_mode) сознательно не
//! загружается.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;
use serde::Deserialize;

#[derive(Debug)]
pub enum DictError {
    /// Файл словаря не найден (в сообщении только имя файла, без путей).
    NotFound { file: &'static str },
    /// Повреждён gzip-поток.
    BadGzip { file: &'static str, source: std::io::Error },
    /// Повреждён или не соответствует схеме JSON.
    BadJson { file: &'static str, source: serde_json::Error },
}

impl std::fmt::Display for DictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DictError::NotFound { file } => write!(f, "dictionary file not found: {file}"),
            DictError::BadGzip { file, source } => {
                write!(f, "corrupted gzip in dictionary file {file}: {source}")
            }
            DictError::BadJson { file, source } => {
                write!(f, "corrupted JSON in dictionary file {file}: {source}")
            }
        }
    }
}

impl std::error::Error for DictError {}

#[derive(Debug, Clone, Default)]
pub struct Dictionaries {
    /// Слово (lower) -> варианты с ударениями, напр. `"замок" -> ["з+амок", "зам+ок"]`.
    pub omographs: HashMap<String, Vec<String>>,
    /// Слово (lower) -> форма с `+`, напр. `"дверь" -> "дв+ерь"`.
    pub accents: HashMap<String, String>,
    /// `е`-слова с подтверждённым `ё`.
    pub yo_words: HashMap<String, String>,
    /// Омографы `е/ё`.
    pub yo_homographs: HashMap<String, String>,
}

const FILES: &[&str] = &[
    "omographs.json.gz",
    "accents.json.gz",
    "yo_words.json.gz",
    "yo_homographs.json.gz",
];

fn dict_path(workdir: &Path, name: &str) -> PathBuf {
    workdir.join("dictionary").join(name)
}

fn read_gz_json(path: &Path, file: &'static str) -> Result<String, DictError> {
    let raw = File::open(path).map_err(|_| DictError::NotFound { file })?;
    let mut decoder = GzDecoder::new(raw);
    let mut text = String::new();
    decoder
        .read_to_string(&mut text)
        .map_err(|source| DictError::BadGzip { file, source })?;
    Ok(text)
}

fn parse_omographs(text: &str, file: &'static str) -> Result<HashMap<String, Vec<String>>, DictError> {
    #[derive(Deserialize)]
    struct Raw(HashMap<String, Vec<String>>);
    let raw: Raw = serde_json::from_str(text).map_err(|source| DictError::BadJson { file, source })?;
    Ok(raw.0)
}

fn parse_string_map(text: &str, file: &'static str) -> Result<HashMap<String, String>, DictError> {
    #[derive(Deserialize)]
    struct Raw(HashMap<String, String>);
    let raw: Raw = serde_json::from_str(text).map_err(|source| DictError::BadJson { file, source })?;
    Ok(raw.0)
}

/// Загрузить все четыре словаря из `<workdir>/dictionary/`.
pub fn load(workdir: &Path) -> Result<Dictionaries, DictError> {
    let mut dicts = Dictionaries::default();
    for name in FILES.iter().copied() {
        let path = dict_path(workdir, name);
        let text = read_gz_json(&path, name)?;
        match name {
            "omographs.json.gz" => dicts.omographs = parse_omographs(&text, name)?,
            "accents.json.gz" => dicts.accents = parse_string_map(&text, name)?,
            "yo_words.json.gz" => dicts.yo_words = parse_string_map(&text, name)?,
            _ => dicts.yo_homographs = parse_string_map(&text, name)?,
        }
    }
    Ok(dicts)
}

/// Применить пользовательские словари, повторяя порядок merge из
/// `ruaccent.py::load` (строки ~90–110):
///
/// 1. `omographs` уже загружен из `omographs.json.gz`;
/// 2. `omographs.update({"коса": ["к+оса", "кос+а"]})` — hardcoded-правка upstream;
/// 3. `omographs.update(custom_homographs)`;
/// 4. `accents` уже загружен из `accents.json.gz`;
/// 5. `accents.update(custom_dict)`;
/// 6. `accents.update(letters_accent)` — hardcoded `{"о": "+о", "О": "+О"}`.
pub fn with_overrides(
    mut dicts: Dictionaries,
    custom_dict: &HashMap<String, String>,
    custom_homographs: &HashMap<String, Vec<String>>,
) -> Dictionaries {
    dicts
        .omographs
        .insert("коса".to_string(), vec!["к+оса".to_string(), "кос+а".to_string()]);
    for (word, variants) in custom_homographs {
        dicts.omographs.insert(word.clone(), variants.clone());
    }
    for (word, accented) in custom_dict {
        dicts.accents.insert(word.clone(), accented.clone());
    }
    dicts.accents.insert("о".to_string(), "+о".to_string());
    dicts.accents.insert("О".to_string(), "+О".to_string());
    dicts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workdir() -> Option<std::path::PathBuf> {
        std::env::var("RUACCENT_TINY_WORKDIR").ok().map(Into::into)
    }

    #[test]
    fn loads_real_tiny_dictionaries() {
        let Some(dir) = workdir() else {
            eprintln!("RUACCENT_TINY_WORKDIR not set; skipping");
            return;
        };
        let dicts = load(&dir).expect("load must succeed");
        assert_eq!(
            dicts.omographs.get("замок").map(|v| v.as_slice()),
            Some(&["з+амок".to_string(), "зам+ок".to_string()][..])
        );
        assert_eq!(dicts.accents.get("дверь").map(String::as_str), Some("дв+ерь"));
        assert_eq!(dicts.omographs.len(), 19741);
        assert_eq!(dicts.accents.len(), 3_194_879);
        assert_eq!(dicts.yo_words.len(), 85_568);
        assert_eq!(dicts.yo_homographs.len(), 637);
    }

    #[test]
    fn missing_file_is_not_found() {
        let err = load(Path::new("Z:/definitely/missing")).unwrap_err();
        assert!(matches!(err, DictError::NotFound { .. }), "got: {err}");
    }

    #[test]
    fn overrides_apply_hardcoded_kosa_and_letters() {
        let dicts = with_overrides(Dictionaries::default(), &HashMap::new(), &HashMap::new());
        assert_eq!(
            dicts.omographs.get("коса").map(|v| v.as_slice()),
            Some(&["к+оса".to_string(), "кос+а".to_string()][..])
        );
        assert_eq!(dicts.accents.get("о").map(String::as_str), Some("+о"));
        assert_eq!(dicts.accents.get("О").map(String::as_str), Some("+О"));
    }

    #[test]
    fn overrides_custom_homographs_override_file() {
        let mut dicts = Dictionaries::default();
        dicts.omographs.insert(
            "замок".to_string(),
            vec!["з+амок".to_string(), "зам+ок".to_string()],
        );
        let custom_homographs: HashMap<String, Vec<String>> =
            [("замок".to_string(), vec!["тест+овый".to_string()])]
                .into_iter()
                .collect();
        let dicts = with_overrides(dicts, &HashMap::new(), &custom_homographs);
        assert_eq!(
            dicts.omographs.get("замок").map(|v| v.as_slice()),
            Some(&["тест+овый".to_string()][..])
        );
    }

    #[test]
    fn overrides_custom_dict_adds_and_overrides() {
        let mut dicts = Dictionaries::default();
        dicts.accents.insert("дверь".to_string(), "дв+ерь".to_string());
        let custom_dict: HashMap<String, String> = [
            ("дверь".to_string(), "тес+т".to_string()),
            ("новый".to_string(), "н+овый".to_string()),
        ]
        .into_iter()
        .collect();
        let dicts = with_overrides(dicts, &custom_dict, &HashMap::new());
        assert_eq!(dicts.accents.get("дверь").map(String::as_str), Some("тес+т"));
        assert_eq!(
            dicts.accents.get("новый").map(String::as_str),
            Some("н+овый")
        );
    }

    #[test]
    fn bad_json_is_reported() {
        let tmp = std::env::temp_dir().join("ruaccent_rs_dict_test");
        std::fs::create_dir_all(tmp.join("dictionary")).unwrap();
        let path = tmp.join("dictionary").join("omographs.json.gz");
        let mut enc = flate2::write::GzEncoder::new(
            File::create(&path).unwrap(),
            flate2::Compression::default(),
        );
        std::io::Write::write_all(&mut enc, b"not json").unwrap();
        drop(enc);
        let err = load(&tmp).unwrap_err();
        assert!(matches!(err, DictError::BadJson { .. }), "got: {err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
