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
    /// Повреждён FST-файл или ошибка mmap.
    BadFst { file: &'static str, reason: String },
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
            DictError::BadFst { file, reason } => {
                write!(f, "corrupted FST in dictionary file {file}: {reason}")
            }
        }
    }
}

impl std::error::Error for DictError {}

/// Хранилище словаря ударений: либо in-memory HashMap (legacy json.gz),
/// либо memmap-отображение скомпилированного FST-файла (`accents.fst`).
#[derive(Debug)]
pub enum AccentStorage {
    /// In-memory HashMap (legacy-режим из `accents.json.gz`).
    HashMap(HashMap<String, String>),
    /// Memory-mapped FST (компактный бинарный словарь `accents.fst`).
    Fst {
        map: fst::Map<memmap2::Mmap>,
        overrides: HashMap<String, String>,
    },
}

impl Default for AccentStorage {
    fn default() -> Self {
        AccentStorage::HashMap(HashMap::new())
    }
}

impl AccentStorage {
    /// Получить форму слова с ударением (`+`).
    pub fn get(&self, word: &str) -> Option<String> {
        match self {
            AccentStorage::HashMap(map) => map.get(word).cloned(),
            AccentStorage::Fst { map, overrides } => {
                if let Some(val) = overrides.get(word) {
                    return Some(val.clone());
                }
                let code = map.get(word.as_bytes())?;
                if code == 0 {
                    Some(word.to_string())
                } else {
                    let offset = (code - 1) as usize;
                    if offset <= word.len() {
                        let mut res = String::with_capacity(word.len() + 1);
                        res.push_str(&word[..offset]);
                        res.push('+');
                        res.push_str(&word[offset..]);
                        Some(res)
                    } else {
                        None
                    }
                }
            }
        }
    }

    /// Добавить или переопределить форму слова.
    pub fn insert(&mut self, word: String, accented: String) {
        match self {
            AccentStorage::HashMap(map) => {
                map.insert(word, accented);
            }
            AccentStorage::Fst { overrides, .. } => {
                overrides.insert(word, accented);
            }
        }
    }

    /// Количество записей в словаре.
    pub fn len(&self) -> usize {
        match self {
            AccentStorage::HashMap(map) => map.len(),
            AccentStorage::Fst { map, overrides } => {
                // overrides может как переопределять уже существующие слова,
                // так и добавлять новые; считаем только добавленные.
                let added = overrides
                    .keys()
                    .filter(|word| map.get(word.as_bytes()).is_none())
                    .count();
                map.len() + added
            }
        }
    }

    /// Пуст ли словарь.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Default)]
pub struct Dictionaries {
    /// Слово (lower) -> варианты с ударениями, напр. `"замок" -> ["з+амок", "зам+ок"]`.
    pub omographs: HashMap<String, Vec<String>>,
    /// Слово (lower) -> форма с `+`, напр. `"дверь" -> "дв+ерь"`.
    pub accents: AccentStorage,
    /// `е`-слова с подтверждённым `ё`.
    pub yo_words: HashMap<String, String>,
    /// Омографы `е/ё`.
    pub yo_homographs: HashMap<String, String>,
}

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

/// Загрузить все словари из `<workdir>/dictionary/`.
/// Если доступен `accents.fst`, он отображается через mmap с нулевыми аллокациями кучи.
/// В противном случае используется `accents.json.gz`.
pub fn load(workdir: &Path) -> Result<Dictionaries, DictError> {
    let mut dicts = Dictionaries::default();

    // 1. omographs
    let omo_path = dict_path(workdir, "omographs.json.gz");
    let omo_text = read_gz_json(&omo_path, "omographs.json.gz")?;
    dicts.omographs = parse_omographs(&omo_text, "omographs.json.gz")?;

    // 2. accents (fst with fallback to json.gz)
    let fst_path = dict_path(workdir, "accents.fst");
    if fst_path.is_file() {
        let file = File::open(&fst_path).map_err(|e| DictError::BadFst {
            file: "accents.fst",
            reason: e.to_string(),
        })?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| DictError::BadFst {
            file: "accents.fst",
            reason: e.to_string(),
        })?;
        let map = fst::Map::new(mmap).map_err(|e| DictError::BadFst {
            file: "accents.fst",
            reason: e.to_string(),
        })?;
        dicts.accents = AccentStorage::Fst {
            map,
            overrides: HashMap::new(),
        };
    } else {
        let path = dict_path(workdir, "accents.json.gz");
        let text = read_gz_json(&path, "accents.json.gz")?;
        dicts.accents = AccentStorage::HashMap(parse_string_map(&text, "accents.json.gz")?);
    }

    // 3. yo_words
    let yo_w_path = dict_path(workdir, "yo_words.json.gz");
    let yo_w_text = read_gz_json(&yo_w_path, "yo_words.json.gz")?;
    dicts.yo_words = parse_string_map(&yo_w_text, "yo_words.json.gz")?;

    // 4. yo_homographs
    let yo_h_path = dict_path(workdir, "yo_homographs.json.gz");
    let yo_h_text = read_gz_json(&yo_h_path, "yo_homographs.json.gz")?;
    dicts.yo_homographs = parse_string_map(&yo_h_text, "yo_homographs.json.gz")?;

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
        assert_eq!(dicts.accents.get("дверь").as_deref(), Some("дв+ерь"));
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
        assert_eq!(dicts.accents.get("о").as_deref(), Some("+о"));
        assert_eq!(dicts.accents.get("О").as_deref(), Some("+О"));
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
        assert_eq!(dicts.accents.get("дверь").as_deref(), Some("тес+т"));
        assert_eq!(
            dicts.accents.get("новый").as_deref(),
            Some("н+овый")
        );
    }

    #[test]
    fn fst_storage_lookup_and_override() {
        use fst::MapBuilder;
        let mut build_buf = Vec::new();
        let mut builder = MapBuilder::new(&mut build_buf).unwrap();
        let acc_word = "дв+ерь";
        let plus_byte = acc_word.find('+').unwrap() as u64;
        builder.insert("дверь", plus_byte + 1).unwrap();
        builder.finish().unwrap();

        let tmp = std::env::temp_dir().join("ruaccent_fst_test.fst");
        std::fs::write(&tmp, build_buf).unwrap();
        let file = File::open(&tmp).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        let map = fst::Map::new(mmap).unwrap();
        let mut storage = AccentStorage::Fst {
            map,
            overrides: HashMap::new(),
        };

        assert_eq!(storage.get("дверь").as_deref(), Some("дв+ерь"));
        assert_eq!(storage.get("неизвестно"), None);

        // Test override
        storage.insert("дверь".to_string(), "двер+ь".to_string());
        assert_eq!(storage.get("дверь").as_deref(), Some("двер+ь"));

        // Test inserting new word into overrides
        storage.insert("замок".to_string(), "зам+ок".to_string());
        assert_eq!(storage.get("замок").as_deref(), Some("зам+ок"));

        drop(storage);
        let _ = std::fs::remove_file(&tmp);
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
