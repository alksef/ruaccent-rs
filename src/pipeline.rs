//! Публичный пайплайн `RuAccent`: загрузка workdir (словари + 4 модели) и
//! `process_all` / `process_all_with_skip`. Порт `RUAccent.process_all`,
//! `process_all_internal`, `_process_omographs`, `_process_accent` из
//! `ruaccent/ruaccent.py`.

use std::collections::HashMap;
use std::path::Path;

use regex::Regex;

use crate::dict::{self, DictError, Dictionaries};
use crate::models::accent::{AccentModel, AccentModelError};
use crate::models::omograph::{OmographCandidate, OmographModel, OmographModelError};
use crate::models::stress::{StressModelError, StressUsagePredictor};
use crate::models::yo::{process_yo, YoHomographModel, YoModelError};
use crate::text::{delete_spaces_before_punc, normalize, split_by_sentences, split_by_words};

#[derive(Debug)]
pub enum RuAccentError {
    Dict(DictError),
    Yo(YoModelError),
    Omograph(OmographModelError),
    Accent(AccentModelError),
    Stress(StressModelError),
    InvalidOmographModel(String),
}

impl std::fmt::Display for RuAccentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuAccentError::Dict(e) => write!(f, "{e}"),
            RuAccentError::Yo(e) => write!(f, "{e}"),
            RuAccentError::Omograph(e) => write!(f, "{e}"),
            RuAccentError::Accent(e) => write!(f, "{e}"),
            RuAccentError::Stress(e) => write!(f, "{e}"),
            RuAccentError::InvalidOmographModel(name) => {
                write!(f, "invalid omograph model id: {name:?}")
            }
        }
    }
}

impl std::error::Error for RuAccentError {}

impl From<DictError> for RuAccentError {
    fn from(e: DictError) -> Self {
        RuAccentError::Dict(e)
    }
}
impl From<YoModelError> for RuAccentError {
    fn from(e: YoModelError) -> Self {
        RuAccentError::Yo(e)
    }
}
impl From<OmographModelError> for RuAccentError {
    fn from(e: OmographModelError) -> Self {
        RuAccentError::Omograph(e)
    }
}
impl From<AccentModelError> for RuAccentError {
    fn from(e: AccentModelError) -> Self {
        RuAccentError::Accent(e)
    }
}
impl From<StressModelError> for RuAccentError {
    fn from(e: StressModelError) -> Self {
        RuAccentError::Stress(e)
    }
}

pub struct RuAccent {
    dicts: Dictionaries,
    yo: YoHomographModel,
    omograph: OmographModel,
    accent: AccentModel,
    stress: StressUsagePredictor,
}

impl RuAccent {
    pub fn load(workdir: &Path) -> Result<Self, RuAccentError> {
        Self::load_with_omograph(workdir, "tiny")
    }

    pub fn load_with_custom(
        workdir: &Path,
        custom_dict: &HashMap<String, String>,
        custom_homographs: &HashMap<String, Vec<String>>,
    ) -> Result<Self, RuAccentError> {
        Self::load_with_omograph_and_custom(workdir, "tiny", custom_dict, custom_homographs)
    }

    /// Загрузить workdir с выбранным каталогом модели омографов из
    /// `workdir/nn/nn_omograph/<model>`.
    pub fn load_with_omograph(workdir: &Path, omograph_model: &str) -> Result<Self, RuAccentError> {
        Self::load_with_omograph_and_custom(
            workdir,
            omograph_model,
            &HashMap::new(),
            &HashMap::new(),
        )
    }

    /// Загрузить workdir с выбранным каталогом модели омографов и
    /// пользовательскими словарями/омографами.
    pub fn load_with_omograph_and_custom(
        workdir: &Path,
        omograph_model: &str,
        custom_dict: &HashMap<String, String>,
        custom_homographs: &HashMap<String, Vec<String>>,
    ) -> Result<Self, RuAccentError> {
        let omograph_dir = omograph_dir(workdir, omograph_model)?;
        let dicts = dict::with_overrides(dict::load(workdir)?, custom_dict, custom_homographs);
        let yo = YoHomographModel::load(&workdir.join("nn/nn_yo_homograph_resolver"))?;
        let omograph = OmographModel::load(&omograph_dir)?;
        let accent = {
            let nn = workdir.join("nn/nn_accent");
            AccentModel::load(&nn.join("model.onnx"), &nn.join("config.json"), &nn.join("vocab.txt"))?
        };
        let stress = StressUsagePredictor::load(&workdir.join("nn/nn_stress_usage_predictor"))?;
        Ok(RuAccent { dicts, yo, omograph, accent, stress })
    }

    /// Порт `RUAccent.process_all` без skip_regex.
    pub fn process_all(&mut self, text: &str) -> String {
        self.process_all_internal(text)
    }

    /// Порт `RUAccent.process_all` с skip_regex: сегменты, совпавшие с
    /// регуляркой, пропускаются (не нормализуются и не обрабатываются).
    pub fn process_all_with_skip(&mut self, text: &str, skip_regex: &str) -> String {
        let Ok(pattern) = Regex::new(skip_regex) else {
            return self.process_all_internal(text);
        };
        let indices: Vec<(usize, usize)> =
            pattern.find_iter(text).map(|m| (m.start(), m.end())).collect();
        let skipped: Vec<String> = indices.iter().map(|&(l, r)| text[l..r].to_string()).collect();

        if indices.is_empty() {
            return self.process_all_internal(text);
        }

        let mut elems: Vec<String> = Vec::new();
        for pair in indices.windows(2) {
            elems.push(text[pair[0].1..pair[1].0].to_string());
        }
        let first_elem = text[..indices[0].0].to_string();
        let last_elem = text[indices[indices.len() - 1].1..].to_string();
        let mut all_elems = Vec::with_capacity(elems.len() + 2);
        all_elems.push(first_elem);
        all_elems.extend(elems);
        all_elems.push(last_elem);

        let mut results: Vec<String> = Vec::with_capacity(all_elems.len());
        for e in &all_elems {
            if e.is_empty() {
                results.push(e.clone());
            } else {
                results.push(self.process_all_internal(e));
            }
        }

        let mut out = String::new();
        out.push_str(&results[0]);
        for (s, r) in skipped.iter().zip(results.iter().skip(1)) {
            out.push_str(s);
            out.push_str(r);
        }
        out
    }

    fn process_all_internal(&mut self, text: &str) -> String {
        let normalized = normalize(text);
        let sentences = split_by_sentences(&normalized);
        let mut outputs = Vec::new();
        for sentence in sentences {
            let (words, remaining_text) = split_by_words(&sentence);
            if words.is_empty() {
                outputs.push(remaining_text.concat());
                continue;
            }
            let stress_usages: Vec<String> = self
                .stress
                .predict(&sentence)
                .map(|ents| ents.into_iter().map(|e| e.entity).collect())
                .unwrap_or_default();
            let mut processed_words = process_yo(&words, &sentence, &self.dicts, &mut self.yo);
            processed_words = self.process_omographs(processed_words);
            processed_words = self.process_accent(processed_words, &stress_usages);

            let mut processed_sentence = String::new();
            for (l, r) in remaining_text.iter().zip(processed_words.iter()) {
                processed_sentence.push_str(l);
                processed_sentence.push_str(r);
            }
            processed_sentence.push_str(remaining_text.last().map(String::as_str).unwrap_or(""));
            outputs.push(delete_spaces_before_punc(&processed_sentence));
        }
        outputs.concat()
    }

    /// Порт `_process_omographs`: находит омографы (точное совпадение, без
    /// lower), классифицирует и подставляет выбранный вариант.
    fn process_omographs(&mut self, words: Vec<String>) -> Vec<String> {
        let mut candidates: Vec<OmographCandidate> = Vec::new();
        for (i, word) in words.iter().enumerate() {
            if let Some(variants) = self.dicts.omographs.get(word) {
                candidates.push(OmographCandidate {
                    word: word.clone(),
                    variants: variants.clone(),
                    position: i,
                });
            }
        }
        let mut result = words;
        if candidates.is_empty() {
            return result;
        }
        let selected = self.omograph.classify(&result, &candidates);
        for (cls_index, cand) in candidates.iter().enumerate() {
            result[cand.position] = selected[cls_index].clone();
        }
        result
    }

    /// Порт `_process_accent`.
    fn process_accent(&mut self, words: Vec<String>, stress_usages: &[String]) -> Vec<String> {
        let mut result = words;
        for (i, item) in result.iter_mut().enumerate() {
            if item.contains('+') {
                continue;
            }
            if stress_usages.get(i).map(String::as_str) != Some("STRESS") {
                continue;
            }
            let word = item.clone();
            let lower_word = word.to_lowercase();
            let stressed_word = self
                .dicts
                .accents
                .get(&lower_word)
                .unwrap_or_else(|| lower_word.clone());
            if stressed_word == lower_word
                && !has_punctuation(&lower_word)
                && count_vowels(&lower_word) > 1
            {
                *item = self.accent.put_accent(&word);
            } else {
                *item = transfer_plus(&word, &stressed_word);
            }
        }
        result
    }
}

/// Проверить имя модели омографов: не пустое, не `.`/`..`, не содержит
/// разделителей пути и не является абсолютным путём. Возвращает осмысленную
/// ошибку до любого IO.
fn validate_omograph_model(model: &str) -> Result<(), RuAccentError> {
    use std::path::Component;
    let path = Path::new(model);
    let invalid = model.is_empty()
        || path.is_absolute()
        || path.components().any(|c| {
            matches!(
                c,
                Component::CurDir
                    | Component::ParentDir
                    | Component::RootDir
                    | Component::Prefix(_)
            )
        })
        || model.contains('/')
        || model.contains('\\');
    if invalid {
        Err(RuAccentError::InvalidOmographModel(model.to_string()))
    } else {
        Ok(())
    }
}

/// Каталог модели омографов `workdir/nn/nn_omograph/<model>`.
fn omograph_dir(workdir: &Path, model: &str) -> Result<std::path::PathBuf, RuAccentError> {
    validate_omograph_model(model)?;
    Ok(workdir.join("nn/nn_omograph").join(model))
}

/// Порт `RUAccent.has_punctuation`.
fn has_punctuation(text: &str) -> bool {
    const PUNCT: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    text.chars().any(|c| PUNCT.contains(c))
}

/// Порт `RUAccent.count_vowels`.
fn count_vowels(text: &str) -> usize {
    text.chars().filter(|c| "аеёиоуыэюяАЕЁИОУЫЭЮЯ".contains(*c)).count()
}

/// Порт переноса `+` из `stressed_word` в оригинальное слово (регистр
/// сохраняется): `word_fixed = word_fixed[:e.start()+j] + ["+"] +
/// word[e.start():]` для j-го `+`.
fn transfer_plus(word: &str, stressed_word: &str) -> String {
    let word_chars: Vec<char> = word.chars().collect();
    let plus_positions: Vec<usize> = stressed_word
        .chars()
        .enumerate()
        .filter(|(_, c)| *c == '+')
        .map(|(i, _)| i)
        .collect();
    let mut word_fixed: Vec<char> = word_chars.clone();
    for (j, e_start) in plus_positions.iter().enumerate() {
        let mut new_fixed: Vec<char> = word_fixed[..e_start + j].to_vec();
        new_fixed.push('+');
        new_fixed.extend_from_slice(&word_chars[*e_start..]);
        word_fixed = new_fixed;
    }
    word_fixed.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Option<std::path::PathBuf> {
        std::env::var("RUACCENT_TINY_WORKDIR").ok().map(Into::into)
    }

    fn records() -> Vec<(String, String)> {
        let golden =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/golden/tiny.jsonl"))
                .unwrap();
        golden
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .map(|r| {
                (
                    r["input"].as_str().unwrap().to_string(),
                    r["final"].as_str().unwrap().to_string(),
                )
            })
            .collect()
    }

    #[test]
    fn process_all_matches_golden_final() {
        let Some(dir) = tiny() else {
            eprintln!("RUACCENT_TINY_WORKDIR not set; skipping");
            return;
        };
        let mut accentor = RuAccent::load(&dir).unwrap();
        let records = records();
        assert_eq!(records.len(), 20, "expected 20 golden records");
        for (input, expected) in &records {
            let got = accentor.process_all(input);
            assert_eq!(&got, expected, "mismatch for {input:?}");
        }
    }

    #[test]
    fn skip_regex_matches_fixtures() {
        let Some(dir) = tiny() else {
            eprintln!("RUACCENT_TINY_WORKDIR not set; skipping");
            return;
        };
        let mut accentor = RuAccent::load(&dir).unwrap();

        let cases: [(&str, &str, &str); 4] = [
            ("Мука 123 уже 456 стоит.", r"\d+", "М+ука 123 уж+е 456 сто+ит."),
            (
                "Ссылка https://example.com/path уже стоит.",
                r"https?://\S+",
                "Сс+ылка https://example.com/path уж+е сто+ит.",
            ),
            ("На двери висит замок.", "замок", "На дв+ери вис+ит замок."),
            ("Он все понял и ушел.", "все", "+Он все п+онял и уш+ёл."),
        ];
        for (input, skip, expected) in cases {
            assert_eq!(
                &accentor.process_all_with_skip(input, skip),
                expected,
                "skip {skip:?} for {input:?}"
            );
        }

        assert_eq!(
            accentor.process_all("Ссылка https://example.com/path уже стоит."),
            "Сс+ылка https:example.compath уж+е сто+ит."
        );
        assert_eq!(
            accentor.process_all("На двери висит замок."),
            "На дв+ери вис+ит зам+ок."
        );
        assert_eq!(
            accentor.process_all("Он все понял и ушел."),
            "+Он всё п+онял и уш+ёл."
        );
    }

    #[test]
    fn transfer_plus_preserves_original_case() {
        assert_eq!(transfer_plus("Дверь", "дв+ерь"), "Дв+ерь");
    }

    #[test]
    fn omograph_dir_builds_path_for_valid_model_ids() {
        let workdir = std::path::Path::new("/workdir");
        for id in ["tiny", "medium", "tiny-v2", "base_lite"] {
            let dir = omograph_dir(workdir, id).unwrap();
            assert_eq!(dir, workdir.join("nn/nn_omograph").join(id));
        }
    }

    #[test]
    fn omograph_dir_rejects_invalid_model_ids() {
        let workdir = std::path::Path::new("/workdir");
        let invalid = [
            "",
            ".",
            "..",
            "tiny/../../secret",
            "tiny\\..\\..\\secret",
            "foo/bar",
            "foo\\bar",
            "/absolute",
            "C:\\abs\\path",
        ];
        for id in invalid {
            assert!(
                omograph_dir(workdir, id).is_err(),
                "expected rejection for {id:?}"
            );
        }
    }

    #[test]
    fn invalid_omograph_model_returns_typed_error() {
        let err = omograph_dir(std::path::Path::new("/workdir"), "../escape").unwrap_err();
        assert!(matches!(err, RuAccentError::InvalidOmographModel(_)));
    }
}
