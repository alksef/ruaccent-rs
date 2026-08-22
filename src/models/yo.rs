//! nn_yo_homograph_resolver: выбор `е`/`ё`. Порт `YoHomographModel`
//! (token-classification, offset_mapping, AVERAGE-агрегация подслов) и
//! `RUAccent._process_yo` (yo_words-словарь + YO→yo_homographs + fix_capital)
//! из `ruaccent/yo_homograph_model.py` и `ruaccent/ruaccent.py`.
//!
//! Пайплайн одного вызова: токенизация нижнего регистра предложения с
//! offsets (в символах) и special_tokens_mask; в сессию идут только
//! `input_ids` + `attention_mask` (int64); логиты `[1, seq, 3]` -> per-token
//! softmax по оси классов; пропуск special-токенов, подслова (`##`) сливаются
//! в слово через WordPiece-decoder со средним по группе и `argmax` первого
//! максимума.

use std::fs;
use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::dict::Dictionaries;
use crate::text::fix_capital;

/// Найденная моделью сущность `е/ё`: слово, метка, уверенность и границы.
#[derive(Debug, Clone)]
pub struct YoEntity {
    pub word: String,
    pub entity: String,
    pub score: f32,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug)]
pub enum YoModelError {
    NotFound { file: &'static str },
    BadConfig { reason: String },
    Tokenizer { reason: String },
    Ort { step: &'static str, source: ort::Error },
}

impl std::fmt::Display for YoModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            YoModelError::NotFound { file } => write!(f, "file not found: {file}"),
            YoModelError::BadConfig { reason } => write!(f, "bad nn_yo_homograph_resolver config: {reason}"),
            YoModelError::Tokenizer { reason } => write!(f, "tokenizer error: {reason}"),
            YoModelError::Ort { step, source } => write!(f, "ort error at {step}: {source}"),
        }
    }
}

impl std::error::Error for YoModelError {}

pub struct YoHomographModel {
    session: Session,
    tokenizer: Tokenizer,
    /// Индекс класса -> label (NO_YO / PUNCT / YO).
    id2label: Vec<String>,
}

/// Токенизированное предложение со всеми артефактами, нужными для сверки.
#[derive(Debug)]
struct EncodedYo {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    offsets: Vec<(usize, usize)>,
    special_tokens_mask: Vec<u32>,
}

/// Сырая сущность подслова до агрегации (порт `collect_pre_entities`).
#[derive(Debug)]
struct PreEntity {
    id: u32,
    /// Токен или (для unk) исходный спан текста — как `entity["word"]` в Python.
    word: String,
    scores: Vec<f32>,
    start: usize,
    end: usize,
    is_subword: bool,
}

impl YoHomographModel {
    /// Загрузить сессию, id2label и токенизатор из каталога модели.
    pub fn load(dir: &Path) -> Result<Self, YoModelError> {
        let model_path = dir.join("model.onnx");
        let tokenizer_path = dir.join("tokenizer.json");
        let config_path = dir.join("config.json");

        let raw = fs::read_to_string(&config_path).map_err(|_| YoModelError::NotFound {
            file: "nn_yo_homograph_resolver/config.json",
        })?;
        let cfg: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| YoModelError::BadConfig { reason: e.to_string() })?;
        let id2label_map = cfg
            .get("id2label")
            .and_then(|v| v.as_object())
            .ok_or_else(|| YoModelError::BadConfig { reason: "no id2label".into() })?;
        let n = id2label_map.len();
        let mut id2label = vec![String::new(); n];
        for (idx, label) in id2label_map {
            let i: usize = idx
                .parse()
                .map_err(|_| YoModelError::BadConfig { reason: format!("bad class index {idx}") })?;
            id2label[i] = label
                .as_str()
                .ok_or_else(|| YoModelError::BadConfig { reason: "non-string label".into() })?
                .to_string();
        }

        let session = Session::builder()
            .and_then(|mut b| b.commit_from_file(&model_path))
            .map_err(|e| YoModelError::Ort { step: "session build", source: e })?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| YoModelError::Tokenizer { reason: e.to_string() })?;

        Ok(YoHomographModel { session, tokenizer, id2label })
    }

    /// Токенизация как в эталоне: offsets в символах + special_tokens_mask,
    /// без паддинга и переопределения truncation (встроен в tokenizer.json).
    fn encode(&self, text: &str) -> Result<EncodedYo, YoModelError> {
        let enc = self
            .tokenizer
            .encode_char_offsets(text, true)
            .map_err(|e| YoModelError::Tokenizer { reason: e.to_string() })?;
        Ok(EncodedYo {
            input_ids: enc.get_ids().iter().map(|&i| i as i64).collect(),
            attention_mask: enc.get_attention_mask().iter().map(|&m| m as i64).collect(),
            offsets: enc.get_offsets().to_vec(),
            special_tokens_mask: enc.get_special_tokens_mask().to_vec(),
        })
    }

    /// Прогнать токенизированное предложение через модель, вернуть логиты
    /// `[1, seq, 3]` (flat).
    fn run(&mut self, enc: &EncodedYo) -> Result<Vec<f32>, YoModelError> {
        let n = enc.input_ids.len();
        let ids = Tensor::from_array(([1usize, n], enc.input_ids.clone()))
            .map_err(|e| YoModelError::Ort { step: "input_ids tensor", source: e })?;
        let mask = Tensor::from_array(([1usize, n], enc.attention_mask.clone()))
            .map_err(|e| YoModelError::Ort { step: "attention_mask tensor", source: e })?;
        let outputs = self
            .session
            .run(ort::inputs!["input_ids" => ids, "attention_mask" => mask])
            .map_err(|e| YoModelError::Ort { step: "run", source: e })?;
        let (_shape, data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| YoModelError::Ort { step: "extract logits", source: e })?;
        Ok(data.to_vec())
    }

    /// Порт `predict_yo_homographs`.
    pub fn predict(&mut self, sentence: &str) -> Result<Vec<YoEntity>, YoModelError> {
        let enc = self.encode(sentence)?;
        let logits = self.run(&enc)?;
        let n_classes = self.id2label.len();
        let scores = softmax_rows(&logits, n_classes);

        let sentence_chars: Vec<char> = sentence.chars().collect();
        let unk_id = self.tokenizer.token_to_id("[UNK]").unwrap_or(1);

        let mut pre = Vec::new();
        for (idx, (&id, &(start, end))) in enc.input_ids.iter().zip(enc.offsets.iter()).enumerate() {
            if enc.special_tokens_mask[idx] != 0 {
                continue;
            }
            let id_u32 = id as u32;
            let word_token = self.tokenizer.id_to_token(id_u32).unwrap_or_default();
            let word_ref: String = sentence_chars[start..end].iter().collect();
            let mut is_subword = word_token.chars().count() != word_ref.chars().count();
            let word = if id_u32 == unk_id { word_ref } else { word_token };
            if id_u32 == unk_id {
                is_subword = false;
            }
            pre.push(PreEntity {
                id: id_u32,
                word,
                scores: scores[idx * n_classes..(idx + 1) * n_classes].to_vec(),
                start,
                end,
                is_subword,
            });
        }

        Ok(aggregate_words(&pre, &self.id2label, &self.tokenizer, unk_id))
    }
}

/// Per-token softmax по оси классов (порт `np.exp(logits - max)/sum`).
fn softmax_rows(logits: &[f32], n_classes: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(logits.len());
    for row in logits.chunks(n_classes) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|v| (v - max).exp()).sum();
        out.extend(row.iter().map(|v| (v - max).exp() / sum));
    }
    out
}

/// Порт `aggregate_word` (AVERAGE) + `aggregate_words`.
fn aggregate_words(
    pre: &[PreEntity],
    id2label: &[String],
    tokenizer: &Tokenizer,
    unk_id: u32,
) -> Vec<YoEntity> {
    let mut result = Vec::new();
    let mut group: Vec<&PreEntity> = Vec::new();
    for e in pre {
        if group.is_empty() || e.is_subword {
            group.push(e);
        } else {
            result.push(aggregate_word(&group, id2label, tokenizer, unk_id));
            group = vec![e];
        }
    }
    if !group.is_empty() {
        result.push(aggregate_word(&group, id2label, tokenizer, unk_id));
    }
    result
}

fn aggregate_word(
    group: &[&PreEntity],
    id2label: &[String],
    tokenizer: &Tokenizer,
    unk_id: u32,
) -> YoEntity {
    let ids: Vec<u32> = group.iter().map(|e| e.id).collect();
    let word = match group.iter().find(|e| e.id == unk_id) {
        Some(unk) => unk.word.clone(),
        None => tokenizer.decode(&ids, false).unwrap_or_default(),
    };

    let n_classes = group[0].scores.len();
    let mut average = vec![0f32; n_classes];
    for e in group {
        for (i, s) in e.scores.iter().enumerate() {
            average[i] += s;
        }
    }
    let count = group.len() as f32;
    for a in average.iter_mut() {
        *a /= count;
    }

    let mut entity_idx = 0usize;
    for (i, &a) in average.iter().enumerate() {
        if a > average[entity_idx] {
            entity_idx = i;
        }
    }

    YoEntity {
        word,
        entity: id2label[entity_idx].clone(),
        score: average[entity_idx],
        start: group[0].start,
        end: group[group.len() - 1].end,
    }
}

/// Порт `RUAccent._process_yo`.
pub fn process_yo(
    words: &[String],
    sentence: &str,
    dicts: &Dictionaries,
    model: &mut YoHomographModel,
) -> Vec<String> {
    let lower_sentence = sentence.to_lowercase();
    let yo_predictions: Option<Vec<String>> = if lower_sentence.contains('е') {
        model
            .predict(&lower_sentence)
            .ok()
            .map(|ents| ents.into_iter().map(|e| e.entity).collect())
    } else {
        None
    };

    let mut result = words.to_vec();
    for (i, word) in words.iter().enumerate() {
        let lower_word = word.to_lowercase();
        result[i] = fix_capital(
            word,
            dicts.yo_words.get(&lower_word).map(String::as_str).unwrap_or(word),
        );
        if let Some(ref preds) = yo_predictions {
            if preds.get(i).map(String::as_str) == Some("YO") {
                result[i] = fix_capital(
                    word,
                    dicts.yo_homographs.get(&lower_word).map(String::as_str).unwrap_or(word),
                );
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tiny() -> Option<PathBuf> {
        std::env::var("RUACCENT_TINY_WORKDIR").ok().map(Into::into)
    }

    fn model(dir: &PathBuf) -> YoHomographModel {
        let nn = dir.join("nn/nn_yo_homograph_resolver");
        YoHomographModel::load(&nn).unwrap()
    }

    fn records() -> Vec<serde_json::Value> {
        let golden =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/golden/tiny.jsonl"))
                .unwrap();
        golden
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .collect()
    }

    fn sentences_of(rec: &serde_json::Value) -> Vec<String> {
        rec["sentences"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect()
    }

    fn words_of(rec: &serde_json::Value) -> Vec<Vec<String>> {
        rec["words"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect()
            })
            .collect()
    }

    fn flat_i64(v: &serde_json::Value) -> Vec<i64> {
        v.as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter())
            .map(|x| x.as_i64().unwrap())
            .collect()
    }

    /// Плоский `[batch, seq, 3]`-тензор logits в `[batch*seq*3]` (три уровня).
    fn flat_f32(v: &serde_json::Value) -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .flat_map(|batch| batch.as_array().unwrap().iter())
            .flat_map(|row| row.as_array().unwrap().iter())
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }

    #[test]
    fn yo_nn_calls_match_golden_bitwise() {
        let Some(dir) = tiny() else {
            eprintln!("RUACCENT_TINY_WORKDIR not set; skipping");
            return;
        };
        let mut model = model(&dir);
        let records = records();
        assert_eq!(records.len(), 20, "expected 20 golden records");
        for rec in &records {
            let sentences = sentences_of(rec);
            let words = words_of(rec);
            let nn_calls = rec["yo_nn_calls"].as_array().unwrap();
            let mut call_idx = 0usize;
            for i in 0..words.len() {
                if words[i].is_empty() {
                    continue;
                }
                let lower = sentences[i].to_lowercase();
                if !lower.contains('е') {
                    continue;
                }
                let enc = model.encode(&lower).unwrap();
                let golden_call = &nn_calls[call_idx];
                let expected_ids = flat_i64(&golden_call["input_ids"]);
                let expected_mask = flat_i64(&golden_call["attention_mask"]);
                assert_eq!(
                    enc.input_ids, expected_ids,
                    "input_ids mismatch for {:?}",
                    rec["input"]
                );
                assert_eq!(
                    enc.attention_mask, expected_mask,
                    "attention_mask mismatch for {:?}",
                    rec["input"]
                );
                let logits = model.run(&enc).unwrap();
                let expected_logits = flat_f32(&golden_call["logits"]);
                assert_eq!(logits.len(), expected_logits.len(), "logits shape for {:?}", rec["input"]);
                for (g, e) in logits.iter().zip(&expected_logits) {
                    assert!((g - e).abs() <= 1e-4, "logit diff too big: {g} vs {e}");
                }
                call_idx += 1;
            }
            assert_eq!(call_idx, nn_calls.len(), "mismatched yo_nn_calls count for {:?}", rec["input"]);
        }
    }

    #[test]
    fn yo_entities_match_golden() {
        let Some(dir) = tiny() else {
            eprintln!("RUACCENT_TINY_WORKDIR not set; skipping");
            return;
        };
        let mut model = model(&dir);
        for rec in &records() {
            let sentences = sentences_of(rec);
            let words = words_of(rec);
            let entities = rec["yo_entities"].as_array().unwrap();
            let mut call_idx = 0usize;
            for i in 0..words.len() {
                if words[i].is_empty() {
                    continue;
                }
                let lower = sentences[i].to_lowercase();
                if !lower.contains('е') {
                    continue;
                }
                let got = model.predict(&lower).unwrap();
                let expected = entities[call_idx].as_array().unwrap();
                assert_eq!(got.len(), expected.len(), "entity count mismatch for {:?}", rec["input"]);
                for (g, e) in got.iter().zip(expected.iter()) {
                    assert_eq!(g.word, e["word"].as_str().unwrap(), "word mismatch for {:?}", rec["input"]);
                    assert_eq!(g.entity, e["entity"].as_str().unwrap(), "entity mismatch for {:?}", rec["input"]);
                    assert_eq!(g.start, e["start"].as_u64().unwrap() as usize, "start mismatch for {:?}", rec["input"]);
                    assert_eq!(g.end, e["end"].as_u64().unwrap() as usize, "end mismatch for {:?}", rec["input"]);
                    assert!(
                        (g.score - e["score"].as_f64().unwrap() as f32).abs() <= 1e-5,
                        "score diff too big: {} vs {}",
                        g.score,
                        e["score"].as_f64().unwrap()
                    );
                }
                call_idx += 1;
            }
            assert_eq!(call_idx, entities.len(), "mismatched yo_entities count for {:?}", rec["input"]);
        }
    }

    #[test]
    fn process_yo_matches_golden() {
        let Some(dir) = tiny() else {
            eprintln!("RUACCENT_TINY_WORKDIR not set; skipping");
            return;
        };
        let dicts = crate::dict::load(&dir).unwrap();
        let mut model = model(&dir);
        for rec in &records() {
            let sentences = sentences_of(rec);
            let words = words_of(rec);
            let yo_words = rec["yo_words"].as_array().unwrap();
            let mut yo_idx = 0usize;
            for i in 0..words.len() {
                if words[i].is_empty() {
                    continue;
                }
                let got = process_yo(&words[i], &sentences[i], &dicts, &mut model);
                let expected: Vec<String> = yo_words[yo_idx]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect();
                assert_eq!(got, expected, "yo_words mismatch for {:?}", rec["input"]);
                yo_idx += 1;
            }
            assert_eq!(yo_idx, yo_words.len(), "mismatched yo_words count for {:?}", rec["input"]);
        }
    }

    #[test]
    fn unk_group_word_is_original_span() {
        let pre = PreEntity {
            id: 1,
            word: "🙂".to_string(),
            scores: vec![0.1, 0.6, 0.3],
            start: 0,
            end: 1,
            is_subword: false,
        };
        let id2label = vec!["NO_YO".to_string(), "PUNCT".to_string(), "YO".to_string()];
        let tokenizer = Tokenizer::new(tokenizers::models::wordpiece::WordPiece::default());
        let entity = aggregate_word(&[&pre], &id2label, &tokenizer, 1);
        assert_eq!(entity.word, "🙂");
    }
}
