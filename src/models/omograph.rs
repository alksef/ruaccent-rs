//! nn_omograph: выбор варианта омографа. Порт `OmographModel.classify` из
//! `ruaccent/omograph_model.py` поверх WordPiece-токенизатора
//! (`tokenizer.json`) и ONNX-сессии (`model.onnx`).
//!
//! Пайплайн одной фразы: `_process_omographs` помечает слово-омограф
//! маркерами ` <w>…</w> ` в списке слов, склеивает через пробел и чистит
//! пробелы перед пунктуацией (`delete_spaces_before_punc`); затем `classify`
//! сжимает оставшиеся пробелы перед `[,.?!:;…]`, токенизирует пары
//! (текст, гипотеза) и выбирает вариант с максимальной вероятностью класса
//! «ударение истинно» (индекс 1). В чётной ветке выбор сводится к сравнению
//! сырых logits класса 1: эталон нормирует весь батч одной глобальной softmax
//! (`omograph_model.py:16–18, 90–108`), поэтому `outputs[i][1]` монотонен по
//! сырому logit — это осознанный порт эталона, а не ошибка.

use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::text::delete_spaces_before_punc;

/// Найденный в тексте омограф: слово (lowercase), варианты с `+` и позиция
/// в списке слов предложения.
#[derive(Debug, Clone)]
pub struct OmographCandidate {
    pub word: String,
    pub variants: Vec<String>,
    pub position: usize,
}

#[derive(Debug)]
pub enum OmographModelError {
    NotFound { file: &'static str },
    Tokenizer { reason: String },
    Ort { step: &'static str, source: ort::Error },
}

impl std::fmt::Display for OmographModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OmographModelError::NotFound { file } => write!(f, "file not found: {file}"),
            OmographModelError::Tokenizer { reason } => write!(f, "tokenizer error: {reason}"),
            OmographModelError::Ort { step, source } => write!(f, "ort error at {step}: {source}"),
        }
    }
}

impl std::error::Error for OmographModelError {}

/// Токенизированный батч (пары «текст, гипотеза»), готовый к ONNX-инференсу.
#[derive(Debug, Clone)]
struct EncodedBatch {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    shape: [usize; 2],
}

pub struct OmographModel {
    session: Session,
    tokenizer: Tokenizer,
}

/// Слова, для которых `group_words` разбивает большие группы на подгруппы
/// по 3 (нечётная ветка `classify`).
const SPECIAL_WORDS: &[&str] = &[
    "балчуга",
    "вертела",
    "волоки",
    "волоку",
    "воронью",
    "выбродите",
    "вывозите",
    "выносите",
    "выноситесь",
    "выходите",
    "железы",
    "начала",
    "округа",
    "перепела",
    "развитая",
    "развитого",
    "развитое",
    "развитой",
    "развитом",
    "развитому",
    "развитою",
    "развитую",
    "развитые",
    "развитым",
    "развитыми",
    "развитых",
    "сторожа",
    "сторожи",
    "сторожу",
    "удало",
    "начался",
    "началась",
    "началось",
    "бутиках",
    "ожила",
    "создало",
    "коротки",
    "проклята",
    "роженица",
    "роженицы",
    "рожениц",
    "роженице",
    "роженицам",
    "роженицу",
    "роженицей",
    "роженицею",
    "роженицами",
    "роженицах",
    "пристава",
    "приставов",
    "приставам",
    "приставами",
    "приставах",
    "пережитое",
    "пережитого",
    "пережитые",
    "пережитых",
    "пережитому",
    "пережитым",
    "пережитыми",
    "пережитом",
    "нипоняла",
];

/// Символы из `re.sub(r'\s+(?=(?:[,.?!:;…]))', '', text)`.
const PUNC_LOOKAHEAD: &[char] = &[',', '.', '?', '!', ':', ';', '…'];

impl OmographModel {
    /// Загрузить ONNX-сессию и WordPiece-токенизатор из каталога модели.
    pub fn load(dir: &Path) -> Result<Self, OmographModelError> {
        let model_path = dir.join("model.onnx");
        let tokenizer_path = dir.join("tokenizer.json");

        let session = Session::builder()
            .and_then(|mut b| b.commit_from_file(&model_path))
            .map_err(|e| OmographModelError::Ort { step: "session build", source: e })?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| OmographModelError::Tokenizer { reason: e.to_string() })?;
        tokenizer
            .with_truncation(Some(TruncationParams::default()))
            .map_err(|e| OmographModelError::Tokenizer { reason: e.to_string() })?;
        tokenizer.with_padding(Some(PaddingParams::default()));

        Ok(OmographModel { session, tokenizer })
    }

    /// Выбрать форму для каждого кандидата (порт `_process_omographs` +
    /// `classify`). Возвращает выбранный вариант в порядке кандидатов.
    pub fn classify(
        &mut self,
        text_words: &[String],
        candidates: &[OmographCandidate],
    ) -> Vec<String> {
        if candidates.is_empty() {
            return Vec::new();
        }
        let (texts, hypotheses, num_hypotheses) = build_batches(text_words, candidates);
        match self.select(&texts, &hypotheses, &num_hypotheses) {
            Ok(outs) => outs,
            Err(_) => candidates
                .iter()
                .map(|c| c.variants.first().cloned().unwrap_or_default())
                .collect(),
        }
    }

    /// Токенизировать батч пар (текст, гипотеза) с паддингом по максимуму.
    fn tokenize_pairs(
        &self,
        texts: &[String],
        hypotheses: &[String],
    ) -> Result<EncodedBatch, OmographModelError> {
        let pairs: Vec<(String, String)> = texts
            .iter()
            .cloned()
            .zip(hypotheses.iter().cloned())
            .collect();
        let encodings = self
            .tokenizer
            .encode_batch(pairs, true)
            .map_err(|e| OmographModelError::Tokenizer { reason: e.to_string() })?;
        let batch = encodings.len();
        let seq = encodings.first().map(|e| e.len()).unwrap_or(0);
        let mut input_ids = Vec::with_capacity(batch * seq);
        let mut attention_mask = Vec::with_capacity(batch * seq);
        for enc in &encodings {
            input_ids.extend(enc.get_ids().iter().map(|&id| id as i64));
            attention_mask.extend(enc.get_attention_mask().iter().map(|&m| m as i64));
        }
        Ok(EncodedBatch { input_ids, attention_mask, shape: [batch, seq] })
    }

    /// Прогнать батч через модель, вернуть сырые logits [batch, 2] (flat).
    fn run(&mut self, batch: &EncodedBatch) -> Result<Vec<f32>, OmographModelError> {
        let ids = Tensor::from_array((batch.shape, batch.input_ids.clone()))
            .map_err(|e| OmographModelError::Ort { step: "input_ids tensor", source: e })?;
        let mask = Tensor::from_array((batch.shape, batch.attention_mask.clone()))
            .map_err(|e| OmographModelError::Ort { step: "attention_mask tensor", source: e })?;
        let outputs = self
            .session
            .run(ort::inputs!["input_ids" => ids, "attention_mask" => mask])
            .map_err(|e| OmographModelError::Ort { step: "run", source: e })?;
        let (_shape, data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| OmographModelError::Ort { step: "extract logits", source: e })?;
        Ok(data.to_vec())
    }

    /// Порт `classify(texts, hypotheses, num_hypotheses)`.
    fn select(
        &mut self,
        preprocessed_texts: &[String],
        hypotheses: &[String],
        num_hypotheses: &[usize],
    ) -> Result<Vec<String>, OmographModelError> {
        if num_hypotheses.iter().all(|n| n % 2 == 0) {
            let batch = self.tokenize_pairs(preprocessed_texts, hypotheses)?;
            let logits = self.run(&batch)?;
            let mut outs = Vec::with_capacity(hypotheses.len() / 2);
            for k in (0..hypotheses.len()).step_by(2) {
                let chosen =
                    select_pair_variant(logits[2 * k + 1], logits[2 * (k + 1) + 1]);
                outs.push(hypotheses[k + chosen].clone());
            }
            Ok(outs)
        } else {
            let grouped_h = group_words(hypotheses);
            let grouped_t = transfer_grouping(&grouped_h, preprocessed_texts);
            let mut outs = Vec::new();
            for (h, t) in grouped_h.iter().zip(grouped_t.iter()) {
                let mut probs = Vec::with_capacity(h.len());
                for hp in h {
                    let batch =
                        self.tokenize_pairs(std::slice::from_ref(&t[0]), std::slice::from_ref(hp))?;
                    let logits = self.run(&batch)?;
                    probs.push(softmax_class1(logits[0], logits[1]));
                }
                let mut best = 0usize;
                for (i, p) in probs.iter().enumerate().skip(1) {
                    if *p > probs[best] {
                        best = i;
                    }
                }
                outs.push(h[best].clone());
            }
            Ok(outs)
        }
    }
}

/// Построить батчи текстов и гипотез как `_process_omographs` +
/// `classify` (тексты уже после `delete_spaces_before_punc` и сжатия
/// пробелов перед `[,.?!:;…]`).
fn build_batches(
    text_words: &[String],
    candidates: &[OmographCandidate],
) -> (Vec<String>, Vec<String>, Vec<usize>) {
    let mut texts = Vec::new();
    let mut hypotheses = Vec::new();
    let mut num_hypotheses = Vec::with_capacity(candidates.len());
    for cand in candidates {
        num_hypotheses.push(cand.variants.len());
        let mut marked = text_words.to_vec();
        marked[cand.position] = format!(" <w>{}</w> ", text_words[cand.position]);
        let text = strip_ws_before_punc(&delete_spaces_before_punc(&marked.join(" ")));
        for _ in 0..cand.variants.len() {
            texts.push(text.clone());
        }
        for v in &cand.variants {
            hypotheses.push(v.clone());
        }
    }
    (texts, hypotheses, num_hypotheses)
}

/// Порт `re.sub(r'\s+(?=(?:[,.?!:;…]))', '', text)`.
fn strip_ws_before_punc(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_whitespace() {
            let mut j = i;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && PUNC_LOOKAHEAD.contains(&chars[j]) {
                // пробелы перед пунктуацией удаляются
            } else {
                out.extend(&chars[i..j]);
            }
            i = j;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Порт `OmographModel.group_words`: группирует гипотезы по основе слова и
/// дробит группы длиннее трёх на подгруппы по 3 (special_words) или по 2.
fn group_words(words: &[String]) -> Vec<Vec<String>> {
    if words.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut current_group = vec![words[0].clone()];
    let mut current_base = words[0].replace('+', "");
    for word in &words[1..] {
        let base_word = word.replace('+', "");
        if base_word == current_base {
            current_group.push(word.clone());
        } else {
            push_group(&mut result, &current_group, &current_base);
            current_group = vec![word.clone()];
            current_base = base_word;
        }
    }
    push_group(&mut result, &current_group, &current_base);
    result
}

fn push_group(result: &mut Vec<Vec<String>>, group: &[String], base: &str) {
    if SPECIAL_WORDS.contains(&base) && group.len() > 3 {
        result.extend(group.chunks(3).map(|c| c.to_vec()));
    } else if group.len() > 3 && group.len().is_multiple_of(2) {
        result.extend(group.chunks(2).map(|c| c.to_vec()));
    } else {
        result.push(group.to_vec());
    }
}

/// Порт `OmographModel.transfer_grouping`: перегруппировать `target` по
/// размерам групп из `grouped`.
fn transfer_grouping<T: Clone>(grouped: &[Vec<String>], target: &[T]) -> Vec<Vec<T>> {
    let mut result = Vec::new();
    let mut start = 0;
    for group in grouped {
        let len = group.len();
        result.push(target[start..start + len].to_vec());
        start += len;
    }
    result
}

/// Вероятность класса «label is true» (индекс 1) после softmax по двум
/// классам, как `OmographModel.softmax` + `outputs[i][1]`.
fn softmax_class1(l0: f32, l1: f32) -> f32 {
    let max = l0.max(l1);
    let e0 = (l0 - max).exp();
    let e1 = (l1 - max).exp();
    e1 / (e0 + e1)
}

/// Вернуть индекс выбранного варианта (0 или 1) для пары строк logits.
/// Чётная ветка `select` портирует сравнение `outputs[i][1]` эталона: из-за
/// глобальной softmax по всему батчу (`omograph_model.py:16–18`) они монотонны
/// по сырым logits класса 1, поэтому выбор эквивалентен сравнению `l1_first`
/// и `l1_second`. Равенство → первый вариант (`pair2.index(max(pair2))`,
/// `omograph_model.py:90–108`).
fn select_pair_variant(l1_first: f32, l1_second: f32) -> usize {
    usize::from(l1_second > l1_first)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tiny() -> Option<PathBuf> {
        std::env::var("RUACCENT_TINY_WORKDIR").ok().map(Into::into)
    }

    fn model(dir: &PathBuf) -> OmographModel {
        let nn = dir.join("nn/nn_omograph/tiny");
        OmographModel::load(&nn).unwrap()
    }

    fn omograph_records() -> Vec<serde_json::Value> {
        let golden = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/golden/tiny.jsonl"))
            .unwrap();
        golden
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .filter(|r| {
                r["omograph_candidates"]
                    .as_array()
                    .map(|a| !a.is_empty())
                    .unwrap_or(false)
            })
            .collect()
    }

    fn candidates_of(rec: &serde_json::Value) -> Vec<OmographCandidate> {
        rec["omograph_candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| OmographCandidate {
                word: c["word"].as_str().unwrap().to_string(),
                variants: c["variants"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect(),
                position: c["position"].as_u64().unwrap() as usize,
            })
            .collect()
    }

    fn words_of(rec: &serde_json::Value) -> Vec<String> {
        rec["yo_words"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
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

    fn flat_f32(v: &serde_json::Value) -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter())
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }

    #[test]
    fn input_ids_match_golden_bitwise() {
        let Some(dir) = tiny() else {
            eprintln!("RUACCENT_TINY_WORKDIR not set; skipping");
            return;
        };
        let model = model(&dir);
        let records = omograph_records();
        assert_eq!(records.len(), 14, "expected 14 records with omographs");
        for rec in &records {
            let words = words_of(rec);
            let candidates = candidates_of(rec);
            let (texts, hypotheses, num_hypotheses) = build_batches(&words, &candidates);
            assert!(
                num_hypotheses.iter().all(|n| n % 2 == 0),
                "golden corpus uses only even batches"
            );
            let batch = model.tokenize_pairs(&texts, &hypotheses).unwrap();

            let golden_inputs = &rec["omograph_inputs"][0];
            let expected_ids = flat_i64(&golden_inputs["input_ids"]);
            let expected_mask = flat_i64(&golden_inputs["attention_mask"]);
            assert_eq!(
                batch.input_ids, expected_ids,
                "input_ids mismatch for {:?}",
                rec["input"]
            );
            assert_eq!(
                batch.attention_mask, expected_mask,
                "attention_mask mismatch for {:?}",
                rec["input"]
            );
        }
    }

    #[test]
    fn classify_matches_golden_resolved() {
        let Some(dir) = tiny() else {
            eprintln!("RUACCENT_TINY_WORKDIR not set; skipping");
            return;
        };
        let mut model = model(&dir);
        for rec in &omograph_records() {
            let words = words_of(rec);
            let candidates = candidates_of(rec);
            let selected = model.classify(&words, &candidates);
            let resolved = rec["omograph_resolved"][0].as_array().unwrap();
            assert_eq!(selected.len(), candidates.len());
            for (i, cand) in candidates.iter().enumerate() {
                assert_eq!(
                    selected[i],
                    resolved[cand.position].as_str().unwrap(),
                    "record {:?} candidate {}",
                    rec["input"],
                    i
                );
            }
        }
    }

    #[test]
    fn logits_match_golden_within_1e4() {
        let Some(dir) = tiny() else {
            eprintln!("RUACCENT_TINY_WORKDIR not set; skipping");
            return;
        };
        let mut model = model(&dir);
        for rec in &omograph_records() {
            let words = words_of(rec);
            let candidates = candidates_of(rec);
            let (texts, hypotheses, _) = build_batches(&words, &candidates);
            let batch = model.tokenize_pairs(&texts, &hypotheses).unwrap();
            let logits = model.run(&batch).unwrap();
            let expected = flat_f32(&rec["omograph_inputs"][0]["logits"]);
            assert_eq!(logits.len(), expected.len());
            for (g, e) in logits.iter().zip(&expected) {
                assert!((g - e).abs() <= 1e-4, "logit diff too big: {g} vs {e}");
            }
        }
    }

    #[test]
    fn select_pair_variant_uses_raw_logit_class1() {
        let chosen = select_pair_variant(3.0, 2.9);
        assert_eq!(
            chosen, 0,
            "row A logits (0.0, 3.0), row B (-5.0, 2.9): per-row softmax would pick \
             B, but the reference global softmax picks A (raw class-1 logit 3.0 > 2.9)"
        );
    }

    #[test]
    fn select_pair_variant_tie_keeps_first() {
        assert_eq!(select_pair_variant(3.0, 3.0), 0);
    }
}
