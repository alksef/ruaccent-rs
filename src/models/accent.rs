//! nn_accent: ONNX-инференс поверх CharTokenizer, порт `AccentModel`
//! из `ruaccent/accent_model.py` (put_accent + render_stress).

use std::fs;
use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;

use crate::tokenization::char::CharTokenizer;

#[derive(Debug)]
pub enum AccentModelError {
    NotFound { file: &'static str },
    BadConfig { reason: String },
    Ort { step: &'static str, source: ort::Error },
}

impl std::fmt::Display for AccentModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AccentModelError::NotFound { file } => write!(f, "file not found: {file}"),
            AccentModelError::BadConfig { reason } => write!(f, "bad nn_accent config: {reason}"),
            AccentModelError::Ort { step, source } => write!(f, "ort error at {step}: {source}"),
        }
    }
}

impl std::error::Error for AccentModelError {}

impl From<crate::tokenization::char::TokenizerError> for AccentModelError {
    fn from(e: crate::tokenization::char::TokenizerError) -> Self {
        AccentModelError::BadConfig { reason: e.to_string() }
    }
}

pub struct AccentModel {
    session: Session,
    /// Индекс класса -> label (NO / STRESS_PRIMARY / STRESS_SECONDARY).
    id2label: Vec<String>,
    tokenizer: CharTokenizer,
}

const SCORE_THRESHOLD: f32 = 0.55;

impl AccentModel {
    /// Загрузить сессию, id2label и токенизатор из каталога nn_accent.
    pub fn load(model_path: &Path, config_path: &Path, vocab_path: &Path) -> Result<Self, AccentModelError> {
        let raw = fs::read_to_string(config_path).map_err(|_| AccentModelError::NotFound {
            file: "nn_accent/config.json",
        })?;
        let cfg: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| AccentModelError::BadConfig { reason: e.to_string() })?;
        let id2label_map = cfg
            .get("id2label")
            .and_then(|v| v.as_object())
            .ok_or_else(|| AccentModelError::BadConfig { reason: "no id2label".into() })?;
        let n = id2label_map.len();
        let mut id2label = vec![String::new(); n];
        for (idx, label) in id2label_map {
            let i: usize = idx
                .parse()
                .map_err(|_| AccentModelError::BadConfig { reason: format!("bad class index {idx}") })?;
            id2label[i] = label
                .as_str()
                .ok_or_else(|| AccentModelError::BadConfig { reason: "non-string label".into() })?
                .to_string();
        }

        let session = Session::builder()
            .and_then(|mut b| b.commit_from_file(model_path))
            .map_err(|e| AccentModelError::Ort { step: "session build", source: e })?;
        let tokenizer = CharTokenizer::from_vocab_file(vocab_path)?;
        Ok(AccentModel { session, id2label, tokenizer })
    }

    /// Прогнать слово через модель, вернуть logits [len+2, n_classes].
    pub fn logits(&mut self, word: &str) -> Result<Vec<f32>, AccentModelError> {
        let enc = self.tokenizer.encode_word(word);
        let n = enc.input_ids.len();
        let ids = Tensor::from_array(([1usize, n], enc.input_ids.clone()))
            .map_err(|e| AccentModelError::Ort { step: "input_ids tensor", source: e })?;
        let mask = Tensor::from_array(([1usize, n], enc.attention_mask.clone()))
            .map_err(|e| AccentModelError::Ort { step: "attention_mask tensor", source: e })?;
        let tt = Tensor::from_array(([1usize, n], enc.token_type_ids.clone()))
            .map_err(|e| AccentModelError::Ort { step: "token_type_ids tensor", source: e })?;
        let outputs = self
            .session
            .run(ort::inputs!["input_ids" => ids, "attention_mask" => mask, "token_type_ids" => tt])
            .map_err(|e| AccentModelError::Ort { step: "run", source: e })?;
        let (_shape, data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| AccentModelError::Ort { step: "extract logits", source: e })?;
        Ok(data.to_vec())
    }

    /// Порт put_accent: разметка `+` перед ударной гласной.
    pub fn put_accent(&mut self, word: &str) -> String {
        let logits = match self.logits(word) {
            Ok(l) => l,
            Err(_) => return word.to_string(),
        };
        let n_classes = self.id2label.len();
        let chars: Vec<char> = word.chars().collect();
        let mut out = String::with_capacity(word.len() + 2);
        // Позиция i соответствует i-му символу слова (первые два логита —
        // bos и первый символ; вставка идёт перед text[i-1] в Python).
        for (i, ch) in chars.iter().enumerate() {
            let row = &logits[(i + 1) * n_classes..(i + 2) * n_classes];
            let (best, &logit) = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .expect("non-empty class row");
            let label = self.id2label.get(best).map(String::as_str).unwrap_or("NO");
            // softmax только для winning score: p = e^l / sum(e^lj)
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|v| (v - max).exp()).sum();
            let score = (logit - max).exp() / sum;
            if label != "NO" && label != "STRESS_SECONDARY" && score >= SCORE_THRESHOLD && i > 0 {
                out.push('+');
            }
            out.push(*ch);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tiny() -> Option<PathBuf> {
        std::env::var("RUACCENT_TINY_WORKDIR").ok().map(Into::into)
    }

    fn model(dir: &PathBuf) -> AccentModel {
        let nn = dir.join("nn/nn_accent");
        AccentModel::load(&nn.join("model.onnx"), &nn.join("config.json"), &nn.join("vocab.txt")).unwrap()
    }

    #[test]
    fn logits_match_golden_within_1e4() {
        let Some(dir) = tiny() else { eprintln!("skip"); return };
        let mut m = model(&dir);
        let golden = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/golden/tiny.jsonl")).unwrap();
        let rec = golden
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .find(|r| r["accent_nn_calls"].as_array().map(|a| !a.is_empty()).unwrap_or(false))
            .expect("golden record with accent_nn_calls");
        let call = &rec["accent_nn_calls"][0];
        assert_eq!(call["word"].as_str().unwrap(), "кварделямбу");
        let expected: Vec<f32> = call["logits"][0]
            .as_array()
            .expect("logits rows")
            .iter()
            .flat_map(|row| row.as_array().expect("logit row").iter())
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let got = m.logits("кварделямбу").unwrap();
        assert_eq!(got.len(), expected.len(), "logits shape");
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() <= 1e-4, "logit diff too big: {g} vs {e}");
        }
    }

    #[test]
    fn put_accent_matches_golden_string() {
        let Some(dir) = tiny() else { eprintln!("skip"); return };
        let mut m = model(&dir);
        assert_eq!(m.put_accent("кварделямбу"), "квардел+ямбу");
        assert_eq!(m.put_accent("дверь"), "дв+ерь");
    }
}
