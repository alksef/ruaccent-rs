//! Text layer of the `ruaccent-rs` port.
//!
//! Faithful ports of ruaccent's text splitting routines:
//! * [`split_by_sentences`] — razdel `sentenize` + gluing of the text between
//!   and after sentences (`TextPreprocessor.split_by_sentences`).
//! * [`split_by_words`] — exact port of `TextPreprocessor.split_by_words`,
//!   including the `" - "` -> `" ~ "` replacement and `remaining_text`
//!   computation.

mod sentenize;
mod words;

use std::sync::OnceLock;

use regex::Regex;

pub use sentenize::split_by_sentences;
pub use words::split_by_words;

/// Порт `RUAccent.delete_spaces_before_punc`: убирает пробелы перед
/// пунктуацией и заменяет `~` на `-`.
pub fn delete_spaces_before_punc(text: &str) -> String {
    const PUNCTUATION: &str = "!\"#$%&'()*,./:;<=>?@[\\]^_`{|}-";
    let mut out = text.to_string();
    for c in PUNCTUATION.chars() {
        if c == '-' {
            out = out.replace(" -", "-").replace("- ", "-");
        }
        out = out.replace(&format!(" {c}"), &c.to_string());
    }
    out.replace('~', "-")
}

/// Порт `self.normalize` (`re.sub(r"[^...]", "", text)`): удаляет все символы
/// вне класса `[a-zA-Z0-9\sа-яА-ЯёЁ—.,!?:;""''(){}\[\]«»„“”-]`.
pub fn normalize(text: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(r#"[^a-zA-Z0-9\sа-яА-ЯёЁ—.,!?:;""''(){}\[\]«»„“”-]"#)
            .expect("normalize regex is valid")
    });
    re.replace_all(text, "").into_owned()
}

/// Port of `text_postprocessor.fix_capital`: переносит регистр `source` на
/// `target` посимвольно (когда длины в символах совпадают). При несовпадении
/// длины возвращает `target` как есть.
pub fn fix_capital(source: &str, target: &str) -> String {
    let src: Vec<char> = source.chars().collect();
    let tgt: Vec<char> = target.chars().collect();
    if src.len() != tgt.len() {
        return target.to_string();
    }
    let mut out = String::with_capacity(target.len());
    for (s, t) in src.iter().zip(tgt.iter()) {
        if s.is_uppercase() {
            out.extend(t.to_uppercase());
        } else {
            out.extend(t.to_lowercase());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::fix_capital;

    #[test]
    fn keeps_first_letter_capital() {
        assert_eq!(fix_capital("Осел", "осёл"), "Осёл");
    }

    #[test]
    fn all_lowercase_stays_lowercase() {
        assert_eq!(fix_capital("все", "всё"), "всё");
    }

    #[test]
    fn all_uppercase_uppercases_target() {
        assert_eq!(fix_capital("ВСЕ", "всё"), "ВСЁ");
    }

    #[test]
    fn different_length_returns_target() {
        assert_eq!(fix_capital("слово", "ёж"), "ёж");
    }
}
