//! Port of razdel `sentenize` plus ruaccent's `split_by_sentences` wrapper.
//!
//! The relevant reference files are `razdel/segmenters/{base,sentenize,punct,sokr}.py`
//! and `ruaccent/text_preprocessor.py` (`split_by_sentences`). Delimiters and
//! join rules are reproduced faithfully; the abbreviation (sokr) dictionaries
//! are inlined from `razdel/segmenters/sokr.py`.

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// punct.py constants
// ---------------------------------------------------------------------------

const ENDINGS: &str = ".?!…";
const DASHES: &str = "‑–—−-";
const _OPEN_QUOTES: &str = "«“‘";
const CLOSE_QUOTES: &str = "»”’";
const GENERIC_QUOTES: &str = "\"„'";
const CLOSE_BRACKETS: &str = ")]}";
const QUOTES: &str = "«“‘»”’\"„'";

const BULLET_BOUNDS: &str = ".)";
const BULLET_SIZE: usize = 20;

/// `ENDINGS + ';' + GENERIC_QUOTES + CLOSE_QUOTES + CLOSE_BRACKETS`.
const DELIMITERS: &str = ".?!…;\"„'»”’)]}";

const SMILES_HEAD: &str = "=:;";

const WINDOW: usize = 10;

// ---------------------------------------------------------------------------
// Substring (razdel.substring.Substring)
// ---------------------------------------------------------------------------

struct Substring {
    start: usize,
    stop: usize,
    text: String,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Port of `TextPreprocessor.split_by_sentences`:
/// razdel `sentenize` followed by gluing the text between/after sentences onto
/// the following/last sentence.
pub fn split_by_sentences(text: &str) -> Vec<String> {
    let sentences = sentenize(text);
    if sentences.is_empty() {
        return Vec::new();
    }

    let chars: Vec<char> = text.chars().collect();
    let mut result: Vec<String> = Vec::with_capacity(sentences.len());

    for (i, r) in sentences.iter().enumerate() {
        let l_stop = if i == 0 { 0 } else { sentences[i - 1].stop };
        if l_stop != r.start {
            let mut piece = slice_chars(&chars, l_stop, r.start);
            piece.push_str(&r.text);
            result.push(piece);
        } else {
            result.push(r.text.clone());
        }
    }

    let last_stop = sentences[sentences.len() - 1].stop;
    let tail = slice_chars(&chars, last_stop, chars.len());
    if let Some(last) = result.last_mut() {
        last.push_str(&tail);
    }

    result
}

// ---------------------------------------------------------------------------
// sentenize
// ---------------------------------------------------------------------------

fn sentenize(text: &str) -> Vec<Substring> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();

    let delims = find_delimiters(&chars);

    // Splitter: yield C0, Split0, C1, Split1, ..., Cn where Ck are text chunks
    // and Splitk is the delimiter at position k. `segment` below folds them.
    let mut buffer = slice_chars(&chars, 0, delims.first().map(|d| d.0).unwrap_or(n));

    let mut chunks: Vec<String> = Vec::new();

    for (idx, &(start, stop, ref delim)) in delims.iter().enumerate() {
        let next_start = delims.get(idx + 1).map(|d| d.0).unwrap_or(n);
        let right = slice_chars(&chars, stop, next_start);

        let left_start = start.saturating_sub(WINDOW);
        let left: Vec<char> = chars[left_start..start].to_vec();
        let right_win_end = (stop + WINDOW).min(n);
        let right_win: Vec<char> = chars[stop..right_win_end].to_vec();

        let split = Split {
            delimiter: delim.clone(),
            left,
            right: right_win,
            buffer: buffer.clone(),
        };

        if joins(&split) {
            buffer.push_str(delim);
            buffer.push_str(&right);
        } else {
            let mut sentence = std::mem::take(&mut buffer);
            sentence.push_str(delim);
            chunks.push(sentence);
            buffer = right;
        }
    }
    chunks.push(buffer);

    // post: strip each chunk, then map back onto the original text.
    let mut result = Vec::with_capacity(chunks.len());
    let mut offset = 0usize;
    for chunk in &chunks {
        let stripped = chunk.trim().to_string();
        let (start, stop) = find_chars(&chars, &stripped.chars().collect::<Vec<_>>(), offset);
        result.push(Substring {
            start,
            stop,
            text: stripped,
        });
        offset = stop;
    }

    result
}

// ---------------------------------------------------------------------------
// Delimiter scanning (razdel.split.SentSplitter)
// ---------------------------------------------------------------------------

fn find_delimiters(chars: &[char]) -> Vec<(usize, usize, String)> {
    let n = chars.len();
    let mut delims = Vec::new();
    let mut i = 0;
    while i < n {
        if let Some((end, smile)) = match_smile(chars, i) {
            delims.push((i, end, smile));
            i = end;
            continue;
        }
        if DELIMITERS.contains(chars[i]) {
            delims.push((i, i + 1, chars[i].to_string()));
        }
        i += 1;
    }
    delims
}

/// `SMILES = r'[=:;]-?[)(]{1,3}'` at position `i`. Returns the exclusive end and
/// the matched smile string.
fn match_smile(chars: &[char], i: usize) -> Option<(usize, String)> {
    if i >= chars.len() || !SMILES_HEAD.contains(chars[i]) {
        return None;
    }
    let mut j = i + 1;
    if j < chars.len() && chars[j] == '-' {
        j += 1;
    }
    let paren_start = j;
    while j < chars.len() && (chars[j] == ')' || chars[j] == '(') && j - paren_start < 3 {
        j += 1;
    }
    if j == paren_start {
        return None;
    }
    Some((j, chars[i..j].iter().collect()))
}

// ---------------------------------------------------------------------------
// Split + join rules (razdel.segmenters.base + razdel.segmenters.sentenize)
// ---------------------------------------------------------------------------

struct Split {
    delimiter: String,
    left: Vec<char>,
    right: Vec<char>,
    buffer: String,
}

fn joins(split: &Split) -> bool {
    let left_token = last_token(&split.left);
    let right_token = first_token(&split.right);

    // empty_side
    if left_token.is_none() || right_token.is_none() {
        return true;
    }
    let left_token = left_token.unwrap();
    let right_token = right_token.unwrap();

    // no_space_prefix
    if !right_space_prefix(&split.right) {
        return true;
    }

    // lower_right
    if is_lower_alpha(&right_token) {
        return true;
    }

    // delimiter_right
    {
        let right_str: String = right_token.iter().collect();
        if !GENERIC_QUOTES.contains(right_str.as_str())
            && (DELIMITERS.contains(right_str.as_str()) || smile_prefix(&split.right))
        {
            return true;
        }
    }

    // sokr_left
    if rule_sokr_left(split, &left_token, &right_token) {
        return true;
    }
    // inside_pair_sokr
    if rule_inside_pair_sokr(split, &left_token, &right_token) {
        return true;
    }
    // initials_left
    if rule_initials_left(split, &left_token) {
        return true;
    }
    // list_item
    if rule_list_item(split) {
        return true;
    }
    // close_quote
    if rule_close_quote(split, &left_token) {
        return true;
    }
    // close_bracket
    if rule_close_bracket(split, &left_token) {
        return true;
    }
    // dash_right
    if rule_dash_right(split, &right_token) {
        return true;
    }

    false
}

fn is_lower_alpha(token: &[char]) -> bool {
    !token.is_empty()
        && token.iter().all(|c| c.is_alphabetic())
        && token.iter().all(|c| c.is_lowercase())
}

fn is_sokr(token: &[char]) -> bool {
    if token.is_empty() {
        return false;
    }
    if token.iter().all(|c| c.is_numeric()) {
        return true;
    }
    if !token.iter().all(|c| c.is_alphabetic()) {
        return true; // punctuation
    }
    token.iter().all(|c| c.is_lowercase())
}

fn close_bound(left_token: &[char]) -> bool {
    let s: String = left_token.iter().collect();
    !ENDINGS.contains(s.as_str())
}

fn rule_sokr_left(split: &Split, left_token: &[char], right_token: &[char]) -> bool {
    if split.delimiter != "." {
        return false;
    }
    if let Some((a, b)) = pair_sokr(&split.left) {
        let pair = (a.to_lowercase().to_string(), b.to_lowercase().to_string());
        if head_pair_sokrs().contains(&pair) {
            return true;
        }
        if pair_sokrs().contains(&pair) {
            return is_sokr(right_token);
        }
    }
    let left_lower = lower_string(left_token);
    if head_sokrs().contains(&left_lower) {
        return true;
    }
    if sokrs().contains(&left_lower) && is_sokr(right_token) {
        return true;
    }
    false
}

fn rule_inside_pair_sokr(split: &Split, left_token: &[char], right_token: &[char]) -> bool {
    if split.delimiter != "." {
        return false;
    }
    let pair = (lower_string(left_token), lower_string(right_token));
    pair_sokrs().contains(&pair)
}

fn rule_initials_left(split: &Split, left_token: &[char]) -> bool {
    if split.delimiter != "." {
        return false;
    }
    if left_token.len() == 1 && left_token[0].is_uppercase() {
        return true;
    }
    initials().contains(&lower_string(left_token))
}

fn rule_list_item(split: &Split) -> bool {
    if !BULLET_BOUNDS.contains(split.delimiter.as_str()) {
        return false;
    }
    if split.buffer.chars().count() > BULLET_SIZE {
        return false;
    }
    token_findall(&split.buffer).iter().all(|t| is_bullet(t))
}

fn rule_close_quote(split: &Split, left_token: &[char]) -> bool {
    if !QUOTES.contains(split.delimiter.as_str()) {
        return false;
    }
    if CLOSE_QUOTES.contains(split.delimiter.as_str()) {
        return close_bound(left_token);
    }
    if GENERIC_QUOTES.contains(split.delimiter.as_str()) {
        if !left_space_suffix(&split.left) {
            return close_bound(left_token);
        }
        return true;
    }
    false
}

fn rule_close_bracket(split: &Split, left_token: &[char]) -> bool {
    if CLOSE_BRACKETS.contains(split.delimiter.as_str()) {
        return close_bound(left_token);
    }
    false
}

fn rule_dash_right(split: &Split, right_token: &[char]) -> bool {
    let rt: String = right_token.iter().collect();
    if !DASHES.contains(rt.as_str()) {
        return false;
    }
    if let Some(word) = first_word(&split.right) {
        if is_lower_alpha(&word) {
            return true;
        }
    }
    false
}

fn is_bullet(token: &[char]) -> bool {
    if token.is_empty() {
        return false;
    }
    if token.iter().all(|c| c.is_numeric()) {
        return true;
    }
    let s: String = token.iter().collect();
    if BULLET_BOUNDS.contains(s.as_str()) {
        return true;
    }
    let lower: String = token.iter().flat_map(|c| c.to_lowercase()).collect();
    if lower.chars().count() == 1 {
        if let Some(c) = lower.chars().next() {
            if bullet_chars().contains(&c) {
                return true;
            }
        }
    }
    roman_match(token)
}

fn roman_match(token: &[char]) -> bool {
    !token.is_empty() && token.iter().all(|c| "IVXML".contains(*c))
}

// ---------------------------------------------------------------------------
// Token regexes (razdel.segmenters.sentenize)
// ---------------------------------------------------------------------------

/// `FIRST_TOKEN = ^\s*([^\W\d]+|\d+|[^\w\s])`
fn first_token(s: &[char]) -> Option<Vec<char>> {
    let mut i = 0;
    while i < s.len() && is_space(s[i]) {
        i += 1;
    }
    if i >= s.len() {
        return None;
    }
    read_token(s, i)
}

/// `LAST_TOKEN = ([^\W\d]+|\d+|[^\w\s])\s*$`
fn last_token(s: &[char]) -> Option<Vec<char>> {
    let mut end = s.len();
    while end > 0 && is_space(s[end - 1]) {
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    read_token_back(s, end)
}

/// `WORD = ([^\W\d]+|\d+)` — first word-ish token anywhere.
fn first_word(s: &[char]) -> Option<Vec<char>> {
    let mut i = 0;
    while i < s.len() {
        if is_letter(s[i]) || is_digit(s[i]) {
            return Some(read_word_at(s, i));
        }
        i += 1;
    }
    None
}

/// `PAIR_SOKR = (\w)\s*\.\s*(\w)\s*$`
fn pair_sokr(s: &[char]) -> Option<(char, char)> {
    let mut pos = s.len();
    while pos > 0 && is_space(s[pos - 1]) {
        pos -= 1;
    }
    if pos == 0 || !is_word(s[pos - 1]) {
        return None;
    }
    let b = s[pos - 1];
    pos -= 1;
    while pos > 0 && is_space(s[pos - 1]) {
        pos -= 1;
    }
    if pos == 0 || s[pos - 1] != '.' {
        return None;
    }
    pos -= 1;
    while pos > 0 && is_space(s[pos - 1]) {
        pos -= 1;
    }
    if pos == 0 || !is_word(s[pos - 1]) {
        return None;
    }
    let a = s[pos - 1];
    Some((a, b))
}

/// `TOKEN = ([^\W\d]+|\d+|[^\w\s])` — all non-overlapping tokens.
fn token_findall(buffer: &str) -> Vec<Vec<char>> {
    let chars: Vec<char> = buffer.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if let Some(tok) = read_token(&chars, i) {
            i += tok.len();
            out.push(tok);
        } else {
            i += 1;
        }
    }
    out
}

/// Read a token starting at `i` (`[^\W\d]+|\d+|[^\w\s]`).
fn read_token(s: &[char], i: usize) -> Option<Vec<char>> {
    let c = s[i];
    if is_letter(c) {
        let mut j = i;
        while j < s.len() && is_letter(s[j]) {
            j += 1;
        }
        Some(s[i..j].to_vec())
    } else if is_digit(c) {
        let mut j = i;
        while j < s.len() && is_digit(s[j]) {
            j += 1;
        }
        Some(s[i..j].to_vec())
    } else if is_punct(c) {
        Some(vec![c])
    } else {
        None
    }
}

/// Read a token ending just before `end`.
fn read_token_back(s: &[char], end: usize) -> Option<Vec<char>> {
    let c = s[end - 1];
    if is_letter(c) {
        let mut j = end;
        while j > 0 && is_letter(s[j - 1]) {
            j -= 1;
        }
        Some(s[j..end].to_vec())
    } else if is_digit(c) {
        let mut j = end;
        while j > 0 && is_digit(s[j - 1]) {
            j -= 1;
        }
        Some(s[j..end].to_vec())
    } else if is_punct(c) {
        Some(vec![c])
    } else {
        None
    }
}

/// Read a word-ish run (`[^\W\d]+|\d+`) starting at `i`.
fn read_word_at(s: &[char], i: usize) -> Vec<char> {
    let mut j = i;
    if is_letter(s[i]) {
        while j < s.len() && is_letter(s[j]) {
            j += 1;
        }
    } else {
        while j < s.len() && is_digit(s[j]) {
            j += 1;
        }
    }
    s[i..j].to_vec()
}

// ---------------------------------------------------------------------------
// Anchors
// ---------------------------------------------------------------------------

/// `SPACE_PREFIX = ^\s`
fn right_space_prefix(right: &[char]) -> bool {
    right.first().is_some_and(|c| is_space(*c))
}

/// `SPACE_SUFFIX = \s$`
fn left_space_suffix(left: &[char]) -> bool {
    left.last().is_some_and(|c| is_space(*c))
}

/// `SMILE_PREFIX = ^\s*SMILES`
fn smile_prefix(right: &[char]) -> bool {
    let mut i = 0;
    while i < right.len() && is_space(right[i]) {
        i += 1;
    }
    match_smile(right, i).is_some()
}

// ---------------------------------------------------------------------------
// Character classes
// ---------------------------------------------------------------------------

fn is_letter(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

fn is_digit(c: char) -> bool {
    c.is_numeric()
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn is_space(c: char) -> bool {
    c.is_whitespace()
}

fn is_punct(c: char) -> bool {
    !is_word(c) && !is_space(c)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn lower_string(token: &[char]) -> String {
    token.iter().flat_map(|c| c.to_lowercase()).collect()
}

fn slice_chars(chars: &[char], start: usize, end: usize) -> String {
    chars[start..end].iter().collect()
}

/// `str.find(needle, offset)` over `Vec<char>`; returns `(start, stop)` char
/// indices. An empty needle is found at `offset` (mirroring Python).
fn find_chars(haystack: &[char], needle: &[char], offset: usize) -> (usize, usize) {
    if needle.is_empty() {
        return (offset.min(haystack.len()), offset.min(haystack.len()));
    }
    if haystack.len() < needle.len() {
        return (haystack.len(), haystack.len());
    }
    for i in offset..=haystack.len() - needle.len() {
        if &haystack[i..i + needle.len()] == needle {
            return (i, i + needle.len());
        }
    }
    (haystack.len(), haystack.len())
}

// ---------------------------------------------------------------------------
// sokr dictionaries (razdel.segmenters.sokr)
// ---------------------------------------------------------------------------

fn sokrs() -> HashSet<String> {
    let mut set = tail_sokrs();
    set.extend(head_sokrs());
    set.extend(other_sokrs());
    set
}

fn tail_sokrs() -> HashSet<String> {
    words(&[
        "дес тыс млн млрд",
        "дол долл",
        "коп руб р",
        "проц",
        "га",
        "барр",
        "куб",
        "кв км",
        "см",
        "час мин сек",
        "в вв",
        "г гг",
        "с стр",
        "co corp inc",
        "изд ed",
        "др",
        "al",
    ])
}

fn head_sokrs() -> HashSet<String> {
    words(&[
        "букв",
        "ст",
        "трад",
        "лат венг исп кат укр нем англ фр итал греч",
        "евр араб яп слав кит рус русск латв",
        "словацк хорв",
        "mr mrs ms dr vs",
        "св",
        "арх зав зам проф акад",
        "кн",
        "корр",
        "ред",
        "гр",
        "ср",
        "чл корр",
        "им",
        "тов",
        "нач пол",
        "chap",
        "п пп ст ч чч гл стр абз пт",
        "no",
        "просп пр ул ш г гор д стр к корп пер корп обл эт пом ауд оф ком комн каб",
        "домовлад лит",
        "т",
        "рп пос с х",
        "пл",
        "bd",
        "о оз",
        "р",
        "а",
        "обр",
        "ум",
        "ок",
        "откр",
        "пс ps",
        "upd",
        "см",
        "напр",
        "доп",
        "юр физ",
        "тел",
        "сб",
        "внутр",
        "дифф",
        "гос",
        "отм",
    ])
}

fn other_sokrs() -> HashSet<String> {
    words(&["сокр рис искл прим", "яз", "устар", "шутл"])
}

fn words(lines: &[&str]) -> HashSet<String> {
    lines
        .iter()
        .flat_map(|line| line.split_whitespace())
        .map(|w| w.to_string())
        .collect()
}

fn head_pair_sokrs() -> HashSet<(String, String)> {
    pairs(&["т е", "т к", "т н", "и о", "к н", "к п", "п н", "к т", "т н", "л д"])
}

fn pair_sokrs() -> HashSet<(String, String)> {
    let mut set = pairs(&[
        "т п", "т д", "у е", "н э", "p m", "a m", "с г", "р х", "с ш", "з д", "л с", "ч т",
    ]);
    set.extend(head_pair_sokrs());
    set.extend(pairs(&["ед ч", "мн ч", "повел накл"]));
    set
}

fn pairs(lines: &[&str]) -> HashSet<(String, String)> {
    lines
        .iter()
        .map(|line| {
            let mut it = line.split_whitespace();
            (
                it.next().unwrap().to_string(),
                it.next().unwrap().to_string(),
            )
        })
        .collect()
}

fn initials() -> HashSet<String> {
    ["дж", "ed", "вс"].iter().map(|s| s.to_string()).collect()
}

fn bullet_chars() -> HashSet<char> {
    "§абвгдеabcdef".chars().collect()
}

#[cfg(test)]
mod tests {
    use super::split_by_sentences;

    #[test]
    fn empty_input() {
        assert_eq!(split_by_sentences(""), vec![String::new()]);
    }

    #[test]
    fn single_sentence() {
        assert_eq!(
            split_by_sentences("На двери висит замок."),
            vec!["На двери висит замок."]
        );
    }

    #[test]
    fn punctuation_only() {
        assert_eq!(split_by_sentences("!!! ..."), vec!["!!! ..."]);
    }
}
