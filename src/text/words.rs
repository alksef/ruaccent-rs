//! Port of `TextPreprocessor.split_by_words`.
//!
//! The original implementation:
//! ```python
//! def split_by_words(string):
//!     string = string.replace(" - ", ' ~ ')
//!     match = list(re.finditer(r"\w*(?:\+\w+)*|[^\w\s]+", string.lower()))
//!     remaining_text = [string[l.end():r.start()] for l,r in zip(match, match[1:])]
//!     words = [string[x.start():x.end()] for x in match]
//!     words_mask = [i for i, w in enumerate(words) if w]
//!     valid_words = [words[i] for i in words_mask]
//!     if len(words_mask) == 0:
//!         return valid_words, ["", ""]
//!     remaining_text_res = ["".join(remaining_text[:words_mask[0]])] + \
//!         ["".join(remaining_text[l+1:r]) for l, r in zip(words_mask, words_mask[1:])]
//!     remaining_text_res.append("".join(remaining_text[words_mask[-1]+1:]))
//!     return valid_words, remaining_text_res
//! ```
//!
//! The `regex` engine in Python is a backtracking engine whose `finditer`
//! emits empty matches for `\w*` and advances past them with a `must_advance`
//! flag (only the *first* match attempt at the current position is constrained
//! to be non-empty). We replicate that exact scan order manually.

/// Split a sentence into `(words, remaining_text)`.
///
/// * `words` are the non-empty tokens matched by `\w*(?:\+\w+)*|[^\w\s]+`
///   (punctuation runs are kept as single tokens).
/// * `remaining_text` always has `words.len() + 1` entries and holds the text
///   between consecutive words (edges included); the first and last entries are
///   the text before the first and after the last word.
pub fn split_by_words(sentence: &str) -> (Vec<String>, Vec<String>) {
    let string = sentence.replace(" - ", " ~ ");
    let chars: Vec<char> = string.chars().collect();

    // Python matches on `string.lower()` but slices the *original* string with
    // the same character indices. For Cyrillic/ASCII lowercasing is length
    // preserving, so indices stay aligned.
    let lower: String = string.to_lowercase();
    let lower_chars: Vec<char> = lower.chars().collect();

    // If lowercasing changed the length (exotic Unicode), fall back to matching
    // the original string so indices never go out of bounds.
    let match_chars: &[char] = if lower_chars.len() == chars.len() {
        &lower_chars
    } else {
        &chars
    };

    let matches = find_matches(match_chars);

    let mut remaining_text: Vec<String> = Vec::with_capacity(matches.len().saturating_sub(1));
    for pair in matches.windows(2) {
        remaining_text.push(slice_chars(&chars, pair[0].1, pair[1].0));
    }

    let words: Vec<String> = matches
        .iter()
        .map(|&(s, e)| slice_chars(&chars, s, e))
        .collect();

    let words_mask: Vec<usize> = words
        .iter()
        .enumerate()
        .filter(|(_, w)| !w.is_empty())
        .map(|(i, _)| i)
        .collect();

    let valid_words: Vec<String> = words_mask.iter().map(|&i| words[i].clone()).collect();

    if words_mask.is_empty() {
        return (valid_words, vec![String::new(), String::new()]);
    }

    let mut remaining_text_res: Vec<String> = Vec::with_capacity(valid_words.len() + 1);
    remaining_text_res.push(remaining_text[..words_mask[0]].concat());
    for pair in words_mask.windows(2) {
        remaining_text_res.push(remaining_text[pair[0] + 1..pair[1]].concat());
    }
    remaining_text_res.push(remaining_text[words_mask[words_mask.len() - 1] + 1..].concat());

    (valid_words, remaining_text_res)
}

/// Emulate `re.finditer(r"\w*(?:\+\w+)*|[^\w\s]+", text)` on the char slice,
/// returning `(start, end)` character-index pairs (empty matches included).
fn find_matches(chars: &[char]) -> Vec<(usize, usize)> {
    let n = chars.len();
    let mut result = Vec::new();
    let mut start = 0usize;
    let mut must_advance = false;

    while start <= n {
        match search(chars, start, must_advance) {
            Some((ms, me)) => {
                result.push((ms, me));
                must_advance = me == ms;
                start = me;
            }
            None => break,
        }
    }

    result
}

/// Leftmost match at or after `start`. When `must_advance` is set, the *first*
/// attempt (at `start` itself) rejects an empty match; subsequent positions are
/// scanned without that constraint, mirroring CPython's `SRE(search)`.
fn search(chars: &[char], start: usize, must_advance: bool) -> Option<(usize, usize)> {
    let n = chars.len();

    if let Some(m) = match_at(chars, start, must_advance) {
        return Some(m);
    }

    let mut p = start + 1;
    while p <= n {
        if let Some(m) = match_at(chars, p, false) {
            return Some(m);
        }
        p += 1;
    }

    None
}

/// A single match attempt at position `p`.
fn match_at(chars: &[char], p: usize, must_advance: bool) -> Option<(usize, usize)> {
    // Alternative 1: `\w*(?:\+\w+)*`
    let end = word_run_at(chars, p);
    if end > p {
        return Some((p, end));
    }

    // `\w*` matched empty.
    if !must_advance {
        return Some((p, p));
    }

    // Alternative 2: `[^\w\s]+` (only reached when the empty match is rejected).
    let end = punct_run_at(chars, p);
    if end > p {
        return Some((p, end));
    }

    None
}

/// Greedy `\w*(?:\+\w+)*` at `p`: a run of word characters followed by zero or
/// more `+word` groups. Returns the exclusive end index.
fn word_run_at(chars: &[char], p: usize) -> usize {
    let n = chars.len();
    let mut q = p;
    while q < n && is_word(chars[q]) {
        q += 1;
    }
    loop {
        if q < n && chars[q] == '+' {
            let mut j = q + 1;
            if j >= n || !is_word(chars[j]) {
                break;
            }
            while j < n && is_word(chars[j]) {
                j += 1;
            }
            q = j;
        } else {
            break;
        }
    }
    q
}

/// Greedy `[^\w\s]+` at `p`. Returns the exclusive end index.
fn punct_run_at(chars: &[char], p: usize) -> usize {
    let n = chars.len();
    let mut q = p;
    while q < n && !is_word(chars[q]) && !is_space(chars[q]) {
        q += 1;
    }
    q
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn is_space(c: char) -> bool {
    c.is_whitespace()
}

fn slice_chars(chars: &[char], start: usize, end: usize) -> String {
    chars[start..end].iter().collect()
}

#[cfg(test)]
mod tests {
    use super::split_by_words;

    #[test]
    fn empty_input() {
        assert_eq!(split_by_words(""), (vec![], vec![String::new(), String::new()]));
    }

    #[test]
    fn punctuation_only() {
        let (words, remaining) = split_by_words("!!! ...");
        assert_eq!(words, vec!["!!!", "..."]);
        assert_eq!(remaining, vec!["", " ", ""]);
    }

    #[test]
    fn plain_sentence() {
        let (words, remaining) = split_by_words("На двери висит замок.");
        assert_eq!(words, vec!["На", "двери", "висит", "замок", "."]);
        assert_eq!(remaining, vec!["", " ", " ", " ", "", ""]);
    }
}
