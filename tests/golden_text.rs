use serde::Deserialize;

#[derive(Deserialize)]
struct Record {
    input: String,
    sentences: Vec<Vec<String>>,
    words: Vec<Vec<String>>,
    remaining_text: Vec<Vec<String>>,
}

#[test]
fn golden_text_splitting() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/golden/tiny.jsonl");
    let content = std::fs::read_to_string(path).expect("failed to read golden/tiny.jsonl");

    let mut count = 0usize;

    for (idx, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let rec: Record =
            serde_json::from_str(line).unwrap_or_else(|e| panic!("bad JSON on line {}: {e}", idx + 1));

        assert_eq!(
            rec.sentences.len(),
            1,
            "record {} must hold exactly one sentence group",
            idx + 1
        );

        let sentences = ruaccent_rs::text::split_by_sentences(&rec.input);
        assert_eq!(sentences, rec.sentences[0], "record {} sentences", idx + 1);

        assert_eq!(rec.words.len(), sentences.len(), "record {} words groups", idx + 1);
        assert_eq!(
            rec.remaining_text.len(),
            sentences.len(),
            "record {} remaining_text groups",
            idx + 1
        );

        for (i, sentence) in sentences.iter().enumerate() {
            let (words, remaining) = ruaccent_rs::text::split_by_words(sentence);
            assert_eq!(words, rec.words[i], "record {} sentence {} words", idx + 1, i);
            assert_eq!(
                remaining, rec.remaining_text[i],
                "record {} sentence {} remaining_text",
                idx + 1, i
            );
        }

        count += 1;
    }

    assert_eq!(count, 20, "golden/tiny.jsonl must contain 20 records");
}
