# ruaccent-rs

Native Rust port of [RUAccent](https://github.com/Den4ikAI/ruaccent) —
automatic Russian stress placement and homograph disambiguation
(`за́мок/замо́к`, `пла́чу/плачу́`, ...). No Python, no sidecar process:
ONNX inference via [`ort`](https://ort.pyke.io), tokenization via the
official [`tokenizers`](https://github.com/huggingface/tokenizers) crate,
dictionaries as gzip+JSON.

The port is **bit-compatible** with the original Python pipeline: every
stage (sentence splitting, word splitting, tokenizer ids, ONNX logits,
homograph choices, final strings) is verified against a captured golden
reference of the upstream implementation (`golden/tiny.jsonl`,
`tools/golden_dump.py`).

## Status

`tiny` model, `use_dictionary=true`, CPU — complete and golden-verified.
Word stress + homographs + `е/ё` + stress-usage predictor, custom
dict/homograph overrides, `skip_regex` support.

Benchmarks vs the Python original on the same machine (Ryzen 7 5700X):
**~2–3× faster** per phrase (`process_all`), see
[docs/benchmarks.md](docs/benchmarks.md).

## Usage

```rust
use ruaccent_rs::pipeline::RuAccent;

let mut accentor = RuAccent::load("path/to/tiny".as_ref())?;
let stressed = accentor.process_all("На двери висит замок.");
// "На дв+ери вис+ит зам+ок."
```

Custom dictionaries override the bundled ones:

```rust
let mut accentor = RuAccent::load_with_custom(
    workdir, &custom_dict, &custom_homographs,
)?;
```

The output uses upstream's `+` notation (a `+` precedes the stressed vowel).
It is an accentor-level format — convert to whatever your TTS expects
(U+0301, SSML, phonemes) in your own adapter.

## Getting the models

Model files are **not** in this repository. Download the `tiny` model from
[Hugging Face `ruaccent/accentuator`](https://huggingface.co/ruaccent/accentuator)
(files under `nn/` and `dictionary/`, MIT license) into a directory, or build
a redistributable, checksummed pack:

```powershell
rustup run stable-x86_64-pc-windows-msvc cargo run --release --bin pack -- `
  <workdir-with-nn-and-dictionary> <out-dir> [pack-id]
```

This produces `<out-dir>/<pack-id>/{manifest.json,nn/,dictionary/}` where
`manifest.json` records SHA-256 and size of every file plus upstream license
notices — the layout expected by the model-pack discovery of the consuming
application.

## License

MIT — see [LICENSE](LICENSE). This is a port of RUAccent by Denis Petrov
(MIT); model files come from `ruaccent/accentuator` (MIT) and are not
redistributed from this repository.
