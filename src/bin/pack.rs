//! Сборка distributable model pack из workdir ruaccent tiny.
//!
//! `pack <workdir> <out-dir> [<pack-id>]` копирует `nn/` и `dictionary/`
//! и пишет `manifest.json` со списком файлов, SHA-256 и размерами.
//! Layout совместим с discovery ROADMAP-081 (app-tts-v2):
//! `<pack-id>/manifest.json`, `<pack-id>/nn/...`, `<pack-id>/dictionary/...`.

use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: pack <workdir> <out-dir> [pack-id]");
        std::process::exit(2);
    }
    let workdir = PathBuf::from(&args[1]);
    let out_root = PathBuf::from(&args[2]);
    let pack_id = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "tiny-1.5.8.3".to_string());

    if let Err(e) = build_pack(&workdir, &out_root, &pack_id) {
        eprintln!("pack failed: {e}");
        std::process::exit(1);
    }
}

fn build_pack(workdir: &Path, out_root: &Path, pack_id: &str) -> Result<(), String> {
    let out = out_root.join(pack_id);
    for dir in ["nn", "dictionary"] {
        let src = workdir.join(dir);
        if !src.is_dir() {
            return Err(format!("missing directory: {}/", dir));
        }
        copy_dir(&src, &out.join(dir))?;
    }

    let out_dict = out.join("dictionary");
    let fst_path = out_dict.join("accents.fst");
    let gz_path = out_dict.join("accents.json.gz");
    if !fst_path.exists() && gz_path.exists() {
        println!("compiling accents.fst from accents.json.gz...");
        compile_accents_fst(&gz_path, &fst_path)?;
    }

    // manifest: отсортированный список файлов относительно корня pack'а.
    let mut files = Vec::new();
    collect_files(&out, &out, &mut files)?;
    files.sort_by_key(|f| f["path"].as_str().unwrap_or("").to_string());

    let manifest = serde_json::json!({
        "schema_version": 1,
        "id": pack_id,
        "name": format!("RUAccent tiny model pack ({pack_id})"),
        "origin": {
            "project": "ruaccent-rs (Rust port of RUAccent)",
            "upstream": "https://github.com/Den4ikAI/ruaccent",
            "models": "https://huggingface.co/ruaccent/accentuator",
            "model_size": "tiny"
        },
        "license": {
            "code": "MIT",
            "upstream_copyright": "Copyright 2026 Denis Petrov (RUAccent)",
            "notice": "Port of RUAccent (MIT). Model files from ruaccent/accentuator (MIT)."
        },
        "files": files,
    });
    let manifest_path = out.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest).unwrap() + "\n")
        .map_err(|e| format!("write manifest: {e}"))?;
    println!("wrote {} ({} files)", manifest_path.display(), files.len());
    Ok(())
}

fn copy_dir(src: &Path, dst: &Path) -> Result<(), String> {
    fs::create_dir_all(dst).map_err(|e| format!("mkdir {}: {e}", dst.display()))?;
    for entry in fs::read_dir(src).map_err(|e| format!("read {}: {e}", src.display()))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let ty = entry.file_type().map_err(|e| e.to_string())?;
        if ty.is_dir() {
            copy_dir(&entry.path(), &dst.join(entry.file_name()))?;
        } else {
            fs::copy(entry.path(), dst.join(entry.file_name()))
                .map_err(|e| format!("copy {}: {e}", entry.path().display()))?;
        }
    }
    Ok(())
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<serde_json::Value>) -> Result<(), String> {
    for entry in fs::read_dir(dir).map_err(|e| format!("read {}: {e}", dir.display()))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let ty = entry.file_type().map_err(|e| e.to_string())?;
        let path = entry.path();
        if ty.is_dir() {
            collect_files(root, &path, out)?;
        } else {
            let data = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
            let rel = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            out.push(serde_json::json!({
                "path": rel,
                "size": data.len(),
                "sha256": hex(&Sha256::digest(&data)),
            }));
        }
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn compile_accents_fst(json_gz_path: &Path, fst_out_path: &Path) -> Result<(), String> {
    use flate2::read::GzDecoder;
    use std::io::Read;

    let file = fs::File::open(json_gz_path).map_err(|e| format!("open json.gz: {e}"))?;
    let mut decoder = GzDecoder::new(file);
    let mut json_str = String::new();
    decoder
        .read_to_string(&mut json_str)
        .map_err(|e| format!("decompress json.gz: {e}"))?;

    let map: std::collections::HashMap<String, String> =
        serde_json::from_str(&json_str).map_err(|e| format!("parse json: {e}"))?;
    drop(json_str);

    let mut entries: Vec<(String, u64)> = Vec::with_capacity(map.len());
    for (k, v) in map {
        let code = if let Some(plus_idx) = v.find('+') {
            (plus_idx as u64) + 1
        } else {
            0
        };
        entries.push((k, code));
    }
    entries.sort_unstable_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));

    let out_file = fs::File::create(fst_out_path).map_err(|e| format!("create fst: {e}"))?;
    let writer = std::io::BufWriter::with_capacity(4 * 1024 * 1024, out_file);
    let mut builder = fst::MapBuilder::new(writer).map_err(|e| format!("fst builder: {e}"))?;

    for (k, val) in entries {
        builder
            .insert(k.as_bytes(), val)
            .map_err(|e| format!("fst insert: {e}"))?;
    }
    builder.finish().map_err(|e| format!("fst finish: {e}"))?;
    Ok(())
}
