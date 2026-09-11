#!/usr/bin/env python3
"""WhisperX word-level alignment stage."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Align Whisper transcript to audio with WhisperX")
    p.add_argument("--transcript", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--compute-type", default=None)
    p.add_argument("--language", default=None)
    args = p.parse_args()
    try:
        import torch
        import whisperx
    except ImportError as e:
        raise RuntimeError("Install WhisperX: pip install whisperx") from e
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.transcript.read_text(encoding="utf-8"))
    language = args.language or data.get("language")
    if not language:
        raise RuntimeError("Language is missing; pass --language or use transcript.json with detected language")
    device = args.device
    compute = args.compute_type or ("float16" if device == "cuda" else "int8")
    print(f"[ALIGN] Loading alignment model | language={language} | device={device}")
    audio = whisperx.load_audio(str(args.audio))
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = {"segments": data.get("segments", []), "language": language}
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    (out / "aligned.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Aligned transcript -> {out / 'aligned.json'}")

if __name__ == "__main__": main()
