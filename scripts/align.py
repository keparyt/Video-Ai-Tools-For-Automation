#!/usr/bin/env python3
"""WhisperX word-level alignment stage."""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Align Whisper transcript to audio with WhisperX")
    p.add_argument("--transcript", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--compute-type", default=None)
    p.add_argument("--language", default=None)
    p.add_argument("--live", action="store_true", help="Print alignment progress")
    args = p.parse_args()

    try:
        import torch
        import whisperx
    except ImportError as e:
        raise RuntimeError("Install WhisperX: pip install whisperx") from e

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.transcript.read_text(encoding="utf-8"))
    language = args.language or data.get("language")
    if not language:
        raise RuntimeError("Language is missing; pass --language or use transcript.json with detected language")

    device = args.device
    compute = args.compute_type or ("float16" if device == "cuda" else "int8")
    print(f"[ALIGN] Loading alignment model | language={language} | device={device}", flush=True)

    # WhisperX can emit one warning for segments whose phoneme lattice cannot
    # be backtracked. It explicitly keeps the original segment timing in that
    # case. Keep the useful summary output without flooding the console.
    logger = logging.getLogger("whisperx.alignment")
    old_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        audio = whisperx.load_audio(str(args.audio))
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        result = {"segments": data.get("segments", []), "language": language}
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    finally:
        logger.setLevel(old_level)

    # Count segments that received no word-level timestamps. These are the
    # practical alignment fallbacks worth reviewing rather than the internal
    # backtracking warning itself.
    segments = result.get("segments", [])
    fallback_count = 0
    for segment in segments:
        if segment.get("words") is None or len(segment.get("words") or []) == 0:
            fallback_count += 1

    result["alignment_summary"] = {
        "segments": len(segments),
        "segments_without_word_alignment": fallback_count,
        "status": "complete" if fallback_count == 0 else "complete_with_fallbacks",
    }

    (out / "aligned.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.live:
        print(f"[ALIGN] Segments: {len(segments)} | word-timing fallbacks: {fallback_count}", flush=True)
    elif fallback_count:
        print(f"[ALIGN] Word-timing fallbacks: {fallback_count}", flush=True)

    print(f"[OK] Aligned transcript -> {out / 'aligned.json'}", flush=True)


if __name__ == "__main__":
    main()
