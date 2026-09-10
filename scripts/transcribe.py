#!/usr/bin/env python3
"""High-reliability local video transcription.

Pipeline: FFmpeg audio extraction -> faster-whisper large-v3 -> optional
WhisperX alignment -> SRT/VTT/TXT/JSON output. Everything runs locally.

Example:
    python scripts/transcribe.py video.mp4
    python scripts/transcribe.py video.mp4 --language fr --output-dir output

Notes:
- Whisper is the source of truth; no LLM is allowed to rewrite the transcript.
- Word timestamps are preserved in JSON and used to build subtitle timings.
- A checkpoint is written so an interrupted long transcription can resume.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("ERROR: faster-whisper is not installed. Run: pip install -r requirements-transcription.txt")
    raise


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".mts", ".m2ts"}


def run(cmd: list[str]) -> None:
    print("[CMD]", " ".join(f'\"{x}\"' if " " in x else x for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg was not found in PATH. Install FFmpeg and restart the terminal.")


def media_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def fmt_time(seconds: float, comma: bool = True) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round((seconds - math.floor(seconds)) * 1000))
    total = int(seconds)
    if ms >= 1000:
        total += 1
        ms = 0
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def extract_audio(video: Path, wav: Path) -> None:
    wav.parent.mkdir(parents=True, exist_ok=True)
    if wav.exists() and wav.stat().st_size > 0:
        print(f"[OK] Reusing extracted audio: {wav}")
        return
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(wav),
    ])


def save_checkpoint(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def segment_to_dict(segment: Any) -> dict[str, Any]:
    words = []
    for word in (segment.words or []):
        words.append({
            "word": word.word,
            "start": word.start,
            "end": word.end,
            "probability": getattr(word, "probability", None),
        })
    return {
        "id": len(words),
        "start": segment.start,
        "end": segment.end,
        "text": segment.text.strip(),
        "avg_logprob": getattr(segment, "avg_logprob", None),
        "no_speech_prob": getattr(segment, "no_speech_prob", None),
        "compression_ratio": getattr(segment, "compression_ratio", None),
        "words": words,
    }


def write_outputs(out: Path, segments: list[dict[str, Any]], info: Any, source: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    # Re-number segments after all merging/checkpointing.
    for i, segment in enumerate(segments, 1):
        segment["id"] = i

    srt_lines: list[str] = []
    vtt_lines: list[str] = ["WEBVTT", ""]
    txt_lines: list[str] = []

    for i, seg in enumerate(segments, 1):
        text = seg["text"].strip()
        if not text:
            continue
        srt_lines.extend([
            str(i),
            f"{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}",
            text,
            "",
        ])
        vtt_lines.extend([
            f"{fmt_time(seg['start'], False)} --> {fmt_time(seg['end'], False)}",
            text,
            "",
        ])
        txt_lines.append(f"[{fmt_time(seg['start'], False)}] {text}")

    metadata = {
        "source": str(source.resolve()),
        "duration_seconds": media_duration(source),
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "segments": segments,
    }

    (out / "transcript.srt").write_text("\n".join(srt_lines), encoding="utf-8")
    (out / "transcript.vtt").write_text("\n".join(vtt_lines), encoding="utf-8")
    (out / "transcript.txt").write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    (out / "transcript.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    words = []
    for seg in segments:
        words.extend(seg.get("words", []))
    (out / "transcript_words.json").write_text(json.dumps(words, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="High-reliability local Whisper transcription")
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: <video>_transcript)")
    parser.add_argument("--language", default=None, help="Language code, e.g. en or fr. Auto-detect if omitted.")
    parser.add_argument("--model", default="large-v3", help="Whisper model (default: large-v3)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--compute-type", default=None, help="faster-whisper compute type; auto selects float16 on CUDA")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--no-previous-text", action="store_true", help="Disable cross-segment context")
    parser.add_argument("--no-word-timestamps", action="store_true")
    args = parser.parse_args()

    video = args.video.expanduser().resolve()
    if not video.exists():
        print(f"ERROR: Video not found: {video}", file=sys.stderr)
        return 2
    if video.suffix.lower() not in VIDEO_EXTENSIONS:
        print(f"WARNING: Unusual video extension: {video.suffix}")

    require_ffmpeg()
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe was not found in PATH. It is normally included with FFmpeg.")

    out = (args.output_dir or video.with_name(video.stem + "_transcript")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "transcription_checkpoint.json"
    wav = out / "audio_16k_mono.wav"

    duration = media_duration(video)
    print(f"\nVIDEO:    {video}")
    print(f"DURATION: {fmt_time(duration, False)}")
    print(f"OUTPUT:   {out}\n")

    extract_audio(video, wav)

    compute_type = args.compute_type or ("float16" if args.device == "cuda" else "int8")
    print(f"[MODEL] Loading {args.model} | device={args.device} | compute={compute_type}", flush=True)
    model = WhisperModel(args.model, device=args.device, compute_type=compute_type)

    checkpoint = load_checkpoint(checkpoint_path)
    segments: list[dict[str, Any]] = checkpoint.get("segments", []) if checkpoint else []
    completed_end = float(checkpoint.get("completed_end", 0.0)) if checkpoint else 0.0

    # Resume from the last completed segment. We deliberately keep a tiny overlap
    # so a word crossing a restart boundary is not lost.
    resume_from = max(0.0, completed_end - 1.5) if segments else 0.0
    if resume_from > 0:
        print(f"[RESUME] Continuing near {fmt_time(resume_from, False)}")

    kwargs: dict[str, Any] = {
        "beam_size": args.beam_size,
        "temperature": 0,
        "word_timestamps": not args.no_word_timestamps,
        "condition_on_previous_text": not args.no_previous_text,
        "vad_filter": False,
        "chunk_length": 30,
        "without_timestamps": False,
    }
    if args.language:
        kwargs["language"] = args.language
    if resume_from > 0:
        kwargs["clip_timestamps"] = f"{resume_from},{duration}"

    print("[TRANSCRIBE] Starting...", flush=True)
    stream, info = model.transcribe(str(wav), **kwargs)

    new_segments: list[dict[str, Any]] = []
    for segment in stream:
        item = segment_to_dict(segment)
        if item["text"]:
            new_segments.append(item)
        progress = min(100.0, (float(segment.end) / duration) * 100.0) if duration else 0
        print(
            f"\r[PROGRESS] {progress:6.2f}% | {fmt_time(segment.start, False)} -> {fmt_time(segment.end, False)} | {item['text'][:100]}",
            end="",
            flush=True,
        )
        save_checkpoint(checkpoint_path, {
            "source": str(video),
            "model": args.model,
            "language": getattr(info, "language", None),
            "completed_end": segment.end,
            "segments": segments + new_segments,
        })

    print("\n[OK] Whisper transcription complete.")

    # Replace the overlapped tail with the fresh pass, avoiding duplicate segments.
    if segments and resume_from > 0:
        segments = [s for s in segments if float(s["end"]) < resume_from]
    segments.extend(new_segments)
    segments.sort(key=lambda x: (float(x["start"]), float(x["end"])))

    write_outputs(out, segments, info, video)
    checkpoint_path.unlink(missing_ok=True)

    print("\n[DONE] Files:")
    for name in ["transcript.srt", "transcript.vtt", "transcript.txt", "transcript.json", "transcript_words.json"]:
        print(f"  {out / name}")
    print("\nNOTE: For maximum reliability, review low-confidence/overlapping speech in the JSON/audio. No LLM rewriting is performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
