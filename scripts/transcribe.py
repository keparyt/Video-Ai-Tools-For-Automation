#!/usr/bin/env python3
"""High-reliability local Whisper transcription with repetition protection."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

AI_MODELS_DIR = Path.cwd() / "aimodels"
AI_MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(AI_MODELS_DIR)
os.environ["HF_HUB_CACHE"] = str(AI_MODELS_DIR / "hub")
os.environ["HF_ASSETS_CACHE"] = str(AI_MODELS_DIR / "assets")
os.environ["TRANSFORMERS_CACHE"] = str(AI_MODELS_DIR / "transformers")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

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
    run(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])


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
        "id": 0,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text.strip(),
        "avg_logprob": getattr(segment, "avg_logprob", None),
        "no_speech_prob": getattr(segment, "no_speech_prob", None),
        "compression_ratio": getattr(segment, "compression_ratio", None),
        "words": words,
    }


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9']+", " ", text.lower()).strip()


def write_outputs(out: Path, segments: list[dict[str, Any]], info: Any, source: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for i, segment in enumerate(segments, 1):
        segment["id"] = i
    srt_lines: list[str] = []
    vtt_lines: list[str] = ["WEBVTT", ""]
    txt_lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        text = seg["text"].strip()
        if not text:
            continue
        srt_lines.extend([str(i), f"{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}", text, ""])
        vtt_lines.extend([f"{fmt_time(seg['start'], False)} --> {fmt_time(seg['end'], False)}", text, ""])
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


def print_live(segment: dict[str, Any], duration: float, index: int) -> None:
    progress = min(100.0, (float(segment["end"]) / duration) * 100.0) if duration else 0.0
    print(
        f"[LIVE {index:05d}] {progress:6.2f}% | "
        f"{fmt_time(segment['start'], False)} -> {fmt_time(segment['end'], False)} | "
        f"{segment['text']}",
        flush=True,
    )


def transcribe_stream(model: Any, wav: Path, kwargs: dict[str, Any], args: argparse.Namespace) -> Any:
    return model.transcribe(str(wav), **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="High-reliability local Whisper transcription")
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--condition-on-previous-text", action="store_true", help="Enable previous-window text context (disabled by default to reduce repetition loops)")
    parser.add_argument("--no-previous-text", action="store_true", help="Compatibility alias; previous-window context is already disabled by default")
    parser.add_argument("--no-word-timestamps", action="store_true")
    parser.add_argument("--live", action="store_true", help="Print each accepted Whisper segment immediately")
    parser.add_argument("--max-repetition-retries", type=int, default=2, help="Maximum recovery attempts after a repeated-text loop")
    parser.add_argument("--repeat-trigger", type=int, default=3, help="Repeated identical segment count that triggers recovery")
    args = parser.parse_args()

    if args.condition_on_previous_text and args.no_previous_text:
        parser.error("--condition-on-previous-text and --no-previous-text cannot be used together")

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
    recovery_path = out / "transcription_recovery.json"
    wav = out / "audio_16k_mono.wav"
    duration = media_duration(video)

    print(f"\nVIDEO:    {video}")
    print(f"DURATION: {fmt_time(duration, False)}")
    print(f"OUTPUT:   {out}")
    print(f"AI MODELS:{AI_MODELS_DIR}")
    print(f"LIVE:     {'ON - every accepted segment will print immediately' if args.live else 'OFF'}")
    print("CONTEXT:  previous-window text OFF by default (repetition-safe)\n")

    extract_audio(video, wav)
    compute_type = args.compute_type or ("float16" if args.device == "cuda" else "int8")
    print(f"[MODEL] Loading {args.model} | device={args.device} | compute={compute_type}", flush=True)
    model = WhisperModel(args.model, device=args.device, compute_type=compute_type, download_root=str(AI_MODELS_DIR))

    checkpoint = load_checkpoint(checkpoint_path)
    segments: list[dict[str, Any]] = checkpoint.get("segments", []) if checkpoint else []
    completed_end = float(checkpoint.get("completed_end", 0.0)) if checkpoint else 0.0
    resume_from = max(0.0, completed_end - 1.5) if segments else 0.0
    if resume_from > 0:
        print(f"[RESUME] Continuing near {fmt_time(resume_from, False)}")

    base_kwargs: dict[str, Any] = {
        "beam_size": args.beam_size,
        "best_of": 5,
        "temperature": [0.0, 0.2, 0.4, 0.6],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "repetition_penalty": 1.08,
        "no_repeat_ngram_size": 3,
        "word_timestamps": not args.no_word_timestamps,
        "condition_on_previous_text": bool(args.condition_on_previous_text),
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500, "speech_pad_ms": 400},
        "chunk_length": 30,
        "without_timestamps": False,
        "hallucination_silence_threshold": 1.0,
        "max_new_tokens": 256,
    }
    if args.language:
        base_kwargs["language"] = args.language
    if resume_from > 0:
        base_kwargs["clip_timestamps"] = f"{resume_from},{duration}"

    all_segments = list(segments)
    recovery_events: list[dict[str, Any]] = []
    current_kwargs = dict(base_kwargs)
    recovery_attempts = 0
    live_index = len(all_segments) + 1
    last_good_end = resume_from

    while True:
        print("[TRANSCRIBE] Starting..." if recovery_attempts == 0 else f"[RECOVERY] Retry attempt {recovery_attempts}/{args.max_repetition_retries} from {fmt_time(last_good_end, False)}", flush=True)
        stream, info = transcribe_stream(model, wav, current_kwargs, args)
        new_segments: list[dict[str, Any]] = []
        recent_same: list[tuple[str, float]] = []
        triggered = False

        for segment in stream:
            item = segment_to_dict(segment)
            start = float(item["start"])
            end = float(item["end"])
            text = item["text"].strip()
            if end <= start or end <= last_good_end + 0.05 or not text:
                continue
            if new_segments and start < float(new_segments[-1]["end"]) - 0.05:
                continue

            normalized = normalize_text(text)
            if normalized:
                recent_same = [(t, e) for t, e in recent_same if end - e <= 30.0]
                recent_same.append((normalized, end))
                same_count = sum(1 for t, _ in recent_same if t == normalized)
            else:
                same_count = 0

            if same_count >= args.repeat_trigger:
                print(
                    f"[HALT] Repetition loop detected at {fmt_time(start, False)}: "
                    f"{args.repeat_trigger}x '{text[:80]}'",
                    flush=True,
                )
                triggered = True
                recovery_events.append({
                    "start": start,
                    "end": end,
                    "text": text,
                    "reason": "repeated_text_loop",
                    "attempt": recovery_attempts,
                })
                break

            new_segments.append(item)
            last_good_end = end
            if args.live:
                print_live(item, duration, live_index)
                live_index += 1
            else:
                progress = min(100.0, (end / duration) * 100.0) if duration else 0.0
                print(f"\r[PROGRESS] {progress:6.2f}% | {fmt_time(start, False)} -> {fmt_time(end, False)}", end="", flush=True)

            save_checkpoint(checkpoint_path, {
                "source": str(video),
                "model": args.model,
                "language": getattr(info, "language", None),
                "completed_end": end,
                "segments": all_segments + new_segments,
                "recovery_attempts": recovery_attempts,
            })

        all_segments.extend(new_segments)
        if not triggered:
            break
        if recovery_attempts >= args.max_repetition_retries:
            print("[HALT] Maximum repetition recovery attempts reached; leaving region unresolved instead of looping.", flush=True)
            break

        recovery_attempts += 1
        current_kwargs = dict(base_kwargs)
        current_kwargs["condition_on_previous_text"] = False
        current_kwargs["temperature"] = [0.2, 0.4, 0.6, 0.8]
        current_kwargs["repetition_penalty"] = 1.12
        current_kwargs["no_repeat_ngram_size"] = 2
        current_kwargs["clip_timestamps"] = f"{max(0.0, last_good_end - 0.5)},{duration}"

    if recovery_events:
        recovery_path.write_text(
            json.dumps({"events": recovery_events, "max_retries": args.max_repetition_retries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("\n[OK] Whisper transcription complete.")
    all_segments.sort(key=lambda x: (float(x["start"]), float(x["end"])))
    write_outputs(out, all_segments, info, video)
    checkpoint_path.unlink(missing_ok=True)

    print("\n[DONE] Files:")
    for name in ["transcript.srt", "transcript.vtt", "transcript.txt", "transcript.json", "transcript_words.json"]:
        print(f"  {out / name}")
    if recovery_events:
        print(f"  {recovery_path}")
    print("\nNOTE: Repetition loops are halted and retried a limited number of times instead of being allowed to run indefinitely.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
