#!/usr/bin/env python3
"""Silero VAD stage. Produces speech-only WAV plus speech intervals."""
from __future__ import annotations
import argparse, json, shutil, subprocess
from pathlib import Path


def run(cmd):
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description="Detect speech with Silero VAD")
    p.add_argument("--input", type=Path, required=True, help="16 kHz mono WAV or source video")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-speech-ms", type=int, default=250)
    p.add_argument("--min-silence-ms", type=int, default=100)
    args = p.parse_args()
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required")
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    source = args.input.resolve()
    wav = out / "vad_input_16k_mono.wav"
    if source.suffix.lower() == ".wav":
        wav = source
    else:
        run(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])
    try:
        from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, save_audio, collect_chunks
    except ImportError as e:
        raise RuntimeError("Install silero-vad: pip install silero-vad") from e
    print("[VAD] Loading Silero VAD...")
    model = load_silero_vad()
    audio = read_audio(str(wav), sampling_rate=16000)
    timestamps = get_speech_timestamps(audio, model, sampling_rate=16000, threshold=args.threshold, min_speech_duration_ms=args.min_speech_ms, min_silence_duration_ms=args.min_silence_ms)
    speech = collect_chunks(timestamps, audio)
    speech_wav = out / "speech_only.wav"
    save_audio(str(speech_wav), speech, sampling_rate=16000)
    data = [{"start": x["start"] / 16000.0, "end": x["end"] / 16000.0, "duration": (x["end"] - x["start"]) / 16000.0} for x in timestamps]
    (out / "vad.json").write_text(json.dumps({"source": str(source), "sample_rate": 16000, "segments": data}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {len(data)} speech regions -> {speech_wav}")

if __name__ == "__main__": main()
