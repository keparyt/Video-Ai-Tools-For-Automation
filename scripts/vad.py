#!/usr/bin/env python3
"""Windows-safe Silero VAD stage with direct PCM WAV I/O."""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import torch


def run(cmd):
    subprocess.run(cmd, check=True)


def read_pcm_wav(path: Path) -> torch.Tensor:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if channels != 1:
        raise RuntimeError(f"Expected mono WAV, got {channels} channels: {path}")
    if sample_width != 2:
        raise RuntimeError(f"Expected 16-bit PCM WAV, got {sample_width * 8}-bit: {path}")
    if sample_rate != 16000:
        raise RuntimeError(f"Expected 16000 Hz WAV, got {sample_rate} Hz: {path}")

    count = len(frames) // 2
    samples = struct.unpack(f"<{count}h", frames)
    return torch.tensor(samples, dtype=torch.float32) / 32768.0


def write_pcm_wav(path: Path, audio: torch.Tensor, sample_rate: int = 16000):
    data = audio.detach().cpu().clamp(-1.0, 1.0)
    pcm = (data * 32767.0).to(torch.int16).numpy().tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def main():
    p = argparse.ArgumentParser(description="Detect speech with Silero VAD")
    p.add_argument("--input", type=Path, required=True, help="16 kHz mono WAV or source video")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-speech-ms", type=int, default=250)
    p.add_argument("--min-silence-ms", type=int, default=100)
    p.add_argument("--live", action="store_true", help="Print each detected speech region immediately")
    args = p.parse_args()

    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.input.resolve()
    wav = out / "vad_input_16k_mono.wav"

    if source.suffix.lower() == ".wav":
        wav = source
    else:
        print("[VAD] Extracting 16 kHz mono PCM WAV with FFmpeg...", flush=True)
        run([
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(wav)
        ])

    try:
        from silero_vad import load_silero_vad, get_speech_timestamps
    except ImportError as e:
        raise RuntimeError("Install silero-vad: pip install silero-vad") from e

    print("[VAD] Loading Silero VAD...", flush=True)
    model = load_silero_vad()

    print(f"[VAD] Reading PCM WAV directly: {wav.name}", flush=True)
    audio = read_pcm_wav(wav)
    duration = len(audio) / 16000.0
    print(f"[VAD] Audio: {duration:.2f}s", flush=True)
    print("[VAD] Detecting speech...", flush=True)

    timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=16000,
        threshold=args.threshold,
        min_speech_duration_ms=args.min_speech_ms,
        min_silence_duration_ms=args.min_silence_ms,
    )

    data = []
    for i, x in enumerate(timestamps, 1):
        start = x["start"] / 16000.0
        end = x["end"] / 16000.0
        row = {"start": start, "end": end, "duration": end - start}
        data.append(row)
        if args.live:
            print(f"[VAD {i:05d}] {start:09.3f} -> {end:09.3f} | speech", flush=True)

    speech = torch.cat([audio[x["start"]:x["end"]] for x in timestamps]) if timestamps else audio[:0]
    speech_wav = out / "speech_only.wav"
    write_pcm_wav(speech_wav, speech)

    vad_path = out / "vad.json"
    vad_path.write_text(
        json.dumps({
            "source": str(source),
            "sample_rate": 16000,
            "segments": data,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[VAD] Found {len(data)} speech regions", flush=True)
    print(f"[VAD] Saved: {vad_path}", flush=True)
    print(f"[VAD] Saved speech audio: {speech_wav}", flush=True)


if __name__ == "__main__":
    main()
