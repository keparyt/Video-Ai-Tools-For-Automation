#!/usr/bin/env python3
"""Optional pyannote speaker diarization stage."""
from __future__ import annotations

import argparse
import json
import os
import wave
from pathlib import Path


def load_env_file() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path.cwd() / ".env", override=False)
    except ImportError:
        pass


def read_pcm_wav(path: Path):
    import torch
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if channels != 1 or sample_width != 2 or sample_rate != 16000:
        raise RuntimeError(
            f"Expected mono 16-bit 16kHz PCM WAV, got channels={channels}, "
            f"sample_width={sample_width}, sample_rate={sample_rate}"
        )
    audio = torch.frombuffer(frames, dtype=torch.int16).clone().float() / 32768.0
    return audio.unsqueeze(0), sample_rate


def main() -> int:
    p = argparse.ArgumentParser(description="Speaker diarization with pyannote")
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--token", default=None, help="Hugging Face token; otherwise HF_TOKEN is used")
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--strict", action="store_true", help="Fail when diarization cannot run")
    p.add_argument("--live", action="store_true")
    args = p.parse_args()

    load_env_file()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "diarization_status.json"
    token = args.token or os.getenv("HF_TOKEN")

    if not token:
        message = "HF_TOKEN is not configured; speaker diarization skipped."
        print(f"[DIARIZE] {message}", flush=True)
        status_path.write_text(
            json.dumps({"status": "skipped", "reason": "missing_hf_token"}, indent=2),
            encoding="utf-8",
        )
        if args.strict:
            raise RuntimeError("Speaker diarization requires HF_TOKEN and accepted access to the pyannote model")
        return 0

    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        if args.strict:
            raise RuntimeError("Install pyannote.audio: pip install pyannote.audio") from e
        print("[DIARIZE] pyannote.audio is unavailable; speaker diarization skipped.", flush=True)
        status_path.write_text(json.dumps({"status": "skipped", "reason": "pyannote_unavailable"}, indent=2), encoding="utf-8")
        return 0

    print("[DIARIZE] Loading pyannote speaker diarization...", flush=True)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)

    kwargs = {}
    if args.min_speakers is not None:
        kwargs["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        kwargs["max_speakers"] = args.max_speakers

    # Pass an in-memory waveform to pyannote. This bypasses torchcodec/FFmpeg
    # binary loading issues that are common on Windows.
    waveform, sample_rate = read_pcm_wav(args.audio.resolve())
    input_data = {"waveform": waveform, "sample_rate": sample_rate}
    print(f"[DIARIZE] Processing {waveform.shape[-1] / sample_rate:.2f}s of audio", flush=True)
    diarization = pipeline(input_data, **kwargs)

    rows = []
    for index, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True), 1):
        row = {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
        rows.append(row)
        if args.live:
            print(f"[DIAR {index:05d}] {turn.start:09.3f} -> {turn.end:09.3f} | {speaker}", flush=True)

    (out / "diarization.json").write_text(
        json.dumps({"audio": str(args.audio.resolve()), "segments": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    status_path.write_text(
        json.dumps({"status": "complete", "segments": len(rows)}, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] {len(rows)} speaker regions -> {out / 'diarization.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
