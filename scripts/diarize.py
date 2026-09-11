#!/usr/bin/env python3
"""Optional local pyannote speaker diarization stage."""
from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
import wave
from pathlib import Path

MODEL_ID = "pyannote/speaker-diarization-community-1"
AI_MODELS_DIR = Path.cwd() / "aimodels"


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

    # Copy through bytearray so torch does not receive a non-writable buffer.
    audio = torch.frombuffer(bytearray(frames), dtype=torch.int16).float() / 32768.0
    return audio.unsqueeze(0), sample_rate


def write_status(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def configure_quiet_output() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"torchcodec is not installed correctly.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*built-in audio decoding will fail.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The given buffer is not writable.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"TensorFloat-32 \(TF32\) has been disabled.*",
        category=UserWarning,
    )

    logging.getLogger("pyannote.audio.core.io").setLevel(logging.ERROR)


def extract_segments(output):
    """Return (turn, speaker) tuples for pyannote 4.x and older outputs."""
    # Community-1 exposes both standard and exclusive annotations. The
    # exclusive version is preferable when reconciling against ASR timestamps.
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    source = "exclusive_speaker_diarization"
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", None)
        source = "speaker_diarization"
    if annotation is not None:
        return [(turn, speaker) for turn, speaker in annotation], source

    # Legacy pyannote outputs may be Annotation objects directly.
    if hasattr(output, "itertracks"):
        return list(output.itertracks(yield_label=True)), "legacy_itertracks"

    raise RuntimeError(f"Unsupported pyannote output type: {type(output).__name__}")


def main() -> int:
    p = argparse.ArgumentParser(description="Local speaker diarization with pyannote")
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--token", default=None, help="Hugging Face token; otherwise HF_TOKEN is used")
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--strict", action="store_true", help="Fail when diarization cannot run")
    p.add_argument("--live", action="store_true", help="Print speaker regions as they are produced")
    args = p.parse_args()

    load_env_file()
    configure_quiet_output()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "diarization_status.json"
    token = args.token or os.getenv("HF_TOKEN")

    if not token:
        message = "HF_TOKEN is not configured; speaker diarization skipped."
        print(f"[DIARIZE] {message}", flush=True)
        print("[DIARIZE] Add HF_TOKEN to .env and accept the model access conditions on Hugging Face.", flush=True)
        write_status(status_path, {"status": "skipped", "reason": "missing_hf_token", "model": MODEL_ID})
        if args.strict:
            raise RuntimeError(f"Speaker diarization requires HF_TOKEN and accepted access to {MODEL_ID}")
        return 0

    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as e:
        message = "pyannote.audio/torch is unavailable; speaker diarization skipped."
        print(f"[DIARIZE] {message}", flush=True)
        write_status(status_path, {"status": "skipped", "reason": "pyannote_or_torch_unavailable", "model": MODEL_ID})
        if args.strict:
            raise RuntimeError("Install pyannote.audio and torch: pip install pyannote.audio torch") from e
        return 0

    audio_path = args.audio.resolve()
    if not audio_path.exists():
        message = f"Audio not found: {audio_path}"
        print(f"[DIARIZE] {message}", flush=True)
        write_status(status_path, {"status": "skipped", "reason": "audio_missing", "model": MODEL_ID})
        if args.strict:
            raise FileNotFoundError(audio_path)
        return 0

    print(f"[DIARIZE] Loading pyannote pipeline: {MODEL_ID}", flush=True)
    try:
        pipeline = Pipeline.from_pretrained(
            MODEL_ID,
            token=token,
            cache_dir=str(AI_MODELS_DIR),
        )
        pipeline.to(torch.device(args.device))
    except Exception as e:
        error_text = str(e)
        is_access_error = (
            "403 Client Error" in error_text
            or "Cannot access gated repo" in error_text
            or "not in the authorized list" in error_text
        )
        if is_access_error:
            print("[DIARIZE] Hugging Face denied access to the diarization model.", flush=True)
            print(f"[DIARIZE] Accept access for: {MODEL_ID}", flush=True)
            print("[DIARIZE] Then rerun the diarization stage; your token is already read from .env.", flush=True)
            reason = "huggingface_model_access_denied"
        else:
            print(f"[DIARIZE] Pipeline could not be loaded; speaker diarization skipped: {e}", flush=True)
            reason = "pipeline_load_failed"
        write_status(status_path, {"status": "skipped", "reason": reason, "model": MODEL_ID, "error": error_text})
        if args.strict:
            raise
        return 0

    kwargs = {}
    if args.min_speakers is not None:
        kwargs["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        kwargs["max_speakers"] = args.max_speakers

    waveform, sample_rate = read_pcm_wav(audio_path)
    input_data = {"waveform": waveform, "sample_rate": sample_rate}
    duration = waveform.shape[-1] / sample_rate
    print(f"[DIARIZE] Processing {duration:.2f}s of audio on {args.device}", flush=True)

    try:
        diarization = pipeline(input_data, **kwargs)
    except Exception as e:
        error_text = str(e)
        print(f"[DIARIZE] Inference failed; speaker diarization skipped: {error_text}", flush=True)
        write_status(status_path, {"status": "failed", "reason": "inference_failed", "model": MODEL_ID, "error": error_text})
        if args.strict:
            raise
        return 0

    try:
        tracks, output_source = extract_segments(diarization)
    except Exception as e:
        print(f"[DIARIZE] Could not read pyannote output: {e}", flush=True)
        write_status(status_path, {"status": "failed", "reason": "output_parse_failed", "model": MODEL_ID, "error": str(e)})
        if args.strict:
            raise
        return 0

    rows = []
    for index, (turn, speaker) in enumerate(tracks, 1):
        row = {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
        rows.append(row)
        if args.live:
            print(f"[DIAR {index:05d}] {turn.start:09.3f} -> {turn.end:09.3f} | {speaker}", flush=True)

    (out / "diarization.json").write_text(
        json.dumps(
            {
                "audio": str(audio_path),
                "model": MODEL_ID,
                "device": args.device,
                "segments": rows,
                "output_source": output_source,
                "exclusive": output_source == "exclusive_speaker_diarization",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_status(
        status_path,
        {"status": "complete", "segments": len(rows), "model": MODEL_ID, "output_source": output_source},
    )
    print(f"[OK] {len(rows)} speaker regions -> {out / 'diarization.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
