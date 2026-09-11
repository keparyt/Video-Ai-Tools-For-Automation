#!/usr/bin/env python3
"""Optional pyannote speaker diarization stage."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description="Speaker diarization with pyannote")
    p.add_argument("--audio",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--token",default=None,help="Hugging Face token; otherwise HF_TOKEN is used")
    p.add_argument("--min-speakers",type=int,default=None)
    p.add_argument("--max-speakers",type=int,default=None)
    args=p.parse_args()
    try:
        from pyannote.audio import Pipeline
    except ImportError as e: raise RuntimeError("Install pyannote.audio: pip install pyannote.audio") from e
    token=args.token or os.getenv("HF_TOKEN")
    if not token: raise RuntimeError("Speaker diarization requires HF_TOKEN and access to the pyannote diarization model")
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    print("[DIARIZE] Loading pyannote speaker diarization...")
    pipeline=Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",use_auth_token=token)
    kwargs={}
    if args.min_speakers is not None: kwargs["min_speakers"]=args.min_speakers
    if args.max_speakers is not None: kwargs["max_speakers"]=args.max_speakers
    diarization=pipeline(str(args.audio),**kwargs)
    rows=[]
    for turn,_,speaker in diarization.itertracks(yield_label=True): rows.append({"start":turn.start,"end":turn.end,"speaker":speaker})
    (out/"diarization.json").write_text(json.dumps({"audio":str(args.audio.resolve()),"segments":rows},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"[OK] {len(rows)} speaker regions -> {out/'diarization.json'}")

if __name__ == "__main__": main()
