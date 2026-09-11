#!/usr/bin/env python3
"""Run the complete video transcription workflow or any individual stage."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

STAGES=["transcribe","vad","align","diarize","verify","export","all"]

def call(script, args):
    cmd=[sys.executable,str(Path(__file__).with_name(script)),*args]
    print("\n"+"="*72+f"\n[MAIN] {script}\n"+"="*72,flush=True)
    subprocess.run(cmd,check=True)

def main():
    p=argparse.ArgumentParser(description="Video AI Tools workflow controller")
    p.add_argument("stage",choices=STAGES,help="all, or one stage: transcribe/vad/align/diarize/verify/export")
    p.add_argument("video",type=Path,help="Input video")
    p.add_argument("--output-dir",type=Path,default=None)
    p.add_argument("--language",default=None)
    p.add_argument("--model",default="large-v3")
    p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    p.add_argument("--compute-type",default=None)
    p.add_argument("--beam-size",type=int,default=5)
    p.add_argument("--skip-vad",action="store_true")
    p.add_argument("--skip-diarize",action="store_true")
    p.add_argument("--force",action="store_true",help="Delete the existing workflow output before starting")
    p.add_argument("--live",action="store_true",help="Print every accepted transcription segment immediately")
    args=p.parse_args()
    video=args.video.resolve(); out=(args.output_dir or video.with_name(video.stem+"_transcript")).resolve()
    if args.force and out.exists():
        import shutil; print(f"[MAIN] Removing {out}"); shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)
    audio=out/"audio_16k_mono.wav"; transcript=out/"transcript.json"; aligned=out/"aligned.json"; diar=out/"diarization.json"
    common=["--output-dir",str(out)]
    if args.stage in ("all","transcribe"):
        transcribe_args=[str(video),*common,"--model",args.model,"--device",args.device,"--beam-size",str(args.beam_size)]
        if args.language: transcribe_args += ["--language",args.language]
        if args.compute_type: transcribe_args += ["--compute-type",args.compute_type]
        if args.live: transcribe_args += ["--live"]
        call("transcribe.py",transcribe_args)
    if args.stage in ("all","vad") and not args.skip_vad: call("vad.py",["--input",str(audio),*common])
    if args.stage in ("all","align"): call("align.py",["--transcript",str(transcript),"--audio",str(audio),*common,"--device",args.device,*( ["--language",args.language] if args.language else [] ),*( ["--compute-type",args.compute_type] if args.compute_type else [] )])
    if args.stage in ("all","diarize") and not args.skip_diarize: call("diarize.py",["--audio",str(audio),*common])
    if args.stage in ("all","verify"): call("verify.py",["--transcript",str(transcript),*common])
    if args.stage in ("all","export"):
        source=aligned if aligned.exists() else transcript
        ex=["--input",str(source),*common]
        if diar.exists(): ex += ["--diarization",str(diar)]
        call("export_transcript.py",ex)
    print("\n[MAIN] WORKFLOW COMPLETE")
    print(f"[OUTPUT] {out}")
    return 0

if __name__=="__main__": raise SystemExit(main())
