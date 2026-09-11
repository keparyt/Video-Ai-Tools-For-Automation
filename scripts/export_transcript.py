#!/usr/bin/env python3
"""Export an aligned/diarized JSON transcript to SRT, VTT and TXT."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path


def ts(x, comma=True):
    x=max(0,float(x)); ms=round((x-int(x))*1000); total=int(x)+(ms//1000); ms%=1000; h,r=divmod(total,3600); m,s=divmod(r,60); return f"{h:02d}:{m:02d}:{s:02d}{',' if comma else '.'}{ms:03d}"

def speaker_at(start,end,diar):
    best=None; score=0
    for d in diar:
        overlap=max(0,min(end,d['end'])-max(start,d['start']))
        if overlap>score: score=overlap; best=d['speaker']
    return best

def main():
    p=argparse.ArgumentParser(description="Export transcript files")
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--diarization",type=Path,default=None)
    args=p.parse_args(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    data=json.loads(args.input.read_text(encoding='utf-8')); diar=[]
    if args.diarization and args.diarization.exists(): diar=json.loads(args.diarization.read_text(encoding='utf-8')).get('segments',[])
    segs=data.get('segments',data if isinstance(data,list) else [])
    srt=[]; vtt=['WEBVTT','']; txt=[]
    for i,s in enumerate(segs,1):
        start=float(s.get('start',0)); end=float(s.get('end',start)); text=s.get('text','').strip()
        sp=speaker_at(start,end,diar) if diar else None
        if sp: text=f"[{sp}] {text}"
        if not text: continue
        srt += [str(i),f"{ts(start)} --> {ts(end)}",text,'']
        vtt += [f"{ts(start,False)} --> {ts(end,False)}",text,'']
        txt.append(f"[{ts(start,False)}] {text}")
    (out/'final.srt').write_text('\n'.join(srt),encoding='utf-8'); (out/'final.vtt').write_text('\n'.join(vtt),encoding='utf-8'); (out/'final.txt').write_text('\n'.join(txt)+'\n',encoding='utf-8')
    print(f"[OK] Exported {len(segs)} transcript segments -> {out}")

if __name__=='__main__': main()
