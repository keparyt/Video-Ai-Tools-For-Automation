#!/usr/bin/env python3
"""Flag transcript segments that deserve targeted manual/re-transcription review."""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Verify Whisper transcript quality")
    p.add_argument("--transcript", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--logprob", type=float, default=-1.0)
    p.add_argument("--no-speech", type=float, default=0.65)
    p.add_argument("--compression", type=float, default=2.4)
    args = p.parse_args()
    data = json.loads(args.transcript.read_text(encoding="utf-8"))
    flagged=[]
    for i, s in enumerate(data.get("segments", []), 1):
        reasons=[]
        lp=s.get("avg_logprob"); ns=s.get("no_speech_prob"); cr=s.get("compression_ratio")
        if lp is not None and lp < args.logprob: reasons.append(f"low_logprob:{lp:.3f}")
        if ns is not None and ns > args.no_speech: reasons.append(f"high_no_speech:{ns:.3f}")
        if cr is not None and cr > args.compression: reasons.append(f"high_compression:{cr:.3f}")
        text=s.get("text", "").strip()
        if re.search(r"(\b\w+\b)(?:\s+\1){2,}", text, re.I): reasons.append("repeated_text")
        if text and len(text) >= 30 and len(re.findall(r"[^\w\s']", text)) / max(1, len(text)) > .25: reasons.append("unusual_punctuation")
        words=s.get("words") or []
        if words:
            bad=[w for w in words if w.get("probability") is not None and w["probability"] < .35]
            if bad: reasons.append(f"low_word_probability:{len(bad)}")
        if reasons: flagged.append({"segment_id":i,"start":s.get("start"),"end":s.get("end"),"text":text,"reasons":reasons})
    out=args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    report={"source":str(args.transcript.resolve()),"flagged_count":len(flagged),"segments":flagged}
    (out/"verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    with (out/"review.txt").open("w",encoding="utf-8") as f:
        for x in flagged: f.write(f"[{x['start']:.3f} -> {x['end']:.3f}] {x['text']}\n  {'; '.join(x['reasons'])}\n\n")
    print(f"[VERIFY] Flagged {len(flagged)} segments")
    print(f"[OK] {out/'verification.json'}")

if __name__ == "__main__": main()
