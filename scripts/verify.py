#!/usr/bin/env python3
"""Flag suspicious transcript regions while preventing runaway retry loops."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def similarity(a: str, b: str) -> float:
    a_words = norm(a).split()
    b_words = norm(b).split()
    if not a_words or not b_words:
        return 0.0
    sa, sb = set(a_words), set(b_words)
    return len(sa & sb) / max(1, len(sa | sb))


def main() -> int:
    p = argparse.ArgumentParser(description="Verify Whisper transcript quality")
    p.add_argument("--transcript", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--logprob", type=float, default=-1.0)
    p.add_argument("--no-speech", type=float, default=0.65)
    p.add_argument("--compression", type=float, default=2.4)
    p.add_argument("--repeat-similarity", type=float, default=0.82)
    p.add_argument("--max-retries", type=int, default=2)
    args = p.parse_args()

    data = json.loads(args.transcript.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    flagged = []
    retry_regions = []
    previous_text = ""
    previous_end = None

    for i, s in enumerate(segments, 1):
        reasons = []
        lp = s.get("avg_logprob")
        ns = s.get("no_speech_prob")
        cr = s.get("compression_ratio")
        text = s.get("text", "").strip()

        if lp is not None and lp < args.logprob:
            reasons.append(f"low_logprob:{lp:.3f}")
        if ns is not None and ns > args.no_speech:
            reasons.append(f"high_no_speech:{ns:.3f}")
        if cr is not None and cr > args.compression:
            reasons.append(f"high_compression:{cr:.3f}")
        if re.search(r"(\b\w+\b)(?:\s+\1){2,}", text, re.I):
            reasons.append("repeated_words")

        sim = similarity(previous_text, text) if previous_text else 0.0
        gap = None if previous_end is None else float(s.get("start", 0)) - previous_end
        if previous_text and sim >= args.repeat_similarity and gap is not None and gap <= 2.0:
            reasons.append(f"near_duplicate_previous:{sim:.2f}")

        words = s.get("words") or []
        bad = [w for w in words if w.get("probability") is not None and w["probability"] < .35]
        if bad:
            reasons.append(f"low_word_probability:{len(bad)}")

        if reasons:
            item = {
                "segment_id": i,
                "start": s.get("start"),
                "end": s.get("end"),
                "text": text,
                "reasons": reasons,
                "max_retries": args.max_retries,
            }
            flagged.append(item)
            retry_regions.append(item)

        if text:
            previous_text = text
            previous_end = float(s.get("end", previous_end or 0))

    # Merge overlapping/nearby suspicious segments into a single retry region.
    # This prevents a retry worker from repeatedly processing the same audio.
    merged = []
    for item in retry_regions:
        start = max(0.0, float(item["start"]) - 1.0)
        end = float(item["end"]) + 1.0
        if merged and start <= merged[-1]["end"] + 1.0:
            merged[-1]["end"] = max(merged[-1]["end"], end)
            merged[-1]["segment_ids"].append(item["segment_id"])
        else:
            merged.append({
                "start": start,
                "end": end,
                "segment_ids": [item["segment_id"]],
                "max_retries": args.max_retries,
            })

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(args.transcript.resolve()),
        "flagged_count": len(flagged),
        "retry_region_count": len(merged),
        "max_retries_per_region": args.max_retries,
        "segments": flagged,
        "retry_regions": merged,
    }
    (out / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out / "review.txt").open("w", encoding="utf-8") as f:
        for x in flagged:
            f.write(f"[{float(x['start']):.3f} -> {float(x['end']):.3f}] {x['text']}\n")
            f.write(f"  {'; '.join(x['reasons'])}\n\n")

    print(f"[VERIFY] Flagged {len(flagged)} segments in {len(merged)} retry regions")
    print(f"[VERIFY] Retry limit: {args.max_retries} per region")
    print(f"[OK] {out / 'verification.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
