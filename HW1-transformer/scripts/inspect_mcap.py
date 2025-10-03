#!/usr/bin/env python3
"""
inspect_mcap.py

Scan MCAP files whose names contain "calib" and summarize their contents:
- topics and schema names
- number of messages and time range
- approximate message rate
- attempt to decode payload as JSON; if successful, report example keys and shapes
- dump a Markdown summary and a JSON report

Usage:
  python inspect_mcap.py --root /path/to/data/raw [--limit 5] [--max-msgs 200]

Notes:
- No ROS required. Works with Foxglove's Python SDK recordings (JSON messages).
- If payloads are not JSON, the script reports them as binary and shows byte sizes.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
from mcap.reader import make_reader


def human_rate(n: int, t0_ns: Optional[int], t1_ns: Optional[int]) -> Optional[float]:
    if n <= 1 or t0_ns is None or t1_ns is None or t1_ns <= t0_ns:
        return None
    dur_s = (t1_ns - t0_ns) / 1e9
    if dur_s <= 0:
        return None
    return n / dur_s


def try_json_decode(b: bytes) -> Tuple[bool, Any]:
    try:
        return True, json.loads(b.decode("utf-8"))
    except Exception:
        return False, None


def summarize_json(msgs: List[Any], max_examples: int = 3) -> Dict[str, Any]:
    """Summarize JSON dict/list structure across a few messages."""
    examples = []
    key_counter = Counter()
    arr_len_counter = Counter()

    for m in msgs[:max_examples]:
        examples.append(m)
        if isinstance(m, dict):
            key_counter.update(m.keys())
            # crude: if dict has an array value, record its length
            for k, v in m.items():
                if isinstance(v, list):
                    arr_len_counter[(k, "list_len")] += len(v)
        elif isinstance(m, list):
            arr_len_counter[("root_list", "list_len")] += len(m)

    summary = {
        "common_keys": list(k for k, _ in key_counter.most_common(20)),
        "array_field_lengths_example_sum": {str(k): v for k, v in arr_len_counter.items()},
        "examples": examples,
    }
    return summary


def inspect_file(path: Path, max_msgs: int = 200) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "file": str(path),
        "topics": {},
    }
    with open(path, "rb") as f:
        reader = make_reader(f)

        # Collect a little per-topic info
        per_topic: Dict[str, Dict[str, Any]] = {}
        per_topic_raw_samples: Dict[str, List[bytes]] = defaultdict(list)
        for schema, channel, message in reader.iter_messages():
            t = channel.topic
            ent = per_topic.setdefault(t, {
                "schema_name": schema.name or "",
                "message_encoding": schema.encoding or "",
                "count": 0,
                "first_time_ns": None,
                "last_time_ns": None,
            })
            ent["count"] += 1
            if ent["first_time_ns"] is None:
                ent["first_time_ns"] = message.log_time
            ent["last_time_ns"] = message.log_time

            # Keep up to max_msgs raw samples for JSON probing
            if len(per_topic_raw_samples[t]) < max_msgs:
                per_topic_raw_samples[t].append(message.data)

        # Build summaries
        for topic, ent in per_topic.items():
            t0 = ent["first_time_ns"]
            t1 = ent["last_time_ns"]
            rate = human_rate(ent["count"], t0, t1)
            topic_summary: Dict[str, Any] = {
                "schema_name": ent["schema_name"],
                "message_encoding": ent["message_encoding"],
                "count": ent["count"],
                "first_time_ns": t0,
                "last_time_ns": t1,
                "approx_rate_hz": rate,
            }

            # Try decode as JSON
            decoded_json = []
            decode_ok = True
            for b in per_topic_raw_samples[topic][: min(10, len(per_topic_raw_samples[topic]))]:
                ok, obj = try_json_decode(b)
                if ok:
                    decoded_json.append(obj)
                else:
                    decode_ok = False
                    break
            if decode_ok and decoded_json:
                topic_summary["payload_type"] = "json"
                topic_summary["json_summary"] = summarize_json(decoded_json, max_examples=3)
            else:
                topic_summary["payload_type"] = "binary"
                sizes = [len(b) for b in per_topic_raw_samples[topic][:10]]
                topic_summary["binary_sizes_sample"] = sizes

            info["topics"][topic] = topic_summary

    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Directory to scan for *calib*.mcap files")
    ap.add_argument("--limit", type=int, default=100, help="Max number of files to scan")
    ap.add_argument("--max-msgs", type=int, default=200, help="Max messages per topic to sample for JSON probing")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted([p for p in root.rglob("*.mcap") if "calib" in p.name])[: args.limit]
    if not files:
        print(f"No MCAP files containing 'calib' found under {root}")
        return

    reports: List[Dict[str, Any]] = []
    for p in files:
        print(f"Scanning {p.name} ...")
        try:
            rep = inspect_file(p, max_msgs=args.max_msgs)
            reports.append(rep)
        except Exception as e:
            reports.append({"file": str(p), "error": str(e)})
            print(f"  Error: {e}")

    # Write JSON and Markdown
    out_json = root / "mcap_calib_inspection.json"
    out_md = root / "mcap_calib_inspection.md"

    with open(out_json, "w") as f:
        json.dump(reports, f, indent=2)

    def md_block(rep: Dict[str, Any]) -> str:
        if "error" in rep:
            return f"### {Path(rep['file']).name}\n\n**Error:** {rep['error']}\n"
        lines = [f"### {Path(rep['file']).name}",
                 "",
                 "| Topic | Schema | Encoding | Count | Rate (Hz) |",
                 "|---|---|---:|---:|---:|"]
        for topic, tinfo in rep["topics"].items():
            lines.append(f"| `{topic}` | `{tinfo.get('schema_name','')}` | `{tinfo.get('message_encoding','')}` | {tinfo.get('count',0)} | {tinfo.get('approx_rate_hz') or '-'} |")
        lines.append("")
        # Show JSON summaries
        for topic, tinfo in rep["topics"].items():
            if tinfo.get("payload_type") == "json":
                js = tinfo.get("json_summary", {})
                lines.append(f"- **{topic}** JSON keys: `{', '.join(js.get('common_keys', []))}`")
                examples = js.get("examples", [])[:1]
                if examples:
                    ex = json.dumps(examples[0], indent=2)[:800]
                    lines.append("  - example:")
                    lines.append("")
                    lines.append("```json")
                    lines.append(ex)
                    lines.append("```")
            else:
                sizes = tinfo.get("binary_sizes_sample", [])
                lines.append(f"- **{topic}** binary payload; sample sizes: {sizes}")
        lines.append("")
        return "\n".join(lines)

    with open(out_md, "w") as f:
        f.write("# MCAP calibration files: inspection report\n\n")
        for rep in reports:
            f.write(md_block(rep))
            f.write("\n")

    print(f"\nWrote:\n  - {out_json}\n  - {out_md}\n")
    print("Open the Markdown for a human-friendly overview.")
    

if __name__ == "__main__":
    main()
