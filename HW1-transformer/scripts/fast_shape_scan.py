#!/usr/bin/env python3
"""
fast_shape_scan.py

Efficiently summarize large Foxglove MCAP datasets by sampling only a few messages
per topic (to infer shapes/keys) and then stopping. Also scans NPZ tensors.

Outputs (in --out dir):
  - fast_shape_report.md
  - fast_shape_report.json

Usage:
  python fast_shape_scan.py --root ./data --out ./data/_reports \
    --mcap-samples 16 --json-examples 1

Dependencies:
  pip install mcap numpy
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mcap.reader import make_reader


def try_json(b: bytes) -> Tuple[bool, Optional[Any]]:
    try:
        return True, json.loads(b.decode("utf-8"))
    except Exception:
        return False, None


def summarize_json(msgs: List[Any], max_examples: int = 1) -> Dict[str, Any]:
    """
    Summarize top-level JSON structure across a few samples:
      - most common keys (dict)
      - per-key list lengths (if lists)
      - small example blob
    """
    keys = Counter()
    list_lens = {}  # key -> list of lengths
    examples = []

    for m in msgs[:max_examples]:
        examples.append(m)
    for m in msgs:
        if isinstance(m, dict):
            keys.update(m.keys())
            for k, v in m.items():
                if isinstance(v, list):
                    list_lens.setdefault(k, []).append(len(v))
        elif isinstance(m, list):
            # root-level list
            list_lens.setdefault("<root>", []).append(len(m))

    list_stats = {
        k: {
            "min": int(min(v)) if v else None,
            "max": int(max(v)) if v else None,
            "mean": float(sum(v) / len(v)) if v else None,
            "n": len(v),
        }
        for k, v in list_lens.items()
    }

    return {
        "common_keys": [k for k, _ in keys.most_common(24)],
        "list_field_stats": list_stats,
        "example": examples[0] if examples else None,
    }


def scan_mcap_shapes(
    path: Path,
    per_topic_samples: int = 16,
    json_examples: int = 1,
) -> Dict[str, Any]:
    """
    Sample the first `per_topic_samples` messages per topic and stop early once all
    seen topics have enough samples. This avoids a full pass on huge files.
    """
    out: Dict[str, Any] = {"file": str(path), "type": "mcap", "topics": {}}

    with open(path, "rb") as f:
        reader = make_reader(f)

        topic_info: Dict[str, Dict[str, Any]] = {}
        samples: Dict[str, List[bytes]] = defaultdict(list)
        t0_topic: Dict[str, int] = {}
        t1_topic: Dict[str, int] = {}

        # We don't know all topics upfront; we stop when every *seen* topic
        # has reached the sample cap.
        def all_capped() -> bool:
            if not topic_info:
                return False
            return all(len(samples[t]) >= per_topic_samples for t in topic_info.keys())

        for schema, channel, message in reader.iter_messages():
            t = channel.topic
            if t not in topic_info:
                topic_info[t] = {
                    "schema_name": schema.name or "",
                    "encoding": schema.encoding or "",
                    "count_sampled": 0,
                }
                t0_topic[t] = message.log_time

            # Collect only up to per_topic_samples raw payloads per topic
            if len(samples[t]) < per_topic_samples:
                samples[t].append(message.data)
                topic_info[t]["count_sampled"] += 1
                t1_topic[t] = message.log_time

            # Early stop if all current topics have hit the cap
            if all_capped():
                break

        # Build summaries from sampled data
        for t, info in topic_info.items():
            payloads = samples.get(t, [])
            if not payloads:
                continue

            # Try JSON; if fails -> binary summary (size sample only)
            json_ok = True
            decoded_json = []
            for b in payloads[:min(8, len(payloads))]:
                ok, obj = try_json(b)
                if ok:
                    decoded_json.append(obj)
                else:
                    json_ok = False
                    break

            topic_summary: Dict[str, Any] = {
                "schema_name": info["schema_name"],
                "encoding": info["encoding"],
                "count_sampled": info["count_sampled"],
                "first_time_ns_sampled": t0_topic.get(t),
                "last_time_ns_sampled": t1_topic.get(t),
            }

            if json_ok and decoded_json:
                topic_summary["payload"] = "json"
                topic_summary["json_summary"] = summarize_json(
                    decoded_json, max_examples=json_examples
                )
            else:
                topic_summary["payload"] = "binary"
                topic_summary["binary_size_bytes_sample"] = [
                    len(b) for b in payloads[:min(8, len(payloads))]
                ]

            out["topics"][t] = topic_summary

    return out


def scan_npz_shapes(path: Path) -> Dict[str, Any]:
    """
    Read NPZ header and report array names and shapes. (np.load reads lazily;
    shape metadata is cheap even for large arrays.)
    """
    out: Dict[str, Any] = {"file": str(path), "type": "npz", "arrays": {}}
    with np.load(path, allow_pickle=True) as data:
        for k in data.files:
            arr = data[k]
            # Avoid materializing big arrays; we only touch .shape and .dtype
            out["arrays"][k] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    return out


def walk_and_scan(
    root: Path,
    per_topic_samples: int,
    json_examples: int,
    include_npz: bool,
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    files = list(root.rglob("*"))
    files.sort()

    for p in files:
        if p.suffix.lower() == ".mcap":
            try:
                rep = scan_mcap_shapes(p, per_topic_samples, json_examples)
            except Exception as e:
                rep = {"file": str(p), "type": "mcap", "error": str(e)}
            reports.append(rep)
        elif include_npz and p.suffix.lower() == ".npz":
            try:
                rep = scan_npz_shapes(p)
            except Exception as e:
                rep = {"file": str(p), "type": "npz", "error": str(e)}
            reports.append(rep)
    return reports


def write_reports(reports: List[Dict[str, Any]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / "fast_shape_report.json"
    mpath = out_dir / "fast_shape_report.md"

    with open(jpath, "w") as f:
        json.dump(reports, f, indent=2)

    def md_for(rep: Dict[str, Any]) -> str:
        name = Path(rep["file"]).name
        if "error" in rep:
            return f"### {name}\n\n**Error:** {rep['error']}\n"
        if rep["type"] == "npz":
            lines = [f"### {name} (NPZ)\n", "| Array | Shape | Dtype |", "|---|---|---|"]
            for k, a in rep["arrays"].items():
                lines.append(f"| `{k}` | `{tuple(a['shape'])}` | `{a['dtype']}` |")
            lines.append("")
            return "\n".join(lines)

        # MCAP
        lines = [
            f"### {name} (MCAP)\n",
            "| Topic | Schema | Encoding | Sampled | JSON? | List fields (min..max / mean) |",
            "|---|---|---|---:|:---:|---|",
        ]
        for topic, tinfo in rep.get("topics", {}).items():
            if tinfo.get("payload") == "json":
                js = tinfo.get("json_summary", {})
                # Compact list field stats line
                lf = []
                for fk, st in js.get("list_field_stats", {}).items():
                    rng = f"{st['min']}..{st['max']}" if st["min"] is not None else "-"
                    mean = f"{st['mean']:.2f}" if st["mean"] is not None else "-"
                    lf.append(f"{fk}: {rng} / {mean}")
                lf_str = "; ".join(lf) if lf else "-"
                lines.append(
                    f"| `{topic}` | `{tinfo.get('schema_name','')}` | `{tinfo.get('encoding','')}` | "
                    f"{tinfo.get('count_sampled',0)} | ✅ | {lf_str} |"
                )
            else:
                sizes = tinfo.get("binary_size_bytes_sample", [])
                size_str = f"bytes={sizes[:4]}..." if sizes else "-"
                lines.append(
                    f"| `{topic}` | `{tinfo.get('schema_name','')}` | `{tinfo.get('encoding','')}` | "
                    f"{tinfo.get('count_sampled',0)} | ❌ | {size_str} |"
                )
        lines.append("")
        return "\n".join(lines)

    with open(mpath, "w") as f:
        f.write("# Fast shape scan\n\n")
        for rep in reports:
            f.write(md_for(rep))

    print(f"Wrote:\n  - {jpath}\n  - {mpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Data directory to scan recursively")
    ap.add_argument("--out", required=True, help="Output directory for reports")
    ap.add_argument("--mcap-samples", type=int, default=16, help="Messages to sample per topic (stop early)")
    ap.add_argument("--json-examples", type=int, default=1, help="JSON examples to store per topic")
    ap.add_argument("--include-npz", action="store_true", help="Also summarize .npz files (X/y shapes)")
    args = ap.parse_args()

    root = Path(args.root)
    reports = walk_and_scan(
        root=root,
        per_topic_samples=args.mcap_samples,
        json_examples=args.json_examples,
        include_npz=args.include_npz,
    )
    write_reports(reports, Path(args.out))


if __name__ == "__main__":
    main()

