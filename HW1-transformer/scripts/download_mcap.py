#!/usr/bin/env python3
"""
remote_inspect_mcap.py

Download MCAP files from a list of URLs (one per line) and inspect them:
- topics, schema, encoding, counts, time ranges, approximate Hz
- try to decode JSON payloads and show example keys/values
Outputs a Markdown and JSON report.

Usage:
  python remote_inspect_mcap.py --urls urls.txt --out ./downloaded --filter calib

Notes:
- No ROS required.
- Works with Foxglove Cloud "Download" or "Share" links that return MCAP bytes.
- Auth: if your URLs require a header/token, you can:
    * pass `--header "Authorization: Bearer <TOKEN>"` multiple times, or
    * pre-generate time-limited links from the Foxglove UI.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict, Counter

import requests
from mcap.reader import make_reader


def human_rate(n: int, t0_ns: Optional[int], t1_ns: Optional[int]) -> Optional[float]:
    if n <= 1 or t0_ns is None or t1_ns is None or t1_ns <= t0_ns:
        return None
    dur_s = (t1_ns - t0_ns) / 1e9
    if dur_s <= 0:
        return None
    return n / dur_s


def try_json_decode(b: bytes):
    try:
        import json as _json
        return True, _json.loads(b.decode("utf-8"))
    except Exception:
        return False, None


def summarize_json(msgs: List[Any], max_examples: int = 3) -> Dict[str, Any]:
    examples = []
    key_counter = Counter()
    arr_len_counter = Counter()
    for m in msgs[:max_examples]:
        examples.append(m)
        if isinstance(m, dict):
            key_counter.update(m.keys())
            for k, v in m.items():
                if isinstance(v, list):
                    arr_len_counter[(k, "list_len")] += len(v)
        elif isinstance(m, list):
            arr_len_counter[("root_list", "list_len")] += len(m)
    return {
        "common_keys": list(k for k, _ in key_counter.most_common(20)),
        "array_field_lengths_example_sum": {str(k): v for k, v in arr_len_counter.items()},
        "examples": examples,
    }


def inspect_mcap_path(path: Path, max_msgs: int = 200) -> Dict[str, Any]:
    info: Dict[str, Any] = {"file": str(path), "topics": {}}
    with open(path, "rb") as f:
        reader = make_reader(f)
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
            if len(per_topic_raw_samples[t]) < max_msgs:
                per_topic_raw_samples[t].append(message.data)
        for topic, ent in per_topic.items():
            t0 = ent["first_time_ns"]; t1 = ent["last_time_ns"]
            rate = human_rate(ent["count"], t0, t1)
            topic_summary: Dict[str, Any] = {
                "schema_name": ent["schema_name"],
                "message_encoding": ent["message_encoding"],
                "count": ent["count"],
                "first_time_ns": t0,
                "last_time_ns": t1,
                "approx_rate_hz": rate,
            }
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
    ap.add_argument("--urls", required=True, help="Text file with one MCAP URL per line")
    ap.add_argument("--out", required=True, help="Directory to save MCAPs")
    ap.add_argument("--filter", default="", help="Only download/inspect files whose URL or name contains this substring")
    ap.add_argument("--header", action="append", default=[], help='HTTP header, e.g. --header "Authorization: Bearer XYZ"')
    ap.add_argument("--limit", type=int, default=100, help="Max number of URLs to process")
    ap.add_argument("--max-msgs", type=int, default=200, help="Max messages per topic to sample for JSON probing")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    headers = {}
    for h in args.header:
        if ":" in h:
            k, v = h.split(":", 1)
            headers[k.strip()] = v.strip()

    # Read URLs
    urls = []
    with open(args.urls, "r") as f:
        for line in f:
            url = line.strip()
            if not url: continue
            if args.filter and args.filter not in url:
                # also allow filter on last path component
                name = url.split("?")[0].split("/")[-1]
                if args.filter not in name:
                    continue
            urls.append(url)
            if len(urls) >= args.limit:
                break

    if not urls:
        print("No URLs to process after filtering.")
        return

    reports = []
    for i, url in enumerate(urls, 1):
        name = url.split("?")[0].split("/")[-1]
        if not name.endswith(".mcap"):
            name = name + ".mcap"
        dest = out_dir / name
        print(f"[{i}/{len(urls)}] Downloading {url} → {dest.name}")
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1<<20):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            print(f"  Download failed: {e}")
            reports.append({"file": str(dest), "error": f"download: {e}"})
            continue

        try:
            rep = inspect_mcap_path(dest, max_msgs=args.max_msgs)
            reports.append(rep)
        except Exception as e:
            print(f"  Inspect failed: {e}")
            reports.append({"file": str(dest), "error": f"inspect: {e}"})
            continue

    # Write reports
    out_json = out_dir / "remote_mcap_inspection.json"
    out_md = out_dir / "remote_mcap_inspection.md"

    with open(out_json, "w") as f:
        json.dump(reports, f, indent=2)

    def md_block(rep: Dict[str, Any]) -> str:
        from pathlib import Path as _P
        if "error" in rep:
            return f"### {(_P(rep['file']).name)}\n\n**Error:** {rep['error']}\n"
        lines = [f"### {(_P(rep['file']).name)}",
                 "",
                 "| Topic | Schema | Encoding | Count | Rate (Hz) |",
                 "|---|---|---:|---:|---:|"]
        for topic, tinfo in rep["topics"].items():
            lines.append(f"| `{topic}` | `{tinfo.get('schema_name','')}` | `{tinfo.get('message_encoding','')}` | {tinfo.get('count',0)} | {tinfo.get('approx_rate_hz') or '-'} |")
        lines.append("")
        for topic, tinfo in rep["topics"].items():
            if tinfo.get("payload_type") == "json":
                js = tinfo.get("json_summary", {})
                lines.append(f"- **{topic}** JSON keys: `{', '.join(js.get('common_keys', []))}`")
                examples = js.get("examples", [])[:1]
                if examples:
                    import json as _json
                    ex = _json.dumps(examples[0], indent=2)[:800]
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
        f.write("# Remote MCAP: inspection report\n\n")
        for rep in reports:
            f.write(md_block(rep))
            f.write("\n")

    print(f"\nWrote:\n  - {out_json}\n  - {out_md}\n")
    print("Done.")


if __name__ == "__main__":
    main()

