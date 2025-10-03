#!/usr/bin/env python3
"""
prepare_ee_dataset.py (VAMP-only, KISS)

Reads /sensor_raw and /joint_states from MCAP, computes EE [x,y,z] with VAMP's compiled Panda model,
and writes Parquet shards: [timestamp_ns, sensor_raw (padded), ee_x, ee_y, ee_z].

Dependencies:
  uv pip install mcap mcap-ros2-support pyarrow numpy vamp-planner
Usage:
  python prepare_ee_dataset.py \
    --mcap_glob "HW1-transformer/data/*.mcap" \
    --out_dir ./train_out \
    --max_align_ms 15 \
    --target_hz 30
"""

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple
from types import SimpleNamespace

import numpy as np

# Parquet (required)
import pyarrow as pa
import pyarrow.parquet as pq

# MCAP decoded reader w/ ROS2 CDR
import inspect
from mcap_ros2.decoder import DecoderFactory as Ros2DecoderFactory
from mcap.reader import make_reader

# import inspect


# ---------- VAMP (required) ----------
try:
    import vamp  # expects vamp.panda with eefk(q)->4x4
except Exception as e:
    print("ERROR: VAMP not available. Install vamp-planner and ensure vamp.panda is importable.", file=sys.stderr)
    raise


# Panda joint order expected by VAMP
PANDA_Q_ORDER = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)



def iter_ros2_messages_decoded(mcap_path: str, topics: list[str]):
    """
    Yield SimpleNamespace(topic, ros_msg, log_time) for decoded ROS 2 messages.

    Per mcap.reader API, supply the ROS2 CDR decoder via make_reader(..., decoder_factories=[...])
    and then call iter_decoded_messages() with no decoder args.
    """
    with open(mcap_path, "rb") as f:
        rdr = make_reader(f, decoder_factories=[Ros2DecoderFactory()])
        for schema, channel, message, ros2_msg in rdr.iter_decoded_messages(topics=topics):
            yield SimpleNamespace(
                topic=channel.topic,
                ros_msg=ros2_msg,
                log_time=message.log_time,
            )




def decode_joint_state(msg) -> Tuple[int, Dict[str, float]]:
    """
    Returns (timestamp_ns, {joint_name: position})
    """
    t = msg.log_time  # ns
    js = msg.ros_msg  # sensor_msgs/msg/JointState
    names = list(js.name)
    pos = list(js.position)
    return t, dict(zip(names, pos))


def decode_sensor_raw(msg) -> Tuple[int, np.ndarray]:
    """
    Returns (timestamp_ns, np.int32[N]) for std_msgs/msg/Int32MultiArray
    """
    t = msg.log_time
    m = msg.ros_msg
    arr = np.asarray(m.data, dtype=np.int32)
    return t, arr


@dataclass
class VampFKSolver:
    def __post_init__(self):
        if not hasattr(vamp, "panda"):
            raise RuntimeError("vamp.panda not found. Is VAMP installed correctly?")
        self.robot = vamp.panda
        self.q_order = PANDA_Q_ORDER

    def fk_xyz(self, joint_state: Dict[str, float]) -> np.ndarray:
        q = np.array([float(joint_state.get(name, 0.0)) for name in self.q_order], dtype=np.float64)
        T = self.robot.eefk(q)  # 4x4
        if getattr(T, "shape", None) != (4, 4):
            raise RuntimeError("vamp.panda.eefk(q) did not return a 4x4 transform.")
        return np.asarray(T[:3, 3], dtype=np.float64)


def nearest_time_index(target_ns: int, times_ns: np.ndarray) -> int:
    """
    Returns index of nearest time in times_ns to target_ns.
    """
    idx = np.searchsorted(times_ns, target_ns)
    if idx <= 0:
        return 0
    if idx >= len(times_ns):
        return len(times_ns) - 1
    before = times_ns[idx - 1]
    after = times_ns[idx]
    return idx - 1 if abs(target_ns - before) <= abs(after - target_ns) else idx


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap_glob", required=True, help="Glob for input MCAPs (e.g., 'data/*.mcap').")
    ap.add_argument("--out_dir", required=True, help="Output directory for Parquet shards.")
    ap.add_argument("--max_align_ms", type=float, default=15.0, help="Max |Δt| between sensor and joint_states.")
    ap.add_argument("--target_hz", type=float, default=30.0, help="Downsample /sensor_raw to ~this rate (stride).")
    ap.add_argument("--rows_per_shard", type=int, default=100_000)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    mcap_paths = sorted(glob.glob(os.path.expanduser(args.mcap_glob)))
    if not mcap_paths:
        print("No MCAPs matched.", file=sys.stderr)
        sys.exit(1)

    fk = VampFKSolver()

    shard_rows = args.rows_per_shard
    max_delta_ns = int(args.max_align_ms * 1e6)
    out_idx = 0
    buf_ts: List[int] = []
    buf_s: List[np.ndarray] = []
    buf_ee: List[np.ndarray] = []

    def flush_shard():
        nonlocal out_idx, buf_ts, buf_s, buf_ee
        if not buf_ts:
            return
        shard_base = os.path.join(args.out_dir, f"ee_dataset_{out_idx:05d}")
        # Pad ragged sensor arrays to consistent width per shard.
        max_len = max(x.shape[0] for x in buf_s)
        sensor_2d = np.zeros((len(buf_s), max_len), dtype=np.int32)
        for i, v in enumerate(buf_s):
            sensor_2d[i, :v.shape[0]] = v

        table = pa.table({
            "timestamp_ns": pa.array(buf_ts, type=pa.int64()),
            "ee_x": pa.array([float(v[0]) for v in buf_ee], type=pa.float32()),
            "ee_y": pa.array([float(v[1]) for v in buf_ee], type=pa.float32()),
            "ee_z": pa.array([float(v[2]) for v in buf_ee], type=pa.float32()),
            "sensor_raw": pa.FixedSizeListArray.from_arrays(
                pa.array(sensor_2d.reshape(-1), type=pa.int32()),
                list_size=max_len
            ),
        })
        pq.write_table(table, shard_base + ".parquet", compression="zstd")
        print(f"[write] {shard_base}.parquet rows={len(buf_ts)}")

        out_idx += 1
        buf_ts.clear()
        buf_s.clear()
        buf_ee.clear()

    for mcap_path in mcap_paths:
        # First pass: collect times/values for only the two topics we need
        js_times: List[int] = []
        js_vecs: List[Dict[str, float]] = []
        sr_times: List[int] = []
        sr_vals: List[np.ndarray] = []

        print(f"[read] {mcap_path}")
        for msg in iter_ros2_messages_decoded(mcap_path, topics=["/joint_states", "/sensor_raw"]):
            if msg.topic == "/joint_states":
                t, jdict = decode_joint_state(msg)
                js_times.append(t)
                js_vecs.append(jdict)
            elif msg.topic == "/sensor_raw":
                t, arr = decode_sensor_raw(msg)
                sr_times.append(t)
                sr_vals.append(arr)

        if not js_times or not sr_times:
            print(f"[warn] Missing required topics in {mcap_path}; skipping.", file=sys.stderr)
            continue

        js_times = np.asarray(js_times, dtype=np.int64)
        sr_times = np.asarray(sr_times, dtype=np.int64)

        # Downsample /sensor_raw by stride to ~target_hz
        if args.target_hz > 0 and len(sr_times) > 1:
            duration_s = (sr_times[-1] - sr_times[0]) * 1e-9
            approx_rate = len(sr_times) / max(duration_s, 1e-9)
            stride = max(1, int(round(approx_rate / args.target_hz)))
        else:
            stride = 1

        for i in range(0, len(sr_times), stride):
            t = sr_times[i]
            s = sr_vals[i]
            j_idx = nearest_time_index(t, js_times)
            if abs(int(t) - int(js_times[j_idx])) > max_delta_ns:
                continue  # no nearby joint_state for alignment

            ee = fk.fk_xyz(js_vecs[j_idx]).astype(np.float32)  # [x,y,z]
            buf_ts.append(int(t))
            buf_s.append(s)
            buf_ee.append(ee)

            if len(buf_ts) >= shard_rows:
                flush_shard()

    flush_shard()
    print("Done.")


if __name__ == "__main__":
    main()
