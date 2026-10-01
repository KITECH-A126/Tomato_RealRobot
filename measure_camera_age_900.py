"""Read-only joint + dual-camera frame-age diagnostic. No robot setters.

RealRobot.__init__ is bypassed to avoid speed changes. Joint receive connections
come from observe_home_hold; camera setup mirrors current real_robot.py modes.
Actual RealRobot grab methods perform conversion/alignment. Ages are measured
immediately AFTER return, not at USB arrival. No image files are written.
"""
import argparse
from collections import deque
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from observe_home_hold import connect_reader


class FrameTap:
    """Capture exactly the frameset chosen by grab; SDK keeps writing to deque."""
    def __init__(self, queue):
        self.queue, self.frame = queue, None

    def __len__(self):
        return len(self.queue)

    def __getitem__(self, index):
        frame = self.queue[index]
        self.frame = frame
        return frame

    def clear(self):
        self.queue.clear()


def connect_cameras(bot, metadata):
    import pyorbbecsdk as sdk
    bot._camera_sdk = sdk
    bot._camera_context = sdk.Context()
    bot._camera_context.set_timestamp_clock_type(sdk.OBClockType.REALTIME)
    if bot._camera_context.get_timestamp_clock_type() != sdk.OBClockType.REALTIME:
        raise RuntimeError("SDK clock is not REALTIME")
    devices = bot._camera_context.query_devices()
    metadata["orbbec_core_version"] = str(sdk.get_version())
    for role, pid, height, fmt in (("wrist", 0x0840, 800, sdk.OBFormat.YUYV),
                                  ("gaze", 0x0800, 720, sdk.OBFormat.MJPG)):
        matches = [i for i in range(devices.get_count()) if devices.get_device_pid_by_index(i) == pid]
        if len(matches) != 1:
            raise RuntimeError(f"{role}: expected exactly one camera, found {len(matches)}")
        device = devices.get_device_by_index(matches[0])
        if not device.is_global_timestamp_supported():
            raise RuntimeError(f"{role}: global timestamp unsupported")
        device.enable_global_timestamp(True)
        pipeline = sdk.Pipeline(device)
        pipeline.disable_health_monitor()
        profile = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR).get_video_stream_profile(
            1280, height, fmt, 30)
        if (profile.get_width(), profile.get_height(), profile.get_format(), profile.get_fps()) != (1280, height, fmt, 30):
            raise RuntimeError(f"{role}: requested mode unavailable; no fallback")
        config = sdk.Config()
        config.disable_all_stream()
        config.enable_stream(profile)
        align = None
        if role == "gaze":
            depth = pipeline.get_stream_profile_list(sdk.OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
            if depth.get_format() != sdk.OBFormat.Y16:
                raise RuntimeError("Depth must be Y16")
            config.enable_stream(depth)
            config.set_align_mode(sdk.OBAlignMode.DISABLE)
            config.set_frame_aggregate_output_mode(sdk.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            pipeline.enable_frame_sync()
            align = sdk.AlignFilter(align_to_stream=sdk.OBStreamType.COLOR_STREAM)
            align.set_match_target_resolution(True)
        for name in ("OB_PROP_COLOR_MIRROR_BOOL", "OB_PROP_COLOR_FLIP_BOOL", "OB_PROP_COLOR_ROTATE_INT"):
            prop = getattr(sdk.OBPropertyID, name)
            if device.is_property_supported(prop, sdk.OBPermissionType.PERMISSION_READ):
                value = device.get_int_property(prop) if name.endswith("_INT") else device.get_bool_property(prop)
                if value:
                    raise RuntimeError(f"{role}: unsupported geometry setting {name}={value}")
        queue = deque(maxlen=1)
        bot._camera_streams[role] = {"pipeline": pipeline, "device": device,
                                     "latest": FrameTap(queue), "align": align}
        pipeline.start(config, queue.append)
        info = device.get_device_info()
        metadata[role] = {"serial": info.get_serial_number(), "firmware": info.get_firmware_version(),
                          "width": 1280, "height": height, "format": str(fmt), "fps": 30}
        deadline = time.monotonic() + 3
        while not queue:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{role}: no first frame")
            time.sleep(.001)


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"count": 0}
    return {"count": len(a), "median_ms": float(np.median(a)*1000),
            "p95_ms": float(np.percentile(a, 95)*1000), "p99_ms": float(np.percentile(a, 99)*1000),
            "max_ms": float(a.max()*1000), "over_100ms": int((a > .1).sum()),
            "negative_count": int((a < 0).sum())}


def collect(bot, report, target=900, timeout=180):
    seen = {r: set() for r in ("wrist", "gaze")}
    counts = {r: 0 for r in seen}
    begin = time.monotonic()
    offset0 = time.time() - begin
    report["measurement_start_monotonic"] = begin
    next_progress = begin + 5
    while min(counts.values()) < target:
        if time.monotonic() - begin >= timeout:
            raise TimeoutError(f"Distinct frames not completed: {counts}")
        j = bot.read_joints()
        joint_host = time.time()
        joint_stamps = [float(bot._sample_times[a]) for a in ("nero", "piper")]
        q = np.asarray(list(j["nero"]) + list(j["piper"]), dtype=float)
        if q.shape != (13,) or not np.isfinite(q).all():
            raise RuntimeError("Invalid joints")
        attempt = len(report["joints"])
        report["joints"].append([joint_host, float(j["t"]), *joint_stamps, *q.tolist()])
        if any(not np.isfinite(t) or not 0 <= joint_host - t <= 1 for t in joint_stamps):
            raise RuntimeError("Joint timestamp stale/inconsistent; no correction")
        # Keep BOTH calls running even if one camera reaches its quota earlier.
        w = bot.grab_wrist()
        wh = time.time()
        wm = time.monotonic()
        wf = bot._camera_streams["wrist"]["latest"].frame
        g = bot.grab_gaze()
        gh = time.time()
        gm = time.monotonic()
        gf = bot._camera_streams["gaze"]["latest"].frame
        for role, result, host, mono, frames in (("wrist", w, wh, wm, wf), ("gaze", g, gh, gm, gf)):
            color = frames.get_color_frame()
            index, device_us = int(color.get_index()), int(color.get_timestamp_us())
            global_us = int(color.get_global_timestamp_us())
            t = float(result[-1])
            if global_us <= 0 or not np.isfinite(t) or t != float(global_us * 1e-6):
                raise RuntimeError(f"{role}: invalid/mismatched frame timestamp")
            key = (index, device_us)
            duplicate = key in seen[role]
            selected = not duplicate and counts[role] < target
            seen[role].add(key)
            counts[role] += int(selected)
            report[role].append([attempt, index, device_us, global_us, t, host, host-t,
                                 mono, int(duplicate), int(selected), float(j["t"])])
            if "received_profile" not in report["camera_metadata"][role]:
                profile = color.get_stream_profile().as_video_stream_profile()
                report["camera_metadata"][role]["received_profile"] = {
                    "width": profile.get_width(), "height": profile.get_height(),
                    "format": str(color.get_format()), "fps": profile.get_fps(),
                    "rgb_shape": list(result[0].shape)}
        if abs((gh-gm) - offset0) > .01:
            raise RuntimeError("Host wall/monotonic offset changed >10ms (or scheduling delay); inspect partial data")
        if time.monotonic() >= next_progress:
            print(f"Distinct frames: {counts}; read_joints calls: {len(report['joints'])}", flush=True)
            next_progress = time.monotonic() + 5
        time.sleep(.001)  # Polling backoff only, not a 30 Hz controller.
    report["measurement_elapsed_s"] = time.monotonic() - begin


def save(out, report):
    arrays = {r: np.asarray(report[r], dtype=float).reshape(-1, 11) for r in ("wrist", "gaze")}
    summary = {}
    payload = {"joint_columns": np.array(["host_time", "joint_t", "nero_t", "piper_t"] +
                  [f"nero_j{i}" for i in range(1,8)] + [f"piper_j{i}" for i in range(1,7)]),
               "joints": np.asarray(report["joints"], float).reshape(-1,17)}
    columns = ["attempt", "frame_index", "device_us", "global_us", "frame_t", "host_time",
               "age_s", "host_monotonic", "duplicate", "selected", "joint_t"]
    payload["camera_columns"] = np.array(columns)
    lines = ["Camera | count | median_ms | p95_ms | p99_ms | max_ms | >100ms"]
    for role, a in arrays.items():
        selected = a[a[:,9] == 1]
        summary[role] = stats(selected[:,6])
        summary[role]["duplicate_calls"] = int(a[:,8].sum())
        payload[role + "_all_calls"] = a
        payload[role + "_selected"] = selected
        s = summary[role]
        if s["count"]:
            lines.append(f"{role} | {s['count']} | {s['median_ms']:.3f} | {s['p95_ms']:.3f} | "
                         f"{s['p99_ms']:.3f} | {s['max_ms']:.3f} | {s['over_100ms']}")
        else:
            lines.append(f"{role} | 0 | N/A | N/A | N/A | N/A | N/A")
    np.savez(out / "frame_ages.npz", **payload, status=np.array(report["status"]))
    metadata = {k:v for k,v in report.items() if k not in ("wrist", "gaze", "joints")}
    metadata.update(summary=summary, read_joints_calls=len(report["joints"]))
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    name = "summary.txt" if report["status"] == "completed" else "partial_summary.txt"
    (out / name).write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Status:", report["status"], report.get("error", ""))
    print("\n".join(lines))
    print("Results:", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--timeout", type=float, default=180, help="Collection timeout seconds, excluding initialization")
    args = ap.parse_args()
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        ap.error("timeout must be positive and finite")
    out = args.out or Path(__file__).resolve().parent / ("camera_age_900_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    report = {"status":"incomplete", "created_at":datetime.now().astimezone().isoformat(),
              "wrist":[], "gaze":[], "joints":[], "camera_metadata":{}, "target_per_camera":900,
              "method":"read_joints -> grab_wrist -> grab_gaze; immediate time.time()-color t after each return",
              "dedup":"(frame index, device timestamp us); latest-frame cache, not guaranteed consecutive sensor frames",
              "warmup":"No deliberate warmup discard; collection starts after both streams ready",
              "poll_sleep_s":.001, "percentile_method":"numpy percentile default linear",
              "timestamp_conversion":"SDK global REALTIME timestamp us * 1e-6; no manual correction"}
    source = Path(__file__).with_name("real_robot.py")
    report["real_robot_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    bot = None
    print("Read-only robot queries; both cameras required. No motion/enable/disable/speed commands.", flush=True)
    try:
        bot = connect_reader()
        connect_cameras(bot, report["camera_metadata"])
        collect(bot, report, timeout=args.timeout)
        report["status"] = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="stopped", error=f"{type(exc).__name__}: {exc}")
    finally:
        if bot is not None:
            try:
                bot.close()
            except Exception as exc:
                report.update(status="stopped", close_error=str(exc))
        save(out, report)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
