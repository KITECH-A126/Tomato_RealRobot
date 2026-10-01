"""Observe HOME while replay_trajectory waits at B. Never sends motion commands.

Dedicated SDK connections receive joints; firmware queries may transmit CAN
requests. RealRobot.__init__ is deliberately NOT called: it changes speed and
opens cameras. Only its existing read_joints/close methods are used.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from real_robot import RealRobot
import replay_trajectory as replay


def connect_reader():
    from pyAgxArm import AgxArmFactory, create_agx_arm_config, resolve_firmware_profile
    bot = RealRobot.__new__(RealRobot)
    bot._closed = False
    bot._arms = {}
    bot._sample_times = {}
    bot._camera_streams = {}
    bot._camera_context = None
    bot._camera_sdk = None
    try:
        for arm, count in (("nero", 7), ("piper", 6)):
            config = create_agx_arm_config(robot=arm, comm="can", channel=f"can_{arm}")
            driver = AgxArmFactory.create_arm(config)
            bot._arms[arm] = driver
            driver.connect()
            deadline = time.monotonic() + 3
            while True:
                fps = driver.get_fps()
                if np.isfinite(fps) and fps > 0:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"{arm}: CAN feedback timeout")
                time.sleep(.01)
            firmware = driver.get_firmware(timeout=1.0)
            if not firmware:
                raise RuntimeError(f"{arm}: firmware query failed")
            profile = resolve_firmware_profile(arm, firmware["software_version"])
            if profile != config["firmeware_version"]:
                driver.disconnect()
                del bot._arms[arm]
                config = create_agx_arm_config(robot=arm, comm="can", channel=f"can_{arm}",
                                              firmeware_version=profile)
                driver = AgxArmFactory.create_arm(config)
                bot._arms[arm] = driver
                driver.connect()
            if driver.joint_nums != count or list(config["joint_names"]) != [f"joint{i}" for i in range(1, count + 1)]:
                raise RuntimeError(f"{arm}: unexpected joint order/count")
        deadline = time.monotonic() + 3
        while True:
            try:
                bot.read_joints()
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.01)
        return bot
    except BaseException:
        bot.close()
        raise


def observe(bot, home, rows, duration=10.0):
    begin = time.monotonic()
    while True:
        state = bot.read_joints()
        host = time.time()
        elapsed = time.monotonic() - begin
        actual = np.array(state["nero"] + state["piper"], dtype=float)
        if actual.shape != (13,) or not np.isfinite(actual).all():
            raise RuntimeError("Invalid joint feedback")
        ages = {arm: host - stamp for arm, stamp in bot._sample_times.items()}
        error = np.abs(actual - home)
        rows.append({"elapsed_s": elapsed, "sample_t": float(state["t"]),
                     "arm_sample_t": dict(bot._sample_times), "feedback_age_s": ages,
                     "actual_rad": actual.tolist(), "abs_error_rad": error.tolist()})
        if any(not np.isfinite(age) or age < 0 or age > 1 for age in ages.values()):
            raise RuntimeError("Feedback timestamp stale or inconsistent; no correction")
        if np.degrees(error).max() > 10.0:
            raise RuntimeError(f"HOME error {np.degrees(error).max():.6f} deg > 10 deg")
        if elapsed >= duration:
            return
        time.sleep(min(1 / 30, duration - elapsed))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", default=str(replay.TRAJ), help="Must match the waiting replay process")
    ap.add_argument("--out", type=Path, help="New directory; existing paths are never overwritten")
    args = ap.parse_args()
    if replay.TRACK_STOP_DEG != 10.0:
        raise RuntimeError("Replay threshold is no longer 10 degrees; review before measuring")
    with np.load(args.traj, allow_pickle=False) as data:
        nero = np.asarray(data["q_cmd"][0], dtype=float)
        piper = np.asarray(data["gaze_q"], dtype=float)
    if nero.shape != (7,) or piper.shape != (6,):
        raise ValueError("Invalid HOME shapes")
    home = np.concatenate((nero, piper))
    if not np.isfinite(home).all():
        raise ValueError("Non-finite HOME")
    print("Keep replay waiting at B; do NOT press Enter or operate another controller during measurement.")
    print("Read-only observation: no move/enable/disable/speed/stop commands; no camera connection.")
    print("NERO HOME:", nero.tolist())
    print("PiPER HOME:", piper.tolist())
    out = args.out or Path(__file__).resolve().parent / (
        "home_observation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    report = {"started_at": datetime.now().astimezone().isoformat(),
              "traj": str(Path(args.traj).resolve()), "home_nero_rad": nero.tolist(),
              "home_piper_rad": piper.tolist(), "threshold_deg": 10.0,
              "requested_seconds": 10.0, "samples": [], "status": "incomplete",
              "reference": "NPZ q_cmd[0] / gaze_q, not live controller target readback",
              "note": "Stopping observer does NOT stop robot; software is not an E-stop"}
    bot = None
    try:
        bot = connect_reader()
        observe(bot, home, report["samples"])
        report["status"] = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="stopped", reason=f"{type(exc).__name__}: {exc}")
    finally:
        if bot is not None:
            try:
                bot.close()
            except Exception as exc:
                report.update(status="stopped", close_error=str(exc))
        rows = report["samples"]
        lines = [f"HOME observation: {report['status']}", f"Samples: {len(rows)}",
                 "Reference: NERO q_cmd[0] / PiPER gaze_q; threshold 10 deg",
                 report.get("reason", "")]
        if rows:
            maximum = np.max([row["abs_error_rad"] for row in rows], axis=0)
            report["max_abs_error_rad"] = maximum.tolist()
            report["max_abs_error_deg"] = np.degrees(maximum).tolist()
            report["observed_seconds"] = rows[-1]["elapsed_s"]
            report["distinct_arm_timestamps"] = {
                arm: len({row["arm_sample_t"][arm] for row in rows}) for arm in ("nero", "piper")}
            lines.append(f"Observed: {report['observed_seconds']:.6f} seconds")
            for i, value in enumerate(maximum):
                label = f"NERO J{i+1}" if i < 7 else f"PiPER J{i-6}"
                lines.append(f"{label}: max |actual - HOME| = {value:.9f} rad / {np.degrees(value):.6f} deg")
        lines.append("Stopped/partial results are NOT a completed 10-second test. No robot stop command sent.")
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        print("Report:", out.resolve())
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
