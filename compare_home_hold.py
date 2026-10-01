"""Compare fixed HOME once vs 30 Hz repetition while replay waits at B.

Default is target preview without hardware connection. --execute and explicit
confirmation authorize commands to BOTH arms. No approach motion is generated.
Uses observe_home_hold.connect_reader, bypassing RealRobot.__init__ and cameras.
After confirmation, set NERO speed to 100% once, matching replay initialization.
PiPER retains its SDK default 50%. No enable or limit changes.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import replay_trajectory as replay
from observe_home_hold import connect_reader

HZ = 30.0
STOP_DEG = 10.0
SETTLE_S = 3.0
OBSERVE_S = 10.0


def vector(state):
    n, p = np.asarray(state["nero"], float), np.asarray(state["piper"], float)
    if n.shape != (7,) or p.shape != (6,):
        raise RuntimeError("Invalid joint count")
    q = np.concatenate((n, p))
    if not np.isfinite(q).all():
        raise RuntimeError("Invalid joint values")
    return q


def check_limits(q, limits):
    if ((q < limits[:, 0]) | (q > limits[:, 1])).any():
        raise RuntimeError("URDF joint limit violation; no clamping")


def check_ready(bot):
    """Read cached status; do not enable, clear faults or switch modes."""
    now = time.time()
    for arm, driver in bot._arms.items():
        sample = driver.get_arm_status()
        if sample is None or not 0 <= now - sample.timestamp <= 1:
            raise RuntimeError(f"{arm}: stale/missing arm status")
        s = sample.msg
        val = lambda x: getattr(x, "value", x)
        if val(s.ctrl_mode) != 1 or val(s.arm_status) != 0:
            raise RuntimeError(f"{arm}: not normal CAN control: {s}")
        for i in range(1, driver.joint_nums + 1):
            if (getattr(s.err_status, f"joint_{i}_angle_limit") or
                    getattr(s.err_status, f"communication_status_joint_{i}")):
                raise RuntimeError(f"{arm} J{i}: limit/communication fault")
            feedback = driver.get_driver_states(i)
            if feedback is None or not 0 <= now - feedback.timestamp <= 1:
                raise RuntimeError(f"{arm} J{i}: stale driver feedback")
            flags = feedback.msg.foc_status
            if not flags.driver_enable_status or any(getattr(flags, name) for name in (
                    "voltage_too_low", "motor_overheating", "driver_overcurrent",
                    "driver_overheating", "collision_status", "driver_error_status", "stall_status")):
                raise RuntimeError(f"{arm} J{i}: disabled/faulted")


def sample(bot, home, phase, report):
    state = bot.read_joints()
    host, mono = time.time(), time.monotonic()
    q = vector(state)
    stamps = [float(bot._sample_times[a]) for a in ("nero", "piper")]
    error = q - home
    report["samples"].append({"phase": phase, "monotonic": mono, "host_time": host,
                              "sample_t": float(state["t"]), "arm_t": stamps,
                              "q": q.tolist(), "error": error.tolist()})
    if any(not math.isfinite(t) or not 0 <= host - t <= 1 for t in stamps):
        raise RuntimeError("Joint feedback stale/invalid; timestamps not corrected")
    if np.max(np.abs(error)) > math.radians(STOP_DEG):
        raise RuntimeError(f"HOME error {np.degrees(np.abs(error).max()):.6f} deg > 10 deg")
    return state


def send(bot, q, phase, report):
    cmd = {"nero": q[:7].tolist(), "piper": q[7:].tolist(), "grip": 0.0, "t": time.time()}
    row = {"phase": phase, "monotonic": time.monotonic(), "host_time": cmd["t"],
           "q": q.tolist(), "returned": False}
    report["commands"].append(row)  # Failed calls can already have reached one arm.
    bot.command_joints(cmd)
    row.update(returned=True, return_monotonic=time.monotonic())


def window(bot, home, phase, seconds, repeat, report):
    start = time.monotonic()
    end = start + seconds
    deadline = start
    slot = 0
    slots = int(math.ceil(seconds * HZ))
    skipped = 0
    report["windows"][phase] = {"start_monotonic": start, "requested_s": seconds}
    try:
        while slot < slots and time.monotonic() < end:
            wait = deadline - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            if time.monotonic() >= end:
                break
            check_ready(bot)
            # Validate current feedback BEFORE sending another HOME command.
            sample(bot, home, phase + "_pre", report)
            if repeat:
                send(bot, home, phase, report)
            sample(bot, home, phase, report)
            slot += 1
            deadline = start + slot / HZ
            now = time.monotonic()
            if deadline <= now:
                next_slot = math.floor((now - start) * HZ) + 1
                missed = max(0, min(next_slot, slots) - slot)
                skipped += missed
                slot = next_slot
                deadline = start + slot / HZ  # Skip missed slots; never burst to catch up.
            deadline = min(deadline, end)
        remaining = end - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        check_ready(bot)
        sample(bot, home, phase + "_end", report)
    finally:
        report["windows"][phase].update(elapsed_s=time.monotonic() - start, skipped_slots=skipped)


def measure(bot, home, condition, report):
    check_ready(bot)
    sample(bot, home, "preflight", report)
    send(bot, home, "initial", report)
    window(bot, home, "settle", SETTLE_S, False, report)
    window(bot, home, "observe", OBSERVE_S, condition == "repeat", report)


def save(out, report, home):
    rows, cmds = report["samples"], report["commands"]
    # Statistics use one post-command/read sample per slot and the endpoint.
    obs = [r for r in rows if r["phase"] in ("observe", "observe_end")]
    report["observation_samples"] = len(obs)
    report["scheduling"] = "time.monotonic(), absolute 30 Hz deadlines; missed slots skipped, no catch-up burst"
    report["error_definition"] = "measured - HOME; sample-weighted signed mean and maximum absolute error"
    lines = [f"Condition: {report['condition']} | approach: {report['approach']}"]
    if obs:
        errors = np.array([r["error"] for r in obs])
        mean, maximum = errors.mean(0), np.abs(errors).max(0)
        report["mean_signed_error_rad"] = mean.tolist()
        report["max_abs_error_rad"] = maximum.tolist()
        report["distinct_arm_timestamps"] = [len({r["arm_t"][i] for r in obs}) for i in range(2)]
        for i in range(13):
            arm, j = ("NERO", i+1) if i < 7 else ("PiPER", i-6)
            lines.append(f"{arm} J{j}: mean {mean[i]:+.9f} rad ({np.degrees(mean[i]):+.6f} deg), "
                         f"max abs {maximum[i]:.9f} rad ({np.degrees(maximum[i]):.6f} deg)")
    times = [r["monotonic"] for r in cmds if r["phase"] == "observe"]
    if len(times) > 1:
        dt = np.diff(times)
        report["command_intervals_s"] = {"median": float(np.median(dt)), "max": float(dt.max()),
                                          "mean_hz": float(1/dt.mean())}
    np.savez(out / "samples.npz", home=home,
             phase=np.array([r["phase"] for r in rows], dtype="U24"),
             t_monotonic=np.array([r["monotonic"] for r in rows]),
             t_host=np.array([r["host_time"] for r in rows]),
             sample_t=np.array([r["sample_t"] for r in rows]),
             arm_sample_t=np.array([r["arm_t"] for r in rows]).reshape(-1, 2),
             q=np.array([r["q"] for r in rows]).reshape(-1, 13),
             error=np.array([r["error"] for r in rows]).reshape(-1, 13),
             command_phase=np.array([r["phase"] for r in cmds], dtype="U24"),
             command_t_monotonic=np.array([r["monotonic"] for r in cmds]),
             command_t_host=np.array([r["host_time"] for r in cmds]),
             command_returned=np.array([r["returned"] for r in cmds], dtype=bool),
             command_q=np.array([r["q"] for r in cmds]).reshape(-1, 13))
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (out / "report.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Status: {report['status']}; observation samples: {len(obs)}")
    if report['status'] != 'completed':
        print("Partial result, NOT a completed test:", report.get("reason", report.get("close_error", "")))
    print("\n".join(lines))
    print("Results:", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True, choices=("once", "repeat"))
    ap.add_argument("--approach", required=True, choices=("plus20", "minus20"), help="Label only; no approach motion")
    ap.add_argument("--traj", type=Path, default=replay.TRAJ)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    with np.load(a.traj, allow_pickle=False) as data:
        home = vector({"nero": data["q_cmd"][0], "piper": data["gaze_q"]})
    limits = np.concatenate(replay.joint_limits())
    check_limits(home, limits)
    if replay.TRACK_STOP_DEG != STOP_DEG:
        raise RuntimeError("Replay threshold differs from 10 deg; review required")
    print("NERO HOME:", home[:7].tolist(), "\nPiPER HOME:", home[7:].tolist())
    if not a.execute:
        print("PREVIEW ONLY: no hardware connected. Add --execute to measure.")
        return 0
    print("Replay MUST wait at B. Do not press Enter there during this test.")
    print("Both arms receive fixed HOME; NERO speed is set to 100% once, PiPER stays at 50%, matching replay.")
    print("No auto-enable. Clear workspace, prepare physical E-stop; approach pose is NOT verified.")
    if input("Type HOME to authorize measurement commands: ").strip() != "HOME":
        print("Cancelled; no connection.")
        return 0
    out = a.out or Path(__file__).resolve().parent / (
        f"home_compare_{a.approach}_{a.condition}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "incomplete", "condition": a.condition, "approach": a.approach,
              "created_at": datetime.now().astimezone().isoformat(), "samples": [], "commands": [],
              "windows": {}, "traj": str(a.traj.resolve()),
              "traj_sha256": hashlib.sha256(a.traj.read_bytes()).hexdigest(),
              "stop_deg": STOP_DEG, "speed_percent": {"nero": 100, "piper": 50}}
    bot = None
    try:
        bot = connect_reader()  # Firmware queries only; no camera/speed/enable initialization.
        bot._configs = {arm: driver.get_config() for arm, driver in bot._arms.items()}
        for arm, driver in bot._arms.items():
            speed = driver._msg_mode.move_spd_rate_ctrl
            if speed != 50:
                raise RuntimeError(f"{arm}: SDK default is {speed}, expected 50; not changed")
        # Validate before sending a mode frame, which can affect controller state.
        check_ready(bot)
        sample(bot, home, "before_speed_setup", report)
        report["speed_setup_attempted"] = True
        bot._arms["nero"].set_speed_percent(100)
        error = bot._arms["nero"]._ctx.get_comm_error()
        if error is not None:
            raise RuntimeError(f"NERO speed setup transport error: {error}")
        report["sdk_speed_cache_percent"] = {
            arm: int(driver._msg_mode.move_spd_rate_ctrl) for arm, driver in bot._arms.items()}
        if report["sdk_speed_cache_percent"] != report["speed_percent"]:
            raise RuntimeError("SDK speed cache differs from requested settings")
        measure(bot, home, a.condition, report)
        report["status"] = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="stopped", reason=f"{type(exc).__name__}: {exc}")
        if bot is not None and report["commands"]:
            try:
                state = bot.read_joints()
                now = time.time()
                if any(not 0 <= now-t <= 1 for t in bot._sample_times.values()):
                    raise RuntimeError("No fresh feedback for emergency hold")
                q = vector(state)
                check_limits(q, limits)
                send(bot, q, "failure_hold", report)
                report["failure_hold"] = "Current-position command once; NOT an E-stop"
            except Exception as stop_exc:
                report["failure_hold"] = f"Failed: {stop_exc}; physical E-stop may be needed"
    finally:
        if bot is not None:
            try:
                bot.close()
            except Exception as exc:
                report.update(status="stopped", close_error=str(exc))
        save(out, report, home)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
