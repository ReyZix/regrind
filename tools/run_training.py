"""Run released REGRIND training inside the active Colab kernel and back it up."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import time
from zipfile import is_zipfile

REPO = Path("/content/regrind")
DRIVE = Path("/content/drive/MyDrive")
TASK = "Regrind-WujiHand-Screwdriver-v0"
EXPERIMENT = REPO / "logs/rsl_rl/wujihand_screwdriver"
CHECKPOINT = re.compile(r"model_(\d+)\.pt")
HEARTBEAT_SECONDS = 180
BACKUP_SECONDS = 1800

# Read training reward with the TensorBoard installed in the training environment.
READ_REWARDS = """
import json
import math
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

events = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0})
events.Reload()
tag = "Train/mean_reward"
scalars = events.Scalars(tag) if tag in events.Tags()["scalars"] else []
scores = {event.step: event.value for event in scalars if math.isfinite(event.value)}
print(json.dumps(scores, allow_nan=False))
"""


def copy_file(source, destination, *, growing=False):
    """Publish complete snapshots while preserving the previous copy on failure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    stat = source.stat()
    if destination.exists():
        saved = destination.stat()
        if stat.st_size == saved.st_size and stat.st_mtime_ns == saved.st_mtime_ns:
            return True
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copy2(source, temporary)
        after = source.stat()
        if growing or (after.st_size, after.st_mtime_ns) == (stat.st_size, stat.st_mtime_ns):
            temporary.replace(destination)
            return True
        return False
    finally:
        temporary.unlink(missing_ok=True)


def reward_scores(python, run):
    result = subprocess.run(
        [python, "-c", READ_REWARDS, run], text=True, capture_output=True, timeout=60
    )
    if result.returncode:
        lines = result.stderr.strip().splitlines()
        print(f"Could not rank checkpoints by reward: {lines[-1] if lines else result.returncode}", flush=True)
        return {}
    return {int(step): reward for step, reward in json.loads(result.stdout).items()}


def backup(python, run, destination, training_log):
    checkpoints = []
    for file in run.glob("model_*.pt"):
        match = CHECKPOINT.fullmatch(file.name)
        if match and is_zipfile(file):
            checkpoints.append((int(match.group(1)), file))
    checkpoints.sort()
    scores = reward_scores(python, run) if checkpoints else {}
    ranked = [(scores[step], step, file) for step, file in checkpoints if step in scores]
    best = max(ranked)[2] if ranked else None
    marker = destination / "best_checkpoint.txt"
    if best is None and marker.is_file():
        old_name = marker.read_text().strip()
        if CHECKPOINT.fullmatch(old_name) and (destination / old_name).is_file():
            best = run / old_name

    selected = {file for _, file in checkpoints[-5:]}
    if best is not None and best.is_file():
        selected.add(best)
    for file in sorted(selected):
        if not copy_file(file, destination / file.name):
            raise OSError(f"Checkpoint changed during backup: {file.name}")
    if best is not None and (destination / best.name).is_file():
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(best.name + "\n")
        temporary.replace(marker)

    keep = {file.name for file in selected}
    if best is not None:
        keep.add(best.name)
    for file in destination.glob("model_*.pt"):
        if CHECKPOINT.fullmatch(file.name) and file.name not in keep:
            file.unlink()

    for file in run.glob("events.out.tfevents.*"):
        copy_file(file, destination / file.name, growing=True)
    for file in (run / "params").glob("*.yaml"):
        copy_file(file, destination / "params" / file.name)
    copy_file(training_log, destination / training_log.name, growing=True)
    print(f"Drive backup: {len(keep)} checkpoints, best={best.name if best else 'pending reward data'}",
          flush=True)


def show_tail(log):
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 16384))
        lines = stream.read().decode(errors="replace").splitlines()[-14:]
    if lines:
        print("\n".join(lines), flush=True)


def main():
    python = REPO / ".venv/bin/python"
    if not python.is_file() or not (REPO / "scripts/rsl_rl/train.py").is_file():
        raise RuntimeError("Set up REGRIND and its Python 3.11 .venv in /content/regrind first.")
    if not DRIVE.is_dir():
        raise RuntimeError("Mount Google Drive at /content/drive before starting training.")
    num_envs = int(os.environ.get("REGRIND_NUM_ENVS", "4096"))
    run_name = "colab-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    training_log = REPO / "logs" / f"{run_name}.log"
    training_log.parent.mkdir(parents=True, exist_ok=True)
    command = [python, "-u", "scripts/rsl_rl/train.py", "--task", TASK,
               "--headless", "--num_envs", str(num_envs), "--run_name", run_name]
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", REGRIND_ROOT=str(REPO),
               REGRIND_DATA_DIR=str(REPO / "data"))

    def sync():
        runs = list(EXPERIMENT.glob(f"*_{run_name}"))
        if len(runs) == 1:
            try:
                if not DRIVE.is_dir():
                    raise OSError("Drive mount is unavailable.")
                destination = DRIVE / "regrind_runs" / runs[0].name
                backup(python, runs[0], destination, training_log)
                return True
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                print(f"Drive backup failed: {error}", flush=True)
        return False

    with training_log.open("wb") as log:
        child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"Training process {child.pid}, {num_envs} environments, log {training_log}", flush=True)
        next_backup = time.monotonic()
        try:
            # Print shortly after startup so the Mac can verify that iterations are logging.
            wait_seconds = 30
            while True:
                try:
                    status = child.wait(timeout=wait_seconds)
                    break
                except subprocess.TimeoutExpired:
                    print(f"Training process {child.pid} is active.", flush=True)
                    show_tail(training_log)
                    if time.monotonic() >= next_backup and sync():
                        next_backup = time.monotonic() + BACKUP_SECONDS
                    wait_seconds = HEARTBEAT_SECONDS
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
        finally:
            show_tail(training_log)
            backed_up = sync()

    if status:
        raise subprocess.CalledProcessError(status, command)
    if not backed_up:
        raise RuntimeError(f"Training completed but the final Drive backup failed. VM log: {training_log}")
    print("Training and final Drive backup completed.", flush=True)


if __name__ == "__main__":
    main()
