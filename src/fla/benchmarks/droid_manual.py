from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def launch_manual_eval(
    *,
    left_camera_id: str,
    right_camera_id: str,
    wrist_camera_id: str,
    external_camera: str,
    remote_host: str,
    remote_port: int,
    max_timesteps: int,
    open_loop_horizon: int,
) -> dict:
    """Launch the interactive DROID evaluation script."""
    script_path = Path(__file__).resolve().parents[3] / "examples" / "droid" / "main.py"
    if not script_path.exists():
        raise FileNotFoundError(f"DROID eval script not found at {script_path}")

    cmd = [
        sys.executable,
        str(script_path),
        f"--left-camera-id={left_camera_id}",
        f"--right-camera-id={right_camera_id}",
        f"--wrist-camera-id={wrist_camera_id}",
        f"--external-camera={external_camera}",
        f"--remote-host={remote_host}",
        f"--remote-port={remote_port}",
        f"--max-timesteps={max_timesteps}",
        f"--open-loop-horizon={open_loop_horizon}",
    ]

    subprocess.run(cmd, check=True)
    return {"suite": "droid_manual", "status": "completed"}
