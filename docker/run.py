#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""Start mofo-ai:latest with the repo, ~/.ssh, mapping.yml mounts, GPU and WSLg GUI.

usage: run.py [command ...]     (no command: interactive bash)
"""
import os
import sys
from pathlib import Path

import yaml

IMAGE = "mofo-ai:latest"
NAME = "mofo-ai"
HOME_IN = "/home/vscode"

docker_dir = Path(__file__).resolve().parent
repo_root = docker_dir.parent


def mounts():
    vols = [f"{repo_root}:{HOME_IN}/{repo_root.name}"]
    ssh = Path.home() / ".ssh"
    if ssh.is_dir():
        vols.append(f"{ssh}:{HOME_IN}/.ssh")

    entries = yaml.safe_load((docker_dir / "mapping.yml").read_text()) or []
    missing = []
    for entry in entries:
        host, _, target = entry.rpartition(":")
        host_path = (repo_root / host).resolve()
        if not host_path.exists():
            missing.append(str(host_path))
        vols.append(f"{host_path}:{target}")
    if missing:
        sys.exit("missing host paths (run docker/install.py):\n  " + "\n  ".join(missing))
    return vols


def gui_args():
    # WSLg: X11 socket, Wayland/PulseAudio sockets under /mnt/wslg.
    args = []
    if Path("/tmp/.X11-unix").exists():
        args += ["-v", "/tmp/.X11-unix:/tmp/.X11-unix"]
    if Path("/mnt/wslg").exists():
        args += [
            "-v", "/mnt/wslg:/mnt/wslg",
            "-e", "XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir",
            "-e", "PULSE_SERVER=unix:/mnt/wslg/PulseServer",
        ]
    args += ["-e", f"DISPLAY={os.environ.get('DISPLAY', ':0')}"]
    args += ["-e", f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY', 'wayland-0')}"]
    return args


def main():
    cmd = ["docker", "run", "--rm", "--name", NAME, "--hostname", NAME,
           "--gpus", "all", "--net", "host", "--ipc", "host"]
    if sys.stdin.isatty():
        cmd.append("-it")
    for v in mounts():
        cmd += ["-v", v]
    cmd += gui_args()
    cmd += ["-w", f"{HOME_IN}/{repo_root.name}", IMAGE]
    cmd += sys.argv[1:] or ["bash"]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
