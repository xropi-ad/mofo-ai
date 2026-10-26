#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""Start mofo-ai:latest with the repo, ~/.ssh, mapping.yml mounts, GPU and WSLg GUI.

The host "root" of the cwd is mounted at /mount: the nearest folder, walking up
from the cwd, that holds a file named 'root'. The walk stops at ~ (or at / when
the cwd is outside home). No root found: the cwd itself is mounted at /mount.

Start folder: the cwd's place under ~/mofo-ai when the cwd is in the repo,
else its place under /mount.

usage: run.py [command ...]     (no command: interactive bash)
"""
import os
import sys
from pathlib import Path

import yaml

IMAGE = "mofo-ai:latest"
NAME = "mofo-ai"
HOME_IN = "/home/vscode"
MOUNT = "/mount"
ROOT_MARKER = "root"

docker_dir = Path(__file__).resolve().parent
repo_root = docker_dir.parent


def find_root(cwd):
    home = Path.home()
    stop = home if cwd.is_relative_to(home) else Path("/")
    for folder in [cwd, *cwd.parents]:
        if (folder / ROOT_MARKER).is_file():
            return folder
        if folder == stop:
            return None
    return None


def workdir(cwd, root):
    if cwd.is_relative_to(repo_root):
        return (Path(HOME_IN) / repo_root.name / cwd.relative_to(repo_root)).as_posix()
    if root:
        return (Path(MOUNT) / cwd.relative_to(root)).as_posix()
    return MOUNT


def mounts(mount_src):
    vols = [f"{repo_root}:{HOME_IN}/{repo_root.name}", f"{mount_src}:{MOUNT}"]
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
    cwd = Path.cwd()
    root = find_root(cwd)
    for v in mounts(root or cwd):
        cmd += ["-v", v]
    cmd += gui_args()
    cmd += ["-w", workdir(cwd, root), IMAGE]
    cmd += sys.argv[1:] or ["bash"]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
