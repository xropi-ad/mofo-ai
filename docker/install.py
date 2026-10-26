#!/usr/bin/env python3
"""Host setup for the mofo-ai image (WSL2 Ubuntu).

1. Docker Engine (Docker's apt repo), current user in the docker group.
2. NVIDIA Container Toolkit, registered as a docker runtime (--gpus all).
3. uv (run.py is a uv script).
4. docker/mapped-folders-template -> docker/+mapped-folders, existing files kept.

Every step is skipped when already done. Plain stdlib: runs before uv exists.
"""
import getpass
import grp
import shutil
import subprocess
import sys
from pathlib import Path

docker_dir = Path(__file__).resolve().parent
template_dir = docker_dir / "mapped-folders-template"
mapped_dir = docker_dir / "+mapped-folders"

# Files that only exist so git keeps an otherwise empty folder.
PLACEHOLDERS = {"dummy"}

notes = []


def sh(script):
    print(f"$ {script}")
    subprocess.run(["sh", "-c", script], check=True)


def os_codename():
    info = dict(
        line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line
    )
    return info.get("UBUNTU_CODENAME", info.get("VERSION_CODENAME", "")).strip('"')


def install_docker():
    if shutil.which("docker"):
        print("docker: already installed")
    else:
        print("docker: installing")
        sh("sudo apt-get update && sudo apt-get install -y ca-certificates curl")
        sh("sudo install -m 0755 -d /etc/apt/keyrings")
        sh("sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc")
        sh("sudo chmod a+r /etc/apt/keyrings/docker.asc")
        sh(
            'echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] '
            f'https://download.docker.com/linux/ubuntu {os_codename()} stable" '
            "| sudo tee /etc/apt/sources.list.d/docker.list > /dev/null"
        )
        sh(
            "sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io "
            "docker-buildx-plugin docker-compose-plugin"
        )
        sh("sudo systemctl enable --now docker")

    user = getpass.getuser()
    if user in grp.getgrnam("docker").gr_mem:
        print(f"docker group: {user} already a member")
    else:
        sh(f"sudo usermod -aG docker {user}")
        notes.append("You were added to the docker group: open a new shell (or run 'newgrp docker').")

    wsl_conf = Path("/etc/wsl.conf")
    if wsl_conf.exists() and "systemd=true" in wsl_conf.read_text().replace(" ", ""):
        print("wsl.conf: systemd already on")
    else:
        notes.append(
            "Docker needs systemd in WSL: add '[boot]' + 'systemd=true' to /etc/wsl.conf, "
            "then run 'wsl --shutdown' from Windows."
        )


def install_nvidia_toolkit():
    if shutil.which("nvidia-ctk"):
        print("nvidia-container-toolkit: already installed")
    else:
        print("nvidia-container-toolkit: installing")
        keyring = "/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        sh(
            "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey "
            f"| sudo gpg --batch --yes --dearmor -o {keyring}"
        )
        sh(
            "curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list "
            f"| sed 's#deb https://#deb [signed-by={keyring}] https://#g' "
            "| sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null"
        )
        sh("sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit")

    daemon_json = Path("/etc/docker/daemon.json")
    if daemon_json.exists() and "nvidia" in daemon_json.read_text():
        print("docker nvidia runtime: already configured")
    else:
        sh("sudo nvidia-ctk runtime configure --runtime=docker")
        sh("sudo systemctl restart docker")


def install_uv():
    if shutil.which("uv") or (Path.home() / ".local/bin/uv").exists():
        print("uv: already installed")
    else:
        sh("curl -LsSf https://astral.sh/uv/install.sh | sh")


def copy_mapped_folders():
    copied = 0
    for src in template_dir.rglob("*"):
        dst = mapped_dir / src.relative_to(template_dir)
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        elif src.name not in PLACEHOLDERS and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    print(f"+mapped-folders: {copied} new file(s) copied to {mapped_dir}")


def main():
    if not sys.platform.startswith("linux"):
        sys.exit("install.py: run this inside WSL2 (Linux)")
    install_docker()
    install_nvidia_toolkit()
    install_uv()
    copy_mapped_folders()
    for note in notes:
        print(f"NOTE: {note}")


if __name__ == "__main__":
    main()
