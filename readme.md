# mofo-ai

Hands-on AI/ML experiments. Everything runs in one docker image
(CUDA 12.9, PyTorch, ONNX Runtime C++/Python, uv), on WSL2.

## Setup (WSL2)

1. `docker/install.py` — installs Docker Engine, NVIDIA Container Toolkit, uv,
   and creates your `docker/+mapped-folders`. Safe to re-run: done steps are skipped.
   Follow the `NOTE:` lines it prints (docker group, systemd in `/etc/wsl.conf`).
2. `docker/build-cached.sh` — builds `mofo-ai:latest`.
   `docker/build.sh` does a clean rebuild (no cache, fresh base image).
3. `docker/run.py [command]` — starts the container (GPU, WSLg GUI, the repo
   at `~/mofo-ai`, your `~/.ssh`). No command: interactive bash.
   - `/mount` = the host "root" of your cwd: the nearest folder above the cwd
     with a file named `root`. The search stops at `~` (or `/` outside home).
     No `root` file found: your cwd itself is mounted.
   - Start folder: your cwd's place under `~/mofo-ai` when you are in the repo,
     else its place under `/mount`.
4. `docker/attach.sh` — another shell in the running container.

VS Code: "Reopen in Container" uses `.devcontainer/` with the same image and mounts.

## Folder mapping

`docker/mapping.yml` lists extra mounts as `"host:container"`.

- Host path: relative to the repo root.
- Container path: absolute.

Use it for state that must survive the container (`--rm` deletes the container
on exit): tool configs in `~/.config/*`, `~/.local`, `~/.custom-bashrc`,
`~/on-path-mapped` (on `PATH`).

To add a mount:

1. Add the line to `docker/mapping.yml`.
2. Add the folder or file to `docker/mapped-folders-template/`.
3. Add the target to the `mkdir -p` in `docker/Dockerfile` (else docker creates it root-owned).
4. Mirror it in `.devcontainer/devcontainer.json`.
5. Re-run `docker/install.py`, then rebuild.

## Why `+mapped-folders` is not tracked

The mapped content is per user: your aliases, tool configs, scripts.

- `docker/mapped-folders-template/` (tracked) — the starting point.
- `docker/+mapped-folders/` (git-ignored by the `**/+*` rule) — your copy.

`install.py` copies the template into `+mapped-folders` and never overwrites
an existing file. So you can customize your container freely, without changing
the repo, and repo updates never overwrite your changes. New template files
reach you on the next `install.py` run.
