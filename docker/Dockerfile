# mofo-ai dev image: PyTorch / ML, C++ with ONNX Runtime (CUDA), dataset tools.
# Runs as user vscode (VS Code devcontainer convention), GUI through WSLg.
#
# CUDA 12.9: the host driver (576.x) supports CUDA <= 12.9, so CUDA 13 images
# and wheels would not run on it.
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
ARG ORT_VERSION=1.31.0
ARG DUCKDB_VERSION=2.0.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu129

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# --- apt packages ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        # base
        sudo ca-certificates curl wget gnupg locales less nano vim tmux htop \
        bash-completion procps file tree \
        # C++
        build-essential cmake ninja-build gdb clang clangd clang-format \
        ccache pkg-config git git-lfs openssh-client \
        # Python (system interpreter for the venv, headers for extensions)
        python3.12 python3.12-dev python3.12-venv \
        # dataset / data handling
        jq sqlite3 unzip zip p7zip-full aria2 rsync pv hdf5-tools \
        imagemagick ffmpeg \
        # GUI (WSLg: X11 + Wayland), clipboard, audio
        libgl1 libegl1 libglib2.0-0 libxkbcommon-x11-0 libxcb-cursor0 \
        libxcb-xinerama0 libxrender1 libxext6 libsm6 libdbus-1-3 \
        x11-apps x11-utils mesa-utils xclip wl-clipboard fonts-dejavu \
        tk pulseaudio-utils \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

# --- user: vscode, uid/gid = host user, passwordless sudo ---------------------
# The Ubuntu 24.04 base ships user "ubuntu" with uid 1000; remove it first.
RUN if id ubuntu >/dev/null 2>&1; then userdel -r ubuntu; fi \
    && if getent group ${USER_GID} >/dev/null; then groupmod -n ${USERNAME} "$(getent group ${USER_GID} | cut -d: -f1)"; \
       else groupadd --gid ${USER_GID} ${USERNAME}; fi \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# --- DuckDB CLI ----------------------------------------------------------------
RUN curl -fsSL "https://github.com/duckdb/duckdb/releases/download/v${DUCKDB_VERSION}/duckdb-cli-linux-amd64.tar.gz" \
        | tar -xz -C /usr/local/bin \
    && chmod 755 /usr/local/bin/duckdb

# --- ONNX Runtime C/C++ (CUDA 12 build) ----------------------------------------
RUN mkdir -p /opt/onnxruntime \
    && curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu_cuda12-${ORT_VERSION}.tgz" \
        | tar -xz -C /opt/onnxruntime --strip-components=1 \
    && echo /opt/onnxruntime/lib > /etc/ld.so.conf.d/onnxruntime.conf \
    && ldconfig
ENV ONNXRUNTIME_ROOT=/opt/onnxruntime \
    CMAKE_PREFIX_PATH=/opt/onnxruntime

# --- uv + Python venv ------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh \
    && mkdir -p /opt/venv && chown ${USER_UID}:${USER_GID} /opt/venv

# '.' last: a script in the current folder never shadows a system command.
ENV VIRTUAL_ENV=/opt/venv \
    UV_NO_CACHE=1 \
    PATH=/home/${USERNAME}/on-path-mapped:/home/${USERNAME}/.local/bin:/opt/venv/bin:/usr/local/cuda/bin:${PATH}:.

USER ${USERNAME}

RUN uv venv --python /usr/bin/python3.12 /opt/venv \
    && uv pip install torch torchvision torchaudio --index-url ${TORCH_INDEX} \
    && uv pip install \
        numpy pandas polars pyarrow scipy scikit-learn \
        matplotlib seaborn plotly tensorboard \
        jupyterlab ipywidgets ipykernel tqdm rich \
        datasets huggingface_hub kaggle dvc h5py \
        opencv-python pillow \
        onnx onnxscript \
        # 1.26 is the last PyPI onnxruntime-gpu built for CUDA 12; later ones need CUDA 13
        "onnxruntime-gpu~=1.26.0"

# --- home folder ---------------------------------------------------------------
# Pre-create the mount targets, so docker does not create them root-owned.
RUN mkdir -p ~/.ssh ~/.local/bin ~/.local/share ~/on-path-mapped \
        ~/.config/ccx ~/.config/cpx ~/.config/dbx ~/.config/edx ~/.config/gtx \
        ~/.config/hex ~/.config/searchx ~/.config/syncx ~/.config/trx ~/.config/wxl \
    && chmod 700 ~/.ssh \
    && printf '%s\n' '<c> Allen & Robert' > ~/xropi \
    && printf '%s\n' \
        '' \
        '# per-user customizations, mapped from mofo-ai/docker/+mapped-folders' \
        '[ -f "$HOME/.custom-bashrc" ] && . "$HOME/.custom-bashrc"' \
        >> ~/.bashrc

WORKDIR /home/${USERNAME}
CMD ["bash"]
