#!/bin/sh
# Install Docker + NVIDIA Container Toolkit for start.sh. Idempotent:
# anything already present is skipped. Ubuntu/Debian (apt) installs Docker;
# RHEL/CentOS needs Docker preinstalled, then installs the toolkit via dnf/yum.
# Needs root or sudo.

set -e

# --- Install Docker if not present ---
ensure_docker() {
    if command -v docker >/dev/null 2>&1; then
        return 0
    fi
    if [ "$(id -u)" -ne 0 ] && ! command -v sudo >/dev/null 2>&1; then
        echo "[ERROR] Docker not found and cannot install without root/sudo."
        return 1
    fi
    _run() { [ "$(id -u)" -eq 0 ] && "$@" || sudo "$@"; }
    if command -v apt-get >/dev/null 2>&1; then
        echo "[INFO] Docker not found. Installing Docker..."
        echo "[INFO] Step 1/6: apt update and install ca-certificates, curl..."
        _run env DEBIAN_FRONTEND=noninteractive apt-get update -qq || true
        _run env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ca-certificates curl 2>/dev/null || true
        echo "[INFO] Step 2/6: adding Docker GPG key..."
        _run install -m 0755 -d /etc/apt/keyrings
        _run curl -fsSL --connect-timeout 30 --max-time 60 https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
        _run chmod a+r /etc/apt/keyrings/docker.asc
        echo "[INFO] Step 3/6: adding Docker repository..."
        _suite="noble"
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            _suite="${UBUNTU_CODENAME:-$VERSION_CODENAME}"
            [ -z "$_suite" ] && _suite="noble"
        fi
        {
            echo "Types: deb"
            echo "URIs: https://download.docker.com/linux/ubuntu"
            echo "Suites: $_suite"
            echo "Components: stable"
            echo "Signed-By: /etc/apt/keyrings/docker.asc"
        } | _run tee /etc/apt/sources.list.d/docker.sources >/dev/null
        echo "[INFO] Step 4/6: apt update..."
        _run env DEBIAN_FRONTEND=noninteractive apt-get update -qq || true
        echo "[INFO] Step 5/6: installing Docker packages..."
        _run env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin 2>/dev/null || return 1
        echo "[INFO] Step 6/6: starting Docker..."
        _run systemctl start docker 2>/dev/null || _run service docker start 2>/dev/null || true
        echo "[INFO] Docker installed and started."
    elif command -v yum >/dev/null 2>&1 || command -v dnf >/dev/null 2>&1; then
        echo "[INFO] Docker not found. On RHEL/CentOS please install Docker first (e.g. dnf install docker-ce) then re-run this script."
        return 1
    else
        echo "[ERROR] Docker not found and no supported package manager to install it."
        return 1
    fi
}

# --- Install nvidia-container-toolkit if Docker has no NVIDIA runtime ---
ensure_nvidia_container_toolkit() {
    if docker info 2>/dev/null | grep -q 'nvidia'; then
        return 0
    fi
    if [ "$(id -u)" -ne 0 ] && ! command -v sudo >/dev/null 2>&1; then
        return 1
    fi
    _run() { [ "$(id -u)" -eq 0 ] && "$@" || sudo "$@"; }
    if command -v apt-get >/dev/null 2>&1; then
        echo "[INFO] Installing nvidia-container-toolkit - required for GPU..."
        echo "[INFO] Step 1/7: apt-get update..."
        _run env DEBIAN_FRONTEND=noninteractive apt-get update -qq || true
        echo "[INFO] Step 2/7: installing ca-certificates and curl..."
        _run env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ca-certificates curl 2>/dev/null || true
        echo "[INFO] Step 3/7: adding NVIDIA repo GPG key..."
        curl -fsSL --connect-timeout 30 --max-time 60 https://nvidia.github.io/libnvidia-container/gpgkey | _run gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
        echo "[INFO] Step 4/7: adding NVIDIA repo list..."
        curl -s -L --connect-timeout 30 --max-time 60 https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            _run tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
        echo "[INFO] Step 5/7: apt-get update (NVIDIA repo)..."
        _run env DEBIAN_FRONTEND=noninteractive apt-get update -qq || true
        echo "[INFO] Step 6/7: installing nvidia-container-toolkit..."
        _run env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-container-toolkit 2>/dev/null || return 1
        echo "[INFO] Step 7/7: configuring runtime and restarting Docker (may take 1-2 min)..."
        _run nvidia-ctk runtime configure --runtime=docker 2>/dev/null || true
        _run systemctl restart docker 2>/dev/null || _run service docker restart 2>/dev/null || true
        echo "[INFO] nvidia-container-toolkit installed. Docker restarted."
    elif command -v yum >/dev/null 2>&1 || command -v dnf >/dev/null 2>&1; then
        echo "[INFO] Installing nvidia-container-toolkit - required for GPU..."
        _pkginstall() { command -v dnf >/dev/null 2>&1 && _run dnf install -y "$@" || _run yum install -y "$@"; }
        echo "[INFO] Step 1/5: installing curl..."
        _pkginstall curl
        echo "[INFO] Step 2/5: adding NVIDIA repo..."
        curl -s -L --connect-timeout 30 --max-time 60 https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | _run tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null
        echo "[INFO] Step 3/5: installing nvidia-container-toolkit..."
        _pkginstall nvidia-container-toolkit 2>/dev/null || return 1
        echo "[INFO] Step 4/5: configuring runtime and restarting Docker (may take 1-2 min)..."
        _run nvidia-ctk runtime configure --runtime=docker 2>/dev/null || true
        _run systemctl restart docker 2>/dev/null || _run service docker restart 2>/dev/null || true
        echo "[INFO] nvidia-container-toolkit installed. Docker restarted."
    else
        return 1
    fi
    sleep 2
}

if ! command -v docker >/dev/null 2>&1; then
    echo "[INFO] Docker not detected. Attempting to install Docker..."
    if ! ensure_docker; then
        echo "[ERROR] Could not install Docker. Please install Docker and re-run this script."
        exit 1
    fi
fi

DOCKER_HAS_NVIDIA=0
if docker info 2>/dev/null | grep -q 'nvidia'; then
    DOCKER_HAS_NVIDIA=1
fi

if [ "$DOCKER_HAS_NVIDIA" -eq 0 ]; then
    echo "[INFO] NVIDIA runtime not detected. Attempting to install nvidia-container-toolkit..."
    if ensure_nvidia_container_toolkit; then
        if docker info 2>/dev/null | grep -q 'nvidia'; then
            DOCKER_HAS_NVIDIA=1
            echo "[INFO] NVIDIA runtime is now available."
        fi
    fi
fi


if [ "$DOCKER_HAS_NVIDIA" -eq 0 ]; then
    echo "[ERROR] Docker has no NVIDIA runtime. Install nvidia-container-toolkit manually."
    exit 1
fi

echo "[INFO] Docker + NVIDIA runtime ready. Smoke test: docker run --rm --gpus all ubuntu nvidia-smi"
if [ "$(id -u)" -ne 0 ] && ! docker info >/dev/null 2>&1; then
    echo "[INFO] To run docker without sudo: sudo usermod -aG docker \"$USER\" then log out and back in."
fi
echo "[INFO] Next: ./start.sh"
