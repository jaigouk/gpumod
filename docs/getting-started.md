---
title: Getting Started with gpumod
description: Install gpumod, initialize the database, and manage GPU services with uv on Linux/NVIDIA systems.
---

# Getting Started

## Requirements

- [uv](https://docs.astral.sh/uv/) >= 0.4
- Python >= 3.12
- Linux with NVIDIA GPU and `nvidia-smi` in PATH

## Installation

```bash
git clone https://github.com/jaigouk/gpumod.git
cd gpumod
uv sync
```

### Make `gpumod` available globally (recommended)

Install gpumod as a [uv tool](https://docs.astral.sh/uv/guides/tools/) so
the command is always on your PATH — no need to activate a virtualenv or
prefix with `uv run`:

```bash
uv tool install -e /path/to/gpumod
```

The `-e` (editable) flag means changes to the source are picked up
immediately. uv places the binary in `~/.local/bin/` (or the uv tool bin
directory). If that's not on your PATH yet, run:

```bash
uv tool update-shell
```

After this, `gpumod` works from any directory:

```bash
gpumod status
gpumod service list
```

??? note "Alternative: use `uv run` or activate the venv"

    If you prefer not to install globally, you can still use:

    ```bash
    # Option A: prefix every command
    uv run gpumod status

    # Option B: activate the project venv
    source .venv/bin/activate
    gpumod status
    ```

## First-time Setup

Initialize the database and load built-in presets:

```bash
gpumod init
```

This creates `~/.config/gpumod/gpumod.db` with:

- GPU hardware profile (detected from `nvidia-smi`)
- Built-in service presets loaded from `presets/`
- Default mode definitions

## Deploying a Service

gpumod auto-generates systemd unit files from presets — you never write them
by hand.

### 1. Preview the generated unit file

```bash
gpumod template generate vllm-chat
```

### 2. Enable user-level systemd lingering

gpumod uses **user-level systemd** (`systemctl --user`) so no `sudo` is
needed for service management. To ensure your services start at boot and
persist after logout, enable lingering:

```bash
sudo loginctl enable-linger $USER
```

### 3. Install the unit file

This writes the unit to `~/.config/systemd/user/`:

```bash
gpumod template install vllm-chat --yes
```

After installing, reload the user daemon so systemd picks up the new unit:

```bash
systemctl --user daemon-reload
```

### 4. Start and manage services

```bash
# Start a service
gpumod service start vllm-chat

# Check service status
gpumod service status vllm-chat

# Stop a service
gpumod service stop vllm-chat
```

## Basic Workflow

```bash
# Check GPU status and VRAM usage
gpumod status

# List registered services
gpumod service list

# Simulate a mode switch before committing
gpumod simulate mode coding-mode

# Switch to a mode (starts/stops services to fit VRAM budget)
gpumod mode switch coding-mode

# Launch the interactive TUI
gpumod tui
```

See the [CLI Reference](cli.md) for all available commands.
