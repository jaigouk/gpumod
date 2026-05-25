# OOM Protection Configs — Moved

The drop-in configs and installer have moved to the production location:

    scripts/oom-protection/

Files:

- `scripts/oom-protection/code-server-protect.conf` — code-server drop-in
- `scripts/oom-protection/oomd-gpumod.conf` — systemd-oomd drop-in
- `scripts/oom-protection/install.sh` — installer script

Verify installation with:

    gpumod doctor oom-protection
