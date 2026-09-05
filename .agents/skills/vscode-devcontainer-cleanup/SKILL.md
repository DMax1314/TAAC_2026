---
name: vscode-devcontainer-cleanup
description: Diagnose slow VS Code Dev Container attachment or clean stale server processes, logs, and caches inside the container.
---

# VS Code Dev Container Cleanup

Use the [bundled script](scripts/cleanup-vscode-devcontainer.sh). Commands below
run from the repository root inside the dev container.

Inspect before cleanup:

```bash
./.agents/skills/vscode-devcontainer-cleanup/scripts/cleanup-vscode-devcontainer.sh inspect
```

When cleanup is requested and inspection confirms the intended targets:

```bash
./.agents/skills/vscode-devcontainer-cleanup/scripts/cleanup-vscode-devcontainer.sh cleanup
```

The script targets `/root/.vscode-server` and `/tmp/.X11-unix`. Preserve active
sessions using its process/socket checks. If the layout is not recognized,
inspect the script and running processes before cleanup; do not replace the
checks with ad hoc deletion commands.
