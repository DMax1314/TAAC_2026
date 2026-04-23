---
name: vscode-devcontainer-cleanup
description: 'Clean stale VS Code Remote Containers helpers, X11 sockets, old VS Code server versions, server logs, and extension caches inside a dev container. Use when Dev Container attach is slow, /tmp/.X11-unix contains many X displays, /root/.vscode-server has stale versions, or you want safe VS Code server housekeeping.'
argument-hint: 'inspect or cleanup'
user-invocable: true
---

# VS Code Dev Container Cleanup

## When to Use

- Dev Container attach is slow and the logs show long X11 probing.
- `/tmp/.X11-unix` contains many `X0`, `X1`, ... entries.
- `/root/.vscode-server/bin` has old VS Code server versions.
- `/root/.vscode-server/data/logs` has many old log directories.
- `/root/.vscode-server/extensionsCache` is taking unnecessary space.

## What This Skill Does

This skill performs safe housekeeping for VS Code inside a dev container:

- Kills stale `vscode-remote-containers-server-*.js` helper processes from builds that are no longer actively attached.
- Removes stale X11 socket entries while preserving active sockets and the current `DISPLAY`.
- Deletes old VS Code server version directories that have no active non-helper processes.
- Removes inactive VS Code log directories while preserving logs referenced by active processes.
- Clears the extension download cache without removing installed extensions.

## Safety Model

- Active VS Code builds are detected from live non-helper processes under `/root/.vscode-server/bin/<build>/`.
- Old Remote Containers helper processes are only terminated when their build is not currently in use.
- X11 cleanup preserves sockets still present in `/proc/net/unix` and the current display socket.
- Log cleanup preserves any log directory referenced by active VS Code processes.
- Extension cleanup only clears `/root/.vscode-server/extensionsCache`.

## Procedure

1. Inspect the current state first from the repository root:

```bash
./.agents/skills/vscode-devcontainer-cleanup/scripts/cleanup-vscode-devcontainer.sh inspect
```

2. Run the cleanup when the inspection looks correct:

```bash
./.agents/skills/vscode-devcontainer-cleanup/scripts/cleanup-vscode-devcontainer.sh cleanup
```

3. Review the summary at the end for:

- Active builds kept
- Stale helper processes removed
- X11 entries removed
- Old server versions removed
- Old log directories removed
- Extension cache size after cleanup

## Notes

- Run this skill inside the dev container with permissions to manage `/root/.vscode-server` and `/tmp/.X11-unix`.
- If multiple VS Code clients are actively attached, the script keeps all builds that still have non-helper processes.
- The main workflow is implemented in [cleanup-vscode-devcontainer.sh](./scripts/cleanup-vscode-devcontainer.sh).