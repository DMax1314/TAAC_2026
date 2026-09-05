---
name: taac-competition-environment
description: Set up or diagnose TAAC Python/CUDA environments, or change online bundle packaging and execution.
---

# TAAC Competition Environment

Choose the affected boundary before reading implementation details:

- Local dependencies or CUDA setup: [getting started](../../../docs/getting-started.md)
  and the relevant dependency/profile in `pyproject.toml`.
- Platform Python, CUDA, proxies, or package indexes:
  [online environment](../../../docs/guide/competition-online-server.md).
- Bundle contents, bootstrap, or local/platform runner differences:
  [bundle format and execution](../../../docs/guide/online-training-bundle.md).

Use these guides' source entrypoints to follow the affected behavior. Ordinary
local commands use the `uv run` convention in `AGENTS.md` without needing this
workflow.

For bundle changes, inspect the generated archive and exercise the affected
entrypoint outside the repository import context. A successful import at the
repository root can conceal missing packaged files or local dependencies.
Build, inspection, and platform simulation commands live in the bundle guide;
test selection lives in [the testing guide](../../../docs/guide/testing.md).
