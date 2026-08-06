# AGENTS.md

This file defines repository-wide operating constraints for coding agents in
the TAAC 2026 experiment workspace. It is not a project manual. Put workflows,
commands, design explanations, and experiment-specific knowledge in `docs/`.

Keep this file lean. Add a rule only when every agent needs it, the rule cannot
be enforced by code or tooling, and an existing rule cannot express it.

## Sources Of Truth

Read the relevant source and tests before changing code. Do not infer current
behavior from filenames, comments, or stale documentation.

Use these documents as routing points instead of duplicating their content
here:

- `README.md`: project overview, setup, and entrypoints.
- `docs/architecture.md`: layering, ownership, runtime flow, and contracts.
- `docs/guide/testing.md`: test selection, CI, GPU validation, and smoke tests.
- `docs/guide/online-training-bundle.md`: bundle format and platform behavior.
- `docs/experiments/index.md` and the relevant experiment page: experiment
  intent, defaults, and validation.

When documentation and implementation disagree, establish the intended
behavior from tests and current requirements, then update all three together.

## Engineering Principles

- Do not preserve backward compatibility. Replace obsolete contracts and
  remove their implementations, callers, flags, fallbacks, and tests in the
  same change. Do not maintain old and new paths in parallel.
- Choose the simplest implementation that fully meets the current
  requirements. Avoid speculative abstractions, configuration, registries,
  extension points, and indirection.
- Grow the system in working layers. Start with the smallest end-to-end slice,
  verify it, and add the next capability on top. Never trade a working product
  for unfinished complexity.
- Make architectural boundaries for the long term, while implementing only
  what the current change needs. Do not accept code intended to be replaced
  later.
- Keep components modular, with one clear responsibility and explicit inputs
  and outputs. Prefer composition over condition-heavy shared code.
- Use established, well-maintained libraries when they reduce total complexity
  or improve reliability. Do not reimplement common functionality without a
  concrete reason.
- Inspect existing dependencies, their documentation, and their types before
  writing new infrastructure or adding a package. Do not assume an installed
  library lacks a capability.
- Keep changes coherent and scoped. A narrow request does not justify unrelated
  cleanup; a contract change does require updating every affected in-repository
  consumer.

## Work Method

1. Inspect the working tree, relevant implementation, tests, and documentation.
2. State the current behavior, required behavior, ownership boundary, and the
   smallest end-to-end change that satisfies the requirement.
3. Implement one path. Remove the path it replaces instead of adding adapters,
   compatibility shims, or fallback behavior.
4. Validate the narrowest affected contract first, then broaden validation in
   proportion to the change's blast radius.
5. Review the final diff for duplication, dead code, accidental generated
   files, silent fallback, and documentation drift.
6. Report what changed, what was verified, and any remaining risk.

Do not leave architectural TODOs, placeholder branches, disabled replacement
code, or two competing sources of truth. If the complete coherent change is too
large for the current task, reduce the feature scope rather than landing a
temporary architecture.

## Architecture Boundaries

Respect this dependency direction:

```text
experiments -> taac2026.api
taac2026.api -> domain/application/infrastructure
application -> domain + infrastructure
infrastructure -> domain
domain -> standard library and lightweight type dependencies
```

- `domain/` owns pure contracts, validated configuration, schema, requests,
  metrics, and sidecar models. It must not perform CLI parsing, filesystem IO,
  environment probing, or framework-specific execution.
- `application/` owns train, evaluation, inference, packaging, and bootstrap
  use-case orchestration. It coordinates domain contracts and infrastructure.
- `infrastructure/` owns data access, IO, runtime, modeling primitives,
  optimization, accelerators, bundles, and platform adapters. It must not
  select experiments or encode experiment policy.
- `experiments/` owns experiment identity, defaults, model composition, and
  genuinely private model code. It must not copy shared data, training,
  checkpoint, evaluation, inference, or packaging workflows.

Experiment packages should depend on the stable `taac2026.api` facade. Add to
that facade only when a capability is intentionally public to experiments.

Keep generic data and schema code model-agnostic. It may expose canonical
values, masks, timestamps, and structural metadata; model-specific feature
selection, grouping, tokenization, and interpretation belong with model
composition. Do not add experiment-specific flags or feature semantics to a
shared data contract.

Move behavior into shared framework code only when it has a stable,
experiment-independent contract and an actual shared consumer. Similar-looking
code is not sufficient evidence for a shared abstraction.

## Contracts And Configuration

- Treat checkpoint sidecars, schema serialization, manifests, model inputs,
  and package metadata as explicit boundaries. Change producers and consumers
  atomically across training, evaluation, inference, and bundles.
- Use structured, typed configuration. Reject unknown fields at serialized or
  plugin boundaries instead of silently ignoring them.
- Use Pydantic models derived from `TAACBoundaryModel` for JSON, manifest,
  sidecar, platform, and plugin payloads. Use dataclasses or dedicated tensor
  carriers for internal immutable configuration and hot-path data.
- Keep one authoritative representation of each setting. Do not mirror the
  same option across flat dictionaries, constructor introspection, custom hooks,
  and documentation.
- Fail early on invalid schema, shape, dtype, configuration, or unsupported
  capability. Do not silently drop arguments or synthesize fallback behavior.
- Prefer `pathlib.Path`, structured parsers, and repository IO helpers over raw
  path manipulation or ad hoc string parsing.

## Dependencies And Environment

This is a Linux project managed by `uv`.

- Use the dependencies already declared in `pyproject.toml` before adding new
  ones.
- Add a dependency only when it makes the complete implementation simpler or
  more reliable, not merely shorter at one call site.
- Use `uv` to change dependency and lock state. Do not hand-edit `uv.lock`.
- Keep local development assumptions separate from platform bundle behavior.
  Platform execution must not depend on undeclared local tools or paths.

## Validation

Follow `docs/guide/testing.md` for authoritative commands and test selection.

- Test observable behavior and boundary contracts, not private implementation
  details.
- Run focused tests first. Run the full CPU-safe gate when shared behavior or a
  cross-layer contract changes.
- A model or training-contract change requires an end-to-end CPU smoke run when
  feasible.
- CPU results do not establish CUDA, TileLang, Triton, or accelerator behavior.
  Verify accelerator changes on available GPU hardware and report the exact
  environment and command.
- Packaging changes require tests plus inspection of actual generated archive
  contents.
- Documentation changes require a strict site build. Edit `docs/` and
  `zensical.toml`, never generated `site/` output.
- Do not hide runtime warnings through global filters, environment variables,
  pytest configuration, or blanket command-line ignores. Fix their cause or
  leave them visible unless the task explicitly requires a narrow exception.
- If a required validation cannot run, state why and identify the unverified
  behavior precisely.

## Repository Safety

Assume the working tree contains user work.

- Preserve changes you did not make. Work with overlapping edits and ignore
  unrelated ones.
- Do not run destructive Git commands or broad delete operations unless the
  user explicitly requests them and the exact target has been verified.
- Do not commit generated caches, `outputs/`, `.venv/`, `site/`, coverage data,
  benchmark artifacts, or `__pycache__/`.
- Do not hand-edit generated artifacts when their source or generator can be
  changed instead.
- Keep comments short and limited to decisions or constraints the code cannot
  express clearly.
