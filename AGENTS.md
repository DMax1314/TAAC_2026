# AGENTS.md

Keep repository-wide decision boundaries here, task-specific workflows in
skills, and commands and implementation details in `docs/`. Add instructions
only when they change a decision and are not already expressed or enforced.

## Scope And Completion

- Use the source and tests relevant to the requested behavior. Consult docs
  when their topic matters; these are routing points, not a reading list:
  - `README.md` and `docs/getting-started.md`: setup and entrypoints.
  - `docs/architecture.md`: ownership and runtime contracts.
  - `docs/guide/contributing.md`: experiment integration.
  - `docs/guide/testing.md`: test selection and authoritative commands.
  - `docs/guide/online-training-bundle.md`: bundle format and platform execution.
  - `docs/experiments/`: experiment intent, defaults, and validation.
- Treat `docs/archive/` as historical reference, not the live runtime contract.
  Resolve discrepancies using current requirements and tests, then update the
  affected code, tests, and docs together.
- Complete the requested change and its required validation. Fix failures
  caused by the change without asking for approval at each local iteration.
  Once checks pass, expand or repeat them only for new changes, failures, or
  unresolved concerns. Report the result, verification, and remaining limits.
- Keep changes scoped. A contract change includes every affected repository
  consumer; a narrow task does not justify unrelated cleanup.
- Do not preserve backward compatibility. Remove replaced implementations,
  callers, flags, fallbacks, and tests together. Leave one working path, with
  no architectural TODOs, disabled replacements, or placeholder branches.
- Choose the simplest complete implementation with durable ownership
  boundaries. Reuse declared dependencies before adding infrastructure or a
  package; inspect their capabilities first. Avoid speculative abstractions.

## Environment

- This is a Linux project managed by `uv`. Run local Python, pytest, and console
  scripts through `uv run`; system `python`/`python3` may be absent.
- `bash run.sh train|val|eval|infer` dispatches through `uv` locally. Generated
  online bundles use platform Python and must not require `uv`, dev extras,
  `uv.lock`, or repository-only paths. Bare Python is appropriate when testing
  or documenting that platform execution path.
- Use `uv` to update dependencies and lock state. Do not hand-edit `uv.lock`.

## Architecture

Respect this dependency direction:

```text
experiments -> taac2026.api
taac2026.api -> domain/application/infrastructure
application -> domain + infrastructure
infrastructure -> domain
domain -> standard library and lightweight type dependencies
```

- `domain/` owns pure contracts, configuration, schema, requests, metrics, and
  sidecars. No CLI parsing, filesystem IO, environment probing, or framework
  execution belongs here.
- `application/` orchestrates training, evaluation, inference, packaging, and
  bootstrap use cases.
- `infrastructure/` owns IO, data access, runtime, modeling primitives,
  optimization, accelerators, bundles, and platform adapters. It must not
  select experiments or encode experiment policy.
- `experiments/` owns identity, defaults, model composition, and private model
  code. It must not copy shared data, training, checkpoint, evaluation,
  inference, or packaging workflows. Expose shared capabilities to experiments
  through `taac2026.api` only when intentionally public.
- Keep generic data and schema model-agnostic. Feature selection, grouping,
  tokenization, and interpretation belong with model composition.
- Share behavior only with a stable, experiment-independent contract and an
  actual shared consumer; similar-looking code alone is insufficient.

## Boundary Contracts

- Change producers and consumers of checkpoint sidecars, schema serialization,
  manifests, model inputs, and package metadata atomically across training,
  evaluation, inference, and bundles.
- Use `TAACBoundaryModel` subclasses for JSON, manifest, sidecar, platform, and
  plugin payloads, rejecting unknown fields. Use dataclasses or dedicated
  tensor carriers for internal immutable configuration and hot-path data.
- Keep one authoritative representation of each setting. Fail early on invalid
  schema, shape, dtype, configuration, or unsupported capability; do not drop
  arguments or synthesize silent fallbacks.

## Validation

Select commands from `docs/guide/testing.md` and test observable contracts.

- Start with affected tests. Shared behavior or cross-layer contract changes
  require the full CPU-safe gate; model or training-contract changes also
  require an end-to-end CPU smoke run when feasible.
- Accelerator changes require available GPU validation with the environment
  and command reported. CPU results do not establish CUDA, TileLang, or Triton
  behavior. State precisely what could not be verified and why.
- Packaging changes require tests and inspection of generated archive contents.
- Documentation changes require a strict site build. Edit `docs/` and
  `zensical.toml`, never generated `site/` output.
- Keep runtime warnings visible. Do not use global filters, environment flags,
  pytest settings, or blanket ignores to hide them; any narrow exception must
  be explicitly required by the task.

## Repository Safety

- Preserve user edits, including overlapping work. Do not run destructive Git
  commands or broad deletions unless explicitly requested and the exact target
  has been verified.
- Do not commit caches, `outputs/`, `.venv/`, `site/`, coverage, benchmark
  artifacts, or `__pycache__/`. Change a generated artifact's source or generator
  instead of editing the output.
