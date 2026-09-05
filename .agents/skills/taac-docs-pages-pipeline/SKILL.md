---
name: taac-docs-pages-pipeline
description: Configure or diagnose TAAC Zensical builds, navigation, and GitHub Pages deployment.
---

# TAAC Docs Pages Pipeline

Use [the local site guide](../../../docs/guide/local-site.md) for build commands,
navigation, and workflow validation. Ordinary prose edits follow that guide's
writing conventions and the strict-build requirement in `AGENTS.md`.

- For navigation, assets, or page-path issues, inspect the affected page and
  `zensical.toml`; consult the section `index.md` when its role changes.
- For CI or Pages claims, inspect `.github/workflows/deploy-docs.yml` and
  `.github/workflows/ci.yml`. Trace the relevant event, path filters, job
  conditions, and checkout ref. PR validation and main deployment differ.
- For mixed code/docs changes, account for the CI-completion path before
  interpreting a skipped direct deployment as a failure.
- Compare the deployed commit and workflow result when diagnosing stale Pages.
  Local `site/` contents alone do not establish deployment status.

Use the guide's workflow checks for YAML changes; a successful documentation
build establishes local rendering, not successful remote deployment.
