---
name: github-actions-logs
description: Retrieve and diagnose GitHub Actions logs from a run, job, or signed log URL.
---

# GitHub Actions Logs

Resolve the repository, run, and optional job from the supplied URL. Preserve
the requested attempt when a run has been rerun.

Use an available GitHub connector or authenticated `gh`. With `gh`, inspect
status before choosing logs (replace `OWNER/REPO`, `RUN_ID`, and `JOB_ID`):

```bash
gh run view RUN_ID --repo OWNER/REPO --json status,conclusion,headSha,attempt,jobs
gh run view RUN_ID --repo OWNER/REPO --log-failed
```

For a particular job, use `gh run view --repo OWNER/REPO --job JOB_ID --log`.
Save large logs to a temporary file and inspect the failing step with context.
Report the run/attempt, commit, failure cause, and supporting log excerpt;
distinguish retrieval failures from build/test failures.

## Access And Availability

- If `gh` is absent, use an available connector or the
  [GitHub REST API](https://docs.github.com/en/rest/actions/workflow-jobs).
  Installing system packages is not a prerequisite for this workflow.
- Logs for an active job may not be available yet. If completed-job logs are
  available, inspect those; otherwise report the available status and the
  missing evidence. Do not assume the REST endpoint provides live logs.
- Fetch a supplied signed log URL with an available HTTP tool. Treat its query
  string as a credential; do not echo it in reports. If it expires, retrieve a
  fresh download through the run/job API when access permits.
- If access is missing, report what could be inspected and the specific access
  needed. Never print credentials or silently fall back to another run.
- Log inspection does not authorize rerunning, cancelling, or deleting runs.

CLI options and log retrieval limitations are documented in
[gh run view](https://cli.github.com/manual/gh_run_view).
