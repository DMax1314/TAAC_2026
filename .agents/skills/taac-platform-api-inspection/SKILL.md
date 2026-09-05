---
name: taac-platform-api-inspection
description: Inspect TAAC/Taiji training checkpoints and scalar metrics through an authenticated platform session.
---

# TAAC Platform API Inspection

Use structured API data for exact checkpoint and metric values. Read
[API access and response layout](references/api.md) when fetching live data.
For an already downloaded payload, use the response-layout section directly.

Return data relevant to the request: checkpoint inventory for checkpoint
questions; first/latest values, extrema with steps, and relevant intervals for
training analysis. Summarize large payloads near the data; if a tool already
wrote a complete local file, inspect it rather than repeating the request.

Interpret metrics using the experiment and run that produced them. For current
Symbiosis diagnostics, consult [the experiment page](../../../docs/experiments/symbiosis.md).
Validate suspected overfitting against held-out AUC/LogLoss, and match a metric's
best step to an existing checkpoint before recommending a checkpoint. Missing
metrics are not zero; fixed percentage thresholds do not fit every scalar.

Use the existing authenticated session without exposing cookies, tokens, or
authorization headers. If login is required, ask the user to log in; continue
analysis of any data already available. Publishing, deleting, cancelling, or
submitting platform jobs requires the user's request for that action.
Keep downloaded platform payloads and screenshots out of commits.
