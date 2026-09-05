# TAAC Platform API Access

## Live Requests

Use an available authenticated browser session on `taiji.algo.qq.com`. Prefer
in-page `fetch` with `credentials: "include"` so credentials stay in the browser.
In browser adapters where `page.context().request.get()` fails with
`Storage.getCookies: Method not found`, in-page fetch avoids that adapter issue;
this is not a restriction on other functioning authenticated clients.

Known instance endpoints share this prefix:

```text
https://taiji.algo.qq.com/taskmanagement/api/v1/instances/external/{instance_id}/
```

- `get_ckpt`: checkpoint inventory, sizes, and publication state.
- `tf_events`: scalar histories.

Resolve the instance ID from the training/checkpoint page URL. If the route or
response differs, inspect the page's network requests; a reload with response
interception can discover the current endpoints. Limit captures to relevant
instance requests and return URLs/statuses rather than unrelated response bodies.
WeChat login probes and telemetry errors alone are not evidence of failed training.

When Playwright is available, this example inventories metric groups. Set
`metricsUrl` to the resolved `tf_events` URL and run in the authenticated page:

```javascript
const inventory = await page.evaluate(async (url) => {
  const response = await fetch(url, { credentials: "include" });
  if (!response.ok) throw new Error(`Metric request failed: HTTP ${response.status}`);
  const payload = await response.json();
  const groups = payload.data?.data;
  if (!groups || typeof groups !== "object" || Array.isArray(groups)) {
    throw new Error("Unexpected tf_events response; inspect login and response shape");
  }
  return Object.entries(groups)
    .filter(([, charts]) => Array.isArray(charts))
    .map(([group, charts]) => ({ group, charts: charts.length }));
}, metricsUrl);
```

Inspect redirects/login responses and application errors as well as HTTP
status. An authentication error or changed shape must not become an empty
metric report.

## Scalar Response Layout

The observed `tf_events` payload nests groups under `payload.data.data`.
Some group entries are metadata rather than chart arrays. Each chart uses:

| Field         | Meaning                                  |
| ------------- | ---------------------------------------- |
| `date`        | Steps shared by the series in this chart |
| `title[i]`    | Name of series `i`                       |
| `value[i][j]` | Value of series `i` at `date[j]`         |

Check these shapes before interpreting a new payload. Preserve step/value
alignment; flag missing or mismatched values. Exclude null and empty values
before numeric conversion so missing values do not become zero, then reject
non-finite numbers. Check ordering and duplicate steps before computing trends
across resumed runs.

For a metric summary, retain the group/name, observed step range, valid point
count, first/latest point, and min/max with steps. Inspect surrounding points
when those summaries cannot establish convergence or a reversal. Select metric
meaning and direction from the producing experiment, not from a universal
percentage threshold.
