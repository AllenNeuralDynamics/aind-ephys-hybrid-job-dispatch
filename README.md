# aind-ephys-hybrid-job-dispatch

This capsule is used in the `aind-ephys-hybrid-benchmark` when using existing pre-generated hybrid data.

In this case, this caspule will substitute the `aind-ephys-job-dispatch`+`aind-ephys-hybrid-generation` capsules.

## Parameters

The capsule reads its parameters from `../params.json` (it fails if the file is missing), using the `job_dispatch` section:

| Key | Default | Description |
|-----|---------|-------------|
| `debug` | `false` | Whether to run in debug mode, clipping recordings and GT sortings |
| `debug_duration` | `30` | Duration in seconds of the clipped recording in debug mode |

An optional top-level `logging` section with `"package": "logging"` passes `logging_cfg` to `logging.basicConfig`.

Minimal example `params.json`:

```json
{
    "job_dispatch": {
        "debug": false,
        "debug_duration": 30
    }
}
```
