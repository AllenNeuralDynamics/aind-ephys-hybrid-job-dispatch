# aind-ephys-hybrid-job-dispatch

This capsule is used in the `aind-ephys-hybrid-benchmark` when using existing pre-generated hybrid data.

In this case, this caspule will substitute the `aind-ephys-job-dispatch`+`aind-ephys-hybrid-generation` capsules.

## Parameters

The capsule reads its parameters from `../params.json` (it fails if the file is missing), using the `job_dispatch` section:

| Key | Default | Description |
|-----|---------|-------------|
| `debug` | `false` | Whether to run in debug mode, clipping recordings and GT sortings |
| `debug_duration` | `30` | Duration in seconds of the clipped recording in debug mode |
| `max_recordings` | `null` | Maximum number of recordings to process, randomly sub-sampled. `null` or `-1` means all |
| `s3_path` | `null` | S3 location of the hybrid datasets (`s3://.../<session>/<case>/{recording,sorting}.zarr`). `null`, `""` or `"none"` means use `../data` |
| `seed` | `null` | Random seed for the `max_recordings` sub-sampling |

An optional top-level `logging` section configures logging: `"package": "logging"` passes `logging_cfg` to `logging.basicConfig`, `"package": "log-schema"` sets up `log_schema` (with `pipeline_name` and `acquisition_name`).

Without `s3_path`, the data folder must contain one mapped folder with `<session>/<case>/{recording,sorting}.zarr`.

Minimal example `params.json`:

```json
{
    "job_dispatch": {
        "debug": false,
        "debug_duration": 30,
        "max_recordings": null,
        "s3_path": null,
        "seed": null
    }
}
```
