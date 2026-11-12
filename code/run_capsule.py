import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# GENERAL IMPORTS
import sys
from glob import glob
import numpy as np
import warnings
from pathlib import Path
import json
import logging


# SPIKEINTERFACE
import spikeinterface as si

from spikeinterface.core.core_tools import SIJsonEncoder


data_folder = Path("../data")
results_folder = Path("../results")
PARAMS_FILE = Path("../params.json")


# Here we look for all "recording.zarr" recordings in the data folder with associated "sorting.zarr"
# We assume the following folder organization:
# session1 -- case1 -- recording.zarr
#          |        |- sorting.zarr
#          |- caseN -- recording.zarr
#                   |- sorting.zarr
# sessionN ...


def run() -> None:
    if not PARAMS_FILE.is_file():
        raise ValueError(f"Params file not found: {PARAMS_FILE}")
    with open(PARAMS_FILE, "r") as f:
        full_params = json.load(f)
    job_dispatch_params = full_params["job_dispatch"]
    LOGGING = full_params.get("logging", None)

    # setup logging before any other logging call
    if LOGGING is None:
        logging.basicConfig(level="INFO", stream=sys.stdout, format="%(message)s")
    else:
        if LOGGING["package"] == "logging":
            logging_cfg = LOGGING.get("logging_cfg", {})
            logging.basicConfig(stream=sys.stdout, **logging_cfg)
        elif LOGGING["package"] == "log-schema":
            import log_schema

            pipeline_name = LOGGING.get("pipeline_name", "AIND Ephys Hybrid Benchmark")
            acquisition_name = LOGGING.get("acquisition_name", "unknown")

            config = LOGGING.get("logging_cfg")
            if config is not None and len(config) == 0:
                config = None
            log_schema.setup_logging(
                config=config,
                model={
                    "pipeline_name": pipeline_name,
                    "acquisition_name": acquisition_name,
                    "process_name": "Hybrid Job Dispatch"
                }
            )

    logging.info("Begin processing...", extra={"event_type": "stage_start"})


    DEBUG = job_dispatch_params.get("debug", False)
    DEBUG_DURATION = float(job_dispatch_params.get("debug_duration", 30))


    logging.info(f"Running job dispatcher with the following parameters:")
    logging.info(f"\tDEBUG: {DEBUG}")
    logging.info(f"\tDEBUG DURATION: {DEBUG_DURATION}")

    recording_dict = {}

    include_annotations = False
    recordings_folder = results_folder / "recordings"
    flattened_folder = results_folder / "flattened"
    recordings_folder.mkdir(parents=True, exist_ok=True)
    flattened_folder.mkdir(parents=True, exist_ok=True)

    zarr_folders = [
        Path(p) for p in list(glob(str(data_folder / "**/*.zarr"), recursive=True)) if "recording" in Path(p).name
    ]
    logging.info(f"Number of zarr recording folders found: {len(zarr_folders)}")
    i = 0
    logging.info("Recording to be processed in parallel:")
    for recording_zarr_folder in zarr_folders:
        sorting_zarr_folder = recording_zarr_folder.parent / "sorting.zarr"
        if not sorting_zarr_folder.is_dir():
            continue
        session_name = recording_zarr_folder.parents[1].name
        recording_name = recording_zarr_folder.parent.name
        recording = si.load(recording_zarr_folder)
        gt_sorting = si.load(sorting_zarr_folder)

        if DEBUG:
            recording = recording.frame_slice(
                start_frame=0,
                end_frame=min(
                    int(DEBUG_DURATION * recording.sampling_frequency), recording.get_num_samples()
                )
            )
            gt_sorting = gt_sorting.frame_slice(
                start_frame=0,
                end_frame=min(
                    int(DEBUG_DURATION * recording.sampling_frequency), recording.get_num_samples()
                )
            )
        duration = np.round(recording.get_total_duration(), 2)

        job_dict = dict(
            session_name=session_name,
            recording_name=recording_name,
            recording_dict=recording.to_dict(recursive=True, include_annotations=include_annotations, relative_to=data_folder),
            duration=duration,
            input_folder=session_name,
            debug=DEBUG,
        )
        rec_str = f"\t{recording_name}\n\t\tDuration: {duration} s - Num. channels: {recording.get_num_channels()}"
        logging.info(rec_str)

        # we use double _ here for easy parsing
        recording_name = f"{job_dict['session_name']}__{job_dict['recording_name']}"
        job_dict["recording_name"] = recording_name
        with open(recordings_folder / f"job_{recording_name}.json", "w") as f:
            json.dump(job_dict, f, indent=4, cls=SIJsonEncoder)
        with open(flattened_folder / f"job_{recording_name}.json", "w") as f:
            json.dump(job_dict, f, indent=4, cls=SIJsonEncoder)
        # save sorting
        gt_sorting.dump_to_json(flattened_folder / f"gt_{recording_name}.json", relative_to=data_folder)
        i += 1
    logging.info(f"Generated {i} hybrid config files")
    logging.info("Pipeline stage completed", extra={"event_type": "stage_complete"})


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        logging.exception("Pipeline stage failed", extra={"event_type": "stage_error"})
        raise
