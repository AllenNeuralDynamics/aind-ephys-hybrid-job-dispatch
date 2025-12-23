import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# GENERAL IMPORTS
import argparse
import os
import sys
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


# Here we look for all "recording.zarr" recordings in the data folder with associated "sorting.zarr"
# We assume the following folder organization:
# session1 -- case1 -- recording.zarr
#          |        |- sorting.zarr
#          |- caseN -- recording.zarr
#                   |- sorting.zarr
# sessionN ...


# Define argument parser
parser = argparse.ArgumentParser(description="Dispatch jobs for AIND ephys pipeline")

debug_group = parser.add_mutually_exclusive_group()
debug_help = "Whether to run in DEBUG mode. Default: False"
debug_group.add_argument("--debug", action="store_true", help=debug_help)
debug_group.add_argument("static_debug", nargs="?", help=debug_help)

debug_duration_group = parser.add_mutually_exclusive_group()
debug_duration_help = (
    "Duration of clipped recording in debug mode. Only used if debug is enabled. Default: 30 seconds"
)
debug_duration_group.add_argument("--debug-duration", default=30, help=debug_duration_help)
debug_duration_group.add_argument("static_debug_duration", nargs="?", default=None, help=debug_duration_help)

max_num_regordings_group = parser.add_mutually_exclusive_group()
max_num_regordings_help = (
    "Maximum number of recordings to process. Default: process all recordings"
)
max_num_regordings_group.add_argument("--max-recordings", default=None, help=max_num_regordings_help)
max_num_regordings_group.add_argument("static_max_recordings", nargs="?", default=None, help=max_num_regordings_help)

parser.add_argument("--params", default=None, help="Path to the parameters file or JSON string. If given, it will override all other arguments.")

if __name__ == "__main__":
    args = parser.parse_args()

    # if params is given, override all other arguments
    PARAMS = args.params
    if PARAMS is not None:
        # try to parse the JSON string first to avoid file name too long error
        try:
            params = json.loads(PARAMS)
        except json.JSONDecodeError:
            if Path(PARAMS).is_file():
                with open(PARAMS, "r") as f:
                    params = json.load(f)
            else:
                raise ValueError(f"Invalid parameters: {PARAMS} is not a valid JSON string or file path")

        DEBUG = params.get("debug", False)
        DEBUG_DURATION = float(params.get("debug_duration"))
        MAX_RECORDINGS = params.get("max_recordings", None)
    else:
        DEBUG = (
            args.static_debug.lower() == "true" if args.static_debug
            else args.debug
        )
        DEBUG_DURATION = float(args.static_debug_duration or args.debug_duration)
        MAX_RECORDINGS = args.static_max_recordings or args.max_recordings
        if MAX_RECORDINGS is not None:
            MAX_RECORDINGS = int(MAX_RECORDINGS)
            # if -1, set to None
            if MAX_RECORDINGS == -1:
                MAX_RECORDINGS = None
   
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

    logging.info(f"Running job dispatcher with the following parameters:")
    logging.info(f"\tDEBUG: {DEBUG}")
    logging.info(f"\tDEBUG DURATION: {DEBUG_DURATION}")
    logging.info(f"\tMAX RECORDINGS: {MAX_RECORDINGS}")

    recording_dict = {}

    include_annotations = False
    recordings_folder = results_folder / "recordings"
    flattened_folder = results_folder / "flattened"
    recordings_folder.mkdir(parents=True, exist_ok=True)
    flattened_folder.mkdir(parents=True, exist_ok=True)

    logging.info(f"Looking for recordings in {data_folder}")
    zarr_folders = []
    for root, dirs, files in os.walk(data_folder, followlinks=True):
        for d in dirs:
            if d == "recording.zarr":
                zarr_folders.append(Path(root) / d)
    logging.info(f"Number of zarr recording folders found: {len(zarr_folders)}")
    i = 0
    logging.info("Recording to be processed in parallel:")

    if MAX_RECORDINGS is not None and MAX_RECORDINGS < len(zarr_folders):
        logging.info(f"Randomly sampling {MAX_RECORDINGS} recordings")
        rng = np.random.default_rng(seed=0)
        zarr_folders = rng.choice(zarr_folders, size=MAX_RECORDINGS, replace=False)
    
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
            duration = np.round(recording.get_total_duration(), 2)
            gt_sorting = gt_sorting.frame_slice(
                start_frame=0,
                end_frame=min(
                    int(DEBUG_DURATION * recording.sampling_frequency), recording.get_num_samples()
                )
            )
        else:
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
