import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# GENERAL IMPORTS
import argparse
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

s3_path_group = parser.add_mutually_exclusive_group()
s3_path_help = "Path to s3 location with s3 data"
s3_path_group.add_argument("--s3-path", default=None, help=s3_path_help)
s3_path_group.add_argument("static_s3_path", nargs="?", default=None, help=max_num_regordings_help)

seed_group = parser.add_mutually_exclusive_group()
seed_help = "Random seed to initializa random sun-sampling"
seed_group.add_argument("--seed", default=None, help=s3_path_help)
seed_group.add_argument("static_seed", nargs="?", default=None, help=max_num_regordings_help)

parser.add_argument("--params", default=None, help="Path to the parameters file or JSON string. If given, it will override all other arguments.")



def get_list_of_recordings_from_s3(s3_path):
    """
    Returns list of zarr recordings/sortings from a pre-generated s3 path with hybrid data.

    Here we look for all "recording.zarr" recordings in the data folder with associated "sorting.zarr"
    We assume the following folder organization:
    session1 -- case1 -- recording.zarr
             |        |- sorting.zarr
             |- caseN -- recording.zarr
                      |
                      - sorting.zarr
    sessionN ...

    Parameters
    ----------
    s3_path: str
        String to s3 location of root folder for hybrid data

    Returns
    -------
    list:
        List of recording zarr paths
    list:
        List of sorting zarr paths
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    # Pass this config to your client or resource initialization
    s3_client_unsigned = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    s3_client_signed = boto3.client('s3')

    # Remove s3://
    if not s3_path.endswith("/"):
        s3_path = f"{s3_path}/"
    s3_path_ = s3_path[5:]
    bucket_name = s3_path_.split("/")[0]
    prefix_path = "/".join(s3_path_.split("/")[1:])

    for s3_client in [s3_client_unsigned, s3_client_signed]:
        try:
            # Fetch immediate items
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix_path,
                Delimiter='/'
            )
        except:
            continue

        sessions = []
        if 'CommonPrefixes' in response:
            for folder in response['CommonPrefixes']:
                sessions.append(folder['Prefix'])
        logging.info(f"Found {len(sessions)} sessions")

        recording_zarr_folders = []
        sorting_zarr_folders = []
        for session in sessions:
            logging.info(f"\t{session}")
            # Fetch immediate items
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=session,
                Delimiter='/'
            )
            if 'CommonPrefixes' in response:
                for folder in response['CommonPrefixes']:
                    prefix = f"s3://{bucket_name}/{folder['Prefix']}"
                    recording_zarr_folders.append(prefix + "recording.zarr")
                    sorting_zarr_folders.append(prefix + "sorting.zarr")
        break
    return recording_zarr_folders, sorting_zarr_folders



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
        S3_PATH = params.get("s3_path")
        SEED = params.get("seed")
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
        S3_PATH = args.static_s3_path
        if S3_PATH == "" or S3_PATH == "none":
            S3_PATH = None
        SEED = args.static_seed
        if SEED == "":
            SEED = None
        if SEED is not None:
            SEED = int(SEED)
   
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

    logging.info(f"Running job dispatcher with the following parameters:")
    logging.info(f"\tDEBUG: {DEBUG}")
    logging.info(f"\tDEBUG DURATION: {DEBUG_DURATION}")
    logging.info(f"\tMAX RECORDINGS: {MAX_RECORDINGS}")
    logging.info(f"\tS3 PATH: {S3_PATH}")
    logging.info(f"\tSEED: {SEED}")

    recording_dict = {}

    include_annotations = False
    recordings_folder = results_folder / "recordings"
    flattened_folder = results_folder / "flattened"
    recordings_folder.mkdir(parents=True, exist_ok=True)
    flattened_folder.mkdir(parents=True, exist_ok=True)

    if S3_PATH is None:
        logging.info(f"Looking for recordings in {data_folder}")
        recording_zarr_folders = []
        sorting_zarr_folders = []
        # here we assume that the data_folder includes a mapped "ecephys" folder which 
        # contains multiple sessions, each with multiple cases (with a recording.zarr and sorting.zarr)
        ecephys_session_folder = [s for s in data_folder.iterdir() if s.is_dir()][0]
        sessions = [s for s in ecephys_session_folder.iterdir() if s.is_dir()]
        logging.info(f"Number of sessions found: {len(sessions)}")
        for session in sessions:
            cases = [c for c in session.iterdir() if c.is_dir()]
            logging.info(f"\tNumber of cases found in session {session.name}: {len(cases)}")
            for case in cases:
                recording_zarr_folder = case / "recording.zarr"
                sorting_zarr_folder = case / "sorting.zarr"
                if recording_zarr_folder.is_dir() and sorting_zarr_folder.is_dir():
                    recording_zarr_folders.append(recording_zarr_folder)
                    sorting_zarr_folders.append(sorting_zarr_folder)
    else:
        logging.info(f"Looking for recordings in {S3_PATH}")
        recording_zarr_folders, sorting_zarr_folders = get_list_of_recordings_from_s3(S3_PATH)

    if len(recording_zarr_folders) == 0:
        raise FileNotFoundError(f"No hybrid sessions found. Exiting.")
    
    logging.info(f"Number of zarr recording found: {len(recording_zarr_folders)}")

    if MAX_RECORDINGS is not None and MAX_RECORDINGS < len(recording_zarr_folders):
        logging.info(f"Randomly sampling {MAX_RECORDINGS} recordings")
        rng = np.random.default_rng()
        random_indices = rng.choice(np.arange(len(recording_zarr_folders), dtype=int), size=MAX_RECORDINGS, replace=False)
        recording_zarr_folders = np.array(recording_zarr_folders)[random_indices]
        sorting_zarr_folders = np.array(sorting_zarr_folders)[random_indices]

    i = 0
    logging.info("Recording to be processed in parallel:")
    for recording_zarr_folder, sorting_zarr_folder in zip(recording_zarr_folders, sorting_zarr_folders):
        recording = si.load(recording_zarr_folder)
        gt_sorting = si.load(sorting_zarr_folder)

        session_name = Path(recording_zarr_folder).parents[1].name
        recording_name = Path(recording_zarr_folder).parent.name

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
        rec_str = f"\t{session_name} -- {recording_name}\n\t\tDuration: {duration} s - Num. channels: {recording.get_num_channels()}"
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
