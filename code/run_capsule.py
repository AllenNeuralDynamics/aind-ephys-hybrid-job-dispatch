import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# GENERAL IMPORTS
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
PARAMS_FILE = Path("../params.json")


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
    MAX_RECORDINGS = job_dispatch_params.get("max_recordings", None)
    if MAX_RECORDINGS is not None:
        MAX_RECORDINGS = int(MAX_RECORDINGS)
        # if -1, set to None
        if MAX_RECORDINGS == -1:
            MAX_RECORDINGS = None
    S3_PATH = job_dispatch_params.get("s3_path", None)
    if S3_PATH in ("", "none"):
        S3_PATH = None
    SEED = job_dispatch_params.get("seed", None)
    if SEED is not None:
        SEED = int(SEED)

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
        rng = np.random.default_rng(SEED)
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
    logging.info("Pipeline stage completed", extra={"event_type": "stage_complete"})


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        logging.exception("Pipeline stage failed", extra={"event_type": "stage_error"})
        raise
