"""Search for a specific data file

You can use this as a stand alone script: search.py --help, 
but intended to be used from other python scripts

Commandline Usage (2 methods):
    * search.py /path/to/data -for vmib_007 vt -ignore unrelated_file
    * search.py /path/to/data -p vmib_007 -t vt
    
"""

import argparse
from datetime import datetime
import glob
import logging
from pathlib import Path
import re

from tqdm import tqdm
import pandas as pd

import hmpldat.file.dflow
import hmpldat.file.cortex
import hmpldat.file.rawetg
import hmpldat.file.video

LOG = logging.getLogger(__name__)
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)



DONT_CARE = [
    "zeno",
    "#",
    ".anb",
    ".forces",
    ".png",
    ".pdf",
    ".doc",
    ".xls",
    ".zip",
    ".ds_store",
]
"""list of strings that identify files I don't want to find"""

PARTICIPANT_ID_PATTERN = re.compile("([\D]{4}_[\d]{3})")


def bundle_associated(
    data_folder: Path, participant: str, task=None, probe_for_duration=True
):
    """Group files by participant and task

    Todo: search for label file

    Args:
        data_folder (Path): where to look
        participant (str): e.g. 'vmib_007'
        task (:obj: `str`, optional): task string from rawetg file
        probe_for_duration (bool): default=True, do you want to probe for duration here? 

    Returns:
        A list of associated files
        
    """

    file_identifiers = {
        "cortex": ["cortex", "trc"],
        "dflow_mc": ["mc", ".txt"],
        "dflow_rd": ["rd", ".txt"],
        "rawetg": ["rawetg", ".txt"],
        "video": ["30Hz", ".avi"],
    }

    _files = []

    # return all tasks (organized by task)
    # return a list of files that we unable to be matched.
    for file_type in file_identifiers:

        # _files[file_type] = {}

        identifiers = file_identifiers[file_type] + [participant]

        if task is not None:
            identifiers = identifiers + [task]

        # search for files
        found = files(data_folder, identifiers)

        i = 0

        # probe file duration
        for each_file in found:

            if file_type == "cortex":
                task_name = hmpldat.file.cortex.get_task_name(each_file)
                if probe_for_duration:
                    elpsd_time = hmpldat.file.cortex.probe_elapsed_time(each_file)

            elif "dflow" in file_type:
                task_name = hmpldat.file.dflow.get_task_name(each_file)
                if probe_for_duration:
                    elpsd_time = hmpldat.file.dflow.probe_elapsed_time(each_file)

            elif file_type == "rawetg":
                task_name = None
                if probe_for_duration:
                    elpsd_time = hmpldat.file.rawetg.probe_elapsed_time(each_file)

            elif file_type == "video":
                task_name = None
                if probe_for_duration:
                    elpsd_time = hmpldat.file.video.probe_elapsed_time(each_file)

            if probe_for_duration:
                _files.append((task_name, file_type, each_file, elpsd_time))
            else:
                _files.append((task_name, file_type, each_file, datetime.fromtimestamp(each_file.stat().st_mtime)))

    if probe_for_duration:
        return pd.DataFrame.from_records(
            _files, columns=["task_name", "type", "path", "duration"]
        )
    else:
        return pd.DataFrame.from_records(_files, columns=["task_name", "type", "path", "creation"])


def files(
    data_folder: Path, descriptors, dont_care=["generated_C3D", "capture_files"]
) -> []:
    """Find files based on descriptor list of strings

    Args:
        data_folder (pathlib.Path): study folder, e.g. full path to VMIB
        descriptors ([str]): string or list of strings to help us find a specific file
        dont_care (:obj: `[str]`, optional): string or list of strings to ignore

    Returns: 
        list of files that match all descriptor strings in list

    Example:
        Search for rawetg files:
        search(Path(<path/to/data>), ['rawetg'], [])
        
    """

    # LOG.info(f"searching for: {descriptors}")

    # empty list to append results to
    found = []

    # handle user input
    # turn all strings lower case and make it a list if it ain't a list
    if isinstance(descriptors, list):
        descriptors = [s.lower() for s in descriptors]
    else:
        descriptors = [descriptors.lower()]

    if dont_care is not None:
        if isinstance(dont_care, list):
            dont_care = [s.lower() for s in dont_care]
        else:
            dont_care = [dont_care.lower()]
    else:
        dont_care = []

    # add in default don't care strings (avoid these files)
    dont_care.extend(DONT_CARE)

    for a_file in glob.glob(str(data_folder / "**/*"), recursive=True):
        a_file = Path(a_file)
        if (
            all(find in str(a_file).lower() for find in descriptors)
            and not any(ignore in str(a_file).lower() for ignore in dont_care)
            and a_file.is_file()
        ):
            found.append(a_file)

    return found


def participants(data_folder: Path, participant=None) -> []:
    """return all participants or list of participants in a study

    Args:
        data_folder (pathlib.Path): study folder, e.g. full path to VMIB
        participant (:obj: `[str]`, optional): list of participants to look for (not implemented)

    Returns: 
        a pd.Series of participants
        
    """

    participants_found = []

    if participant:
        raise NotImplementedError("so you want this feature? send Ian an email")

    for a_file in tqdm(data_folder.glob("**/*")):

        # if the folder is a directory
        if a_file.is_dir():

            # if dir matches this regex
            result = PARTICIPANT_ID_PATTERN.match(a_file.name)
            if result:

                a_participant = result.group(1).lower()

                # add to list of participants
                if a_participant not in participants_found:
                    participants_found.append(a_participant)

    return pd.Series(participants_found).sort_values().reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="use this search function to find files of interest"
    )

    parser.add_argument(dest="data_folder", type=Path)
    parser.add_argument("-for", dest="descriptors", nargs="+", type=str, default=[])
    parser.add_argument(
        "-ignore", dest="dont_care", nargs="+", type=str, default=[],
    )
    parser.add_argument("-p", dest="participant", type=str)
    parser.add_argument("-t", dest="task", type=str)

    FLAGS = parser.parse_args()

    if FLAGS.participant is not None and FLAGS.task is not None:

        print(FLAGS)

        files = bundle_associated(
            FLAGS.data_folder, FLAGS.participant.lower(), FLAGS.task.lower()
        )

        for x in files:
            print(x, files[x])

    else:
        # lower case user input arguments
        FLAGS.descriptors = [x.lower() for x in FLAGS.descriptors]
        FLAGS.dont_care = [x.lower() for x in FLAGS.dont_care]

        print(FLAGS)

        for x in search(FLAGS.data_folder, FLAGS.descriptors, FLAGS.dont_care):
            print(x)
