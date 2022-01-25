""" Methods to get video info

compare lengths
probe elapsed time

"""

from pathlib import Path

import cv2
import pandas as pd


def compare_lengths(videos: [Path], expected_length: float) -> Path:
    """Decide which video from a list is closest to expected length

    Args:
        videos (list(Path)): list of videos as Path objects
        expected_length (float): in seconds

    Returns:
        A path object to video that is closest to expected length

    """

    length_diff = {}

    for video in videos:
        capture_length = probe_elapsed_time(str(video)) / 1000
        length_diff[video] = abs(capture_length - expected_length)

    return min(length_diff, key=length_diff.get)


def probe_elapsed_time(path: Path) -> float:
    """get elapsed time from video file

    Args:
        path: pathobject to video file
    
    Returns:
        elapsed time as a timedelta

    """

    video = cv2.VideoCapture(str(path))

    # set video to the end
    # Does not work anymore? sets 
    # video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)

    elapsed_time = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
    print(elapsed_time)

    video.release()

    return pd.Timedelta(elapsed_time, unit="s")
