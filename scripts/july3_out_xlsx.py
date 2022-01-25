""" Clean files for Rodney

open dflow rd files for VMIB_038 
- reformat
- save as excel

open detected objects and unpacked video:
- add reported video time column to detected objects
    - function to check that steps are consistant
    - function to calculate by reported frame number (to compare)
- reformat detected objects
- save as excel

"""

import argparse
from pathlib import Path
from pprint import pprint

import cv2
import numpy as np
import pandas as pd

# import hmpldat.file.rawetg
import hmpldat.file.detected
import hmpldat.file.dflow

FLAGS = None

OBJECTS_O_INTEREST = [
    "cross",
    "target",
    "user",
    "safezone",
    "grid",
    "Ready?",
    "three",
    "two",
    "one",
    "Done",
]


def get_frame_times(video):

    print(video.get(cv2.CAP_PROP_FRAME_COUNT))
    times = []

    while(True):
        frame_number = video.get(cv2.CAP_PROP_POS_FRAMES)
        """
        CAP_PROP_POS_FRAMES 
            0-based index of the frame to be decoded/captured next. 
        """

        frame_time_ms = video.get(cv2.CAP_PROP_POS_MSEC)
        """
        CAP_PROP_POS_MSEC 
            Current position of the video file in milliseconds.
        """

        ret, frame = video.read()

        if not ret:
            break
        
        # if frame_number % 1000 == 0:
        #     print(frame_number)

        times.append([frame_number, frame_time_ms])

    return times


def count_back(frame_time, frame_number, fps):
    """
    """

    times = []
    
    for x in reversed(range(frame_number)):
        frame_time = frame_time - 1000/fps

        times.append((x, frame_time))

    times.reverse()

    return pd.DataFrame(times, columns=["frame_number", "corrected_frame_time"])


def check_video_step(s):
    """
    check that no step sizes are different than median step
    """


    steps = s.diff()

    step_std = steps.std()
    step_median = steps.median()
    
    odd_steps = steps[(steps < (step_median - 3*step_std)) | (steps > (step_median + 3*step_std))]

    print(odd_steps)

    if len(odd_steps) > 0:
        print("WOOOOW weird steps man")
    else:
        print("no prob bob")


def main():

    # pprint(vars(FLAGS))

    ### open video
    video = cv2.VideoCapture(str(FLAGS.video))
    fps = video.get(cv2.CAP_PROP_FPS)

    # calculate from video
    # record frame number, frame times
    # frame_times = get_frame_times(video)
    # frame_times_df = pd.DataFrame(frame_times, columns=pd.MultiIndex.from_product([["OpenCV"], ["frame_number", "frame_time"]]))

    
    # open objects
    detected_objects = hmpldat.file.detected.open(FLAGS.objects)

    if "frame_time" in detected_objects.columns:

        # detected_objects = detected_objects.set_index(["frame_time", "frame_number", "object"])[["col", "row", "score"]]
        print(detected_objects["frame_time"])

        # correct frame time 
        check_video_step(detected_objects["frame_time"])

        time_at_frame_10 = detected_objects[detected_objects["frame_number"] == 10]["frame_time"].iloc[0]
        corrected_times = count_back(time_at_frame_10, 10, fps)

        detected_objects["corrected_frame_time"] = detected_objects["frame_time"]
        detected_objects = pd.merge(detected_objects, corrected_times, how='outer', on="frame_number")
        detected_objects["corrected_frame_time"] = detected_objects["corrected_frame_time_y"].fillna(detected_objects["corrected_frame_time_x"])
        detected_objects = detected_objects.drop(columns=["corrected_frame_time_y", "corrected_frame_time_x"])
        print(detected_objects)

    else:
        detected_objects = detected_objects.set_index(["frame_number", "object"])[["col", "row", "score"]]

    detected_objects["visible"] = np.where(detected_objects["score"] > .98, True, False)

    # reformat 
    reformated_objects = hmpldat.file.detected.reformat(detected_objects)[OBJECTS_O_INTEREST]
    # print(reformated_objects)

    # join with frame_times
    # objects = pd.merge(frame_times_df, reformated_objects, how="left", left_on=[("OpenCV","frame_number")], right_index=True)

    objects = objects.set_index(("corrected", "frame_time"))
    objects.index.name = "corrected_frame_time[ms]"
    
    # # save to excel
    objects.to_excel("vmib_038_detected_objects.xlsx", float_format="%.2f")

    # for each dflow file 
    # regex for "_rd" in file name
    for f in FLAGS.dflow.glob("*_rd*.txt"):

        # print(f)

        # regex for task name
        # TODO: fails sometimes, handle on the inside
        task_name = hmpldat.file.dflow.get_task_name(f)
        print(task_name)

        if task_name is None:
            continue

        # skip these for now
        if any(t in task_name for t in ["vt", "duck"]):
            continue

        dflow_environment = hmpldat.file.dflow.rd_open(f)
        time = dflow_environment["Time"]

        #drop grid rotation 
        dflow_environment = dflow_environment.drop("Grid.RotX", axis=1)

        # reformat column headers "object": ["visible", "x", "y", "z"]
        reformated_dflow_environment = hmpldat.file.dflow.reformat(dflow_environment)

        # convert to mm
        reformated_dflow_environment.loc[:, reformated_dflow_environment.columns.get_level_values(1).str.contains(r"[x,y,z]")] = reformated_dflow_environment.loc[:, reformated_dflow_environment.columns.get_level_values(1).str.contains(r"[x,y,z]")] * 1000
        
        # join time column
        reformated_dflow_environment = reformated_dflow_environment.set_index(time.to_numpy() * 1000)
        reformated_dflow_environment.index.name = "Time[ms]"
        # print(reformated_dflow_environment)

        # save to excel
        reformated_dflow_environment.to_excel(f"vmib_038_{task_name}_dflow_rd.xlsx", float_format="%.2f")



if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-video',
        default="/home/irz0002/Documents/projects/HMP/Projects/VData_Integration/VMIB_038_30Hz_Square_recordingtime_unpack_bframes.avi",
        type=Path,
    )

    parser.add_argument(
        '-objects',
        default="/home/irz0002/Documents/projects/hmpldat/output_/vmib_038-1-unpack.txt",
        type=Path,
    )

    parser.add_argument(
        '-dflow',
        default="/home/irz0002/Documents/projects/HMP/Projects/VMIB/Data/DFlow/VMIB_038",
        type=Path,
    )

    FLAGS, _ =  parser.parse_known_args()

    main()