"""Test alignment


"""
from pathlib import Path
import re
import time

import pandas as pd
import cv2
from tqdm import tqdm
from pprint import pprint

import hmpldat.file.detected as detected
import hmpldat.align.temporal


DATA = Path("/mnt/hdd/VMI_data/vmi/datasets/VMIB/Data")
LABELED = Path("/mnt/hdd/VMI_data/14oct2019/output")

tqdm.pandas()


def main():

    # for each rawetg data file, find corresponding video, objects file, and all corresponding dflow+cortex
    for rawetg_file in search.search(DATA, ["rawetg"], ["vmib_003"]):

        print("rawetg: ", rawetg_file)

        participant = re.match(".*([\D]{4}_[\d]{3}).*", rawetg_file.name).group(1)
        print(participant)

        rawetg_df = rawetg.open(rawetg_file)
        if rawetg_df is None:
            print("raw etg skipped cause she didn't show (read error): %s", rawetg_df)
            continue

        objects_file = search.search(LABELED, [participant], [])
        print("objects: ", objects_file)

        if len(objects_file) == 0:
            print("No objects file found for ", participant)
            continue
        else:
            objects_file = objects_file[0]

        # match video file to objects file
        # remove any strings added to file name when creating objects file
        video_file = search.search(DATA, [objects_file.name.split(".")[0], "30hz"], [])
        print(video_file)

        if video_file is None:
            print("No video found for ", participant)
            continue
        elif len(video_file) == 1:
            video_file = video_file[0]
        else:
            raise ValueError("too many videos found")

        objects_df = detected.open(objects_file)
        capture = cv2.VideoCapture(str(video_file))

        objts_of_interest = "Ready?|three|two|one|Done|target|cross|grid|user|safezone"
        ignore_for_now = "disk|hair"

        objects_df = objects_df[
            (
                objects_df.index.get_level_values("object").str.match(objts_of_interest)
                & ~objects_df.index.get_level_values("object").str.contains(
                    ignore_for_now
                )
            )
        ]

        objects_df = detected.reformat(objects_df)

        objects_df = objects_df.assign(
            calc_video_time=detected.calc_video_time(
                objects_df, capture.get(cv2.CAP_PROP_FPS)
            )
        )

        # upsample 30Hz objects data to 60Hz rawetg data (not interpolated)
        combined = hmpldat.align.temporal.objects_to_rawetg(objects_df, rawetg_df)

        combined = combined.assign(expected_focus=detected.expected_focus)

        print(combined)
        print(combined.columns)

        combined.to_csv("check_me_out.csv")

        # print(objects_df[''])

        # print(rawetg_df)
        # print(video.probe_elapsed_time(video_file))

        break


if __name__ == "__main__":
    main()
