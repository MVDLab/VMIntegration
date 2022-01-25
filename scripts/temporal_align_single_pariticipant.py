"""

Auto-alignment is dumb

Ian Zurutuza
Juneteenth, 2020
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
# import xarray as xr

import hmpldat.file.cortex
import hmpldat.file.detected
import hmpldat.file.dflow
import hmpldat.file.rawetg
import hmpldat.file.search

import hmpldat.align.temporal

TASKS_TO_ALIGN = [
    "fix_1",  # fixation
    "pp_1",  # peripheral pursuit
    "hp_1",  # horizontal pursuit
    "ap_1",  # similar to avoid & intercept, but user ball
    "int_1",  # intercept
    "avoid_1",  # avoid
]

FLAGS = None


def main():

    print(FLAGS.participant)

    # find and match corresponding dflow & cortex files
    bundled_files_df = (
        hmpldat.file.search.bundle_associated(
            FLAGS.study_folder, FLAGS.participant, probe_for_duration=False
        )
        .set_index("task_name")
        .sort_index()
    )
    # bundled_files_df = bundled_files_df.loc[TASKS_TO_ALIGN]
    print(bundled_files_df)

    # merge corresponding dflow_rd and cortex data

    # stack them up?

    # open rawetg & detected objects
    rawetg_path = "Z:\HMP\Projects\VMIB\Data\ETG\Metrics Export\VMIB_004_RawETG.txt"
    video_path = "Z:\HMP\Projects\VData_Integration\Exported VMIB Videos\Scan Path_VMIB_004-1-recording.avi"
    
   # hmpldat.file.search.files(
    #    FLAGS.study_folder, [FLAGS.participant, ".avi"]
    #)[0]
    detected_path = hmpldat.file.search.files(FLAGS.detected_files, [FLAGS.participant])[0]

    print(rawetg_path)
    print(str(video_path))
    print(detected_path)

    # get fps
    video = cv2.VideoCapture(str(video_path))
    frame_count_cv = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = video.get(cv2.CAP_PROP_FOURCC)
    frame_size = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"opencv video info: {fps}, {frame_count_cv}, {fourcc}, {frame_size}")

    writer = cv2.VideoWriter(FLAGS.participant + '_aligned.avi', int(fourcc), fps, frame_size)

    # open rawetg file
    rawetg_df = hmpldat.file.rawetg.open(rawetg_path)

    # calculate frame number for each rawetg frame
    rawetg_df["frame_number"] = hmpldat.file.rawetg.frame_number(rawetg_df, fps=fps)
    # print(rawetg_df["RecordingTime [ms]"])
    # print(rawetg_df["frame_number"])
    # print(frame_count)

    # open detected objects
    detected_df = hmpldat.file.detected.open(detected_path)
    # detected_df = hmpldat.file.detected.reformat(detected_df)
    # detected_df.columns = detected_df.columns.to_flat_index()

    # print(detected_df)

    # merge rawetg & detected objects on frame number
    # rawetg_and_detected_df = pd.merge(
    #     rawetg_df,
    #     detected_df,
    #     how="outer",
    #     on="frame_number",
    #     indicator="rawetg_and_detected_merge_source",
    # ).sort_values(by=["frame_number", "RecordingTime [ms]"])

    # rawetg_and_detected_df["rawetg_and_detected_merge_source"] = rawetg_and_detected_df[
    #     "rawetg_and_detected_merge_source"
    # ].cat.rename_categories(["rawetg_only", "detected_only", "both"])

    # # print(rawetg_and_detected_df[["frame_number", "object"]])

    # rawetg_and_detected_df.to_csv("rawetg_and_detected_merged.csv")

    # test video open "physical" frame count
    print("Video open") if video.isOpened() else print("video read error")

    while True:
        frame_number = video.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = video.read()
        
        if not ret:
            break

        gaze = rawetg_df[rawetg_df["frame_number"] == frame_number][["Point of Regard Binocular X [px]","Point of Regard Binocular Y [px]"]].dropna()
        objects = detected_df[detected_df["frame_number"] == frame_number]

        frame_with_detected_objects = frame.copy()

        # draw each object center on frame with cross and label
        for _, each_object in objects.iterrows():
            # print(each_object)
            name, score, col, row = each_object[["object", "score", "ctr_bb_col", "ctr_bb_row"]]
            name = name + ", " + str(round(score * 100,1)) + "%"
            col = int(round(col))
            row = int(round(row))
            # print(name, col, row)

            frame_with_detected_objects = cv2.drawMarker(frame_with_detected_objects, (col,row), (144,0,255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            frame_with_detected_objects = cv2.putText(frame_with_detected_objects, name, (col+5,row+15), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.5, color=(0,0,0), thickness=2)

    
        if len(gaze) != 0:        
            # draw each gaze position expecting ~2 per frame
            for _, each_gaze in gaze.iterrows():
                # print(each_gaze)
                col, row = each_gaze[["Point of Regard Binocular X [px]","Point of Regard Binocular Y [px]"]]
                col = int(round(col))
                row = int(round(row))

                with_gaze = cv2.drawMarker(frame_with_detected_objects, (col,row), (20,255,57), markerType=cv2.MARKER_DIAMOND, markerSize=20, thickness=2)

        else:
            with_gaze = frame_with_detected_objects

        writer.write(with_gaze)
        cv2.imshow("testing", with_gaze)
        k = cv2.waitKey(1)

        if k == ord('q'):
            break

    video.release()
    writer.release()
    cv2.destroyAllWindows()
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--participant",
        required=True,
        help='the participant for which you want to align data (e.g. "vmib_004")',
    )

    parser.add_argument(
        "--study_folder",
        type=Path,
        default="/home/irz0002/Documents/projects/HMP/Projects/VMIB",
    )

    parser.add_argument(
        "--detected_files",
        type=Path,
        default="/home/irz0002/Documents/projects/HMP/Projects/VData_Integration",
    )

    FLAGS, _ = parser.parse_known_args()

    main()
