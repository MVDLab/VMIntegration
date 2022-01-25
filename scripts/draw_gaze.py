"""write frame numbers and times to file

Usage: 
python .\scripts\file_frame_number.py 
    -v "Z:\HMP\Projects\VData_Integration\Exported VMIB Videos\VMIB_038_30Hz_Square_recordingtime.avi" 
    -d  "Z:\HMP\Projects\VData_Integration\Exported VMIB Videos\VMIB_038_RawData_06262020.txt"

29 June 2020
Ian Zurutuza
"""

import argparse
from collections import deque
from itertools import count
from pathlib import Path
from pprint import pprint
import os

import cv2
import numpy as np
import pandas as pd

import hmpldat.file.rawetg

# print(cv2.__version__)
# print(cv2.getBuildInformation())

# OpenCV uses BGR ordering
# TODO (for fun): generate with bit orderings
COLORS = [
    (0, 0, 0),  # black
    (0, 0, 255),  # red
    (0, 165, 255), # orange
    (0, 255, 255),  # yellow
    (0, 255, 0),  # green
    (255, 0, 0),  # blue
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (255, 255, 255),  # "white":
]

SIZES = count(12, 8)

GAZE_MARKER_STYLE = list(zip(SIZES, COLORS))
# print(GAZE_MARKER_STYLE)
# input()

FLAGS = None


def get_video_properties(video):
    """ print video properties """

    frame_count_cv = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    """Number of frames in the video file. not accurate"""

    fps = video.get(cv2.CAP_PROP_FPS)
    """Frame rate"""

    fourcc = "".join(
        [chr((int(video.get(cv2.CAP_PROP_FOURCC)) >> 8 * i) & 0xFF) for i in range(4)]
    )
    """4-character code of codec"""

    frame_size = (
        int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    # print video info
    print()
    print(f"video properties: {FLAGS.video.name}")
    print("-" * 50)
    print(f"\tfps: {fps}")
    print(f"\tapprox. frame count: {frame_count_cv}")
    print(f"\tfourcc: {fourcc}")
    print(f"\tframe size: {frame_size}")
    print()

    return frame_count_cv, fps, fourcc, frame_size


def main():

    # Open a video file
    video = cv2.VideoCapture(str(FLAGS.video))
    frame_count_cv, fps, fourcc, frame_size = get_video_properties(video)

    # read rawetg file
    rawetg = pd.read_csv(FLAGS.rawetg, sep="\t")
    # calculate a frame number for each gaze position
    rawetg["frame_number"] = hmpldat.file.rawetg.frame_number(rawetg, fps=fps)
    # get initial RecordingTime [ms] for offset to OpenCV CAP_PROP_POS_MSEC
    offset = rawetg["RecordingTime [ms]"].iloc[0]

    # if you want to save, create VideoWriter object
    if FLAGS.save:
        name, extension = FLAGS.video.name.split(".")
        writer = cv2.VideoWriter(
            name + "_with_gaze." + extension, cv2.VideoWriter_fourcc(*fourcc), fps, frame_size
        )

    # store previous gaze positions
    gaze_history = deque(maxlen=len(GAZE_MARKER_STYLE))

    # frame number of when to save frames
    start = [0, 20]
    ts = [3162, 3169]
    fix = [20379, 20395]
    hp = [24700, 24740]
    pp = [26265, 26295]
    ap = [33625, 33635]  # angled pursuit
    # safezone = [42480, 42565]
    end = [frame_count_cv - 15, frame_count_cv]

    grab_list = [ts, fix, hp, pp, ap, end]

    # set auto playback and initial playback wait time if showing video
    wait = 0
    prev = int(1000 / fps)

    # handle user quit if showing video
    user_quit = False

    while True:
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
        """
            combines VideoCapture::grab() and VideoCapture::retrieve() in one call. 

            This is the most convenient method for reading video files or
            capturing data from decode and returns the just grabbed frame. 

            If no frames has been grabbed (camera has been disconnected, 
            or there are no more frames in video file), 
            the method returns false and the function returns empty image 
        """

        # if no frame is returned break this loop
        if not ret or user_quit:
            break

        corresponding_gaze = rawetg[rawetg["frame_number"] == frame_number][
            ["Point of Regard Binocular X [px]", "Point of Regard Binocular Y [px]"]
        ]
        # corresponding_objects = detected_df[detected_df["frame_number"] == frame_number]

        gaze_history.extendleft(corresponding_gaze.values)

        # if FLAGS.objects is not None:
        #     # draw each object center on frame with cross and label
        #     for _, each_object in objects.iterrows():
        #         # print(each_object)
        #         name, score, col, row = each_object[
        #             ["object", "score", "ctr_bb_col", "ctr_bb_row"]
        #         ]
        #         name = name + ", " + str(round(score * 100, 1)) + "%"
        #         col = int(round(col))
        #         row = int(round(row))
        #         # print(name, col, row)

        #         # draw marker then label
        #         frame_with_detected_objects = cv2.drawMarker(
        #             frame_with_detected_objects,
        #             (col, row),
        #             (144, 0, 255),
        #             markerType=cv2.MARKER_CROSS,
        #             markerSize=15,
        #             thickness=2,
        #         )
        #         frame_with_detected_objects = cv2.putText(
        #             frame_with_detected_objects,
        #             name,
        #             (col + 5, row + 15),
        #             fontFace=cv2.FONT_HERSHEY_DUPLEX,
        #             fontScale=1
        #             color=(0, 0, 0),
        #             thickness=1,
        #         )

        gaze_to_draw = list(zip(gaze_history, GAZE_MARKER_STYLE))
        gaze_to_draw.reverse()

        ### draw gaze locations on frame
        for gaze, how in gaze_to_draw:

            size, color = how

            try:
                # convert gaze string to tuple
                gaze = list(map(float, gaze))
                gaze = list(map(round, gaze))
                gaze = tuple(map(int, gaze))

                frame = cv2.drawMarker(
                    frame,
                    gaze,
                    color,
                    markerType=cv2.MARKER_DIAMOND,
                    markerSize=size,
                    thickness=2,
                )
            except:
                pass

        ### write timing information for this frame
        # opencv frame time plus offset(first rawetg "RecordingTime [ms]")
        frame = cv2.putText(
            frame,
            str(round(frame_time_ms + offset, 1)),
            (10, 55),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            color=(144, 0, 255),
            thickness=1,
        )
        # opencv frame number
        frame = cv2.putText(
            frame,
            str(int(frame_number)),
            (10, 85),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            color=(144, 0, 255),
            thickness=1,
        )
        # opencv frame time
        frame = cv2.putText(
            frame,
            str(round(frame_time_ms, 1)),
            (10, 115),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            color=(144, 0, 255),
            thickness=1,
        )

        if FLAGS.show:
            cv2.imshow(FLAGS.video.name, frame)
            k = cv2.waitKey(wait)

            if wait == 0:
                if k == ord(" "):
                    wait = prev
                elif k == ord("q"):
                    user_quit = True
            else:
                if k == ord("]"):
                    wait = int(wait / 1.1)
                    if wait > 1:
                        wait = 1
                        print("no more faster")
                elif k == ord("["):
                    try:
                        wait = int(wait * 2)
                    except OverflowError:
                        print("too much wait")
                        wait = int(1000 // fps) * 1000
                elif k == ord("'"):
                    wait = int(1000 // fps)
                elif k == ord("q"):
                    user_quit = True
                elif k == ord(" "):
                    prev = wait
                    wait = 0

        # write video to file
        if FLAGS.save:
            writer.write(frame)

            if any(
                [
                    (frame_number >= x[0]) and (frame_number <= x[1])
                    for x in grab_list
                ]
            ):
                cv2.imwrite(
                    str(FLAGS.output/ f"{int(frame_number):07d}.jpg"),
                    frame,
                )

    # clean up
    video.release()
    cv2.destroyAllWindows()

    if writer.isOpened():
        writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-video", help="path to video file", type=Path, required=True)

    parser.add_argument("-rawetg", help="path to rawetg file", type=Path, required=True)

    parser.add_argument("-objects", help="path to objects file", type=Path)

    parser.add_argument("-output", help="folder for output files", type=Path, default="output")

    parser.add_argument("-show", action="store_true")

    parser.add_argument("-save", action="store_true")

    FLAGS, _ = parser.parse_known_args()
    # print(FLAGS)

    if not FLAGS.output.exists():
        os.mkdir(FLAGS.output)

    main()
