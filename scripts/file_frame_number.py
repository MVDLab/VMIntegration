"""write frame numbers and times to file

Usage: 
python .\scripts\file_frame_number.py 
    -v "Z:\HMP\Projects\VData_Integration\Exported VMIB Videos\VMIB_038_30Hz_Square_recordingtime.avi" 
    -d  "Z:\HMP\Projects\VData_Integration\Exported VMIB Videos\VMIB_038_RawData_06262020.txt"

29 June 2020
Ian Zurutuza
"""

import argparse
from pathlib import Path
from pprint import pprint
import os

import cv2
import numpy as np
import pandas as pd

import hmpldat.file.search as search

print(cv2.__version__)
print(cv2.getBuildInformation())

SEARCH_FOR = [".avi"]
"""strings to identify video"""

FLAGS = None


def print_video_properties(video):
    """ print video properties """

    frame_count_cv = int(video.get(
        cv2.CAP_PROP_FRAME_COUNT
    ))
    """Number of frames in the video file. not accurate"""

    fps = video.get(cv2.CAP_PROP_FPS)
    """Frame rate"""

    fourcc = "".join(
        [
            chr((int(video.get(cv2.CAP_PROP_FOURCC)) >> 8 * i) & 0xFF)
            for i in range(4)
        ]
    )  
    """4-character code of codec"""

    frame_size = (
        int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    # print video info
    print()
    print(f"video properties: {FLAGS.video.name}")
    print("-"*50)
    print(f"\tfps: {fps}")
    print(f"\tapprox. frame count: {frame_count_cv}")
    print(f"\tfourcc: {fourcc}")
    print(f"\tframe size: {frame_size}")
    print()

    return frame_count_cv, fps, fourcc, frame_size


def main():

    # Open a video file
    video = cv2.VideoCapture(str(FLAGS.video))
    frame_count_cv, _, _, _ = print_video_properties(video)
    
    # read first data line of rawetg file (get RecordingTime [ms] offset)
    rawetg = pd.read_csv(FLAGS.rawetg, sep="\t")
    offset = rawetg["RecordingTime [ms]"].iloc[0]

    # where to stop
    first = 0
    one_third = frame_count_cv * 1 / 3
    two_thirds = frame_count_cv * 2 / 3
    last = frame_count_cv - 20
    
    user_quit = False
    counter = 1
    data = []

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
        frame_time_ms_offset = frame_time_ms + offset

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

        reported_frame_time = np.NaN

        if any(
            [
                (counter >= x) and (counter <= x + 20)
                for x in [first, one_third, two_thirds]
            ]
        ) or (counter >= last):
            tframe = frame.copy()  
            
            # frame to draw text on
            tframe = cv2.putText(
                tframe,
                str(round(frame_time_ms_offset,1)),
                (10, 55),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=(144, 0, 255),
                thickness=1,
            )
            tframe = cv2.putText(
                tframe,
                str(int(frame_number)),
                (10, 85),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=(144, 0, 255),
                thickness=1,
            )
            tframe = cv2.putText(
                tframe,
                str(round(frame_time_ms,1)),
                (10,115),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=(144, 0, 255),
                thickness=1,
            )

            cv2.imshow("test", tframe)
            cv2.waitKey(33)

            #write images to file
            save_as = Path.cwd() / "output" / FLAGS.video.name.split(".")[0] / f"{int(frame_number):07}.jpg"
            # print(save_as)
            cv2.imwrite(str(save_as), tframe)

            # request user to input frame time on image
            if FLAGS.begaze_reported:
                while True:
                    try:
                        reported_frame_time = input(f"frame#{int(frame_number):>7}: reported_time=")
                        reported_frame_time = int(reported_frame_time)
                    except ValueError:
                        if reported_frame_time == "q":
                            user_quit = True
                            reported_frame_time = np.NaN
                        else:
                            continue
                    break
            
        data.append([
            counter,
            frame_number,
            frame_time_ms,
            frame_time_ms_offset,
            reported_frame_time,
        ])

        # increment counter
        counter += 1

    # clean up
    video.release()
    cv2.destroyAllWindows()

    # record data to file
    df = pd.DataFrame(
        data,
        columns=["my_count", "cv_frame_number", "cv_frame_time[ms]", "cv_frame_time_plus_offset", "BeGaze_frame_time"],
    )
    
    file_name = "_".join([FLAGS.video.name.split(".")[0]] + ["frame_times.xlsx"])
    df.to_excel(file_name, float_format="%.3f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-video",
        help="path to video file",
        type=Path,
        required=True
    )

    parser.add_argument(
        "-rawetg",
        help="path to rawetg file",
        type=Path,
        required=True
    )

    parser.add_argument(
        '-begaze_reported',
        action="store_true"
    )

    FLAGS, _ = parser.parse_known_args()

    if not (Path.cwd() / "output" / FLAGS.video.name.split(".")[0]).exists():
        os.makedirs(Path.cwd() / "output" / FLAGS.video.name.split(".")[0])        
    else:
        print("EXIT due to output folder already existing, please change name") 
        exit()
        
    main()
