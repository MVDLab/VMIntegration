"""Methods to draw info on image frame

main function has initial code for displaying objects drawn on video

Todo: separate main, make sure I didn't break

"""
import time
import argparse
from pathlib import Path
import re
import logging
from pprint import pprint

import numpy as np
import pandas as pd
import cv2


FLAGS = None
LOG = logging.getLogger(__name__)

EXPECTED_FRAME_TIME = 1000 / 30.00003000003


def draw_gaze(frame, center):
    """draws a 3 degree ring on the image at the gaze location based on ratio of pixels to horizontal camera field of view

    .. math:: 
    
        \\frac{\\text{camera horizontal field of view}}{\\text{central vision focus}} = \\frac{60^\circ}{3^\circ} = \\frac{960px}{48px} 

    again using radius:  

    .. math:: 

        \\frac{30^\circ}{1.5^\circ} = \\frac{\pi / 6}{\pi / 120} = \\frac{480px}{24px}

    Args:
        frame: an image frame
        center: tuple of gaze location

    Returns:
        frame: an image frame with drawn gaze location
        
    """

    frame = cv2.circle(
        frame,
        center=center,
        radius=24,  # 3 degree cone around Binocular Point of Reference
        color=(0, 255, 255),
        thickness=4,
        lineType=cv2.LINE_AA,
    )

    return frame


def draw_boxes(frame, boxes):
    """Show location of detected objects on an image frame

    Args:
        frame: image frame
        boxes: dataframe of detected objects for associated frame

    Returns:
        video frame with boxes drawn on top

    """

    for _, box in boxes.iterrows():

        # only draw boxes with a score higher than min_score
        if box.score > FLAGS.min_score:
            object_name = (
                " ".join([box.object, str(box.score * 100).split(".")[0]]) + "%"
            )

            # draw box
            frame = cv2.rectangle(
                frame,
                pt1=(round(box.left), round(box.top)),
                pt2=(round(box.right), round(box.bottom)),
                color=(255, 128, 180),
                thickness=4,
                lineType=cv2.LINE_AA,
            )

            # put text with backing
            (text_w, text_h) = cv2.getTextSize(object_name, 5, 0.5, 1)[0]
            frame = cv2.rectangle(
                frame,
                pt1=(round(box.left), round(box.top) - text_h),
                pt2=(round(box.left) + text_w, round(box.top)),
                color=(255, 128, 180),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            frame = cv2.putText(
                frame,
                text=object_name.upper(),
                org=(round(box.left), round(box.top)),
                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                fontScale=0.5,
                color=(0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    return frame


def main():

    for rawetg_file in search.search(FLAGS.data_dir, ["rawetg"], ["vmib_003"]):

        participant = re.match(".*([\D]{4}_[\d]{3}).*", rawetg_file).group(1)
        print(participant)

        rawetg = rawetg.open(rawetg_file)
        if rawetg is None:
            LOG.error("raw etg skipped cause she didn't show (read error): %s", rawetg)
            continue

        video_files = search(path, [participant, ".avi", "30Hz"], ["rawetg", ".rmf"])

        if len(video_files) == 0:
            LOG.error("No video found for %s", participant)
            continue
        elif len(video_files) > 1:
            LOG.error(
                "multiple video files found, include catch from old/video_handler.find_video"
            )

        # TODO: add 'labeled' to file names of label txt files
        # then move folder to data dir
        label_file = search.search(
            str(FLAGS.labels_dir), [video_file.name.split(".")[0]], []
        )
        if len(label_file) == 0:
            LOG.error("No label file found for %s", participant)
            continue
        else:
            label_file = label_file[0]

        labels = objects.open(label_file)

        # find label file starts Ready? or three or two or one
        starts = labels[
            labels.object.str.match("Ready?|three|two|one") & labels.score.gt(0.99)
        ]

        potential_starts = starts.frame_number.diff().gt(35).cumsum()

        # pprint([potential_starts.where(potential_starts.values == x).dropna() for x in potential_starts.unique()])
        # print(starts.to_string())

        task_count = 0

        for s in potential_starts.unique():

            a_potential_start_event = potential_starts.where(
                potential_starts.values == s
            ).dropna()

            # the length of our start "event" is at least a second long (should be about 4 seconds, but particpant may not be looking at "Ready?" when appears")
            if a_potential_start_event.size > 30:

                task_count = task_count + 1

                print(labels.iloc[a_potential_start_event.index[0]].frame_number)

                vid = cv2.VideoCapture(str(video_file))
                vid = vid.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    labels.iloc[a_potential_start_event.index[0]].frame_number - 5,
                )
                # vid = video_handler.safe_seek(vid, labels.iloc[a_potential_start_event.index[0]].frame_number - 5, how="frame")

                # print(vid.get(cv2.CAP_PROP_FOURCC))
                # print(vid.get(cv2.CAP_PROP_FPS))
                # print(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                # print(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if input() == "s":
                    continue

                writer = cv2.VideoWriter(
                    "_".join([participant, str(task_count)]) + ".avi",
                    fourcc=int(vid.get(cv2.CAP_PROP_FOURCC)),
                    fps=2 * vid.get(cv2.CAP_PROP_FPS),
                    frameSize=(
                        int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    ),
                )

                print(writer.isOpened())

                new_vid_time = 0
                count = 0
                while True:

                    ret, frame = vid.read()
                    if not ret:
                        break

                    frame_time = pd.to_timedelta(
                        vid.get(cv2.CAP_PROP_POS_MSEC), unit="ms"
                    )
                    frame_number = vid.get(cv2.CAP_PROP_POS_FRAMES)

                    # frames with close video times (within 16 ms right now)
                    corresponding_rawetg = rawetg[
                        abs(
                            frame_time.total_seconds()
                            - rawetg["Video Time [h:m:s:ms]"].dt.total_seconds()
                        )
                        < 0.020
                    ]

                    # draw_bounding boxes
                    bframe = draw_boxes(
                        frame, labels[labels.frame_number.eq(frame_number)]
                    )

                    k = None  # waitkey

                    print(len(corresponding_rawetg))

                    # for each gaze draw display frame
                    for _, gaze in corresponding_rawetg.iterrows():

                        print("video:  ", frame_time)
                        print("rawetg: ", gaze["Video Time [h:m:s:ms]"])

                        print("diff: ", abs(frame_time - gaze["Video Time [h:m:s:ms]"]))
                        # print(gaze.name)

                        if gaze.in_range:

                            gaze_center = (
                                gaze["Point of Regard Binocular X [px]"],
                                gaze["Point of Regard Binocular Y [px]"],
                            )
                            # TODO insert check to see if gaze recording time has repeated
                            # log event for now -> if it is an issue (happens alot fix)

                            if all(not np.isnan(val) for val in gaze_center):
                                gframe = draw_gaze(
                                    bframe.copy(), tuple(map(round, gaze_center))
                                )
                            else:
                                gframe = bframe.copy()
                        else:
                            print("gaze out of range")
                            gframe = bframe.copy()

                        cv2.imshow(participant, gframe)

                        writer.write(gframe)
                        # cv2.imwrite("tmp/frame%d.jpg" % count, gframe)
                        # count = count + 1

                        k = cv2.waitKey(30)

                        if k == ord(" "):
                            while cv2.waitKey(500) == ord(" "):
                                ## hang out
                                pass

                        # Press 'q' to quit
                        if k == ord("q"):
                            break

                    # Press 'q' to quit
                    if k == ord("q"):
                        break

                vid.release()
                cv2.destroyAllWindows()
                if writer is not None:
                    writer.release()

            else:
                print("I don't think this is a start")

        input()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        dest="data_dir",
        type=Path,
        default=Path("/home/irz0002/Projects/vmi/data/VMIB/Data"),
    )
    parser.add_argument(
        "-labels", "-label_files", dest="labels_dir", type=Path,
    )
    parser.add_argument(
        "-show", action="store_true",
    )
    parser.add_argument(
        "-save", action="store_true",
    )

    parser.add_argument(
        "-min_score", type=float, default=0.6,
    )

    FLAGS = parser.parse_args()

    main()
