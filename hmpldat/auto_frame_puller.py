"""
TODO: fix my project import statements

auto-frame-puller.py - pull frames after jumping to task location in video

Ian Zurutuza
2019 Sept 9
"""

import os
import time
import re
import logging
import pathlib
import numpy as np
import pandas as pd
import random
from pick import pick
import argparse
from pathlib import Path
import cv2

# import hmpldat.utils.file_handler as file_handler
# import hmpldat.utils.video_handler as video_handler

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger("auto_frame_puller.py")

LINE_WIDTH = 180

FLAGS = None


def get_labels(row: pd.Series) -> str:
    """
    helper function - return labels from the row of the data frame 
    """

    labels = []

    for i in row.index:
        if "Bool" in i:
            if row[i] == 1:
                labels.append(i.split("V")[0])

    if labels:
        return ", ".join(labels)
    else:
        return "none"


def save(frame, labels, num):
    print(f"saving {labels}")
    fname = str(num) + ".jpg"
    fname = FLAGS.output_dir / fname
    print(fname)
    cv2.imwrite(str(fname), frame)


def main():
    """
    alignment checker main
    """

    random.seed()

    # search for rawetg files
    raw_etg_files = file_handler.search(FLAGS.data_dir, ["rawetg"], ["vmib_003"])

    frames_saved = 0

    for raw_etg in raw_etg_files:

        fname = os.path.basename(raw_etg)

        # get participant info from file name of raw_etg
        participant = re.match(r"(\D+_\d+)\(?", fname).group(1)

        # open raw etg file
        raw_etg = file_handler.open_file(raw_etg)

        if raw_etg is None:
            LOG.error("raw etg skipped cause she didn't show (read error): %s", raw_etg)
            continue

        # find video
        video = video_handler.find_video(FLAGS.data_dir, participant, raw_etg)

        if video is None:
            LOG.error("No video found for %s", participant)
            continue

        # get task information from raw_etg data
        tasks = file_handler.get_tasks(raw_etg)

        if tasks.empty:
            LOG.error("no task info found! (%s)\n", fname)
            continue
        else:
            LOG.debug(
                "tasks in file: \n\n%s\n",
                "\n".join(f"\t{task}" for task in tasks.index),
            )

        # open video
        vid = video_handler.video_open(video)

        # find corresponding files for annotated tasks in raw etg
        for task in tasks.index:
            new_task_name = file_handler.fix_task_name(task)

            files = file_handler.find_task_files(
                FLAGS.data_dir, participant, new_task_name
            )

            if files is None:
                LOG.info("participant %s task %s skipped", participant, task)
                continue
            else:
                LOG.info(
                    "\n%s", "\n".join(f"\t{file}: {files[file]}" for file in files)
                )

            LOG.debug("\n%s", tasks.loc[task])

            merged = file_handler.align(
                files, raw_etg[tasks.loc[task]["start"] : tasks.loc[task]["end"]]
            )

            vid = video_handler.safe_seek(
                vid,
                (tasks["vid_start"][task] - pd.Timedelta(seconds=FLAGS.buffer)).value
                / int(1e6),
            )

            if vid is None:
                LOG.error("set failed, skipping this task")
                continue

            data_frame_time = merged["Video Time [h:m:s:ms]"][0]
            vid_frame_time = pd.to_datetime(
                vid.get(cv2.CAP_PROP_POS_MSEC), unit="ms", origin="unix"
            )
            ret, frame = vid.read()

            frame_name = participant + "_" + task

            estimated_frame_count = video_handler.estimate_frame_count(
                vid, tasks.loc[task]["end"] - tasks.loc[task]["start"]
            )

            per_of_task = FLAGS.want / estimated_frame_count
            task_save_count = 0

            for i, row in merged.iterrows():

                if not ret:
                    LOG.error("video read error")
                    break

                # get labels associated with this frame
                msg = get_labels(row)

                # BUG: this logic no good with video faster than 30fps
                if (
                    row["Video Time [h:m:s:ms]"] is pd.NaT
                    or row["Video Time [h:m:s:ms]"] == data_frame_time
                ):
                    # stay on same frame
                    pass
                else:
                    # go to next frame
                    data_frame_time = row["Video Time [h:m:s:ms]"]
                    vid_frame_time = pd.to_datetime(
                        vid.get(cv2.CAP_PROP_POS_MSEC), unit="ms", origin="unix"
                    )
                    ret, frame = vid.read()

                    if task_save_count >= FLAGS.want:
                        break

                    if random.random() < per_of_task:
                        print(f"saving #{vid.get(cv2.CAP_PROP_POS_FRAMES)}")
                        save(frame, msg, frames_saved)
                        frames_saved += 1
                        task_save_count += 1

            cv2.destroyAllWindows()

        input()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        dest="data_dir",
        type=Path,
        default=Path("/mnt/hdd/Backup/vmi/datasets/VMIB/Data"),
    )
    parser.add_argument(
        "-o",
        dest="output_dir",
        type=Path,
        default=Path.cwd() / "created_data" / "pulled_frames" / "numbered",
    )
    parser.add_argument(
        "-want",
        type=int,
        default=10,
        help="how many images fromeach participant/task do you want",
    )
    parser.add_argument(
        # TODO: test read as timedelta
        "-buffer",
        type=float,
        default=10.0,
        help="also grab frames from x seconds before and after task",
    )

    FLAGS = parser.parse_args()

    FLAGS.output_dir.mkdir(parents=True, exist_ok=False)

    main()
