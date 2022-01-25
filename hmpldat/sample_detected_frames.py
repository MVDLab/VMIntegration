"""Pull frames according to detector label and score

Uses multiprocessing to pull frames from multiple videos at the same time
"""

import argparse
from itertools import chain
import multiprocessing
from pathlib import Path
from pprint import pprint
import re
import sys

import cv2
import pandas as pd

import hmpldat.file.detected
import hmpldat.file.search as search
import hmpldat.utils.filter

FLAGS = None

CPUs = multiprocessing.cpu_count()

PARTICIPANT_ID = "[a-zA-Z]{4}_[0-9]{3}"


def pull_frames(x):
    """ pull requested frames from a video file

    This function may be called on a single video, but is expected to be 

    Args:
        x: tuple (Path("path/to/a/video/file"), [list of frame numbers to pull])

    Returns:
        list of frame images

    """

    vid = x[0]
    frames_to_pull = x[1]

    frame_list = []

    # open video read until requested frame and add to frame list
    cap = cv2.VideoCapture(str(vid))

    while True:

        if len(frames_to_pull) == 0:
            break

        frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()

        if frame_num % 10000 == 0:
            print(vid, " @ ", frame_num)

        if not ret:
            break

        if frame_num in frames_to_pull:
            frames_to_pull.remove(frame_num)
            frame_list.append(frame)

    print("finished video ", vid)
    return frame_list


def main():

    # record each frame that we want to label
    frame_list = []

    # find detected objects files
    detected_objects_files = search.files(
        FLAGS.data_path, ["vmib", "labeled_files", ".txt"]
    )

    # find video files
    video_files = search.files(FLAGS.video_path, ["30hz", "scan path", ".avi"])

    # define file dictionary with detected object file names
    d_file_dict = {f.name.split(".")[0]: {"detected": f} for f in detected_objects_files}
    v_file_dict = {f.name.split(".")[0]: {"video": f} for f in video_files}

    # associate files that have same name (minus file extension)
    file_dict = {}
    for k in d_file_dict.keys():
        file_dict[k] = {**d_file_dict[k], **v_file_dict[k]}

    pprint(file_dict)

    drop_keys = []
    detected_objects = []

    for k, files in file_dict.items():
        print(k)

        # open detected objects
        detected_df = hmpldat.file.detected.open(file_dict[k]["detected"])
        # reformat & sample objects detected with score > 0.995
        detected_df = hmpldat.file.detected.reformat(
            detected_df[detected_df["score"] > 0.99]
        )

        # object may not exist in the recorded frame
        try:
            frames = detected_df[FLAGS.find].dropna()
            frames = frames.reset_index()
        except KeyError:
            print(f"No {FLAGS.find} in file: {file_dict[k]['detected']}\t\t skipping")
            drop_keys.append(k)
            continue

        # remove target instances when multiple targets are detected
        if FLAGS.find == "target":
            frames = frames[frames["multiple"] == False]

        # group object of interest by appearance ("trial" number, but incorrect since participant may look away from screen)
        frames = frames.assign(
            appearance_number=(frames.frame_number.diff() > 30)
            .apply(lambda x: 1 if x else 0)
            .cumsum(),
            file_name=k,
        )

        # remove detected objects that touch the frame edge
        on_edge = hmpldat.utils.filter.at_frame_edge(frames, 5).any(axis=1)
        frames = frames[~on_edge]

        # find sector for each object and join to dataframe
        sectors, bins = hmpldat.utils.filter.sector_of_frame(
            frames, row_bins=FLAGS.row_bins, col_bins=FLAGS.col_bins
        )
        frames = frames.join(sectors)

        # append this dataframe of detected object from a video to a list
        detected_objects.append(frames)

    # join each particiapant's specified detected objects into one frame
    detected_objects = pd.concat(detected_objects)

    # group all detected instances by detected object sector (in frame)
    objects_by_sector = detected_objects.groupby(["sector_row", "sector_col"])

    sampled_frames = []

    # sample frames from each group sector (specify random_state for repeatability)
    for sec, grp in objects_by_sector:

        # sample this many frame from each sector
        n = FLAGS.total // (FLAGS.row_bins * FLAGS.col_bins)

        print(f"{sec}: {grp.size}")

        if n > len(grp):
            n = len(grp)

        sampled_frames.append(grp.sample(n))

    sampled_frames = pd.concat(sampled_frames)
    print(len(sampled_frames))

    # group sampled instances by participant_id
    sampled_frames_by_participant = sampled_frames.groupby("file_name")

    files_to_process = []

    # sort each group by frame number (so the video can be search in one direction)
    for participant, df in sampled_frames_by_participant:
        frames_to_pull = list(df["frame_number"].sort_values().values)

        # record filenames from which we sampled frames
        files_to_process.append(participant)

        file_dict[participant]["to_pull"] = frames_to_pull

    to_process = list(
        (file_dict[k]["video"], file_dict[k]["to_pull"]) for k in files_to_process
    )

    # pull frames from each video in parallel
    with multiprocessing.Pool(CPUs) as p:
        frame_list = p.map(pull_frames, to_process)

    frame_list = list(chain.from_iterable(frame_list))
    print(len(frame_list))

    i = 0
    # save each frame to new folder
    while len(frame_list) > 0:

        name = f"{FLAGS.find}_{i:04d}.jpg"
        f = frame_list.pop()
        i += 1

        cv2.imwrite(str(FLAGS.output_path / name), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default=Path("/mnt/hdd/VMI_data/vmi/Projects"))

    parser.add_argument("--video_path", default=Path("/mnt/ssd/30Hz"))

    # total number of frames to pull (evenly divided by number of files found)
    parser.add_argument(
        "-t", "--total", required=True, type=int, help="total number of frames to pull"
    )

    parser.add_argument(
        "-o", "--output_path", default=Path.cwd() / "sampled_vid_frames", type=Path
    )

    parser.add_argument(
        "-f",
        "--find",
        required=True,
        help="object to find in frame, one of ['cross', 'target', 'user', 'safezone', 'grid', ...]",
    )

    parser.add_argument(
        "-rbins",
        "--row_bins",
        default=5,
        type=int,
        help="number of vertical sectors to split recorded scene frame into for distributing sampling",
    )

    parser.add_argument(
        "-cbins",
        "--col_bins",
        default=5,
        type=int,
        help="number of horizontal sectors to split recorded scene frame into for distributing sampling",
    )

    FLAGS, _ = parser.parse_known_args()

    # don't accidentally overwrite data
    if FLAGS.output_path.exists():
        ask = f"Output folder: {str(FLAGS.output_path)} exists!\nData MAY be \033[91m\033[1mOVERWRITTEN\033[0m continue? [Y/n]"
        if input(ask).lower() == "n":
            sys.exit("Rename folder or specify output folder with '-o <PATH>' argument")
    else:
        FLAGS.output_path.mkdir()

    main()
