"""This script demonstrates how to use alignment methods 

TODO: update import statements
TODO: Test that this still works

Ty and Rhythm this is the script I am talking about:
    Put a comment that starts with "WTF!?!" anywhere I need to explain better

Ian Zurutuza
22 December 2019
"""

# import builtin python libs
import argparse
import logging
from pathlib import Path
import re
import sys
from pprint import pprint
import logging

# import 3rd party
import numpy as np
import pandas as pd
from tqdm import tqdm  # progress bars
import cv2  # video processing (opencv)

# import my python "utility" files
import hmpldat.utils as utils


LOG = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

FLAGS = None


def main():

    # find all participants
    participants_found = utils.search.participants(
        FLAGS.study_folder
    )  # returns a pd.Series
    # print(participants_found)
    # input()

    # Consider replacing `participants_found` with a pd.Series or numpy of participants e.g. pd.Series(['vmib_007', 'vmib_23', ...])
    # OR simply remove the '.values' from the for loop declaration below and use a list

    for participant in participants_found.values:
        # or like this maybe
        # for participant in ['vmib_006']:
        print(participant)

        ### FIND FILE LOCATIONS!
        # either search or read the file generate by a previous search
        # bundle all files for a participant
        df_bf = utils.search.bundle_associated(
            FLAGS.study_folder, participant, probe_for_duration=False
        )

        # or read excel file
        # dict_bf = pd.read_excel('study_file_listing.xls', sheets=None) # sheets=None reads all sheet of excel file
        # this may some require some other syntax changes

        ### check that we found required files (if not just continue = skip this loop)
        # check for video
        if len(df_bf[df_bf["type"] == "video"]["path"]) == 0:
            print("video not found!")
            continue

        # Todo: handle multiple videos!!
        video_path = df_bf[df_bf["type"] == "video"]["path"].values
        if len(video_path) == 1:
            video_path = video_path[0]
        else:
            print("multiple videos found")
            continue
            # raise ValueError('too many videos found')

        # check for rawetg
        if len(df_bf[df_bf["type"] == "rawetg"]["path"]) == 0:
            print("rawetg not found!\n")
            continue

        # rawetg exists (only look at first available)
        rawetg_path = df_bf[df_bf["type"] == "rawetg"]["path"].values
        if len(rawetg_path) == 1:
            rawetg_path = rawetg_path[0]
        else:
            # Todo: properly handle this
            rawetg_path = rawetg_path[0]
            # raise ValueError('too many rawetgs found')

        # open rawetg and check that it worked
        rawetg_df = utils.rawetg.open(rawetg_path)
        if rawetg_df is None:
            LOG.error(
                "raw etg skipped cause she didn't show (read error): %s", rawetg_df
            )
            continue

        # find and open detected objects
        label_file = utils.search.files(
            Path("/mnt/hdd/VMI_data/14oct2019/output"),
            [video_path.name.split(".")[0]],
            [],
        )
        if len(label_file) == 0:
            LOG.error("No label file found for %s", participant)
            continue
        else:
            label_file = label_file[0]

        # open file check that I got something!
        objects_df = utils.objects.open(label_file)
        if objects_df is None:
            LOG.error("Object open failed objects :(")
            continue

        ### Filter for the object of interest
        objts_of_interest = "Ready?|three|two|one|Done|target|cross|grid|user|safezone"
        ignore_for_now = "disk|hair"

        # looks more complicated than it is.
        # reformat dataframe after filtering
        # Current filter:
        #   - when the string match any of the strings above
        #   - and when the score is above 98%
        objects_reformated = utils.objects.reformat(
            objects_df[
                (
                    objects_df.index.get_level_values("object").str.match(
                        objts_of_interest
                    )
                    & ~objects_df.index.get_level_values("object").str.contains(
                        ignore_for_now
                    )
                )
                & objects_df.score.gt(0.96)
            ]
        )

        ### You can assume that the videos are record at 30Hz
        # but this is how you would check and calculate frame time

        # open video and probe for frames per second
        video = cv2.VideoCapture(str(video_path))
        fps = video.get(cv2.CAP_PROP_FPS)

        # objects_reformated.index*(1000/fps)
        objects_reformated = objects_reformated.assign(
            calc_video_time=utils.objects.calc_video_time(objects_reformated, fps),
        )

        ### merge detected objects to rawetg data (30 -> 60Hz)
        # requires column 'calc_video_time'
        objects_and_rawetg = utils.align.objects_to_rawetg(
            objects_reformated, rawetg_df
        )

        ### UPSAMPLE HERE??

        ### DETECT TASKS
        # first, find where Readys and dones happen
        # utils.objects.group() returns a list of start times for the object in question
        task_starts = utils.objects.group(objects_and_rawetg, "Ready?")
        task_ends = utils.objects.group(objects_and_rawetg, "Done", pd.Timedelta("10s"))
        # print(list(reversed(task_ends)))

        # now, match potential starts with ends to represent complete tasks (less than 6 minutes long and have start + done)
        potential_tasks = []
        for each_start in reversed(task_starts):
            for each_end in reversed(task_ends):
                if (each_start < each_end) and (each_end - each_start) < pd.Timedelta(
                    "6m"
                ):
                    potential_tasks.append(
                        {
                            "start": each_start,
                            "end": each_end,
                            "duration": each_end - each_start,
                        }
                    )
                    task_ends.remove(each_end)
                    break

        # detected task locations (from object detector), ordered by start time according to video
        detected_tasks = (
            pd.DataFrame(potential_tasks).sort_values("start").reset_index(drop=True)
        )

        # then, check that we have at least some of each of the three file types before preceding
        if all(
            x not in df_bf["type"].values for x in ["dflow_rd", "dflow_mc", "cortex"]
        ):
            LOG.info("no dflow or cortex files found")
            print("\033[31mERROR: no dflow or cortex files found \033[0m")
            continue

        # calc group size
        df_bf["group_size"] = df_bf.groupby("task_name", as_index=True)[
            "type"
        ].transform("size")
        df_bf = df_bf.set_index(["group_size", "task_name"]).sort_index(level=[0, 1])

        dflow_times = {}
        # find start and end times from dflow_mc files
        # check that we have the necessary files
        for gname, g in df_bf.groupby(
            ["group_size", "task_name"]
        ):  # google: "pandas dataframe.groupby()"

            # for these tasks there is no dflow_rd
            if gname[1].split("_")[0] in ["closed", "open", "cross"]:

                # check that I have all expected filetypes
                if set(g["type"].unique()) == set(["cortex", "dflow_mc"]):
                    dflow_times[gname[1]] = utils.dflow.probe_elapsed_time(
                        g[g["type"] == "dflow_mc"]["path"].values[0], start_and_end=True
                    )

            # check that I have all expected filetypes
            elif set(g["type"].unique()) == set(["cortex", "dflow_mc", "dflow_rd"]):
                dflow_times[gname[1]] = utils.dflow.probe_elapsed_time(
                    g[g["type"] == "dflow_mc"]["path"].values[0], start_and_end=True
                )

            else:  # error :(
                print("unable to match files associated with %s" % gname[1])

        # sort by start to find task order according to dflow (mostly correct)
        dflow_times_df = pd.DataFrame.from_dict(
            dflow_times, orient="index"
        ).sort_values("start")
        # print(dflow_times_df)

        ### Match tasks together automatically based on duration -> this returns a dataframe or None when failure
        matched = utils.align.match_task_lengths(detected_tasks, dflow_times_df)

        # sometimes autoalignmnet fails -> STOP! Do NOT pass go! Do NOT collect $200!
        if matched is None:
            print("align by length failure")
            continue

        # drop 'group_size' indexer (maybe you don't want to do this?)
        df_bf = df_bf.reset_index("group_size")

        # hold first cross time for each task from respective detected and dflow point of views
        first_cross = {}  
        dflow_and_cortex_dict = {}  # hold each merged dflow+cortex dataframe

        # currently find first cross for each matched task
        # Todo: do this for all tasks
        #       when unable to detect guessimate another way
        for i, row in matched[matched["start_detected"].notnull()].iterrows():
            # print(i)

            first_cross[i] = {"dflow": [], "detected": []}

            # find files that match this task
            files = df_bf[df_bf.index == i]

            # find files that match each type
            cortex_files = files[files["type"] == "cortex"]
            dflow_mc_files = files[files["type"] == "dflow_mc"]
            dflow_rd_files = files[files["type"] == "dflow_rd"]

            # currently require all three
            # todo: change dependent on task
            if any(len(x) == 0 for x in [cortex_files, dflow_mc_files, dflow_rd_files]):
                print("failure file not found")
                continue

            # Todo handle duplicate files (or manually edit file listing excel file)
            cortex_df = utils.cortex.open(cortex_files["path"].head(1).values[0])
            dflow_mc_df = utils.dflow.mc_open(dflow_mc_files["path"].head(1).values[0])
            dflow_rd_df = utils.dflow.rd_open(dflow_rd_files["path"].head(1).values[0])

            # merge dflow and cortex
            merged = utils.align.dflow_to_cortex(dflow_mc_df, dflow_rd_df, cortex_df)

            # add to a dictionary
            dflow_and_cortex_dict[i] = merged

            # print(merged)
            # input()

            # try to detect a cross (return no time if failure)
            # failure caused here when file is improperly read?
            try:
                dflow_first_cross = (
                    merged[merged["CrossVisible.Bool"] == 1].iloc[0].time_rd
                )
                first_cross[i]["dflow"] = dflow_first_cross
            except:
                dflow_first_cross = pd.NaT
                first_cross[i]["dflow"] = dflow_first_cross

            # find first cross in each detected task
            detected_crosses = objects_and_rawetg[
                objects_and_rawetg["rawetg_recording_time_from_zero"].between(
                    row.start_detected, row.end_detected
                )
            ]
            first_cross[i]["detected"] = detected_crosses[
                detected_crosses["cross_score"].notnull()
            ].iloc[0]["rawetg_recording_time_from_zero"]

        first_cross = pd.DataFrame(first_cross).T

        first_cross["difference"] = first_cross["dflow"] - first_cross["detected"]

        # print('mean', first_cross['difference'].mean())
        # print('mode', first_cross['difference'].dt.round('500ms').mode())

        ### for tasks we can't detect for alignment (qs_open, qs_closed, vt)
        # we can estimate alignment from mode
        # check difference between mean and mode after rounding values?
        # the only time this is wrong is when we have different videos or dflow restarted

        first_cross = first_cross.assign(
            dflow_adj_by_dif=first_cross.dflow.subtract(first_cross["difference"])
        )

        # create a dictionary of our merge dflow+cortex dataframes by task
        # you can probably update inplace instead of creating a separate structure for results
        tasks = {}

        for k, df in dflow_and_cortex_dict.items():
            print(k, df.shape)

            if k in tasks.keys():
                print("ERROR: a duplicate task! how did we get this far??")
                break

            df["task_name"] = k  # add column with task_name to each

            # adjust dflow time with time difference between alignment point (currently just first cross)
            df["time_mc_adj"] = df["time_mc"].subtract(first_cross.at[k, "difference"])
            df = df.set_index("time_mc_adj", drop=False)  # set index to time for merge

            tasks[k] = df

        if FLAGS.eachT:
            for k, t in tasks.items():
                # print('task: %s' % k)

                merged = utils.align.rawetg_and_objects_to_dflow_and_cortex(
                    objects_and_rawetg, t
                )

                save_as = "_".join([participant, k, "merged.csv"])
                print(
                    "\nsaving to %s\t\t... should only take a moment, have a wonderful day! :)\n"
                    % save_as
                )

                merged.to_csv(save_as)

        if FLAGS.oneBD:
            # wants a dictionary of dflow tasks and objects+rawetg dataframe
            merged = utils.align.dflow_and_cortex_to_rawetg_and_objects(
                tasks, objects_and_rawetg
            )

            save_as = participant + "_merged.csv"
            print("\nsaving to %s\t\t\t... this may take a hawt minute\n" % save_as)

            merged = merged.to_csv(save_as)

        input(
            "\n\nFINISH! hit me baby one more time!\t\t[ENTER] to auto align next participant"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="perform auto-alignment")

    parser.add_argument("study_folder", type=Path)
    parser.add_argument("--oneBD", "--one_big_dataframe", action="store_true")
    parser.add_argument("--eachT", "--each_task", action="store_true")

    FLAGS = parser.parse_args()

    print(FLAGS)

    if not (FLAGS.oneBD or FLAGS.eachT):
        print('please select output, "eachT" or "oneBD" must be supplied!')
        exit(-1)

    main()
