"""Search and associate study files

Todo: test with other studies

usage: file_search.py [-h] study_folder output_file

Diplays files found with colored output
    Red = duration difference is larger than 5 seconds
    Blue = duplicate files?
    Green = All good!
    Yellow = something else

This script demostrates how to use search methods and group files

Ian Zurutuza
16 December 2019
"""

# import builtin python libs
import argparse
import logging
from pathlib import Path
import re
import sys

# import 3rd party (these must be installed)
import colorama
import numpy as np
import pandas as pd

# import my python files
import hmpldat.file.search

# set-up colored output for the machine you are on (linux vs. windows)
colorama.init()

LOG = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# holds commandline arguments
FLAGS = None


def main():

    # find all participants for a task
    participants_found = hmpldat.file.search.participants(FLAGS.study_folder)
    print(participants_found)
    print()

    # define excel file to save output to
    with pd.ExcelWriter(
        FLAGS.output_file, datetime_format="hh:mm:ss.000000", engine="xlsxwriter"
    ) as writer:

        # For each participant found, search and bundle all tasks
        for participant in participants_found.values:
            print(participant)

            # bundle all files for a participant
            df_bf = hmpldat.file.search.bundle_associated(FLAGS.study_folder, participant)

            # make sure we found dflow and cortex files
            if all(
                x not in df_bf["type"].values
                for x in ["dflow_rd", "dflow_mc", "cortex"]
            ):
                LOG.info("no dflow or cortex files found")
                print("\033[31m no dflow or cortex files found \033[0m")
                continue

            # calc group size
            df_bf["group_size"] = df_bf.groupby("task_name", as_index=True)[
                "type"
            ].transform("size")

            # calc difference between max and min durations
            duration_diff = df_bf.groupby("task_name", as_index=True)["duration"].agg(
                lambda g: g.max() - g.min()
            )
            df_bf = df_bf.join(duration_diff, on="task_name", rsuffix="_diff")

            df_bf = df_bf.set_index(["task_name", "group_size"]).sort_index(
                level=[0, 1]
            )

            # keep? or drop?
            save_to_file = df_bf.droplevel("group_size")
            save_to_file = df_bf.copy()

            save_to_file = save_to_file.sort_index(level=[1, 0])

            # change timedelta into datetime object to format time for saving to excel
            save_to_file["duration"] = save_to_file["duration"] + pd.Timestamp(0)
            save_to_file["duration_diff"] = save_to_file[
                "duration_diff"
            ] + pd.Timestamp(0)

            # save output to file
            save_to_file.to_excel(writer, sheet_name=participant)

            fwarn = writer.book.add_format(
                {"bg_color": "#FFC7CE", "font_color": "#9C0006"}
            )
            fdups = writer.book.add_format(
                {"bg_color": "#ADD8E6", "font_color": "#0000FF"}
            )

            # formating columns
            sheet = writer.sheets[participant]
            sheet.conditional_format(
                "F1:F400",
                {
                    "type": "date",
                    "criteria": "greater than",
                    "value": pd.Timestamp(0) + pd.Timedelta("6s"),
                    "format": fwarn,
                },
            )
            sheet.conditional_format(
                "B1:B400",
                {
                    "type": "cell",
                    "criteria": "greater than",
                    "value": 3,
                    "format": fdups,
                },
            )
            sheet.set_column("A:A", 15)
            sheet.set_column("B:B", 15)
            sheet.set_column("C:C", 15)
            sheet.set_column("D:D", 150)
            sheet.set_column("E:E", 15)
            sheet.set_column("F:F", 15)

            # print output to terminal, colored!
            pd.options.display.max_colwidth = 120
            for gname, g in df_bf.groupby(["group_size", "task_name"]):

                if gname[0] == 1:

                    # these seem to only have a cortex file
                    # we cannot align these
                    if any(s in gname[1] for s in ["static", "dynamic"]):
                        print("\033[30;1m%s" % g.to_string())

                    # somethings wrong here, usually just ducks
                    else:
                        print("\033[33m%s" % g.to_string())

                elif gname[0] == 2:

                    # durations are not similar
                    if g["duration"].max() - g["duration"].min() > pd.Timedelta(
                        5, unit="s"
                    ):
                        print("\033[31m%s" % g.to_string())

                    # everything seems fine
                    elif any(s in gname[1] for s in ["closed", "open", "cross"]):
                        print("\033[32m%s" % g.to_string())

                    # possible duplicate files
                    elif any(s in gname[1] for s in ["static", "dynamic"]):
                        print("\033[36m%s" % g.to_string())

                    # somethings wrong
                    else:
                        print("\033[33m%s" % g.to_string())

                elif gname[0] == 3:

                    # durations are not similar
                    if g["duration"].max() - g["duration"].min() > pd.Timedelta(
                        5, unit="s"
                    ):
                        print("\033[31m%s" % g.to_string())

                    # possible duplicate files
                    elif any(s in gname[1] for s in ["closed", "open", "cross"]):
                        print("\033[36m%s" % g.to_string())

                    # everything seems fine
                    else:
                        print("\033[32m%s" % g.to_string())

                # possible duplicate files
                elif gname[0] == 4:
                    print("\033[36m%s" % g.to_string())

                else:
                    print("\033[30;1m%s" % g.to_string())

                print("\033[0m\n")

            if len(df_bf[df_bf["type"] == "video"]["path"]) > 0:
                print(
                    "\033[32m%s\033[0;0m\n"
                    % df_bf[df_bf["type"] == "video"].to_string()
                )
            else:
                print("\033[31mvideo not found!\033[0m\n")

            if len(df_bf[df_bf["type"] == "rawetg"]["path"]) > 0:
                print(
                    "\033[32m%s\033[0;0m\n"
                    % df_bf[df_bf["type"] == "rawetg"].to_string()
                )
            else:
                print("\033[31mrawetg not found!\033[0m\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
        search for all tasks for each participants in a study folder,
        spit colored output to terminal,
        saved file listing to excel file""",
    )
    parser.add_argument("study_folder", type=Path, help="path to a study folder")
    parser.add_argument("output_file", type=Path, help="filename for .xlsx output")

    FLAGS = parser.parse_args()

    # add a correct file identifier if user doesn't supply
    match = re.match(r".*.xlsx?", FLAGS.output_file.name)
    if not match:
        FLAGS.output_file = FLAGS.output_file.with_suffix(".xlsx")

    main()
