"""Compare gaze angles between groups, plot gaze heatmap

compare data etg data between groups
requires group file to know which groups to compare (name='VMIB_group_td or asd_ageGroup.txt)
In this file there should be a list of individuals each separated with a newline character (an ENTER)

TODO: fix my project import statements

"""

import os
import sys
import argparse
import re
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import plotly
import plotly.offline as offline
import colorlover as cl
import plotly.graph_objs as go
import cv2

import hmpldat.utils as utils


# formatting stuff
class _MyFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
):
    pass


LOG = logging.getLogger(__name__)

# output excel document
AVG_OUT = "output_averages.xlsx"
TRIM_OUT = "trim_info.xlsx"
IND_OUT = "output_data.xlsx"
ANNOT_OUT = "output_annotation_info.xlsx"

# command line args are stored here
FLAGS = None

RADIUS = 2.49
"""distance to screen from (0,0,0), in meters"""

HEAD = {"x": 480, "y": 360}
"""
Our original assumption for where the participant was expected to be looking (Early Summer 2019)

where @ is the location of our head vector
::
    0         480    
  0 +----------------------+
    |                      |
    |                      |
 360|          @           |
    |                      |
    |                      | 
    +----------------------+

Notice Y values increase in the negative direction. This is standard for images.

"""


def angle(points):
    """calc angle between binocular PoR and head vector

    Args:
        points: gaze location tranformed to new origin, expected focus

    Returns:
        gaze angle from expected focus

    Notes:

        Is the 46 degrees actually true? 

        * = 720/960 (scene camera resolution)
        * = 60/80 (eye tracking ability)
        * != 46?/60 (camera field of view) ... why not 45?
    
        I calculated the difference between the calculated vertical and horizontal scalars (code block below).
        So to simplify the calculations, when working in polar, I assume that the vertical and horizontal scales are equal

        >>> x_scalar = np.tan(np.pi / 6) / 480
        >>> y_scalar = np.tan(23 * np.pi / 180) / 360
        >>> abs(x_scalar - y_scalar)
        2.3716349118373982e-05
        # My ti calculator says even smaller difference = 4.229e-07
        
        This is the vertical, "y", component if the max vertical angle is 22.5 (45/2), not 23 (46/2) degrees 

        >>> np.tan(np.pi / 8) / 360)

    .. figure:: ../figures/gaze_vec_calc.png
        :align: center

        Method for calculating gaze angle
    
    See Also:
        .. function:: objects.in_range() 
        
            methods based off of max angle derivation

    """

    if FLAGS.p:
        # polar representation assume a circle (I could probably derive this as a function on a oval
        # This assumption seems more reasonable to me (discussed again in the next comment)
        # print(pd.DataFrame(np.arctan(points['r'] * np.tan(np.pi / 6) / 480)))
        return pd.DataFrame(np.arctan(points["r"] * np.tan(np.pi / 6) / 480))
    else:
        x_ang = np.arctan(points["centered_x"] * np.tan(np.pi / 6) / 480)
        y_ang = np.arctan(points["centered_y"] * np.tan(23 * np.pi / 180) / 360)

        return x_ang, y_ang


def avg_individual(df: pd.DataFrame, name: str):
    """Calculate averages for one participant

    Todo: 
        use plot function outside of this method

    Args:
        df: dataframe of gaze data
        name: participant name (used to name plot)

    """

    # info will be added to this dataframe and returned
    info_avg_df = pd.DataFrame(
        columns=["in_task", "out_task"],
        index=[
            "percentage_of_total",
            "percentage_visual_input_or_saccade",
            "percentage_time_OoR",
            "switches_to_OoR_vs_total",
            "in_range_saccades_vs_total",
        ],
    )

    angle_avg_df = {}

    octant_avg_df = pd.DataFrame(columns=["in_task", "out_task"])

    bool_dict = {True: "in_task", False: "out_task"}

    # % of total in task / out of task
    split_info = df["in_task"].value_counts(normalize=True)
    split_info.index = [bool_dict[x] for x in split_info.index]
    split_info.name = "percentage_of_total_recording"
    info_avg_df.loc["percentage_of_total"] = split_info

    # split the data into dictionary
    data_dict = {"in_task": df[df["in_task"]], "out_task": df[~df["in_task"]]}

    # for each in and out of task data
    for key in data_dict:

        # remove any data that is not visual input or saccade
        orig_len = len(data_dict[key])
        data_dict[key] = data_dict[key][
            (data_dict[key]["Category Binocular"] == "Saccade")
            | (data_dict[key]["Category Binocular"] == "Visual Intake")
        ]

        # print(len(data_dict[key]) / orig_len)
        info_avg_df.loc["percentage_visual_input_or_saccade", key] = (
            len(data_dict[key]) / orig_len
        )

        info_avg_df.loc["percentage_time_OoR", key] = (
            ~data_dict[key]["in_range"]
        ).sum() / len(data_dict[key])

        info_avg_df.loc["switches_to_OoR_vs_total", key] = (
            ~data_dict[key]["in_range"]
            & (data_dict[key]["in_range"] != data_dict[key]["in_range"].shift())
        ).sum() / len(data_dict[key])

        # select only in range data
        data_dict[key] = data_dict[key][data_dict[key]["in_range"]]

        info_avg_df.loc["in_range_saccades_vs_total", key] = (
            (data_dict[key]["Category Binocular"] == "Saccade")
            & (
                data_dict[key]["Category Binocular"]
                != data_dict[key]["Category Binocular"].shift()
            )
        ).sum() / len(data_dict[key])

        angle_avg_df[key] = {}

        # average gaze angles
        for col in data_dict[key].columns:
            if "gaze" in col:
                angle_avg_df[key][col + "_avg"] = data_dict[key][col].mean()
                angle_avg_df[key][col + "_std"] = data_dict[key][col].std()

        # when normalized == True:
        # then the counts are divided by the total to return a percentage from [0.0, 1.0]
        #       now each participant is worth the same weight in the total average
        #       (set false and each data point is worth the same weight)
        octant_avg_df[key] = data_dict[key]["octant"].value_counts(normalize=True)
        octant_avg_df[key].index.name = "octant in radians"

    # plot data trimmed data (in range and in task)
    # if FLAGS.plot:
    plot_2d_hist(data_dict["in_task"], name)

    angle_avg_df = pd.DataFrame(angle_avg_df)

    print(
        info_avg_df.to_string(
            col_space=10,
            float_format=lambda o: "{:.3f}%".format(o * 100),
            justify="center",
        )
    )
    print(
        angle_avg_df.to_string(
            col_space=10,
            float_format=lambda o: "{:.3f}\u00b0".format(o * 180 / np.pi),
            justify="center",
        )
    )
    print(
        octant_avg_df.to_string(
            col_space=10,
            float_format=lambda o: "{:.3f}%".format(o * 100),
            justify="center",
        )
    )

    return info_avg_df, angle_avg_df, octant_avg_df


def calc(df: pd.DataFrame):
    """for each gaze location center point then calc angle and gaze "octant"

    Different methods for rectanglar 

    Args:
        df: participant data

    Returns:
        gaze angles in rectangular or polar cooridinates

    """

    gaze = df[["Point of Regard Binocular X [px]", "Point of Regard Binocular Y [px]"]]

    print(type(gaze))
    gaze = gaze.rename(
        columns={
            "Point of Regard Binocular X [px]": "x",
            "Point of Regard Binocular Y [px]": "y",
        }
    )

    obj = df[["ctr_bb_col", "ctr_bb_row"]]
    obj = obj.rename(columns={"ctr_bb_col": "x", "ctr_bb_row": "y"})

    centered = center(gaze, obj)

    print(centered)
    if FLAGS.p:
        r, theta = to_polar(centered)
        polar = centered.assign(r=r, theta=theta)
        polar = polar.assign(gaze_angle=angle(polar), octant=get_orthant(polar))

        print(polar)
        return polar

    else:
        rect = centered
        x_ang, y_ang = angle(rect)
        rect = rect.assign(x_gaze_ang=x_ang, y_gaze_ang=y_ang, octant=get_orthant(rect))

        return rect


# TODO: use euclidean distance in math.py (remove this)
def dist(gaze, center={"x": 0, "y": 0}) -> float:
    """2D euclidean distance

    Calculates distance from center to gaze location
    Gaze location is transformed so the "expeced focus" is the origin.
    See: transform_to_relative_center()

    Args:
        gaze: gaze locations 
        center: DEFAULT=(0,0) 
    
    Returns: 
        distance

    """

    return (
        (gaze["centered_x"] - center["x"]) ** 2
        + (gaze["centered_y"] - center["y"]) ** 2
    ) ** 0.5


def get_orthant(gaze):
    """calculate "orthant" where the gaze vector is located
    
    Args:
        gaze: gaze location transformed around expected focus

    Returns:
        integer representing which quadrant participant is looking

    """
    # divide into this many quadrants
    n = 8  # allow options: make this dependant on command line flag?

    # TODO: if you want to set your own quadrants, read from file?? add flag?

    # do this in rectangular ?
    # I calculate gaze angle (transform to polar)
    if "theta" not in gaze.columns:
        gaze["theta"] = np.arctan2(gaze["centered_y"], gaze["centered_x"])

    # to make binning easier lets change the bounds of our angles from (-pi, pi) -> (-9pi/8, 7pi/8)
    if n == 8:
        gaze.loc[gaze.theta > 7 * np.pi / 8, "theta"] += -2 * np.pi

    offset = np.pi / n if n % 2 == 0 else 0
    start = -np.pi - offset

    bounds = [start]
    for i in range(n):
        bounds.append(start + (i + 1) * np.pi / (n / 2))

    return pd.cut(gaze["theta"], bounds)


def final_avg(df):
    """calculate average and std of each row by group: helper function

    Args:
        df: DataFrame where each column is a participant's averages
    
    Returns:
        final averages as pd.DataFrame

    """

    # drop any rows with 'std' (individual standard deviation calculations)
    if type(df.index) is not pd.CategoricalIndex:
        for i in df.index:
            if "std" in i:
                df.drop(index=i, inplace=True)

    mean = df.groupby(axis=1, level=[0, 2]).mean()
    std = df.groupby(axis=1, level=[0, 2]).std()

    out = mean.join(std, lsuffix="_avg", rsuffix="_std")

    out.sort_index(axis=1, level=1, inplace=True)
    out = out.reorder_levels([1, 0], axis=1)

    return out


def plot_2d_hist(df: pd.DataFrame, name: str):
    """plot data 2d histogram to file

    Todo: update this method to use seaborne and matplotlib

    Args:
        df: data to plot
        name: title for figure
    
    """

    scl = cl.scales["9"]["seq"]["BuPu"]
    colorscale = [[float(i) / float(len(scl) - 1), scl[i]] for i in range(len(scl))]

    print("plot this")
    print(df)
    print(df.describe())

    xaxis = dict(
        range=[-698, 698],
        ticks="outside",
        showgrid=True,
        zeroline=True,
        showline=True,
        mirror="ticks",
        gridcolor="#bdbdbd",
        gridwidth=2,
        zerolinecolor="#000000",
        zerolinewidth=2,
        linewidth=2,
        linecolor="#444",
    )

    yaxis = dict(
        range=[-490, 490],
        ticks="outside",
        showgrid=True,
        zeroline=True,
        showline=True,
        mirror="ticks",
        gridcolor="#bdbdbd",
        gridwidth=2,
        zerolinecolor="#000000",
        zerolinewidth=2,
        linewidth=2,
        linecolor="#444",
    )

    fig = {
        "data": [
            go.Histogram2dContour(
                x=df["centered_x"],
                y=df["centered_y"],
                colorscale=colorscale,
                hoverinfo=None,
                opacity=0.85,
            )
        ],
        "layout": go.Layout(
            xaxis=xaxis,
            yaxis=yaxis,
            width=750,
            height=500,
            autosize=False,
            hovermode=False,
            title=name,
        ),
    }

    plotly.io.write_image(fig, "plots/" + name + ".svg")
    # fig.write_image('plots/' + name + '.svg')
    # offline.plot(fig, 'plots/' + name + '.svg', image='svg')


def to_polar(gaze):
    """Convert rectangular coordinates to polar form

    Args:
        gaze: DataFrame of centered X and Y points
    
    Returns: 
        polar representation of points in relation to our "expected focus" origin

    """
    return dist(gaze), np.arctan2(gaze["centered_y"], gaze["centered_x"])


def transform_to_relative_center(rect_points, expected_focus):
    """Move gaze location relative to new origin (expected focus)

    Args:
        rect_points: df of X, Y point from raw etg
        expected_focus: the actual expected_focus for the data dict
    
    Returns: 
        A dictionary (X, Y) Binocular PoR points transformed to new expected_focus

    Example:
        where @ is the location of our head vector (h, k) (480, 360)
        ::
            0         480
          0 +----------------------+
            |                      |
            |                      |
         360|          @           |
            |                      |
            |                      |
            +----------------------+

        Since the center point is not the expected_focus we cannot directly convert to polar.

                (h, k) -> (0, 0)
        
        for each point (x', y') = (x-h, k-y)
        ::
          -480         0
         360+----------------------+
            |                      |
            |                      |
         0  |          @           |
            |                      |
            |                      |
            +----------------------+

        Now we can easily convert to polar and decide which quadrant we are in
        And can ignore subtraction for angle calculations

        This method also holds true when using an object center as the expected_focus

    """

    return pd.DataFrame(
        {
            "centered_x": rect_points["Point of Regard Binocular X [px]"]
            - expected_focus["x"],
            "centered_y": expected_focus["y"]
            - rect_points["Point of Regard Binocular Y [px]"],
        }
    )


def main(_):

    found_group_file = False

    ind_info = {}
    avg_stats = {}
    groups = {}
    duplicates = []
    count = 0

    for rawetg_file in search.search(FLAGS.data_dir, ["rawetg"], ["vmib_003"]):

        participant = re.match(".*([\D]{4}_[\d]{3}).*", rawetg_file.name).group(1)
        print(participant)

        rawetg_df = rawetg.open(rawetg_file)
        if rawetg_df is None:
            LOG.error(
                "raw etg skipped cause she didn't show (read error): %s", rawetg_df
            )
            continue

        video_file = search.search(FLAGS.data_dir, [participant, "30hz"], [])
        print(video_file)
        if video_file is None:
            LOG.error("No video found for %s", participant)
            continue
        elif len(video_file) == 1:
            video_file = video_file[0]
        else:
            raise ValueError("too many videos found")

        label_file = search.search(FLAGS.labels, [video_file.name.split(".")[0]], [])
        if len(label_file) == 0:
            LOG.error("No label file found for %s", participant)
            continue
        else:
            label_file = label_file[0]

        print(label_file)
        objects_df = objects.open(label_file)

        if participant in avg_stats.keys():
            duplicates.append(participant)
            if FLAGS.v == 1:
                print("Potentially duplicate file: {}".format(file))

        avg_stats[participant] = {}

        objts_of_interest = "Ready?|three|two|one|Done|target|cross|grid|user|safezone"
        ignore_for_now = "disk|hair"

        objects_reformated = objects.reformat(
            objects_df[
                (
                    objects_df.index.get_level_values("object").str.match(
                        objts_of_interest
                    )
                    & ~objects_df.index.get_level_values("object").str.contains(
                        ignore_for_now
                    )
                )
                & objects_df.score.gt(0.98)
            ]
        )

        # calc_video time here
        video = cv2.VideoCapture(str(video_file))
        fps = video.get(cv2.CAP_PROP_FPS)

        # objects_reformated.index*(1000/fps)
        objects_reformated = objects_reformated.assign(
            calc_video_time=objects.calc_video_time(objects_reformated, fps),
        )

        print(objects_reformated.columns)

        # objects_reformated.sort_values('calc_video_time', inplace=True)
        # objects_reformated.to_csv('objects_reformated.csv')

        rawetg_df = rawetg_df.dropna(subset=["Video Time [h:m:s:ms]"])
        # rawetg_df.to_csv('rawetg_b4_merge.csv')

        # up_sample object of interest (if one exists) from labels file align to rawetg
        merged = align.objects_to_rawetg(objects_reformated, rawetg_df)

        merged = merged.assign(identifier=objects.expected_focus(merged))

        merged.to_csv("check_this_out.csv")

        # drop columns that are not related target or cross
        # combine cross and target locations into columns ctr_bb_col, ctr_bb_row
        merged = objects.filter_for

        # calc gaze angle and octant for each data point
        info = smaller.join(calc(smaller))

        print(smaller.size)
        print(info.size)

        # for saving individual info to file
        if FLAGS.s == 2:
            ind_info[participant] = info

        (
            avg_stats[participant]["info"],
            avg_stats[participant]["angles"],
            avg_stats[participant]["octants"],
        ) = avg_individual(info, participant)

    # read group file
    for group_file in search.search(FLAGS.data_dir, ["group", ".txt"], []):
        with open(group_file) as f:
            gname = group_file.split("_")
            gname = gname[2] + "_" + gname[3].split(".")[0]

            if gname not in groups:
                groups[gname] = f.read().splitlines()
                found_group_file = True

            else:
                groups[gname] += f.read().splitlines()

    sys.stdout.write("\r" + "read & calc finish!".ljust(40, " ") + "\n\n")
    sys.stdout.flush()

    # print(groups.keys())
    # write individual data to file?
    if FLAGS.s == 2:
        sys.stdout.write("writing individual data to " + IND_OUT + "\n")
        sys.stdout.flush()

        with pd.ExcelWriter(IND_OUT) as writer:
            i = 0
            for ind in ind_info:
                i += 1
                sys.stdout.write(
                    "\r"
                    + "writing {} to file ({}/{})".format(ind, i, len(ind_info)).ljust(
                        30, " "
                    )
                )
                sys.stdout.flush()
                ind_info[ind].to_excel(writer, sheet_name=ind)

        sys.stdout.write("\n\n")
        sys.stdout.flush()

    assert found_group_file, "No group file found! I don't know who to compare to who!"

    if duplicates:
        print("Duplicate files found for trials: {}".format(", ".join(duplicates)))
        print("\tthese files are potentially not duplicates")
        print("\tand only the data from one file is used (I don't know which)")
        print("\tplease remove duplicate files\n")
        # TODO: print('\tor ignore unwanted files with command line arg \'--i\'')

    # split avg_stats into dictionary by group and data (quad averages | angle averages)
    # this will help display and compare the values we want to compare
    angles_by_group = {}
    quads_by_group = {}
    trim_by_group = {}
    na = []

    for g in groups:
        angles_by_group[g] = {}
        quads_by_group[g] = {}
        trim_by_group[g] = {}

        for name in groups[g]:

            if name in avg_stats.keys():
                quads_by_group[g][name] = avg_stats[name].pop("octants")
                trim_by_group[g][name] = avg_stats[name].pop("info")
                angles_by_group[g][name] = avg_stats[name].pop("angles")

            else:
                na.append(name)

    if na:
        print(
            "No raw etg files found for the following participants named in group files:"
        )
        print(end="\t")
        print(
            "\n\t".join(
                [
                    ", ".join(map(str, sl))
                    for sl in [na[i : i + 9] for i in range(0, len(na), 9)]
                ]
            )
        )
        print()

    # bookkeeping stuff, convert my dictionary into pandas DataFrame,
    # now I save to excel easily and information will be properly organized to calculate total averages
    tuples = [(x, y) for x in [*quads_by_group] for y in [*quads_by_group[x]]]

    _q = pd.concat(
        [quads_by_group[x][y] for x in [*quads_by_group] for y in [*quads_by_group[x]]],
        keys=tuples,
        axis=1,
    )
    _a = pd.concat(
        [
            angles_by_group[x][y]
            for x in [*angles_by_group]
            for y in [*angles_by_group[x]]
        ],
        keys=tuples,
        axis=1,
    )
    _t = pd.concat(
        [trim_by_group[x][y] for x in [*trim_by_group] for y in [*trim_by_group[x]]],
        keys=tuples,
        axis=1,
    ).astype("float64")

    # calculate final averages
    _q_totals = final_avg(_q)
    _a_totals = final_avg(_a)
    _t_totals = final_avg(_t)

    # save to file
    if FLAGS.s:
        with pd.ExcelWriter(AVG_OUT) as writer:
            _a.to_excel(writer, sheet_name="gaze_angles_averages")
            _a_totals.to_excel(writer, sheet_name="gaze_angle_total_averages")
            _q.to_excel(
                writer, sheet_name="octant_averages", index_label="octant in radians"
            )
            _q_totals.to_excel(writer, sheet_name="octant_tot_averages")
            _t.to_excel(writer, sheet_name="trimming_averages")
            _t_totals.to_excel(writer, sheet_name="trimming_tot_averages")

            print("averages saved to: ", AVG_OUT)

    print()
    print("Average octant distribution".center(80, "-"))
    print(
        _q_totals.to_string(
            col_space=10,
            formatters={
                "index": lambda k: "({:7.2f}\u00b0, {:7.2f}\u00b0]".format(
                    k.left, k.right
                )
            },
            float_format=lambda o: "{:.3f}%".format(o * 100),
            justify="center",
        )
    )

    print()
    print("Average gaze angles".center(80, "-"))
    print(
        _a_totals.to_string(
            col_space=10,
            float_format=lambda o: "{:.3f}\u00b0".format(o * 180 / np.pi),
            justify="center",
        )
    )

    print()
    print("Average trimming info".center(80, "-"))
    print(
        _t_totals.to_string(
            col_space=8,
            float_format=lambda o: "{:.3f}%".format(o * 100),
            justify="center",
        )
    )


def test():
    """
    Test stuff, print results to terminal

    Todo:
        make a pytest module ;)

    """

    # ANGLE TEST
    print("    ANGLE TEST    ".center(40, "#"))

    # test angle for different pixel values
    if FLAGS.p:
        test_df = pd.DataFrame({"r": [0, 10, 240, 480, 698]})

        print(
            test_df.join(angle(test_df), lsuffix="_px", rsuffix="_angle").to_string(
                col_space=2,
                float_format=lambda o: "{:.5f}\u00b0".format(o * 180 / np.pi),
                justify="center",
            )
        )
    else:
        test_df = pd.DataFrame(
            {
                "X": [0, 10, 240, 480, 698, -698, -480, -240],
                "Y": [0, 10, 180, 360, 490, -490, -360, -180],
            }
        )

        df = pd.DataFrame(angle(test_df)).T
        print(
            test_df.join(df, lsuffix="_px", rsuffix="_angle").to_string(
                col_space=2,
                float_format=lambda o: "{:.5f}\u00b0".format(o * 180 / np.pi),
                justify="center",
            )
        )
    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
                                     calculate angle between gaze vector and center from raw ETG data
                                     default: do these calculations in rectangular
                                     """,
        formatter_class=_MyFormatter,
    )

    parser.add_argument(
        "--data_dir",
        type=Path,
        default="/mnt/hdd/VMI_data/vmi/datasets/VMIB/Data",
        help="where is my raw ETG data?\n",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default="/mnt/hdd/VMI_data/14oct2019/output",
        help="where is label files?\n",
    )
    parser.add_argument(
        "--select",
        metavar="IDENTIFIER",
        nargs="+",
        type=str,
        default=[],
        help="select specific video, accepts a list of arguments\n"
        'example: "--specify VMIB 024" will only process file for VMIB participant 024\n',
    )
    parser.add_argument(
        "--p",
        "--polar",
        action="store_true",
        default=True,
        help="perform calculations in polar\n",
    )
    parser.add_argument(
        "--s",
        "--save",
        action="count",
        default=2,
        help="save generated data? default only prints to command line\n"
        "'--s'  = averages saved to excel file\n",
    )
    # '\'--s --s\' = individual information saved\n')
    parser.add_argument(
        "--v",
        "--verbose",
        action="count",
        default=0,
        help="how much info to show\n" 'more v\'s = more verbose (max=2 "--vv")\n',
    )
    parser.add_argument(
        "--t",
        "--test",
        action="store_true",
        default=False,
        help="run test function: " "\t- demonstrates angle calculation ",
    )
    parser.add_argument(
        "--a", action="store_true", default=False, help="save annotation information"
    )

    FLAGS, _ = parser.parse_known_args()

    FLAGS.select.append("RawETG")

    if FLAGS.t:
        print("testing ... ")
        test()

    main(_)
