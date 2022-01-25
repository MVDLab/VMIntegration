"""Select objects by location

TODO: add check for required attributes are missing from dataframe for each op

"""

import numpy as np
import pandas as pd

from pprint import pprint


def centered_in_frame(df, px=60, w=960, h=720):
    """

    Args:
        df: DataFrame containing attributes for a single detected object: ctr_bb_col, ctr_bb_row
        w: frame width
        h: frame height
        px: pixels from frame edge within which instances will be removed

    Returns:
        DataFrame of instances within center bounds

    Note:
        (0,0) is top left corner of image frame 
        Target object at it's largets is ~105px diameter

    """

    df = df[
        (df["detected"]["ctr_bb_row"] > h/6)
        & (df["detected"]["ctr_bb_row"] < h - h/6)
        & (df["detected"]["ctr_bb_col"] > w/6)
        & (df["detected"]["ctr_bb_col"] < w - w/6)
    ]
  
    # df = df[
    #     (df["detected"]["ctr_bb_row"] > px)
    #     & (df["detected"]["ctr_bb_row"] < h - px)
    #     & (df["detected"]["ctr_bb_col"] > px)
    #     & (df["detected"]["ctr_bb_col"] < w - px)
    # ]

    return df


def detected(df, score=0.99):
    """

    Args:
        Dataframe containing single detected object attributes: left, right, top, bottom, score 

    Returns:
        df of instances detected that are approximately square

    Note:
        detected object should be bounded by a square (or close to it)

    """

    # score is above specified minimum
    df = df[df["score"] >= score]

    # define detected object sizes
    df = df.assign(width=df["right"] - df["left"], height=df["bottom"] - df["top"])

    # calculate aspect ratio of detected object
    df = df.assign(aspect=df["width"] / df["height"])

    # remove instances where object detected is not close to a square
    df = df[abs(df["aspect"] - 1) < 0.2]

    return df


def each_trial(df):
    """

    Args:
        DataFrame containing mocap positions [x,y,z] for a specific object

    Returns:
        df with instances where dflow says the object is visible 
        and an additional column assigning each group of instances a "trial number"

    Notes:
        Expect 330 trials with the target object
        Expect 346 trials with cross 

    """

    visible = df[df["visible"] == 1]

    # make time index a column to perform required filtering ops
    # difference between two instances larger than one second?
    # assign each trial a number (0 indexed)
    visible = visible.assign(
        trial_number=(visible.index.to_series().diff() > pd.Timedelta("1s"))
        .apply(lambda x: 1 if x else 0)
        .cumsum()
    )

    return visible


def at_frame_edge(df, px_from_edge=5, frame_width=960, frame_height=720):
    """
    given a detected object's bounding box is the object near the edge?

    """

    on_edge = df.assign(
        on_top=df["top"] < px_from_edge,
        on_bottom=df["bottom"] > frame_height - px_from_edge,
        on_left=df["left"] < px_from_edge,
        on_right=df["right"] > frame_width - px_from_edge,
    )
    # top edge of bounding box near top edge of frame
    # bottom edge of bounding box near bottom edge of frame
    # left edge of bounding box near left of frame
    # right edge of bounding box near right edge of frame

    return on_edge.filter(like="on_")
    

def sector_of_frame(df, row_bins=5, col_bins=5, frame_width=960, frame_height=720):
    """Assign a sector to each detected object

    Args:
        data frame with detected locations for a SINGLE object

    Notes:        
        bins with default args:  
        row_bins = [(0.0, 144.0] < (144.0, 288.0] < (288.0, 432.0] < (432.0, 576.0] < (576.0, 720.0]]
        col_bins = [(0.0, 192.0] < (192.0, 384.0] < (384.0, 576.0] < (576.0, 768.0] < (768.0, 960.0]]

    """

    # define bin size
    bin_height = frame_height / row_bins 
    bin_width = frame_width / col_bins

    # define each bin (list of tuples)
    row_bins = pd.IntervalIndex.from_breaks([i*bin_height for i in range(row_bins + 1)])
    col_bins = pd.IntervalIndex.from_breaks([i*bin_width for i in range(col_bins + 1)])

    # assign bin number to each detected object 
    # for rows
    binned_by_row, row_bins = pd.cut(df["ctr_bb_row"], row_bins, retbins=True)
    # rename returned series object
    binned_by_row.name = "sector_row"

    # for columns
    binned_by_col, col_bins = pd.cut(df["ctr_bb_col"], col_bins, retbins=True)
    binned_by_col.name = "sector_col"

    binned = pd.merge(binned_by_col, binned_by_row, left_index=True, right_index=True)

    # print(binned.groupby(["sector_col", "sector_row"]).apply(pd.DataFrame.sample, n=2, replace=False))

    return binned, (row_bins, col_bins)

def task_height(df):
    """

    with cross position know we can determine height used to set object positions

    Args:
        dataframe of cross positions [x,y,z] and task name attribute

    Returns:
        series index=task 

    """

    # tasks 1,2,3,4
    t_1234 = ["fix", "bm", "hp", "pp"]

    # tasks 5,6,7
    t_567 = ["ap", "avoid", "int"]

    task_height = {}

    # cross y positions (for debugging)
    cross_pos = df.groupby("task_name").nth([0])[["x","y","z"]]

    # for each task calculate participant rhead.y used to set projection height
    for t, d in df.groupby("task_name"):

        # the cross position is constant for all trials of each task
        if any(s in t for s in t_1234):
            task_height[t] = d["y"].values[0]
        elif any(s in t for s in t_567):
            task_height[t] = -3230 / np.tan(
                np.arccos((d["z"].values[0] + 3230) / -2800) - np.pi / 2
            )
        else:
            print("error")

    projection_height = pd.concat([cross_pos, pd.Series(task_height, name="calc_via_cross_pos")], axis=1)

    # To check that cross y position matches for tasks 5,6,7 (with new calculated value)
    # projection_height = projection_height.assign(cross_y_recalc=projection_height["calc_via_cross_pos"] + 2800 * np.sin(np.arctan(-3230 / projection_height["calc_via_cross_pos"]) + np.pi/2))

    return projection_height
