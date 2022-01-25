""" Methods for handling detected object files

"""
import logging
from pathlib import Path
import time
from pprint import pprint

import pandas as pd
from tqdm import tqdm

LOG = logging.getLogger(__name__)


def calc_video_time(df: pd.DataFrame, fps):
    """calculate frame time based on frame number and video fps

    .. math:: 
    
        frame_{time} = frame_{\#} * \\frac{1000}{fps}

    Args:
        df: a data frame indexed by frame_number
        fps: frames per second from video.get(cv2.CAP_PROP_FPS)

    Returns:
        a single dataframe column of calculated video time

    """

    return pd.to_timedelta(df.index * (1000 / fps), unit="ms")


def expected_focus(df: pd.DataFrame):
    """created expected focus column
    
    Todo:
        split into separate functions for marking:
            * starts and ends
            * objects in dataframe
            * calibration (ts) task

    If a cross exists that is the expected focus
    If a single "target" ball exists that is the expected focus
    Cross takes priority

    Args:
        group: reformated objects dataframe

    Returns:
        The expected object of focus (new column of strings ['target', 'cross', '])

    """

    mask = pd.DataFrame(None, columns=["mask"], index=df.index)

    # print(df.filter(like='target').dropna().shape)

    # exptd_focus = ((df.filter(regex='cross|target').any(1)) and not df['target_multiple'])
    # with pd.option_context('display.min_rows', 50):
    #     print(exptd_focus)
    #     # print(df.filter(regex='cross|target'))
    #     # print(df.filter(regex='cross|target').any(1))
    #     # print(df.filter(regex='target_multiple'))

    # print(df[df.filter(like='score').gt(.95).])

    # filter by score
    # TODO: move this outside?
    # TODO: increase score threshold?
    # TODO: make score threshold global
    # df = df[
    #     (
    #         (df['cross_score'].gt(0.95)) |
    #         (df['target_score'].gt(0.95)) |
    #         (df['Done_score'].gt(0.98)) |
    #         (df['Ready?_score'].gt(0.95)) |
    #         (df['three_score'].gt(0.95)) |
    #         (df['two_score'].gt(0.95)) |
    #         (df['one_score'].gt(0.95))
    #     )
    # ]

    mask = pd.DataFrame(None, columns=["mask"], index=df.index)

    # The following two line must be in this order ("target" and "target_multiple" or target in target_multiple == True)
    mask[
        df.filter(like="target").any(1).rolling(32, 11, center=True).sum() >= 16
    ] = "target"
    mask[df["target_multiple"].rolling(32, 11, center=True).sum() >= 11] = "ts"

    mask[
        df.filter(like="cross").any(1).rolling(32, 11, center=True).sum() >= 16
    ] = "cross"

    # The following two lines must be in this order ("one" and "Done" regex together)
    mask[
        df.filter(regex="Ready?|three|two|one")
        .any(1)
        .rolling(61, 11, center=True)
        .sum()
        >= 32
    ] = "start"
    mask[
        df.filter(like="Done").any(1).rolling(61, 11, center=True).sum() >= 32
    ] = "done"

    # print(mask['mask'].value_counts())

    return mask


def group(
    df, object_name, time_between=pd.Timedelta("1s"), duration=pd.Timedelta("250ms")
):
    """identify start times for an object

    Args:
        df: 
        object: object to look for e.g. 'Ready?' or 'Done'
        time_between: minimum change in time between each object appearance 
        duration: object must appear for at least this amount of time

    Returns:
        list of start times for object

    """
    # print(object_name)
    # drop rows where our object does not exist
    df = df.dropna(subset=["_".join([object_name, "score"])])

    # count number of object locations
    df = df.assign(
        count=(df["rawetg_recording_time_from_zero"].diff() > time_between).cumsum()
    )

    # get start time from each group that exceeds requested a min duration
    object_starts = []
    for gname, group in df.groupby(df["count"]):
        if (
            group["rawetg_recording_time_from_zero"].max()
            - group["rawetg_recording_time_from_zero"].min()
        ) > duration:
            object_starts.append(group["rawetg_recording_time_from_zero"].iat[0])

    return object_starts


def identify_location(df: pd.DataFrame):
    """Guesstimate task locations from detected objects

    DO NOT USE

    Args:
        df: 

    Returns:
        Dataframe (a single column) that identifies [start, end, in_task]

    Notes: 
        * current method combine related columns 
        * roll a centered window sum boolean values
        * if the sum is greater than half window length then it should belong to identifier

    See: 
        expected_focus()

    """

    mask = pd.DataFrame(None, columns=["mask"], index=df.index)

    # in task
    intask_columns = [
        "grid_visible",
        "target_visible",
        "user_visible",
        "safezone_visible",
        "cross_visible",
    ]
    mask[df[intask_columns].any(1).rolling(200, 1, center=True).sum() >= 101] = "intask"

    # start
    start_columns = ["Ready?_visible", "three_visible", "two_visible", "one_visible"]
    mask[df[start_columns].any(1).rolling(32, 1, center=True).sum() >= 17] = "start"

    # end
    mask[df["Done_visible"].rolling(15, 1, center=True).sum() >= 9] = "end"

    # trouble shooting task is easy to identify
    mask[df["multiple_targets"].rolling(15, 1, center=True).sum() >= 9] = "ts"

    return mask


def open(path: Path) -> pd.DataFrame:
    """Open my file bounding boxes info to dataframe for drawing
       
    Returns:
        pandas dataframe multi-indexed by [frame number, object], with columns
        
        .. code-block:: python

            object, score,
            left, right, top, bottom,

    Notes:
        does not add additional columns:
            * ctr_bb_row: (bottom + top)/2 take top and bottom edges of bounding box and calculate midpoint 
            * ctr_bb_col: (right + left)/2 take right and left edges of bounding box and calculate midpoint
    """

    # LOG.info(" Reading label file: %s", path)
    objects_df = pd.read_csv(
        path,
        )

    # strip space out of column names
    # required for some older detection files 
    objects_df.columns = objects_df.columns.str.strip()
    # objects_df.index.names = ["frame_number", "object"]

    objects_df['frame_time'] = pd.to_timedelta(objects_df['frame_time'], unit='ms')

    # objects_df = objects_df.set_index(["frame_number"])
    # if LOG.isEnabledFor(logging.DEBUG):
    #     LOG.debug('file as read into memory:\n%s', objects_df.to_string(max_rows=20, line_width=LINE_WIDTH))

    # print(objects_df)
    return objects_df


def milestone(df, width=960, height=720):
    """

    select instances when object is fully in the frame

    horizontal bounds
    vertical bounds

    """

    hb = width / 7
    vb = height / 7

    df = df[(df["CrossVisible.Bool"] == 1) & (df["cross_score"] >= 0.9)].copy()

    count = {}
    count["detected_and_visible"] = len(df)

    df["time_diff"] = df["time_mc_adj"].diff() > pd.Timedelta("1s")
    df["cross#"] = df["time_diff"].apply(lambda x: 1 if x else 0).cumsum()

    # remove instances outside of center bounds
    df = df[
        (df["cross_ctr_bb_col"] > hb)
        & (df["cross_ctr_bb_col"] < (width - hb))
        & (df["cross_ctr_bb_row"] > vb)
        & (df["cross_ctr_bb_row"] < (height - vb))
    ]
    count["in_frame"] = len(df)

    df = df.set_index(["cross#"], append=True).swaplevel()

    return df, count


def lr_milestone(df, width=960, height=720):
    """

    select instances when cross is on left or right third

    horizontal bounds
    vertical bounds

    """

    hb = width / 3
    vb = height / 3

    df = df[(df["CrossVisible.Bool"] == 1) & (df["cross_score"] >= 0.9)].copy()

    count = {}
    count["detected_and_visible"] = len(df)

    df["time_diff"] = df["time_mc_adj"].diff() > pd.Timedelta("1s")
    df["cross#"] = df["time_diff"].apply(lambda x: 1 if x else 0).cumsum()

    # remove instances outside of center bounds
    df = df[
        (df["cross_ctr_bb_col"] < hb)
        | (df["cross_ctr_bb_col"] > (width - hb))
        | (df["cross_ctr_bb_row"] < vb)
        | (df["cross_ctr_bb_row"] > (height - vb))
    ]
    count["edge_frame"] = len(df)

    df = df.set_index(["cross#"], append=True).swaplevel()

    return df, count


def multiindex(df):
    """
    Distance between each detected object and gaze location

    Returns as list closest first

    """

    info_col = ["score", "left", "right", "top", "bottom", "ctr_bb_row", "ctr_bb_col"]
    rstr = "|".join(["(.*)_" + s for s in info_col])

    # multi-index object
    object_names = df.filter(regex=rstr).columns.values
    object_names = [s.split("_", 1) for s in object_names]

    objects = df.filter(regex=rstr)
    objects.columns = pd.MultiIndex.from_tuples(object_names)

    return objects


def reformat(df) -> pd.DataFrame:
    """Reformat objects dataframe for joining to the rest of our data

    Consider filtering by score before running this function (e.g. df[df['score'] > 0.95])
    calculates whether multiple targets exist on frame (we otherwise lose that info here)

    Args: 
        df: objects as a dataframe from 

    Returns:
        a new dataframe indexed by frame_number, with columns

        # TODO: update this
        .. code-block:: python

            ('object', 'score'),
            ('object', 'left'), ('object', 'right'), ('object', 'top'), ('object', 'bottom'), 
            ('object', 'ctr_bb_row'), ('object', 'ctr_bb_col'), 
            ... 
            ('target', 'multiple'),

    """

    # groupby index (frame_number, frame_time, object)
    grouped = df.groupby(["frame_number", "frame_time", "object"], dropna=False)

    # return size of each group (used to calc multiple targets)
    group_sizes = grouped.size()

    # each group already ordered by score, so take the first object from each 
    # group (highest score) to represent the object that exists
    grouped = grouped.first()

    # unstack and then swap multi-indexed column names
    reformated = grouped.unstack().swaplevel(axis=1)
    reformated = reformated.reset_index()

    # calculate and reformat whether multiple targets exist
    multiple_targets = group_sizes[group_sizes.index.get_level_values(1) == "target"]
    multiple_targets.index = multiple_targets.index.droplevel(1)
    multiple_targets.name = ("target", "count")

    # add multiple targets column
    reformated = pd.concat([reformated, multiple_targets], axis=1)  

    # remove frame number & time, so they aren't affected by the formatting applied next
    frame_tickers = reformated[['frame_number', 'frame_time']]
    frame_tickers.columns = [t[0] for t in frame_tickers.columns.to_flat_index()]
    
    # TODO: this statement creates warning (performance?), investigate
    reformated = reformated.drop(['frame_number', 'frame_time'], axis=1, level=0)

    # drop "NaN" object, this is occuring because a record is created for every image frame 
    # even if no objects are detected
    reformated = reformated.drop(labels=pd.NA, axis=1, level=0)

    # I want the columns in this order
    columns_to_order = ['cross', 'target', 'user', 'safezone', 'grid', 'Done', 'Ready?', 'three', 'two', 'one']

    # using defined order first, then add the rest of the columns to the end
    new_column_order = columns_to_order + reformated.drop(labels=columns_to_order, axis=1, level=0).columns.get_level_values(0).unique().to_list()

    # apply
    reformated = reformated.reindex(columns=new_column_order, level="object")
    
    # change multi-indexed columns into single level
    reformated.columns = ['.'.join(t) for t in reformated.columns.to_flat_index()]

    # join frame number & time back to df
    reformated = frame_tickers.join(reformated)

    return reformated


def count_back(frame_time, frame_number, fps):
    """
    """

    times = []
    
    for x in reversed(range(frame_number)):
        frame_time = frame_time - pd.Timedelta(1000/fps, 'ms')

        times.append((x, frame_time))

    times.reverse()

    return pd.DataFrame(times, columns=["frame_number", "corrected_frame_time"])


def fix_frame_time(df):
    """
    Adjust frame time column to correctly count upwards

    from:
    0             0 days 00:00:00              
    1             0 days 00:00:00              
    2             0 days 00:00:00.033333300    
    3             0 days 00:00:00.066666600    
    4             0 days 00:00:00.033333300    

    to:
    0            -1 days +23:59:59.966666370    ~= -00:00:00.033333          
    1            -1 days +23:59:59.999999703    ~= -00:00:00.000000
    2             0 days 00:00:00.033333300    
    3             0 days 00:00:00.066666600    
    4             0 days 00:00:00.099999900    

    """

    # df = df.reset_index()

    time_at_frame_10 = df[df["frame_number"] == 10]["frame_time"].iloc[0]
    corrected_times = count_back(time_at_frame_10, 10, fps=30)

    df["corrected_frame_time"] = df["frame_time"]
    df = pd.merge(df, corrected_times, how='outer', on="frame_number")

    df["corrected_frame_time"] = df["corrected_frame_time_y"].fillna(df["corrected_frame_time_x"])
    df = df.drop(columns=["corrected_frame_time_y", "corrected_frame_time_x"])

    return df



