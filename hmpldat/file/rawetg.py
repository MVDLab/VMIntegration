"""Methods for opening exported RawETG files from BeGaze software

"""
from pathlib import Path
import logging

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def fix_task_name(task_name: str) -> str:
    """Reformat task string from annotation for search if necessary

    Args:
        task_name: a string

    Returns:
        corrected task string that will match dflow and cortex file names

    """

    if "int" in task_name:
        new_name = "int"
    elif "fix" in task_name:
        new_name = "fix"
    elif "qs" in task_name:
        if "ec" in task_name:
            new_name = "closed"
        elif "cross" in task_name:
            new_name = "cross"
        else:
            new_name = "open"
    elif "calib" in task_name:
        new_name = "ts"
    else:
        new_name = task_name

    LOG.debug("\n\n %s:", task_name)
    LOG.debug(new_name)

    return new_name

def frame_number_from_vidtime(df):
    """
    """

    # print(df)
    df['Video Time [h:m:s:ms]'] = df['Video Time [h:m:s:ms]'].fillna(method='ffill')
    # print(len(df))

    frame_number = df.groupby('Video Time [h:m:s:ms]').count()
    # print(frame_number)

    frame_number = frame_number.reset_index()

    frame_number = frame_number.loc[frame_number.index.repeat(frame_number['in_range'])]

    frame_number = frame_number.index.to_list()

    # print(len(frame_number))

    # frame_number = df.groupby('Video Time [h:m:s:ms]').ngroup()
    # print(frame_number, len(frame_number))

    # df['frame_number'] = frame_number

    return frame_number




def frame_number(df, fps=30.00003000003, how='ceiling'):
    """

    """
    # print(df)

    video_time_step_sec = 1 / fps

    zeroed = df["RecordingTime [ms]"] - df["RecordingTime [ms]"].iloc[0]

    zeroed.name = 'recording_time_from_zero'
    # zeroed = df.index - df.index[0]

    # convert timedelta to float
    frame_number_as_float = (zeroed / video_time_step_sec)
    frame_number_as_float.name = 'frame_number'
    # print(frame_number_as_float)

    if how == 'ceiling':
        frame_number = np.ceil(frame_number_as_float)
    elif how == 'round':
        frame_number = np.round(frame_number_as_float, decimals=0) # to int
    elif how == 'floor':
        frame_number = np.floor(frame_number_as_float)

    # print(frame_number)

    return frame_number


def get_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """Read annotation data from exported rawetg data.

    Ignores area of interest, "AOI", annotations

    Args:
        df (pd.DataFrame): raw etg as pd DataFrame

    Returns:
        A dataframe indexed by task name: start & end times

    """

    tasks = {}

    # columns are not hard coded, that what this mess is for.
    # TODO: find column number for recording time
    annot_col_num = 0
    cat_col_num = 0
    col_num = 0
    for col in df.columns:
        col_num += 1
        if "annotation" in col.lower():
            annot_col_num = col_num
        elif "category" in col.lower():
            cat_col_num = col_num

    for row in df[df["Annotation Name"] != "-"].itertuples():

        # annotation is not an AOI (area of interest)
        if "AOI" not in row[annot_col_num]:

            task = row[annot_col_num].lower()
            # print(task, row[1])

            if " " in task:
                task = task.replace(" ", "")

            if "end" in row[cat_col_num].lower():
                tasks[task].update({"end": row[1]})
            elif "start" in row[cat_col_num].lower():
                tasks[task] = {"start": row[1]}
            elif "instant" in row[cat_col_num].lower():
                if "end" in task:
                    tasks[task.split("_")[0]].update({"end": row[1]})
                elif "start" in task:
                    tasks[task.split("_")[0]] = {"start": row[1]}

    tasks = pd.DataFrame(tasks)

    # for task in tasks.index:
    #     print(task, fix_task_name(task))

    return tasks


def in_range(df: pd.DataFrame):
    """Is participant gaze within accurate bounds?

    Binocular PoR xrange [-218, 1178] & yrange [-130, 850]

    Args:
        raw (pd.DataFrame): rawETG as a dataframe

    Returns:
        A dataFrame (a single boolean column) representing in|out of range for the entire trial

    Notes:
        .. figure:: ../figures/etg_reliable_range.png
            :align: center
                        
            Method for deriving max ranges in pixels (irrelevant of head vector length)

    """

    # create a data frame to act as a mask
    mask = pd.DataFrame(False, columns=["mask"], index=df.index)

    mask[
        (
            df["Point of Regard Binocular X [px]"].between(-218, 1178)
            & df["Point of Regard Binocular Y [px]"].between(-130, 850)
        )
    ] = True

    return mask


def in_task(df: pd.DataFrame):
    """According to annotations is this row within task?

    Args:
        df (pd.DataFrame): rawETG data

    Returns:
        A mask (a single boolean column) representing in task vs. out of task
    
    """

    # get tasks
    tasks = get_tasks(df)
    # print(tasks)
    # print(df['RecordingTime [ms]'])

    # create a data frame to act as a mask
    mask = pd.DataFrame(False, columns=["mask"], index=df.index)

    # build mask
    for task in tasks:
        mask[
            ((tasks[task]["start"] <= df['RecordingTime [ms]']) & (tasks[task]["end"] >= df['RecordingTime [ms]']))
        ] = True

    return mask


def open(path: Path) -> pd.DataFrame:
    """Open rawETG data exported from BeGaze

    Args:
        path (Path): path to rawETG data file
            
    Returns:
        A pandas dataframe with columns

        .. code-block:: python
        
            RecordingTime [ms], Video Time [h:m:s:ms],
            Tracking Ratio [%],
            Category Binocular,
            Pupil Diameter Right [mm], Pupil Diameter Left [mm],
            Point of Regard Binocular X [px], Point of Regard Binocular Y [px],
            Annotation Name,

    """
    LOG.info("reading rawetg file: %s", str(path))

    # this defines columns we want and converts stuff if necessary
    converters = {
        "RecordingTime [ms]": None,
        "Video Time [h:m:s:ms]": lambda t: pd.NaT
        if "-" in t
        else ".".join(t.rsplit(":", 1)),
        "Tracking Ratio [%]": lambda p: np.NaN if "-" in p else float(p),
        "Category Binocular": lambda s: "-" if None  else str(s),
        "Pupil Diameter Right [mm]": lambda r: np.NaN if "-" in r else float(r),
        "Pupil Diameter Left [mm]": lambda l: np.NaN if "-" in l else float(l),
        "Point of Regard Binocular X [px]": lambda x: np.NaN if x == "-" else float(x),
        "Point of Regard Binocular Y [px]": lambda y: np.NaN if y == "-" else float(y),
        "Annotation Name": lambda s: "-" if None else str(s),
    }

    dtype = {"Category Binocular": str, "Annotation Name": str}

    # read from file
    rawetg_df = pd.read_csv(
        path, sep="\t", usecols=converters.keys(), converters=converters
    )

    # The rows with annotation do not contain a "Video Time [h:m:s:ms]" entry
    # backfill this column (there doesn't seem to be any gaps longer than 2 rows)
    # rawetg_df["Video Time [h:m:s:ms]"] = rawetg_df["Video Time [h:m:s:ms]"].fillna(
    #     method="bfill", limit=2
    # )

    rawetg_df["RecordingTime [ms]"] = pd.to_timedelta(
        rawetg_df["RecordingTime [ms]"], unit="ms"
    )

    # start recording time at 0
    # rawetg_df["rawetg_recording_time_from_zero"] = (
    #     rawetg_df["RecordingTime [ms]"] - rawetg_df["RecordingTime [ms]"].iat[0]
    # )

    # convert recording time into timedeltaindex
    # rawetg_df.set_index('RecordingTime [ms]', inplace=True)

    # add new column to represent whether data is in task (according to annot) and in range
    # rawetg_df = rawetg_df.assign(in_task=in_task, in_range=in_range)
    # print(rawetg_df[['in_task', 'in_range']].dtypes)

    # print(rawetg_df['Video Time [h:m:s:ms]'][(rawetg_df.index > pd.Timedelta('01:21:30')) & (rawetg_df.index < pd.Timedelta('01:21:35'))].to_string())
    # input()

    # convert video time to a time delta
    # rawetg_df["Video Time [h:m:s:ms]"] = pd.to_timedelta(
    #     rawetg_df["Video Time [h:m:s:ms]"]
    # )

    # print(round(rawetg_df['Video Time [h:m:s:ms]'] / pd.to_timedelta(1000/30.0003000003, unit='ms')))

    # if LOG.isEnabledFor(logging.DEBUG):
    #     LOG.debug('file as read into memory:\n%s', rawetg_df.to_string(max_rows=20, line_width=LINE_WIDTH))
    #     LOG.debug("last modified: %s", time.ctime(os.path.getctime(path)))
    #     # LOG.debug('pandas DataFrame.info(): \n%s', rawetg_df.info())

    return rawetg_df



def probe_elapsed_time(path: Path) -> pd.Timedelta:
    """get elapsed time from rawETG file

    Args:
        path: pathobject to rawetg file
    
    Returns:
        elapsed time as a timedelta

    """

    LOG.info("probe duration for: %s" % str(path))

    try:
        df = pd.read_csv(path, sep="\t", usecols=["RecordingTime [ms]"])
    except pd.errors.ParserError:
        return pd.NaT

    if df.size > 1:
        return pd.Timedelta(
            df["RecordingTime [ms]"].iloc[-1] - df["RecordingTime [ms]"].iloc[0],
            unit="ms",
        )
    else:
        LOG.error("unable to calc elapsed time :(")


def quadrant(Binoc_PoR, origin={"x": 480, "y": 360}):
    """what quadrant is this point in?

    Args:
        points: a dataframe containing rawetg data (columns `Point of Regard Binocular X|Y [px]` )
        origin: default origin of the image (360, 480)
    
    Returns:
        quadrant number or np.NaN (if no gaze loc) 

    """

    # if any(np.isnan(val) for val in Binoc_PoR):
    #     quadrant = None
    # # else:
    # if Binoc_PoR['x'] < origin['x'] and Binoc_PoR['y'] < origin['y']:
    #     quadrant = 2
    # if Binoc_PoR['x'] > origin['x'] and Binoc_PoR['y'] < origin['y']:
    #     quadrant = 1
    # elif Binoc_PoR['x'] > origin['x'] and Binoc_PoR['y'] > origin['y']:
    #     quadrant = 4
    # elif Binoc_PoR['x'] < origin['x'] and Binoc_PoR['y'] > origin['y']:
    #     quadrant = 3

    # right of vertical axis
    if Binoc_PoR["x"] > origin["x"]:
        # Q I
        if Binoc_PoR["y"] > origin["y"]:
            quadrant = 1

        # Q IV
        elif Binoc_PoR["y"] < origin["y"]:
            quadrant = 4

        # centered on horizontal axis
        # techinacally not in a quadrant
        else:
            quadrant = 1

    # left of vertical axis
    elif Binoc_PoR["x"] < origin["x"]:
        # Q II
        if Binoc_PoR["y"] > origin["y"]:
            quadrant = 2

        # Q III
        elif Binoc_PoR["y"] < origin["y"]:
            quadrant = 3

        # centered on horizontal axis
        # techinacally not in a quadrant
        else:
            quadrant = 2

    # centered on vertical axis
    else:
        # techinacally not in a quadrant

        if Binoc_PoR["y"] > origin["y"]:
            quadrant = 1

        elif Binoc_PoR["y"] < origin["y"]:
            quadrant = 4

        else:
            quadrant = 1

    return quadrant
