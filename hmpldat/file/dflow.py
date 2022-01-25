"""Methods for opening dflow files

"""
from pathlib import Path
from pprint import pprint
import re
import logging

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


SCREEN_RADIUS = 2490.0
# SCREEN_RADIUS = 2.49



def mc_open(path: Path, mocap=False):
    """Open dflow mc file

    Args:
        path: path object to file
        mocap: default = False 
            read only force plate data from this file
            if true also returns the mocap data 

    Returns: 
        A pandas dataframe, with columns (mocap=False)
        
        .. code-block:: python

            Time,
            FP1.CopX, FP1.CopY, FP1.CopZ,
            FP1.ForX, FP1.ForY, FP1.ForZ,
            FP1.MomX, FP1.MomY, FP1.MomZ,
            FP2.CopX, FP2.CopY, FP2.CopZ,
            FP2.ForX, FP2.ForY, FP2.ForZ,
            FP2.MomX, FP2.MomY, FP2.MomZ,
            Channel16.Anlg

    """
    LOG.info("reading dflow mc file: %s", str(path))

    if not mocap:
        # we aren't using this mocap data, but we need the force plate data
        # No conversion necessary, just trying to keep everything consistent
        converters = {
            "TimeStamp": None,
            "FP1.CopX": None,
            "FP1.CopY": None,
            "FP1.CopZ": None,
            "FP1.ForX": None,
            "FP1.ForY": None,
            "FP1.ForZ": None,
            "FP1.MomX": None,
            "FP1.MomY": None,
            "FP1.MomZ": None,
            "FP2.CopX": None,
            "FP2.CopY": None,
            "FP2.CopZ": None,
            "FP2.ForX": None,
            "FP2.ForY": None,
            "FP2.ForZ": None,
            "FP2.MomX": None,
            "FP2.MomY": None,
            "FP2.MomZ": None,
            "Channel16.Anlg": None,
        }

        # read from file (ignoring columns we don't need)
        dflow_mc_df = pd.read_csv(
            path, sep="\t", usecols=converters.keys(), comment="#",
        )
    else:
        # read all columns
        dflow_mc_df = pd.read_csv(path, sep="\t", comment="#")

        # TODO: convert meters to mm in this motion capture data

    # convert TimeStamp into timedeltaindex(I'm using this index to align)
    dflow_mc_df["TimeStamp"] = pd.to_timedelta(dflow_mc_df["TimeStamp"], unit="s")
    dflow_mc_df.set_index("TimeStamp", inplace=True)

    # TODO: convert Channel16 to boolean

    # change name to match dflow_rd and cortex
    dflow_mc_df.index.name = "dflow_mc_time"

    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(
            "file as read into memory:\n%s",
            dflow_mc_df.to_string(max_rows=20, line_width=200),
        )
        # LOG.debug("last modified: %s", time.ctime(os.path.getctime(path)))

        # LOG.debug('pandas DataFrame.info(): \n%s', dflow_mc_df.info(verbose=True))

    return dflow_mc_df


def rd_open(path: str):
    """Open dflow rd file

    Todo: convert dflow-rd Pos from meter to millimeters

    Returns: 
        A pandas dataframe indexed by timedelta, with columns
        
        .. code-block:: python

            Tasknr, 
            CrossVisible.Bool, Cross.PosX, Cross.PosY, Cross.PosZ, 
            TargetVisible.Bool, Target.PosX, Target.PosY, Target.PosZ, 
            UserVisible.Bool, User.PosX, User.PosY, User.PosZ, 
            SafezoneVisible.Bool, Safezone.PosX, Safezone.PosY, Safezone.PosZ,
            GridVisible.Bool, Grid.PosX, Grid.PosY, Grid.PosZ, Grid.RotX, 
            RandomIndexNr, RNP

    Note:
        returns all columns from file

    """
    LOG.info(" Reading dflow rd (vr) file: %s", path)

    # convert meters to mm
    converters = {
        "Time": None,
        "Tasknr": None,
        "CrossVisible.Bool": None,
        "Cross.PosX": lambda n: n * 1000,
        "Cross.PosY": lambda n: n * 1000,
        "Cross.PosZ": lambda n: n * 1000,
        "TargetVisible.Bool": None,
        "Target.PosX": lambda n: n * 1000,
        "Target.PosY": lambda n: n * 1000,
        "Target.PosZ": lambda n: n * 1000,
        "UserVisible.Bool": None,
        "User.PosX": lambda n: n * 1000,
        "User.PosY": lambda n: n * 1000,
        "User.PosZ": lambda n: n * 1000,
        "SafezoneVisible.Bool": None,
        "Safezone.PosX": lambda n: n * 1000,
        "Safezone.PosY": lambda n: n * 1000,
        "Safezone.PosZ": lambda n: n * 1000,
        "GridVisible.Bool": None,
        "Grid.PosX": lambda n: n * 1000,
        "Grid.PosY": lambda n: n * 1000,
        "Grid.PosZ": lambda n: n * 1000,
        "Grid.RotX": None,  # TODO: figure out how to interpret this
        "RandomIndexNr": None,
        "RNP": None,
    }

    # read from file
    # dflow_rd_df = pd.read_csv(path, sep="\t", converters=converters, comment="#")
    dflow_rd_df = pd.read_csv(path, sep="\t", comment="#")

    # convert Time into timedeltaindex (I'm using this index to align)
    # dflow_rd_df["Time"] = pd.to_timedelta(dflow_rd_df["Time"], unit="s")
    # dflow_rd_df.set_index("Time", inplace=True)

    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(
            "file as read into memory:\n%s",
            dflow_rd_df.to_string(max_rows=20, line_width=200),
        )
        # LOG.debug("last modified: %s", time.ctime(os.path.getctime(path)))
        # LOG.debug('pandas DataFrame.info(): \n%s', dflow_rd_df.info(verbose=True))

    return dflow_rd_df

# TODO: look for 5V pulse
def probe_elapsed_time(path: Path, start_and_end=False):
    """get elapsed time from dflow rd or mc file

    Args:
        path: pathobject to dflow file
    
    Returns:
        elapsed time as a timedelta

    """

    LOG.info("probing duration for: %s", str(path))

    time_column_name = None

    if "rd" in path.name:
        time_column_name = "Time"
    elif "mc" in path.name:
        time_column_name = "TimeStamp"

    if time_column_name is None:
        raise ValueError("this ain't a dflow file (or it is improperly named)")

    try:
        df = pd.read_csv(path, sep="\t", usecols=[time_column_name], comment="#")
    except:
        print("ERROR: unable to find time column in file: %s" % path)
        return pd.NaT

    # if the data is not read as numbers then there was a problem reading the file
    if df[time_column_name].dtypes != np.float:
        return pd.NaT

    start_time = pd.Timedelta(df[time_column_name].iloc[0], unit="s")
    end_time = pd.Timedelta(df[time_column_name].iloc[-1], unit="s")

    if start_and_end:
        return {"start": start_time, "end": end_time, "duration": end_time - start_time}
    else:
        return end_time - start_time


# def not_centered(df, dist):
#     """
#     return target objects that are not centered in the frame

#     Args:
#         df: dataframe with object positions
#         dist: distance from center mm

#     Returns:
#         df

#     """

#     df = df["target"]["x"]
   

def project_object_onto_screen(x_pos, y_pos, z_pos, y_view):
    """Uses the position of the object in the VR world (dflow_rd data) to find projection onto the screen.

    Args:
        [x,y,z]_pos: position of an object in mocap space
        y_view: y height for viewpoint (calculated from cross position)

    Returns:
        [x,y,z]_proj: projection of object onto to screen

    Notes:
        Todo: see `helpful_documents/project_object_onto_screen.pdf`

    """

    # object exists on viewpoint.
    # this should never occur -> return np.NaN
    if x_pos == z_pos == 0.0:
        return np.NaN, np.NaN, np.NaN

    # True, but the else statement would also handle this case
    elif x_pos == 0.0:
        x_proj = 0
        z_proj = -SCREEN_RADIUS  # mm

    # this never happens
    if z_pos == 0.0:
        z_proj = 0
        x_proj = SCREEN_RADIUS  # mm

        # when x_position is negative projection should be negative
        if x_pos < 0:
            x_proj = -x_proj

    # the normal case
    else:
        z_proj = SCREEN_RADIUS / np.sqrt((x_pos / z_pos) ** 2 + 1)

        # when z_pos is negative projection should also be negative
        if z_pos < 0:
            z_proj = -z_proj

        x_proj = np.sqrt(SCREEN_RADIUS ** 2 - z_proj ** 2)

        # when x_position is negative projection should be negative
        if x_pos < 0:
            x_proj = -x_proj

    y_proj = (SCREEN_RADIUS * (y_pos - y_view)) / np.sqrt(x_pos ** 2 + z_pos ** 2) + y_view

    return x_proj, y_proj, z_proj


def get_task_name(path: Path):
    """return task accoring to file name"""

    # return none if regex fails
    task_name = None

    match = re.search(r".*?([a-z]+)_[a-z]{2}([\d]{4}).*?", path.name.lower())
    if match:
        task_name = "_".join([match.group(1), match.group(2).lstrip("0")])

    return task_name


def reformat(df):
    """
    transform dflow rd data to use multi-indexed columns

    Args:

    Returns:

    """

    objs_pos = df.loc[:, df.columns.str.contains(r"\.pos|\.rot", case=False)]
    objs_visible = df.loc[:, df.columns.str.contains(r"visible\.bool", case=False)]

    # rename columns to match
    objs_pos.columns = [c.lower().replace(".pos", ".") for c in objs_pos.columns]
    objs_visible.columns = [
        c.lower().replace("visible.bool", ".visible") for c in objs_visible.columns
    ]

    # join object info together
    objs = objs_pos.join(objs_visible)

    # multi-index
    objs.columns = pd.MultiIndex.from_tuples(c.split(".") for c in objs.columns)

    # organize 
    objs = objs.sort_index(axis="columns", level=0)

    return objs
