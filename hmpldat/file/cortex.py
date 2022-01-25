"""Methods for opening cortex files

"""

import logging
from pathlib import Path
from pprint import pprint
import re

import pandas as pd

LOG = logging.getLogger(__name__)


def open(path: Path):
    """Open a cortex motion capture file

    Restructures columns to use a single index

    Returns:
        A pandas dataframe index by timedelta with columns

        .. code-block:: python

            Frame#,
            FHEAD.X, FHEAD.Y, FHEAD.Z,
            RHEAD.X, RHEAD.Y, RHEAD.Z,
            THEAD.X, THEAD.Y, THEAD.Z,
            LHEAD.X, LHEAD.Y, LHEAD.Z,
            C7.X, C7.Y, C7.Z,
            BBAC.X, BBAC.Y, BBAC.Z, 
            Offset_Nav.X, Offset_Nav.Y,Offset_Nav.Z,
            XYPH.X, XYPH.Y, XYPH.Z,
            STRN.X, STRN.Y, STRN.Z,
            LSHO.X, LSHO.Y, LSHO.Z,
            RSHO.X, RSHO.Y, RSHO.Z,
            LASIS.X, LASIS.Y, LASIS.Z,
            LPSIS.X, LPSIS.Y, LPSIS.Z,
            RASIS.X, RASIS.Y, RASIS.Z,
            RPSIS.X, RPSIS.Y, RPSIS.Z,
            SACRUM.X, SACRUM.Y, SACRUM.Z,
            RLM.X, RLM.Y, RLM.Z,
            RHEE.X, RHEE.Y, RHEE.Z,
            RTOE.X, RTOE.Y, RTOE.Z,
            RMT5.X, RMT5.Y, RMT5.Z,
            LHEE.X, LHEE.Y, LHEE.Z,
            LTOE.X, LTOE.Y, LTOE.Z,
            LMT5.X, LMT5.Y, LMT5.Z,
            V_RShoulder.X, V_RShoulder.Y, V_RShoulder.Z,
            V_LShoulder.X, V_LShoulder.Y, V_LShoulder.Z,
            V_Neck.X, V_Neck.Y, V_Neck.Z,
            V_LShoulder_Dyn.X, V_LShoulder_Dyn.Y, V_LShoulder_Dyn.Z,
            V_RShoulder_Dyn.X, V_RShoulder_Dyn.Y, V_RShoulder_Dyn.Z,
            V_Mid_ASIS_Dyn.X, V_Mid_ASIS_Dyn.Y, V_Mid_ASIS_Dyn.Z,
            V_Head.X, V_Head.Y, V_Head.Z,
            V_Mid_PSIS.X, V_Mid_PSIS.Y, V_Mid_PSIS.Z,
            V_RFoot.X, V_RFoot.Y, V_RFoot.Z,
            V_LFoot.X, V_LFoot.Y, V_LFoot.Z,
            V_RAnkle_Dyn.X, V_RAnkle_Dyn.Y, V_RAnkle_Dyn.Z,
            V_Neck2.X, V_Neck2.Y, V_Neck2.Z,
            V_Pelvis.X, V_Pelvis.Y, V_Pelvis.Z,
            V_Sacrum.X, V_Sacrum.Y, V_Sacrum.Z

    Note: 
        may recieve warnings due to extra data existing to the right of the recorded data, safe to ignore

    """
    LOG.info("reading cortex trc file: %s", str(path))

    # first read only the column names
    try:
        cols = pd.read_csv(
            path, sep="\t", header=[3, 4], nrows=0, error_bad_lines=False
        )
    except (pd.errors.ParserError):
        cols = pd.read_csv(
            path,
            sep="\t",
            header=[3, 4],
            nrows=0,
            engine="python",
            error_bad_lines=False,
        )

    # LOG.debug("%s", cols.to_string())

    # edit column names to make column names into single level index similar to dflow data
    # I am droping the number value associated with each marker (We need only X, Y, Z)
    new_cols = []
    a_col_name = ""
    for col in cols:
        if "Unnamed" in col[0]:
            if "Unnamed" not in col[1]:
                new_cols.append(a_col_name + "." + col[1][0])
            else:
                pass
        else:
            if "Unnamed" in col[1]:
                new_cols.append((col[0]))
            else:
                new_cols.append((col[0] + "." + col[1][0]))
            a_col_name = col[0]

    # LOG.debug("\n%s\ntotal=%d", '\n'.join(f"\t{s}" for s in new_cols), len(new_cols))

    # read the data but use my edited column names
    try:
        _df = pd.read_csv(
            path,
            sep="\t",
            names=new_cols,
            usecols=new_cols,
            header=None,
            skiprows=[0, 1, 2, 3, 4, 5],
            index_col=False,
            skip_blank_lines=False,
            error_bad_lines=False,
        )
    except (pd.errors.ParserError):
        _df = pd.read_csv(
            path,
            sep="\t",
            names=new_cols,
            usecols=new_cols,
            header=None,
            skiprows=[0, 1, 2, 3, 4, 5],
            index_col=False,
            skip_blank_lines=False,
            error_bad_lines=False,
            engine="python",
        )

    # convert to a timedelta index
    _df["Time"] = pd.to_timedelta(_df["Time"], unit="s")
    _df.set_index("Time", inplace=True)

    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(
            "file as read into memory:\n%s", _df.to_string(max_rows=20, line_width=200)
        )
        # LOG.debug("last modified: %s", time.ctime(os.path.getctime(path)))

        # LOG.debug('pandas DataFrame.info(): \n%s', _df.info(verbose=True))

    return _df


def probe_elapsed_time(path: Path):
    """get elapsed time from cortex file

    Args:
        path: pathobject to cortex file
    
    Returns:
        elapsed time as a timedelta

    """
    LOG.info("probing duration for: %s", str(path))

    try:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["Frame#", "Time"],
            usecols=["Frame#", "Time"],
            skiprows=[0, 1, 2, 3, 4, 5],
            skip_blank_lines=False,
            error_bad_lines=False,
        )
    except (pd.errors.ParserError):
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["Frame#", "Time"],
            usecols=["Frame#", "Time"],
            engine="python",
            skiprows=[0, 1, 2, 3, 4, 5],
            skip_blank_lines=False,
            error_bad_lines=False,
        )

    if df["Time"].size > 1:
        return pd.Timedelta(df["Time"].iloc[-1] - df["Time"].iloc[0], unit="s")
    else:
        LOG.error("unable to parse elapsed time")
        return pd.NaT


def get_task_name(path: Path):
    """return task according to file name"""

    # return none if regex fails
    task_name = None

    match = re.search(r".*?_([a-z]+)\.?(\d{1,2}).*?", path.name.lower())
    if match:
        task_name = "_".join(match.groups())

    return task_name


def multiindex(df):
    """ 
    transform cortex mocap data to use multi-indexed columns

    Returns:

    """

    # grab columns
    mocap = df.loc[:, df.columns.str.contains(r"\.[x,y,z]{1}", case=False)]

    # multi-index
    mocap.columns = pd.MultiIndex.from_tuples(
        c.lower().split(".") for c in mocap.columns
    )

    return mocap
