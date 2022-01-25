"""Define camera pose 

"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm

import hmpldat.file.search as search
from hmpldat.utils.math import euclidean_distance
import hmpldat.utils.plot


def define(lhead, rhead, fhead, filename):
    """
    Define eyetracking glasses in camera space
    
    Args:
        lhead: m by n array of m observations and n dimensions
        rhead: m by n array of m observations and n dimensions
        fhead: m by n array of m observations and n dimensions

    Returns:
        lhead, rhead, fhead in camera (etg) space

    Note:
        TODO: use this distance method in other places

        .. figure helpful_documents/etg_model.pdf

    """

    # TODO: add check array shapes are the same

    # try to do it with matricies if you have enough ram
    # else do it row-wise
    try:
        d = np.mean(np.diagonal(cdist(lhead, rhead, "euclidean")))
        c = np.mean(np.diagonal(cdist(lhead, fhead, "euclidean")))
        e = np.mean(np.diagonal(cdist(fhead, rhead, "euclidean")))
    except MemoryError as err:

        print(f"\n\t{err}\n\n\tContinuing with row-wise operation\n")

        d = []
        c = []
        e = []

        for l, r, f in tqdm(zip(lhead.values, rhead.values, fhead.values)):

            d.append(euclidean_distance(l, r))
            c.append(euclidean_distance(l, f))
            e.append(euclidean_distance(f, r))

        d = np.mean(d)
        c = np.mean(c)
        e = np.mean(e)

        # d = np.mean([euclidean_distance(u, v) for u, v in ])

    a = (c ** 2 + d ** 2 - e ** 2) / (2 * d)
    b = (c ** 2 - a ** 2) ** 0.5

    # print(f"d={d}\ta={a}\tb={b}\tc={c}\te={e}")

    lhead = [0, 0, 0]
    rhead = [d, 0, 0]
    fhead = [a, b, 0]

    # creation date
    date = [datetime.now()]

    model = pd.DataFrame(
        date + lhead + rhead + fhead,
        index=[
            "creation_date",
            "lhead.x",
            "lhead.y",
            "lhead.z",
            "rhead.x",
            "rhead.y",
            "rhead.z",
            "fhead.x",
            "fhead.y",
            "fhead.z",
        ],
        columns=[filename],
    ).T

    return model


def save_camera_model(model, path):
    """
    Append camera model to csv
    
    """

    # if this file does not exist, create it
    if not path.exists():
        model.to_csv(path, index_label="filename", mode="w")
    else:
        model.to_csv(path, mode="a", header=False)


def open(path):
    """
    opens

    """

    df = pd.read_csv(path, index_col=0)

    return df


def multiindex(model):
    """ 
    given a single camera model reformat for future calculations

    index = multi-index (filename, [x,y,z])
    attributes = [lhead, rhead, fhead]

    Returns:
        model (dataframe)
        fname (str)
        creation_date (str)

    Note:

    """

    fname = model.name
    creation = model.pop("creation_date")

    model.index = pd.MultiIndex.from_tuples(
        list(tuple(s.split(".")) for s in model.index)
    )
    model = pd.DataFrame(model).T

    return model, fname, creation


def best_guess(model):
    """
    given a camera model, initialize points [camera, cam_vector_origin, left_vector_origin]

    """
    print(model)

    camera_vector_origin = (model["lhead"] + model["rhead"]) / 2
    camera_vector_origin.index = ["camera_vector_origin"]

    camera = camera_vector_origin.copy()
    camera["y"] = camera["y"] + model[("fhead", "y")].values  # mm
    camera["z"] = model[("fhead", "z")].values  # 0 by our definition of camera model
    camera.index = ["camera"]

    left_vector_origin = camera_vector_origin.copy()
    left_vector_origin["x"] += camera["y"].values * np.tan(np.pi / 6)
    left_vector_origin.index = ["left_vector_origin"]

    best_guess = pd.concat([camera, camera_vector_origin, left_vector_origin]).T

    print(best_guess.T.to_markdown())
    return best_guess


def median_camera_point(lhead, rhead, fhead):
    """
    what is my median camera point in mocap space


    """

    # reset column names for clarity ("LHEAD.X"  ->  "x")
    lhead.columns = [c.split(".")[1].lower() for c in lhead.columns]
    rhead.columns = [c.split(".")[1].lower() for c in rhead.columns]
    fhead.columns = [c.split(".")[1].lower() for c in fhead.columns]

    median_lhead = lhead.median()
    median_rhead = rhead.median()
    median_fhead = fhead.median()

    # name each series (will become column names when concatinating before plot)
    median_lhead.name = "median_lhead"
    median_rhead.name = "median_rhead"
    median_fhead.name = "median_fhead"

    # calculate "median" camera position
    median_camera_x = (median_lhead["x"] + median_rhead["x"]) / 2
    median_camera_y = (median_lhead["y"] + median_rhead["y"]) / 2
    median_camera_z = median_fhead["z"]

    median_camera = pd.Series(
        [median_camera_x, median_camera_y, median_camera_z],
        index=["x", "y", "z"],
        name="median_cam",
    )

    # calculate median horizontal-forward (-z) camera vector
    median_camera_vector = median_camera.copy()
    median_camera_vector["z"] = median_camera_vector["z"] - 1
    median_camera_vector.name = "median_cam_vec"

    # calculate median horizontal-forward LEFT camera vector
    median_left_camera_vector = median_camera_vector.copy()
    median_left_camera_vector["x"] = median_left_camera_vector["x"] - np.tan(np.pi / 6)
    median_left_camera_vector.name = "median_left_cam_vec"

    df = pd.concat(
        [
            median_lhead,
            median_rhead,
            median_fhead,
            median_camera,
            median_camera_vector,
            median_left_camera_vector,
        ],
        axis=1,
    )
    print(df.to_markdown())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    hmpldat.utils.plot.glasses(df, ax)

    ax.legend()
    plt.suptitle('median glasses points and "virtual median camera points"\n' + f.name)

    plt.show()


if __name__ == "__main__":

    for f in search.files(
        Path("/home/raphy/proj/hmpldat/sample_datas/merged/all_tasks_from_one_session"),
        [],
    ):

        # info[f.name] = {}

        print(f)

        df = pd.read_csv(f, low_memory=False)

        # TODO: create new sample merged data, then remove this
        df[df.filter(like=".Pos").columns] = (
            df.filter(like=".Pos") * 1000
        )  ## convert m to mm
        # print(df.filter(like="Cross.Pos"))

        df = df.dropna(subset=["Frame#"])
        # x = df.dropna(subset=[df.filter(regex=r"[FLR]{1}HEAD.[XYZ]{1}")])

        median_camera_point(
            df[["LHEAD.X", "LHEAD.Y", "LHEAD.Z"]],
            df[["RHEAD.X", "RHEAD.Y", "RHEAD.Z"]],
            df[["FHEAD.X", "FHEAD.Y", "FHEAD.Z"]],
        )

        # TODO: remove, Not necessary for this op?
        # df["time_mc_adj"] = pd.to_timedelta(df["time_mc_adj"])
        # y_view = df["RHEAD.Y"].dropna().values[0]

        # drop instances with no mocap data (cortex typically starts recording shortly after dflow)
        # df = df.dropna(subset=["Frame#"])

        # df = df.dropna(subset=[df.filter(regex=r"[FLR]{1}HEAD.[XYZ]{1}")])

        # df = df.filter(regex=r"[FLR]{1}HEAD.[XYZ]{1}")
        # # print(df)

        # df = df.dropna()
        # # print(df.isna().sum())

        # model = define(
        #     df[["LHEAD.X", "LHEAD.Y", "LHEAD.Z"]],
        #     df[["RHEAD.X", "RHEAD.Y", "RHEAD.Z"]],
        #     df[["FHEAD.X", "FHEAD.Y", "FHEAD.Z"]],
        #     f.name,
        # )

        # save_camera_model(model, Path("test_camera_model.csv"))
