"""
Object to represent eyetracking glasses model.

TODO: add methods to save camera model (currently only etg glasses information) to file, with creation date
TODO: mark/make changes required to handle arbitrary number of points

17 April 2020
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from hmpldat.utils.math import (
    euclidean_distance,
    rigid_transform_3D,
    ray_plane_intersect,
    vectorized_euclidean_dist,
)
import hmpldat.file.search as search
import hmpldat.file.cortex
import hmpldat.utils.plot


class etg_model:
    def __init__(self):

        # median in mocap space 2d pandas dataframe index = [x,y,z] columns=each point
        self.median = None

        # in glasses space 2d pandas dataframe index = [x,y,z] columns=each point
        self.in_glasses_space = None

        # TODO: record camera model as part of this object?
        # self.camera_model= None

        self.from_file = None  # filename string
        self.creation_date = None  # datetime object

    def new_define(self, df, filename):
        """from a set of glasses marker points df

        Note:
            df should have multi-indexed columns names: [(markerA, x), ...]
            centroid is placed at the origin

        """

        self.creation_date = datetime.now()
        self.from_file = filename

        # mean across each set of x, y, z columns provided in input dataframe
        centroid = df.mean(axis=1, level=1)

        # subtract centroid from each glasses marker
        with_centroid_as_origin = df.subtract(centroid, level=1)

        # calculate median across all instances
        # unstack(0) -> 2D array columns=marker_names; rows=[x,y,z]
        median_glasses_model = with_centroid_as_origin.median(axis=0).unstack(0)

        self.in_glasses_space = median_glasses_model

    def define(self, lhead, rhead, fhead, filename):
        """
        glasses in glasses space and find median mocap values

        from a set of instances lhead, rhead, fhead

        .. figure helpful_documents/etg_model.pdf
        """

        self.creation_date = datetime.now()
        self.from_file = filename

        ### DEFINE GLASSES MODEL IN "GLASSES" SPACE
        # try to do it with matrices, if you have enough ram
        # else do it row-wise
        try:
            d = np.mean(np.diagonal(cdist(lhead, rhead, "euclidean")))
            c = np.mean(np.diagonal(cdist(lhead, fhead, "euclidean")))
            e = np.mean(np.diagonal(cdist(fhead, rhead, "euclidean")))

            # TODO : implement this here (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5

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

        # TODO: assert d = d and check sklearn pairwise distance method

        a = (c ** 2 + d ** 2 - e ** 2) / (2 * d)
        b = (c ** 2 - a ** 2) ** 0.5
        # print(f"d={d}\ta={a}\tb={b}\tc={c}\te={e}")

        gs_lhead = [0, 0, 0]
        gs_rhead = [d, 0, 0]
        gs_fhead = [a, b, 0]

        self.in_glasses_space = pd.DataFrame(
            zip(gs_lhead, gs_rhead, gs_fhead),
            columns=["lhead", "rhead", "fhead"],
            index=["x", "y", "z"],
        )

        # print(self.in_glasses_space)

        ### DEFINE MEDIAN GLASSES POSITION IN MOCAP SPACE
        median_lhead = lhead.median()
        median_rhead = rhead.median()
        median_fhead = fhead.median()

        self.median = pd.concat([median_lhead, median_rhead, median_fhead], axis=1)
        self.median.columns = ["lhead", "rhead", "fhead"]

    def load(self, filepath, modelname):
        """ read from file 
        
        TODO: handle multiple models for smae filename: allow user to choose, sort by date

        Returns:
            True, on success
        """

        success = False

        if not filepath.exists():
            return success

        try:
            df = pd.read_csv(filepath)

            # select row reprenting specified filename
            select = df[df["filename"] == modelname]

            # TODO: if more than one model have user choose
            if select.shape[0] > 1:
                raise NotImplementedError(
                    f"multiple models built for `{filepath}` please delete old for now."
                )

            if select.shape[0] == 0:
                raise KeyError

            self.from_file = select["filename"]
            self.creation_date = select["created"]

            # create multi-index columns and unflatten data
            g_columns = pd.MultiIndex.from_tuples(
                select.filter(regex=r"glasses_space.*").columns.str.split(".").to_list()
            )
            m_columns = pd.MultiIndex.from_tuples(
                select.filter(regex=r"median.*").columns.str.split(".").to_list()
            )

            gs = select.filter(regex=r"glasses_space.*")
            gs.columns = g_columns.droplevel()
            gs = gs.stack().reset_index(level=0, drop=True)

            med = select.filter(regex=r"median.*")
            med.columns = m_columns.droplevel()
            med = med.stack().reset_index(level=0, drop=True)

            self.in_glasses_space = gs
            self.median = med

            success = True

        except KeyError:
            print(
                f"\n\tERROR: etg model not found for file: {modelname} in {filepath.name}\n"
            )

        return success

    def cam_best_guess_orig(self):
        """ original methods for devising best guess in camera space using avg distance between markers"""

        camera_vector_origin = pd.Series(
            (self.in_glasses_space["lhead"] + self.in_glasses_space["rhead"]) / 2,
            name="camera_vector_origin",
        )

        camera = pd.Series(
            {
                "x": camera_vector_origin["x"],
                "y": camera_vector_origin["y"] + self.in_glasses_space["fhead"]["y"],
                "z": self.in_glasses_space["fhead"]["z"],
            },
            name="camera",
        )

        left_vector_origin = pd.Series(
            {
                "x": camera_vector_origin["x"] + camera["y"] * np.tan(np.pi / 6),
                "y": camera_vector_origin["y"],
                "z": camera_vector_origin["z"],
            },
            name="left_vector_origin",
        )

        top_vector_origin = pd.Series(
            {
                "x": camera_vector_origin["x"],
                "y": camera_vector_origin["y"],
                "z": camera_vector_origin["z"] - camera["y"] * np.tan(23 * np.pi / 180),
            },
            name="top_vector_origin",
        )

        # merge into a single dataframe series names become column names
        best_guess = pd.concat(
            [camera, camera_vector_origin, left_vector_origin, top_vector_origin],
            axis=1,
        )

        return best_guess

    def cam_best_guess_pass_thru(self):
        """best guess camera model with pass through points"""

        cam = pd.Series({"x": 0, "y": 0, "z": 0}, name="cam")

        cam_vec_pass_thru = pd.Series(
            {"x": 0, "y": 0, "z": -1000}, name="cam_vec_pass_thru"
        )

        left_vec_pass_thru = pd.Series(
            {"x": -1000 * np.tan(np.pi / 6), "y": 0, "z": -1000},
            name="left_vec_pass_thru",
        )

        top_vec_pass_thru = pd.Series(
            {"x": 0, "y": 1000 * np.tan((23 * np.pi) / 180), "z": -1000},
            name="top_vec_pass_thru",
        )

        # merge into a single dataframe series names become column names
        best_guess = pd.concat(
            [cam, cam_vec_pass_thru, left_vec_pass_thru, top_vec_pass_thru], axis=1,
        )

        return best_guess

    def cam_best_guess_ray(self):
        """ use median values and optimal rotation to etg model in glasses space to define best guess in camera space """

        self.median = self.median.astype("float64")

        # define optimal rotation+translation between median to glasses space models
        # median -> glasses space
        r, t = rigid_transform_3D(self.median, self.in_glasses_space)

        # define etg plane normal vector
        etg_plane_normal_vec = np.cross(
            self.median["lhead"] - self.median["rhead"],
            self.median["lhead"] - self.median["fhead"],
        )

        # define "median" position of points [camera, camera_vector_origin, left_vector_origin] in mocap space
        camera = pd.Series(
            {
                "x": (self.median["lhead"]["x"] + self.median["rhead"]["x"]) / 2,
                "y": (self.median["lhead"]["y"] + self.median["rhead"]["y"]) / 2,
                "z": self.median["fhead"]["z"],
            },
            name="camera",
        )

        # a vector through this point intersects the etg plane to represent camera_vector_origin
        rayTo_camera_vector_origin = pd.Series(
            {"x": camera["x"], "y": camera["y"], "z": camera["z"] + 1.0,},
            name="rayTo_camera_vector_origin",
        )

        # a vector through this point intersects the etg plane to represent left_vector_origin
        rayTo_left_vector_origin = pd.Series(
            {
                "x": rayTo_camera_vector_origin["x"]
                + abs(camera["z"] - rayTo_camera_vector_origin["z"])
                * np.tan(np.pi / 6),
                "y": rayTo_camera_vector_origin["y"],
                "z": rayTo_camera_vector_origin["z"],
            },
            name="rayTo_left_vector_origin",
        )

        # find plane intersection points
        camera_vector_origin = pd.Series(
            ray_plane_intersect(
                camera,
                rayTo_camera_vector_origin,
                etg_plane_normal_vec,
                self.median["lhead"],
            ),
            name="camera_vector_origin",
        )

        left_vector_origin = pd.Series(
            ray_plane_intersect(
                camera,
                rayTo_left_vector_origin,
                etg_plane_normal_vec,
                self.median["lhead"],
            ),
            name="left_vector_origin",
        )

        # join points into a single dataframe
        best_guess = pd.concat(
            [camera, camera_vector_origin, left_vector_origin], axis=1
        )

        # apply rotation+translation to "median" best guess points [camera, camera_vector_origin, left_vector_origin]
        best_guess = r @ best_guess + np.tile(t, (best_guess.shape[1], 1)).T
        best_guess.index = ["x", "y", "z"]

        return best_guess

    def save(self, path):
        """append calculated camera model to csv file"""

        # don't change stored structure
        gs = self.in_glasses_space
        med = self.median

        # add identifier to respective dataframes
        gs.columns = pd.MultiIndex.from_product([["glasses_space"], gs.columns])
        # med.columns = pd.MultiIndex.from_product([["median"], med.columns])

        # flatten data
        to_file = pd.merge(gs, med, left_index=True, right_index=True).unstack()
        to_file.index = to_file.index.to_flat_index()

        # add filename & creation date
        fname_and_date = pd.Series(
            [self.from_file, self.creation_date], index=["filename", "created"]
        )

        to_file = fname_and_date.append(to_file).to_frame().T

        # join each flattened column name tuple into a string
        to_file.columns = [
            ".".join(c) if isinstance(c, tuple) else c for c in to_file.columns
        ]

        # If we are creating this file append a header, else just append a line of data
        if not path.exists():
            to_file.to_csv(path, index=False, mode="w")
        else:
            to_file.to_csv(path, index=False, mode="a", header=False)


if __name__ == "__main__":

    etg_model_path = Path("/home/raphy/proj/hmpldat/new_test_etg_models.csv")

    for f in search.files(
        Path("/home/irz0002/Projects/hmpldat/sample_data/merged"), []
    ):

        print(f.name)

        # instance empty glasses object
        g = etg_model()

        ### load corresponding model from file
        found = g.load(etg_model_path, f.name)
        print(found)

        if not found:  # create and save etg model

            # read data from file
            df = pd.read_csv(f, low_memory=False)

            df = df.filter(regex=r"[FLR]{1}HEAD.[XYZ]{1}")
            df = hmpldat.file.cortex.multiindex(df)
            df = df.dropna()

            g = etg_model()

            # for testing handling of more than three points
            fake_head_lr = df[["lhead", "rhead"]].mean(axis=1, level=1)
            fake_head_lr.columns = pd.MultiIndex.from_product(
                [["fake_head_lr"], fake_head_lr.columns]
            )

            fake_head_fl = df[["fhead", "lhead"]].mean(axis=1, level=1)
            fake_head_fl.columns = pd.MultiIndex.from_product(
                [["fake_head_fl"], fake_head_fl.columns]
            )

            fake_head_rf = df[["rhead", "fhead"]].mean(axis=1, level=1)
            fake_head_rf.columns = pd.MultiIndex.from_product(
                [["fake_head_rf"], fake_head_rf.columns]
            )

            df = pd.concat([df, fake_head_lr, fake_head_fl, fake_head_rf], axis=1)

            # define etg model in glasses space and median values
            g.new_define(
                df[
                    [
                        "lhead",
                        "rhead",
                        "fhead",
                    ]
                ],
                f.name,
            )

            print(g.in_glasses_space)
            print(g.cam_best_guess_pass_thru())

            g.save(etg_model_path)

            ### plot median rotated and translate to etg space to check that I am insane
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection="3d")

            # hmpldat.utils.plot.glasses(
            #     g.in_glasses_space, ax,
            # )
            # # hmpldat.utils.plot.screen(ax)

            # ax.legend()
            # plt.show()
            ### end plot (remove after debug)

            # # distances between median glasses points
            # median_dist_lr = euclidean_distance(median_glasses_model["lhead"], median_glasses_model["rhead"])
            # median_dist_rf = euclidean_distance(median_glasses_model["rhead"], median_glasses_model["fhead"])
            # median_dist_lf = euclidean_distance(median_glasses_model["lhead"], median_glasses_model["fhead"])

            # # print((df["lhead"] - df["rhead"]) ** 2)
            # # print(((df["lhead"] - df["rhead"]) ** 2).sum(axis=1))
            # # print(((df["lhead"] - df["rhead"]) ** 2).sum(axis=1) ** 0.5)

            # # vectorized mean distance between each instance of marker points
            # mean_dist_lr = (((df["lhead"] - df["rhead"]) ** 2).sum(axis=1) ** 0.5).mean()
            # mean_dist_rf = (((df["rhead"] - df["fhead"]) ** 2).sum(axis=1) ** 0.5).mean()
            # mean_dist_lf = (((df["lhead"] - df["fhead"]) ** 2).sum(axis=1) ** 0.5).mean()

            # print(median_dist_lr, mean_dist_lr)
            # print(median_dist_rf, mean_dist_rf)
            # print(median_dist_lf, mean_dist_lf)
            # print(median_dist_lr - mean_dist_lr)
            # print(median_dist_rf - mean_dist_rf)
            # print(median_dist_lf - mean_dist_lf)

            # g.save(etg_model_path)

        # # calculate "best guess" with respective methods
        # orig_best_guess = g.cam_best_guess_orig()
        # new_best_guess = g.cam_best_guess_ray()

        # print(orig_best_guess.to_markdown())
        # print(new_best_guess.to_markdown())

        ### plot median rotated and translate to etg space to check that I am insane
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")

        # hmpldat.utils.plot.glasses(
        #     pd.merge(
        #         orig_best_guess,
        #         new_best_guess,
        #         left_index=True,
        #         right_index=True,
        #         suffixes=("_etg_space", "_medians_+rot&trans"),
        #     ),
        #     ax,
        # )

        # plt.suptitle("median points rotated+translated to etg space")
        # ax.legend()
        # plt.show()
        ### end plot (remove after debug)

        # # rotation from etg space to median MoCap
        # r, t = rigid_transform_3D(g.in_glasses_space, g.median)

        # # Move etg best guess to MoCap space by optimal rotation and translation
        # new_best_guess_mocap = r @ new_best_guess + np.tile(t, (3, 1)).T
        # new_best_guess_mocap.index = ["x", "y", "z"]

        # ### Plot new best guess and median camera model rotatado and translato to glasses space
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")

        # inMoCap = pd.merge(
        #     new_best_guess_mocap, g.median, left_index=True, right_index=True
        # )

        # print(inMoCap.to_markdown())

        # hmpldat.utils.plot.glasses(
        #     inMoCap, ax,
        # )

        # plt.suptitle("median points and resultant camera model in MoCap Space")
        # ax.legend()
        # plt.show()
        # ### end plot (remove after debug)

        #         ### Plot in mocap space
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")

        # hmpldat.utils.plot.glasses(
        #     pd.merge(
        #         best_guess,
        #         self.median,
        #         left_index=True,
        #         right_index=True,
        #         suffixes=("_guess_mocap", "_medians"),
        #     ),
        #     ax,
        # )

        # plt.suptitle("median points new best guess methods in MoCap space")
        # ax.legend()
        # plt.show()
        # ### end plot (remove after debug)
