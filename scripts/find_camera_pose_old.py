"""

For each participant:
    find camera "pose" 
    use three points 

"""

from pathlib import Path
from pprint import pprint
from random import random
from time import time_ns

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt 

import hmpldat.utils.camera_model
import hmpldat.file.search as search
import hmpldat.align.spatial
from hmpldat.utils.math import (
    euclidean_distance,
    sq_euclidean_distance,
    unit_vector,
    point_plane_distance,
    rigid_transform_3D,
    angle_between_vectors,
)
import hmpldat.file.dflow
import hmpldat.file.detected
import hmpldat.utils.plot


LAMBDA = 1e6  # error factor
MAX_ITER = 5

XY_PLANE = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0])]

fout = open("single_instance_across_optimizer_iterations.csv", "w")

def is_edge_case(camera, camera_vector_origin, left_vector_origin):
    """Use this to drop data that does not follow the garden path

    Returns
        True when this value falls outside my garden
    
    """

    drop = False

    if np.linalg.norm(camera - camera_vector_origin) <= 10 or np.linalg.norm(camera - left_vector_origin) <= 10:
        drop = True

    return drop


def constraint_cam_fov(x):
    """
    The angle between these vectors must be 30 degrees

    defined by camera field of view
    """

    camera = x[:3]
    camera_vector_origin = x[3:6]
    left_vector_origin = x[6:]

    cam_vec = camera - camera_vector_origin
    left_vec = camera - left_vector_origin

    # Equality constraint means that the constraint function result is to be zero
    return angle_between_vectors(cam_vec, left_vec) - np.pi / 6


def constraint_center_cam(x, *args):
    """
    camera point should be approx equidistance from lhead and rhead

    """

    camera = x[:3].flatten()

    lhead, rhead = args
  
    return abs(sq_euclidean_distance(camera, lhead.flatten()) - sq_euclidean_distance(camera, rhead.flatten()))


def constraint_dist(x):
    """
    """

    camera = x[0:3]
    cam_vector_origin = x[3:6]

    # ineq constraint constraint function result is greater than 0

    d = euclidean_distance(camera, cam_vector_origin) - 10
    return d


def objective(x, *args):
    """
    for a a set of instances, estimate gaze location

    then find squared euclidean distance between object location and estimated gaze
    
    Args:
        x : 1-D array with shape (n,) 
        args : a tuple of the fixed parameters needed to completely specify the function.

    Returns:
        a float squared euclidean distance between object location and estimated gaze location

    """
    # print("\n\nObjective()")
    # print("+"*90)
 
    # save output to file
    # ddd = [str(q) for q in x]
    # data_str = ",".join(ddd) + "\n"
    # fout.write(data_str)
    
    # unpack independent variables (these are in camera space)
    camera = np.array(x[:3], dtype=np.float64)
    camera_vector_origin = np.array(x[3:6], dtype=np.float64)
    left_vector_origin = np.array(x[6:], dtype=np.float64)

    # reshape to apply rotation
    cam_space = np.array([camera, camera_vector_origin, left_vector_origin]).T
    print(cam_space)

    # unpack args tuple
    gaze, cross_projection, rotation_dict, model, video_time = args
    # print(gaze, cross_projection, rotation_dict, model)

    # reset distance sum
    sq_dist_list = []

    # for each instance
    for cross_num, instance in gaze.index:

        opt_rotaion = rotation_dict[instance]
        frame_time = video_time.loc[(cross_num, instance)]
        # print(gaze.loc[(cross_num, instance)])
        # print(cross_projection[])

        # rotate then translate points to mocap space
        mocap_space = opt_rotaion["r"] @ cam_space + np.tile(opt_rotaion["t"], (3, 1)).T
        # print(instance)
        
        # print()
        # pprint(mocap_space)
        # input()

        # write the values for this frame time to file
        if frame_time == '0 days 00:26:17.008000000':
            ddd = [str(q) for q in mocap_space.T.flatten()]
            data_str = ",".join(ddd) + "\n"
            fout.write(data_str)

        # split points to for input to function
        cam_mocap, cam_vector_origin_mocap, left_vector_origin_mocap = np.split(
            mocap_space, 3, axis=1
        )

        # if this point falls outside the garden path
        # then continue (start next iteration of for loop)
        if is_edge_case(cam_mocap, cam_vector_origin_mocap, left_vector_origin_mocap):
            print("points too close together")
            return np.inf
            continue

        est_mocap_gaze = hmpldat.align.spatial.estimate_gaze_location(
            cam_mocap.flatten(),
            cam_vector_origin_mocap.flatten(),
            left_vector_origin_mocap.flatten(),
            gaze.loc[(cross_num, instance)],
        )

        # print(est_mocap_gaze)

        sq_dist = sq_euclidean_distance(
            cross_projection.loc[(cross_num, instance)], est_mocap_gaze
        )

        sq_dist_list.append(sq_dist)

        # error (distance) between estimated gaze and actual cross location
        # sq_dist_sum += sq_euclidean_distance(cross_projection.loc[(cross_num, instance)], est_mocap_gaze)

    # print([x ** 0.5 for x in sq_dist_list])
    ccc = np.count_nonzero(np.isnan(sq_dist_list))

    # how I keep the otimizer chugging when shit hits the fan
    # replace all the nans with big boys
    sq_dist_list = np.nan_to_num(sq_dist_list, nan=np.inf)
    sq_dist_sum = np.nansum(sq_dist_list)

    # origin and left_origin belong on etg plane
    sq_dist_vector_origin_to_xy_plane = (
        point_plane_distance(XY_PLANE, camera_vector_origin) ** 2
        + point_plane_distance(XY_PLANE, left_vector_origin) ** 2
    )

    sq_dist_fhead_to_cam = sq_euclidean_distance(model["fhead"].values.flatten(), camera)

    ### safe to ignore these futile attempts to tame the optimizer
    # camera point distance to lhead vs. rhead (in camera space (lhead=(0,0,0)))
    # lhead_to_cam_vec_orgn = sum(camera_vector_origin ** 2)
    # lhead_to_left_vec_orgn = sum(left_vector_origin ** 2)

    # print(lhead_to_cam_vec_orgn, lhead_to_left_vec_orgn, lhead_to_left_vec_orgn - lhead_to_cam_vec_orgn)
 
    # actl_dist_btwn_vctr_orgn = lhead_to_left_vec_orgn - lhead_to_cam_vec_orgn
    # exptd_dist_btwn_vctr_orgn = euclidean_distance(camera_vector_origin, camera) * np.tan(np.pi / 6)
    # diff = ((exptd_dist_btwn_vctr_orgn - actl_dist_btwn_vctr_orgn) ** 2)
        
    # print((exptd_dist_btwn_vctr_orgn - actl_dist_btwn_vctr_orgn) ** 2)
    # print(exptd_dist_btwn_vctr_orgn, actl_dist_btwn_vctr_orgn)

    # f = 0
    # if lhead_to_cam_vec_orgn > lhead_to_left_vec_orgn:
    #     f = 1e20

    # cam_vec_orgn_to_cam = sq_euclidean_distance(camera, camera_vector_origin)

    # camera point distance to lhead vs. rhead (in camera)
    # cam_to_lhead = sq_euclidean_distance(model["lhead"].values.flatten(), camera)
    # cam_to_rhead = sq_euclidean_distance(model["rhead"].values.flatten(), camera)
    # lhead_to_rhead = sq_euclidean_distance(model["lhead"].values.flatten(), model["rhead"].values.flatten())

    # if cam_vec_orgn_to_cam > cam_to_lhead:
    #     # input(str(sq_dist_sum) + "\tINSANE")
    #     sq_dist_sum += 1e15

    # if cam_vec_orgn_to_cam > cam_to_rhead:
    #     # input(str(sq_dist_sum) + "\tINSANE")
    #     sq_dist_sum += 1e15

    print(sq_dist_sum, sq_dist_vector_origin_to_xy_plane, sq_dist_fhead_to_cam)

    return sq_dist_sum + 1000 * sq_dist_vector_origin_to_xy_plane + sq_dist_fhead_to_cam #+ diff


def find_optimal_rotation(model, real):
    """
    Find rotation & translation for each instance

    Returns:
        dict[index] = {r: (3,3) array, t: (3,) array}

    Note:
        TODO: decide calculate optimal translation or use lhead?
    
    """

    # print(model)
    # print("real")
    # print(real)
    # input()

    info = {}

    # for every instance in real
    for i, instance in real.groupby(level=1):

        ### 2 ways to use rigid transform:
        ### Provide translation
        # rotation, translation = rigid_transform_3D(
        #     model.values, instance.values, instance["lhead"].values
        # )

        ### Or, calculate OPTIMAL rotation
        rotation, translation = rigid_transform_3D(model.values, instance.values)

        info[i] = {"r": rotation, "t": translation}

    return info


def apply_rotation_translation(a, r, t):
    """ 

    Args:
        a: (3,n) a set of points
        r: (3,3) corresponding optimal rotation matrix 
        t: (3,) translation (typically lhead)

    Returns:
        dataframe set of points rotated (r) then translated (t)

    """

    b = r @ a + np.tile(t, (a.shape[1], 1)).T
    b.index = ["x", "y", "z"]

    return b


def main():
    """ """

    # load camera models
    camera_models = hmpldat.utils.camera_model.open(Path("test_camera_models.csv"))
    print(camera_models)

    info = {}

    for f in search.files(Path("/home/raphy/proj/hmpldat/sample_datas/merged"), ["38", "pp"]):

        info[f.name] = {}

        # choose corresponding camera  (ignore name and creation_date)
        camera_model, _fname, _creation_date = hmpldat.utils.camera_model.multiindex(
            camera_models.loc[f.name]
        )

        print("camera_model=")
        print(camera_model.stack().reset_index(0, drop=True))
        # input()

        # initial location of camera, camera_vector_origin, left_vector_origin
        initial_guess = hmpldat.utils.camera_model.best_guess(camera_model)
        print("initial guess=")
        print(initial_guess)
        
        # column headers for recorded optimizer values to file
        ffff = ["_".join(e) for e in initial_guess.unstack().index]
        fout.write(",".join(ffff) + "\n")
        
        # # calculate angle between vectors
        # print(
        #     hmpldat.utils.math.angle_between_vectors(
        #         initial_guess["camera"] - initial_guess["camera_vector_origin"],
        #         initial_guess["camera"] - initial_guess["left_vector_origin"],
        #     )
        #     * 180
        #     / np.pi
        # )
        # input()

        print(f"file= {f.name}")
        input("hit [ENTER] to continue")
        # if input() == "s":
        #     continue

        # open file
        df = pd.read_csv(f, low_memory=False)

        #### Book Keeping stuff        # TODO: create new sample merged data, then remove this
        df[df.filter(like=".Pos").columns] = (
            df.filter(like=".Pos") * 1000
        )  ## convert m to mm
        # print(df.filter(like="Cross.Pos"))

        df["time_mc_adj"] = pd.to_timedelta(df["time_mc_adj"])

        # define projection height
        y_views = df.dropna(subset="RHEAD.Y")

        # drop instances with no mocap data (cortex typically starts recording shortly after dflow)
        df = df.dropna(subset=["Frame#"])

        # objects = hmpldat.file.detected.multiindex_object_columns(df)
        # print(objects)

        # gaze = df[
        #     ["Point of Regard Binocular X [px]", "Point of Regard Binocular Y [px]"]
        # ]

        # print(len(gaze))
        # print(len(objects))

        # for objt, df in objects.groupby(axis="columns", level=0):
        #     df.columns = df.columns.droplevel()
        #     gaze_err = np.linalg.norm(df[["ctr_bb_col", "ctr_bb_row"]].values - gaze.values, axis=1)
        #     gaze_err = pd.DataFrame(gaze_err, index=df.index)
        #     print(objt)
        #     print(gaze_err.dropna())

        # input()


        #### filter for "milestone" events
        df = hmpldat.align.spatial.find_milestone_instances(df)

        # sample N from each group
        # n = 2
        # df = df.groupby(level=[0], axis="index", group_keys=False).apply(
        #     lambda g: g.sample(n, random_state=12345) #keep the same numbers
        # )

        df["px_gaze_err"] = np.linalg.norm(df[["cross_ctr_bb_col", "cross_ctr_bb_row"]].values - df[
            ["Point of Regard Binocular X [px]", "Point of Regard Binocular Y [px]"]
        ].values, axis=1)

        # df = df[df["px_gaze_err"] < 25]

        # transform data to multi-indexed columns (with lower cased column names)
        mocap = hmpldat.file.cortex.multiindex(df)
        vr = hmpldat.file.dflow.multiindex(df)

        # select required data
        mocap_etg = mocap[["fhead", "lhead", "rhead"]]
        cross_position = vr["cross"]
        gaze = df[
            ["Point of Regard Binocular X [px]", "Point of Regard Binocular Y [px]"]
        ]
        video_time = df["Video Time [h:m:s:ms]"]
        cross_ctr = df[["cross_ctr_bb_col", "cross_ctr_bb_row"]]

        # print(cross_ctr)
        # print(gaze)

        # print(df["px_gaze_err"])
        # print(df["px_gaze_err"].mean())
        # input()
        # # print(gaze)

        # calculate cross projection location
        cross_projection = cross_position.apply(
            lambda row: hmpldat.file.dflow.project_object_onto_screen(
                row.loc["x"], row.loc["y"], row.loc["z"], y_view
            ),
            axis=1,
        )
        # split tuples into attributes of dataframe
        cross_projection = pd.DataFrame(
            cross_projection.to_list(),
            index=cross_projection.index,
            columns=["x","y","z"],
        )
        
        ### calculate optimal rotation for each instance
        rotation_dict = find_optimal_rotation(camera_model.stack(), mocap_etg.stack())
            
        # rot_all_etg = apply_rotation_translation(all_etg, rotation_dict[a_key]["r"], rotation_dict[a_key]["t"])
        # print(rot_all_etg)

        # print(mocap_etg.head(1).reset_index(level=[0,1], drop=True))

        # hmpldat.utils.plot.compare_etg(
        #     mocap_etg.head(1).reset_index(level=[0,1], drop=True).stack().T,
        #     rot_all_etg.T
        # )

        # the values I am optimizing
        # print(initial_guess)
        # input()

        # print(gaze)
        # print(cross_projection)
        # print(rotation_dict)

        # for i in MAX_ITER:

        print("initial guess")
        print(initial_guess)
        # input()

        bounds=((-10, 150), (-10, 100), (-50, 50), (-10, 150), (-10, 150), (-50, 50), (-10, 150), (-10, 100), (-50, 50))

        # with inequality contraints scipy uses Sequential Least SQuares Programming
        # nelder-mead is the ameoba triangle 
        res = opt.minimize(
            objective,
            x0=initial_guess.unstack().values,
            args=(cross_ctr, cross_projection, rotation_dict, camera_model, video_time),
            # method="nelder-mead",
            # tol=1,
            bounds=bounds,
            # constraints={"type": "ineq", "fun": constraint_dist},
            # options={"disp": True, "ftol": 1e-6},
        )

        res_df = pd.DataFrame(res.x, index=initial_guess.unstack().index)
        pprint(res)

        print(f.name)

        print("initial guess")
        print(initial_guess.T)

        print("\noptimized result")
        print(res_df.unstack())

        print(camera_model)

        a_key = list(rotation_dict.keys())[0]

        res_df = pd.merge(res_df.unstack().T.reset_index(0, drop=True), camera_model.stack().reset_index(0, drop=True), right_index=True, left_index=True)
        print(res_df)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        rot_result_etg = apply_rotation_translation(res_df, rotation_dict[a_key]["r"], rotation_dict[a_key]["t"])
        print(rot_result_etg)     

        hmpldat.utils.plot.camera_model(rot_result_etg, ax)

        ax.legend()
        ax.set_xlim(-150,150)
        ax.set_ylim(150, -150) # z-limits
        ax.set_zlim(1500, 1800) # y-limits
        ax.set_xlabel('X')
        ax.set_ylabel('Z') # swap axis labels to match our environment
        ax.set_zlabel('Y')
        plt.suptitle('camera_model + optimized points transformed to MoCap Space\n' + f.name)
               
        print(mocap_etg.head(1).reset_index(level=[0,1], drop=True).stack())
        print()

        plt.show(block=False)
        plt.pause(0.01)
        
        while True:
            if input("hit [SPACE BAR] to end.") == ' ':
                plt.close()
                break
            else:
                print("y no pres spcebarr?")


if __name__ == "__main__":
    main()
    
    # close file that you may have been writing datas to.
    fout.close()
    
    # plot()
