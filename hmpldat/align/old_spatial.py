""" Methods to perform spatial alignment

"""

import math
import time
import random
from pprint import pprint
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.spatial.transform import Rotation

import hmpldat.utils.math
from hmpldat.file.dflow import project_vr_object_onto_screen


def split_undefined_slope(df, slope):
    """ separate datas to be handled differently

    Returns:
        2 dataframes: instances with slope, instances with undefined slope

    """

    # TODO: check indicies for df and slope are equal

    return df[slope.run != 0], df[slope.run == 0]


def find_vector_screen_intersection(camera, origin):
    """Calculate in MoCap Space where the vector intersects the screen
    
    The two points, camera and origin, define this vector

    Args:
        camera: camera point in mocap space (x, y, z)
        origin: point on head marker plane (x, y, z)
        slope: if false then slope 

    Returns:
        intersection: the intersection of the vector and the screen

    TODO: reference slide one
    
    .. figure: path/to/figure

    """

    if camera.shape != origin.shape:
        raise ValueError("inputs are different shapes")

    # empty dataframe to hold intersection point
    intersection = pd.DataFrame(index=camera.index, columns=camera.columns)

    # calc xz slope (returned as rise(z) and run(x) attributes)
    slope = hmpldat.utils.math.slope_2d(camera, origin)

    # split datas
    camera_default, camera_undefined_slope = split_undefined_slope(camera, slope)
    origin_default, origin_undefined_slope = split_undefined_slope(origin, slope)
    slope_default, slope_undefined_slope = split_undefined_slope(slope, slope)
    intersect_default, intersect_boundary = split_undefined_slope(intersection, slope)

    ### perform ops when slope is defined (default)
    # TODO: XZ_SLOPE needs to be the same length as camera_default
    if len(camera_default) > 0:
        slope_default["xz"] = slope_default.rise / slope_default.run

        n = camera_default.X - camera_default.Z / (slope_default.xz)

        a = 1 + 1 / slope_default.xz ** 2
        b = (2 * n) / slope_default.xz
        c = slope_default.xz ** 2 - 2490 ** 2

        # calc z intersect
        z_intersection_solns = hmpldat.utils.math.quadratic_formula(a, b, c)

        print(z_intersect_solns)

        # TODO: pick correct solution, not the first.
        # TODO: handle no solutions??
        itersect_default.Z = z_intersection_solns

        # calc x intersect
        itersect_default.X = (
            itersect_default.Z - camera_default.Z
        ) / slope_default.xz + camera_default.X

    ### perform ops for undefined slope
    # refer to notes
    if len(camera_undefined_slope) > 0:

        # camera_undefined_slope
        intersect_boundary.X = camera_undefined_slope.X
        intersect_boundary.Z = (2490 ** 2 - intersect_boundary.X) ** 0.5

    # combine data back together
    camera = pd.concat([camera_default, camera_undefined_slope]).sort_index()
    origin = pd.concat([origin_default, origin_undefined_slope]).sort_index()
    intersection = pd.concat([intersect_default, intersect_boundary]).sort_index()

    # calculate xz distance
    xz_dist_o_cam = hmpldat.utils.math.euclidean_distance(
        camera[["X", "Z"]], origin[["X", "Z"]]
    )
    xz_dist_cam_intersect = hmpldat.utils.math.euclidean_distance(
        intersection[["X", "Z"]], camera[["X", "Z"]]
    )

    # calculate intersection y location
    intersection.Y = (
        camera.Y + (camera.Y - origin.Y) * xz_dist_cam_intersect / xz_dist_o_cam
    )

    return intersection


def find_horizontal_frame_center_left(camera, frame_center):
    """return the left frame center as if the participant is NOT tilting their head left or right.

    Args:
        camera:
        frame_center:

    Returns:
        MoCap position as if the participant is not tilting their head left or right (not on screen)

    .. figure: slide 2
    
    """

    dist_camera_fc = hmpldat.utils.math.euclidean_distance(camera, frame_center)
    o = dist_camera_fc * math.tan(math.pi / 6)

    horizontal_left_frame_center = pd.DataFrame(
        data=frame_center.Y, index=frame_center.index, columns=frame_center.columns
    )

    # split data to apply methods separately for boundary case (camera.X == frame_center.X)
    camera_default, camera_boundary = (
        camera[camera.X != frame_center.X],
        camera[camera.X == frame_center.X],
    )
    frame_center_default, frame_center_boundary = (
        frame_center[camera.X != frame_center.X],
        frame_center[camera.X == frame_center.X],
    )
    hlfc_default, hlfc_boundary = (
        horizontal_left_frame_center[camera.X != frame_center.X],
        horizontal_left_frame_center[camera.X == frame_center.X],
    )

    if len(frame_center_default) > 0:

        k = (frame_center_default.Z - camera_default.Z) / (
            frame_center_default.X - camera_default.X
        )

        a = k ** 2 + 1
        b = 2 * a * frame_center_default.Z
        c = frame_center_default.Z ** 2 - o ** 2

        possible_z_hlfc = hmpldat.utils.math.quadratic_formula(a, b, c)

        # TODO: choose correct soln, not first
        hlfc_default.Z = possible_z_hlfc[0]

        hlfc_default.X = frame_center_default.X - k * (
            hlfc_default.Z - frame_center_default.Z
        )

    if len(frame_center_boundary) > 0:

        hlfc_boundary.Z = frame_center_boundary.Z
        hlfc_boundary.X = frame_center_boundary.X - o

    # combine datas
    horizontal_left_frame_center = pd.concat([hlfc_default, hlfc_boundary]).sort_index()

    return horizontal_left_frame_center


def find_left_frame_center(camera, origin_left, frame_center):
    """find actual location of left frame center including participant head left/right tilt 

    Args:
        camera:
        origin_left:
        frame_center:

    Returns:

    .. figure: slide 3
    
    """

    left_frame_center = pd.DataFrame(
        index=frame_center.index, columns=frame_center.columns
    )

    # this vector is normal to captured image frame
    camera_vector = camera - frame_center

    # change camera_vector column name to <a,b,c>
    camera_vector = camera_vector.rename(columns={"X": "a", "Y": "b", "Z": "c"})
    print(camera_vector)

    xz_slope = (camera.Z - origin_left.Z) / (camera.X - origin_left.X)

    h = (
        -camera_vector.a / (camera_vector.b * xz_slope)
        - camera_vector.c / camera_vector.b
    )
    j = (
        (camera_vector.a / (camera_vector.b * xz_slope)) * camera.Z
        + (camera_vector.c / camera_vector.b) * frame_center.Z
        + (camera_vector.a / camera_vector.b) * (frame_center.X - camera.X)
        + frame_center.Y
    )

    dist_camera_fc = hmpldat.utils.math.euclidean_distance(camera, frame_center)
    o = dist_camera_fc * math.tan(math.pi / 6)

    p = camera.Z / xz_slope + camera.X - frame_center.X
    q = j - frame_center.Y

    a = 1 / xz_slope ** 2 + h ** 2 + 1
    b = 2 * p / xz_slope + 2 * h * q - 2 * frame_center.Z
    c = p ** 2 + q ** 2 + frame_center.Z ** 2 - o ** 2

    possible_z_lfc = hmpldat.utils.math.quadratic_formula(a, b, c)

    # TODO: choose correct, not first
    left_frame_center.Z = possible_z_lfc[0]

    left_frame_center.X = (left_frame_center.Z - camera.Z) / xz_slope + camera["X"]
    left_frame_center.Y = h * left_frame_center.Z + j

    return left_frame_center


def find_frame_rotation(left_frame_center, horizontal_left_frame_center):
    """

    """
    dist_camera_fc = utils.math.euclidean_distance(camera, frame_center)
    o = dist_camera_fc * math.tan(math.pi / 6)

    dist_hlfc_lfc = utils.math.euclidean_distance(
        left_frame_center.values, horizontal_left_frame_center.values
    )

    theta = 2 * np.arcsin(dist_hlfc_lfc / (2 * o))

    return theta


def find_gaze_coords_frame_uv(gaze_PoR, o):
    """
    """

    u = (abs(gaze_PoR["X"] - 480) * o) / 480
    v = (abs(gaze_PoR["Y"] - 360) * o) / 360

    # what quadrant was my gaze in?
    quad = utils.rawetg.quadrant(gaze_PoR)

    if quad == 2:
        u = -u
    elif quad == 3:
        u = -u
        v = -v
    elif quad == 4:
        v = -v

    frame_gaze = [u, v, 0]

    return frame_gaze


def find_frame_gaze_mocap_coords(theta, u, uv_frame_gaze, frame_center):
    """
    """

    # scipy spatial transform
    r = Rotation.from_matrix(
        [
            [
                np.cos(theta) + u["X"] ** 2 * (1 - np.cos(theta)),
                u["X"] * u["Y"] * (1 - np.cos(theta)) - u["Z"] * np.sin(theta),
                u["X"] * u["Z"] * (1 - np.cos(theta)) - u["Y"] * np.sin(theta),
            ],
            [
                u["Y"] * u["X"] * (1 - np.cos(theta)) - u["Z"] * np.sin(theta),
                np.cos(theta) + u["Y"] ** 2 * (1 - np.cos(theta)),
                u["Y"] * u["Z"] * (1 - np.cos(theta)) - u["X"] * np.sin(theta),
            ],
            [
                u["Z"] * u["X"] * (1 - np.cos(theta)) - u["Y"] * np.sin(theta),
                u["Z"] * u["Y"] * (1 - np.cos(theta)) - u["X"] * np.sin(theta),
                np.cos(theta) + u["Z"] ** 2 * (1 - np.cos(theta)),
            ],
        ]
    )

    # TODO: check that frame center and rotated frame gaze are same shapes
    # gaze in mocap space (NOT on screen)
    mocap_frame_gaze = r.apply(uv_frame_gaze) + frame_center

    return mocap_frame_gaze


# def calc_milestone_event():
#     """filter gaze locations occuring durring the appearance of an object

#     average reaction time = 250ms
#     discard records that are "saccade"
#     discard initial 300 ms

#     """


def initialize_camera_position(df, rotation_matrix=None, translation=None):
    """Use to initialize participants for gaze estimation

    if rotation matrix is given then use it to initialize else use educated guess

    Args:
        df: merge pandas dataframe

    Returns:
        the same data frame with additional columns with values initialized according to notes.

    Notes:
        Requires cortex data.

        o = origin of vector on head marker plane

        o = (lhead + rhead) / 2
        > This point directly between lhead and rhead should be close to directly behind cam_captr_pt and it is on hd_mrkr_pln. 
        > This is currently being used for cam_captr_pt, but it could be problematic for cam_captr_pt to start on the hd_mrkr_pln; 
        > whereas, cam_cntr_v_pln_orig must always be on hd_mrkr_pln.

        camera(x, y, z): (cam_cntr_v_pln_orig_x, cam_cntr_v_pln_orig_y - 10mm, fhead_z) 
        > This should be an okay initial point, as long as the participant is mostly facing forward.
        
        o_left = lhead + (rhead - lhead)*5/8
        > This point is on hd_mrkr_pln, as required, and slightly to the camera’s right side, as required. 
        > It’s not necessarily a great estimate for the y value, but getting one on the plane is more work than it is worth. 
        > Let the solver figure it out.
        > origin of vector that points the the left center of the screen
        
    """

    if rotation_matrix is None:
        # select lhead, rhead, fhead columns from big dataframe
        lhead = df.filter(like="LHEAD")
        rhead = df.filter(like="RHEAD")
        fhead = df.filter(like="FHEAD")

        # drop identifiers from column names (makes code more readable)
        # We are using these to create new points anyways. e.g. 'LHEAD.X' -> 'X'
        lhead.columns = [c.split(".")[1] for c in lhead.columns]
        rhead.columns = [c.split(".")[1] for c in rhead.columns]
        fhead.columns = [c.split(".")[1] for c in fhead.columns]

        # check that everone is still the same shape
        assert lhead.shape == rhead.shape == fhead.shape

        ### initialize points
        o = (lhead - rhead) / 2

        camera = o.copy()
        camera["Y"] = camera["Y"] - 10
        camera["Z"] = fhead["Z"]

        o_left = lhead + (rhead - lhead) * (5 / 8)

    else:
        # use a rotation matrix to estimate these points
        raise NotImplementedError
        # TODO: implement this

    # print(cam_cntr_v_pln_orig.dropna().head())
    # print(cam_captr_pt.dropna().head())

    # change col names before merge back into one df
    o.columns = [".".join(["o", c]) for c in o.columns]
    camera.columns = [".".join(["camera", c]) for c in camera.columns]
    o_left.columns = [".".join(["o_left", c]) for c in o_left.columns]

    # merge everybody back together
    new_merged = pd.merge(o, camera, left_index=True, right_index=True)
    new_merged = pd.merge(new_merged, o_left, left_index=True, right_index=True)
    new_merged = pd.merge(df, new_merged, left_index=True, right_index=True)

    return new_merged


def rigid_transform_3D(A, B):
    """
    TODO: replace matrix with ndarray

    Should we be using this method? - for only three points I think there are more efficient methods 

    See:
        https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
        
        This should help with double checking this and adding a test case.
    
    Takes a set of points defining configuration A and finds the rigid transform taking them to a set of points defining configuration B.
    Configuration A and configuration B must have the same number of points.

    [[x1, x2, x3, x4, ..., xn],
     [y1, y2, y3, y4, ..., yn],
     [z1, z2, z3, z4, ..., zn]]

    Args:
        A (array): An array of n cordinate points describing configuration A.
        B (array): An array of n cordinate points describing configuration B.

    Returns:
        R: Rotation matrix
        t: Translation matrix
        
    """

    if A.shape != B.shape:
        raise ValueError("A and B must be the same shape")

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # subtract mean
    Am = A - np.tile(centroid_A, (1, num_cols))
    Bm = B - np.tile(centroid_B, (1, num_cols))

    # dot is matrix multiplication for array
    H = Am * np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...\n")
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = -R * centroid_A + centroid_B

    return R, t


def est_gaze_error(df):
    """

    o = 

    Notes:
        a.) Determine constant slope of camera capture angle (du, dv) to upper left corner of scene’s image:
        du = tan(30), dv = tan(23), both in degrees
        b.) Determine the slope (dx, dy, dz) of the vector from cam_cntr_v_pln_orig to cam_captr_pt: 
        = cam_captr_pt - cam_cntr_v_pln_orig
        c.) Determine slope(cam_00_v_pln_orig, cam_captr_pt) = cam_captr_pt - cam_00_v_pln_orig
        d.) Determine slope(cam_captr_pt, est_mocap_gz_pt) based on (a), (b), (d), etg_gz_img_col (aka etg_gz_img_u), and etg_gz_img_v; = I’ll leave this one for you to think through (send me all your equations for this)
        e.) Determine est_mocap_gz_pt using cam_captr_pt, (d), and knowledge of the screen; equations/code should exist already

    Args:
        df: pandas dataframe of each task merged

    Returns:

    """

    # a) Determine constant slope of camera capture angle (du, dv) to upper left corner of scene’s image:
    du = np.degrees(np.tan(np.pi / 6))
    dv = np.degrees(np.tan(23 * np.pi / 180))

    # print(" constant angle to upper left corner ".center(50, "-"))
    # print(f"\tdu={du}\n\tdv={dv}")
    # print("-" * 50)

    ### initialize points: cam, o, ol
    # if previous participant: use that rotation matrix + translation
    # rigid_transform?

    # else: best guess
    df = initialize_camera_position(df)

    # project each object onto screen
    project_vr_object_onto_screen()

    # find slope of "camera" | "head" vector
    # apply euclidean dist to each instance cam_point - o_point
    hmpldat.utils.math.euclidean_dist()

    # find slope of vector through camera point pointing to left frame center
    # apply euclidean dist to each instance cam_point - ol_point
    hmpldat.utils.math.euclidean_dist()
