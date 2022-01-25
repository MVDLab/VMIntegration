""" Methods to perform spatial alignment

TODO: unpack arguments inside of each method

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
from scipy.spatial.distance import euclidean

import hmpldat.utils.math
from hmpldat.utils.math import euclidean_distance
from hmpldat.file.rawetg import quadrant

# from hmpldat.file.dflow import project_vr_object_onto_screen

SCREEN_RADIUS = 2490.0  # mm
# SCREEN_RADIUS = 2.49  # meters
# TODO(future): define constants some place known
# FOV_HORIZONTAL_DEGREES = 60
# FOV_VERTICAL_
# FRAME_WIDTH_PX
# FRAME_HEIGHT_PX
RADIANS_PER_H_PIXEL = 60/960 * np.pi/180
RADIANS_PER_V_PIXEL = 46/720 * np.pi/180


def split_undefined_slope(df, slope):
    """ separate datas to be handled differently

    Returns:
        2 dataframes: instances with slope, instances with undefined slope

    """

    # TODO: check indicies for df and slope are equal

    return df[slope.run != 0], df[slope.run == 0]


def find_vector_screen_intersection(camera, origin):
    """Calculate in MoCap Space where the vector intersects the screen
    
    Two points in MoCap space, camera and origin, define this vector

    Args:
        camera : array_like
                 camera position [x,y,z]
        origin : array_like
                 vector origin position [x,y,z]

    Returns:
        intersection: the intersection of the vector and the screen

    TODO: reference slide one
    
    .. figure: path/to/figure

    """
    # print(f"screen_intersect() " + "-"*71 + "\n")

    # print(f"vector screen intersection({camera},{origin})")

    cam_x, cam_y, cam_z = camera
    origin_x, origin_y, origin_z = origin

    # slope in xz_plane
    rise = cam_z - origin_z
    run = cam_x - origin_x

    xz_dist_o_cam = hmpldat.utils.math.euclidean_distance(
        [origin_x, origin_z], [cam_x, cam_z]
    )

    # print(f"\trise={rise}\n\trun={run}\n")

    if run == 0 and rise == 0:
        return (
            np.NaN
        )  # participant is either looking between their feet or up at the ceiling.

    # edge: slope is undefined!
    # this occurs when camera vector is parallel with z-axis
    elif run == 0:
        intersect_x = cam_x
        intersect_z = (SCREEN_RADIUS ** 2 - intersect_x ** 2) ** 0.5

        if rise < 0:
            intersect_z = -intersect_z

    # edge: slope is zero!
    # this occurs when camera vector is parallel with x-axis
    elif rise == 0:
        intersect_z = cam_z
        intersect_x = (SCREEN_RADIUS ** 2 - intersect_z ** 2) ** 0.5

        if run < 0:
            intersect_x = -intersect_x

    # we need quadratic to solve
    else:
        xz_slope = rise / run
        n = cam_x - cam_z / xz_slope

        a = 1 + 1 / xz_slope ** 2
        b = (2 * n) / xz_slope
        c = n ** 2 - SCREEN_RADIUS ** 2

        # should be negative (there is no screen behind you)
        possible_intersect_z = hmpldat.utils.math.quadratic_formula(a, b, c)
        # print(f"zs={possible_intersect_z}")

        # one soln | no soln
        if isinstance(possible_intersect_z, float):
            if np.isnan(possible_intersect_z):
                return np.NaN, np.NaN, np.NaN

            intersect_z = possible_intersect_z
            intersect_x = (intersect_z - cam_z) / xz_slope + cam_x

        else:  # choose solution

            possible_intersect_x = [
                (intersect_z - cam_z) / xz_slope + cam_x
                for intersect_z in possible_intersect_z
            ]

            soln = []

            # calculate both solutions
            for intersect_x, intersect_z in zip(
                possible_intersect_x, possible_intersect_z
            ):

                xz_dist_cam_intersect = hmpldat.utils.math.euclidean_distance(
                    [cam_x, cam_z], [intersect_x, intersect_z]
                )

                # calculate y intersection
                intersect_y = (
                    cam_y + ((cam_y - origin_y) * xz_dist_cam_intersect) / xz_dist_o_cam
                )

                soln.append([intersect_x, intersect_y, intersect_z])

            # print()
            # pprint(soln)
            real = hmpldat.utils.math.unit_vector(camera, origin)
            # print("\nreal=", real)

            ps = []
            # calc unit vectors
            for s in soln:
                ps.append(hmpldat.utils.math.unit_vector(s, origin))

            # pprint(ps)

            pt = list(np.allclose(real, p, rtol=1, atol=0.1) for p in ps)
            # print(pt)

            # print()
            if pt[0]:
                intersect_x, intersect_y, intersect_z = soln[0]
            elif pt[1]:
                intersect_x, intersect_y, intersect_z = soln[1]
            else:
                print("fail choosing solution vector_screen_intersect()")

            return intersect_x, intersect_y, intersect_z

    xz_dist_cam_intersect = hmpldat.utils.math.euclidean_distance(
        [cam_x, cam_z], [intersect_x, intersect_z]
    )

    # calculate y intersection
    intersect_y = cam_y + ((cam_y - origin_y) * xz_dist_cam_intersect) / xz_dist_o_cam

    return intersect_x, intersect_y, intersect_z


def find_horizontal_left_frame_center2(camera, frame_center, len_fc_lfc):
    """return the left frame center as if the participant is NOT tilting their head left or right.

    Args:
        camera : array_like 
                 camera position [x,y,z] in MoCap Space
        frame_center : array_like
                       frame center position in MoCap space
        len_fc_lfc : float 
                     distance between frame_center and left_frame_center

    Returns:
        horizontal_left_frame_center: MoCap position of the left frame center as if the participant is not tilting (roll) their head left or right (not on screen)

    .. figure: slide 2
    
    """
    # print()
    # print(f"hlfc() " + "-"*84 + "\n")

    # print(frame_center)

    cam_x, _cam_y, cam_z = camera
    fc_x, fc_y, fc_z = frame_center

    # as if participant is NOT tilting (roll) their head
    # hlfc_y = fc_y

    # participant's camera vector is parallel with the z-axis
    if fc_x == cam_x:
        # print(f"\tcamera_vector || z-axis\n")
        hlfc_z = fc_z

        # TODO: just check fc_z - cam_z < 0??
        # assuming participant has NOT gone horror movie possessed
        ang = np.arctan2(fc_z, fc_x)
        # quadrant 3&4 (typical e.g. looking at the screen)
        if ang < 0:
            hlfc_x = fc_x - len_fc_lfc
        # quadrant 1&2
        else:
            hlfc_x = fc_x + len_fc_lfc

    # participant's camera vector is parallel with the x-axis
    elif fc_z == cam_z:
        # print(f"\tcamera_vector || x-axis\n")

        hlfc_x = fc_x

        # assuming participant has NOT gone horror movie possessed
        ang = abs(np.arctan2(fc_z, fc_x))
        # quadrant 1&4
        if ang < np.pi / 2:
            hlfc_z = fc_z + len_fc_lfc
        # quadrant 2&3
        else:
            hlfc_z = fc_z - len_fc_lfc

    # we need to solve quadratic
    else:
        # print("hlfc normy")
        # print()
        # print(frame_center)
        # print()
        # print(camera)
        # print()
        # print(f"\tsolving quadratic")
        k = (fc_z - cam_z) / (fc_x - cam_x)

        a = k ** 2 + 1
        b = -2 * a * fc_z
        c = a * fc_z ** 2 - len_fc_lfc ** 2

        # print(f"a={a}\tb={b}\tc={c}")
        # print(len_fc_lfc)

        poss_hlfc_z = hmpldat.utils.math.quadratic_formula(a, b, c)

        # print(f"\tposs_z_solns={poss_hlfc_z}\n")

        # one soln | no soln
        if isinstance(poss_hlfc_z, float):

            if np.isnan(poss_hlfc_z):
                return np.NaN, np.NaN, np.NaN

            hlfc_z = poss_hlfc_z
            hlfc_x = fc_x - k * (hlfc_z - fc_z)

        # two solutions found
        elif isinstance(poss_hlfc_z, tuple):

            # solve for each possible x
            poss_hlfc_x = [fc_x - k * (hlfc_z - fc_z) for hlfc_z in poss_hlfc_z]
            # poss_hlfc_x = [fc_x - np.sqrt(len_fc_lfc **2 - (hlfc_z - fc_z) **2) for hlfc_z in poss_hlfc_z]

            #### CHANGE and comment this
            # print(poss_hlfc_z)
            # print(poss_hlfc_x)

            # angle between frame_center and +x axis
            ang_to_frame_center = np.arctan2(fc_z, fc_x)
            # print(f"\tfc angle={ang_to_frame_center - np.pi/6}\n")

            expected_ang_to_lfc = ang_to_frame_center - np.pi / 6

            # keep angle between [-pi,pi]
            if expected_ang_to_lfc < -np.pi:
                expected_ang_to_lfc += 2 * np.pi

            # print(f"\tfc angle={ang_to_frame_center + np.pi/6}\n")

            # find angle from +x axis to possible solutions
            soln_angle = np.arctan2(poss_hlfc_z, poss_hlfc_x)
            # print(soln_angle)

            solns = zip(poss_hlfc_x, poss_hlfc_z, soln_angle - expected_ang_to_lfc)

            ### choose more correct solution
            # minimum = solns[0]
            # for s in solns:
            #     if s[2] < minimum[2]:
            #         minimum = s

            hlfc_x, hlfc_z, _ = min(solns, key=lambda x: abs(x[2]))

            # print(f"\t(x,z,angle) {list(zip(poss_hlfc_x, poss_hlfc_z, soln_angle - expected_ang_to_lfc))}\n")

    # print("\treturning")
    # print(f"\t{hlfc_x}, {fc_y}, {hlfc_z}\n")
    # input()

    # print()
    # print("-"*90)

    return hlfc_x, fc_y, hlfc_z


def find_horizontal_left_frame_center(camera, frame_center, dist_fc_lfc):
    """simple method to find horizontal left frame center

    Args:
        camera: a point [x,y,z]
        frame_center: a point [x,y,z]
        dist_fc_lfc: (o from slides) either d*tan(pi/6) or ecl_dist(fc,lfc)
    
    Returns:
        horizontal left frame center [x,y,z]

    Note:
        right hand rule! 

    """

    # normal vector defining captured image frame
    a = camera - frame_center
    
    # vector pointing straight up 
    b = [0, 1, 0]

    # "u" points towards horizontal left frame center
    u = np.cross(a, b)

    # turn u into a unit vector
    u = u / np.linalg.norm(u)

    # return actual [x,y,z] 
    return frame_center + dist_fc_lfc * u


def find_left_frame_center(camera, origin_left, frame_center, len_fc_lfc):
    """find actual location of left frame center including participant head left/right tilt (roll)

    Args:
        camera : array_like 
                 camera position [x,y,z] in MoCap Space
        origin_left : array_like
                      origin left position in MoCap space
        frame_center : array_like
                       frame center position in MoCap space
        len_fc_lfc : float 
                     distance between frame_center and left_frame_center

    Returns:
        left frame center in MoCap space (not on screen)

    .. figure: slide 3
    
    """
    # print(f"lfc() " + "-"*85 + "\n")

    # print(f"lfc()")

    cam_x, cam_y, cam_z = camera
    ol_x, ol_y, ol_z = origin_left
    fc_x, fc_y, fc_z = frame_center

    # print(frame_center)

    # cam_vector_[a,b,c]
    cam_vector_a, cam_vector_b, cam_vector_c = fc_x - cam_x, fc_y - cam_y, fc_z - cam_z
    # print(f"cam_vector = <{cam_vector_a}, {cam_vector_b}, {cam_vector_c}>")

    # xz slope of cam->lfc vector
    rise = cam_z - ol_z
    run = cam_x - ol_x

    # camera vector has no pitch (forward|backwards tilt)
    if cam_vector_b == 0:
        input("ZZ")

        # camera vector parallel to z-axis (vector origin_left -> left_frame_center must have a non-zero and defined m)
        if cam_vector_a == 0:

            m = rise / run
            if m == 0 or abs(m) == np.inf:  # this should never happen
                raise ValueError(f"m must have a non-zero|inf value")

            lfc_z = fc_z
            lfc_x = (lfc_z - cam_z) / m + cam_x

            # TODO: check this could be positive or negative??
            lfc_y = fc_y + (len_fc_lfc ** 2 - (lfc_x - fc_x) ** 2) ** 0.5

        # camera vector parallel to x-axis (vector origin_left -> left_frame_center must have a non-zero and defined m)
        elif cam_vector_c == 0:

            m = rise / run
            if m == 0 or abs(m) == np.inf:  # this should never happen
                raise ValueError(f"m must have a non-zero|inf value")

            lfc_x = fc_x
            lfc_z = (lfc_x - cam_x) * m + cam_z

            # TODO: check this could be positive or negative??
            lfc_y = fc_y + (len_fc_lfc ** 2 - (lfc_z - fc_z) ** 2) ** 0.5

        # potential for xz slope of cam->lfc vector to be zero|undefined
        else:

            # vector origin_left -> left_frame_center is undefined (parallel to the z-axis)
            if run == 0:

                lfc_x = cam_x
                lfc_z = fc_z - (cam_vector_a / cam_vector_c) * (lfc_x - fc_x)

                # TODO: check this could be positive or negative??
                lfc_y = (
                    fc_y
                    + (len_fc_lfc ** 2 - (lfc_x - fc_x) ** 2 - (lfc_z - fc_z) ** 2)
                    ** 0.5
                )

            # vector from origin_left -> left_frame_center is 0 (parallel to the x-axis)
            elif rise == 0:

                lfc_z = cam_z
                lfc_x = fc_x - (cam_vector_c / cam_vector_a) * (lfc_z - fc_z)

                # TODO: check this could be positive or negative??
                lfc_y = (
                    fc_y
                    + (len_fc_lfc ** 2 - (lfc_x - fc_x) ** 2 - (lfc_z - fc_z) ** 2)
                    ** 0.5
                )

            # we almost in the middle now boyz
            else:

                m = rise / run

                lfc_z = ((-cam_vector_c / cam_vector_a) * fc_z - cam_z / m + cam_x) / (
                    (-cam_vector_c / cam_vector_a) - 1 / m
                )
                lfc_x = fc_x - (cam_vector_c / cam_vector_a) * (lfc_z - fc_z)

                # TODO: check this could be positive or negative??
                lfc_y = (
                    fc_y
                    + (len_fc_lfc ** 2 - (lfc_x - fc_x) ** 2 - (lfc_z - fc_z) ** 2)
                    ** 0.5
                )

        return lfc_x, lfc_y, lfc_z

    # camera vector has pitch! (b != 0)
    else:

        # vector origin_left -> left_frame_center is undefined (parallel to the z-axis)
        if run == 0:
            input("YY")

            lfc_x = cam_x

            if cam_vector_c == 0:
                raise ValueError(
                    f"camera vector a == {cam_vector_c}. This shan't happen in this edge case"
                )

            s = (cam_vector_b / cam_vector_c) * fc_y - (cam_vector_a / cam_vector_c) * (
                lfc_x - fc_x
            )

            a = (cam_vector_b / cam_vector_c) ** 2 + 1
            b = -2 * (((cam_vector_b * s) / cam_vector_c) + fc_y)
            c = s ** 2 + fc_y ** 2 + (lfc_x - fc_x) ** 2 - len_fc_lfc ** 2

            possible_lfc_y = hmpldat.utils.math.quadratic_formula(a, b, c)

            if np.isnan(possible_lfc_y):
                return np.NaN, np.NaN, np.NaN

            possible_lfc_z = [
                fc_z
                - cam_vector_b / cam_vector_c * (y - fc_y)
                - cam_vector_a / cam_vector_c * (lfc_x - fc_x)
                for y in possible_lfc_y
            ]

            # TODO: look at this!!
            lfc_y = possible_lfc_y[0]
            lfc_z = possible_lfc_z[0]

        # vector from origin_left -> left_frame_center is 0 (parallel to the x-axis)
        elif rise == 0:
            input("XX")

            lfc_z = cam_z

            if cam_vector_a == 0:
                raise ValueError(
                    f"camera vector a == {cam_vector_a}. This shan't happen in this edge case"
                )

            r = (cam_vector_b / cam_vector_a) * fc_y - (cam_vector_c / cam_vector_a) * (
                lfc_z - fc_z
            )

            a = (cam_vector_b / cam_vector_a) ** 2 + 1
            b = -2 * (((cam_vector_b * r) / cam_vector_a) + fc_y)
            c = r ** 2 + fc_y ** 2 + (lfc_z - fc_z) ** 2 - len_fc_lfc ** 2

            possible_lfc_y = hmpldat.utils.math.quadratic_formula(a, b, c)

            if np.isnan(possible_lfc_y):
                return np.NaN, np.NaN, np.NaN

            possible_lfc_x = [
                fc_x
                - cam_vector_b / cam_vector_c * (y - fc_y)
                - cam_vector_c / cam_vector_a * (lfc_z - fc_z)
                for y in possible_lfc_y
            ]

            # TODO: look at this!!!
            lfc_y = possible_lfc_y[0]
            lfc_x = possible_lfc_x[0]

        # the normal case :)
        else:
            # print("normy")

            m = rise / run

            h = -cam_vector_a / (cam_vector_b * m) - cam_vector_c / cam_vector_b
            j = (
                (cam_vector_a / (cam_vector_b * m)) * cam_z
                + (cam_vector_c / cam_vector_b) * fc_z
                + (cam_vector_a / cam_vector_b) * (fc_x - cam_x)
                + fc_y
            )

            p = -cam_z / m + cam_x - fc_x
            q = j - fc_y

            a = 1 / m ** 2 + h ** 2 + 1
            b = 2 * p / m + 2 * h * q - 2 * fc_z
            c = p ** 2 + q ** 2 + fc_z ** 2 - len_fc_lfc ** 2

            possible_lfc_z = hmpldat.utils.math.quadratic_formula(a, b, c)
            # print(possible_lfc_z)

            # single solution or NaN
            if isinstance(possible_lfc_z, float):

                # no solns
                if np.isnan(possible_lfc_z):
                    return np.NaN, np.NaN, np.NaN

                lfc_z = possible_lfc_z
                lfc_y = h * lfc_z + j
                lfc_x = (lfc_z - cam_z) / m + cam_x

            # two solns
            else:

                possible_lfc_y = [h * z + j for z in possible_lfc_z]
                possible_lfc_x = [(z - cam_z) / m + cam_x for z in possible_lfc_z]

                soln_a, soln_b = zip(possible_lfc_x, possible_lfc_y, possible_lfc_z)
                # print(f"soln_a={soln_a}")
                # print(f"soln_b={soln_b}")

                unit_vec_ol_cam = hmpldat.utils.math.unit_vector(
                    (cam_x, cam_y, cam_z), (ol_x, ol_y, ol_z)
                )
                unit_vec_cam_soln_a = hmpldat.utils.math.unit_vector(
                    soln_a, (ol_x, ol_y, ol_z)
                )
                unit_vec_cam_soln_b = hmpldat.utils.math.unit_vector(
                    soln_b, (ol_x, ol_y, ol_z)
                )

                if sum((unit_vec_ol_cam - unit_vec_cam_soln_a) ** 2) < sum(
                    (unit_vec_ol_cam - unit_vec_cam_soln_b) ** 2
                ):
                    lfc_x, lfc_y, lfc_z = soln_a
                else:
                    lfc_x, lfc_y, lfc_z = soln_b

    return lfc_x, lfc_y, lfc_z


def find_frame_rotation(fc, hlfc, lfc, cam):
    """If horizontal-lfc and lfc are known, then calculate head rotation

    triangle

    Args:
        fc: frame center point [x,y,z]
        hlfc: horizontal left frame center point [x,y,z]
        lfc: left frame center point [x,y,z]

    Notes:
        Slide 8 (for now)
        Need to correctly rotate up or down?
        check cross product.

        TODO: what is positive angle?

    """

    # fc = np.array(fc)
    # hlfc = np.array(hlfc)
    # lfc = np.array(lfc)
    # cam = np.array(cam)

    # define vectors
    vec_fc_lfc = lfc - fc
    vec_fc_hlfc = hlfc - fc
    vec_hlfc_lfc = lfc - hlfc
    vec_cam = fc - cam

    ### two possible directions, (positive roll left) or (negative roll right)
    # compare cross product of vectors <fc->hlfc> x <hlfc->lfc> to camera vector decide
    # right hand rule.
    uvec_cam = vec_cam / np.linalg.norm(vec_cam)
    # print(uvec_cam)

    vec_dir = np.cross(vec_fc_hlfc, vec_hlfc_lfc)
    uvec_dir = vec_dir / np.linalg.norm(vec_dir)
    # print(uvec_dir)

    length_lfc_hlfc = np.linalg.norm(vec_hlfc_lfc)
    length_fc_lfc = np.linalg.norm(vec_fc_lfc)

    theta = 2 * np.arcsin(length_lfc_hlfc / (2 * length_fc_lfc))

    # if directions are the same (or similar) make length_lfc_hlfc negative
    # else nothing (leave length_lfc_hlfc positive) (sanity check: directions should be opposite)
    # this will correctly roll the head left or right
    if np.allclose(uvec_cam, uvec_dir):
        theta = -theta

    return theta


def find_coords_on_scene_frame(pixel, dist_fc_lfc, dist_fc_tfc):
    """

    with a known length (in mocap space) between frame center and left frame center
    we can calculate the pixel location in terms of distance

    Args:
        binocular_point_of_reference: DataFrame
        dist_frame_center_to_left_frame_center: distance between frame center and left frame center
        TESTING dist_cam_to_frame_center: distance between frame center and camera

    Returns:
        Real gaze location on imaginary "frame plane"
        u, v 

    """

    x, y = pixel.values

    # translate frame origin (0,0) to frame center (479.5, 359.5) (y is down)
    # opencv image center
    x = x - 479.5
    y = 359.5 - y

    # calculate relative pixel location in real distance
    # as pixel per mm
    u = (x * dist_fc_lfc) / 480
    v = (y * dist_fc_tfc) / 360

    # as radians per pixel
    # u = dist_cam_to_frame_center * np.tan(x * RADIANS_PER_H_PIXEL)
    # v = dist_cam_to_frame_center * np.tan(y * RADIANS_PER_V_PIXEL)  

    return u, v


def find_frame_mocap_coords(
    theta, camera_unit_vector, col, row, frame_center
):
    """

    Rotate pixel from location on imaginary frame plane to real frame plane in MoCap space

    Args:
        theta: head tilt (roll) radians
        camera_unit_vector: (fc - cam) / mag(fc - cam) = <x,y,z>
        col: float; real horizontal distance from frame center 
        row: float; real vertical distance from frame center
        frame_center: [x,y,z] point in MoCap space

    Notes:
        utilizes on Scipy's Rotation object
        .. https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html

    """

    # u_x, u_y, u_z = camera_unit_vector
    fc_x, fc_y, fc_z = frame_center

    r = Rotation.from_rotvec(camera_unit_vector * theta)

    # gaze in mocap space (NOT on screen)
    mocap_frame_gaze = r.apply([col, row, 0]) + [fc_x, fc_y, fc_z]

    return mocap_frame_gaze


def find_milestone_instances(df):
    """filter gaze locations occuring durring the appearance of an object

    average reaction time = 250ms
    discard records that are "saccade" | "blink"
    discard initial 300 ms

    Returns:
        milestone_instances: filtered instances 
        count: number removed

    """
    count = {}

    # when dflow says cross is visible AND when cross is detected
    df = df[(df["CrossVisible.Bool"] == 1) & (df["cross_score"] >= 0.9)].copy()

    count["detected_and_visible"] = len(df)

    df["time_diff"] = df["time_mc_adj"].diff() > pd.Timedelta("1s")
    df["cross#"] = df["time_diff"].apply(lambda x: 1 if x else 0).cumsum()

    event_groups = df.groupby(["cross#"])

    # remove instances before 300ms after the cross apears
    milestone_instances = event_groups.apply(
        lambda g: g[(g["time_mc_adj"] - g["time_mc_adj"].iat[0]) > pd.Timedelta("300ms")]
    )

    count["after_300ms"] = len(milestone_instances)
    count["blink_after_300ms"] = len(milestone_instances[milestone_instances["Category Binocular"] == "Saccade"])
    count["saccade_after_300ms"] = len(milestone_instances[milestone_instances["Category Binocular"] == "Blink"])
    count["visual_intake_after_300ms"] = len(milestone_instances[milestone_instances["Category Binocular"] == "Visual Intake"])

    # select only instances when ETG is labeled gaze as "Visual Intake"
    milestone_instances = milestone_instances[
        milestone_instances["Category Binocular"] == "Visual Intake"
    ]
    
    return milestone_instances, count


def est_gaze_loc_ray():

    for cross_num, instance in instances:

        opt_rotaion = rotation_dict[instance]
        frame_time = video_time.loc[(cross_num, instance)]
    
        # rotate then translate cam_model_guess by this instance's optimal rotation+translation
        cam_mocap_model = opt_rotaion["r"] @ cam_model + np.tile(opt_rotaion["t"], (cam_model.shape[1], 1)).T

        cam_mocap = cam_mocap_model[:,0]
        cam_vector_origin_mocap = cam_mocap_model[:,1]
        left_vector_origin_mocap = cam_mocap_model[:,2]

        # find frame center
        frame_center = hmpldat.utils.math.ray_cylinder_intersect(cam_vector_origin_mocap, cam_mocap)

        # find left frame center
        left_frame_center = hmpldat.utils.math.ray_plane_intersect(left_vector_origin_mocap, cam_mocap, cam_mocap - frame_center, frame_center)

        # find distance between frame_center and left_frame_center
        dist_fc_lfc = hmpldat.utils.math.euclidean_distance(frame_center, left_frame_center)
        exptd_dist_fc_lfc = hmpldat.utils.math.euclidean_distance(frame_center, cam_mocap) * np.tan(np.pi / 6)

        # record abs|sqd difference between actual(by camera FoV) and estimated length of frame_center -> left_frame_center
        d1 = abs(dist_fc_lfc - exptd_dist_fc_lfc)
        abs_diff_lfc.append(d1)

        # find horizontal left frame center
        horizontal_left_frame_center = hmpldat.align.spatial.find_horizontal_left_frame_center(cam_mocap, frame_center, dist_fc_lfc)

        # find frame rotation
        frame_rotation = hmpldat.align.spatial.find_frame_rotation(frame_center, horizontal_left_frame_center, left_frame_center, cam_mocap)

        # find pixel in terms of real distance on imaginary "frame" plane
        u, v = hmpldat.align.spatial.find_gaze_coords_frame_uv(pixel.loc[(cross_num, instance)], dist_fc_lfc)

        vec_cam = frame_center - cam_mocap
        uvec_cam = vec_cam / np.linalg.norm(vec_cam)

        # rotate pixel to frame plane in MoCap space
        # TODO: fix the return value here
        est_gaze_loc_on_frame = (
            hmpldat.align.spatial.find_frame_gaze_mocap_coords(
                frame_rotation, uvec_cam, u, v, frame_center,
            ),
        )      

        est_gaze_loc_on_frame = est_gaze_loc_on_frame[0]

        # find intersection "gaze" or "pixel" vector and screen
        estimated_location = hmpldat.utils.math.ray_cylinder_intersect(cam_mocap, est_gaze_loc_on_frame)

    return estimated_location


def estimate_gaze_location_orig(camera, origin, origin_left, binoc_por):
    """ wraps required methods to estimate gaze in mocap space """

    # print("est_gaze_loc()" + "="*75 + "\n")

    # print(f"cam={camera}")
    # print(f"camera_vector_origin={origin}")
    # print(f"left_vector_origin={origin_left}")
    # print(f"binoc_por={binoc_por.values}")

    frame_center = find_vector_screen_intersection(camera, origin)
    # print(f"frame_center={frame_center}")
    # print()

    length_camera_vector = euclidean_distance(camera, frame_center)
    # print(f"length(camera vector)={length_camera_vector}")
    # print()

    # known by camera field of view (FoV)
    length_fc_lfc = length_camera_vector * np.tan(np.pi / 6)
    # print(f"length(fc->lfc)={length_fc_lfc}")
    # print()

    horizontal_left_frame_center = find_horizontal_left_frame_center(
        camera, frame_center, length_fc_lfc,
    )

    # print(f"horizontal_left_frame_center={horizontal_left_frame_center}")
    # print()

    left_frame_center = find_left_frame_center(
        camera, origin_left, frame_center, length_fc_lfc
    )
    # print(f"l/eft_frame_center={left_frame_center}")
    # print()

    length_lfc_hlfc = euclidean_distance(
        left_frame_center, horizontal_left_frame_center
    )
    # print(f"length(lfc->hlfc)={length_lfc_hlfc}")
    # print()

    frame_rotation = find_frame_rotation(
        frame_center, horizontal_left_frame_center, left_frame_center, camera
    )
    print(np.degrees(frame_rotation))

    # if np.isnan(frame_rotation):
    #     print(f"length(fc->lfc)={length_fc_lfc}")
    #     print(f"length(lfc->hlfc)={length_lfc_hlfc}")

    #     print(length_lfc_hlfc / (2 * length_fc_lfc))
    #     input()

    # print(f"frame_rotation={frame_rotation}\n")

    u, v = find_gaze_coords_frame_uv(binoc_por, length_fc_lfc)

    unit_camera_vector = hmpldat.utils.math.unit_vector(frame_center, camera)

    # this returns point in mocap space (not on screen) for gaze location
    est_gaze_loc_on_frame = (
        find_frame_gaze_mocap_coords(
            frame_rotation, unit_camera_vector, u, v, frame_center,
        ),
    )

    est_gaze_loc_on_frame = est_gaze_loc_on_frame[0]
    # print(f"frame_gaze_loc_mocap={est_gaze_loc_on_frame}")

    # find point on screen
    est_gaze_loc_on_screen = find_vector_screen_intersection(
        est_gaze_loc_on_frame, camera
    )
    # print(f"gaze_screen={est_gaze_loc_on_screen}")

    return est_gaze_loc_on_screen


if __name__ == "__main__":

    # typical values
    o = (2490 * np.tan(np.pi / 6)) - 20
    gaze = pd.DataFrame([20, 20])

    # print(find_gaze_coords_frame_uv(gaze, o))
