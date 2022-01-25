"""
Contians methods from original trig ppt without any boundary checking

"""

import numpy as np
import pandas as pd

import hmpldat.utils.math

SCREEN_RADIUS = 2490


def find_vector_sceen_intersect(
    camera, origin,
):
    """
    simple ricks: when life is simple

    ref: slide 1
    """

    cam_x, cam_y, cam_z = camera
    origin_x, origin_y, origin_z = origin

    # slope in xz_plane
    rise = cam_z - origin_z
    run = cam_x - origin_x

    xz_dist_o_cam = hmpldat.utils.math.euclidean_distance(
        [origin_x, origin_z], [cam_x, cam_z]
    )

    xz_slope = rise / run
    n = cam_x - cam_z / xz_slope

    a = 1 + 1 / xz_slope ** 2
    b = (2 * n) / xz_slope
    c = n ** 2 - SCREEN_RADIUS ** 2

    possible_intersect_z = hmpldat.utils.math.quadratic_formula(a, b, c)
    # print(f"zs={possible_intersect_z}")

    # one soln | no soln
    if isinstance(possible_intersect_z, float):

        # no soln
        if np.isnan(possible_intersect_z):
            return np.NaN, np.NaN, np.NaN

        intersect_z = possible_intersect_z
        intersect_x = (intersect_z - cam_z) / xz_slope + cam_x

    ### choose a solution
    # based on comparing unit vector directions
    else:

        possible_intersect_x = [
            (intersect_z - cam_z) / xz_slope + cam_x
            for intersect_z in possible_intersect_z
        ]

        soln = []

        # calculate both solutions
        for intersect_x, intersect_z in zip(possible_intersect_x, possible_intersect_z):

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

    # we need to solve quadratic

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
        hlfc_x, hlfc_z, _ = min(solns, key=lambda x: abs(x[2]))

        # print(f"\t(x,z,angle) {list(zip(poss_hlfc_x, poss_hlfc_z, soln_angle - expected_ang_to_lfc))}\n")

    return hlfc_x, fc_y, hlfc_z


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
        left frame center in MoCap space

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
        unit_vec_cam_soln_a = hmpldat.utils.math.unit_vector(soln_a, (ol_x, ol_y, ol_z))
        unit_vec_cam_soln_b = hmpldat.utils.math.unit_vector(soln_b, (ol_x, ol_y, ol_z))

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

    # some of my data is not a np.array
    fc = np.array(fc)
    # hlfc = np.array(hlfc)
    lfc = np.array(lfc)
    # cam = np.array(cam)

    vec_fc_lfc = lfc - fc
    vec_fc_hlfc = hlfc - fc
    vec_hlfc_lfc = lfc - hlfc
    vec_cam = fc - cam

    uvec_cam = vec_cam / np.linalg.norm(vec_cam)
    # print(uvec_cam)

    vec_dir = np.cross(vec_fc_hlfc, vec_hlfc_lfc)
    uvec_dir = vec_dir / np.linalg.norm(vec_dir)
    # print(uvec_dir)

    len_lfc_hlfc = np.linalg.norm(vec_hlfc_lfc)
    len_fc_lfc = np.linalg.norm(vec_fc_lfc)

    # if directions are the same (or similar?) make len_lfc_hlfc negative
    # else nothing (leave len_lfc_hlfc positive) (sanity check: directions should be opposite)
    # this will correctly roll the head left or right
    if np.allclose(uvec_cam, uvec_dir):
        len_lfc_hlfc = -len_lfc_hlfc

    theta = 2 * np.arcsin(len_lfc_hlfc / (2 * len_fc_lfc))

    return theta
