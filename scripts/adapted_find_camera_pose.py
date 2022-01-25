"""Method to find the optimal camera model

This method utilizes ray tracing

"""

import argparse
from pathlib import Path
from pprint import pprint
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
import xarray as xr

import hmpldat.align.spatial
import hmpldat.utils.math
import hmpldat.utils.glasses
import hmpldat.file.search as search
import hmpldat.file.detected
import hmpldat.utils.plot
import hmpldat.utils.filter

import hmpldat.file.participant


# not so rando
RANDOM_STATE = 12345

CCCC = 0

DATA_PATH = Path("./sample_data/merged")

XY_PLANE = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0])]

context = {"0": "annealing", "1": "local search", "2": "dual annealing"}

# will hold any commandline arguments
FLAGS = None

PARTICIPANT_ID = r"([a-zA-Z]{4}_[0-9]{3})"

Z_PLANE = -1000


def callback(xk, a=None, c=None):
    """
    To be called after each iteration of the optimizer

    Plot something?

    """

    cam_model = pd.DataFrame(
        np.array([xk[:3], xk[3:6], xk[6:]]).T,
        index=["x", "y", "z"],
        columns=["cam", "cvo", "lvo"],
    )

    print()
    if not c is None:
        print(context[str(c)])

    print(cam_model)


def objective(x, *args):
    """
    The function we are minimizing

    Args:
        x: current guess
        args: other data required for calculation

    Returns:
        list: total error for that guess across a set of instances
    
    """
    # callback(x)
    # print(args)

    # unpack current guess (in camera space)
    cam = np.array(x[:3], dtype=np.float64)

    # use for bounded, but not fixed z value and only 2 vector origin points
    cam_vector_origin = np.array(x[3:6], dtype=np.float64)
    left_vector_origin = np.array(x[6:], dtype=np.float64)


    # cam_vec_pass_thru = np.append(x[3:5], Z_PLANE).astype(np.float64)
    # left_vec_pass_thru = np.append(x[5:7], Z_PLANE).astype(np.float64)
    # top_vec_pass_thru = np.append(x[7:], Z_PLANE).astype(np.float64)
    
    # reshape
    cam_model = np.array([cam, cam_vec_pass_thru, left_vec_pass_thru, top_vec_pass_thru]).T

    # unpack args
    instances, pixel, projection, rotation_dict, etg_model, video_time = args

    # instances = index of samples to eval
    # pixel = gaze location or center of detected object
    # projection = object position on screen
    # rotation_dict = dictionary of optimal rotations for each instance (where key == index)
    # etg_model = glasses in glasses space

    abs_diff_lfc = []
    sq_distance = []

    # for each instance
    for instance in instances:

        opt_rotation = rotation_dict[instance]
        frame_time = video_time.loc[instance]

        # rotate then translate cam_model_guess by this instance's optimal rotation+translation
        cam_mocap_model = (
            opt_rotation["r"] @ cam_model
            + opt_rotation["t"].reshape(3,1) # into a column vector
            # + np.tile(opt_rotation["t"], (cam_model.shape[1], 1)).T
        )

        cam_mocap = cam_mocap_model[:, 0]
        cam_vec_pass_thru_mocap = cam_mocap_model[:, 1]
        left_vec_pass_thru_mocap = cam_mocap_model[:, 2]
        top_vec_pass_thru_mocap = cam_mocap_model[:, 3]

        # find frame center
        frame_center = hmpldat.utils.math.ray_cylinder_intersect(
            cam_vector_origin_mocap, cam_mocap
            # cam_mocap, cam_vec_pass_thru_mocap
        )

        # find left frame center
        left_frame_center = hmpldat.utils.math.ray_plane_intersect(
            left_vector_origin_mocap, cam_mocap, cam_mocap - frame_center, frame_center
            # cam_mocap, left_vec_pass_thru_mocap, cam_mocap - frame_center, frame_center
        )

        # find top frame center
        top_frame_center = hmpldat.utils.math.ray_plane_intersect(
            left_vector_origin_mocap, cam_mocap, cam_mocap - frame_center, frame_center
            # cam_mocap, top_vec_pass_thru_mocap, cam_mocap - frame_center, frame_center
        )

        # find distance between frame_center and left_frame_center
        dist_fc_lfc = hmpldat.utils.math.euclidean_distance(
            frame_center, left_frame_center
        )
        exptd_dist_fc_lfc = hmpldat.utils.math.euclidean_distance(
            frame_center, cam_mocap
        ) * np.tan(np.pi / 6)

        # # find distance between frame_center and top_frame_center
        # dist_fc_tfc = hmpldat.utils.math.euclidean_distance(
        #     frame_center, top_frame_center
        # )
        # exptd_dist_fc_tfc = hmpldat.utils.math.euclidean_distance(
        #     frame_center, cam_mocap
        # ) * np.tan((23 * np.pi) / 180) 

        # # record abs|sqd difference between actual(by camera FoV) and estimated length of frame_center -> left_frame_center
        # d1 = abs(dist_fc_lfc - exptd_dist_fc_lfc)
        # abs_diff_lfc.append(d1)

        # find horizontal left frame center
        horizontal_left_frame_center = hmpldat.align.spatial.find_horizontal_left_frame_center(
            cam_mocap, frame_center, dist_fc_lfc
        )

        # find frame rotation
        frame_rotation = hmpldat.align.spatial.find_frame_rotation(
            frame_center, horizontal_left_frame_center, left_frame_center, cam_mocap
        )

        # # find distance from camera to frame_center
        # dist_cam_to_frame_center = hmpldat.utils.math.euclidean_distance(
        #     cam_mocap, frame_center
        # )

        # find pixel in terms of real distance on imaginary "frame" plane
        u, v = hmpldat.align.spatial.find_coords_on_scene_frame(
            pixel.loc[instance], 
            dist_fc_lfc,
            dist_fc_lfc,
            # dist_fc_tfc,
        )

        # calculate camera unit vector
        vec_cam = frame_center - cam_mocap
        uvec_cam = vec_cam / np.linalg.norm(vec_cam)

        # rotate pixel to frame plane in MoCap space
        est_loc_on_frame = hmpldat.align.spatial.find_frame_mocap_coords(
            frame_rotation, uvec_cam, u, v, frame_center,
        )

        # find intersection of vector and screen
        estimated_location = hmpldat.utils.math.ray_cylinder_intersect(
            cam_mocap, est_loc_on_frame
        )

        d2 = hmpldat.utils.math.euclidean_distance(
            estimated_location, projection.loc[instance], squared=True
        )
        sq_distance.append(d2)

    # return np.sum([abs_diff_lfc, sq_distance], axis=0)  # row-wise sum
    return sq_distance


def main():
    """

    find and open data files

    optimize camera model for participant

    plot & record results

    """

    # # open previously created glasses models
    # etg_model_file = Path("./sample_data/test_etg_models.csv")

    # dictionaries to record errors and "optimal" camera models
    err = {}
    camera_models = {}

    # NOT USED 
    prev_model = None

    # used to break loop early to check output format
    counter = 0 

    b = ('VMIB', '005')
    a = ('VMIB', '047')

    # for each merged data file
    # for f in search.files(DATA_PATH, []):  # a_task or all_tasks_from_one_session
    for study, participant in [b]:

        p = hmpldat.file.participant.participant(study, participant)

        p.rawetg_path = Path(f'~/Desktop/selected_data/VMIB/Data/ETG/Metrics Export/VMIB_{participant}_RawETG.txt')
        p.detections_path = Path(f'~/Desktop/selected_data/VMIB/Data/detections/vmib_{participant}-1-unpack.txt')

        # p.create_file_listing(Path('selected_data_file_listing.xlsx'), Path('/home/ian/Desktop/selected_data/VMIB/Data'))
        p.load_file_listing(
            Path('/home/ian/Projects/hmpldat/selected_data_file_listing.xlsx')
        )

        p.load_rawetg_and_objects()
        
        p.create_tasks()
        output_data_dir = Path(f'/home/ian/Projects/hmpldat/temporal_alignment_data/{p.experiment}/{p.name}')

        # don't overwrite data by default
        if output_data_dir.exists():
            i = input("directory already exists. continue? [Y/n] ")
            if i.lower() == 'n':
                exit()
        else:
            output_data_dir.mkdir(parents=True)

        # TODO: keep n samples across all tasks -> one camera model per participant
        # samples = 

        # for each task, sample
        for task in p.tasks.values():
            print(task.name, type(task), hasattr(task, 'merged'))

            if task.name != 'int':
                continue
            # TODO: keep n sample per task -> one camera model per task
            # samples = 

            # skip tasks without merged data
            if not hasattr(task, 'merged'):
                continue
            
            task.merged.to_csv(output_data_dir / ("_".join([task.name, str(task.trial)]) + '.csv'))
            # print(task.merged[['frame_number', 'Point of Regard Binocular X [px]', 'Annotation Name', 'cross.u', 'crossvisible.bool']].head(25))

            # drop instances with no mocap data (cortex typically starts recording shortly after dflow)
            # and any time when head markers are not recorded
            # TODO: look at cortex files for any NaN value prior to merge

            # remove rows with NaN values 
            # important columns lhead, rhead, thead, 
            # required_columns = task.merged.loc[:, task.merged.columns.str.contains('.head|target(\.|v)|cross(\.|v)|user(\.|v)|safezone|grid')].columns
            # cleaned = task.merged.dropna(subset=required_columns, thresh=29)
            
            # cleaned.to_csv(output_data_dir / ("_".join([task.name, str(task.trial), 'cleaned']) + '.csv'))

            # print(task.horizon_height())

            # exit()

        continue

        # OLD WAY FOR COMPARISON
        # define projection height as first recorded rhead.y value OF EACH TASK
        # is used to calculate object projection on to screen
        # old_task_height = task_height.assign(
        #     first_recorded_rhead_y=df.groupby("task_name").nth([0])["RHEAD.Y"],
        # )
        # print(task_height, old_task_height, task_height - old_task_height)

        # create and load corresponding glasses model
        # g = hmpldat.utils.glasses.etg_model()
        # ret = g.load(etg_model_file, f.name)
        # if not ret:
        #     input("ERROR: corresponding glasses model not found in file {}")
        
        # TODO: automatically handle whether we have glasses markers: [fhead, lhead, rhead] OR [GL1, GL2, GL3, GR1, GR2, GR3]
        # try except else??
        g = hmpldat.utils.glasses.etg_model()
        g.new_define(mocap[["fhead", "lhead", "rhead"]], f.name)
        # print(g.in_glasses_space)

        # specify initial camera model
        # our best guess method based off of the glasses shape
        # initial_best_guess = g.cam_best_guess_pass_thru()

        # replace values I am bounding to z=-1000 to nans (I drop these values later)
        # initial_best_guess = initial_best_guess.replace(to_replace=-1000, value=np.NaN)
        # print(initial_best_guess)

        # evaluate using average optimized result as best guess (across multiple participants)
        # excluding vmib_005
        initial_best_guess = (
            pd.DataFrame(
                [
                    [91.1725039925567, 39.8040907941888, -15.8159464100071],
                    [93.4250580223207, 2.47495082434566, 9.90577755878422e-11],
                    [112.987914620579, 4.10444705124924, -7.85793882423784e-10]
                ],
                columns=["x", "y", "z"],
                index=["cam", "cvo", "lvo"],
                ).T
            )
        print(initial_best_guess)

        ### SAMPLING CROSSES
        # crosses = vr["cross"]
        # crosses.columns = pd.MultiIndex.from_tuples(
        #     [("visible", ""), ("position", "x"), ("position", "y"), ("position", "z")]
        # )
        # dflow_crosses = hmpldat.utils.filter.each_trial(crosses)

        # detected_crosses = hmpldat.utils.filter.detected(detected["cross"])
        # detected_crosses.columns = pd.MultiIndex.from_product(
        #     [["detected"], detected_crosses.columns]
        # )

        # detected_crosses = hmpldat.utils.filter.centered_in_frame(detected_crosses, 120)

        # # inner join (both left and right indexes must match)
        # # therefore object must be both visible according to dflow, and detected in the associated image frame.
        # cr_df = pd.merge(
        #     dflow_crosses, detected_crosses, left_index=True, right_index=True
        # )

        # # new attribute to remember which object this data came from
        # cr_df = cr_df.assign(object="cross")

        # # sample first and last visible cross
        # sampled_cross_instances = cr_df.groupby("trial_number", as_index=False).nth(
        #     [0, -1]
        # )

        # # join sampled instances with corresponding data
        # df = sampled_cross_instances.join(mocap).join(gaze).join(times).join(task_name)
        # print(df)

        ### SAMPLING TARGETS
        # select instances with a visible target according to dflow
        targets = vr["target"]
        targets.columns = pd.MultiIndex.from_tuples(
            [("visible", ""), ("position", "x"), ("position", "y"), ("position", "z")]
        )
        targets.to_csv("targets_example.csv")

        # TODO: use assign to append this attribute 
        dflow_targets = hmpldat.utils.filter.each_trial(targets)
        # print(dflow_targets)

        # select targets that are detected (and bbx is approx square)
        detected_targets = hmpldat.utils.filter.detected(detected["target"])

        # remove any objects at the edge of the image frame
        edge = hmpldat.utils.filter.at_frame_edge(detected_targets, 5).any(axis=1)

        # multi-level column names to match other dataframes
        detected_targets.columns = pd.MultiIndex.from_product(
            [["detected"], detected_targets.columns]
        )
        # print(detected_targets)

        # remove instances where detected object is on the edge of the captured frame
        detected_targets = detected_targets[edge == False]

        # inner join (both left and right indexes must match)
        target_df = pd.merge(
            dflow_targets, detected_targets, left_index=True, right_index=True
        )
        # print(target_df)

        # where sectors is a dataframe and bins is a tuple containing (vertical bins, horizontal_bins)
        sectors, bins = hmpldat.utils.filter.sector_of_frame(
            target_df["detected"], row_bins=FLAGS.row_bins, col_bins=FLAGS.col_bins
        )
        # bins defined during frame sectoring (not used)
        # vbins, hbins = bins 

        samples = []
        # sample_info = {}

        # sample 60 from each sector (if less than 60 exist take them all)
        for grp, sec in sectors.groupby(["sector_col", "sector_row"]):

            n = 60

            if len(sec) < 60:
                n = len(sec)

            # print(f"sampling{n} from {grp}")

            samples.append(sec.sample(n, random_state=RANDOM_STATE))

        sampled_target_instances = pd.concat(samples)
        sampled_target_instances.columns = pd.MultiIndex.from_product(
            [sampled_target_instances.columns, [""]]
        )

        # left join other data to sampled instances
        df = (
            sampled_target_instances.join(target_df)
            .join(mocap)
            .join(gaze)
            .join(times)
            .join(task_name)
        )
        # print(df)

        # print(df[["position", "task_name"]])

        # calculate projection on screen
        projection = df.apply(
            lambda row: hmpldat.file.dflow.project_object_onto_screen(
                row["position"]["x"],
                row["position"]["y"],
                row["position"]["z"],
                task_height["calc_via_cross_pos"][row["task_name"]].values[0],
            ),
            result_type="expand",
            axis=1,
        )
        projection.columns = pd.MultiIndex.from_product(
            [["projection"], ["x", "y", "z"]]
        )

        # join data
        df = df.join(projection)
        # print(df.stack(-1))

        # # my_first_xarray = xr.Dataset.from_dataframe(df.stack(-1))
        # # print(my_first_xarray)

        # print(my_first_xarray[["fhead", "lhead", "rhead"]])
        # input()

        # calculate and record optimal rotation for each sampled instance
        rotation_dict = {}
        for i, instance in df[["fhead", "lhead", "rhead"]].iterrows():
            r, t = hmpldat.utils.math.rigid_transform_3D(
                g.in_glasses_space.values, instance.unstack(0).values
            )
            rotation_dict[i] = {"r": r, "t": t}

        rotation_array = np.array(rotation_dict.values())


        x0 = initial_best_guess.unstack().copy()
        x0 = x0.dropna()  # drop nan values (what I set the z values to)
        # print(x0)
        # print(x0.index)

        ### define bounds
        # bounds = pd.DataFrame(
        #     [
        #         [-75, 75], # cam.x
        #         [-50, 50], # cam.y
        #         [-75, 75], # cam.z
        #         [initial_best_guess["left_vec_pass_thru"]["x"],-initial_best_guess["left_vec_pass_thru"]["x"]], # cvpt.x
        #         [-initial_best_guess["top_vec_pass_thru"]["y"],initial_best_guess["top_vec_pass_thru"]["y"]], # cvpt.y
        #         # cvpt.z is fixed at Z_PLANE
        #         [2*initial_best_guess["left_vec_pass_thru"]["x"] , 0], # lvpt.x
        #         [-initial_best_guess["top_vec_pass_thru"]["y"],initial_best_guess["top_vec_pass_thru"]["y"]], # lvpt.y
        #         # lvpt.z is fixed at Z_PLANE
        #         [initial_best_guess["left_vec_pass_thru"]["x"],-initial_best_guess["left_vec_pass_thru"]["x"]], # tvpt.x
        #         [0,2*initial_best_guess["top_vec_pass_thru"]["y"]], # tvpt.y
        #         # tvpt.z is fixed at Z_PLANE
        #     ],
        #     columns=["lower_bound", "upper_bound"],
        #     index=x0.index,
        # )
        # print(bounds.to_markdown())

        ### define bounds (NO top frame center)
        bounds = pd.DataFrame(
            [
                [0, np.inf],
                [0, 60],
                [-30, 0],
                [0, np.inf],
                [-np.inf, np.inf],
                [-1e-8, 1e8],
                [0, np.inf],
                [-np.inf, np.inf],
                [-1e-8, 1e8],
            ],
            columns=["lower_bound", "upper_bound"],
            index=pd.MultiIndex.from_tuples(initial_best_guess.unstack().index),
        )
        print(bounds)

        # split data into train and test sets
        # train, test = train_test_split(
        #     df, test_size=1 / 3, random_state=RANDOM_STATE, shuffle=False
        # )

        # No test/train split
        train = df
        test = df

        max_iters = 1

        for i in range(max_iters):

            print()
            print(f"\titeration #{i}")

            res = scipy.optimize.least_squares(
                objective,
                x0,
                bounds=(bounds["lower_bound"], bounds["upper_bound"]),
                # method="dogbox", # [default="trf", "dogbox", "lm"]
                xtol=1e-4,  # default=1e-8
                # loss="soft_l1",  # [default="linear", "soft_l1", "huber", "cauchy", "arctan"]
                x_scale="jac",
                args=(
                    train.index,
                    train["detected"][["ctr_bb_col", "ctr_bb_row"]],  # df["gaze"]
                    train["projection"],
                    rotation_dict,
                    g.in_glasses_space,
                    train["time"],
                ),
                verbose=2,
            )

            x = res.x
            # print(x)

            # split 1D vector into the four points it holds
            inter_cam = np.array(x[:3], dtype=np.float64)
            inter_cvpt = np.append(x[3:5], Z_PLANE).astype(np.float64)
            inter_lvpt = np.append(x[5:7], Z_PLANE).astype(np.float64)
            inter_tvpt = np.append(x[7:], Z_PLANE).astype(np.float64)

            # print("intermediate camera model")
            intermediate_cam_model = pd.DataFrame(
                [inter_cam, inter_cvpt, inter_lvpt, inter_tvpt],
                columns=["x", "y", "z"],
                index=["cam", "cvpt", "lvpt", "tvpt"],
            ).T
            print(intermediate_cam_model)

            # record "optimal" camera model to file
            camera_models[participant] = intermediate_cam_model

            # use objective function to calculate error on test set
            on_test = objective(
                x,
                test.index,
                test["detected"][["ctr_bb_col", "ctr_bb_row"]],  # df["gaze"]
                test["projection"],
                rotation_dict,
                g.in_glasses_space,
                test["time"],
            )

            on_test = (
                (
                    pd.DataFrame(
                        on_test, columns=["error"], index=test.index
                    )
                    ** 0.5
                )
                .join(sectors)
                .join(task_name["task_name"])
            )

            # record info for each participant
            err[participant] = on_test

            # remove instances > mean + std from actual (to be used during next optimization iterations
            # use_instances = (
            #     dist_btw_kwn_and_est[dist_btw_kwn_and_est.lt(mean + 1 * std)]
            #     .dropna()
            #     .index
            # )

        # use to cut off early and check output
        # counter += 1
        # if counter == 3:
        #     break

    # combine errors from each participants data
    err_df = pd.concat(err, names=["participant"]).reset_index()

    # combined average error
    by_sector_err = err_df.groupby(["sector_row", "sector_col"]).mean()
    by_sector_size = err_df.groupby(["sector_row", "sector_col"]).size()
    by_sector_size.name = "count"
    # print(by_sector_err)
    # print(by_sector_size)

    # individual average error
    by_sector_and_participant_err = err_df.groupby(["participant", "sector_row", "sector_col"]).mean()
    by_sector_and_participant_size = err_df.groupby(["participant", "sector_row", "sector_col"]).size()
    by_sector_and_participant_size.name = "count"
    # print(by_sector_and_participant_err)
    # print(by_sector_and_participant_size)

    # each "optimal" camera model 
    camera_models_df = pd.concat(camera_models)
    camera_models_df = camera_models_df.swaplevel().sort_index()
    # print(camera_models_df)

    mean_cam_model = camera_models_df.mean(axis=0, level=0)
    median_cam_model = camera_models_df.median(axis=0, level=0)
    # print(mean_cam_model)
    # print(median_cam_model)
    simple_diff_cam_models_and_mean = camera_models_df.subtract(mean_cam_model, axis=0, level=0)
    simple_diff_cam_models_and_median = camera_models_df.subtract(median_cam_model, axis=0, level=0)
    # print(simple_diff_cam_models_and_mean)
    # print(simple_diff_cam_models_and_median)

    # combine error and counts
    mean_err_by_sector = (
        pd.concat([by_sector_err, by_sector_size], axis=1)
        .unstack()
        .swaplevel(axis=1)
        .sort_index(axis=1)
    )
    # print(mean_err_by_sector)
    
    # combine error and counts
    mean_err_by_participant_and_sector = (
        pd.concat([by_sector_and_participant_err, by_sector_and_participant_size], axis=1)
        .unstack()
        .swaplevel(axis=1)
        .sort_index(axis=1)
        .swaplevel(axis=0)
        .sort_index(axis=0)        
    )
    # print(mean_err_by_participant_and_sector)

    # write to excel file
    with pd.ExcelWriter(
        "out.xlsx", engine="xlsxwriter", options={"nan_inf_to_errors": True}
    ) as writer:
        mean_err_by_sector.to_excel(writer, float_format="%.0f", sheet_name="error_by_sector")
        mean_err_by_participant_and_sector.to_excel(writer, float_format="%.0f", sheet_name="error_by_participant&sector")
        camera_models_df.to_excel(writer, float_format="%.0f", sheet_name="camera_models")
        simple_diff_cam_models_and_median.to_excel(writer, float_format="%.0f", sheet_name="diff_camera_models_and_median")
        simple_diff_cam_models_and_mean.to_excel(writer, float_format="%.0f", sheet_name="diff_camera_models_and_mean")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--row_bins", default=5, type=int,
    )

    parser.add_argument(
        "--col_bins", default=5, type=int,
    )

    # ignore any unknown arguments
    FLAGS, _ = parser.parse_known_args()

    main()
