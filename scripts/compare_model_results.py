"""Method to find the optimal camera model

This method utilizes ray tracing

"""

from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

import hmpldat.align.spatial
import hmpldat.utils.math
import hmpldat.utils.glasses
import hmpldat.file.search as search
import hmpldat.file.detected
import hmpldat.utils.plot


DATA_PATH = Path("./sample_data/merged")


def main():
    """

    find and open data files

    optimize camera model for participant

    plot & record results

    """

    # previously created camera and glasses models
    camera_model_file = Path("./sample.csv")
    etg_model_file = Path("./sample_data/test_etg_models.csv")

    camera_models = pd.read_csv(camera_model_file, index_col=0, header=[0,1],skiprows=[2])
    etg_models = pd.read_csv(etg_model_file, index_col=0)
    etg_models.columns = pd.MultiIndex.from_tuples([col.split('.') for col in etg_models.columns])
    etg_models = etg_models["glasses_space"]

    print(camera_models)

    models = pd.merge(camera_models, etg_models, left_index=True, right_index=True)
    models.index.name = "generated from file:"
    print(models)

    models.to_excel("camera_and_etg_models.xls")

    info = {}

    # for each merged data file
    for f in search.files(DATA_PATH, []): #a_task or all_tasks_from_one_session

        info[f.name] = {}

        print(f"\ndata: {f.name}")
        # input("\tpress ENTER to continue\n\n")

        # open data
        df = pd.read_csv(f, low_memory=False)

        ### handle data quirks
        ### this piece will be removed in the future
        df[df.filter(like=".Pos").columns] = (
            df.filter(like=".Pos") * 1000
        )  ## convert m to mm
        # print(df.filter(like="Cross.Pos"))

        df["time_mc_adj"] = pd.to_timedelta(df["time_mc_adj"])

        # define projection height as first recorded rhead.y value OF EACH TASK
        # is used to calculate object projection on to screen
        y_views = df.dropna(subset=["RHEAD.Y"]).groupby("task_name").nth([0])["RHEAD.Y"]
        # print(y_views)

        # drop instances with no mocap data (cortex typically starts recording shortly after dflow)
        # other times when head markers are not
        df = df.dropna(subset=['Frame#', "FHEAD.X", "LHEAD.X", "RHEAD.X", "Point of Regard Binocular X [px]"])

        # adding count to info dictionary total instance count
        # info[f.name]["dflow_cross_instance_count"] = len(df[df["CrossVisible.Bool"]==1])

        # define milestone instances
        df, num_drpd = hmpldat.file.detected.milestone(df)
        # df, num_drpd = hmpldat.align.spatial.find_milestone_instances(df)

        # pprint(num_drpd)             
        # print(df)

        # add counts to info dict
        # info[f.name].update(num_drpd)

        # Histogram crap
        # df["zeroed_time"] = df["time_mc_adj"].sub(df.groupby(level=[0], axis="index")["time_mc_adj"].first())
        # df["zeroed_time"] = df["zeroed_time"] / pd.Timedelta(milliseconds=1)

        # saccades = df[df["Category Binocular"] == "Saccade"]["zeroed_time"]

        # print(saccades)

        # print(saccades.groupby(level=[0], axis="index").hist(bins=range(0,700,50), normed=1))
        # plt.show()
        # exit

        # (zerod_start / pd.Timedelta(milliseconds=1)).groupby(level=[0], axis="index").hist(bins=range(0,800,50))
        # plt.show()
        # print(sac)
        
        # sample data first and last instance
        df = df.groupby(level=[0], axis="index", as_index=False, group_keys=False).nth([0,-1])

        # organize data
        mocap = hmpldat.file.cortex.multiindex(df)
        vr = hmpldat.file.dflow.multiindex(df)

        etg_mocap = mocap[["fhead", "lhead", "rhead"]]
        task_name = df[["task_name"]]
        task_name.columns = pd.MultiIndex.from_product([["cross"], task_name.columns])
        cross_position = vr[["cross"]].merge(task_name, left_index=True, right_index=True)
        # print(cross_position)

        gaze = df[
            ["Point of Regard Binocular X [px]", "Point of Regard Binocular Y [px]"]
        ]
        gaze.columns = pd.MultiIndex.from_product([["gaze"], ["x", "y"]])

        times = df[["Video Time [h:m:s:ms]", "time_mc"]]
        times.columns = pd.MultiIndex.from_product([["time"], ["video", "MoCap"]])
        
        cross_px = df[["cross_ctr_bb_col", "cross_ctr_bb_row"]]
        cross_px.columns = pd.MultiIndex.from_product([["detected_cross_px"], ["x", "y"]])

        # calculate cross projection on screen
        # cross_projection["cross_projection", "x"], cross_projection["cross_projection", "y"], cross_projection["cross_projection", "z"]  = 
        cross_projection = cross_position["cross"].apply(
            lambda row: hmpldat.file.dflow.project_object_onto_screen(
                row["x"], row["y"], row["z"], y_views.loc[row["task_name"]]
            ),
            result_type='expand',
            axis=1,
        )
        cross_projection.columns = pd.MultiIndex.from_product([["cross_projection"], ["x","y","z"]])
        # print(cross_position)
        # print(cross_projection)

        # for each model
        for participant, model in models.iterrows():

            # participant = "p_" + participant
            info[f.name][participant] = {}

            print(f"model: {participant}")

            model = model.unstack(0)

            g = model[['fhead','lhead','rhead']]
            cam = model[['cam','cvo','lvo']]

            # print(g)
            # print(cam)

            # calculate and record optimal rotation for each instance
            rotation_dict = {}
            for i, instance in etg_mocap.groupby(level=1):  
                r, t = hmpldat.utils.math.rigid_transform_3D(
                    g, instance.stack()
                )
                rotation_dict[i] = {"r": r, "t": t}
        
            # use detected cross center instead of gaze
            gaze = cross_px

            dist_btw_kwn_and_est=[]
            diff_btw_kwn_and_est=[]

            # for all sampled instances
            for cross_num, instance in gaze.index:

                ### using camera model calculate 3d screen location from pixel coordinates.
                opt_rotaion = rotation_dict[instance]
                frame_time = times.loc[(cross_num, instance)]
            
                # rotate then translate intermediate camera model by this instance's optimal rotation+translation
                cam_mocap_model = opt_rotaion["r"] @ cam + np.tile(opt_rotaion["t"], (cam.shape[1], 1)).T

                cam_mocap = cam_mocap_model.values[:,0]
                cam_vector_origin_mocap = cam_mocap_model.values[:,1]
                left_vector_origin_mocap = cam_mocap_model.values[:,2]

                # find frame center
                frame_center = hmpldat.utils.math.ray_cylinder_intersect(cam_vector_origin_mocap, cam_mocap)

                # find left frame center
                left_frame_center = hmpldat.utils.math.ray_plane_intersect(left_vector_origin_mocap, cam_mocap, cam_mocap - frame_center, frame_center)

                # find distance between frame_center and left_frame_center
                dist_fc_lfc = hmpldat.utils.math.euclidean_distance(frame_center, left_frame_center)

                # find horizontal left frame center
                horizontal_left_frame_center = hmpldat.align.spatial.find_horizontal_left_frame_center(cam_mocap, frame_center, dist_fc_lfc)

                # find frame rotation
                frame_rotation = hmpldat.align.spatial.find_frame_rotation(frame_center, horizontal_left_frame_center, left_frame_center, cam_mocap)

                # find pixel in terms of real distance on imaginary "frame" plane
                u, v = hmpldat.align.spatial.find_gaze_coords_frame_uv(gaze.loc[(cross_num, instance)], dist_fc_lfc)

                vec_cam = frame_center - cam_mocap
                uvec_cam = vec_cam / np.linalg.norm(vec_cam)

                # rotate pixel to frame plane in MoCap space
                est_gaze_loc_on_frame = hmpldat.align.spatial.find_frame_gaze_mocap_coords(
                        frame_rotation, uvec_cam, u, v, frame_center,
                    )

                # find intersection "gaze" or "pixel" vector and screen
                estimated_location = hmpldat.utils.math.ray_cylinder_intersect(cam_mocap, est_gaze_loc_on_frame)

                # find euclidean distance between actual and calculated
                d = hmpldat.utils.math.euclidean_distance(estimated_location, cross_projection.loc[(cross_num, instance)])
                dist_btw_kwn_and_est.append(d)
                
                d2 = estimated_location - cross_projection.loc[(cross_num, instance)]
                diff_btw_kwn_and_est.append(d2)

            diff_btw_kwn_and_est = pd.DataFrame(diff_btw_kwn_and_est, index=gaze.index)
            dist_btw_kwn_and_est = pd.DataFrame(dist_btw_kwn_and_est, index=gaze.index)

            # descriptive stats and box plot
            # print(dist_btw_kwn_and_est.describe().to_markdown())
            # print(diff_btw_kwn_and_est.describe().to_markdown())

            mean = dist_btw_kwn_and_est.mean() 
            std = dist_btw_kwn_and_est.std()

            s = f"iter #{i}, mean"
            info[f.name][participant]['mean'] = mean.values[0]
            s = f"iter #{i}, std"
            info[f.name][participant]['std'] = std.values[0]

            # boxplot
            # ax = dist_btw_kwn_and_est.boxplot(return_type='axes')
            # ax.set_ylabel("millimeters")
            # plt.suptitle("Distance between expected and estimated")
            # plt.show()

    pprint(info)
           
    info_df = pd.concat({k: pd.DataFrame(v) for k, v in info.items()})
    info_df = info_df.unstack()
    info_df.index.name = "data sampled from:"

    with pd.ExcelWriter("model_comparison.xls") as writer:

        std = info_df.xs("std", axis=1, level=1)
        mean = info_df.xs("mean", axis=1, level=1)
        
        info_df.to_excel(writer, sheet_name='all')
        mean.to_excel(writer, sheet_name='mean')
        std.to_excel(writer, sheet_name='std')
    

    pprint(info_df)   



if __name__ == "__main__":
    main()
