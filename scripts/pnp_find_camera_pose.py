
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import cv2

import hmpldat.file.search as search
import hmpldat.file.detected 


cam = np.array([
    [960, 0, 480],
    [0, 720, 360],
    [0, 0, 1]
    ])


def main():
    
    pprint(cam)

    for f in search.files(Path("/home/raphy/proj/hmpldat/sample_datas/merged"), ["avoid"]):

        print(f.name)

        df = pd.read_csv(f)

        # make columns simpiler to access
        detected_objects_df = hmpldat.file.detected.multiindex_object_columns(df)
        vr_objects_df = hmpldat.file.dflow.multiindex(df).sort_index(axis='columns', level=0)
        mocap = hmpldat.file.cortex.multiindex(df)

        # change meter to mm
        vr_objects_df.loc[:, (slice(None), ['x','y','z'])] = vr_objects_df.loc[:, (slice(None), ['x','y','z'])] * 1000

        # print(detected_objects_df)
        # print(vr_objects_df)

        # input()
        time_col = df["Video Time [h:m:s:ms]"]

        df = pd.concat([detected_objects_df.loc[:, (slice(None), ['ctr_bb_col', 'ctr_bb_row'])], vr_objects_df, mocap], axis=1).sort_index(axis='columns', level=0)
        # print(df[["cross", "user", "target", "grid", "safezone"]].head())

        df = df[["cross", "user", "target", "grid", "safezone"]]

        # for each instance
        for i, row in df.iterrows():

            # we need 3 points
            count = 0 

            frame_points = []
            mocap_points = []

            # collect each correspondance
            for objt in df.columns.get_level_values(0):

                if any(np.isnan(row[objt])):
                    continue
                else: 
                    count += 1
                    frame_points.append([row[(objt, "ctr_bb_col")], row[(objt, "ctr_bb_row")]])
                    mocap_points.append([row[(objt, "x")], row[(objt, "y")], row[(objt, "z")]])

            mocap_points = np.array(mocap_points)
            frame_points = np.array(frame_points)

            if count >= 3:
                retval, rvec, tvec = cv2.solvePnP(mocap_points, frame_points, cam, np.array([[]]))
                # print(row)
                
                pprint(retval)     
                pprint(rvec)
                pprint(tvec) 

        if input("continue").lower() == "n":
            break

if __name__=="__main__":
    main()