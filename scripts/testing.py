

from pathlib import Path
from dataclasses import dataclass


import pandas as pd


import hmpldat.file.search as search
import hmpldat.align.simple_spatial
import hmpldat.align.spatial



@dataclass
class Point:
    x: float
    y: float
    z: float


def df_point_slice(df, point):
    """ slice point from dataframe and rename columns 

    original column names: ["point.X", "point.Y", "point.Z"]

    """

    # create regex search string
    point_o_interest = point + "\.[X,Y,Z]+"

    # slice correspond point from dataframe 
    slice_df = df.filter(regex=point_o_interest)

    # rename columns to simple [X, Y, Z]
    slice_df = slice_df.rename(columns= lambda s: s.split(".", maxsplit=1)[1])

    return slice_df


def main():

    rotation = None
    translation = None


    for f in search.files(Path("/home/irz0002/Projects/hmpldat/sample_datas/merged"), []):

        df = pd.read_csv(f)

        df = hmpldat.align.spatial.initialize_camera_position(df, rotation, translation)
        
        # drop instances with no mocap data (cortex typically starts recording shortly after dflow)
        df = df.dropna(subset=["Frame#"])

        # slice points from dataframe
        camera = df_point_slice(df, "camera")
        o = df_point_slice(df, "o")
        o_left = df_point_slice(df, "o_left")

        # print(df.filter(regex=r"camera\.[X,Y,Z]"))

        frame_center = df.apply(lambda row: hmpldat.align.simple_spatial.find_vector_screen_intersection(row.filter(regex="camera\.[X,Y,Z]"), row.filter(regex="origin\.[X,Y,Z]")))


if __name__ == "__main__":
    main()