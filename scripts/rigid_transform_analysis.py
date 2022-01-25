"""check rigid transform method for glasses markers and plot

"""

from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import hmpldat.utils.glasses
import hmpldat.file.search as search
import hmpldat.utils.math


DATA_PATH = Path("./sample_data/merged")
ETG_MODELS_FILE = Path("./sample_data/test_etg_models.csv")



def main():

    for f in search.files(DATA_PATH, []):  # a_task or all_tasks_from_one_session
        print(f"\n\tfile= {f.name}\n")

        # read data
        df = pd.read_csv(f, low_memory=False)
        mocap = hmpldat.file.cortex.multiindex(df)

        etg_markers = mocap[["fhead", "lhead", "rhead"]]

        etg_markers = etg_markers.dropna()

        # sample instances
        sampled_instances = etg_markers.sample(200)
        print(sampled_instances)

        # load corresponding glasses model
        g = hmpldat.utils.glasses.etg_model()
        ret = g.load(ETG_MODELS_FILE, f.name)
        if not ret:
            input("ERROR: corresponding glasses model not found in file {}")

        # calculate and record optimal rotation for each sampled instance
        rotation_dict = {}
        for i, instance in sampled_instances.iterrows():
            r, t = hmpldat.utils.math.rigid_transform_3D(
                g.in_glasses_space, instance.unstack(0)
            )
            rotation_dict[i] = {"r": r, "t": t}

        print("glasses model")
        print(g.in_glasses_space)
        print()

        # do the glasses markers, in glasses space, end up at the mocap location
        for i, instance in sampled_instances.iterrows():
            print("instance")
            instance = instance.unstack(0)
            print(instance)
            print()

            print("transform glasses model by optimal rotation+translation")
            new = rotation_dict[i]["r"] @ g.in_glasses_space + np.tile(rotation_dict[i]["t"], (3,1)).T
            new.index = ["x","y","z"]
            print(new)
            print()
            
            rmse = (((new - instance).values ** 2).sum() ** 0.5)/3
            print(f"rmse={rmse}")
            
            input("\nhit ENTER for next instance\n")

if __name__=="__main__":
    main()