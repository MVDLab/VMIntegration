"""compare_centers.py

Match labeled and detected centers

calculate descriptive stats of differences and euclidean distances
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import hmpldat.file.detected
import hmpldat.utils.camera

FLAGS = None


def main():

    detected = hmpldat.file.detected.open(FLAGS.detected)
    # print(detected)

    # assume distance to the screen of 2490 millimeters how big is a pixel
    # returned as tuple w, h
    pixel_size = hmpldat.utils.camera.calc_pixel_size(2490, 60, 46, 960, 720)
    # print(pixel_size)

    # only care about the crosses for now only the highest scoring object from each frame
    detected = detected.query(f"object == '{FLAGS.labeled.name.split('_')[0]}'").droplevel(-1).groupby(by="frame_number").nth([0])
    detected_centers = detected[['ctr_bb_row', 'ctr_bb_col']]

    labeled = pd.read_csv(FLAGS.labeled, index_col=0)

    detected_centers.columns = pd.MultiIndex.from_product([['detected'], ['v','u']])
    labeled.columns = pd.MultiIndex.from_product([['labeled'], labeled.columns])

    # merge frames
    # df = pd.merge(detected_centers, labeled, left_index=True, right_index=True)
    df = labeled.join(detected_centers)
    df = df.sort_index(axis=1)

    not_detected = df[df.isna().any(axis=1)]
    print(f"failed to detect {len(not_detected)} frames")

    # remove instances near border?
    # if FLAGS.filter:
    #     width, height = 960, 720
    #     hb = width / 6
    #     vb = height / 6

    #     df = df[
    #         (df[('labeled', 'u')] > hb)
    #         & (df[('labeled', 'u')] < (width - hb))
    #         & (df[('labeled', 'v')] > vb)
    #         & (df[('labeled', 'v')] < (height - vb))
    #     ]

    # calculate pixel locations in physical units
    physical_units = df.copy()
    physical_units.loc[:, (slice(None), 'u')] = df.loc[:, (slice(None), 'u')] * pixel_size[0]
    physical_units.loc[:, (slice(None), 'v')] = df.loc[:, (slice(None), 'v')] * pixel_size[1]
    
    # simple difference
    df[("diff_pixel", 'u')] = df[('detected', 'u')] - df[('labeled', 'u')]
    df[("diff_pixel", 'v')] = df[('detected', 'v')] - df[('labeled', 'v')]

    df[("diff_physical", 'u')] = df[("diff_pixel", 'u')] * pixel_size[0]
    df[("diff_physical", 'v')] = df[("diff_pixel", 'v')] * pixel_size[1]

    # real distance
    df[("euclidean", "pixel")] = np.diag(cdist(df['detected'], df['labeled'], 'euclidean'))
    df[("euclidean", "dist")] = np.diag(cdist(physical_units['detected'], physical_units['labeled'], 'euclidean'))

    # save results to file
    with pd.ExcelWriter(f"{FLAGS.labeled.name.split('_')[0]}_comparison.xls") as writer:
        df.filter(regex="euclidean|diff*").describe().to_excel(writer, sheet_name="summary", float_format="%.2f")
        df.to_excel(writer, sheet_name="data", float_format="%.2f")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--detected",
        type=Path
    )

    parser.add_argument(
        "--labeled",
        type=Path
    )

    # remove instances on edge of frames
    parser.add_argument(
        '--filter', dest='filter', action='store_true'
        )
    parser.add_argument(
        '--no-filter', dest='filter', action='store_false'
        )
    parser.set_defaults(filter=False)

    FLAGS, _ = parser.parse_known_args()
    main()
