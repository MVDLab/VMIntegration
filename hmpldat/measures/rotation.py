"""Rotation of a point around a line

general task measures
- head rotation about the y-axis (xz-plane)
- 
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import hmpldat.file.participant

X_AXIS = [1, 0, 0]
Y_AXIS = [0, 1, 0]
Z_AXIS = [0, 0, -1]
ORIGIN = [0, 0, 0]

# define by 3 points to keep direction consistent
XZ_PLANE = np.array([ORIGIN, Z_AXIS, X_AXIS])
# YZ_PLANE = 
# XY_PLANE = 


def calc_head_rotations(task):
  


def run_metrics_and_save():

    b = ('VMIB', '005')
    a = ('VMIB', '047')

    # for each merged data file
    # for f in search.files(DATA_PATH, []):  # a_task or all_tasks_from_one_session
    for study, participant in [a, b]:

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

            # TODO: keep n sample per task -> one camera model per task
            # samples = 

            # skip tasks without merged data
            if not hasattr(task, 'merged'):
                continue

            calc_head_rotations(task)




# def project_point_onto_plane(points, plane)
#     """

#     """



# def around_axis(points, axis):
#     """find rotation of point around an axis



#     This axis can be any vector

#     Args:
#         task: task object
#         points: points to calculate rotation
#         axis: any vector

#     """


# # project point on to plane
# # 




# def test_around_axis():
#     """test find rotation around axis"""





def visualize():
    """plot, for sanity"""


if __name__ == "__main__":
    run_metrics_and_save()