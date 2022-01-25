"""Example usage for several functions

If you have any questions, shoot me an email ian.zurutuza@gmail.com
Uncomment a section of code and run `python examples.py` 

Feel free to add and/or change the code to try out what ever you what.

"""

import logging
import time
from pathlib import Path
from pprint import pprint

import pandas as pd

# Import our our methods
from hmpldat.utils import cortex, rawetg, dflow, search, video, align

LOG = logging.getLogger(__name__)

FILES = Path("/mnt/hdd/VMI_data/vmi/datasets/VMIB/Data")
"""change this to match your file path to the shared drive"""


def main():

    ############################### cortex open ##################################
    print('example search for cortex then cortex.open()')
 
    x = search.files(FILES, ['vmib', '007', 'cortex', 'trc', 'bm'], [])
    print(x)

    df = cortex.open(x[0])

    # print columns
    i = 0
    for c in df.columns:
        print(c + ',')

    ############################## dflow rd open ########################################
    # print('example search for dflow_rd then dflow.open()')
    # x = search.files(FILES, ['vmib', '007', 'dflow', 'rd', 'bm'], [])
    # print(x)

    # df = dflow.rd_open(x[0])

    # # print columns
    # i = 0
    # print(','.join(list(df.columns)))

    ####################### Bundle associated ###############################
    # print('search.bundle_associated()')

    # a_dict_of_files = search.bundle_associated(FILES, 'vmib_023', 'vt')

    # print()
    # for key, value in a_dict_of_files.items():
    #     print(f'{key}: {str(value)}')
    #     for each_path in value:

    #         print(search.probe_elapsed_time(each_path, key))

    # # rawetg probe
    # print('example search for rawETG and probe elapsed time')
    # x = search.files(FILES, ['rawetg'], [])
    # print(x)

    # for i in x:
    #     print(rawetg.probe_elapsed_time(i))

    ######################## # search video and probe length ############
    # print('search and probe for videos')

    # videos = search.files(FILES, ['vmib', '.avi', '30hz'], [])

    # for v in videos:
    #     print(v)

    #     print(pd.Timedelta(video.probe_elapsed_time(v), unit='ms'))

    # labels open and reformat
    # print("open and reformat labels")

    # objt_files = search.files(Path('/mnt/hdd/VMI_data/14oct2019/output'), ['.txt', 'vmib'], [])

    # for f in objt_files:
    #     print(str(f))
    #     df = objects.open(f)

    #     objects.reformat(df[df['score'] > 0.95])

    ############## Bundle and align #################################
    # TODO: example match file that belong together (check length if more than one file is returned)
    # print('search.bundle_associated()')

    # a_dict_of_files = search.bundle_associated(FILES, 'vmib_026', 'pp')

    # for key, value in a_dict_of_files.items():
    #     print(f'{key}: {str(value)}')

    # dflow_mc_df = dflow.mc_open(a_dict_of_files['dflow_mc'][0])
    # dflow_rd_df = dflow.rd_open(a_dict_of_files['dflow_rd'][0])
    # cortex_df = cortex.open(a_dict_of_files['cortex'][0])

    # print(align.dflow_to_cortex(dflow_mc_df, dflow_rd_df, cortex_df))

    ######################### estimate gaze error ##############################
    # files = search.files(Path('/mnt/hdd/VMI_data/vmi/datasets/VData_Integration/merged_data'), [])

    # for f in files:
    #     if len(f.name.split('_')) == 5:
    #         df = pd.read_csv(f, low_memory=False)

    #         # spatial_align.est_gaze_error(df)
    #         x = spatial_align.init_head_vector_est(df)

    #         print(x)

    ############################# Check merge #######################################
    # I created a file with some selections from vmib 007 rawetg recording.

    # rawetg_samples = pd.read_excel(
    #     "rawetg_examples_to_merge.xls", sheet_name=None, na_values=["-"]
    # )
    # dflow_sample = pd.read_excel("dflow_small_sample.xls")

    # dflow_sample["TimeStamp"] = pd.to_timedelta(dflow_sample["TimeStamp"], unit="s")
    # # dflow_sample.set_index('TimeStamp', inplace=True)

    # for k, df in rawetg_samples.items():

    #     adj_dflow_sample = dflow_sample

    #     adj_dflow_sample["dflow_time_adj"] = (
    #         adj_dflow_sample["TimeStamp"] - adj_dflow_sample["TimeStamp"].iat[0]
    #     )
    #     adj_dflow_sample = adj_dflow_sample.set_index("dflow_time_adj")

    #     # convert rawetg times into timedelta
    #     df["RecordingTime [ms]"] = pd.to_timedelta(df["RecordingTime [ms]"], unit="ms")
    #     df["Video Time [h:m:s:ms]"] = (
    #         df["Video Time [h:m:s:ms]"].str.rsplit(":", 1).str.join(".")
    #     )
    #     df["Video Time [h:m:s:ms]"] = pd.to_timedelta(
    #         df["Video Time [h:m:s:ms]"]
    #     ).fillna(method="bfill", limit=2)

    #     df["step"] = df["RecordingTime [ms]"].diff()

    #     # print(k)
    #     print(df)
    #     # print(adj_dflow_sample)

    #     # this occurs during rawetg file open
    #     df["rawetg_recording_time_from_zero"] = (
    #         df["RecordingTime [ms]"] - df["RecordingTime [ms]"].iat[0]
    #     )

    #     # columns to match
    #     # perform resample
    #     print(align.upsample_rawetg_and_objects(df))
    #     # print(align.rawetg_and_objects_to_dflow_and_cortex(df, adj_dflow_sample))

    #     input()

        #

        # print()


if __name__ == "__main__":
    main()
