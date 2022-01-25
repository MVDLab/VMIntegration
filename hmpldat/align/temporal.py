""" Methods to perform temporal alignment

"""

import itertools
import logging
from pprint import pprint

import numpy as np
import pandas as pd
from tqdm import tqdm

import hmpldat.file.dflow

LOG = logging.getLogger(__name__)

MERGE_TOLERANCE = pd.Timedelta("5ms")
"""tolerance allowed between matched data (5 milliseconds)"""


def concat_and_interpolate(dflow_mc, dflow_rd, cortex, rawetg_and_objects):
    """
    different method to combine data
    
    create evenly spaced time index
    concatenate with time index

    interpolate gaps for floats
    merge_asof nearest for other data (strings, bools)
    
    return data evenly spaced @ 120Hz

    """
    # TODO: do everything after converting all indices from timedelta to float (do this everywhere?)
    # TODO: clean and simplify

    hz = 120

    # retain original time stamps
    dflow_rd = dflow_rd.assign(time_rd=dflow_rd.index)
    dflow_mc = dflow_mc.assign(time_mc=dflow_mc.index)
    cortex = cortex.assign(time_cortex=cortex.index)
    rawetg_and_objects = rawetg_and_objects.set_index('RecordingTime [ms]', drop=False)
    rawetg_and_objects.index = rawetg_and_objects.index.rename('time')


    # dflow files are recorded by the same system time stamps interspace each other 
    dflow = pd.concat([dflow_mc, dflow_rd], axis=1)

    # set all indices to start from 0
    dflow.index = dflow.index - dflow.index[0]
    
    # cortex always starts from zero (not required)
    if cortex.index[0] > pd.Timedelta(0):
        raise ValueError('cortex time expected to start from zero')
    
    # this should align with the 5V pulse
    alignment_pulse = dflow[dflow['channel16.anlg'] >= 4].index[0]
    cortex.index = cortex.index + alignment_pulse
    
    # rawetg needs start from the time stamp of the first cross appearance
    # (When using annotation to align)
    # TODO: handle other alignment methods
    first_cross_time = dflow[dflow['crossvisible.bool'] == 1].index[0]
    rawetg_and_objects.index = rawetg_and_objects.index - rawetg_and_objects.index[0] + first_cross_time

    ### create index to use to use with all the task data ###
    # length must contain all of the data
    task_duration = max([dflow.index[-1], cortex.index[-1], rawetg_and_objects.index[-1]]).total_seconds()
    # print(task_duration)

    # beaUtiful 120Hz
    evenly_spaced = pd.TimedeltaIndex([pd.Timedelta(x/hz, unit='s') for x in range(int(np.ceil(task_duration * 120)) + 1)])
    empty_series = pd.Series(data=range(len(evenly_spaced)), index=evenly_spaced, name='int_index')

    # retain original column order
    col_order = rawetg_and_objects.columns

    # interpolate differently by type
    # string/object, convert to categorical then interpolate with nearest
    str_and_objects = rawetg_and_objects.select_dtypes(include=['object', 'string'])
    str_and_objects.index = str_and_objects.index.total_seconds()

    # print(rawetg_and_objects.loc[rawetg_and_objects.index.duplicated(False)])
    # # TODO: this next statment should NOT be required, but VMIB_005 int has a duplicate row, drop it.
    rawetg_and_objects = rawetg_and_objects.loc[~rawetg_and_objects.index.duplicated(keep='first')]

    rawetg_and_objects = pd.concat([rawetg_and_objects, empty_series], axis=1)
    # rawetg_and_objects.to_csv('original_rawetg_objects.csv')
    rawetg_and_objects.index = rawetg_and_objects.index.total_seconds()

    # timedelta, convert to float
    # dt accessor only available to series, stack -> one big series, convert to seconds(float), unstack -> back to original
    timedeltas = rawetg_and_objects.select_dtypes(include='timedelta64').stack(dropna=False).dt.total_seconds().unstack()
    floats = rawetg_and_objects.select_dtypes(include=['float64'])

    # combine all float values 
    floats = pd.concat([timedeltas, floats], axis=1)

    # interpolate
    floats = floats.interpolate(method='index', order=3, axis='index', limit=6, limit_area='inside').loc[evenly_spaced.total_seconds(), :]

    rawetg_and_objects = pd.merge_asof(
        floats, 
        str_and_objects,
        left_index=True,
        right_index=True,
        tolerance=pd.Timedelta('10ms').total_seconds(),
        direction='nearest'
    )[col_order]

    # do this again for dflow+cortex data
    dflow_and_cortex = pd.concat([dflow, cortex], axis=1)
    col_order = dflow_and_cortex.columns

    to_match = dflow_and_cortex.select_dtypes(exclude=['float', 'timedelta64'])


    tds = dflow_and_cortex.select_dtypes(include='timedelta64').stack().dt.total_seconds().unstack()
    flts = dflow_and_cortex.select_dtypes(include='float64')

    df = pd.concat([tds, flts, empty_series], axis=1)

    df.index = df.index.total_seconds()
    to_match.index = to_match.index.total_seconds()

    df = df.interpolate(method='index', order=3, axis='index', limit_area='inside').loc[evenly_spaced.total_seconds(), :]

    df = pd.merge_asof(
        df,
        to_match.dropna(),
        left_index=True,
        right_index=True,
        tolerance=0.010,
        direction='nearest'
    )[col_order]

    df.loc[:, df.columns.str.contains('visible')] = df.loc[:, df.columns.str.contains('visible')].fillna(method='ffill')
    
    df = pd.concat([df, rawetg_and_objects], axis=1)
    
    return df


def split_by_type(df, exclude=None):
    """

    Args:
        dataframe: columns may be of different types
        exclude: str or list, string name of types 'object', 'float64', 'timedelta64[ns]'
    
    Returns:
        dictionary of dataframes, {dtype: dataframe_of_one_type}

    Notes:
        Must handle interpolation differently for different column types.

    """

    # if exclude is not None:
    #     raise NotImplementedError("exclude parameter for function split_by_type is not implemented")

    # get dtypes that exist in dataframe
    data_types = set(df.dtypes.astype(str).to_list())

    for this in list(exclude):
        data_types.remove(this) 

    dataframes = {}
    for dt in data_types:
        one_dtype = df.select_dtypes(include=dt)
        dataframes[dt] = one_dtype

    return dataframes


def interpolate(df, evenly_spaced):
    """

    """

    # combine with evenly spaced time index
    df = pd.concat([df, evenly_spaced], axis=1)

    # convert index to float
    df.index = df.index.total_seconds()

    # keep original column order
    col_order = df.columns

    # print(df.dtypes)
    # print(pd.api.types.is_bool(df['crossvisible.bool']))
    
    # handle these columns differently
    timedelta_col_names = [col for col in df.columns if pd.api.types.is_timedelta64_dtype(df[col])]
    string_col_names = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
    bool_col_names = [col for col in df.columns if pd.api.types.is_bool_dtype(df[col])]

    # convert timedelta columns to floats (floats are interpolatable)
    for col in timedelta_col_names:
        df[col] = df[col].dt.total_seconds()

    # split strings & bools from df
    string_columns = df[string_col_names]
    boolean_columns = df[bool_col_names]
    df = df.drop(columns=string_col_names + bool_col_names)

    # interpolate (linear for now)
    # TODO: investigate different methods from scipy interpolate
    df = df.interpolate(method='index', axis='index', limit_area='inside')
    
    # convert strings to categorical, then interpolate
    # convert back
    # string_columns = string_columns.interpolate(method='nearest', axis='index', limit_area='inside')

    boolean_columns = boolean_columns.astype(float).interpolate(method='nearest', axis='index', limit_area='inside').astype('boolean')

    df = pd.concat([df, boolean_columns, string_columns], axis=1).loc[evenly_spaced.index.total_seconds(), col_order]
    
    # df.to_csv('interpolated.csv')

    return df


def dflow_and_cortex_to_rawetg_and_objects(
    dflow_and_cortex_dict, rawetg_and_objects_df
):
    """Merge a dflow+cortex task to rawetg+objects

    Args:

    Returns:

    Notes:
        merge_asof is a left merge (this means the keys of the left dataframe are retained).
        This requires the left data frame to be the rawetg+cortex otherwise we will lose the data outside of task.
        1. upsample 60Hz rawetg+objects to 120Hz 
            interpolate rawetg data by index values 
            interpolate objects data by padding (since this data discretely matches video frames)
        2. then merge dflows+cortex to rawetg+objects

    """

    # Todo: if frequency is not 120 Hz then upsample
    #       pass if already upsampled
    #       else upsample -> like i do here by default :)
    #       probably should upsample before calculating starts (start may align slightly "better")
    rawetg_and_objects_df_upsampled = upsample_rawetg_and_objects(rawetg_and_objects_df)

    stacked_df = pd.concat(dflow_and_cortex_dict).droplevel(0)

    merged = pd.merge_asof(
        rawetg_and_objects_df_upsampled,
        stacked_df,
        left_index=True,
        right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta("8ms"),
    )

    return merged


def dflow_to_cortex(dflow_mc, dflow_rd, cortex) -> pd.DataFrame:
    """Using current methods, aligns files.  
    
    Args:
        dflow_mc: dflow motion capture file as a pandas DataFrame
        dflow_rd: dflow virtual reality file as a pandas DataFrame
        cortex: cortex motion capture file as a pandas DataFrame

    Returns:
        A DataFrame with all data sampled to 120Hz

    Notes:
        * IF (both dflow files contain 5V pulse)
            1. get start time of 5V pulse for rd and mc
            2. add dflow_mc 5V start time to cortex index
            3. merge cortex to dflow_mc index (dflow_mc index is longer) 
            4. merge dflow_rd to combined cortex and dflow_mc dataframe (300 -> 120Hz)
            
        * ELSE (only dflow_mc contains 5V pulse)
            1. merge dflow_rd to dflow_mc (300Hz -> 120Hz)
            2. add merged 5V start to cortex index
            3. merge cortex to combined dflow dataframe

    """

    # append these indexes as columns to perseve time information
    dflow_rd = dflow_rd.assign(time_rd=dflow_rd.index)
    dflow_mc = dflow_mc.assign(time_mc=dflow_mc.index)
    cortex = cortex.assign(time_cortex=cortex.index)

    # check for pulse in both Dflow files
    if "RNP" in dflow_rd.columns:

        # the 5V pulse doesn't stay on for the entire task
        # use the first frame with 5V as start and align by time (cortex turns off when task is over?)

        # get start of 5V in dflow mc file
        dflow_mc_start = dflow_mc[dflow_mc["Channel16.Anlg"].gt(4)].index[0]

        # get start of 5V in dflow rd file
        dflow_rd_start = dflow_rd[dflow_rd["RNP"].gt(4)].index[0]

        # add dflow start pulse to cortex index
        # cortex always starts at zero
        # our new index is the running clock from dflow
        cortex.index = cortex.index + dflow_mc_start

        # align dflow_mc (forceplate data) to cortex motion capture data
        merged = pd.merge_asof(
            dflow_mc,
            cortex,
            right_on="Time",
            left_index=True,
            direction="nearest",
            tolerance=MERGE_TOLERANCE,
        )

        # downsample and align dflow rd to merged(cortex + mc)
        merged = pd.merge_asof(
            merged,
            dflow_rd,
            right_on="Time",
            left_index=True,
            direction="nearest",
            tolerance=MERGE_TOLERANCE,
        )

    # else match together dflow file by timestamp then align to cortex
    else:

        # merge on nearest time (downsample vr environment data from dflow_rd)
        merged = pd.merge_asof(
            dflow_mc,
            dflow_rd,
            right_on="Time",
            left_index=True,
            direction="nearest",
            tolerance=MERGE_TOLERANCE,
        )

        # get start of 5V pulse
        merged_start = merged[merged["Channel16.Anlg"].gt(4)].index[0]

        # add dflow start pulse to cortex index
        # cortex always starts at zero
        # our new index is the running clock from dflow
        cortex.index = cortex.index + merged_start

        # align combined dflow files to cortex
        merged = pd.merge_asof(
            merged,
            cortex,
            right_on="Time",
            left_index=True,
            direction="nearest",
            tolerance=MERGE_TOLERANCE,
        )

    return merged


def combos(list1, list2):
    """Return different orderings of l1 + l2 where sub list order is maintained

    https://stackoverflow.com/questions/26305216/permutations-of-the-elements-of-two-lists-with-order-restrictions?rq=1
    """
    perms = []
    if len(list1) + len(list2) == 1:
        return [list1 or list2]
    if list1:
        for item in combos(list1[1:], list2):
            perms.append([list1[0]] + item)
    if list2:
        for item in combos(list1, list2[1:]):
            perms.append([list2[0]] + item)
    return perms


def match_task_lengths(detected_tasks, dflow_tasks):
    """align tasks according to duration and order

    Todo: Test this with dflow that is non continuous

    Args:
        detected_tasks: a dataframe with task durations
        dflow_tasks: a dataframe with task durations

    Returns:
        single pd.DataFrame indexed by task name

    """

    diff = len(dflow_tasks.index) - len(detected_tasks.index)
    assert diff >= 0, "more tasks detected than dflow recorded"

    # reset this index, thing get weird if you don't
    dflow_tasks = dflow_tasks.reset_index()

    # rename columns
    dflow_tasks.columns = ["_".join([col, "dflow"]) for col in dflow_tasks.columns]
    detected_tasks.columns = [
        "_".join([col, "detected"]) for col in detected_tasks.columns
    ]

    # create lists of detected durations and "undetected" pd.NaT durations
    detected_durations = list(detected_tasks["duration_detected"])
    missing_nans = [pd.NaT for _ in range(diff)]

    # Todo: try changing dflow order with a cycle, if error with autoalignment?
    # keep detected tasks in order, but add in pd.NaT for missing tasks
    # calc ordering with sum squared error
    min_err = np.inf
    for i in combos(detected_durations, missing_nans):
        err = (
            (dflow_tasks["duration_dflow"] - pd.Series(i)).dt.total_seconds() ** 2
        ).sum()
        if err < min_err:
            min_err = err
            dflow_tasks["duration_detected"] = i

    print(min_err)

    # if we still have lots of error something wrong...
    if min_err > 1000:

        LOG.error(
            "Unable to auto align, we detected a complete task that was not found in dflow"
        )
        return None
        # TODO: handle this...
        # dflow_duration_list = list(dflow_tasks['duration_dflow'])

        # for i in combos(dflow_duration_list, [pd.NaT]):
        #     err = ((['duration_dflow'] - pd.Series(i)).dt.total_seconds()**2).sum()
        #     if err < min_err:
        #         min_err = err
        #         dflow_tasks['duration_detected'] = i

    # merge detected to dflow on optimized task duration
    matched = dflow_tasks.merge(detected_tasks, how="left", on="duration_detected")
    matched = matched.set_index("index_dflow")
    matched.index.name = "task"

    print(matched)
    return matched


def objects_to_rawetg(objects_df, rawetg_df, how='time') -> pd.DataFrame:
    """Align reformated objects to rawetg dataframe

    Args:
        objects: objects dataframe with column "corrected_frame_time" or "frame number"
        rawetg: rawetg dataframe with column "RecordingTime [ms]" or "frame number" (calculated, see: rawetg.frame_number(df))
        how: time(default), frame_num

    Returns:
        one dataframe containing all instances of rawetg and detected objects for an entire video (all tasks)

    Notes:
        * objects_df merged to rawetg_df (30Hz -> 60Hz) 
        * data should be interpolated after this join 
        * how='time'
            * match detection video frame time stamp with nearest rawetg recording time 
            * this requires matching the nearest rawetg 'RecordingTime [ms]' to objects corrected frame time 
        * how='frame_num'
            * join rawetg(calculated frame_number) to objects frame number

    """

    if how == 'time':

        # adjust video frame time to match rawetg RecordingTime 
        # objects_df['corrected_frame_time'] = objects_df['corrected_frame_time'] + rawetg_df['RecordingTime [ms]'].iloc[0]

        # OR, adjust RecordTime column to start at 0.
        rawetg_df['RecordingTime [ms]'] = rawetg_df['RecordingTime [ms]'] - rawetg_df['RecordingTime [ms]'].iloc[0]

        # only match one detection(30Hz) row with the nearest rawetg(60Hz) time
        # results in 30Hz
        aligned = pd.merge_asof(
            objects_df,
            rawetg_df,
            direction="nearest",
            left_on="corrected_frame_time",
            right_on="RecordingTime [ms]",
            tolerance=pd.Timedelta(17, unit='ms')
        ) 

        # then merge the result with the entire rawetg(60Hz)
        aligned = rawetg_df.merge(aligned, how='left')
    
    elif 'frame_num' in how: 

        if 'frame_number' not in rawetg_df.columns:
            raise(ValueError, "frame number must be calculated and added as column of rawetg data before alignment")

        # left join returns 60Hz, but leaves use with duplicated entries for 
        # object detection locations
        aligned = pd.merge(
            objects_df,
            rawetg_df,
            how='left',
            on='frame_number',
        ) 

        # so remove the duplicated entries
        aligned = aligned.drop_duplicates(subset='frame_number', keep='first')

        # merge with rawetg -> now 60Hz with gaps we can interpolate
        aligned = rawetg_df.merge(aligned, how='left')
   
    else:
        raise(ValueError, "argument \"how\" must be \"time\" or \"frame_num(ber)\"")

    ### TESTING -- save to file ###
    # a2 = aligned.copy()

    # a2['RecordingTime [ms]'] = a2['RecordingTime [ms]'] + pd.Timestamp('1900-01-01')
    # a2['corrected_frame_time'] = a2['corrected_frame_time'] + pd.Timestamp('1900-01-01')

    # writer = pd.ExcelWriter(f'amanda_please.xlsx',datetime_format='hh:mm:ss.000')
    # a2.to_excel(writer)

    # writer.close()
    ### END TESTING ###
 
    return aligned


def dflow_task_order(file_listing):
    """
    define task order with runnning dflow time

    given a file listing for s SINGLE participant 
    return start and end times for each task recorded in dflow_mc file (ORDERED)

    """

    # create a dictionary 
    dflow_times = {}

    for k, task in file_listing.groupby('task_name'):

        # task has no dflow (then skip)
        if task[task["type"] == "dflow_mc"].empty:
            continue

        # open file
        mc = hmpldat.file.dflow.mc_open(task[task["type"] == "dflow_mc"]['path'][0])

        # record start and end time
        dflow_times[k] = {
            'start': mc.index.to_series().iloc[0],
            'end': mc.index.to_series().iloc[-1],
        }

    dflow_times = pd.DataFrame(dflow_times).T.stack().rename('time').sort_values()

    # print(dflow_times)
    return dflow_times


def rawetg_and_objects_to_dflow_and_cortex(rawetg_and_objects_df, dflow_and_cortex_df):
    """ Merge a section of rawetg+objects to a dflow+cortex task

    Returns:
        a dataframe representing a single task for a participant.

    """

    print(dflow_and_cortex_df)
    print(rawetg_and_objects_df)

    rawetg_and_objects_df = rawetg_and_objects_df.set_index(
        "rawetg_recording_time_from_zero", drop=False
    )

    # when data is not previously upsampled
    # upsample to 120 Hz dflow time -> interpolate, then merge back together
    if True:
        columns_to_match = [
            "RecordingTime",
            "calc_video_time",
            "Video Time",
            "rawetg_recording_time_from_zero",
            "Annotation Name",
            "Category Binocular",
            "Tracking Ratio",
            "in_",
            "Ready?",
            "three",
            "two",
            "one",
            "Done",
            "target",
            "cross",
            "grid",
            "user",
            "safezone",
        ]
        columns_to_interpolate = ["Regard", "Pupil"]

        discrete_merged_to_index = pd.merge_asof(
            dflow_and_cortex_df[[]],
            rawetg_and_objects_df.filter(regex="|".join(columns_to_match)),
            left_index=True,
            right_index=True,
            direction="nearest",
            tolerance=pd.Timedelta("17ms"),
        )

        interpolate_merged_to_index = pd.merge_asof(
            dflow_and_cortex_df[[]],
            rawetg_and_objects_df.filter(regex="|".join(columns_to_interpolate)),
            left_index=True,
            right_index=True,
            direction="nearest",
            tolerance=pd.Timedelta("4100us"),  # 4.10ms
        )

        interpolated_merged_to_index = interpolate_merged_to_index.interpolate(
            "index", limit=5, limit_area="inside"
        ).round(1)

        # now discrete + interpolated data
        rawetg_and_objects = pd.merge(
            discrete_merged_to_index,
            interpolated_merged_to_index,
            left_index=True,
            right_index=True,
        )

    # merge upsampled rawetg+objects
    merged = pd.merge(
        dflow_and_cortex_df, rawetg_and_objects, left_index=True, right_index=True
    )

    return merged

# TODO: interpolate object location between video frames
# TODO: 
def upsample_rawetg_and_objects(df):
    """ Upsample these datas from 60Hz to 120Hz
    
    Args:
        df: merged rawetg_and_cortex dataframe

    Returns:
        a dataframe with perfectly spaced 120Hz 

    Notes:
        This requires reindexing because rawetg RecordingTime is not perfectly spaced at 60Hz to begin with

        columns interpolated:
        
        columns matched to nearest:

    """

    # df_indexed = df.set_index("rawetg_recording_time_from_zero", drop=False)
    df_indexed = df.set_index(df['RecordingTime [ms]'] - df['RecordingTime [ms]'].iloc[0])
    # print(df_indexed)

    columns_to_match = [
        "RecordingTime",
        "calc_video_time",
        "Video Time",
        "rawetg_recording_time_from_zero",
        "Annotation Name",
        "Category Binocular",
        "Tracking Ratio",
        "in_",
        "Ready?",
        "three",
        "two",
        "one",
        "Done",
        "target",
        "cross",
        "grid",
        "user",
        "safezone",
        "frame_number"
    ]
    columns_to_interpolate = ["Regard", "Pupil"]

    # upsample to 120Hz
    df_upsampled = df_indexed.resample("8333333ns").asfreq()

    # now merge_asof (to closest timestamp) from original
    df_upsampled = pd.merge_asof(
        df_upsampled[[]],  # no coloumns
        df_indexed.filter(regex="|".join(columns_to_match)),
        left_index=True,
        right_index=True,
        tolerance=pd.Timedelta("16ms"),  # relaxed
        direction="nearest",
    )

    # now merge_asof (to closest timestamp) from original
    df_upsampled_to_interpolate = pd.merge_asof(
        df_upsampled[[]],  # no columns
        df_indexed.filter(regex="|".join(columns_to_interpolate)),
        left_index=True,
        right_index=True,
        tolerance=pd.Timedelta(
            "4ms"
        ),  # more strict -> this leaves gaps that we will interpolate
        direction="nearest",
    )

    # interpolate gaps (round to original precision)
    df_upsampled_interpolated = df_upsampled_to_interpolate.interpolate(
        "index", limit=5, limit_area="inside"
    ).round(1)

    # inner join (these dataframes have equal keys)
    upsampled = pd.merge(
        df_upsampled, df_upsampled_interpolated, left_index=True, right_index=True,
    )

    return upsampled
