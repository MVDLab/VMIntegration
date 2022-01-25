"""

Ian Zurutuza
8 July 2020
"""

from pprint import pprint

import numpy as np
import pandas as pd

import hmpldat.file.dflow
import hmpldat.file.detected
import hmpldat.align.temporal


def last_object_appearance(
    df, min_duration=pd.Timedelta("266.67ms"), between=pd.Timedelta("10s")
):
    """
    Use to find task start

    return time of last object appearance
    """

    # drop rows without object detection
    df = df.dropna(subset=["visible"])

    # select rows with score above .98
    df = df[df["score"] > 0.98]

    # assign a count to each appearance
    df["count"] = (df.index.to_series().diff() > between).cumsum()

    ks = []

    print()
    for k, g in df.groupby("count"):

        true_detection = False
        # # or by size??
        # if g.shape[0] > 8:
        #     print("true detection by shape")
        #     true_detection = True

        # min duration
        if (g.index.to_series().iloc[-1] - g.index.to_series().iloc[0]) > min_duration:
            # print("true detection by min duration")
            true_detection = True

        # else:
        # print('false detection', k)

        if true_detection:
            ks.append(g.iloc[-1].name)
            # print(g.iloc[-5:-1])

    # pprint(ks)
    # print(df)

    return ks


def first_object_appearance(
    df, min_duration=pd.Timedelta("100ms"), between=pd.Timedelta("10s")
):
    """
    Use to find task ends

    return time of first object appearance

    """

    # select rows with score above .95
    df = df[df["score"] > 0.95]

    # print(an_object.index.to_frame())
    # assign a count to each appearance
    count = (df["time"].diff() > between).cumsum()

    # ks = []
    first_appearance = pd.Series(index=df.columns, dtype=object)

    # print()
    for k, g in df.groupby(count):

        true_detection = False
        # # or by size??
        # if g.shape[0] > 8:
        #     print("true detection by shape")
        #     true_detection = True

        # min duration
        if (g.iloc[-1]["time"] - g.iloc[0]["time"]) > min_duration:
            # print("true detection by min duration")
            true_detection = True

        # else:
        # print('false detection', k)

        if true_detection:
            first_appearance = g.iloc[0]

    return first_appearance


def find_switch(df, a, b):

    # filter failed detections, these do not persist for at least 133ms (4 video frames)
    # for this method count() must be greater than half of the rolling window
    # TODO: investigate behavior of method with even sized window
    a_df = df[df[(a, "score")].rolling(7, center=True).count() > 3]
    b_df = df[df[(b, "score")].rolling(7, center=True).count() > 3]
    time = df["frame_time", "corrected"]

    # did we capture the frame on which the change happens
    intersect = a_df.index.intersection(b_df.index)

    print(intersect)

    if (
        len(intersect) == 1
    ):  # single intersection (we captured object a disappearing and object b appearing in the same frame)
        intersect = df.loc[intersect]
        switch_frame = intersect[[a, b]].iloc[0].name
        print(f"switch at: {switch_frame}\n{intersect[[a, b]].iloc[0].unstack()}")
        switch_time = time.loc[switch_frame]

    elif len(intersect) > 1:  # this should not happen
        # print('Error: too much intersect')
        # print(f'switch at: \n{intersect[[a, b]].iloc[0].unstack()}')
        raise (ValueError)

    else:  # no intersect, (the change happens in between video frames)
        # use the last appearance of first object
        last_a = a_df["frame_time", "corrected"].max()
        # print(last_a)

        # or use first appearance of the next object
        first_b = b_df["frame_time", "corrected"].min()
        # print(first_b)

        if first_b - last_a > pd.Timedelta("35ms"):
            raise (ValueError, "change doesn't happen immediately??")

        # split difference
        switch_time = (first_b + last_a) / 2

        # print(f'switch at: {switch_time}')

    return switch_time


def find_done_and_last_object(df):

    # no grid or cross
    task_objects = ['target', 'user', 'safezone']

    last_object_time = pd.NaT
    first_done_time = pd.NaT

    for n, row in df.iterrows():
        if row['Ready?', 'score'] > 0.95:
            break
        if any(row[(a, 'score')] > 0.95 for a in task_objects):
            last_object_time = row['frame_time', 'corrected']
        if row['Done', 'score'] > 0.95:
            first_done_time = row['frame_time', 'corrected']
            break

    return first_done_time, last_object_time


# def find_ready_start(df):
#     """
#     find ready ,three, two, one identifiers for task start

#     """

#     # TODO: treat as a single object
#     # combine Ready?, three, two, one, detections into a single "start object"
#     start_object = df[[
#         ('Ready?', 'visible'),
#         ('three', 'visible'),
#         ('two', 'visible'),
#         ('one', 'visible'),
#     ]].any(axis='columns')
#     # returned as boolean column
#     # print(start_object.dropna())

#     # use a rolling window to decide object permanence (sum due to boolean conversion with .any() in previous line)
#     ww = 75 # (window_width // 2) * 33.33ms = min time object must be detected
#     starts = df.loc[(start_object.rolling(ww, center=True).sum() > ww//2)].copy()

#     # 0-indexed start count, expect 8 with no recording errors:
#     # if detected starts are at least 5 seconds apart label as a separate start
#     # df['detected_task_number'] = (starts.index.to_series().diff() > pd.Timedelta('5s')).cumsum()
#     # print(df['detected_task_number'].dropna())


#     # # find_task_ends(df)?
#     # for k, g in df.groupby('detected_task_number'):

#     #     print(k)
#     #     # print(g.sort_index().is_lexsorted())

#     #     print(g)
#     #     input()

#     #     # find done occurrence within ~6 minutes of the task start (300 + 10 seconds)
#     #     task_start = g.index.max() # last value of detected start
#     #     max_task_end = task_start + pd.Timedelta('360s')

#     #     print('first loc')
#     #     between = df.loc[task_start:max_task_end]
#     #     print(between)

#     #     ww = 15
#     #     # again rolling window for permanence (count non NaN values this time)
#     #     dones = between[between[('Done', 'visible')].rolling(ww, center=True).count() > ww//2]
#     #     print(dones.index)

#     #     # no done detected? skip;
#     #     # TODO: evaluate if this is what I should do...
#     #     # this is likely a trouble-shooting task
#     #     if dones.empty:
#     #         continue

#     #     print('second loc')
#         # df.loc[dones.index, ['detected_task_number']] = k

#         # two dones detected during this duration ( expected for tasks that are shorter: fix, hp, pp, safezone )
#         # select first done
#         # dones['count'] = dones.index.to_series.diff() > pd.Timedelta('15s').cumsum()

#         # # check for multiple dones in this group
#         # if (done.index.max() - done.index.min()) > pd.Timedelta('15s'):

#         #     done.index = done.index.astype(str)
#         #     done.to_excel('done_check.xls')
#         #     print(f'Extremely long done [ {done.index.min()} - {done.index.max()} ]')
#         # print(done)

#     # individually find start identifiers
#     ready_start = first_object_appearance(df['Ready?'])
#     # three_three_start = first_object_appearance(df['three'])
#     # two_start = first_object_appearance(df['two'])
#     # # one_start = first_object_appearance(df['one'])

#     # starts
#     start_locs = list(zip(map(lambda x: x - pd.Timedelta('1s'), ready_start), map(lambda x: x + pd.Timedelta('6s'), ready_start)))

#     # pprint(start_locs)
#     # for a, b in start_locs:
#     #     print(a, " : ", b)

#     starts = [
#         df.loc[a:b] for a, b in start_locs
#     ]

#     # calculate time between object disappearances
#     #
#     x = 1
#     for start in starts:

#         start['count'] = x
#         # print(start)

#         # time between appearance and disappearance
#         # start[('Ready?', 'visible')] - start[('one', 'visible')].ilo

#         # ready? -> three
#         # print(start[start[('Ready?', 'score')] >= 0.2].index)

#         # find switch between objects
#         ready_to_three = find_switch(start, 'Ready?', 'three')
#         three_to_two = find_switch(start, 'three', 'two')
#         two_to_one = find_switch(start, 'two', 'one')

#         x += 1

#         # else
#         diff = start[start[('three', 'score')] >= 0.2].index.min() - start[start[('Ready?', 'score')] >= 0.2].index.min()
#         print(diff)

#     starts = pd.concat(starts)
#     starts.index = starts.index.astype(str)

#     # starts.to_excel("test.xls")

#     return starts


# def find_end(df, starts):
#     """
#     with task starts defined, find task end

#     Args:
#         df: dataframe of detected objects indexed by frame time
#         starts: list of frame times for detected starts

#     returns:
#         frame times of the last object appearance

#     """

#     # smooth detected "Done"s


#     # iterate over each row from start to first done appearance


def main():

    # TODO: update read detections to do this and only this
    detections_df = pd.read_table(
        "./output_a/vmib_038-1-unpack.txt", sep=",", header=0,
    )

    # convert time column to timedelta
    detections_df["frame_time"] = pd.to_timedelta(
        detections_df["frame_time"], unit="ms"
    )

    # fix frame time at beginning of data
    detections_df = hmpldat.file.detected.fix_frame_time(detections_df)

    # TODO: reformat=True (by default) or just return raw
    detections_df = hmpldat.file.detected.reformat(detections_df)
    # print(detections_df[detections_df['frame_time', 'corrected'].diff() > detections_df['frame_time', 'corrected'].diff().mean()])

    ### END detections formating (read function update)

    # find task starts (by ["Ready?", "3", "2", "1"] detection)

    starts = (
        detections_df[
            [
                ("Ready?", "score"),
                ("three", "score"),
                ("two", "score"),
                ("one", "score"),
            ]
        ]
        >= 0.975
    ).any(1)
    # print(starts)

    ww = 75  # (window_width // 2) * 33.33ms = minimum time object must be detected
    starts = detections_df.loc[(starts.rolling(ww, center=True).sum() > ww // 2)]
    start_num = (starts["frame_time", "corrected"].diff() > pd.Timedelta("5s")).cumsum()

    max_task_length = pd.Timedelta("6m")

    # '#': {'start': , 'end': , name}
    tasks = {}

    # groupby start, then find first object that appears after last start
    for start_count, g in starts.groupby(start_num):
        print(f"start #{start_count} @ {g['frame_time', 'corrected'].iloc[-1]}")

        # get last frame number
        final_start_frame_number = g.iloc[-1].name

        # find the actual task start (first object appearance)
        # find_first_object_appearance(final_start_frame_time, ['cross', 'grid'])
        start_objects = ["grid", "target", "cross"]

        seconds = 3
        fps = 30

        start_location = detections_df.loc[
            g.iloc[-1].name : g.iloc[-1].name + seconds * fps
        ]

        # find first appearance for grid, target, cross
        first_cross = first_object_appearance(
            start_location["cross"].join(
                start_location["frame_time", "corrected"].rename("time")
            )
        )
        first_grid = first_object_appearance(
            start_location["grid"].join(
                start_location["frame_time", "corrected"].rename("time")
            )
        )
        first_target = first_object_appearance(
            start_location["target"].join(
                start_location["frame_time", "corrected"].rename("time")
            )
        )

        # find first done and last object of task (returns last object appearance if no done found)
        done, last_object = find_done_and_last_object(
            detections_df.loc[
                detections_df["frame_time", "corrected"].between(
                    g["frame_time", "corrected"].iloc[-1],
                    g["frame_time", "corrected"].iloc[-1] + max_task_length,
                )
            ]
        )

        tasks[f"{start_count}"] = {
            "ready_three": find_switch(g, "Ready?", "three"),
            "three_two": find_switch(g, "three", "two"),
            "two_one": find_switch(g, "two", "one"),
            "first_grid": first_grid.time,
            "first_cross": first_cross.time,
            "first_target": first_target.time,
            'last_object': last_object,
            'done': done,
            # 'name':
        }

    # first object that appears will be either the cross or the grid.
    detected_tasks = pd.DataFrame(tasks).T
    detected_tasks['duration'] = detected_tasks['last_object'] - detected_tasks[['first_grid', 'first_cross', 'first_target']].min(axis=1)
    # detected_tasks['between'] = detected_tasks[['first_grid', 'first_cross', 'first_target']].min(axis=1).shift(-1) - detected_tasks['last_object']
    
    # print(detected_tasks[['duration', 'between']])


    # open file listing
    file_listing = pd.read_excel(
        "/home/irz0002/Projects/hmpldat/sample_data/listings/VMIB_listing.xlsx",
        sheet_name=None,
        index_col=[0, 1],
    )

    dflow_tasks = hmpldat.align.temporal.dflow_task_order(file_listing["vmib_038"]).unstack()
    dflow_tasks['duration'] = dflow_tasks['end'] - dflow_tasks['start']
    dflow_tasks = dflow_tasks.sort_values('start').reset_index()
    # dflow_tasks['between'] = dflow_tasks['start'].shift(-1) - dflow_tasks['end']

    i = 0
    j = 0

    while True:

        if dflow_tasks['duration'].iloc[0] - tasks[''] 
        detected_task = tasks

        

    print(dflow_tasks)

    # using merge asof to match on duration
    together = pd.merge_asof(dflow_tasks.sort_values('duration'), detected_tasks.sort_values('duration'), on='duration', tolerance=pd.Timedelta('1.5s'), direction='nearest')  
    together = together.sort_values('start').reset_index(drop=True)

    diff_between_starts = together['start'] - together[['first_grid', 'first_cross', 'first_target']].min(axis=1)
    diff_between_starts_match = (diff_between_starts - diff_between_starts.median()) < pd.Timedelta('2s')
    # print(diff_between_starts)
    # print(diff_between_starts_match)

    together = together[diff_between_starts_match]
    together['dflow_between'] = together['start'].shift(-1) - together['end']
    together['detected_between'] = together[['first_grid', 'first_cross', 'first_target']].min(axis=1).shift(-1) - together['last_object']

    with pd.ExcelWriter('merged_matched.xlsx', datetime_format='HH:MM:SS.0000') as writer:
        together.to_excel(writer)
    
    # hmpldat.file.dflow.open()

    # one = last_object_appearance(detections["one"])
    # done = first_object_appearance(detections["Done"])

    # # assert len(one) == len(done)

    # detected_tasks = pd.DataFrame(zip(reversed(one), reversed(done)), columns=["start", "end"])
    # detected_tasks["duration"] = detected_tasks["end"] - detected_tasks["start"]

    # print(detected_tasks)

    # print(detections.columns)
    # print(detections)


if __name__ == "__main__":
    main()
