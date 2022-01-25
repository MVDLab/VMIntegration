"""
Representation of a task object (dflow and cortex data)

Ian Zurutuza
August 7, 2020
"""

# standard library
from pathlib import Path
from pprint import pprint
import re
import sys

# 3rd party
import numpy as np
import pandas as pd
#import xarray as xr


from hmpldat.align.temporal import dflow_to_cortex
import hmpldat.align.temporal 

"""START meaning "Ready?", "3", "2", "1" before task """
TASKS_WITH_START = [
    "ts",  # troubleshooting
    "fix",  # fixation
    "hp",  # horizontal pursuit
    "pp",  # peripheral pursuit
    "ap",  # angled pursuit
    "safezone",  # ... safezone
    "int",  # intercept
    "avoid",  # ... avoid
]

""" Tasks without standard "Ready?", "3", "2", "1" """
TASKS_WITHOUT_START = [
    "qs_open",  # quiet standing, eyes open (blank screen)
    "qs_cross",  # quiet standing, eyes open (cross)
    "qs_closed",  # quiet standing, eyes closed (cross)
    "vt",  # visual tracking (disk match)
    "ducks",  # ducks
]

MERGE_TOLERANCE = pd.Timedelta("5ms")

class task:
    def __init__(self, name, trial, mc_path, cortex_path, rawetg_and_detections=None, rd_path=None, ev_path=None):

        # init file locations for these files
        self.mc_path = mc_path
        self.cortex_path = cortex_path
        self.rd_path = rd_path
        self.ev_path = ev_path
        self.trial = trial
        self.name = name
        print(f"{name}, {trial}")

        # section of dataframe associated with this task
        self.rawetg_and_objects_df = rawetg_and_detections
        
        self.open_mc()
        self.open_cortex()
        if(rd_path is not None):
            self.open_rd()
            # self.align()
        if (ev_path is not None):
            self.open_ev()

        # open these files and align a single dataframe
        # self.data =

        # I don't know which video at this point.
        # self.video_path = video_path
        # self.rawetg_path = rawetg_path
    
    def __repr__(self):
        return str({
            'name': self.name,
            'trial': self.trial,
            'dflow_mc': self.mc_path,
            'dflow_rd': self.rd_path,
            'dflow_ev': self.ev_path,
            'cortex': self.cortex_path,
        })

    def __str__(self):
        return f'task: {self.name} trial: {self.trial}'

    def parse_filenames_for_task_name(self):

        # use a set for taskname representation (this will make it easy to check for differing names)
        task_name = []
        
        # parse task name for dflow mc files
        dflow_mc_match = re.search(r".*?([a-z]{2,})_(e|d)?_mc([\d]{4}).*?", self.mc_path.name.lower())
        if dflow_mc_match:
            task_name.append("_".join([dflow_mc_match.group(1), dflow_mc_match.group(3).lstrip("0")]))

        # parse task name for dflow rd files
        if self.rd_path is not None:
            # parse task name for dflow files
            dflow_rd_match = re.search(r".*?([a-z]{2,})_(e|d)?_rd([\d]{4}).*?", self.rd_path.name.lower())
            if dflow_rd_match:
                task_name.append("_".join([dflow_rd_match.group(1), dflow_rd_match.group(3).lstrip("0")]))

        # parse task name for dflow ev files
        if self.ev_path is not None:
            # parse task name for dflow files
            dflow_ev_match = re.search(r".*?([a-z]{2,})_(e|d)?_ev([\d]{4}).*?", self.ev_path.name.lower())
            if dflow_ev_match:
                task_name.append("_".join([dflow_ev_match.group(1), dflow_ev_match.group(3).lstrip("0")]))

        # parse task name from cortex file
        cortex_match = re.search(r".*_([a-z]{2,}).*?(e|d)?(\d{1,2}).*", self.cortex_path.name.lower())
        if cortex_match:
            task_name.append("_".join([cortex_match.group(1), cortex_match.group(3)]))

        # check filename for e|d character (task with distractors or venice)

        task_name = set(task_name)
        if len(task_name) > 1:
            sys.exit(f'ERROR: parsed task names: {task_name} are not consistent')

        return task_name.pop()

    # TODO: search of ChannelXX.Anlg that contains the 5V pulse 
    def open_mc(self):
        """open dflow mc file into dataframe, this frame will contain 5V pulse and force plate data"""

        # do NOT read motion capture data from this file!
        # see: usecols arg
        df = pd.read_table(
            self.mc_path,
            sep='\t',
            header=0,
            comment='#',
            usecols=lambda x: any(s in x for s in ['TimeStamp', 'FrameNumber', 'FP1', 'FP2', 'Channel'])
        )

        # lower case all column names
        df.columns = df.columns.str.lower()

        # remove 'pos' from column names
        df.columns = df.columns.str.replace('pos', '')

        # rename time column
        df = df.rename({'timestamp': 'time'}, axis='columns')

        # drop frame number 
        df = df.drop('framenumber', axis='columns')

        # set index as time column
        df = df.set_index('time')

        # change index to timedelta
        df.index = pd.to_timedelta(df.index, unit="s")

        # search for dflow anlg channel that goes high, drop analog channels that don't
        channels_df = df.loc[:, df.columns.str.contains('channel')]
        df = df.drop(channels_df.columns, axis='columns')

        # which channels go "high", receive 5V pulse
        # drop channels that do not go high
        pulse_channels = channels_df[channels_df[channels_df >= 4].dropna(axis='columns', how='all').columns]

        if len(pulse_channels.columns) > 1:
            import warnings
            warnings.wain(f"More than one analog channel contain a 5V pulse. [{pulse_channels.columns}]")

        if pulse_channels.columns[0] != "channel16.anlg":
            raise ValueError(f"5V pulse not contained in channel16.anlg [{self.mc_path}]")
        
        # convert column to boolean?
        # pulse_channels = pulse_channels >= 4
      
        df = pd.concat([df, pulse_channels], axis=1)

        self.dflow_mc_df = df
        

    # TODO: handle differently vt, ducks
    def open_rd(self):
        """open rd file into dataframe
        
        Notes: 
            * this frame will contain vr environment information
            * may contain 5V pulse


        """

        df = pd.read_table(
            self.rd_path,
            sep='\t',
            header=0,
            comment='#'
        )

        # lower case all column names
        df.columns = df.columns.str.lower()

        # remove 'pos' from column names
        df.columns = df.columns.str.replace('pos', '')

        # convert <object>visible.bool columns to boolean type
        df.loc[:, df.columns.str.contains(".bool")] = df.loc[:, df.columns.str.contains(".bool")].astype('boolean')

        # set index as time column
        df = df.set_index('time')

        df.index = pd.to_timedelta(df.index, unit="s")

        self.dflow_rd_df = df
        

    def open_ev(self):
        """event data, this file only exists for tasks "vt"(disk match) and "ducks" """

        try:
            self.dflow_ev_df = pd.read_table(
                self.ev_path,
                sep="\t",
                header=0
            )
        except:
            print("Unexpected error:", sys.exc_info()[0])

        # lower case all column names
        self.dflow_ev_df.columns = self.dflow_ev_df.columns.str.lower()

        # set index as time column
        self.dflow_ev_df = self.dflow_ev_df.set_index('time')

        self.dflow_ev_df.index = pd.to_timedelta(self.dflow_ev_df.index, unit="s")


    def open_cortex(self):
        """open cortex file into dataframe, contains motion capture data """

        # first read only the column names
        try:
            cols = pd.read_table(
                self.cortex_path, sep="\t", header=[3, 4], nrows=0
            )
        except (pd.errors.ParserError):
            print('read error')
            cols = pd.read_table(
                self.cortex_path,
                sep="\t",
                header=[3, 4],
                nrows=0,
                engine="python",
                error_bad_lines=False,
            )

        # edit column names to make column names into single level index similar to dflow data?
        # I am droping the number value associated with each marker (We need only "x", "y", "z")
        new_cols = []
        a_col_name = ""
        for col in cols:
            if "Unnamed" in col[0]:
                if "Unnamed" not in col[1]:
                    new_cols.append((a_col_name + "." + col[1][0]).lower())
                else:
                    pass
            else:
                if "Unnamed" in col[1]:
                    new_cols.append(col[0].lower())
                else:
                    new_cols.append((col[0] + "." + col[1][0]).lower())
                a_col_name = col[0]
        
        # read data but use my edited column names
        try:
            self.cortex_df = pd.read_table(
                self.cortex_path,
                sep="\t",
                header=None,
                names=new_cols,
                index_col=False,
                skiprows=[0, 1, 2, 3, 4, 5],
                skip_blank_lines=False,
                error_bad_lines=False,
            )
        except:
            self.cortex_df = pd.read_table(
                self.cortex_path,
                sep="\t",
                header=None,
                names=new_cols,
                skiprows=[0, 1, 2, 3, 4, 5],
                index_col=False,
                skip_blank_lines=False,
                error_bad_lines=False,
                engine="python",
            )

        self.cortex_df = self.cortex_df.set_index('time')
        
        # drop frame number index from cortex system
        self.cortex_df = self.cortex_df.drop('frame#', axis=1)
        
        # # convert to a timedelta index
        self.cortex_df.index = pd.to_timedelta(self.cortex_df.index, unit="s")
        # self.cortex_df.set_index("cortex_time", inplace=True)
      
        # self.cortex_df.columns = self.cortex_df.columns.set_names(['marker', 'axis'])

        # convert to xarray?
        # self.cortex_df = self.cortex_df.stack()
        # self.cortex_xr = self.cortex_df.to_xarray()
        # self.cortex_xr

        # print(self.cortex_xr)


    # TODO: handle task if dflow_rd does not exist
    def align(self):
        """align mc, rd, cortex files"""

        """Using current methods, aligns files.  
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

        # print('aligning')

        self.merged = hmpldat.align.temporal.concat_and_interpolate(
            self.dflow_mc_df, 
            self.dflow_rd_df, 
            self.cortex_df, 
            self.rawetg_and_objects_df,
            )


        # # append these indexes as columns to preserve time information
        # dflow_rd = self.dflow_rd_df.assign(time_rd=self.dflow_rd_df.index)
        # dflow_mc = self.dflow_mc_df.assign(time_mc=self.dflow_mc_df.index)
        # cortex = self.cortex_df.assign(time_cortex=self.cortex_df.index)

        # # check for pulse in both Dflow files
        # if "RNP" in dflow_rd.columns:

        #     # the 5V pulse doesn't stay on for the entire task
        #     # use the first frame with 5V as start and align by time (cortex turns off when task is over?)

        #     # get start of 5V in dflow mc file
        #     dflow_mc_start = dflow_mc[dflow_mc["channel16.anlg"].gt(4)].index[0]

        #     # get start of 5V in dflow rd file
        #     dflow_rd_start = dflow_rd[dflow_rd["rnp"].gt(4)].index[0]

        #     # add dflow start pulse to cortex index
        #     # cortex always starts at zero
        #     # our new index is the running clock from dflow
        #     cortex.index = cortex.index + dflow_mc_start

        #     # align dflow_mc (forceplate data) to cortex motion capture data
        #     merged = pd.merge_asof(
        #         dflow_mc,
        #         cortex,
        #         right_on="time",
        #         left_index=True,
        #         direction="nearest",
        #         tolerance=MERGE_TOLERANCE,
        #     )

        #     # downsample and align dflow rd to merged(cortex + mc)
        #     merged = pd.merge_asof(
        #         merged,
        #         dflow_rd,
        #         right_on="time",
        #         left_index=True,
        #         direction="nearest",
        #         tolerance=MERGE_TOLERANCE,
        #     )

        # # else match together dflow file by timestamp then align to cortex
        # else:

        # create index @ 120Hz
        # this index is the length of our dflow data
        # from 0 to x time
        # len(dflow_mc)
        # print(dflow_mc.index, len(dflow_mc))
        # print(dflow_mc.index[-1] - dflow_mc.index[0], dflow_rd.index[-1] - dflow_mc.index[0])
        
        # _120hz_timeseries = pd.TimedeltaIndex([pd.Timedelta(x/120, unit='s') for x in range(len(dflow_mc)+100)], freq='infer')

        # clean = pd.DataFrame(_120hz_timeseries, columns=['time']).set_index('time')

        # # start dflow indicies @ 0
        # # TODO: 
        # dflow_mc.index = dflow_mc.index - dflow_mc.index[0]
        # dflow_rd.index = dflow_rd.index - dflow_rd.index[0]

        # # merge on nearest time (downsample vr environment data from dflow_rd)
        # merged = pd.merge_asof(
        #     clean,
        #     dflow_mc,
        #     right_index=True,
        #     left_index=True,
        #     direction="nearest",
        #     tolerance=MERGE_TOLERANCE,
        # )

        # # merge on nearest time (downsample vr environment data from dflow_rd)
        # merged = pd.merge_asof(
        #     merged,
        #     dflow_rd,
        #     right_on="time",
        #     left_index=True,
        #     direction="nearest",
        #     tolerance=MERGE_TOLERANCE,
        # )

        # # get start of 5V pulse
        # channel16_pulse = merged[merged["channel16.anlg"].gt(4)].index[0]

        # # add start pulse to cortex index
        # # cortex always starts at zero
        # cortex.index = cortex.index + channel16_pulse
        # # print(cortex)

        # # align combined dflow files to cortex
        # merged = pd.merge_asof(
        #     merged,
        #     cortex,
        #     left_index=True,
        #     right_index=True,
        #     direction="nearest",
        #     tolerance=MERGE_TOLERANCE,
        # )

        # # merged = merged.reset_index(drop=False)
        # # print(merged)

        # ### now merge rawetg+objects to CAREN's task data ###

        # # find index when first cross appears (in VR, dflow_rd)
        # # first_cross_time = merged[merged['crossvisible.bool'] == 1].iloc[0, merged.columns.get_indexer(['time'])].to_numpy()[0]
        # first_cross_time = merged[merged['crossvisible.bool'] == 1].index[0]

        # # print(first_cross_time, type(first_cross_time))

        # rawetg_start = self.rawetg_and_objects_df.iloc[0, self.rawetg_and_objects_df.columns.get_indexer(['RecordingTime [ms]'])].to_numpy()[0]
        # # rawetg_start = self.rawetg_and_objects_df.index[0]

        # # print(rawetg_start, type(rawetg_start))

        # # print(rawetg_start, first_cross_time, first_cross_time - rawetg_start)
        # self.rawetg_and_objects_df = self.rawetg_and_objects_df.assign(
        #     time=self.rawetg_and_objects_df.loc[:, 'RecordingTime [ms]'] - rawetg_start + first_cross_time
        # )

        # self.rawetg_and_objects_df.index = self.rawetg_and_objects_df.index.rename("unique_id")
        # self.rawetg_and_objects_df = self.rawetg_and_objects_df.reset_index()
        # self.rawetg_and_objects_df = self.rawetg_and_objects_df.set_index('time', drop=False)
        # # print(self.rawetg_and_objects_df['time'])

        # print(self.rawetg_and_objects_df)

        # # merge all to 120 Hz 
        # # but this may assign some of the rawetg+objects rows to more the one row of the merged data.
        # all_merged = pd.merge_asof(
        #     merged,
        #     self.rawetg_and_objects_df,
        #     left_index=True,
        #     right_index=True,
        #     direction="nearest",
        #     tolerance=pd.Timedelta('17ms'),
        # )

        # # all_merged = merged.merge(self.rawetg_and_objects_df, how='left', left_index=True, right_index=True)
        # # print(all_merged)

        # difference = abs(all_merged.index - all_merged['time'])
        # all_merged['difference'] = difference

        # rawetg_and_objects_nearest_time = all_merged.loc[all_merged.groupby('unique_id')['difference'].idxmin()]
        # print(rawetg_and_objects_nearest_time[['unique_id', 'difference']].dropna())

        # rawetg_and_objects_nearest_time = rawetg_and_objects_nearest_time[self.rawetg_and_objects_df.columns]

        # closest = all_merged.groupby('unique_id', as_index=False)['difference'].min() # min(self.rawetg_and_objects_df[self.rawetg_and_objects_df['unique_id'] == x].index ))
        # print(closest)

        # from pprint import pprint
        # pprint(grouped.groups)

        # so drop any duplicated rawetg+object rows
        # all_merged = all_merged.drop_duplicates('unique_id')

        # then drop rows without any rawetg data
        # TODO: test with subset=['unique_id'] -> resultant should be the same (maybe faster)
        # rawetg_and_objects_nearest_time = all_merged[rawetg_and_objects_nearest_time.columns].dropna(axis=0, how='all')


        # # now left merge, retains only unique entries from rawetg
        # all_merged = merged.merge(
        #     rawetg_and_objects_nearest_time,
        #     how='left',
        #     left_index=True,
        #     right_index=True,
        #     )

        # # all_merged = all_merged.reset_index()

        # ### TESTING ###
        # # TODO: make this a class method
        # # add time stamp to any Timedelta column
        # # save each to sheets of excel doc 

        # # convert timedeltas to to datetime, so excel will interpret it correctly
        # all_merged['time'] = all_merged['time'] + pd.Timestamp('1900-01-01')
        # all_merged.index = all_merged.index + pd.Timestamp('1900-01-01')

        # merged.index = merged.index + pd.Timestamp('1900-01-01')
        # self.rawetg_and_objects_df.index = self.rawetg_and_objects_df.index + pd.Timestamp('1900-01-01')
        # self.rawetg_and_objects_df['corrected_frame_time'] = self.rawetg_and_objects_df['corrected_frame_time'] + pd.Timestamp('1900-01-01')

        # # create writer, specify datetime format
        # writer = pd.ExcelWriter(f'{self.name}_test.xlsx',datetime_format='hh:mm:ss.000')
                
        # # save sheets
        # all_merged.to_excel(writer, sheet_name='merged')
        # # merged.to_excel(writer, sheet_name='CAREN')
        # self.rawetg_and_objects_df.to_excel(writer, sheet_name='ETG')

        # # save & close document
        # writer.close()
        # ### END TESTING ###

        # print(merged[['frame_number', 'Annotation Name']].head(25))

        # self.merged = all_merged


    def trial_number(self):
        """

        Args:
            DataFrame containing mocap positions [x,y,z] for a specific object

        Returns:
            df with instances where dflow says the object is visible 
            and an additional column assigning each group of instances a "trial number"

        Notes:
            Expect 330 trials with the target object
            Expect 346 trials with cross 

        """

        
        cross_vis = self.merged[self.merged["crossvisible.bool"] == 1]

        # make time index a column to perform required filtering ops
        # difference between two instances larger than one second?
        # assign each trial a number (0 indexed)
        self.merged = self.merged.assign(
            trial_number=(cross_vis.index.to_series().diff() > 1)
            .apply(lambda x: 1 if x else 0)
            .cumsum()
        )

        self.merged['trial_number'] = self.merged['trial_number'].fillna(method='ffill', axis=0)

        return self.merged['trial_number']


    def horizon_height(self):
        """define horizon for this task

        reverse engineered from lua code
        
        Returns:
            Horizon height in millimeters

        Notes:
            if we know where the cross is located then we can calculate the "particiapant height" used to initialize horizon for this task

        """

        cross_postion = self.merged[['cross.x', 'cross.y', 'cross.z']]
        cross_postion.columns = ['x', 'y', 'z']

        # cross position is calculated differently by task
        # but remains static across each task
        if self.name in ["fix", "bm", "hp", "pp"]:
            horizon = cross_postion['y'].to_numpy()[0] * 1000
        elif self.name in ["ap", "avoid", "int"]:
            horizon = -3230 / np.tan(
                np.arccos((cross_postion['z'].to_numpy()[0] * 1000 + 3230) / -2800) - np.pi / 2
            )
        else:
            raise ValueError(f"unable to calculate task height for task {task.name}_{task.trial}")

        self.horizon = horizon
        return horizon


    def task_start(self):
        """when does the first object appear?"""
        # return start
        pass

    def task_end(self):
        """when does the task end?"""
        pass


def test_evopen():

    t = task(
        Path(
            "/home/irz0002/Projects/HMP/Projects/VMIB/Data/DFlow/VMIB_007/VMIB_007_Dflow_VT_mc0001_09142016.txt"
        ),
        Path(
            "/home/irz0002/Projects/HMP/Projects/VMIB/Data/Cortex/Cortex/VMIB_007/TRC/VMIB_007_VT1.trc"
        ),
        Path(
            "/home/irz0002/Projects/HMP/Projects/VMIB/Data/DFlow/VMIB_007/VMIB_007_Dflow_VT_rd0001_09142016.txt"
        ),
        Path(
            "/home/irz0002/Projects/HMP/Projects/VMIB/Data/DFlow/VMIB_007/VMIB_007_Dflow_VT_ev0001_09142016.txt"
        ),
    )

    print(t)

    t.open_cortex()
    t.open_mc()
    t.open_rd()
    t.open_ev()

    print(t.cortex_df)
    print(t.dflow_mc_df)
    print(t.dflow_rd_df)
    print(t.dflow_ev_df)


    t2 = task(
        Path('/home/irz0002/Projects/HMP/Projects/VMAK/DFlow/VMAK_004/Visit2/VMAK_004_P2_Fix_E_mc0001.txt'),
        Path('/home/irz0002/Projects/HMP/Projects/VMAK/Cortex/VMAK_004/Visit2/TRC/VMAK_004_P2_Fix_E1-Eyetracker_Child_Large_Shutter.trc'),
        Path('/home/irz0002/Projects/HMP/Projects/VMAK/DFlow/VMAK_004/Visit2/VMAK_004_P2_Fix_E_rd0001.txt'),
    )

    print(t2)
    t2.open_cortex()
    t2.open_mc()
    t2.open_rd()
    # t2.open_ev()

    print(t.cortex_df)
    print(t.dflow_mc_df)
    print(t.dflow_rd_df)
    # print(t.dflow_ev_df)


if __name__ == "__main__":
    test_evopen()
