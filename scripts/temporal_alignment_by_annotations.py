
# from standard library (no install required)
import glob
import os
from pathlib import Path
import sys
from pprint import pprint

from hmpldat.file.participant import participant
from hmpldat.file.task import task
import hmpldat.file.rawetg 


def main():


    b = ('VMIB', '005')
    a = ('VMIB', '047')

    # for each merged data file
    # for f in search.files(DATA_PATH, []):  # a_task or all_tasks_from_one_session
    for study, participant in [b]:

        p = hmpldat.file.participant.participant(study, participant)

        p.rawetg_path = Path(f'~/Desktop/selected_data/VMIB/Data/ETG/Metrics Export/VMIB_{participant}_RawETG.txt')
        p.detections_path = Path(f'~/Desktop/selected_data/VMIB/Data/detections/vmib_{participant}-1-unpack.txt')

        # p.create_file_listing(Path('selected_data_file_listing.xlsx'), Path('/home/ian/Desktop/selected_data/VMIB/Data'))
        p.load_file_listing(
            Path('/home/ian/Projects/hmpldat/selected_data_file_listing.xlsx')
        )

        p.load_rawetg_and_objects()
        
        p.create_tasks()
        output_data_dir = Path(f'/home/ian/Projects/hmpldat/test/{p.experiment}/{p.name}')

        # don't overwrite data by default
        if output_data_dir.exists():
            pass
            # i = input("directory already exists. continue? [Y/n] ")
            # if i.lower() == 'n':
            #     exit()
        else:
            output_data_dir.mkdir(parents=True)

        # TODO: keep n samples across all tasks -> one camera model per participant
        # samples = 

        # for each task, sample
        for task in p.tasks.values():
            print(task.name, type(task), hasattr(task, 'merged'))

            if task.name == 'int':          
                task.align()

            # skip tasks without merged data
            if not hasattr(task, 'merged'):
                continue
            
            task.merged.to_csv(output_data_dir / ("_".join([task.name, str(task.trial)]) + '.csv'))

            



    # # open rawetg 
    # rawetg_df = hmpldat.file.rawetg.open(p.rawetg_path)
    
    # with_frame_num = rawetg_df.assign(frame_number=hmpldat.file.rawetg.frame_number_from_vidtime(rawetg_df))

    # print(with_frame_num)

    # with_frame_num.to_excel('testing123.xlsx')

    # annotated_tasks = hmpldat.file.rawetg.get_tasks(rawetg_df)
    # print(annotated_tasks)

    # # these names need to change
    # converters = {
    #     'calib': 'ts',
    #     'qs_eo': 'qs_open',
    #     'qs_eo_cross': 'qs_cross'
    #     'qs_ec': 'qs_closed',
    #     'fixation': 'fix',
    #     'intercept': 'int',
    # }
    # # print(rawetg_df)







if __name__ == "__main__":
    main()