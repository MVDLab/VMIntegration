

from datetime import datetime
from pathlib import Path
from pprint import pprint
import re
import time

import pandas as pd

import hmpldat.file.search as s


# change this to match your path to the "Projects" folder on the shared drive
PATH_TO_PROJECTS = Path("/mnt/hdd/VMI_data/vmi/datasets")

STUDIES = ["VMIB", "VMAD", "VMTD", "VMUM"]

# regex string used to match files
# dflow specific
# for all, groups 1 and 2 represent participant ID
# group 5 catches task name
# group 6 (dflow) catches dflow file id (mc, rd, ev)
# group 7 (dflow) catches attempt number for this task (may not exist)
# group 8 creation date string "12182020" -> month day year 
DFLOW_FILE_PATTERN = r"([A-Z]{4})_?([\d]{3})_?(dflow_)?(qs_)?([A-Z]+\d*)_?(mc|rd|ev)([\d]{4})?_?([\d]+)?\.txt"
CORTEX_FILE_PATTERN = r"([A-Z]{2,4})_?([\d]{2,3})_?([A-Z]{0,2}.*_)?_?([A-Z]{2,}[\d]{1})_?(.*)?\.trc"
RAWETG_FILE_PATTERN = r"([A-Z]{4})_?(\d{3})(\(\d\))?_?(.*)\.txt"
VIDEO_FILE_PATTERN = r"*.avi"


class Participant:
    """Class to represent file listing for a participant"""

    def __init__(self, participant_id):

        # when was this object created?
        self.creation_date = datetime.now()

        # dictionary to hold file information for A participant
        # date = datetime.fromtimestamp(file.stat().st_mtime)
        # file_listing[date] = { task: { files } }
        self.file_listing = {}

        self.find_participant_files(participant_id)
        

    def find_participant_files(self, participant_id):
        # participant_id should be like "VMAD_004"
        pass


    def load(self, path):
        """open a previously created file listing"""



p = Participant("VMIB_004")

print("nope")

# for x in PATH_TO_PROJECTS.rglob("*.trc"):
#     print(x)    
#     print(re.match(CORTEX_FILE_PATTERN, x.name, re.I))
pd.set_option("display.max_colwidth", 200)

participants = s.participants(PATH_TO_PROJECTS)
pprint(list(participants))

todos = {}

for p in participants:
    todos[p] = {}
    
    files = s.bundle_associated(PATH_TO_PROJECTS, p, probe_for_duration=False)
    break

pprint(files)

# for task, g in files.groupby("task_name"):
#     print(g)