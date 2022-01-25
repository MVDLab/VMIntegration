"""how accurate is our 5V pulse?

step between previous record and first record with a pulse

Ian Zurutuza
Juneteenth, 2020
"""

from itertools import chain
import multiprocessing
from pathlib import Path
from pprint import pprint
import re

import numpy as np
import pandas as pd

import hmpldat.file.search
import hmpldat.file.dflow



def calc_stuff(x):

    f = x[0]
    pname = x[1]

    try:
        df = pd.read_csv(f, sep="\t", comment="#")
    except pd.errors.EmptyDataError as e:
        pulses = [{
            "filename": f.name,
            "participant": pname,
            "source": None,
            "length[sec]": np.NaN,
            "step_to_previous[sec]": np.NaN,
            "max_voltage": np.NaN,
            "info": e,
            }]
        steps = pd.Series(name=(pname, f.name), dtype=float)

        return pulses, steps

    pulses = find_5v_pulse(df, pname, f.name)
    steps = calc_step_size(df, pname, f.name)

    return pulses, steps


def find_5v_pulse(df, pname, fname):
    """

    Args:
        df: data
        dflow_type: "rd" or "mc" 

    Returns: list of pulse dictionaries
        * source (column Name with pulse, or None if not found)
        * length of pulse
        * step to previous record (NaN if no previous record, data has been trimmed)

    """

    if "mc" in fname:
        potential_pulse_columns = [f"Channel{n}.Anlg" for n in range(1,17)]
        time = "TimeStamp"
    elif "rd" in fname:
        potential_pulse_columns = ["RNP"]
        time = "Time"
    else: 
        raise ValueError("expects 'mc'|'rd' as dflow type")

    pulses = []

    for c in potential_pulse_columns:

        try:
            pulse_col = df[c]
        except KeyError:
            # pulse column not found
            pulse_info = {
                "filename": fname,
                "participant": pname,
                "source": c,
                "length[sec]": np.NaN,
                "step_to_previous[sec]": np.NaN,
                "max_voltage": np.NaN,
                "info": "column does not exist",
                } 

            pulses.append(pulse_info)
            continue 
        
        pulse = pulse_col[pulse_col >= 4]

        if len(pulse) > 0:
            pulse_info = {
                "filename": fname,
                "participant": pname,
                "source": c,
                "length[sec]": df[time].iloc[pulse.index[-1]] - df[time].iloc[pulse.index[0]],
                "step_to_previous[sec]": np.NaN if pulse.index[0] == 0 else df[time].iloc[pulse.index[0]] - df[time].iloc[pulse.index[0]-1],
                "max_voltage": pulse_col.max(),
                "info": "pulse found",
                }            
        else:
            pulse_info = {
                "filename": fname,
                "participant": pname,
                "source": c,
                "length[sec]": np.NaN,
                "step_to_previous[sec]": np.NaN,
                "max_voltage": pulse_col.max(),
                "info": "no pulse found",
                } 

        pulses.append(pulse_info)

    # pprint(pulses)
    return pulses


def calc_step_size(df, pname, fname):
    """

    Args:
        Series of time stamps for each recorded datarow
            - "Time" (dflow_rd)
            - "TimeStamp (dflow_mc)

    Returns:
        series of lengths between each record's timestamp

    """

    if "mc" in fname:
        time = "TimeStamp"
    elif "rd" in fname:
        time = "Time"
    else: 
        raise ValueError("expects 'mc'|'rd' as dflow type")

    times = df[time]
    steps = times.diff()
    steps.name = (pname, fname)

    return steps


def main():

    MAX_NUM = 5

    # find all dflow environment files (dflow_rd)
    dflow_rd = hmpldat.file.search.files(Path("/home/irz0002/Documents/projects/HMP"), ["dflow", "rd"], ['cortex', 'ducks', 'vt']) #[:MAX_NUM]
    # pprint(dflow_rd)

    # find all dflow environment files (dflow_mc)
    dflow_mc = hmpldat.file.search.files(Path("/home/irz0002/Documents/projects/HMP"), ["dflow", "mc"], ['cortex', 'ducks', 'vt'])#[:MAX_NUM]
    # pprint(dflow_mc)

    to_process = []

    pulse_info_list = []
    step_info_list = []

    file_read_failures = []

    for f in dflow_rd + dflow_mc:
        # open each

        pname_a = str(f).split("/")[-2]
        pname_b = str(f).split("/")[-3]
        fname = f.name

        a_match = re.search(r"[a-zA-Z]{4}_[0-9]{3}", pname_a)
        b_match = re.search(r"[a-zA-Z]{4}_[0-9]{3}", pname_b)

        if a_match is not None:
            pname = pname_a.lower()
        elif b_match is not None:
            pname = pname_b.lower()
        else:
            pname = "_".join(fname.split("_")[:2]).lower()

        to_process.append((f, pname))
        # pulse_info_list = pulse_info_list + find_5v_pulse()
        # step_info_list.append(calc_step_size(df, pname, fname))


    with multiprocessing.Pool(16) as p:
        res = p.map(calc_stuff, to_process)

    # print(res)
    pulse_info_list, step_info_list = list(zip(*res))

    # unpack list of lists
    pulse_info_list = chain.from_iterable(pulse_info_list)

    pulse_info = pd.DataFrame(pulse_info_list)
    pulse_info = pulse_info.set_index(["source", "participant", "filename"]).sort_index().dropna(subset=["length[sec]"])
    pulse_info["type"] = np.where(pulse_info.index.get_level_values(-1).str.contains("rd") == True, "rd", "mc")

    step_info = pd.DataFrame(step_info_list)
    step_info.index = pd.MultiIndex.from_tuples(step_info.index, names=["participant", "filename"])
    step_info["type"] = np.where(step_info.index.get_level_values(1).str.contains("rd") == True, "rd", "mc")
    step_info = step_info.set_index("type", append=True).swaplevel().swaplevel(1,0).T

    rd_step_info = step_info["rd"]
    mc_step_info = step_info["mc"]

    print(step_info)
    avg_step_info = {}

    for k in step_info.columns.unique(level=0):
        print(k)

        nk = "_".join(["dflow", k])

        avg_step_info[nk] = {}
        
        avg_step_info[nk]["mean"] = np.nanmean(step_info[k].values)
        avg_step_info[nk]["std"] = np.nanstd(step_info[k].values)
        avg_step_info[nk]["min"] = np.nanmin(step_info[k].values)
        avg_step_info[nk]["25%"] = np.nanquantile(step_info[k].values, .25)
        avg_step_info[nk]["median"] = np.nanquantile(step_info[k].values, .5)
        avg_step_info[nk]["75%"] = np.nanquantile(step_info[k].values, .75)
        avg_step_info[nk]["max"] = np.nanmax(step_info[k].values)

    # 2 columns  
    avg_step_info = pd.DataFrame.from_dict(avg_step_info) #, orient=)
    print(avg_step_info)

    ff = "%.6f" # set float format

    with pd.ExcelWriter("pulse_and_step_info.xlsx") as w:

        pulse_info.to_excel(w, sheet_name="pulse_info", float_format=ff)
        avg_step_info.to_excel(w, sheet_name="overall_step_info", float_format=ff)
        mc_step_info.describe().T.to_excel(w, sheet_name="mc_by_individual", float_format=ff)
        rd_step_info.describe().T.to_excel(w, sheet_name="rd_by_individual", float_format=ff)



if __name__=="__main__":
    main()
