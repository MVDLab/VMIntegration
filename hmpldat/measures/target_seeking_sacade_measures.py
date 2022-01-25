import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math
from matplotlib import gridspec
from hmpldat.file.participant import participant
from pathlib import Path
import peakdetect
from scipy.signal import savgol_filter
import cv2
pd.set_option('mode.chained_assignment', None)
# TODO not used for this paper , fix at a later time
def prepare_merged_data(merged):
    cleaned = merged[merged['trial_number'].notna()]
    for i, dat in cleaned.iterrows():
        if (dat["target.score"] < 0.90):
            cleaned.loc[i, "target.score"] = np.nan
            cleaned.loc[i, "target.right"] = np.nan
            cleaned.loc[i, "target.left"] = np.nan
            cleaned.loc[i, "target.bottom"] = np.nan
            cleaned.loc[i, "target.top"] = np.nan
        if (dat["user.score"] < 0.90):
            cleaned.loc[i, "user.score"] = np.nan
            cleaned.loc[i, "user.right"] = np.nan
            cleaned.loc[i, "user.left"] = np.nan
            cleaned.loc[i, "user.bottom"] = np.nan
            cleaned.loc[i, "user.top"] = np.nan
        if (dat["cross.score"] < 0.90):
        # if (dat["cross.score"] < 0.90) :
            cleaned.loc[i, "cross.score"] = np.nan
            cleaned.loc[i, "cross.u"] = np.nan
            cleaned.loc[i, "cross.v"] = np.nan
    # for i, dat in cleaned.iterrows():
    #     if (((-218 < dat["Point of Regard Binocular X [px]"]) and (dat["Point of Regard Binocular X [px]"] < 1178)) and (
    #             (-130 < dat["Point of Regard Binocular Y [px]"]) and ( dat["Point of Regard Binocular Y [px]"]< 850))):
    #         cleaned.loc[i, "in_range"] = True
    #     else:
    #         cleaned.loc[i, "in_range"] = False
    cleaned = cleaned.assign(
        in_range=(cleaned["Point of Regard Binocular X [px]"].between(-218, 1178)
                  & cleaned["Point of Regard Binocular Y [px]"].between(-130, 850)),
        in_scene=(cleaned["Point of Regard Binocular X [px]"].between(0, 960)
                  & cleaned["Point of Regard Binocular Y [px]"].between(0, 720))
    )

    cleaned["xDif"] = cleaned["Point of Regard Binocular X [px]"].diff()
    cleaned["yDif"] = cleaned["Point of Regard Binocular Y [px]"].diff()
    cleaned["gazeDiff"] = np.sqrt(cleaned["xDif"] ** 2 + cleaned["yDif"] ** 2)
    cleaned["tcenter.x"] = (cleaned["target.right"] + cleaned["target.left"]) / 2
    cleaned["tcenter.y"] = (cleaned["target.bottom"] + cleaned["target.top"]) / 2
    cleaned["ucenter.x"] = (cleaned["user.right"] + cleaned["user.left"]) / 2
    cleaned["ucenter.y"] = (cleaned["user.bottom"] + cleaned["user.top"]) / 2
    cleaned["tcenter.dist"] = np.sqrt(
        (cleaned["tcenter.x"] - cleaned["Point of Regard Binocular X [px]"]) ** 2 +
        (cleaned["tcenter.y"] - cleaned["Point of Regard Binocular Y [px]"]) ** 2)
    cleaned["ucenter.dist"] = np.sqrt(
        (cleaned["ucenter.x"] - cleaned["Point of Regard Binocular X [px]"]) ** 2 +
        (cleaned["ucenter.y"] - cleaned["Point of Regard Binocular Y [px]"]) ** 2)
    cleaned["cross.dist"] = np.sqrt(
        (cleaned["cross.u"] - cleaned["Point of Regard Binocular X [px]"]) ** 2 +
        (cleaned["cross.v"] - cleaned["Point of Regard Binocular Y [px]"]) ** 2)
    cleaned["tcenter.dist"] = cleaned["tcenter.dist"] * 2.895
    cleaned["ucenter.dist"] = cleaned["ucenter.dist"] * 2.895
    cleaned["cross.dist"] = cleaned["cross.dist"] * 2.895
    cleaned["target.position"] = np.abs(cleaned["target.x"])
    cleaned["target.position"] = cleaned["target.position"] * 1000
    cleaned["objective.dist"] = cleaned[['tcenter.dist', 'ucenter.dist']].min(axis=1)
    insideTarget = []
    insideUser =[]
    insideRectangle =[]
    rotations = []
    rotations2 = []
    old=None
    for i, dat in cleaned.iterrows():
        insideTarget.append(
            ((dat["Point of Regard Binocular X [px]"] > dat["target.left"]) and
             (dat["Point of Regard Binocular X [px]"] < dat["target.right"]) and
             (dat["Point of Regard Binocular Y [px]"] > dat["target.top"]) and
             (dat["Point of Regard Binocular Y [px]"] < dat["target.bottom"]))
        )
        insideUser.append(
            ((dat["Point of Regard Binocular X [px]"] > dat["user.left"]) and
             (dat["Point of Regard Binocular X [px]"] < dat["user.right"]) and
             (dat["Point of Regard Binocular Y [px]"] > dat["user.top"]) and
             (dat["Point of Regard Binocular Y [px]"] < dat["user.bottom"]))
        )
        insideRectangle.append(
            ((dat["Point of Regard Binocular X [px]"] > dat["target.left"]) and
             (dat["Point of Regard Binocular X [px]"] < dat["user.right"]) and
             (dat["Point of Regard Binocular Y [px]"] > dat["target.top"]) and
             (dat["Point of Regard Binocular Y [px]"] < dat["user.bottom"])) or
            ((dat["Point of Regard Binocular X [px]"] > dat["user.left"]) and
             (dat["Point of Regard Binocular X [px]"] < dat["target.right"]) and
             (dat["Point of Regard Binocular Y [px]"] > dat["user.top"]) and
             (dat["Point of Regard Binocular Y [px]"] < dat["target.bottom"]))
        )
        if (old is None):
            rotations.append(0)
            rotations2.append(0)
            old=i
        else:
            bat=cleaned.loc[old]
            bbx = math.degrees(math.atan2(bat["rhead.z"] - bat["lhead.z"], bat["rhead.x"] - bat["lhead.x"]))
            ttx = math.degrees(math.atan2(dat["rhead.z"] - dat["lhead.z"], dat["rhead.x"] - dat["lhead.x"]))
            bby = math.degrees(math.atan2(bat["rhead.y"] - bat["lhead.y"], bat["rhead.x"] - bat["lhead.x"]))
            tty = math.degrees(math.atan2(dat["rhead.y"] - dat["lhead.y"], dat["rhead.x"] - dat["lhead.x"]))
            bbz = math.degrees(math.atan2(bat["rhead.y"] - bat["fhead.y"], bat["rhead.z"] - bat["fhead.z"]))
            ttz = math.degrees(math.atan2(dat["rhead.y"] - dat["fhead.y"], dat["rhead.z"] - dat["fhead.z"]))
            rotations.append((ttx-bbx)*125)
            rotations2.append(math.sqrt( (ttx-bbx)*(ttx-bbx) + (tty-bby)*(tty-bby) + (ttz-bbz)*(ttz-bbz) )*125)

            # rotations.append(0)
            old=i
    old = None
    old2 = None
    old3 = None
    # targetmoves = []
    cleaned["target.moves"]=False
    for l, row in cleaned.iterrows():
        if (row["targetvisible.bool"] == False):
            old = None
            old2 = None
            old3 = None
            # targetmoves.append(False)
            continue
        if (old is None or old2 is None or old3 is None):
            old3 = old2
            old2 = old
            old = l
            # targetmoves.append(False)
        else:
            if (row["target.z"] != cleaned.loc[old]["target.z"] and cleaned.loc[old]["target.z"] != cleaned.loc[old2]["target.z"] and cleaned.loc[old2]["target.z"] != cleaned.loc[old3]["target.z"]):
                # targetmoves.append(True)
                cleaned.loc[old3,"target.moves"] = True
            old3 = old2
            old2 = old
            old = l
            # else:
            #     old3 = old2
            #     old2 = old
            #     old = l
            #     targetmoves.append(False)
    cleaned['rotations'] = rotations
    cleaned['rotations2'] = rotations2
    cleaned['insideTarget'] = insideTarget
    cleaned['insideUser'] = insideUser
    cleaned['insideRectangle'] = insideRectangle
    # cleaned["target.moves"] = targetmoves
    cleaned["tcenter.dist"] = cleaned["tcenter.dist"].mask(cleaned["in_range"] == False, np.nan)
    cleaned["ucenter.dist"] = cleaned["ucenter.dist"].mask(cleaned["in_range"] == False, np.nan)
    cleaned["objective.dist"] = cleaned["objective.dist"].mask(cleaned["in_range"] == False, np.nan)
    return cleaned

def multi_plot_prep(task1 , task2):
    if not hasattr(task1, 'merged'):
        #TODO throw error or try to create merged file in the future
        print("does not have merged file")
        return None
    if not hasattr(task2, 'merged'):
        #TODO throw error or try to create merged file in the future
        print("does not have merged file")
        return None
    cleaned1 = task1.merged.copy()
    cleaned2 = task2.merged.copy()
    cleaned1 = prepare_merged_data(cleaned1)
    cleaned2 = prepare_merged_data(cleaned2)
    #Category Binocular
    trials = cleaned1["trial_number"].unique()
    trials1 = []
    trials2 = []
    for trial in trials:
        # if trial not in [4,11,14,32,38,39,77,79,87,88]:
        #     continue
        # if trial not in [14]:
        #     continue
        offset1 = cleaned1[(cleaned1["trial_number"] == trial) & (cleaned1["targetvisible.bool"] == True)].index[0]
        trial_data1 = cleaned1[(cleaned1["trial_number"] == trial)]
        trial_data1["plot_time"] = (trial_data1.index - offset1) * 1000
        trial_data1["plot_time2"] = trial_data1["plot_time"] - trial_data1[trial_data1["target.moves"] == True]["plot_time"].min()
        trial_data1.loc[trial_data1["plot_time"] > 80, "cross.score"] = np.nan
        trial_data1.loc[trial_data1["plot_time"] > 80, "cross.u"] = np.nan
        trial_data1.loc[trial_data1["plot_time"] > 80, "cross.v"] = np.nan
        trial_data1.loc[trial_data1["plot_time"] > 80, "cross.dist"] = np.nan
        # trial_data1["plot_time"] = (trial_data1.index ) * 1000
        trials1.append(trial_data1)

        offset2 = cleaned2[(cleaned2["trial_number"] == trial) & (cleaned2["targetvisible.bool"] == True)].index[0]
        trial_data2 = cleaned2[(cleaned2["trial_number"] == trial)]
        trial_data2["plot_time"] = (trial_data2.index - offset2) * 1000
        trial_data2["plot_time2"] = trial_data2["plot_time"] - trial_data2[trial_data2["target.moves"] == True]["plot_time"].min()
        trial_data2.loc[trial_data2["plot_time"] > 80, "cross.score"] = np.nan
        trial_data2.loc[trial_data2["plot_time"] > 80, "cross.u"] = np.nan
        trial_data2.loc[trial_data2["plot_time"] > 80, "cross.v"] = np.nan
        trial_data2.loc[trial_data2["plot_time"] > 80, "cross.dist"] = np.nan
        # trial_data2["plot_time"] = (trial_data2.index ) * 1000
        trials2.append(trial_data2)

    trials1L = pd.concat(trials1)
    trials1L["plot_time"] = trials1L["plot_time"].round(1)
    trials1L["plot_time2"] = trials1L["plot_time2"].round(3)
    trials2L = pd.concat(trials2)
    trials2L["plot_time"] = trials2L["plot_time"].round(1)
    trials2L["plot_time2"] = trials2L["plot_time2"].round(3)
    return trials1L , trials2L
def multi_plot_int(trials1L,trials2L,group):
    trialTypes={
    "Left to Left" : [0,1,18,52,59,61,65,69,70,82],
    "Left to Mid" : [6,7,12,15,27,42,50,71,80,81],
    "Left to Right" : [4,11,14,32,38,39,77,79,87,88],
    "Mid to Left" : [19,29,37,48,51,57,68,72,74,78],
    "Mid to Mid" : [9,13,21,35,41,66,67,83,84,85],
    "Mid to Right" :[3,22,23,24,43,54,62,63,75,86],
    "Right to Left" : [5,17,25,31,36,40,53,56,73,89],
    "Right to Mid" : [16,26,28,30,34,46,47,58,60,64],
    "Right to Right" : [2,8,10,20,33,44,45,49,55,76]
    }
    target_trials = trialTypes[group]
    trials1L = trials1L[trials1L["trial_number"].isin(target_trials)]
    trials2L = trials2L[trials2L["trial_number"].isin(target_trials)]
    trials1L["target.x"] = trials1L["target.x"] * 1000
    trials2L["target.x"] = trials2L["target.x"] * 1000
    trials1L["user.x"] = trials1L["user.x"] * 1000
    trials2L["user.x"] = trials2L["user.x"] * 1000
    xlim1 = trials1L[(trials1L["targetvisible.bool"] == True) & (trials1L["plot_time2"] > 0)]["plot_time2"].max()
    xlim2 = trials2L[(trials2L["targetvisible.bool"] == True) & (trials2L["plot_time2"] > 0)]["plot_time2"].max()
    print(xlim1)
    print(xlim2)
    if(xlim1>xlim2):
        xlimit = xlim2
    else:
        xlimit = xlim1
    xaxis = "plot_time2"
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle("Intercept " + group + " Trials for 05-47")
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

    ax = plt.subplot(gs[0])
    ax.set_xlim([0, xlimit])
    ax.set_ylabel('target position (mm)')
    ax.title.set_text('Autism')

    ax4 = plt.subplot(gs[1], sharex=ax , sharey=ax)
    ax4.set_ylabel('target position (mm)')
    ax4.title.set_text('Neurotypical')

    ax0 = plt.subplot(gs[2], sharex=ax)
    ax0.set_ylim([0, 1500])
    ax0.set_ylabel('gaze distance (mm)')


    ax1 = plt.subplot(gs[4], sharex=ax)
    ax1.set_ylim([-125, 125])
    ax1.set_ylabel('head angular velocity (°/s)')
    ax1.set_xlabel('time (ms)')

    ax2 = plt.subplot(gs[3], sharex=ax)
    ax2.set_ylim([0, 1500])
    ax2.set_ylabel('gaze distance (mm)')

    ax3 = plt.subplot(gs[5], sharex=ax)
    ax3.set_ylim([-125, 125])
    ax3.set_ylabel('head angular velocity (°/s)')
    ax3.set_xlabel('time (ms)')

    sns.lineplot(data=trials1L, x=xaxis, y="target.x", color='red', ax=ax)
    sns.lineplot(data=trials1L, x=xaxis, y="user.x", color='blue', ax=ax)

    sns.lineplot(data=trials2L, x=xaxis, y="target.x", color='red', ax=ax4)
    sns.lineplot(data=trials2L, x=xaxis, y="user.x", color='blue', ax=ax4)


    # sns.lineplot(data=trials1L, x=xaxis, y="cross.dist", color='gray', ax=ax0)
    sns.lineplot(data=trials1L, x=xaxis, y="ucenter.dist", color='blue', ax=ax0, style=True, dashes=[(2, 2)],
                 legend=False)
    sns.lineplot(data=trials1L, x=xaxis, y="tcenter.dist", color='red', ax=ax0, style=True, dashes=[(4, 1)],
                 legend=False)

    sns.lineplot(data=trials1L, x=xaxis, y="rotations", color='orange', ax=ax1)

    # sns.lineplot(data=trials2L, x=xaxis, y="cross.dist", color='gray', ax=ax2)
    sns.lineplot(data=trials2L, x=xaxis, y="ucenter.dist", color='blue', ax=ax2, style=True, dashes=[(2, 2)],
                 legend=False)
    sns.lineplot(data=trials2L, x=xaxis, y="tcenter.dist", color='red', ax=ax2, style=True, dashes=[(4, 1)],
                 legend=False)

    sns.lineplot(data=trials2L, x=xaxis, y="rotations", color='orange', ax=ax3)

    plt.subplots_adjust(hspace=.0)
    plt.show()
    return trials1L, trials2L
def fix_aligner(trials):
    leftFixTrials = [0,1,4,8,13,15,16,18,23,25]
    trialsAligned = trials.copy()
    trialsAligned.loc[trialsAligned["trial_number"].isin(leftFixTrials), "rotations"]=trialsAligned.loc[trialsAligned["trial_number"].isin(leftFixTrials), "rotations"].apply(lambda x: x*-1)
    return trialsAligned
def multi_plot_fix(trials1L,trials2L,multi = False,group = "Group A"):
    # target_trials = [0,4,9,14,15,23,24]
    trialTypes = {
    "Group A" : [10,12],
    "Group B" : [1, 16, 21,22],
    "Group C" : [2,4,9,15],
    "Group D" : [0,14,23,24],
    "Group E" : [7,17],
    "Group F" : [5,13,20,25],
    "Group G" : [3,19],
    "Group H" : [6,8,11,18],
    "Group Mid" : [10,12,7,17,3,19],
    }
    trials1L = fix_aligner(trials1L)
    trials2L = fix_aligner(trials2L)
    target_trials = trialTypes[group]
    trials1L = trials1L[trials1L["trial_number"].isin(target_trials)]
    trials1LM = trials1L.groupby(['plot_time']).median()
    trials2L = trials2L[trials2L["trial_number"].isin(target_trials)]
    trials2LM = trials2L.groupby(['plot_time']).median()
    xlim1 = trials1L[(trials1L["targetvisible.bool"] == True)]["plot_time"].max()
    xlim2 = trials2L[(trials2L["targetvisible.bool"] == True)]["plot_time"].max()
    print(xlim1)
    print(xlim2)
    if(xlim1>xlim2):
        xlimit = xlim2
    else:
        xlimit = xlim1
    xaxis = "plot_time"
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle("Fixation " + group + " Trial for 05-47")
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

    ax = plt.subplot(gs[0])
    ax.set_xlim([-600, xlimit])
    ax.set_ylim([0, 1500])
    ax.set_ylabel('gaze distance (mm)')
    ax.title.set_text('Autism')

    ax1 = plt.subplot(gs[2], sharex=ax)
    ax1.set_ylim([-125, 125])
    ax1.set_ylabel('head angular velocity (°/s)')
    ax1.set_xlabel('time (ms)')

    ax2 = plt.subplot(gs[1], sharex=ax)
    ax2.set_ylim([0, 1500])
    ax2.set_ylabel('gaze distance (mm)')
    ax2.title.set_text('Neurotypical')

    ax3 = plt.subplot(gs[3], sharex=ax)
    ax3.set_ylim([-125, 125])
    ax3.set_ylabel('head angular velocity (°/s)')
    ax3.set_xlabel('time (ms)')
    if(multi):
        colors=["red","blue","orange","green"]
        i=0
        for trial in target_trials:
            trial_data1 = trials1L[(trials1L["trial_number"] == trial)]
            trial_data2 = trials2L[(trials2L["trial_number"] == trial)]
            sns.lineplot(data=trial_data1, x=xaxis, y="cross.dist", color=colors[i], ax=ax)
            sns.lineplot(data=trial_data1, x=xaxis, y="tcenter.dist", color=colors[i], ax=ax, style=True, dashes=[(4, 1)],
                         legend=False)

            sns.lineplot(data=trial_data1, x=xaxis, y="rotations", color=colors[i], ax=ax1)

            sns.lineplot(data=trial_data2, x=xaxis, y="cross.dist", color=colors[i], ax=ax2)
            sns.lineplot(data=trial_data2, x=xaxis, y="tcenter.dist", color=colors[i], ax=ax2, style=True, dashes=[(4, 1)],
                         legend=False)

            sns.lineplot(data=trial_data2, x=xaxis, y="rotations", color=colors[i], ax=ax3)
            i=i+1
    else:
        lw=1
        sns.lineplot(data=trials1L, x=xaxis, y="cross.dist", color='gray', ax=ax, style=True, dashes=[(4, 0)],
                     legend=False, linewidth=lw)
        sns.lineplot(data=trials1L, x=xaxis, y="tcenter.dist", color='red', ax=ax, style=True, dashes=[(4, 0)],
                     legend=False, linewidth=lw)
        sns.lineplot(data=trials1L, x=xaxis, y="rotations", color='orange', ax=ax1, style=True, dashes=[(4, 0)],
                     legend=False, linewidth=lw)
        sns.lineplot(data=trials2L, x=xaxis, y="cross.dist", color='gray', ax=ax2, style=True, dashes=[(4, 0)],
                     legend=False, linewidth=lw)
        sns.lineplot(data=trials2L, x=xaxis, y="tcenter.dist", color='red', ax=ax2, style=True, dashes=[(4, 0)],
                     legend=False, linewidth=lw)
        sns.lineplot(data=trials2L, x=xaxis, y="rotations", color='orange', ax=ax3, style=True, dashes=[(4, 0)],
                     legend=False, linewidth=lw)


        # sns.lineplot(data=trials1LM, x=xaxis, y="cross.dist", color='gray', ax=ax, style=True, dashes=[(4, 2)],
        #              legend=False)
        # sns.lineplot(data=trials1LM, x=xaxis, y="tcenter.dist", color='red', ax=ax, style=True, dashes=[(4, 2)],
        #              legend=False)
        # sns.lineplot(data=trials1LM, x=xaxis, y="rotations", color='orange', ax=ax1, style=True, dashes=[(4, 2)],
        #              legend=False)
        # sns.lineplot(data=trials2LM, x=xaxis, y="cross.dist", color='gray', ax=ax2, style=True, dashes=[(4, 2)],
        #              legend=False)
        # sns.lineplot(data=trials2LM, x=xaxis, y="tcenter.dist", color='red', ax=ax2, style=True, dashes=[(4, 2)],
        #              legend=False)
        # sns.lineplot(data=trials2LM, x=xaxis, y="rotations", color='orange', ax=ax3, style=True, dashes=[(4, 2)],
        #              legend=False)

    plt.subplots_adjust(hspace=.0)
    plt.show()
    return trials1L, trials2L
def multi_plot_hp(trials1L,trials2L):
    left = [2, 6, 7, 8, 9, 10, 11, 16, 17, 19]
    right = [0, 1, 3, 4, 5, 12, 13, 14, 15, 18]
    target_trials = left
    trials1L = trials1L[trials1L["trial_number"].isin(target_trials)]
    trials2L = trials2L[trials2L["trial_number"].isin(target_trials)]
    trials1L["target.x"] = trials1L["target.x"] * 1000
    trials2L["target.x"] = trials2L["target.x"] * 1000
    trials1L["user.x"] = trials1L["user.x"] * 1000
    trials2L["user.x"] = trials2L["user.x"] * 1000
    xlim1 = trials1L[(trials1L["targetvisible.bool"] == True) & (trials1L["plot_time2"] > 0)]["plot_time2"].max()
    xlim2 = trials2L[(trials2L["targetvisible.bool"] == True) & (trials2L["plot_time2"] > 0)]["plot_time2"].max()
    print(xlim1)
    print(xlim2)
    if(xlim1>xlim2):
        xlimit = xlim2
    else:
        xlimit = xlim1
    xaxis = "plot_time2"
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle("Horizontal Pursuit Left Trials for 05-47")
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

    ax = plt.subplot(gs[0])
    ax.set_xlim([0, xlimit])
    ax.set_ylabel('target position (mm)')
    ax.title.set_text('Autism')

    ax4 = plt.subplot(gs[1], sharex=ax)
    ax4.set_ylabel('target position (mm)')
    ax4.title.set_text('Neurotypical')

    ax0 = plt.subplot(gs[2], sharex=ax)
    ax0.set_ylim([0, 1500])
    ax0.set_ylabel('gaze distance (mm)')

    ax1 = plt.subplot(gs[4], sharex=ax)
    ax1.set_ylim([-125, 125])
    ax1.set_ylabel('head angular velocity (°/s)')
    ax1.set_xlabel('time (ms)')

    ax2 = plt.subplot(gs[3], sharex=ax)
    ax2.set_ylim([0, 1500])
    ax2.set_ylabel('gaze distance (mm)')

    ax3 = plt.subplot(gs[5], sharex=ax)
    ax3.set_ylim([-125, 125])
    ax3.set_ylabel('head angular velocity (°/s)')
    ax3.set_xlabel('time (ms)')

    sns.lineplot(data=trials1L, x=xaxis, y="target.x", color='red', ax=ax)
    # sns.lineplot(data=trials1L, x=xaxis, y="user.x", color='blue', ax=ax)

    sns.lineplot(data=trials2L, x=xaxis, y="target.x", color='red', ax=ax4)
    # sns.lineplot(data=trials2L, x=xaxis, y="user.x", color='blue', ax=ax4)


    sns.lineplot(data=trials1L, x=xaxis, y="cross.dist", color='gray', ax=ax0)
    sns.lineplot(data=trials1L, x=xaxis, y="ucenter.dist", color='blue', ax=ax0, style=True, dashes=[(2, 2)],
                 legend=False)
    sns.lineplot(data=trials1L, x=xaxis, y="tcenter.dist", color='red', ax=ax0, style=True, dashes=[(4, 1)],
                 legend=False)

    sns.lineplot(data=trials1L, x=xaxis, y="rotations", color='orange', ax=ax1)

    sns.lineplot(data=trials2L, x=xaxis, y="cross.dist", color='gray', ax=ax2)
    sns.lineplot(data=trials2L, x=xaxis, y="ucenter.dist", color='blue', ax=ax2, style=True, dashes=[(2, 2)],
                 legend=False)
    sns.lineplot(data=trials2L, x=xaxis, y="tcenter.dist", color='red', ax=ax2, style=True, dashes=[(4, 1)],
                 legend=False)

    sns.lineplot(data=trials2L, x=xaxis, y="rotations", color='orange', ax=ax3)

    plt.subplots_adjust(hspace=.0)
    plt.show()
    return trials1L ,trials2L
def create_tasks(task_name="int"):
    experiment = "VMIB"
    participant_name = '005'
    p = participant(experiment, participant_name)
    p.load_file_listing(
        Path("C:/UNT HTC/file_listing_VMIB.xlsx")
    )
    p.rawetg_path = Path('C:/UNT HTC/selected_data/VMIB/Data/ETG/Metrics Export/VMIB_005_RawETG.txt')
    p.detections_path = Path('C:/UNT HTC/selected_data/VMIB/Data/detections/vmib_005-1-unpack.txt')
    output_data_dir = Path(f'C:/UNT HTC/{p.experiment}/{p.name}')
    p.load_rawetg_and_objects()
    p.create_tasks()
    task1 = p.tasks[(task_name,1)]
    task1.align()
    task1.trial_number()
    # task1.merged.to_csv(output_data_dir / ("_".join([task1.name, str(task1.trial)]) + '.csv'))
    participant_name = '047'
    p2 = participant(experiment, participant_name)
    p2.load_file_listing(
        Path("C:/UNT HTC/file_listing_VMIB.xlsx")
    )
    p2.rawetg_path = Path('C:/UNT HTC/selected_data/VMIB/Data/ETG/Metrics Export/VMIB_047_RawETG.txt')
    p2.detections_path = Path('C:/UNT HTC/selected_data/VMIB/Data/detections/vmib_047-1-unpack.txt')
    output_data_dir = Path(f'C:/UNT HTC/{p.experiment}/{p2.name}')
    p2.load_rawetg_and_objects()
    p2.create_tasks()
    task2 = p2.tasks[(task_name,1)]
    task2.align()
    task2.trial_number()
    # task2.merged.to_csv(output_data_dir / ("_".join([task2.name, str(task2.trial)]) + '.csv'))
    return task1, task2
def parse_sacade_data(sacade):
    # try:
        first = True
        notIn = True
        count = 0
        errorCount = 0
        countP = 0
        errorCountP = 0
        for data in sacade:
            if (np.isnan(data[0][4]) or np.isnan(data[1][4])):
                continue
            if(first):
                first=False
                firstAbs = data[1][4]
                firstPrc = data[1][4] / data[0][4]
            if(notIn and (data[1][4] > data[0][4])):
                errorCount += 1
            if(notIn and (data[0][5])):
                sacadeCount = count
                notIn = False
            count += 1
            if (notIn and (data[1][5])):
                sacadeCount = count
                notIn = False
            if (not notIn):
                countP += 1
                if (data[1][4] > data[0][4]):
                    errorCountP += 1
        return {
            "First sacade error" : firstAbs,
            "first sacade percent error" : firstPrc,
            "Sacade count to reach" : sacadeCount,
            #"Percent of error sacades" : errorCount/sacadeCount,
            "pursuit count" : countP,
            "pursuit error count" : errorCountP
        }
def multi_plot_gaze(trials1L,target_trials = 0, trial_name = "Fixation", person = "Autism", scc=0):
    trials1L = trials1L[trials1L["trial_number"] == target_trials ]
    xlim1 = trials1L[(trials1L["targetvisible.bool"] == True)]["plot_time"].max()
    print(xlim1)
    xlimit = xlim1
    xaxis = "plot_time"
    fig = plt.figure(figsize=(9, 9))
    fig.suptitle(trial_name + " trial " + str(target_trials) + " saccade count: " + str(scc))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    ax = plt.subplot(gs[0])
    ax.set_xlim([-600, xlimit])
    # ax.set_ylim([0, 1500])
    ax.set_ylabel('X-Y gaze location')
    ax.set_xlabel('time (ms)')
    ax.title.set_text(person)

    ax1 = plt.subplot(gs[1], sharex=ax)
    ax1.set_ylim([-125, 125])
    ax1.set_ylabel('head angular velocity (°/s)')

    lw = 1
    c=0
    old=None
    colors={
        'Blink':"red",
        'Visual Intake':"green",
        'Saccade':"blue",
        np.nan:"gray",
    }
    for i, row in trials1L.iterrows():
        if c == 0:
            c += 1
        else:
            data = pd.DataFrame([old,row])
            sns.lineplot(data=data, x=xaxis, y="Point of Regard Binocular X [px]", color=colors[row["Category Binocular"]], ax=ax, style=True, dashes=[(4, 0)],
                         legend=False, linewidth=lw)
            sns.lineplot(data=data, x=xaxis, y="Point of Regard Binocular Y [px]", color=colors[row["Category Binocular"]], ax=ax, style=True, dashes=[(1, 1)],
                         legend=False, linewidth=lw)
            sns.lineplot(data=data, x=xaxis, y="tcenter.x",
                         color="black", ax=ax, style=True, dashes=[(4, 0)],
                         legend=False, linewidth=lw)
            sns.lineplot(data=data, x=xaxis, y="tcenter.y",
                         color="black", ax=ax, style=True, dashes=[(1, 1)],
                         legend=False, linewidth=lw)
            sns.lineplot(data=data, x=xaxis, y="rotations", color="orange", ax=ax1)
        old = row
    plt.subplots_adjust(hspace=.0)
    plt.show()
    return trials1L
def fixation_hp_measures(cleaned,buffer=6,limit=0.10):
    # if not hasattr(task, 'merged'):
    #     #TODO throw error or try to create merged file in the future
    #     print("does not have merged file")
    #     return None
    #Category Binocular
    cleaned = apply_target_buffer(cleaned,buffer)
    trials = cleaned["trial_number"].unique()
    trialData = {}
    maxis = []
    rtt = "rotations2"
    for trial in trials:
        trial_data = cleaned[(cleaned["trial_number"] == trial)]
        maxis.append(abs(max(trial_data[rtt], key=abs)) * 0.10)
    maxi = np.nanmedian(maxis)
    for trial in trials:
        trial_data = cleaned[(cleaned["trial_number"] == trial)]
        xlim1 = trial_data[(trial_data["targetvisible.bool"] == True)]["plot_time"].max()
        time_to_target = np.nan
        n_of_inside = 0
        n_of_time = 0
        n_of_time_after = 0
        head_movement_count=0
        head_up=0
        head_down=0
        head_change=0
        peaks = peakdetect.peakdetect(trial_data[rtt], trial_data["plot_time"],20,0)
        peaks2 = []
        for dat in peaks[0]:
            if(maxi > dat[1]) or (dat[0] < 0):
                continue
            peaks2.append(dat)
        # for dat in peaks[1]:
        #     if (maxi > abs(dat[1])) or (dat[0] < 0):
        #         continue
        #     peaks2.append(dat)
        peak_count = len(peaks2)
        for i,dat in trial_data.iterrows():
            if dat["plot_time"] < 0:
                continue
            if xlim1 < dat["plot_time"]:
                break
            n_of_time += 1
            if dat["insideTarget"]:
                n_of_inside += 1
            if dat["insideTarget"] and np.isnan(time_to_target) :
                time_to_target = dat["plot_time"]
            if not np.isnan(time_to_target) :
                n_of_time_after += 1
            if abs(dat["rotations"])>maxi:
                head_movement_count+=1
        trialData[trial] = {
            "time to target" : time_to_target,
            "frames inside" : n_of_inside,
            "total frames" : n_of_time,
            "frames after reaching" : n_of_time_after,
            "peaks" : peaks2,
            "peak count": peak_count,
            "head movement count": head_movement_count,
        }

    return trialData
def int_measures(cleaned,buffer=6,limit=0.10):
    # if not hasattr(task, 'merged'):
    #     #TODO throw error or try to create merged file in the future
    #     print("does not have merged file")
    #     return None
    #Category Binocular
    cleaned = apply_target_buffer(cleaned, buffer)
    trials = cleaned["trial_number"].unique()
    trialData = {}
    maxis = []
    rtt = "rotations2"
    for trial in trials:
        trial_data = cleaned[(cleaned["trial_number"] == trial)]
        maxis.append(abs(max(trial_data[rtt], key=abs)) * 0.10)
    maxi = np.nanmedian(maxis)
    for trial in trials:
        trial_data = cleaned[(cleaned["trial_number"] == trial)]
        xlim1 = trial_data[(trial_data["targetvisible.bool"] == True)]["plot_time"].max()
        time_to_target = np.nan
        time_to_user = np.nan
        time_to_rectangle = np.nan
        n_of_inside = 0
        n_of_inside_user = 0
        n_of_inside_rectangle = 0
        n_of_time = 0
        n_of_time_after = 0
        n_of_time_after_user = 0
        n_of_time_after_rect = 0
        head_movement_count=0
        peaks = peakdetect.peakdetect(trial_data[rtt], trial_data["plot_time"], 20, 0)
        peaks2 = []
        for dat in peaks[0]:
            if (maxi > dat[1]) or (dat[0] < 0):
                continue
            peaks2.append(dat)
        # for dat in peaks[1]:
        #     if (maxi > abs(dat[1])) or (dat[0] < 0):
        #         continue
        #     peaks2.append(dat)
        peak_count = len(peaks2)
        for i,dat in trial_data.iterrows():
            if dat["plot_time"] < 0:
                continue
            if xlim1 < dat["plot_time"]:
                break
            n_of_time += 1
            if dat["insideTarget"]:
                n_of_inside += 1
            if dat["insideUser"]:
                n_of_inside_user += 1
            if dat["insideRectangle"] and not dat["insideUser"] and not dat["insideTarget"]:
                n_of_inside_rectangle += 1
            if dat["insideTarget"] and np.isnan(time_to_target):
                time_to_target = dat["plot_time"]
            if dat["insideUser"] and np.isnan(time_to_user):
                time_to_user = dat["plot_time"]
            if dat["insideRectangle"] and np.isnan(time_to_rectangle):
                time_to_rectangle = dat["plot_time"]
            if not np.isnan(time_to_target):
                n_of_time_after += 1
            if not np.isnan(time_to_user):
                n_of_time_after_user += 1
            if not np.isnan(time_to_rectangle):
                n_of_time_after_rect += 1
            if abs(dat["rotations"])>maxi:
                head_movement_count+=1
        trialData[trial] = {
            "time to target" : time_to_target,
            "time to user": time_to_user,
            "time to rectangle": time_to_rectangle,
            "frames inside" : n_of_inside,
            "frames inside user": n_of_inside_user,
            "frames inside rectangle": n_of_inside_rectangle,
            "total frames" : n_of_time,
            "frames after reaching" : n_of_time_after,
            "frames after reaching user": n_of_time_after_user,
            "frames after reaching rect": n_of_time_after_rect,
            "peaks" : peaks2,
            "peak count": peak_count,
            "head movement count": head_movement_count,
        }

    return trialData
def data_avg(PlotData,PlotData2,buffer=6,name="fix",limit=0.10):
    data = fixation_hp_measures(PlotData,buffer,limit)
    data2 = fixation_hp_measures(PlotData2,buffer,limit)
    print(name+" with "+str(buffer)+" buffer")
    t2t=[]
    fi=[]
    tf=[]
    far=[]
    head=[]
    keys=[]
    headc=[]
    for key in data:
        t2t.append(data[key]["time to target"])
        fi.append(data[key]["frames inside"]/data[key]["total frames"])
        far.append(data[key]["frames inside"]/(data[key]["frames after reaching"]+0.001))
        head.append(data[key]["peak count"])
        headc.append(data[key]["head movement count"]/data[key]["total frames"])
        keys.append(key)
    plt.scatter(keys, t2t)
    plt.show()
    print("ASD")
    print("Time to target, mean: %s median:   std: %s" % (np.nanmean(t2t),np.nanstd(t2t) ) )
    print("percentage inside target, mean: %s  std: %s" % (np.nanmean(fi), np.nanstd(fi)))
    print("percentage inside target after reaching target, mean: %s  std: %s" % (np.nanmean(far), np.nanstd(far)))
    print("Number of head movements, mean: %s  std: %s" % (np.nanmean(head), np.nanstd(head)))
    print("percentage of head movements, mean: %s  std: %s" % (np.nanmean(headc), np.nanstd(headc)))
    t2t = []
    fi = []
    tf = []
    far = []
    head = []
    keys = []
    for key in data2:
        t2t.append(data2[key]["time to target"])
        fi.append(data2[key]["frames inside"] / data2[key]["total frames"])
        far.append(data2[key]["frames inside"] / (data2[key]["frames after reaching"] + 0.001))
        head.append(data2[key]["peak count"])
        headc.append(data2[key]["head movement count"] / data2[key]["total frames"])
        keys.append(key)
    plt.scatter(keys, t2t)
    plt.show()
    print("TD")
    print("Time to target, mean: %s  std: %s" % (np.nanmean(t2t), np.nanstd(t2t)))
    print("percentage inside target, mean: %s  std: %s" % (np.nanmean(fi), np.nanstd(fi)))
    print("percentage inside target after reaching target, mean: %s  std: %s" % (np.nanmean(far), np.nanstd(far)))
    print("Number of head movements, mean: %s  std: %s" % (np.nanmean(head), np.nanstd(head)))
    print("percentage of head movements, mean: %s  std: %s" % (np.nanmean(headc), np.nanstd(headc)))
def int_avg(PlotData,PlotData2,buffer=6,name="int",limit=0.10):
    data = int_measures(PlotData,buffer,limit)
    data2 = int_measures(PlotData2,buffer,limit)
    print(name + " with " + str(buffer) + " buffer")
    t2o = []
    t2t = []
    t2u = []
    t2r = []
    fi = []
    fiu = []
    fir = []
    far = []
    fau = []
    farr = []
    head = []
    headc = []
    for key in data:
        t2o.append(min( [ data[key]["time to target"] , data[key]["time to user"]] ))
        t2t.append(data[key]["time to target"])
        t2u.append(data[key]["time to user"])
        t2r.append(data[key]["time to rectangle"])
        fi.append(data[key]["frames inside"] / data[key]["total frames"])
        fiu.append(data[key]["frames inside user"] / data[key]["total frames"])
        fir.append(data[key]["frames inside rectangle"] / data[key]["total frames"])
        far.append(data[key]["frames inside"] / (data[key]["frames after reaching"] + 0.001))
        fau.append(data[key]["frames inside user"] / (data[key]["frames after reaching user"] + 0.001))
        farr.append(data[key]["frames inside rectangle"] / (data[key]["frames after reaching rect"] + 0.001))
        head.append(data[key]["peak count"])
        headc.append(data[key]["head movement count"] / data[key]["total frames"])
    print("ASD")
    print("Time to object, mean: %s  std: %s" % (np.nanmean(t2o), np.nanstd(t2o)))
    print("Time to target, mean: %s  std: %s" % (np.nanmean(t2t), np.nanstd(t2t)))
    print("Time to user, mean: %s  std: %s" % (np.nanmean(t2u), np.nanstd(t2u)))
    print("Time to rectangle, mean: %s  std: %s" % (np.nanmean(t2r), np.nanstd(t2r)))
    print("percentage inside target, mean: %s  std: %s" % (np.nanmean(fi), np.nanstd(fi)))
    print("percentage inside user, mean: %s  std: %s" % (np.nanmean(fiu), np.nanstd(fiu)))
    print("percentage inside rectangle, mean: %s  std: %s" % (np.nanmean(fir), np.nanstd(fir)))
    print("percentage inside target after reaching target, mean: %s  std: %s" % (
    np.nanmean(far), np.nanstd(far)))
    print("percentage inside user after reaching user, mean: %s  std: %s" % (
        np.nanmean(fau), np.nanstd(fau)))
    print("percentage inside rectangle after reaching rectangle, mean: %s  std: %s" % (
        np.nanmean(farr), np.nanstd(farr)))
    print("Number of head movements, mean: %s  std: %s" % (np.nanmean(head), np.nanstd(head)))
    print("percentage of head movements, mean: %s  std: %s" % (np.nanmean(headc), np.nanstd(headc)))
    t2o = []
    t2t = []
    t2u = []
    t2r = []
    fi = []
    fiu = []
    fir =[]
    far = []
    fau = []
    farr = []
    head = []
    headc = []
    for key in data2:
        t2o.append(min([data2[key]["time to target"], data2[key]["time to user"]]))
        t2t.append(data2[key]["time to target"])
        t2u.append(data2[key]["time to user"])
        t2r.append(data2[key]["time to rectangle"])
        fi.append(data2[key]["frames inside"] / data2[key]["total frames"])
        fiu.append(data2[key]["frames inside user"] / data2[key]["total frames"])
        fir.append(data2[key]["frames inside rectangle"] / data2[key]["total frames"])
        far.append(data2[key]["frames inside"] / (data2[key]["frames after reaching"] + 0.001))
        fau.append(data2[key]["frames inside user"] / (data2[key]["frames after reaching user"] + 0.001))
        farr.append(data2[key]["frames inside rectangle"] / (data2[key]["frames after reaching rect"] + 0.001))
        head.append(data2[key]["peak count"])
        headc.append(data2[key]["head movement count"] / data2[key]["total frames"])
    print("TD")
    print("Time to object, mean: %s  std: %s" % (np.nanmean(t2o), np.nanstd(t2o)))
    print("Time to target, mean: %s  std: %s" % (np.nanmean(t2t), np.nanstd(t2t)))
    print("Time to user, mean: %s  std: %s" % (np.nanmean(t2u), np.nanstd(t2u)))
    print("Time to rectangle, mean: %s  std: %s" % (np.nanmean(t2r), np.nanstd(t2r)))
    print("percentage inside target, mean: %s  std: %s" % (np.nanmean(fi), np.nanstd(fi)))
    print("percentage inside user, mean: %s  std: %s" % (np.nanmean(fiu), np.nanstd(fiu)))
    print("percentage inside rectangle, mean: %s  std: %s" % (np.nanmean(fir), np.nanstd(fir)))
    print("percentage inside target after reaching target, mean: %s  std: %s" % (np.nanmean(far), np.nanstd(far)))
    print("percentage inside user after reaching user, mean: %s  std: %s" % (
        np.nanmean(fau), np.nanstd(fau)))
    print("percentage inside rectangle after reaching rectangle, mean: %s  std: %s" % (
        np.nanmean(farr), np.nanstd(farr)))
    print("Number of head movements, mean: %s  std: %s" % (np.nanmean(head), np.nanstd(head)))
    print("percentage of head movements, mean: %s  std: %s" % (np.nanmean(headc), np.nanstd(headc)))
def plot_peaks(PlotData,look=50,delta=5,smooth=False,rtt="rotations2"):
    trials = PlotData["trial_number"].unique()
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle("Head Movement Peak Graph With Width = "+str(look)+" and Delta = " + str(delta) + " Smoothed = " + str(smooth))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
    maxis=[]
    # maxi = abs(max(PlotData["rotations"], key=abs)) * 0.10
    for trial in trials:
        trial_data = PlotData[(PlotData["trial_number"] == trial)]
        maxis.append(abs(max(trial_data[rtt], key=abs)) * 0.10)
    maxi=np.nanmedian(maxis)
    for trial in [0,1,2,3,4,5]:
        trial_data = PlotData[(PlotData["trial_number"] == trial)]
        if(smooth):
            trial_data[rtt] = savgol_filter(trial_data[rtt], 21, 3,mode='nearest')
        peaks = peakdetect.peakdetect(trial_data[rtt], trial_data["plot_time"], look,delta)
        # maxi = abs(max(trial_data["rotations"], key=abs))*0.10
        peakX=[]
        peakY=[]
        for dat in peaks[0]:
            if(maxi > dat[1]):
                continue
            peakX.append(dat[0])
            peakY.append(dat[1])
        # for dat in peaks[1]:
        #     if (maxi > abs(dat[1])):
        #         continue
        #     peakX.append(dat[0])
        #     peakY.append(dat[1])
        ax = plt.subplot(gs[trial])
        sns.lineplot(data=trial_data, x="plot_time", y=rtt, color='orange', ax=ax)
        sns.scatterplot(x=peakX, y=peakY,ax=ax)
    plt.subplots_adjust(hspace=.0)
    plt.show()
def apply_target_buffer(cleaned,buffer=6):
    insideTarget = []
    insideUser = []
    insideRectangle = []
    for i, dat in cleaned.iterrows():
        x = buffer
        y = buffer
        insideTarget.append(
            ((dat["Point of Regard Binocular X [px]"] > (dat["target.left"] - x)) and
             (dat["Point of Regard Binocular X [px]"] < (dat["target.right"] + x)) and
             (dat["Point of Regard Binocular Y [px]"] > (dat["target.top"] - y)) and
             (dat["Point of Regard Binocular Y [px]"] < (dat["target.bottom"] + y)))
        )
        insideUser.append(
            ((dat["Point of Regard Binocular X [px]"] > (dat["user.left"] - x)) and
             (dat["Point of Regard Binocular X [px]"] < (dat["user.right"] + x)) and
             (dat["Point of Regard Binocular Y [px]"] > (dat["user.top"] - y)) and
             (dat["Point of Regard Binocular Y [px]"] < (dat["user.bottom"] + y)))
        )
        cx = (dat["tcenter.x"] + dat["ucenter.x"])/2
        cy = (dat["tcenter.y"] + dat["ucenter.y"]) / 2
        d = math.sqrt((dat["tcenter.x"] - dat["ucenter.x"]) * (dat["tcenter.x"] - dat["ucenter.x"]) + (
                    dat["tcenter.y"] - dat["ucenter.y"]) * (dat["tcenter.y"] - dat["ucenter.y"]))
        dist = math.sqrt((dat["Point of Regard Binocular X [px]"] - cx) * (dat["Point of Regard Binocular X [px]"] - cx) + (
                    dat["Point of Regard Binocular Y [px]"] - cy) * (dat["Point of Regard Binocular Y [px]"] - cy))
        insideRectangle.append(dist <= d)
    cleaned['insideTarget'] = insideTarget
    cleaned['insideUser'] = insideUser
    cleaned['insideRectangle'] = insideRectangle
    return cleaned
def plot_video():
    video_name = ""
    cap = cv2.VideoCapture(str(vid))
def prepare_data():
    taskHp1, taskHp2 = create_tasks("hp")
    hpPlotData, hpPlotData2 = multi_plot_prep(taskHp1, taskHp2)
    taskFix1, taskFix2 = create_tasks("fix")
    fixPlotData, fixPlotData2 = multi_plot_prep(taskFix1, taskFix2)
    taskInt1, taskInt2 = create_tasks("int")
    intPlotData, intPlotData2 = multi_plot_prep(taskInt1, taskInt2)
    return intPlotData, intPlotData2,fixPlotData, fixPlotData2, hpPlotData, hpPlotData2