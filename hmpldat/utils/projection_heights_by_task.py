"""

Explore correlation between rhead.y and task cross object height

"""
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import hmpldat.file.search as search

DATA = Path("sample_data/merged")

def main():

    info = {}

    for f in search.files(DATA, []): #a_task or all_tasks_from_one_session

        print(f)
        print()
        info[f.name] = {}
        
        df = pd.read_csv(f, low_memory=False)

        # convert to millimeters
        df["Cross.PosY"] = df["Cross.PosY"] * 1000

        height_by_task = df[["Cross.PosY", "RHEAD.Y", "task_name"]].dropna().groupby("task_name")
        
        for task, group in height_by_task:

            # print(group["Cross.PosY"].iloc[0], group["RHEAD.Y"].iloc[0])

            info[f.name][task] = {
                "mean_diff": (group["Cross.PosY"] - group["RHEAD.Y"]).mean(),
                "init_diff": group["Cross.PosY"].iloc[0] - group["RHEAD.Y"].iloc[0],
                "init_height": group["RHEAD.Y"].iloc[0],
                "avg_height": group["RHEAD.Y"].mean(),
                "cross_height": group["Cross.PosY"].iloc[0]
            }

            # print(f"{task}: {info[f.name][task]}")
    
    info_df = pd.concat({k: pd.DataFrame(v) for k, v in info.items()})
    
    info_df = info_df.unstack().T.unstack()
    info_df.to_excel("cross_v_partic_height.xls")

    # graph height vs. cross position for each task & participant
    # for t, row in info_df.iterrows():

    #     data = row.unstack()

    #     x = np.array(data["init_height"]).reshape(-1, 1)
    #     y = np.array(data["cross_height"]).reshape(-1, 1)

    #     reg = LinearRegression().fit(x, y)  # perform linear regression
    #     data["pred_cross_height"] = reg.predict(x)  # make predictions

    #     title_str = t + f"\ncross_height = {reg.coef_[0][0]:.4f} * init_height + {reg.intercept_[0]:.4f};  r^2 = {reg.score(x, y):.6f};"
      
    #     ax = data.plot.scatter("init_height", "cross_height", title=title_str, legend=True)
    #     data.plot.line("init_height", "pred_cross_height", ax=ax, color='r', legend=True)

    #     plt.savefig(t + '.png')


    # merge together tasks that are created the same way
    tasks_1234 = ["fix_1", "bm_1", "hp_1", "pp_1"]
    tasks_567 = ["ap_1", "int_1", "avoid_1"]

    m_data = info_df.loc[tasks_1234]
    h_data = info_df.loc[tasks_567]
    
    plot_this = True

    for t, data in [(tasks_1234, m_data), (tasks_567, h_data)]:
        ts = " ".join(t)

        data = data.stack(0)
        print(data)

        x = np.array(data["init_height"]).reshape(-1, 1)
        y = np.array(data["cross_height"]).reshape(-1, 1)

        reg = LinearRegression().fit(x, y)  # perform linear regression
        data[f"{reg.coef_[0][0]:.4f} * init_height + {reg.intercept_[0]:.4f}"] = reg.predict(x)  # make predictions

        title_str = ts + f"\ncross_height = {reg.coef_[0][0]:.4f} * init_height + {reg.intercept_[0]:.4f};  r^2 = {reg.score(x, y):.6f};"
      
        fig, ax = plt.subplots(figsize=(8,8))

        #grouped
        gs = data.groupby(level=-1)

        for g, c in zip(gs, ['r', 'b', 'g', 'c', 'y', 'm', 'k']):
            print(g, c)
            n, d = g
            ax.scatter(d["init_height"], d["cross_height"], c=c, label=n)

        plt.plot(data["init_height"], data[f"{reg.coef_[0][0]:.4f} * init_height + {reg.intercept_[0]:.4f}"], alpha=0.7, label=f"{reg.coef_[0][0]:.4f} * init_height + {reg.intercept_[0]:.4f}; r^2={reg.score(x, y):.6f};")
        
        if plot_this:
            plot_this = False
            plt.plot(data["init_height"], data["init_height"], alpha=0.7, label="1 * init_height")

        ax.legend(loc='best')
        plt.xlabel('rhead.y')
        plt.ylabel('cross_position.y')
        plt.title("Tasks: " + ts)

        plt.tight_layout()

        plt.savefig("_".join(t) + '.png')


if __name__=="__main__":

    main()