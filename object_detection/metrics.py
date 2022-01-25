"""
calculate metrics for object detection

https://github.com/rafaelpadilla/review_object_detection_metrics

detections produced with `object_detection/image_detector.py`
"""

from collections import namedtuple
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd

TRUE_LABELS_PATH = Path("/mnt/hdd_2tb/VMI_data/14oct2019/split_images/validation_labels.csv")
DETECTION_LABELS_PATH = Path("/home/ian/Projects/hmpldat/detection_output/validation.txt")

# important objects
SUBSET = ["cross", "target", "user", "safezone", "grid"]

BBOX = namedtuple('boundingbox', 'left right top bottom')
PREDICTION_TRESHOLD = 0.25

def intersection_over_union(truth, prediction):
    """ 
    
    Args:
        truth: true bounding box (left, right, top, bottom)
        prediction: detected bounding box (left, right, top, bottom)

    Returns:
        float: 
  
    """
    # intersection
    dx = min(prediction.right, truth.right) - max(prediction.left, truth.left)
    dy = min(prediction.bottom, truth.bottom) - max(prediction.top, truth.top)
    # print(dx, dy)
    if (dx >= 0) and (dy >= 0):
        intersection = dx * dy
    else:
        # intersection = 0 
        return 0.0
    # print(intersection)

    # union total area - intersection
    union = (
        (prediction.right - prediction.left) * (prediction.bottom - prediction.top)
        + (truth.right - truth.left) * (truth.bottom - truth.top) 
        - intersection
    )
    # print(union)
   
    return intersection / union


def precision_and_recall(df, threshold):
    """ """

    # object detected, and exists
    # labels match and threshold is satisfied
    tp = ((df["detected_object"] == df['true_object']) 
           & (df['iou'] > threshold)).sum()

    print("tp: ", tp)

    # object exists, but is not detected
    # true label exists, but does not satisfy threshold
    fn = (( df['true_object'].str.contains(".*") > 0 ) & (df['iou'] < threshold)).sum()
    print("fn: ", fn)

    # object detected, but does not exist
    # true label does not exist, but threshold is satisfied
    fp = (( df['true_object'].str.contains(".*") == 0 ) & ( df['iou'] > threshold )).sum()
    print("fp: ", fp)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print(f"pre={precision}, recall={recall}")

    return precision, recall


def tp_fp_fn(df, thresh):

     # object detected, and exists
    # labels match and threshold is satisfied
    tp = ((df["detected_object"] == df['true_object']) 
           & (df['iou'] > threshold)).sum()

    # object exists, but is not detected
    # true label exists, but does not satisfy threshold
    fn = (( df['true_object'].str.contains(".*") > 0 ) & (df['iou'] < threshold)).sum()

    # object detected, but does not exist
    # true label does not exist, but threshold is satisfied
    fp = (( df['true_object'].str.contains(".*") == 0 ) & ( df['iou'] > threshold )).sum()

    return tp, fn, fp


def f1(precision, recall):
    """ """
    return 2 * ((precision * recall) / (precision + recall))

def main():

    # read csv files containing true labels and detections
    truth_df = pd.read_table(TRUE_LABELS_PATH, delimiter=",").sort_values(by="filename").drop(columns=["width", "height"])
    detected_df = pd.read_table(DETECTION_LABELS_PATH, delimiter=",").sort_values(by="filename")
    
    # TODO: columns should have same names when these files are produced originally
    truth_df = truth_df.rename(columns={"class": "object", "xmin": "left", "xmax": "right", "ymin": "top", "ymax": "bottom"})
    truth_df.to_csv('truth_renamed')
    # print(truth_df)
    # print(detected_df)

    # all the files exist
    # print(truth_df.index.difference(detected_df.index))

    # get unique filenames
    filenames = truth_df['filename'].unique()

    # keep track of info for each image frame 
    matched_detections = {}
    unmatched_detections = {}

    for fn in filenames:
        # select data by filename
        truth = truth_df[truth_df['filename'] == fn]
        detected = detected_df[detected_df['filename'] == fn]
        # print(truth)
        # print(detected)

        matched_detections[fn] = {}
        unmatched_detections[fn] = {}

        # if multiple of the same object exist in the same image frame match them by IoU score
        not_evaluated = {x for x in detected.index.to_list()}        

        for i, true_bb in truth.iterrows():
            # print(f"true_bb #{i}, {true_bb['object']}, {true_bb['filename']}")
            potential_matches = []
            
            for j, detected_bb in detected[detected['object'] == true_bb['object']].iterrows():
                not_evaluated.discard(j)
                
                iou = intersection_over_union(true_bb, detected_bb)
                
                potential_matches.append((j, iou))
      
            # sort by iou
            potential_matches.sort(key= lambda x: x[1])

            # if not detected (false negative) append (np.NaN, 0.0)
            if len(potential_matches) >= 1:
                # key is index of truth_df
                matched_detections[fn][i] = {"detected_bb": potential_matches[-1][0], "iou": potential_matches[-1][1], "true_object": true_bb['object']}
            else:
                matched_detections[fn][i] = {"detected_bb": np.NaN, "iou": 0.0, "true_object": true_bb['object']}

            # if true_bb['filename'] == '996.jpg':
            #     print(f"true_bb #{i}, {true_bb['object']}, {true_bb['filename']}")
            #     print(f"detected_bb #{j}, {detected_bb['object']}")
            #     print(f"true:     l={true_bb.left} r={true_bb.right} t={true_bb.top} b={true_bb.bottom}")
            #     print(f"detected: l={detected_bb.left:.0f} r={detected_bb.right:.0f} t={detected_bb.top:.0f} b={detected_bb.bottom:.0f}")
            #     print(f"IoU={iou}")
            #     print(f"{matched_detections[fn][i]}")

        # false positives (not labeled, but detected)
        for i, false_detect in detected.loc[not_evaluated, :].iterrows():
            # print(false_detect)
            unmatched_detections[fn][i] = {"detected_bb": i, "iou": 0.0, "true_object": None}



    # pprint(unmatched_detections)
    # df = pd.concat({k: pd.DataFrame(v) for k, v in matched_detections.items()}).unstack()
    df = pd.DataFrame(matched_detections).stack().apply(pd.Series).reset_index(level=0)
    df.columns = ['true_bb', 'detected_bb', 'iou', 'true_object']

    df_fp = pd.DataFrame(unmatched_detections).stack().apply(pd.Series).reset_index(level=0, drop=True)

    df = pd.concat([df, df_fp]).reset_index()
    df = df.rename(columns={'index': "filename"})

    # df = df.merge(detected_df['score'], left_on=detected_bb, right_index=True)

    # add scores to this dataframe
    # df['confidence'] = df.apply(lambda x: df.loc[x["detected_bb"], "score"])

    confidence = []
    for _, row in df.iterrows():
        if not np.isnan(row['detected_bb']):
            confidence.append((row['detected_bb'], detected_df.loc[row['detected_bb'], "score"], detected_df.loc[row['detected_bb'], "object"]))

    confidence = pd.DataFrame(confidence, columns=['detected_bb', 'confidence', "detected_object"])

    df = df.merge(confidence, how='left', on='detected_bb')

    print(df)
    print()
    # remove instance with low prediction confidence
    df = df[df['confidence'] >= 0.2]

    writer = pd.ExcelWriter("model_accuracy_IoU_0.75.xlsx")

    # now calculate F1
    for iou_thresh in [0.75]:
        
        # defines iou > thresh defines true positive
        # iou_thresh = round(iou_thresh, 2)
        each_class = {}
        for c in set(df['detected_object']):
        # for c in SUBSET:

            single_class = df[((df['detected_object'] == c) | (df['true_object'] == c))]
            single_class = single_class.assign(
                tp=(( df["detected_object"] == df['true_object'] ) & ( df['iou'] > iou_thresh )),
                fn=(( df['true_object'].str.contains(".*") > 0 ) & ( df['iou'] < iou_thresh )),
                fp=(( df["detected_object"] != df['true_object'] ))
            )

            sname = c
            if c == "Ready?":
                sname = "Ready"

            single_class.to_excel(writer, sheet_name=f"{sname}") 

            precision = single_class['tp'].sum() / (single_class['tp'].sum() + single_class['fp'].sum())
            recall = single_class['tp'].sum() / (single_class['tp'].sum() + single_class['fn'].sum())
            af1 = f1(precision, recall)

            print(c)
            print(f"P{iou_thresh}: {precision}")
            print(f"RECALL: {recall}")
            print(f"f1={af1}")
            each_class[c] = {'precision': precision, 'recall': recall, 'f1': af1}


        results_by_class = pd.DataFrame(each_class)
        results_by_class.to_excel(writer, sheet_name="metrics")
        
        # TODO: get average across selected objects
        
        # for k, g in results_by_class.loc['precision'].groupby(by=SUBSET):
        #     print(g)

        # results_by_class.to_excel(writer, sheet_name="avg_metrics")


        print(results_by_class)


    writer.close()








 



if __name__ == "__main__":
    main()