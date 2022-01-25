"""Script to label object center with mouse click

Use mouse wheel to zoom in|out
"""
import argparse
from pathlib import Path
from pprint import pprint
import sys

import cv2
import numpy as np
import pandas as pd


FLAGS = None

# only finds these file extensions
IMG_EXT = ["*.jpg", "*.jpeg", "*.png"]

# global variable to update object center and image frame
u, v = -1, -1
frame = np.zeros((512,512,3), np.uint8)
draw_frame = frame.copy()


def draw_center():
    global u, v, frame, draw_frame

    # clear previous drawing
    draw_frame = frame.copy()  

    # bullseye
    cv2.circle(draw_frame, (u,v), 2, (0,0,255), -1) 
    cv2.circle(draw_frame, (u,v), 15, (0,0,255), 1) 
    cv2.circle(draw_frame, (u,v), 30, (0,0,255), 1)  


# mouse callback 
def label_center(event, x, y, flags, param):
    global u, v, frame           

    if event == cv2.EVENT_LBUTTONDOWN:

        u = x
        v = y

        draw_center()

        cv2.imshow("label center", draw_frame)
        print(f"center = ({u}, {v}) \t hit [SPACE] to confirm")


def main():
    global u, v, frame, draw_frame

    # This dictionary collects info, saved to file on prgm exit
    centers = {}

    # if input is a single image
    if not FLAGS.i.is_dir():
        raise NotImplementedError("if you want to label a single image please move the image into it's own folder")

    # find images file in folder
    images = []
    for ext in IMG_EXT:
        images.extend(FLAGS.i.rglob(ext))

    print(f"found {len(images)} images!")

    # create a named window and link mouse callback func
    cv2.namedWindow("label center")
    cv2.setMouseCallback("label center", label_center)

    # for each image found
    for img in images:

        centers[img.name] = {}

        # read image 
        frame = cv2.imread(str(img))

        # failed to read frame, skip
        if frame is None:
            print(f"failed to open: {img.name} \t\t skipping")
            continue

        draw_frame = frame.copy()

        while True:
            
            cv2.imshow("label center", draw_frame)
            key = cv2.waitKey(0)

            # hit [SPACE] to confirm
            if key == ord(" "):
                # record result to dictionary
                centers[img.name]["u"] = u
                centers[img.name]["v"] = v

                print(f"RECORDING {img.name}\t center = ({u}, {v})")
                
                # next image please
                break 

            # skip, don't record center for this frame
            elif key == ord("g"):
                break

            # move center up one pixel
            elif key == ord("w"):
                v -= 1
                draw_center()

            # move center down one pixel
            elif key == ord("s"):
                v += 1
                draw_center()

            # move center left one pixel
            elif key == ord("a"):
                u -= 1
                draw_center()

            # move center right one pixel
            elif key == ord("d"):
                u += 1
                draw_center()

            # hit q to exit program
            elif key == ord("q"):
                if input("are you sure you want to quit? [y/N]").lower() == "y":
                    pd.DataFrame().from_dict(centers).T.dropna().to_csv(FLAGS.o)
                    cv2.destroyAllWindows()
                    sys.exit(f"labels saved to: {str(FLAGS.o)}")

    # all images in folder labeled
    print("Success! You labeled ALL them images!")

    # close file and exit    
    pd.DataFrame().from_dict(centers).T.to_csv(FLAGS.o)
    cv2.destroyAllWindows()

    sys.exit(f"labels saved to: {str(FLAGS.o)}")

    
if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # input folder of frames to label
    parser.add_argument(
        '-i',
        # default=Path("./sample_data/frames"))
        default=Path("sampled_vid_frames"))

    # output file
    parser.add_argument(
        '-o',
        default=Path.cwd() / "labeled_centers.csv"
    )

    FLAGS, _ = parser.parse_known_args()

    # don't accidentally overwrite data
    if FLAGS.o.exists():
        ask = f"Output file: {str(FLAGS.o)} exists!\nFile will be \033[91m\033[1mOVERWRITTEN\033[0m continue? [Y/n]"
        if input(ask).lower() == "n":
            sys.exit("Rename file or specify output path with -o <PATH> argument")

    main()