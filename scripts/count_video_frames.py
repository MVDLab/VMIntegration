"""Physically count the number of frames

"""

import argparse
from pathlib import Path

import cv2
import pandas as pd

import hmpldat.file.search as search
import hmpldat.utils as utils


SEARCH_FOR = [".avi"]
"""strings to identify video"""

FLAGS = None


def main():

    videos = search.files(FLAGS.data_path, [FLAGS.participant] + SEARCH_FOR, [])
    print(videos)

    for v in videos:

        video = cv2.VideoCapture(str(v))
        frame_count_cv = video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = video.get(cv2.CAP_PROP_FPS)
        fourcc = video.get(cv2.CAP_PROP_FOURCC)
        frame_size = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f'video properties: {v.name}')
        print(f'\tfps:{fps}')
        print(f'\test. frame count:{frame_count_cv}')
        print(f'\tframe size:{frame_size}')

        count = 1

        while True:
            frame_num = video.get(cv2.CAP_PROP_POS_FRAMES)
            # frame_time = vid.get(cv2.CAP_PROP_POS_MSEC)
            ret, frame = video.read()

            if not ret:
                break

            count += 1

            # msg = f"time={frame_time}      num={frame_num}"

            # if ret:
            #     if frame_num % 10000 == 0:
            #         print(msg)

            #     last = frame.copy()
            #     lframe_time = frame_time
            #     lmsg = msg

            # else:      

            #     print("last")
            #     print(lmsg)    
            #     print(pd.to_timedelta(lframe_time, "ms"))

            #     cv2.putText(
            #         last, lmsg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            #     )

            #     cv2.imshow("test", last)
            #     k = cv2.waitKey()

        print(f'true frame count: {count}')
        print(f'true frame count: {frame_num + 1}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-p',
        '--participant',
    )

    parser.add_argument(
        '-d',
        '--data_path',
        help='path to folder of videos',
        default=Path('/home/irz0002/Documents/projects/HMP/Projects')
    )

    FLAGS, _ = parser.parse_known_args() 
    
    main()
