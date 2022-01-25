"""
video_detector.py

Ian Zurutuza
Sept 9, 2019
"""

from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
import sys
import argparse

# if you can't import these add <your/path/to>/models/research to your system path
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

FLAGS = None


def save_to_txt(file_name, classes, scores, boxes, num, category_index, image_shape, frame_num, frame_time):
    im_height, im_width, _ = image_shape

    at_least_one_line = False

    with open(file_name, 'a') as f:

        for i in range(0, int(num)):

            if scores[i] > FLAGS.min_score_thresh:
                at_least_one_line = True
                ymin, xmin, ymax, xmax = boxes[i]

                if classes[i] in category_index.keys():
                    class_name = category_index[classes[i]]['name']
                else:
                    class_name = 'N/A'

                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                            ymin * im_height, ymax * im_height)

                line = ','.join([str(frame_time), str(frame_num), class_name, str(scores[i]), str(left), str(right), str(top), str(bottom)])
                f.write(line + '\n')

        if not at_least_one_line:
            line = ','.join([str(frame_time), str(frame_num), '', '', '', '', '', ''])
            f.write(line + '\n')
          

def main():

    PATH_TO_CKPT = FLAGS.frozen_ckpt_dir / 'frozen_inference_graph.pb'

    # Number of classes the object detector can identify
    # TODO: convert to using count classes script
    NUM_CLASSES = 24

    # Load the label map.
    # Label maps map indices to category names, e.g. 5 == 'grid' 
    label_map = label_map_util.load_labelmap(str(FLAGS.label_map))
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # category_index = label_map_util.create_category_index_from_labelmap(str(FLAGS.label_map), use_display_name=True)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.gfile.GFile(str(PATH_TO_CKPT), 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Open video file
    # TODO: integrate file search methods

    video_path = FLAGS.input_dir
    # for video_path in FLAGS.input_dir.rglob('*.avi'):

    print(video_path.name)
    # print(any(x not in video_path.name for x in ['007', '004']))

    # TODO: remove this??
    # if any(x in video_path.name for x in ['007', '004']):
    #     continue
    # else:
    #     pass

    print(video_path.name)
    video = cv2.VideoCapture(str(video_path))
    out = None

    if FLAGS.type == 'vid' or FLAGS.type == 'both':
        # create video writer
        out = cv2.VideoWriter(
            str(FLAGS.output_dir / ('labeled_' + video_path.name)),
            fourcc=cv2.VideoWriter_fourcc(*"MJPG"),
            fps=video.get(cv2.CAP_PROP_FPS),
            frameSize=(int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )

    csv_filename = FLAGS.output_dir / (video_path.name.split('.')[0] + '.txt')

    if FLAGS.type == 'csv' or FLAGS.type == 'both':
        with open(csv_filename, 'w') as f:
            f.write('frame_time,frame_number,object,score,left,right,top,bottom\n')

    while True:

        frame_num = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        frame_time = video.get(cv2.CAP_PROP_POS_MSEC)

        # Acquire frame
        ret, frame = video.read()
        if frame is None:
            break
                        
        # convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame_expanded = np.expand_dims(rgb_frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        if FLAGS.type == 'csv' or FLAGS.type == 'both':
            save_to_txt(
                csv_filename,
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                np.squeeze(boxes),
                np.squeeze(num),
                category_index,
                frame.shape,
                frame_num,
                frame_time)

        # Draw the results of the detection (aka 'visualize the results')
        if FLAGS.type == 'vid' or FLAGS.type == 'both':
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                min_score_thresh=FLAGS.min_score_thresh)

             # output
            out.write(frame)
           

        if FLAGS.show:
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                min_score_thresh=FLAGS.min_score_thresh)

            cv2.imshow('object detector', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break

    # Clean up
    video.release()
    cv2.destroyAllWindows()

    if out is not None:
        out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-i', 
        dest='input_dir', 
        type=Path,
    )
    parser.add_argument(
        '-o',
        dest='output_dir',
        type=Path,
        default=Path.cwd() / 'output_a'
    )
    parser.add_argument(
        '-t', '--type',
        choices=['vid', 'csv', 'both'],
        default='csv',
    )
    parser.add_argument(
        '-f', '-frozen',
        dest='frozen_ckpt_dir',
        # required=True,
        type=Path,
        default=Path.cwd() / 'sample_data' / 'my_models' / 'working',
    )
    parser.add_argument(
        '-l', '-labelmap',
        dest='label_map',
        # required=True,
        type=Path,
        default=Path.cwd() / 'sample_data' / 'my_models' / 'working' / 'labelmap.pbtxt',
    )
    # parser.add_argument(
    #     '-tf-objdt-dir',
    #     dest='object_dection_dir',
    #     type=Path,
    #     default=Path.home() / 'projects/models/research/object_detection',
    #     help="path to tensor"
    # )
    parser.add_argument(
        '-show',
        action='store_true',
        help="you can't show on talon",
    )
    parser.add_argument(
        '-score',
        dest='min_score_thresh',
        type=float,
        default=0.05,
    )
    parser.add_argument(
        '-p', '-pullframes',
        dest='pull_frames',
        type=bool,
        default=False,
        help="save lower confidence frames to folder in output directory between (90-40%)?",
    )
    # parser.add_argument()

    FLAGS = parser.parse_args()

    try:
        FLAGS.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        s = 'Output folder exists! Files may be \033[91m\033[1mOVERWRITTEN\033[0m continue? [Y/n]'

        if len(list(FLAGS.output_dir.rglob('*'))) != 0:
            if input(s).lower() == 'y':
                FLAGS.output_dir.mkdir(parents=True, exist_ok=True)
            else:
                exit()

    main()