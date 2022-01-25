"""
generate tf-records for test and train sets

TODO: add ability to shard one or both validation and train sets
see: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
"""
from pathlib import Path
import argparse
import io
import pandas as pd
from pprint import pprint
import tensorflow as tf

import pandas as pd
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

FLAGS = None


def gen_labelmap_pbtxt(lmap_dict: dict) -> None:
    """
    generate and save labelmap.pbtxt file for future use.
    """
    with open(FLAGS.output / 'labelmap.pbtxt', 'w') as f:
        for item in lmap_dict:
            out = f"item {{\n  id: {lmap_dict[item]}\n  name: '{item}'\n}}\n\n"
            f.write(out)


def create_labelmap_dict(paths: [Path]) -> dict:
    """
    parse csv(s) input to create a dictionary to convert class labels to integers

    output file -> labelmap.pbtxt
    :returns dict: keys=class_labels -> int
    """
    
    labels_list = []

    for path in paths:
        df = pd.read_csv(path, index_col=None)
        labels_list.append(df)
    
    all_labels = pd.concat(labels_list, axis=0, ignore_index=True)

    unique_labels = all_labels['class'].unique()

    return {k:v for k, v in zip(unique_labels, range(1, len(unique_labels)+1))}


def split(df, group):
    """
    helper function - take csv of annotations and splits the csv by 'filename' column

    :returns: a list named tuples (filename, objects in example)
    """
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(example, path, labelmap):

    with tf.io.gfile.GFile(str(path/example.filename), 'rb') as f:
        encoded_jpg = f.read()

    width = example.object.width.values[0]
    height = example.object.height.values[0]

    filename = example.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    # for each object in example
    for index, row in example.object.iterrows():
        # normalize bounding boxes within 0 and 1.0
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        # encode label
        classes_text.append(row['class'].encode('utf8'))
        classes.append(labelmap[row['class']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main():
    labelmap = create_labelmap_dict(list(FLAGS.input.rglob('*.csv')))
    
    gen_labelmap_pbtxt(labelmap)
    
    split_data = [x for x in FLAGS.input.iterdir() if x.is_dir()]

    # if my data is split
    if len(split_data) > 0:
        for folder in split_data:

            writer = tf.io.TFRecordWriter(str(FLAGS.output / (folder.name + '.record')))
            examples = split(pd.read_csv(FLAGS.input / (folder.name + '_labels.csv')), 'filename')

            for ex in examples:
                tf_example = create_tf_example(ex, folder, labelmap)
                writer.write(tf_example.SerializeToString())

            writer.close()

        print('Successfully created the TFRecords: {}'.format(FLAGS.output))

    # TODO: my data is not split 
    else:
        print("not implemented")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        """
        generate tf-record(s)

        TODO: allow for sharding tf record 
        """
    )
    parser.add_argument(
        '-i', '-input',
        dest='input',
        type=Path,
    )
    parser.add_argument(
        '-o', '-output',
        dest='output',
        type=Path,
        default=Path.cwd() / 'new_tfrecords',
    )
    # parser.add_argument()

    FLAGS = parser.parse_args()

    FLAGS.input = FLAGS.input.resolve()
    FLAGS.output.mkdir(parents=True, exist_ok=False)

    main()