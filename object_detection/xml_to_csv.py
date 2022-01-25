"""
Helper function to generate a csv file from labeled images (.xml files) 
for each folder of images in input directory

handles input directory like: (filenames are not important)
    |---labeled_images
        |---validation
        |   |---0.xml
        |   |---0.jpg
        |   |--- ...
        |   |---XXX.xml
        |   |---XXX.jpg
        |
        |---train
            |---1.xml
            |---1.jpg
            |--- ...
            |---YYY.xml
            |---YYY.jpg

or a single input directory of image labels
"""
from pathlib import Path
import argparse

import pandas as pd
import xml.etree.ElementTree as ET

FLAGS = None


def xml_to_csv(path):
    xml_list = []
    for xml_file in path.rglob('*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    member[0].text,
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text)
                    )
            xml_list.append(value)
 
    column_name = ['filename', 'width', 'height', 'object', 'left', 'top', 'right', 'bottom']
    xml_df = pd.DataFrame(xml_list, columns=column_name)

    return xml_df


def annotation_info(xml_df: pd.DataFrame) -> None:
    """
    helper function - returns counts of objects in labeled images
    """
    info = xml_df['object'].value_counts().to_frame(name='count')

    ### find a which image has a string you mistyped
    print(xml_df[xml_df['object'].str.match('face')])

    print(info.to_string())


def main():
    # does input folder contain: (images + labels) or directories?
    labels = True if len(list(FLAGS.input.glob('*.xml'))) > 0 else False

    if labels:
        print(f'\ngerating csv from .xml files in: {FLAGS.input}')
        xml_df = xml_to_csv(FLAGS.input)
        annotation_info(xml_df)
        xml_df.to_csv(FLAGS.input.parent / (str(FLAGS.input.name) + '_labels.csv'), index=None)
        print('Successfully converted xmls to csv.')
    else:

        for folder in FLAGS.input.iterdir():
            if folder.is_dir():
                print(f'\ngerating csv from .xml files in: {folder}')
                xml_df = xml_to_csv(folder)
                annotation_info(xml_df)        
                xml_df.to_csv(FLAGS.input / (folder.name + '_labels.csv'), index=None)
                print('Successfully converted xmls to csv.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        '''
        generate single csv from xml label files

        will handle images and label files to be split into test & training sets
        e.g. (refer to: split_data.py)
            |---labeled_images
                |---test
                |   |---0.xml
                |   |---0.jpg
                |   |--- ...
                |   |---XXX.xml
                |   |---XXX.jpg
                |
                |---train
                    |---1.xml
                    |---1.jpg
                    |--- ...
                    |---YYY.xml
                    |---YYY.jpg

        or single directories of images 
        '''
    )
    parser.add_argument(
        '-i', '-input',
        dest='input',
        type=Path,
        help='input directory'
    )

    FLAGS = parser.parse_args()

    # resolve path (get absolute path)
    FLAGS.input = FLAGS.input.resolve()
    
    main()
