"""
Split data into test and train folders

ignore images without .xml file

This will overwrite any already split images, cuidado
"""
from pathlib import Path
import shutil
import argparse
import random
import pprint

from tqdm import tqdm

FLAGS = None


def overwrite_dir(path: Path):
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        shutil.rmtree(path)
        path.mkdir(parents=True)


def split(data: dict):
    key_list = list(data.keys())

    if FLAGS.shuffle:
        print("shuffle")
        random.shuffle(key_list)

    split = int(FLAGS.split*len(key_list))

    valid = key_list[:split//2]
    test = key_list[split//2:split]
    train = key_list[split:]

    print(f"test={len(test)}")
    print(f"validation={len(valid)}")
    print(f"train={len(train)}")

    if FLAGS.output.exists():
        input(f"about to overwrite: {FLAGS.output}\n Hit ENTER to continue")
    overwrite_dir(FLAGS.output)

    train_dir = FLAGS.output / "train"
    vaild_dir = FLAGS.output / "validation"
    test_dir = FLAGS.output / "test"

    vaild_dir.mkdir()
    train_dir.mkdir()
    test_dir.mkdir()

    for i in tqdm(test):
        shutil.copy2(data[i]['image'], str(test_dir))
        shutil.copy2(data[i]['annot'], str(test_dir))

    for i in tqdm(valid):
        shutil.copy2(data[i]['image'], str(vaild_dir))
        shutil.copy2(data[i]['annot'], str(vaild_dir))

    for i in tqdm(train):
        shutil.copy2(data[i]['image'], str(train_dir))
        shutil.copy2(data[i]['annot'], str(train_dir))
    

# def xml_to_csv(path):
#     xml_list = []
#     for xml_file in path.rglob('*.xml'):
#         tree = ET.parse(xml_file)
#         root = tree.getroot()
#         for member in root.findall('object'):
#             value = (root.find('filename').text,
#                     int(root.find('size')[0].text),
#                     int(root.find('size')[1].text),
#                     member[0].text,
#                     int(member[4][0].text),
#                     int(member[4][1].text),
#                     int(member[4][2].text),
#                     int(member[4][3].text)
#                     )
#             xml_list.append(value)
 
#     column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
#     xml_df = pd.DataFrame(xml_list, columns=column_name)
#     return xml_df


def main():
    print(f"searching: '{FLAGS.input}' for files")

    search_dir = FLAGS.input

    files = [x for x in search_dir.glob('**/*') if x.is_file()]

    data_dict = {}
    for f in files:
        name, ext = f.name.split('.')
        if name not in data_dict.keys():
            data_dict[name] = {'image': None, 'annot': None}
        
        # annotation
        if ext == "xml":
            data_dict[name]['annot'] = f
            
        elif ext == "jpg":
            data_dict[name]['image'] = f

    # get labeled frames only (.xml label file must exist)
    positive_frames = {}
    for item in data_dict:
        if all(v is not None for v in data_dict[item].values()):
            positive_frames.update({item:data_dict[item]})

    split(positive_frames)


if __name__=="__main__":
    parser = argparse.ArgumentParser(
    """
    Split data into validation and train folders

    ignore images without .xml file

    Cuidado, this will overwrite any already split images
    """
    )
    parser.add_argument(
        '-i', '-input',
        dest='input',
        type=Path,
        required=True,
    )
    parser.add_argument(
        '-o', '-output',
        dest='output',
        type=Path,
        default=Path.cwd() / 'split_images',
        help="for talon make this on the scratch drive",
    )
    parser.add_argument(
        '--shuffle',
        action='store_false',
        help='shuffle the data before splitting (default: True)'
    )
    parser.add_argument(
        '--split',
        type=float,
        default=0.2,
        help="percentage of data to be the test set"
    )

    FLAGS = parser.parse_args()

    main()