'''
Reference repo:
    https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/blob/master/generate_tfrecord.py
Install the tensorflow object detection first
'''
from collections import namedtuple
import argparse
import logging
import io
import os

from PIL import Image
import pandas as pd
import tensorflow as tf
from object_detection.utils import dataset_util as dutil

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class TFRecord:
    def __init__(self, labelmap_file) -> None:
        f = open(labelmap_file, "r")
        labelmap = f.read()
        self.class_names = self.init_names(labelmap)

    def init_names(self, labelmap) -> dict:
        items = labelmap.split('item')[1:]
        items_dict = {}
        for item in items:
            name = str(item.split('name')[1].split('"')[1])
            name_id = int(item.split('name')[1].split('id')[1].
                          split(": ")[1].split('}')[0])

            items_dict[name] = name_id
        return items_dict

    def class_text_to_int(self, row_label) -> int:
        if self.class_names[row_label] is not None:
            return self.class_names[row_label]
        else:
            None

    def split(self, df, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in
                zip(gb.groups.keys(), gb.groups)]

    def create_tf(self, group, path):
        with tf.io.gfile.GFile(os.path.join(path, '{}'
                                            .format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(self.class_text_to_int(row['class']))

        tf_sample = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dutil.int64_feature(height),
            'image/width': dutil.int64_feature(width),
            'image/filename': dutil.bytes_feature(filename),
            'image/source_id': dutil.bytes_feature(filename),
            'image/encoded': dutil.bytes_feature(encoded_jpg),
            'image/format': dutil.bytes_feature(image_format),
            'image/object/bbox/xmin': dutil.float_list_feature(xmins),
            'image/object/bbox/xmax': dutil.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dutil.float_list_feature(ymins),
            'image/object/bbox/ymax': dutil.float_list_feature(ymaxs),
            'image/object/class/text': dutil.bytes_list_feature(classes_text),
            'image/object/class/label': dutil.int64_list_feature(classes),
        }))
        return tf_sample

    def generate(self, output_path, image_dir, csv_input) -> None:
        writer = tf.io.TFRecordWriter(output_path)
        path = os.path.join(image_dir)
        data = pd.read_csv(csv_input)
        grouped = self.split(data, 'filename')

        for group in grouped:
            try:
                tf_sample = self.create_tf(group, path)
                writer.write(tf_sample.SerializeToString())
            except Exception:
                continue
        logging.info(
            'Successfully created the TFRecords: {}'.format(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tf record")
    parser.add_argument('-l', '--labelmap_file',
                        help='labelmap path')
    parser.add_argument('-o', '--output_record_path',
                        help='output record path')
    parser.add_argument('-i', '--image_dir',
                        help='image directory')
    parser.add_argument('-csv', '--csv_input',
                        default='train_labels.csv',
                        help='CSV file with images names')
    args = parser.parse_args()

    tf_record = TFRecord(args.labelmap_file)
    tf_record.generate(args.output_record_path, args.image_dir, args.csv_input)
