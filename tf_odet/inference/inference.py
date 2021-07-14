# must be imported at the top for setting env vars
import tf_odet.set_tf_env_vars
import os
import yaml
import argparse
import os.path as osp

import numpy as np
import pandas as pd
from PIL import Image
from six import BytesIO

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import tensorflow as tf

# limit gpu memory growth to required amount only
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export trained tf2 checkpoints in savedmodel fmt using tf2 object detection api')
    parser.add_argument("-c",
                        "--model_config_yaml",
                        default="model_store/mobilenet_v2_train.yaml",
                        help='Def: "model_store/mobilenet_v2_train.yaml". YAML file path with training configs')
    parser.add_argument("-m",
                        "--saved_model_dir",
                        default="inference_graph/saved_model",
                        help="Def: inference_graph/saved_model. Dir containing the trained saved_model files")
    parser.add_argument("-o",
                        "--inference_output_directory",
                        default="detection_result",
                        help="Def: detection_result. Dir where images with detections drawn will be saved to")
    parser.add_argument("-tcsv",
                        "--test_csv_label_path",
                        default="data/coco_person/test_labels.csv",
                        help="Def: data/coco_person/test_labels.csv. Path to test csv label file")
    parser.add_argument("-i",
                        "--images_dir",
                        default="data/coco_person/images",
                        help="Def: data/coco_person/images. Path to images directory")
    args = parser.parse_args()

    return args


def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = \
        output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def load_model_and_cat_idx(labelmap_path, saved_model_output_dir="inference_graph/saved_model"):

    category_index = label_map_util.create_category_index_from_labelmap(
        labelmap_path, use_display_name=True)
    tf.keras.backend.clear_session()
    model = tf.saved_model.load(saved_model_output_dir)
    return model, category_index


def inference_on_images(model, category_index, test_csv_label, images_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    test = pd.read_csv(test_csv_label)
    #  getting 3 random images to test
    images = list(test.sample(n=3)['filename'])

    for image_name in images:
        image_np = load_image_into_numpy_array(
            osp.join(images_dir, image_name))
        output_dict = run_inference_for_single_image(model, image_np)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get(
                'detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        Image.fromarray(image_np).save(f"{osp.join(output_dir, image_name)}")


def main():
    args = parse_args()

    # load config files for training
    with open(args.model_config_yaml) as file:
        config_dict = yaml.full_load(file)

    model, category_index = load_model_and_cat_idx(labelmap_path=config_dict["labelmap_path"],
                                                   saved_model_output_dir=args.saved_model_dir)

    inference_on_images(model=model, category_index=category_index,
                        test_csv_label=args.test_csv_label_path,
                        images_dir=args.images_dir,
                        output_dir=args.inference_output_directory)

    print(
        f"Inferenced images will be saved in {args.inference_output_directory}")


if __name__ == "__main__":
    main()
