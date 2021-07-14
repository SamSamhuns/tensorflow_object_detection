import os
import argparse
import os.path as osp

import cv2
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description='Draw bounding boxes on images using the labels csv')
    parser.add_argument("-o",
                        "--inference_output_directory",
                        default="drawn_images",
                        help="Def: drawn_images. Dir where images with detections drawn will be saved to")
    parser.add_argument("-csv",
                        "--csv_label_file",
                        default="data/coco_person/train_labels.csv",
                        help="Def: data/coco_person/train_labels.csv. Path to csv label file")
    parser.add_argument("-i",
                        "--images_dir",
                        default="data/coco_person/images",
                        help="Def: data/coco_person/images. Path to images directory")
    parser.add_argument("-n",
                        "--number_to_draw",
                        default=100,
                        help="Def: 100. Number of images to draw on")
    args = parser.parse_args()
    return args


def draw_on_images(csv_label_file, images_dir, output_dir, num_draw):
    filter_classes = {'person'}
    os.makedirs(output_dir, exist_ok=True)
    csv_df = pd.read_csv(csv_label_file)
    num_draw = min(num_draw, len(csv_df))

    # filename  width  height   class  xmin  ymin  xmax  ymax
    for ridx in range(num_draw):
        class_name = csv_df.loc[ridx, "class"]
        if class_name in filter_classes:
            image_name = csv_df.loc[ridx, "filename"]
            xmin = int(csv_df.loc[ridx, "xmin"])
            ymin = int(csv_df.loc[ridx, "ymin"])
            xmax = int(csv_df.loc[ridx, "xmax"])
            ymax = int(csv_df.loc[ridx, "ymax"])

            image_path = osp.join(images_dir, image_name)
            print(image_path)
            image = cv2.imread(image_path)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
                          (255, 0, 0), thickness=2)
            cv2.imwrite(f"{osp.join(output_dir, image_name)}", image)


def main():
    args = parse_args()
    draw_on_images(csv_label_file=args.csv_label_file,
                   images_dir=args.images_dir,
                   output_dir=args.inference_output_directory,
                   num_draw=args.number_to_draw)

    print(
        f"Inferenced images will be saved in {args.inference_output_directory}")


if __name__ == "__main__":
    main()
