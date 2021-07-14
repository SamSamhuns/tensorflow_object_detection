import os
import glob
import tqdm
import shutil
import argparse
import os.path as osp
import xml.etree.ElementTree as ET


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pascal VOC to tfrecord generation CSV fmt conversion for object detection training')
    parser.add_argument("-r",
                        "--root_voc",
                        default="VOCdevkit/VOC2007",
                        help='Def: VOCdevkit/VOC2007. Root folder containing VOC Annotations, ImageSets, and JPEGImages')
    parser.add_argument('-l',
                        '--class_label_file',
                        default='VOCdevkit/classes.txt',
                        help='Def: VOCdevkit/classes.txt. Txt file containing class names in newlines')
    parser.add_argument('-e',
                        '--src_img_ext_list',
                        nargs='+',
                        default=["jpg", "jpeg"],
                        help='Def: jpg jpeg. List extension of source images. i.e. -e jpg JPEG png PNG')
    parser.add_argument('-csv',
                        '--target_csv_file',
                        default="train_labels.csv",
                        help='Def: train_labels.csv. CSV file with headers filename,width,height,class,xmin,ymin,xmax,ymax')
    parser.add_argument('-cdir',
                        '--copy_img_annot_dir',
                        help='Def: None. If -cdir DIR_PATH is used, matching class annots and iamges are copied to this dir')
    args = parser.parse_args()
    return args


def get_image_list_from_dir(image_dir, valid_extn=["jpg", "jpeg"]):
    image_list = []
    for ext in valid_extn:
        image_list.extend(glob.glob(osp.join(image_dir, f"*.{ext}")))
    return image_list


def get_class_name_idx_dict(class_label_file):
    """
    get a name ot index dict of classes from a txt file with class names in newlines
    """
    with open(class_label_file, 'r') as class_file:
        class_list = class_file.readlines()
        class_name_idx_dict = {
            cname.strip(): i for i, cname in enumerate(class_list)}
    return class_name_idx_dict


def add_csv_label_from_img_annot(src_image_path, src_annot_path, class_dict, target_csv_fptr, status_dict):
    """
    CSV file will have headers: filename,width,height,class,xmin,ymin,xmax,ymax
    Each line contains annotation for one bounding box in an image
    """
    src_annot_file = open(src_annot_path, 'r')
    base_fname = osp.basename(src_image_path)

    tree = ET.parse(src_annot_file)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    for obj in root.iter('object'):  # foreach labelled object in an image
        difficult = obj.find('difficult').text
        class_name = obj.find('name').text
        if class_name not in class_dict or int(difficult) == 1:
            continue
        xmlbox = obj.find('bndbox')
        xmin, ymin = int(xmlbox.find('xmin').text), int(
            xmlbox.find('ymin').text)
        xmax, ymax = int(xmlbox.find('xmax').text), int(
            xmlbox.find('ymax').text)
        target_csv_fptr.write(
            f"{base_fname},{width},{height},{class_name},{xmin},{ymin},{xmax},{ymax}\n")
        status_dict["class_found"] = True


def generate_csv_from_voc(root_voc_dir, valid_extn, class_label_file, target_csv_file, copy_img_annot_dir):

    if copy_img_annot_dir is not None:
        copy_img_dir = osp.join(copy_img_annot_dir, "images")
        copy_annot_dir = osp.join(copy_img_annot_dir, "annotations")
        os.makedirs(copy_img_dir, exist_ok=True)
        os.makedirs(copy_annot_dir, exist_ok=True)

    image_dir = osp.join(root_voc_dir, "JPEGImages")
    annot_dir = osp.join(root_voc_dir, "Annotations")

    class_dict = get_class_name_idx_dict(class_label_file)
    image_paths = get_image_list_from_dir(image_dir, valid_extn=valid_extn)
    target_csv_fptr = open(target_csv_file, 'w')
    target_csv_fptr.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")  # write headers
    status_dict = {}  # track if classes in class_label_file were foundin the annots

    for src_image_path in tqdm.tqdm(image_paths):
        src_annot_path = osp.join(annot_dir, osp.splitext(
            osp.basename(src_image_path))[0] + ".xml")
        try:
            status_dict["class_found"] = False
            add_csv_label_from_img_annot(src_image_path,
                                         src_annot_path,
                                         class_dict=class_dict,
                                         target_csv_fptr=target_csv_fptr,
                                         status_dict=status_dict)
            if copy_img_annot_dir is not None and status_dict["class_found"]:
                shutil.copy(src_image_path, osp.join(
                    copy_img_dir, osp.basename(src_image_path)))
                shutil.copy(src_annot_path, osp.join(
                    copy_annot_dir, osp.basename(src_annot_path)))
        except Exception as e:
            print(f"{e}. Skipping conv for {src_annot_path}")

    target_csv_fptr.close()
    print(f"Finished processing images in {image_dir}")
    print(f"CSV label file stored in {target_csv_file}")


def main():
    args = parse_args()
    generate_csv_from_voc(root_voc_dir=args.root_voc,
                          valid_extn=args.src_img_ext_list,
                          class_label_file=args.class_label_file,
                          target_csv_file=args.target_csv_file,
                          copy_img_annot_dir=args.copy_img_annot_dir)


if __name__ == "__main__":
    main()
