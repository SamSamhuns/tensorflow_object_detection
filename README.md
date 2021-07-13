# Object Detection with Tensorflow

Note: Using a virtual env with python 3.6

## 1. Repository Setup

```bash
$ git clone https://github.com/tensorflow/models.git
# set up a virtual env / conda env
$ python -m venv venv_tf; source venv_tf/bin/activate
$ cd models/research/
# install protoc in system with sudo bash install_protoc.sh and run the cmd
$ protoc object_detection/protos/*.proto --python_out=.
# install TensorFlow Object Detection API.
$ cp object_detection/packages/tf2/setup.py .
$ python -m pip install .
# run model builder test
$ python object_detection/builders/model_builder_tf2_test.py
```

## 2. Data Setup

### a. Create a `labelmap.pbtxt` file with class id and name and place it inside the dataset directory:

`labelmap.pbtxt` file contents:

        item {
          id: 1
          name: 'object1'
        }

        item {
          id: 2
          name: 'object2'
        }

        item {
          id: 3
          name: 'object3'
        }
        ...

### b. Convert dataset to tfrecords:

Example on the `coco_person` dataset. This dataset was created using `tf_odet/data/voc_to_label_csv.py` from the original coco dataset filtered for only the `person` class.

```bash
$ export dataset=coco_person
$ python tf_odet/data/generate_tf_records.py -l $dataset/labelmap.pbtxt -o $dataset/train.record -i $dataset/images -csv $dataset/train_labels.csv
$ python tf_odet/data/generate_tf_records.py -l $dataset/labelmap.pbtxt -o $dataset/test.record -i $dataset/images -csv $dataset/test_labels.csv
```

## 3. Model Setup

### a. Choose and download model from Tesorflow Model Zoo

**Download model from the [Tensorflow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). Choosing ssd_mobilenet_v2_320x320_coco17_tpu in this case**

```bash
$ mkdir model_store; cd model_store
$ wget http://download.tensorflow.org/models/object_detection/classification/tf2/20200710/mobilenet_v2.tar.gz
$ tar -xvf mobilenet_v2.tar.gz
$ rm mobilenet_v2.tar.gz
```

### b. Choose corresponding config file for TensorFlow Model

**Download corresponding config file for model from [Tensorflow Model Config](https://github.com/tensorflow/models/tree/master/research/object_detection/configs/tf2)**

```bash
$ wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/ssd_mobilenet_v2_320x320_coco17_tpu-8.config
$ mv ssd_mobilenet_v2_320x320_coco17_tpu-8.config mobilenet_v2.config
```

## 4. Model Training

## 5. Model Validation

## 6. Model Export

### Acknowledgements

-   [Custom object detection in the browser using TensorFlow.js by Hugo Zanini](https://blog.tensorflow.org/2021/01/custom-object-detection-in-browser.html)
