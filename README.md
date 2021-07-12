# Object Detection with Tensorflow

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

```bash
$ export dataset=DATASET_DIR_NAME
$ python tf_odet/data/generate_tf_records.py -l $dataset/labelmap.pbtxt -o $dataset/train.record -i images -csv $dataset/train_labels.csv
$ python tf_odet/data/generate_tf_records.py -l $dataset/labelmap.pbtxt -o $dataset/test.record -i images -csv $dataset/test_labels.csv
```

## 3. Model Setup

### a. Choose and download model from Tesorflow Model Zoo

**Download model from the [Tensorflow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). Choosing ssd_mobilenet_v2_320x320_coco17_tpu in this case**

```bash
$ wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
$ tar -xvf ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
$ rm ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
```

### b. Choose corresponding config file for TensorFlow Model

**Download corresponding config file for model from [Tensorflow Model Config](https://github.com/tensorflow/models/tree/master/research/object_detection/configs/tf2)**

```bash
$ wget <https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/ssd_mobilenet_v2_320x320_coco17_tpu-8.config>
```

## 4. Model Training

## 5. Model Validation

## 6. Model Export

### Acknowledgements

-   [Custom object detection in the browser using TensorFlow.js by Hugo Zanini](https://blog.tensorflow.org/2021/01/custom-object-detection-in-browser.html)
