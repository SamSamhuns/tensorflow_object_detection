# Object Detection with Tensorflow

## Instructions

```bash
$ git clone https://github.com/tensorflow/models.git
$ cd models/research/
# install protoc in system and run the cmd
$ protoc object_detection/protos/*.proto --python_out=.
# install TensorFlow Object Detection API.
$ cp object_detection/packages/tf2/setup.py .
$ python -m pip install .
# run model builder test
$ python object_detection/builders/model_builder_tf2_test.py
```
