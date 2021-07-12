FROM tensorflow/tensorflow:latest-gpu

ENV CUDA_DEVICE_ORDER='PCI_BUS_ID'
ENV CUDA_VISIBLE_DEVICES='0'

WORKDIR /tf_odet_root

# install protoc in linux
COPY ./install_protoc.sh /tf_odet_root
RUN /bin/bash install_protoc.sh
RUN pip install --upgrade pip

# setup tensorflow model git repo
RUN git clone https://github.com/tensorflow/models.git
RUN cd models/research/ \
  && protoc object_detection/protos/*.proto --python_out=. \
  && cp object_detection/packages/tf2/setup.py . \
  && python -m pip install . \
  && python object_detection/builders/model_builder_tf2_test.py


CMD ["/bin/bash"]
