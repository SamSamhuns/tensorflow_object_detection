FROM tensorflow/tensorflow:latest-gpu

ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_DEVICE_ORDER='PCI_BUS_ID'
ENV CUDA_VISIBLE_DEVICES='0'

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libgl1-mesa-glx \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget

# Install gcloud and gsutil commands
# https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu
RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y

# Add new user to avoid running as root
RUN useradd -ms /bin/bash tensorflow
USER tensorflow
WORKDIR /home/tensorflow

# Copy this version of of the model garden into the image
COPY --chown=tensorflow models /home/tensorflow/models

# Compile protobuf configs
RUN (cd /home/tensorflow/models/research/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR /home/tensorflow/models/research/

RUN cp object_detection/packages/tf2/setup.py ./
ENV PATH="/home/tensorflow/.local/bin:${PATH}"

RUN pip install --upgrade pip==20.2
RUN python -m pip install .

ENV TF_CPP_MIN_LOG_LEVEL 3

# Add models to our python path
ENV PYTHONPATH="/home/tf/models:$PYTHONPATH"

# end of official repo tf
WORKDIR /home/tensorflow/
# copy required files
RUN mkdir -p /home/tensorflow/model_store
RUN mkdir -p /home/tensorflow/tf_odet

COPY --chown=tensorflow model_store /home/tensorflow/model_store
COPY --chown=tensorflow tf_odet /home/tensorflow/tf_odet

CMD ["python3 -m tf_odet.train.train"]
