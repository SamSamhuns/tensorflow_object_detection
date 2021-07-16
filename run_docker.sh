#!/bin/bash

# check for 4 cmd args
if [ $# -ne 2 ]
  then
    echo "HTTP port must be specified for tensorboard."
		echo "eg. \$ bash run_docker.sh -h 8080"
		exit
fi

# get the http & grpc ports
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--http) http="$2"; shift ;;
        *) echo "Unknown parameter passed: $1";
	exit 1 ;;
    esac
    shift
done

echo "Running docker with exposed tensorboard http port: $http"
# Z flag in shared volume ensures, docker container can access host dirs
docker run -ti --rm \
      --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
      --gpus device="3" \
      --name tf2_odet \
      -v $PWD/data:/home/tensorflow/data:z \
      -p $http:8080 \
      tf2_object_detection:latest \
      bash
