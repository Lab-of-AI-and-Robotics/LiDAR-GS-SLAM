#!/bin/bash

echo "Building docker"
IMAGE_NAME=diff-surfel-spherical-rast

docker build --build-arg UNAME=$(whoami) --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t ${IMAGE_NAME} .

echo "Done!"
