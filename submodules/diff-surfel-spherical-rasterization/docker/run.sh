if [ -n "$1" ]; then
  echo "Source dir: $1"
else
  echo "usage ./docker/.run.sh <SOURCE_DIR>"
fi

IMAGE_NAME=diff-surfel-spherical-rast

xhost +

docker run --gpus 'all,"capabilities=compute,utility,graphics"' \
  -ti \
  -it \
  --rm \
  --env="DISPLAY" \
  --shm-size 24G \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --privileged \
  --network host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v "$1:/workspace/repository/" \
  ${IMAGE_NAME} \
  bash -c /bin/bash
