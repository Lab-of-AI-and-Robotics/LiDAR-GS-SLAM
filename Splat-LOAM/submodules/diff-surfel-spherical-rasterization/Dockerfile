FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel
ARG UNAME=user
ARG UID=1000
ARG GID=1000

WORKDIR /workspace

RUN groupadd -g $GID $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME


USER $UNAME

