FROM nvcr.io/nvidia/pytorch:22.03-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y libsndfile1 ffmpeg

RUN pip install Cython

RUN pip install nemo_toolkit[all]

