FROM ubuntu:20.04
WORKDIR /app
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get -y install python3.8 python3-pip && \
    apt-get install -y curl wget unzip git ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*
COPY setup.sh .
RUN pip install jupyterlab ipywidgets textattack
RUN chmod +x setup.sh && \
    pip config set global.no-cache-dir false && \
    ./setup.sh