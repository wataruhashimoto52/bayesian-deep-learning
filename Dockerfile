FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
ARG WITH_TORCHVISION=1
USER root

ENV TZ='Asia/Tokyo'

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libfreetype6-dev \
         libhdf5-serial-dev \
         libzmq3-dev \
         pkg-config \
         software-properties-common \
         unzip \
         wget \
         vim \
         less \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

ARG PYTHON=python3.7

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y ${PYTHON}

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN ${PYTHON} get-pip.py
RUN ln -sf /usr/bin/${PYTHON} /usr/local/bin/python3
RUN ln -sf /usr/local/bin/pip /usr/local/bin/pip3

RUN pip3 --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt
RUN git clone https://github.com/cybertronai/pytorch-sso.git
WORKDIR /pytorch-sso
RUN python3.7 setup.py install
WORKDIR /
COPY . /app
WORKDIR /app
