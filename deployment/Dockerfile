# syntax=docker/dockerfile:experimental

# The above shebang is used by `--mount=type=ssh`.

# This file is derived from
# https://github.com/sbx126/Mask3D/blob/docker_debug/docker/Dockerfile.
# It installs the prerequistes of Mask3D tbd.

FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive 
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# First update the list of available packages and their versions.
# Then upgrade the current packages. Say yes to all prompts.
# After install all target packages.
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --fix-missing --no-install-recommends \
    curl \
    build-essential \
    cmake \
    git \
    ca-certificates \
    libjpeg-dev \
    # vim \
    gcc \
    make \
    # htop \
    python \
    wget \
    apt-utils \
    zip \
    unzip \
    python-numpy \
    python3 \
    python3-cups \
    python3-numpy \
    gzip \
    python3-pip \
    python3-dev \
    python3-gi \
    python3-setuptools \
    python3-wheel \
    sudo \
    ninja-build \
    libopenblas-dev \
    xterm \
    xauth \
    openssh-server \
    openssh-client \
    tmux \
    mate-desktop-environment-core \
    libpng-dev && \
    apt-get purge -y --auto-remove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install conda with Python 3.10
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH

COPY deployment/requirements_inference.txt /tmp/requirements.txt

# packages
RUN conda update -n base -c defaults conda && \
    conda install openblas-devel -c anaconda && \
    pip3 install --upgrade pip==23.3.2 && \
    pip3 install setuptools && \
    pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 && \
    pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu116.html && \
    pip3 install "cython<3.0.0" "numpy<1.25.0" && \
    pip3 install --no-build-isolation "pycocotools==2.0.4" && \
    pip3 install --no-build-isolation "pyyaml==5.4.1" && \
    pip3 install -r /tmp/requirements.txt && \
    pip3 install protobuf==3.20 wandb==0.16.6
    # pip3 install google-cloud-storage fastapi && \


    # pip3 install torchmetrics==0.11.4

    # pip3 install pytorch-lightning==1.7.2 fire==0.4.0 imageio==2.21.1 tqdm wandb python-dotenv==0.20.0 pyviz3d==0.2.28 scipy==1.9.0 plyfile==0.7.4 scikit-learn==1.1.2 trimesh==3.14.0 loguru==0.6.0 albumentations==1.2.1 volumentations==0.1.8 && \
    # # pip3 install open3d && \
    # pip3 install ninja==1.10.2.3 && \
    # pip3 install antlr4-python3-runtime==4.8 && \
    # pip3 install black==21.4b2 && \
    # pip3 install omegaconf==2.0.6 hydra-core==1.0.5 && \
    # pip3 install torchmetrics==0.11.4 && \
    # pip3 install pillow==9.5.0

# packages
# RUN conda update -n base -c defaults conda && \
#     conda install openblas-devel -c anaconda && \
#     pip3 install --upgrade pip==23.3.2 && \
#     pip3 install setuptools && \
#     pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 && \
#     pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu116.html && \
#     pip3 install "cython<3.0.0" "numpy==1.24.2" \
#     && pip3 install --no-build-isolation "pycocotools==2.0.4" \
#     && pip3 install --no-build-isolation "pyyaml==5.4.1" && \
#     pip3 install pytorch-lightning==1.7.2 fire==0.4.0 imageio==2.21.1 tqdm wandb python-dotenv==0.20.0 pyviz3d==0.2.28 scipy==1.9.0 plyfile==0.7.4 scikit-learn==1.1.2 trimesh==3.14.0 loguru==0.6.0 albumentations==1.2.1 volumentations==0.1.8 && \
#     # pip3 install open3d && \
#     pip3 install ninja==1.10.2.3 && \
#     pip3 install antlr4-python3-runtime==4.8 && \
#     pip3 install black==21.4b2 && \
#     pip3 install omegaconf==2.0.6 hydra-core==1.0.5 && \
#     pip3 install torchmetrics==0.11.4 && \
#     pip3 install pillow==9.5.0

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"

# Copy all relevant files to run inference
WORKDIR /Mask3D
COPY benchmark benchmark
COPY conf conf
COPY datasets datasets
COPY models models
COPY trainer trainer
COPY third_party third_party
COPY utils utils
COPY inference.py inference.py

# Install detectron2
RUN pip3 install natsort fvcore==0.1.5.post20220512 cloudpickle==2.1.0 pycocotools==2.0.4 && \
    pip3 install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

WORKDIR /Mask3D/third_party

# Install MinkowskiEngine
RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine" && \
    cd MinkowskiEngine && \
    git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 && \
    python setup.py install --force_cuda --blas=openblas && \
    cd .. && \
    rm -rf MinkowskiEngine

# Install ScanNet
RUN git clone https://github.com/ScanNet/ScanNet.git && \
    cd ScanNet/Segmentator && \
    git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2 && \
    make && \
    cd ../.. && \
    rm -rf ScanNet

# Install pointnet2
RUN cd pointnet2 && python setup.py install

WORKDIR /Mask3D

# Make port 8081 available to the world outside this container
# EXPOSE 8081
