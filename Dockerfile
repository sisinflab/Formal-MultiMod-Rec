FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y wget unzip git curl software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.11 python3.11-distutils && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py && \
    apt-get install -y python3.11-dev && \
    pip install --upgrade pip && \
    git clone https://github.com/sisinflab/Formal-Multimod-Rec.git && \
    pip install -r Formal-Multimod-Rec/requirements.txt && \
    pip install -r Formal-Multimod-Rec/requirements_torch.txt && \
    pip install -r Formal-Multimod-Rec/requirements_torch_geometric.txt

WORKDIR Formal-Multimod-Rec