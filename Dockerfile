FROM nvcr.io/nvidia/pytorch:25.06-py3

ARG MAX_JOBS=32
ENV MAX_JOBS=${MAX_JOBS}
ARG NVCC_THREADS=2
ENV NVCC_THREADS=${NVCC_THREADS}
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV NODE_OPTIONS=""
ENV PIP_ROOT_USER_ACTION=ignore

ARG TORCH_CUDA_ARCH_LIST="9.0a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG VLLM_FA_CMAKE_GPU_ARCHES="90a-real"
ENV VLLM_FA_CMAKE_GPU_ARCHES=${VLLM_FA_CMAKE_GPU_ARCHES}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt upgrade -y
RUN apt install -y --no-install-recommends \
    curl \
    gcc-12 g++-12 \
    git \
    libibverbs-dev \
    libjpeg-turbo8-dev \
    libpng-dev \
    zlib1g-dev

# Set compiler paths
ENV CC=/usr/bin/gcc-12
ENV CXX=/usr/bin/g++-12

RUN sed -i '/^torch==/d; /^pytorch-triton==/d; /^torchvision==/d; /^sympy==/d; /^packaging==/d; /^setuptools==/d; /^build==/d; /^cmake==/d; /^ninja==/d; /^pybind11==/d; /^wheel==/d' /etc/pip/constraint.txt && echo 'numpy==1.26.4' >> /etc/pip/constraint.txt
# get rid of the nvidia fork of pytorch
RUN pip uninstall -y torch torchvision torchaudio pytorch-triton triton
RUN pip install -U torch torchvision torchaudio triton pytorch-triton --index-url https://download.pytorch.org/whl/cu128

RUN pip install pynvml

RUN pip install accelerate==1.9.0 hf_transfer modelscope bitsandbytes timm boto3 runai-model-streamer runai-model-streamer[s3] tensorizer "transformers==4.52.4" nltk qwen_vl_utils radgraph codetiming datasets dill hydra-core pandas peft "pyarrow>=15.0.0" pybind11 pylatexenc torchdata wandb scikit-image==0.25.2 ensemble-boxes==1.0.9 torchxrayvision==1.3.5 pydicom==3.0.1 sentencepiece==0.2.0 faster-coco-eval==1.6.7 rouge-score==0.1.2 bert-score==0.3.13 radgraph==0.1.18 f1chexbert==0.0.2 torchmetrics==1.8.0 albumentations==2.0.8  sentence-transformers==5.0.0 numba==0.59.1 llvmlite==0.42.0 grpcio==1.62.1 protobuf==4.24.4 scikit_image==0.25.2 torchxrayvision==1.3.5 rouge_score==0.1.2 bert_score==0.3.13 f1chexbert==0.0.2

COPY local-wheels /opt/local-wheels

RUN pip install --no-cache-dir /opt/local-wheels/*

# RUN git clone https://github.com/volcengine/verl.git && \
#     cd verl && \
#     git checkout 8d9e350ea58c7ad4b50dd14d9dcb50577242c55f && \
#     pip install accelerate codetiming datasets \
#     dill hydra-core pandas peft "pyarrow>=15.0.0" pybind11 pylatexenc torchdata wandb && \
#     sed -i '/# Dependencies corresponding to install_requires in setup.py/,/\]/d' pyproject.toml && \
#     pip install -e .
