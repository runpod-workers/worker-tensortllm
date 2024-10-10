# Start with Runpod Python 3.10 base image
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python
RUN apt-get update -y && \
    apt-get install -y git libopenmpi-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Update Torch
RUN pip install torch==2.4.0 torchvision==0.19.0

# Install TensorRT-LLM
RUN pip install tensorrt_llm==0.13.0 -U --extra-index-url https://pypi.nvidia.com

