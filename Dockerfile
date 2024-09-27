# Start with NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python
RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev git libopenmpi-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone TensorRT-LLM repository
RUN git clone https://github.com/NVIDIA/TensorRT-LLM.git /app/TensorRT-LLM

# Set working directory
WORKDIR /app/TensorRT-LLM/examples/llm-api

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Install additional dependencies for the serverless worker
RUN pip3 install --upgrade runpod transformers

# Set the working directory to /app
WORKDIR /app

# Copy the src directory containing handler.py
COPY src /app/src

# Command to run the serverless worker
CMD ["python3", "/app/src/handler.py"]