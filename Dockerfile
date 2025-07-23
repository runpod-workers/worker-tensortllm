# Start with NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python
RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev git libopenmpi-dev cmake build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone TensorRT-LLM repository first
RUN git clone https://github.com/NVIDIA/TensorRT-LLM.git /app/TensorRT-LLM

# Copy requirements file
COPY builder/requirements.txt /app/requirements.txt

# Install Python dependencies
# Install runpod and other dependencies first, then install tensorrt_llm from the cloned repo
RUN pip3 install --upgrade pip && \
    pip3 install runpod~=1.7.13 transformers fastapi uvicorn pydantic numpy torch huggingface-hub python-dotenv && \
    cd /app/TensorRT-LLM && \
    pip3 install -e .

# Copy the src directory containing handler.py
COPY src /app/src

# Copy test_input.json
COPY test_input.json /app/

# Command to run the serverless worker
CMD ["python3", "/app/src/handler.py"]