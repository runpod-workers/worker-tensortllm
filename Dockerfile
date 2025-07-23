# Start with NVIDIA CUDA development image (includes nvcc)
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

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
WORKDIR /app

# Copy requirements.txt from builder directory
COPY builder/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Install additional dependencies for TensorRT-LLM
WORKDIR /app/TensorRT-LLM/examples/llm-api
RUN if [ -f requirements.txt ]; then pip3 install -r requirements.txt; fi

# Set the working directory back to /app
WORKDIR /app

# Copy the src directory containing handler.py
COPY src /app/src

# Copy test_input.json
COPY test_input.json /app/

# Command to run the serverless worker
CMD ["python3", "/app/src/handler.py"]