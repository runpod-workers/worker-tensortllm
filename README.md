# TensorRT-LLM Worker for RunPod

This repository contains a RunPod worker implementation for NVIDIA's TensorRT-LLM, enabling efficient inference of large language models on NVIDIA GPUs.

[![RunPod](https://img.shields.io/badge/RunPod-Ready-success)](https://runpod.io)

## Features

- Optimized LLM inference using NVIDIA's TensorRT-LLM
- OpenAI-compatible API endpoints
- Support for various model architectures
- Configurable parameters via environment variables

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRTLLM_MODEL` | Model path or HuggingFace model name | (Required) |
| `TRTLLM_TOKENIZER` | Tokenizer path or name (defaults to model) | Same as model |
| `TRTLLM_MAX_BEAM_WIDTH` | Maximum beam width | Default from BuildConfig |
| `TRTLLM_MAX_BATCH_SIZE` | Maximum batch size | Default from BuildConfig |
| `TRTLLM_MAX_NUM_TOKENS` | Maximum number of tokens | Default from BuildConfig |
| `TRTLLM_MAX_SEQ_LEN` | Maximum sequence length | Default from BuildConfig |
| `TRTLLM_TP_SIZE` | Tensor parallelism size | 1 |
| `TRTLLM_PP_SIZE` | Pipeline parallelism size | 1 |
| `TRTLLM_KV_CACHE_FREE_GPU_MEMORY_FRACTION` | KV cache free GPU memory fraction | 0.9 |
| `TRTLLM_TRUST_REMOTE_CODE` | Whether to trust remote code | false |
| `HF_TOKEN` | HuggingFace token for private models | (Optional) |

## API Usage

The worker exposes an OpenAI-compatible API. You can use it with the following endpoints:

- `/generate` - Basic generation endpoint
- `/v1/chat/completions` - OpenAI-compatible chat completions
- `/v1/completions` - OpenAI-compatible completions

## Building and Running Locally

```bash
# Build the Docker image
docker build -t tensorrt-llm-worker .

# Run the container
docker run --gpus all -p 8000:8000 -e TRTLLM_MODEL=meta-llama/Llama-2-7b-chat-hf tensorrt-llm-worker
```

## Notes

- TensorRT-LLM is installed from source during the Docker build process
- The worker requires NVIDIA GPUs with appropriate drivers
- For optimal performance, use GPUs with sufficient VRAM for your chosen model

## License

See the [LICENSE](LICENSE) file for details.