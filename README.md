# TensorRT-LLM Worker for RunPod

This repository contains a RunPod serverless worker for running TensorRT-LLM models. TensorRT-LLM is NVIDIA's library for accelerating LLM inference on NVIDIA GPUs.

## Features

- Optimized LLM inference using NVIDIA's TensorRT-LLM
- OpenAI-compatible API endpoints
- Support for various model architectures
- Configurable parameters via environment variables

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRTLLM_MODEL` | Model path or HuggingFace model ID | (Required) |
| `TRTLLM_TOKENIZER` | Tokenizer path or HuggingFace tokenizer ID | Same as model |
| `TRTLLM_MAX_BEAM_WIDTH` | Maximum beam width | Default from BuildConfig |
| `TRTLLM_MAX_BATCH_SIZE` | Maximum batch size | Default from BuildConfig |
| `TRTLLM_MAX_NUM_TOKENS` | Maximum number of tokens | Default from BuildConfig |
| `TRTLLM_MAX_SEQ_LEN` | Maximum sequence length | Default from BuildConfig |
| `TRTLLM_TP_SIZE` | Tensor parallelism size | 1 |
| `TRTLLM_PP_SIZE` | Pipeline parallelism size | 1 |
| `TRTLLM_KV_CACHE_FREE_GPU_MEMORY_FRACTION` | Fraction of GPU memory to use for KV cache | 0.9 |
| `TRTLLM_TRUST_REMOTE_CODE` | Whether to trust remote code | false |
| `HF_TOKEN` | HuggingFace token for accessing gated models | (Optional) |

## API Usage

The worker supports both direct generation and OpenAI-compatible endpoints:

### Direct Generation

```json
{
  "prompt": "What is TensorRT-LLM?",
  "max_tokens": 100,
  "temperature": 0.7
}
```

### OpenAI-compatible Endpoints

```json
{
  "openai_route": "/v1/chat/completions",
  "openai_input": {
    "model": "tensorrt_llm_model",
    "messages": [
      {"role": "user", "content": "What is TensorRT-LLM?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }
}
```

## Building and Running Locally

```bash
docker build -t tensorrt-llm-worker .
docker run --gpus all -e TRTLLM_MODEL=your_model_path tensorrt-llm-worker
```

## License

See the [LICENSE](LICENSE) file for details.