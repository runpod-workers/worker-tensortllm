# TensorRT-LLM Worker for RunPod

This repository contains a RunPod worker implementation for NVIDIA's TensorRT-LLM, enabling efficient inference of large language models on NVIDIA GPUs.

## Features

- Optimized LLM inference using NVIDIA TensorRT-LLM
- Compatible with RunPod Serverless platform
- Supports various model architectures
- Configurable via environment variables
- OpenAI-compatible API endpoints

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
| `TRTLLM_KV_CACHE_FREE_GPU_MEMORY_FRACTION` | KV cache free GPU memory fraction | 0.9 |
| `TRTLLM_TRUST_REMOTE_CODE` | Whether to trust remote code | false |
| `HF_TOKEN` | HuggingFace token for private models | (Optional) |

## Usage

The worker accepts inputs in the following format:

```json
{
  "prompt": "What is TensorRT-LLM?",
  "max_tokens": 100,
  "temperature": 0.7
}
```

You can also use the OpenAI-compatible endpoints by specifying the route and input:

```json
{
  "openai_route": "/v1/chat/completions",
  "openai_input": {
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "What is TensorRT-LLM?"}
    ],
    "temperature": 0.7
  }
}
```

## Building and Running Locally

```bash
docker build -t tensortllm-worker .
docker run -e TRTLLM_MODEL=<model_name_or_path> tensortllm-worker
```

## License

See the [LICENSE](LICENSE) file for details.