

<h1>OpenAI Compatible Tensort-LLM Worker </h1>
A high-performance inference server that combines the power of TensorRT-LLM for optimized model inference with RunPod's serverless infrastructure. This implementation provides an OpenAI-compatible API interface for easy integration with existing applications.

## Features

- TensorRT-LLM optimization for faster inference
- OpenAI-compatible API endpoints
- Flexible configuration through environment variables
- Support for model parallelism (tensor and pipeline)
- Hugging Face model integration
- Streaming response support
- RunPod serverless deployment ready

### Runtime Constraints
- Batch size and sequence length must be determined during engine building time
- Dynamic shape support is limited and may impact performance
- KV-cache size is fixed at build time and affects memory usage
- Changing model parameters requires rebuilding the TensorRT engine

### Build Time Impact
- Engine building can take significant time (hours for large models)
- Each combination of parameters requires a separate engine
- Changes to maximum sequence length or batch size require rebuilding

## Environment Variables

The server can be configured using the following environment variables:

```plaintext
TRTLLM_MODEL              # Required: Path or name of the model to load
TRTLLM_TOKENIZER          # Optional: Path or name of the tokenizer (defaults to model path)
TRTLLM_MAX_BEAM_WIDTH     # Optional: Maximum beam width for beam search
TRTLLM_MAX_BATCH_SIZE     # Optional: Maximum batch size for inference
TRTLLM_MAX_NUM_TOKENS     # Optional: Maximum number of tokens to generate
TRTLLM_MAX_SEQ_LEN        # Optional: Maximum sequence length
TRTLLM_TP_SIZE            # Optional: Tensor parallelism size (default: 1)
TRTLLM_PP_SIZE            # Optional: Pipeline parallelism size (default: 1)
TRTLLM_KV_CACHE_FREE_GPU_MEMORY_FRACTION  # Optional: GPU memory fraction for KV cache (default: 0.9)
TRTLLM_TRUST_REMOTE_CODE  # Optional: Whether to trust remote code (default: false)
HF_TOKEN                  # Optional: Hugging Face API token for protected models
