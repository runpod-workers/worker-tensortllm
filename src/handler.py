import os
import asyncio
from typing import Optional, Dict, Any
from dataclasses import dataclass
import runpod
from transformers import AutoTokenizer
from tensorrt_llm import LLM, BuildConfig
from tensorrt_llm import LlmArgs
from serve import OpenAIServer
from dotenv import load_dotenv
from huggingface_hub import login
import requests

@dataclass
class ServerConfig:
    model: str
    tokenizer: Optional[str] = None
    max_beam_width: Optional[int] = BuildConfig.max_beam_width
    max_batch_size: Optional[int] = BuildConfig.max_batch_size
    max_num_tokens: Optional[int] = BuildConfig.max_num_tokens
    max_seq_len: Optional[int] = BuildConfig.max_seq_len
    tp_size: Optional[int] = 1
    pp_size: Optional[int] = 1
    kv_cache_free_gpu_memory_fraction: Optional[float] = 0.9
    trust_remote_code: bool = False

    @classmethod
    def from_env(cls) -> 'ServerConfig':
        model = os.getenv('TRTLLM_MODEL')
        if not model:
            raise ValueError("TRTLLM_MODEL environment variable must be set")

        return cls(
            model=model,
            tokenizer=os.getenv('TRTLLM_TOKENIZER'),
            max_beam_width=int(os.getenv('TRTLLM_MAX_BEAM_WIDTH', str(BuildConfig.max_beam_width))) if os.getenv('TRTLLM_MAX_BEAM_WIDTH') else None,
            max_batch_size=int(os.getenv('TRTLLM_MAX_BATCH_SIZE', str(BuildConfig.max_batch_size))) if os.getenv('TRTLLM_MAX_BATCH_SIZE') else None,
            max_num_tokens=int(os.getenv('TRTLLM_MAX_NUM_TOKENS', str(BuildConfig.max_num_tokens))) if os.getenv('TRTLLM_MAX_NUM_TOKENS') else None,
            max_seq_len=int(os.getenv('TRTLLM_MAX_SEQ_LEN', str(BuildConfig.max_seq_len))) if os.getenv('TRTLLM_MAX_SEQ_LEN') else None,
            tp_size=int(os.getenv('TRTLLM_TP_SIZE', '1')) if os.getenv('TRTLLM_TP_SIZE') else None,
            pp_size=int(os.getenv('TRTLLM_PP_SIZE', '1')) if os.getenv('TRTLLM_PP_SIZE') else None,
            kv_cache_free_gpu_memory_fraction=float(os.getenv('TRTLLM_KV_CACHE_FREE_GPU_MEMORY_FRACTION', '0.9')) if os.getenv('TRTLLM_KV_CACHE_FREE_GPU_MEMORY_FRACTION') else None,
            trust_remote_code=os.getenv('TRTLLM_TRUST_REMOTE_CODE', '').lower() in ('true', '1', 'yes')
        )

    def validate(self) -> None:
        if not self.model:
            raise ValueError("Model path or name must be provided")

class TensorRTLLMServer:
    """
    Singleton class to manage TensorRT-LLM server instance and handle requests
    """
    # _instance = None
    # _initialized = False

    # def __new__(cls):
    #     if cls._instance is None:
    #         cls._instance = super(TensorRTLLMServer, cls).__new__(cls)
    #     return cls._instance

    def __init__(self):
        self._initialize_server()
        self.host = '0.0.0.0'
        self.port = 8000

    def _initialize_server(self):
        """Initialize the TensorRT-LLM server and load model"""
        # Load environment variables
        load_dotenv()
        
        # Handle HuggingFace login
        huggingface_token = os.getenv("HF_TOKEN")
        if huggingface_token:
            print("Logging in to Hugging Face...")
            login(huggingface_token)

        # Initialize configuration
        self.config = ServerConfig.from_env()
        self.config.validate()

        # Create build configuration
        build_config = BuildConfig(
            max_batch_size=self.config.max_batch_size,
            max_num_tokens=self.config.max_num_tokens,
            max_beam_width=self.config.max_beam_width,
            max_seq_len=self.config.max_seq_len
        )

        # Initialize LLM
        self.llm = LLM(
            model=self.config.model,
            tokenizer=self.config.tokenizer,
            tensor_parallel_size=self.config.tp_size,
            pipeline_parallel_size=self.config.pp_size,
            trust_remote_code=self.config.trust_remote_code,
            build_config=build_config
        )

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer or self.config.model,
            trust_remote_code=self.config.trust_remote_code
        )

        # Initialize OpenAI compatible server
        self.server = OpenAIServer(
            llm=self.llm,
            model=self.config.model,
            hf_tokenizer=self.tokenizer
        )
        
        asyncio.run(self.server(self.host, self.port))

# Initialize the server at module load time
server = TensorRTLLMServer()

async def async_handler(job):
    """Handle the requests asynchronously."""
    job_input = job["input"]
    print(f"JOB_INPUT: {job_input}")
    
    base_url = "http://0.0.0.0:8000"
    
    if job_input.get("openai_route"):
        openai_route, openai_input = job_input.get("openai_route"), job_input.get("openai_input")

        openai_url = f"{base_url}" + openai_route
        headers = {"Content-Type": "application/json"}

        response = requests.post(openai_url, headers=headers, json=openai_input)
        # Process the streamed response
        if openai_input.get("stream", False):
            for formated_chunk in response:
                yield formated_chunk
        else:
            for chunk in response.iter_lines():
                if chunk:
                    decoded_chunk = chunk.decode('utf-8')
                    yield decoded_chunk        
    else:
        generate_url = f"{base_url}/generate"
        headers = {"Content-Type": "application/json"}
        # Directly pass `job_input` to `json`. Can we tell users the possible fields of `job_input`?
        response = requests.post(generate_url, json=job_input, headers=headers)
        if response.status_code == 200:
            yield response.json()
        else:
            yield {"error": f"Generate request failed with status code {response.status_code}", "details": response.text}

runpod.serverless.start({"handler": async_handler, "return_aggregate_stream": True})