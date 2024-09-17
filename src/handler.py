import os
import runpod
from typing import List
from tensorrt_llm import LLM, SamplingParams
from huggingface_hub import login

# Enable build caching
os.environ["TLLM_HLAPI_BUILD_CACHE"] = "1"
# Optionally, set a custom cache directory
# os.environ["TLLM_HLAPI_BUILD_CACHE_ROOT"] = "/path/to/custom/cache"
#HF_TOKEN for downloading models   



hf_token = os.environ["HF_TOKEN"]
login(token=hf_token)



class TRTLLMWorker:
    def __init__(self, model_path: str):
        self.llm = LLM(model=model_path, enable_build_cache=True)
        
    def generate(self, prompts: List[str], max_tokens: int = 100) -> List[str]:
        sampling_params = SamplingParams(max_new_tokens=max_tokens)
        outputs = self.llm.generate(prompts, sampling_params)
        
        results = []
        for output in outputs:
            results.append(output.outputs[0].text)
        return results

# Initialize the worker outside the handler
# This ensures the model is loaded only once when the serverless function starts
# this path is hf model "<org_name>/model_name" egs: meta-llama/Meta-Llama-3.1-8B-Instruct 
model_path = os.environ["MODEL_PATH"]
worker = TRTLLMWorker(model_path)



def handler(job):
    """Handler function that will be used to process jobs."""
    job_input = job['input']
    prompts = job_input.get('prompts', ["Hello, how are you?"])
    max_tokens = job_input.get('max_tokens', 100)
    
    try:
        results = worker.generate(prompts, max_tokens)
        return {"status": "success", "output": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}


runpod.serverless.start({"handler": handler})