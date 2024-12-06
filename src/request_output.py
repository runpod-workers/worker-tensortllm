import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional
import traceback  # For capturing the current exception stack trace


@dataclass
class Output:
    index: int
    text: str
    text_diff: str
    token_ids: List[int]
    token_ids_diff: List[int]
    logprobs: List[float]
    logprobs_diff: List[float]
    length: int
    finish_reason: Optional[str] = None
    stop_reason: Optional[str] = None

class RequestOutput:
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.prompt_token_ids: List[int] = []
        self.outputs: List[Output] = []
        self._done = False

    async def __aiter__(self):
        while not self._done:
            yield self
            await asyncio.sleep(0.1)  # Prevent tight loop
        
    async def aresult(self):
        while not self._done:
            await asyncio.sleep(0.1)  # Prevent tight loop


class CppExecutorError(RuntimeError):
    def __init__(self, message: Optional[str] = None):
        self.message = message
        self.stack_trace = traceback.format_exc()  # Captures the stack trace of the most recent exception
        super().__init__(message)

    def __str__(self):
        return f"{self.message}\nStack trace:\n{self.stack_trace}"