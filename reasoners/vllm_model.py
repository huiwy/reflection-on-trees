import os
import numpy as np
from typing import Optional, Union
import time
import httpx
from . import GenerateOutput

from requests_futures.sessions import FuturesSession

API_BASE = os.getenv("VLLM_API_BASE", None)

class VLLMModel:
    def __init__(self, model: str, max_tokens:int = 200, temperature=0.7):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    
    def generate(self,
                prompt: str,
                max_tokens: int = None,
                num_return_sequences: int = 1,
                stop: Optional[str] = None,
                temperature = None,
                logprobs=False,
                **kwargs) -> GenerateOutput:
        if type(prompt) == str:
            prompt = [prompt]

        gpt_temperature = self.temperature if temperature is None else temperature

        if max_tokens is None:
            max_tokens = self.max_tokens
        
        try:
            session = FuturesSession()
            url = API_BASE + "/completions"

            rs = []
            for p in prompt:
                data = {
                    'model': self.model,
                    'prompt': p,
                    'temperature': gpt_temperature,
                    'n': num_return_sequences,
                    'max_tokens': max_tokens,
                    'stop': stop,
                    'logprobs': logprobs
                }
                rs.append(session.post(url, json=data, timeout=60))

            rs = [r.result() for r in rs]
            responses = [r.json() for r in rs]
            session.close()
            
            log_prob = None
            if logprobs:
                log_prob = [
                    sum(choice["logprobs"]["token_logprobs"][:-1])/(len(choice['logprobs']['token_logprobs'])-1) 
                    for r in responses for choice in r["choices"]
                ]
                
            return GenerateOutput(
                text=[choice["text"] for r in responses for choice in r["choices"]],
                log_prob=log_prob
            )
            
        except Exception as e:
            print(rs)
            raise e

    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              **kwargs) -> list[np.ndarray]:
        if isinstance(prompt, str):
            prompt = [prompt]
        
        session = FuturesSession()

        res = []
        for p in prompt:
            data = {
                'model': self.model,
                'prompt': p,
                'n': 1,
                'max_tokens': 1,
                'logprobs': 1000,
            }

            r = session.post(API_BASE + "/completions", json=data, timeout=60)
            res.append(r)

        responses = [r.result().json() for r in res]
        results = []
        for r in responses:
            cand_log_probs = []
            top_log_probs = r["choices"][0]["logprobs"]["top_logprobs"][0]
            for cand in candidates:
                r = -1000
                for key in top_log_probs:
                    if cand in key and top_log_probs[key] > r:
                        r = top_log_probs[key]

                cand_log_probs.append(r)

            results.append(np.array(cand_log_probs))

        session.close()

        return results
    
    def get_loglikelihood(self,
                    prefixs: list[str],
                    prompts: list[str],
                    **kwargs) -> list[np.ndarray]:
        
        session = FuturesSession()
        res = []

        for prefix, prompt in zip(prefixs, prompts):

            data = {
                'prompt': prompt,
                'prefix': prefix,
            }

            r = session.post(API_BASE + "/logprobs", json=data, timeout=60)
            res.append(r)

        responses = [r.result().json() for r in res]
        results = []

        for r in responses:
            results.append(sum(r["logprobs"]))

        session.close()

        return np.array(results)