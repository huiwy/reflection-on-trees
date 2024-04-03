import sys
import os

path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(1, path)

import json
from reasoners.benchmark import BWEvaluator
import fire

class CoTReasoner():
    def __init__(self, base_model, temperature=0.8):
        self.base_model = base_model
        self.temperature = temperature
    def __call__(self, example, prompt=None):
        inputs = prompt["icl"].replace("<init_state>", example["init"])\
            .replace("<goals>", example["goal"]).replace("<action>", "")
        output = self.base_model.generate([inputs],
                                          do_sample=True,
                                          temperature=self.temperature,
                                          stop='\n[').text[0].strip()
        return output

def main(hf_path, data_path, prompt_path, disable_log=False, batch_size=1, config_file: str = "examples/blocksworld/data/bw_config.yaml", domain_file: str = "examples/blocksworld/data/generated_domain.pddl", resume=0, log_dir=None, temperature=0.8, exllama_mem_map: str = None):

    from reasoners import VLLMModel
    base_model = VLLMModel(model=hf_path)


    with open(prompt_path) as f:
        prompt = json.load(f)

    reasoner = CoTReasoner(base_model, temperature=temperature)
    evaluator = BWEvaluator(config_file=config_file, domain_file=domain_file, data_path=data_path, init_prompt=prompt, disable_log=disable_log, output_extractor=lambda x:x, sample_prompt_type="rap") # rap prompt includes cot
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(f'accuracy: {accuracy:.4f}')
    return 0

if __name__ == '__main__':
    fire.Fire(main)