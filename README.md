# [RoT: Enhancing Large Language Models with Reflection on Search Trees](http://paper.com)

RoT is a reflection framework designed to improve the performance of tree-search-based prompting methods and non-tree-search-based prompting methods such as [RAP](https://arxiv.org/abs/2305.14992), [ToT](https://arxiv.org/abs/2305.10601), and CoT based on previous valuable search experiences.

---
This repo contains the implementation and experiment code of Blocksworld and GSM8K. For the implementation of CraigslistBargain, see [RoT dialogue](https://to-proactivedial), as its tree search process is much different from the above two tasks.

## Quick Start
Install the required libraries.
```bash
conda create -n rot python=3.10
conda activate rot

git clone https://github.com/huiwy/reflection-on-trees --recursive
cd reflection-on-trees
pip install -r requirements.txt
```

RoT uses [vllm](https://github.com/vllm-project/vllm) to support efficient text generation, so you need to first launch a vllm service.
```bash
cd vllm-server
sh phi-2.sh
```

Then you can run RoT to generate the new prompts with guidelines based on the served model.
```bash
export OPENAI_API_BASE=xxx
export OPENAI_API_KEY=xxx

sh blocksworld_rot.sh prompts/bw/pool_prompt_rot.json # the prompt with RoT are generated at prompts/bw/pool_prompt_rot.json
sh gsm8k_rot.sh prompts/gsm8k/prompt_pool_rot.json # the prompt with RoT are generated at prompts/gsm8k/prompt_pool_rot.json
```

Finally add the genereted prompt to prompt dict in `blocksword_control.py` or `gsm8k_control.py`:
```python
prompt_path = {
    'default': 'prompts/bw/pool_prompt_v2_step_{step}.json',
    'rot': 'prompts/bw/pool_prompt_v2_step_{step}_rot.json',
    ...
+   'rot-new': 'prompts/gsm8k/prompt_pool_rot.json'
}
```

and run with the new prompt with guidelines:

```bash
python gsm8k_control.py --mode mcts --n_iter 10 --split train --prompt rot-new
```

## Acknowledgement 
This repo is built on [llm-reasoner](https://llm-reasoners).

## Citation
```
@article{hao2023reasoning,
  title={Reasoning with language model is planning with world model},
  author={Hao, Shibo and Gu, Yi and Ma, Haodi and Hong, Joshua Jiahua and Wang, Zhen and Wang, Daisy Zhe and Hu, Zhiting},
  journal={arXiv preprint arXiv:2305.14992},
  year={2023}
}
```