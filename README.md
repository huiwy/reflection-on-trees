# [RoT: Enhancing Large Language Models with Reflection on Search Trees](http://paper.com)

**Reflection on search Trees (RoT)** is an LLM reflection framework designed to improve the performance of tree search based on previous valuable search experiences.

This repo contains the implementation and experiment code of Blocksworld and GSM8K, for the implementation of CraigslistBargain, see [proactive dialogue](https://to-proactivedial), as its MCTS  much different from the above two tasks.

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

Then you can run RoT based on the served model.
```bash
export OPENAI_API_BASE=xxx
export OPENAI_API_KEY=xxx

sh blocksworld_control.py
```

## Acknowledgement 
This repo are built based on [llm-reasoner](https://llm-reasoners) 