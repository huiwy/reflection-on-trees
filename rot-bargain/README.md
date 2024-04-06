# RoT of CraigslistBargain

## Quick Start
Similar to RoT with deterministic environment, we use [vllm](https://github.com/vllm-project/vllm) to support efficient text generation, so you need to launch a vllm when using a deployed model.
```bash
cd ../vllm-server
sh mixtral.sh
```

Then you can run RoT to generate the new prompts with guidelines based on the served model.
```bash
python bargain_control.py \
  --mode mcts --n_iter 8 \
  --seller mixtral \ # the model name of the seller
  --stage reflect \
  --rot_output_path prompts/rot.json # path to the file to store the guidelines.
```

Finally run LLM with the guidelines:
```bash
python bargain_control.py \
  --mode mcts --n_iter 8 \
  --seller mixtral  --stage evaluate \
  --prompt_path prompts/rot.json
```