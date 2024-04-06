export OPENAI_API_KEY='XXX'

# generate the rot guidelines
python bargain_control.py \
  --mode mcts --n_iter 8 \
  --seller mixtral  --stage reflect \
  --rot_output_path prompts/rot.json 

# evaluate the rot guidelines
python bargain_control.py \
  --mode mcts --n_iter 8 \
  --seller mixtral  --stage evaluate \
  --prompt_path prompts/rot.json