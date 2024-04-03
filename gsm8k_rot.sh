outputs=$(python gsm8k_control.py --mode mcts --n_iter 3 --split train)
log_dir=$(echo $outputs | grep -oP 'log_dir [^ ]*' | cut -d' ' -f2)

python rot_scripts/gsm8k_analysis.py --path $log_dir/algo_output --output_name $log_dir/rot_analysis.json
python rot_scripts/gsm8k_generate_rot_prompt.py --path $log_dir/rot_analysis.json --output_name prompts/gsm8k/prompt_pool_rot_phi_2.json --mode mcts