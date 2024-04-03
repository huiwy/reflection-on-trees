output_name=$1

outputs=$(python blocksworld_control.py --mode mcts --n_iter 10)
log_dir=$(echo $outputs | grep -oP 'log_dir [^ ]*' | cut -d' ' -f2)

python rot_scripts/blocksworld_analysis.py --path $log_dir/algo_output --steps 4 --output_name $log_dir/rot_analysis.json
python rot_scripts/blocksworld_generate_rot_prompt.py --path $log_dir/rot_analysis.json --output_name $output_name --steps 4