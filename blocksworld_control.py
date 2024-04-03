import os
import time
import argparse
from typing import Literal, Optional
from dataclasses import dataclass
from datetime import datetime

os.environ['VAL'] = 'LLMs-Planning/planner_tools/VAL'

model_ip = 'http://10.234.38.2:23100/v1'

prompt_path = {
    'default': 'prompts/bw/pool_prompt_v2_step_{step}.json',
    'rot': 'prompts/bw/pool_prompt_v2_step_{step}_rot.json',
    'rot-iter-2': 'prompts/bw/pool_prompt_v2_step_{step}_rot_iter_2.json',
    'rot-iter-3': 'prompts/bw/pool_prompt_v2_step_{step}_rot_iter_3.json',
    'rot-iter-4': 'prompts/bw/pool_prompt_v2_step_{step}_rot_iter_4.json',
    'rot-iter-5': 'prompts/bw/pool_prompt_v2_step_{step}_rot_iter_5.json',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='default')
    parser.add_argument('--model', type=str, default='phi-2')
    parser.add_argument('--mode', type=str, default='cot')
    parser.add_argument('--n_iters', type=int, default=-1)
    parser.add_argument('--width', type=int, default=-1)
    parser.add_argument('--step', type=int, default=4)
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()

    os.environ['VLLM_API_BASE'] = model_ip

    if args.mode == 'cot':
        command = f'python examples/blocksworld/cot.py --hf_path {args.model}'
    elif args.mode == 'bfs':
        command = f'python examples/blocksworld/bfs.py --hf_path {args.model} --depth_limit {args.step}'
        if args.width != -1:
            command += f' --width {args.width}'
    else:
        command = f'python examples/blocksworld/rap.py --hf_path {args.model}  --depth_limit {args.step} --output_trace_in_each_iter --n_actions 4'
        if args.n_iters != -1:
            command += f' --n_iters {args.n_iters}'

    command += f' --prompt_path {prompt_path[args.prompt].format(step=args.step)}'
    
    log_path = f'logs/bw/step_{args.step}/{args.model}_{args.prompt}_{args.mode}_{args.n_iters}_{datetime.now().strftime("%Y%m%d-%H%M")}'

    command += f' --data_path examples/blocksworld/data/split_v2/split_v2_step_{args.step}_data.json'
    command += f' --batch_size 1'
    command += f' --log_dir {log_path}'
    command += ' --config_file examples/blocksworld/data/bw_config.yaml'
    print(command)

    import time
    t = time.time()
    os.system(command)
    time_consumed = time.time() - t
    
    with open(f'{log_path}/time_consumed.txt', 'w') as f:
        f.write(str(time_consumed))