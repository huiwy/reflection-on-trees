import os
import time
import argparse
from typing import Literal, Optional
from dataclasses import dataclass
from datetime import datetime

model_ip = 'http://10.234.38.2:23100/v1'

prompt_path = {
    'default': 'prompts/gsm8k/prompt_pool.json',
    'rot': 'prompts/gsm8k/prompt_pool_rot.json',
    'rot-2-iter': 'prompts/gsm8k/prompt_pool_rot_2_iter.json',
    'rot-3-iter': 'prompts/gsm8k/prompt_pool_rot_3_iter.json',
    'rot-4-iter': 'prompts/gsm8k/prompt_pool_rot_4_iter.json',
    'rot-5-iter': 'prompts/gsm8k/prompt_pool_rot_5_iter.json',
    'rot-6-iter': 'prompts/gsm8k/prompt_pool_rot_6_iter.json',
    '0.5': 'prompts/gsm8k/prompt_pool_rot_promising_0.5.json',
    'all': 'prompts/gsm8k/prompt_pool_rot_all.json',
    'random': 'prompts/gsm8k/prompt_pool_rot_random.json',
    'cot-default': 'prompts/gsm8k/cot_default.json',
    'cot-rot': 'prompts/gsm8k/cot_rot.json',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='default')
    parser.add_argument('--model', type=str, default='phi-2')
    parser.add_argument('--mode', type=str, default='mcts')
    parser.add_argument('--n_iters', type=int, default=-1)
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()

    os.environ['VLLM_API_BASE'] = model_ip

    if args.mode == 'cot':
        command = f'python examples/gsm8k/cot.py --hf_path {args.model}'
    elif args.mode == 'bfs':
        command = f'python examples/gsm8k/bfs.py --hf_path {args.model} --width {args.n_iters}'
    elif args.mode == 'mcts':
        command = f'python examples/gsm8k/rap.py --hf_path {args.model}'
        if args.n_iters != -1:
            command += f' --n_iters {args.n_iters}'
            
    command += f' --prompt {prompt_path[args.prompt]} --split {args.split}'
    
    log_path = f'logs/gsm8k/{args.model}_{args.prompt}_{args.mode}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    command += f' --log_dir {log_path}'
    print(command)

    import time
    t = time.time()
    os.system(command)
    time_consumed = time.time() - t
    with open(f'{log_path}/time_consumed.txt', 'w') as f:
        f.write(str(time_consumed))