import os
import time
import argparse
from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seller', type=str, default='mixtral')
    parser.add_argument('--buyer', type=str, default='gpt4')
    parser.add_argument('--n_iters', type=int, default=-1)
    parser.add_argument('--panelty', type=float, default=0.0)
    parser.add_argument('--mode', type=str, default='mcts')
    parser.add_argument('--stage', type=str, default='evaluate')
    parser.add_argument('--prompt_path', type=str, default='none')
    parser.add_argument('--rot_output_path', type=str, default='prompts/rot.json')
    args = parser.parse_args()

    if args.mode == 'mcts':
        command = f'python run/run_mcts.py --stage {args.stage}'
    else:
        command = f'python run/run_cot.py --stage {args.stage}'
        
    command += f' --buyer {args.buyer} --seller {args.seller}'

    if args.n_iters != -1:
        command += f' --n_iters {args.n_iters}'

    command += f' --prompt_path {args.prompt_path} --panelty {args.panelty}'
    
    log_path = f'logs/{args.mode}_{args.buyer}_{args.seller}_{args.prompt_path.split("/")[-1]}_{args.n_iters}_{args.panelty}_{datetime.now().strftime("%Y%m%d-%H%M")}'

    command += f' --log_path {log_path}'
    print(command)

    import time
    t = time.time()
    os.system(command)
    time_consumed = time.time() - t
    with open(f'{log_path}/time_consumed.txt', 'w') as f:
        f.write(str(time_consumed))
        
    if args.stage == 'reflect':
        os.system(f'python mcts_reflect.py --path {log_path} --output-path {args.rot_output_path}')