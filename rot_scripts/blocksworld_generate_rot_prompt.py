import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)
parser.add_argument('--output_name', type=str, default='bw_summarization.json')
parser.add_argument('--steps', type=str, default='4')

args = parser.parse_args()

prompts = json.load(open(f'prompts/bw/pool_prompt_v2_step_{args.steps}_template.json'))
rot = json.load(open(f'{args.path}'))

prompts['intro'] = prompts['intro'].format(template=rot)
prompts['self-eval'] = prompts['self-eval'].format(template=rot)

with open(args.output_name, 'w') as f:
    json.dump(prompts, f, indent=2)