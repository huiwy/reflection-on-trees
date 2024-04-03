import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)
parser.add_argument('--output_name', type=str)
parser.add_argument('--mode', type=str, default='cot')

args = parser.parse_args()

rot = json.load(open(f'{args.path}'))

if args.mode == 'cot':
    prompts = json.load(open(f'prompts/gsm8k/cot_default.json'))
    prompts['guidelines'] = rot
else:
    prompts = json.load(open(f'/home/wenyanghui/projects/reflection-on-trees/prompts/gsm8k/prompt_pool_template.json'))
    prompts['instruction'] = prompts['instruction'].format(rot=rot)

with open(args.output_name, 'w') as f:
    json.dump(prompts, f, indent=2)