from gpt4_utils import query_gpt4

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'examples', 'blocksworld')))

import os
import json

from tqdm import trange
from reasoners.visualization import visualize, analyze, TreeLog
import reasoners.benchmark.bw_utils as bw_utils

os.environ['VAL'] = 'LLMs-Planning/planner_tools/VAL'


class Evaluator:
    def __init__(self, data, depth, config_file, domain_file) -> None:
        self.data = data
        self.depth = depth
        self.config_file = config_file
        self.domain_file = domain_file

    def is_terminal(self, node):
        if 'step_idx' not in node.data:
            return False
        print(node.data['step_idx'], self.depth)
        return node.data['step_idx'] == self.depth
    
    def get_score(self, node):
        output = node.data['history_actions'].replace(',', '\n')
        bw_utils.text_to_plan_blocksworld(output, self.data["instance_file"], self.config_file, self.domain_file, 'tmp.plan')
        correct = bw_utils.validate_plan(self.domain_file, self.data["instance_file"], 'tmp.plan')[0]

        return correct
    
config_file = 'examples/blocksworld/data/bw_config.yaml'
domain_file = 'examples/blocksworld/data/generated_domain.pddl'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--output_name', type=str, default='bw_summarization.json')
    parser.add_argument('--thres', type=float, default=0.1)
    parser.add_argument('--steps', type=int)    
    args = parser.parse_args()
    
    data_file = f'examples/blocksworld/data/split_v2/split_v2_step_{args.steps}_data.json'
    path = args.path
    data = bw_utils.load_blocksworld(config_file, domain_file, data_file)

    descriptions = []
    for i, d in enumerate(data):
        aa = analyze.Analysis.from_file(f"{path}/{i+1}.pkl", Evaluator(d, args.steps, config_file, domain_file))
        
        aa.backprop_rewards()
        nodes = aa.get_improving_nodes(args.thres)
        
        for n in nodes:
            descriptions.append({
                "goal": d['goal'],
                "details": aa.get_node_details(n)
            })
            
    def get_gpt4_summarizations(descriptions):
        summarizations = []
        NL= '\n'
        batchsize = 5
        for i in trange(0, len(descriptions), batchsize):
            batch = descriptions[i:i+batchsize]
            prompt = ''
            for i, b in enumerate(batch):
                prompt += f"Goal {i}: {b['goal']}\n\n"
                prompt += f"State {i}: {b['details'][0]}\nActions {i}:\n{NL.join(b['details'][1])}\n\n"

            messages = [
                {
                    "role": "system",
                    "content": "BlocksWorld is game that requires the agent to apply a sequence of actions to make configurations of blocks match a goal configuration. Please summarize the following action and corresponding rewards given a state into a policy to achieve higher reward."
                },
                {
                    "role": "user",
                    "content": prompt + "Note: Since the reward is not given when playing the game, your policy should avoid directly use the reward as information.  Your policy should be specific to help avoid making the same mistake. Please follow this format: Summarization: ...\nPolicy: ..."
                }
            ]
            
            summarizations.append(query_gpt4(messages))
            
            if len(summarizations) > 8:
                break

        with open('summarizations.json', 'r') as f:
            summarizations = json.load(f)[0]

        policies = [
            s.split("Policy:")[1].strip() for s in summarizations
        ]
        
        messages = [{
            "role": "user",
            "content": "Please merge the following policies into one:\n" + NL.join(policies)
        }]

        merged_policy = query_gpt4(messages)
            
        return summarizations, merged_policy

    summarizations = get_gpt4_summarizations(descriptions)

    with open('summarizations.json', 'w') as f:
        json.dump(summarizations, f, indent=4)
        
        