import os
import json
from collections import defaultdict
import numpy as np
from utils import *


def covnert_graph(relations, nodes):
    old_relations = relations

    actions_nodes = defaultdict(dict)
    for key, value in old_relations.items():
        if '__' in key:
            node = key.split('__')[0]
            action = key.split('__')[1]

            if action not in actions_nodes[node]:
                actions_nodes[node][action] = {
                    'value': 0,
                    'children': []
                }
            actions_nodes[node][action]['children'].extend(value)

    return actions_nodes

def compute_action_value(relations, nodes):
    for node, actions in relations.items():
        for action, info in actions.items():
            info['value'] = sum([nodes[child]['value'] for child in info['children']]) / len(info['children'])

    return relations

def find_good_actions_nodes(action_nodes):
    node_value = {
        node: [action_nodes[node][a]['value'] for a in action_nodes[node]] for node in action_nodes
    }

    node_avg = {
        node: np.mean(node_value[node]) for node in node_value
    }

    node_var = {
        node: np.var(node_value[node]) for node in node_value
    }

    node_max = {
        node: np.max(node_value[node]) for node in node_value
    }

    node_gain = {
        node: node_max[node] - node_avg[node] for node in node_value
    }

    return node_avg, node_var, node_max, node_gain

def merge_guidelines(guidelines):
    
    messages = [{
        "role": "user",
        "content": "Please merge the following policies into one:\n" + "\n".join(guidelines)
    }]

    res = query_gpt4(messages)
    return res

def process(args):
    gpt4_analysis = []

    path = args.path
    
    bargains = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    
    
    for i in bargains:
        p = os.path.join(path, i)
        
        relations = json.load(open(f'{p}/relationship.json'))
        nodes = json.load(open(f'{p}/node_info.json'))
        background = json.load(open(f'{p}/game.json'))

        recover_dialogue(relations, nodes, 'node_0', background['history'][:2])
        action_nodes = covnert_graph(relations, nodes)

        action_nodes = compute_action_value(action_nodes, nodes)
        good_nodes = find_good_actions_nodes(action_nodes)

        selected_nodes = [k for k in good_nodes[3] if good_nodes[3][k] > 0.1]

        for n in selected_nodes:
            res = query_gpt4_for_advise_node(action_nodes, n, nodes)
            
            gpt4_analysis.append({
                'node': n,
                'res': res.split('Summary:')[1].strip()
            })
            
    guidelines = merge_guidelines([g['res'] for g in gpt4_analysis])
    
    with open(args.output_path, 'w') as f:
        json.dump(guidelines, f, indent=4)
    

def query_gpt4_for_advise_node(action_nodes, node, nodes):
    history = nodes[node]['history']
    history = '\n\n'.join([f'Buyer: {h}' if i % 2 == 0 else f'Seller: {h}' for i, h in enumerate(history)]) + '\n\n'

    strategies = action_nodes[node].keys()
    strategies = '\n'.join([f'Strategy {i}: {s} Value: {action_nodes[node][s]["value"]}' for i, s in enumerate(strategies) if s != 'history'])

    prompt = f"Dialogue history:\n{history}\nThe strategies of the seller used to reply in the current response and their rewards are listed below:\n{strategies}\n\n" + \
             f"Can you analyze the reason and then summarize the findings into a policy in one sentence in the format of Analysis: ...\nSummary:\n..."
    
    res = query_gpt4([
        {
            'role': 'system',
            'content': 'You are GPT4 trained by OpenAI. Now please act as a dialogue analyzer who can evaluate the behaviour of the seller in a bargining dialogue.',
        },
        {
            'role': 'user',
            'content': prompt
        }
    ])
    return res


import argparse

def recover_dialogue(relation, node, current_node, history):
    node[current_node]['history'] = history + [node[current_node]['response']]

    for r in relation:
        if r.split('__')[0] == current_node:
            for child in relation[r]:
                recover_dialogue(relation, node, child, history)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--output-path', type=str, default='guidelines.json')
    args = parser.parse_args()

    process(args)