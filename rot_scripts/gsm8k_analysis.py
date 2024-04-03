from gpt4_utils import query_gpt4

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'examples', 'gsm8k')))

import datasets
import json

from reasoners.visualization import visualize, TreeLog, analyze
from reasoners.visualization.tree_snapshot import NodeData
from reasoners.algorithm.mcts import MCTSNode
from utils import retrieve_answer, retrieve_answer_from_dataset, judge_answer

class Evaluator:
    def __init__(self, data) -> None:
        self.data = data
    def is_terminal(self, node):
        if node.data['question'] is None:
            return False
        return 'Now we can answer the question:' in node.data['question']
    
    def get_score(self, node):
        pred = retrieve_answer(node.data['answer'])
        gold = retrieve_answer_from_dataset(self.data)
        return int(judge_answer(pred, gold))

def compose_solution(solution):
    text = ''
    for i, s in enumerate(solution):
        text += (f"sub-question {i}: {s.data['question']}\nsub-answer {i}: {s.data['answer']}\n")
    return text

def compose_summarization_prompt(case):
    question = case[0]['question']
    answer = case[0]['answer'].split('####')[1].strip()
    correct = compose_solution(case[1][0])
    wrong = compose_solution(case[1][1])

    return f"Below is a math word problem:\n{question}\nIts answer is: {answer}\nThe following are two solutions.\n\nCorrect solution:\n{correct}\nWrong solution:\n{wrong}\nPlease compare the above 2 solutions, briefly analyze how the mistake is made and give a practical policy to avoid this mistake.\nPlease write in the following format: Analysis: ...\nPolicy: ..."

def gsm_node_data_factory(x: MCTSNode):
    if not x.state:
        return {"question": x.action, "answer": "Not finished"}
    return {"question": x.action, "answer": x.state[-1].sub_answer}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--output_name', type=str, default='gsm8k_summarization.json')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--thres', type=float, default=0.1)
    parser.add_argument('--method', type=str, default='improving')
    parser.add_argument('--old_policy', type=str, default=None)
    
    args = parser.parse_args()

    full_dataset = datasets.load_dataset('gsm8k', 'main', split=args.split)
    max_idx = max([int(p.replace('.pkl', '')) for p in os.listdir(args.path)])

    res = []
    for i in range(max_idx):
        try:
            path = os.path.join(args.path, f'{i+1}.pkl')

            analysis = analyze.Analysis.from_file(path, evaluator=Evaluator(full_dataset[i]), node_data_factory=gsm_node_data_factory)
            analysis.backprop_rewards()
            
            if args.method == 'improving':
                nodes = analysis.get_improving_nodes(threshold=args.thres)
            else:
                nodes = analysis.get_improving_nodes(method=args.method)
            for node in nodes:
                res.append((full_dataset[i], analysis.get_node_details_gsm8k(node)))
        except Exception as e:
            pass
    
    with open('gsm8k_summarization.json', 'w') as f:
        json.dump(res, f, indent=4)


    cases = [x for x in res if x is not None][:30]

    batchsize = 5
    analysises = []
    for i in range(0, len(cases), batchsize):
        batch = cases[i:i+batchsize]
        NL = '\n'
        prompt = ''
        for i, bb in enumerate(batch):
                prompt += f"State {i}:\nQuestion {i}: {bb[0]['question']}\nPartial solution {i}: {bb[1][0]}"
                prompt += f"Possible Action & Response & Rewards:\n\n{NL.join([d.format(idx=ii) for ii, d in enumerate(bb[1][1])])}"

        messages = [
            {
                "role": "system",
                "content": "You are a math tutor. Now you are teaching a student who is solving a math word problem. Please summarize the following action, response and corresponding rewards given a state into a policy to achieve higher reward."
            },
            {
                "role": "user",
                "content": "Below are some examples of the process.\n" + prompt + "\n\nPlease give suggestion on how to ask subquestions (take action) and answer the subquestions (response) to achieve higher reward. Please first analyze the mistakes and then summarize a policy. Your policy should be specific to help avoid making the same mistake. Please follow this format: Summarization: ...\nPolicy: ..."
            }
        ]
    
        res = query_gpt4(messages)

        analysises.append({
            'sample': str(bb),
            'analysis': res.split('Summarization:')[1].split('Policy:')[0].strip(),
            'policy': res.split('Policy:')[1].strip(),
        })

        with open('gsm8k_summarization.json', 'w') as f:
            json.dump(analysises, f)

    analysises = json.load(open('gsm8k_summarization.json'))

    policies = [a['policy'] for a in analysises]

    messages = [
        {
            'content': f"The following are some policies. Please merge them as a comprehensive policy. Note you should keep the details in the merged policy and make sure that the merged policy is not too general.\n" + '\n\n'.join([f"Policy {i}:\n"+ s for i, s in enumerate(policies)]),
            'role': 'user'
        }
    ]

    res = query_gpt4(messages)

    with open(args.output_name, 'w') as f:
        json.dump(res, f)