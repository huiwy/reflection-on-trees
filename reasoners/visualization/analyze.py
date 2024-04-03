from .tree_log import TreeLog
from .tree_snapshot import TreeSnapshot, NodeData

import pickle
import random
import math

def default_node_data_factory(n):
    if not n.state:
        return NodeData({})
    # transform any object to dict
    if hasattr(n.state, "_asdict"):
        # if the state is a NamedTuple
        state_dict = n.state._asdict()
    elif isinstance(n.state, list):
        state_dict = {idx: value for idx, value in enumerate(n.state)}
    else:
        try:
            state_dict = dict(n.state)
        except TypeError:
            raise TypeError("The type of the state is not supported. "
                            "Please provide a node_data_factory function to transform the state to a dict.")
    return NodeData(state_dict)

class Analysis:
    def __init__(self, tree_log: TreeLog, evaluator, tree_log_raw, node_data_factory=default_node_data_factory):
        self.tree_log = tree_log
        self.tree_snapshot: TreeSnapshot = tree_log[-1]
        self.evaluator = evaluator
        self.tree_log_raw = tree_log_raw
        self.node_data_factory = node_data_factory
        

    @classmethod
    def from_file(cls, filename, evaluator, **kwargs):
        with open(filename, 'rb') as f:
            tree_log_raw = pickle.load(f)
        tree_log = TreeLog.from_mcts_results(tree_log_raw, **kwargs)

        return cls(tree_log, evaluator, tree_log_raw, **kwargs)

    def get_improving_nodes(self, threshold=0.5, method='random'):
        for n in self.tree_snapshot.nodes:
            nn = self.tree_snapshot.nodes[n]
            if self.evaluator.is_terminal(nn):
                nn.value = self.tree_snapshot.in_edges(n)[0].data['Q']
            else:
                nn.value = sum([self.tree_snapshot.nodes[c].data['Q'] for c in self.tree_snapshot.children(n)])/len(self.tree_snapshot.children(n))
            
        good_nodes = set()
        if method == 'random':
            for e in self.tree_snapshot.edges:
                ee = self.tree_snapshot.edges[e]
                if random.random() < 0.3:
                    good_nodes.add(ee.source)
        if method == 'all':
            for e in self.tree_snapshot.edges:
                good_nodes.add(self.tree_snapshot.edges[e].source)
        else:
            for e in self.tree_snapshot.edges:
                ee = self.tree_snapshot.edges[e]
                if self.tree_snapshot.nodes[ee.target].data['Q'] - self.tree_snapshot.nodes[ee.source].value >= threshold:
                    good_nodes.add(ee.source)

        return good_nodes

    def get_iteration_res(self, iter):
        node = self.tree_log_raw.tree_state_after_each_iter[iter]
        _, path = self.dfs_node([node])
        
        terminal = path[-1]
        node = TreeSnapshot.Node(0, data=self.node_data_factory(terminal))
        return self.evaluator.get_score(node)

        
    def dfs_node(self, path):
        cur = path[-1]
        if cur.is_terminal:
            return sum([node.reward for node in path[1:]]), path
        if cur.children is None:
            return -math.inf, path
        visited_children = [x for x in cur.children if x.state is not None]
        if len(visited_children) == 0:
            return -math.inf, path
        return max((self.dfs_node(path + [child]) for child in visited_children), key=lambda x: x[0])        
    
    def get_first_res_iter(self, max_iter):
        for i in range(max_iter):
            if self.get_iteration_res(i):
                return i
        return None

    def get_contrastive_sample(self):
        terminal_nodes = []
        for n in self.tree_snapshot.nodes:
            if self.evaluator.is_terminal(self.tree_snapshot.nodes[n]):
                terminal_nodes.append([self.tree_snapshot.nodes[n], self.evaluator.get_score(self.tree_snapshot.nodes[n])])

        if len(set([n[1] for n in terminal_nodes])) == 1:
            return None
        
        good_node = max(terminal_nodes, key=lambda x: x[1])
        bad_node = min(terminal_nodes, key=lambda x: x[1])

        good_path = []
        bad_path = []

        cur_node = good_node[0]
        while cur_node.id != 0:
            good_path.append(cur_node)
            cur_node = self.tree_snapshot.nodes[self.tree_snapshot.parent(cur_node.id)]
        
        cur_node = bad_node[0]
        while cur_node.id != 0:
            bad_path.append(cur_node)
            cur_node = self.tree_snapshot.nodes[self.tree_snapshot.parent(cur_node.id)]

        good_path = good_path[::-1]
        bad_path = bad_path[::-1]

        return good_path, bad_path
    
    def backprop_rewards(self, from_leaves=True, reward=False):
        self.remove_empty_nodes()
        if not reward:
            for n in self.tree_snapshot.nodes:
                nn = self.tree_snapshot.nodes[n]
                if self.evaluator.is_terminal(nn):
                    nn.data['reward'] = self.evaluator.get_score(nn)
                    nn.data['Q'] = nn.data['reward']
                    nn.data['N'] = 1
                    print(nn.data)

        if from_leaves:
            for n in self.tree_snapshot.nodes:
                nn = self.tree_snapshot.nodes[n]
                if self.evaluator.is_terminal(nn):                
                    self.backprop_rewards_from_node(nn)
                
                
        for n in self.tree_snapshot.nodes:
            nn = self.tree_snapshot.nodes[n]
            if self.evaluator.is_terminal(nn):
                continue
            print(nn.data)
            print(self.tree_snapshot.children(nn.id))
            if nn.data['blocks_state'] == 'the blue block is clear, the orange block is in the hand, the yellow block is clear, the hand is holding the orange block, the yellow block is on top of the red block, the blue block is on the table, and the red block is on the table.':
                breakpoint()
            nn.data['Q'] /= nn.data['N']
            
    def backprop_rewards_from_node(self, node: TreeSnapshot.Node):
        current_node_id = node.id
        Q = node.data['Q']
        
        while True:
            if not current_node_id in self.tree_snapshot._parent:
                break
            
            parent_node_id = self.tree_snapshot.parent(current_node_id)
            parent_node = self.tree_snapshot.nodes[parent_node_id]
            
            if not "N" in parent_node.data:
                parent_node.data['N'] = 0
                parent_node.data['Q'] = 0
                
            parent_node.data['Q'] += Q
            parent_node.data['N'] += 1
            
            current_node_id = parent_node_id
            
    def remove_empty_nodes(self):
        to_remove = []
        for n in self.tree_snapshot.nodes:
            if ((len(self.tree_snapshot.nodes[n].data) == 0 or self.is_gsm8k_empty(n)) and self.tree_snapshot.nodes[n].id != 0) or \
                ((not self.evaluator.is_terminal(self.tree_snapshot.nodes[n]) and len(self.tree_snapshot.children(n)) == 0)):
                to_remove.append(n)

        if to_remove == []:
            return
        
        for n in to_remove:
            del self.tree_snapshot.nodes[n]
        parent_ids = [self.tree_snapshot.parent(n) for n in to_remove]
        
        for r, p in zip(to_remove, parent_ids):
            self.tree_snapshot._children[p].remove(r)
            del self.tree_snapshot._parent[r]
        
        to_remove_edges = []
        for e in self.tree_snapshot.edges:
            if self.tree_snapshot.edges[e].source in to_remove or self.tree_snapshot.edges[e].target in to_remove:
                to_remove_edges.append(e)
                
        for e in to_remove_edges:
            del self.tree_snapshot.edges[e]
        
        self.remove_empty_nodes()
            
    def get_node_details(self, node):
        state = self.tree_snapshot.nodes[node].data['blocks_state']
        
        children = self.tree_snapshot.children(node)
        
        action_value_pairs = [
            f"Action: {self.tree_snapshot.nodes[c].data['history_actions'].split(',')[-1].strip()} Reward: {self.tree_snapshot.nodes[c].data['Q']}" for c in children
        ]
        
        return state, action_value_pairs
    
    def get_node_details_gsm8k(self, node):
        trace = self.tree_snapshot.trace(node)
        
        state = '\n'.join([f'Subquestion: {self.tree_snapshot.nodes[n].data["question"]} Subanswer: {self.tree_snapshot.nodes[n].data["answer"]} Reward: {self.tree_snapshot.nodes[n].data["Q"]}' for n in trace])
        
        children = self.tree_snapshot.children(node)
        action_value_pairs = [
            f"Subquestion: {self.tree_snapshot.nodes[n].data['question']} Subanswer: {self.tree_snapshot.nodes[n].data['answer']} Reward: {self.tree_snapshot.nodes[n].data['Q']}" for n in children
        ]
        
        return state, action_value_pairs
    
    def get_node_details_gsm8k_train(self, node):
        trace = self.tree_snapshot.trace(node)
        
        state = ' '.join([self.tree_snapshot.nodes[n].data["question"] for n in trace])
        
        children = self.tree_snapshot.children(node)
        best_child = max(children, key=lambda x: self.tree_snapshot.nodes[x].data['Q'])
        worst_child = min(children, key=lambda x: self.tree_snapshot.nodes[x].data['Q'])

        action_value_pairs = {
            'chosen': self.tree_snapshot.nodes[best_child].data['question'],
            'rejected': self.tree_snapshot.nodes[worst_child].data['question']
        }
        
        return state, action_value_pairs
    
    def contains_correct_node(self, iter):
        for n in self.tree_log[iter].nodes:
            if self.evaluator.is_terminal(self.tree_log[iter].nodes[n]) and self.evaluator.get_score(self.tree_log[iter].nodes[n]):
                return True
    
    def find_first_iter(self, max_iter=10):
        for i in range(max_iter):
            if self.contains_correct_node(i):
                return i
            
        return 10
        
    def is_gsm8k_empty(self, node):
        if 'answer' in self.tree_snapshot.nodes[node].data:
            return self.tree_snapshot.nodes[node].data['answer'] == 'Not finished'
        return False