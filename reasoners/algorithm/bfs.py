import pickle
from os import PathLike
import math
from copy import deepcopy
from typing import Generic, Optional, NamedTuple, Callable, Hashable, List, Tuple
import itertools
from abc import ABC
from collections import defaultdict

import numpy as np
from tqdm import trange

from .. import SearchAlgorithm, WorldModel, SearchConfig, State, Action, Example, Trace


class BFSNode(Generic[State, Action]):
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self, state: Optional[State], action: Optional[Action], parent: "Optional[BFSNode]" = None,
                 fast_reward: float = 0., fast_reward_details=None,
                 is_terminal: bool = False):
        """
        A node in the MCTS search tree

        :param state: the current state
        :param action: the action of the last step, i.e., the action from parent node to current node
        :param parent: the parent node, None if root of the tree
        :param fast_reward: an estimation of the reward of the last step
        :param is_terminal: whether the current state is a terminal state
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        """
        self.id = next(BFSNode.id_iter)
        if fast_reward_details is None:
            fast_reward_details = {}
        self.cum_rewards: List[float] = []
        self.fast_reward = self.reward = fast_reward
        self.fast_reward_details = fast_reward_details
        self.is_terminal = is_terminal
        self.action = action
        self.state = state
        self.parent = parent
        self.children: 'Optional[List[BFSNode]]' = None
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

class BFSResult(NamedTuple):
    terminal_state: State
    trace: Trace
    trace_of_nodes: List[BFSNode]
    tree_state: BFSNode

class BFS(SearchAlgorithm, Generic[State, Action, Example]):
    def __init__(self,
                 depth_limit: int = 5,
                 width: int = 5,
                 disable_tqdm: bool = True,
                 node_visualizer: Callable[[BFSNode], dict] = lambda x: x.__dict__):
        super().__init__()
        self.world_model = None
        self.search_config = None
        self.terminals = []
        self.width = width
        self.depth_limit = depth_limit
        self._output_iter: List[BFSNode] = None
        self.trace_in_each_iter: List[List[BFSNode]] = None
        self.root: Optional[BFSNode] = None
        self.disable_tqdm = disable_tqdm
        self.node_visualizer = node_visualizer

    def _is_terminal_with_depth_limit(self, node: BFSNode):
        return node.is_terminal or node.depth >= self.depth_limit

    def _expand(self, node: BFSNode):
        if node.state is None:
            node.state, aux = self.world_model.step(node.parent.state, node.action)
            # reward is calculated after the state is updated, so that the
            # information can be cached and passed from the world model
            # to the reward function with **aux without repetitive computation
            node.reward, node.reward_details = self.search_config. \
                reward(node.parent.state, node.action, **node.fast_reward_details, **aux)
            node.is_terminal = self.world_model.is_terminal(node.state)

        if node.is_terminal:
            return

        children = []
        actions = self.search_config.get_actions(node.state)
        
        if getattr(self.search_config, 'fast_rewards', None) is None:
            for action in actions:
                fast_reward, fast_reward_details = self.search_config.fast_reward(node.state, action)
                child = BFSNode(state=None, action=action, parent=node,
                                fast_reward=fast_reward, fast_reward_details=fast_reward_details)
                children.append(child)
        else:
            fast_rewards, fast_reward_detailss = self.search_config.fast_rewards(node.state, actions)
            for r, d, a in zip(fast_rewards, fast_reward_detailss, actions):
                children.append(BFSNode(state=None, action=a, parent=node,
                                        fast_reward=r, fast_reward_details=d))

        node.children = children
        
        return children

    def search(self):
        self._output_cum_reward = -math.inf
        self._output_iter = None
        self.root = BFSNode(state=self.world_model.init_state(), action=None, parent=None)
        terminals = []
        frontier = [self.root]
        for _ in trange(self.depth_limit+1, disable=self.disable_tqdm):
            next_frontier = []
            for node in frontier:
                children = self._expand(node)
                if children is None:
                    continue

                non_terminal_children = [child for child in children if not child.is_terminal]                
                next_frontier.extend(non_terminal_children)

            terminals.extend([node for node in frontier if node.is_terminal])
            frontier = sorted(next_frontier, key=lambda x: x.fast_reward, reverse=True)[:self.width]
        if len(terminals) == 0:
            breakpoint()
        best_state = sorted(terminals, key=lambda x: x.reward, reverse=True)[0]

        self._output_iter = []
        
        while best_state.parent is not None:
            self._output_iter.append(best_state)
            best_state = best_state.parent
        self._output_iter.append(best_state)
        self._output_iter = self._output_iter[::-1]
        
    def __call__(self,
                 world_model: WorldModel[State, Action, Example],
                 search_config: SearchConfig[State, Action, Example],
                 **kwargs) -> BFSResult:
        BFSNode.reset_id()
        self.world_model = world_model
        self.search_config = search_config
        self.search()

        if self._output_iter is None:
            terminal_state = trace = None
        else:
            terminal_state = self._output_iter[-1].state
            trace = [node.state for node in self._output_iter], [node.action for node in self._output_iter[1:]]
            
        result = BFSResult(
            terminal_state=terminal_state,
            trace=trace,
            trace_of_nodes=self._output_iter,
            tree_state=self.root
        )
        
        l = [self.root]
        while l:
            node = l.pop()
            if node.children is not None:
                l.extend(node.children)
        
        return result
