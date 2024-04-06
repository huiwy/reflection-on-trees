import numpy as np
import logging
import math

from core import *
from user_simulators.naive_simulator import UserSimulator
from user_simulators.strategy_simulator import StrategyUserSimulator

from collections import defaultdict
from functools import partial

import time

class MCTSConfig:
	def __init__(self) -> None:
		self.cpuct = 1
		self.Q_0 = -0.5
		self.num_simulations = 5
		self.max_realizations = 2
		self.smoothing = 1

class DialogueNode:
	def __init__(self, data, parent) -> None:
		self.data = data
		self.parent = parent
		self.children = []
		self.visits = 0
		self.value = 0

		parent.children.append(self)
	
class OpenLoopMCTS:
	def __init__(self, configs, simulator=None) -> None:
		self.configs = configs

		self.Ns: dict = {}
		self.Nsa: dict = {}
		self.Q: dict = {}
		# utility
		self.valid_moves: dict = {}
		self.terminals: dict = {}
		self.simulator = simulator
		# debugging / more information
		self.Vs: dict = {}
		self.realizations: dict = defaultdict(list)
		self.realizations_Vs: dict = defaultdict(dict)
		self.realizations_Ns: dict = defaultdict(dict)
		self.max_realizations = configs.max_realizations

		self.simulation_results = defaultdict(list)

	def dump(self, name):
		import pickle
		with open(f'{name}', 'wb') as f:
			pickle.dump(self, f)

	def _init_node(self, state: DialogueGameState):
		game = DialogueGame.resume_gamestate(state)
		allowed_actions = game.get_valid_actions()

		self.valid_moves[state] = allowed_actions

		self.Ns[state] = 0
		self.Nsa[state] = {action: 0 for action in self.valid_moves[state]}
		self.Q[state] = {action: self.configs.Q_0 for action in self.valid_moves[state]}

		v = [1/len(allowed_actions) for _ in allowed_actions]
		self.Vs[state] = v
   
		return v

	def search(self, state):
		if not state in self.valid_moves:
			self._init_node(state)

		best_action = self.select_action(state)
		game = DialogueGame.resume_gamestate(state)

		realization_key = f"{state}__{best_action}"

		if not realization_key in self.realizations or len(self.realizations[realization_key]) < self.max_realizations:
			# mcts agent action
			finished, r, realization_state = game.step_and_state(action=best_action)
			rs = realization_state.copy()
			self.realizations[realization_key].append(rs)
			
			if finished:
				self.update(state, best_action, r)
				self.update_realizations(realization_key, realization_state, r)

				return r
			
			history, r = self.simulator.simulate(realization_state)
			self.simulation_results[rs].append((history, r))
			self.update(state, best_action, r)
			self.update_realizations(realization_key, realization_state, r)
			
			return r
		else:
			realization_state = random.choice(self.realizations[realization_key])
			game = DialogueGame.resume_gamestate(realization_state)

		# the other agent action
		finished, r, next_state = game.step_and_state()

		if finished:
			self.update(state, best_action, r)
			self.update_realizations(realization_key, realization_state, r)

			self.Vs[next_state] = r
			self.Q[next_state] = {'': r}

			return r
		
		v = self.search(next_state)

		self.update(state, best_action, v)
		self.update_realizations(realization_key, realization_state, v)
  
		return v

	def get_best_realization(self, state, action):
		best_v = -10000
		best_realization = None
		for realization in self.realizations[f"{state}__{action}"]:
			if realization in self.realizations_Vs[f"{state}__{action}"]:
				v = self.realizations_Vs[f"{state}__{action}"][realization]
				if v > best_v:
					best_v = v
					best_realization = realization
		return best_realization

	def select_action(self, state):
		best_uct = -100
		best_actions = []
		for i, a in enumerate(self.valid_moves[state]):
			Ns = self.Ns[state]
			uct = self.Q[state][a] + self.configs.cpuct * math.sqrt(Ns) / (1 + self.Nsa[state][a])
			if abs(uct - best_uct) < 1e-5:
				best_actions.append(a)
			elif uct > best_uct:
				best_uct = uct
				best_actions = [a]
		return random.choice(best_actions)

	def update(self, state, action, v):
		self.Q[state][action] = (self.Nsa[state][action] * self.Q[state][action] + v) / (self.Nsa[state][action] + 1)
		self.Ns[state] += 1
		self.Nsa[state][action] += 1
	
	def update_realizations(self, realization_key, realization_state, v):
		if realization_state in self.realizations_Vs[realization_key]:
			vv = self.realizations_Vs[realization_key][realization_state]
			n = self.realizations_Ns[realization_key][realization_state]
		else:
			vv = 0
			n = 0
   
		self.realizations_Vs[realization_key][realization_state] = (n * vv + v) / (n + 1)
		self.realizations_Ns[realization_key][realization_state] = n + 1

	def get_best_action(self, state):
		best_action = None
		best_v = -10000
		for a in self.valid_moves[state]:
			v = self.Q[state][a]
			if v > best_v:
				best_v = v
				best_action = a
		return best_action

class MCTSUserSimulator(UserSimulator):
	def __init__(self, system, messages, actions, configs, history_start_idx=0, simulator=None, model='gpt4') -> None:
		self.actions = actions
		self.configs = configs
		self.simulator = simulator
		self.mcts = None
		if self.simulator is not None:
			self.simulator.agent = self
		super().__init__(system, messages, history_start_idx, model=model)

	def chat(self, **kwargs):		
		best_action, utterance = self.get_action_utterance()
		return utterance

	def get_valid_actions(self):
		return self.actions
	
	def get_action_utterance(self):
		if self.configs.reuse_mcts and self.mcts is not None:
			mcts = self.mcts
		else:
			mcts = OpenLoopMCTS(self.configs, self.simulator)
			self.mcts = mcts

		game = self.game

		gamestate = self.game.gamestate()
		# replace all MCTSSimulator with StrategySimulator
		for i in range(len(gamestate.agents)):
			if isinstance(gamestate.agents[i], MCTSUserSimulator):
				gamestate.agents[i].chat = partial(StrategyUserSimulator.chat, gamestate.agents[i])

		t = time.time()
		for i in range(self.configs.num_simulations):
			mcts.search(gamestate)
			print(f'{i+1}/{self.configs.num_simulations} {time.time()-t}')

		for a in game.agents:
			a.game = game
		
		best_action = mcts.get_best_action(gamestate)
		best_realization = mcts.get_best_realization(gamestate, best_action)
		
		if best_realization is None:
			utterance = self.chat(action=best_action)
		else:
			utterance = best_realization.history[-1]
		
		for i in range(len(gamestate.agents)):
			if isinstance(gamestate.agents[i], MCTSUserSimulator):
				gamestate.agents[i].chat = partial(type(gamestate.agents[i]).chat, gamestate.agents[i])

		return best_action, utterance