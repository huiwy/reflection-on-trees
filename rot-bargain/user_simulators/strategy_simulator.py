from .naive_simulator import UserSimulator
from utils import *
import random

class StrategyUserSimulator(UserSimulator):
    def __init__(self, system, messages, actions, configs, history_start_idx=0, model='gpt4') -> None:
        self.actions = actions
        self.configs = configs

        super().__init__(system, messages, history_start_idx, model=model)

    def chat(self, **kwargs):
        messages = self.prepare_messages()

        if 'action' in kwargs:
            action = kwargs['action']
        else:
            action = self.get_action()

        if not action == '':
            messages = messages[:-1] + [{
                'role': 'user',
                'content': f'Please reply with the following strategy: {action}. ' + messages[-1]['content'] 
            }]

        if self.model == 'gpt4':
            res = query_gpt4(messages)
        elif self.model == 'chatgpt':
            res = query_chatgpt(messages)
        else:
            res = query_mixtral(messages)
            
        return res

    def get_action(self):
        return random.choice(self.actions)

    def get_valid_actions(self):
        return self.actions


class FixedStrategyUserSimulator(UserSimulator):
    def __init__(self, system, messages, history_start_idx=0, strategy='', model='gpt4') -> None:
        self.strategy = strategy

        super().__init__(system, messages, history_start_idx, model=model)

    def chat(self, **kwargs):
        messages = self.prepare_messages()

        new_messages = messages[:-1] + [{
            'role': 'user',
            'content': f'Please reply with the following strategy: {self.strategy}. ' + messages[-1]['content'] 
        }]

        if self.model == 'gpt4':
            res = query_gpt4(new_messages)
        elif self.model == 'chatgpt':
            res = query_chatgpt(new_messages)
        elif self.model == 'mistral':
            res = query_mixtral(new_messages)

        return res

    def get_action(self):
        return random.choice(self.actions)

    def get_valid_actions(self):
        return self.actions