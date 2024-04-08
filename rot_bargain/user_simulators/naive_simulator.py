from utils import *
import numpy as np
class UserSimulator:
    def __init__(self, system, bg_messages, history_start_idx=0, model='gpt4') -> None:
        self.system = system
        self.bg_messages = bg_messages
        self.history_start_idx = history_start_idx
        self.model=model
        self.game = None

    def chat(self, **kwargs):
        messages = self.prepare_messages()

        if self.model == 'gpt4':
            res = query_gpt4(messages)
        elif self.model == 'chatgpt':
            res = query_chatgpt(messages)
        elif self.model == 'mixtral':
            res = query_mixtral(messages)

        return res

    def get_valid_actions(self):
        return ['None']
    
    def predict(self):
        return np.array([1]), 0
    
    def prepare_messages(self):
        guidelines = getattr(self, 'guidelines', '')

        system_message = [{
            'role': 'system',
            'content': self.system + guidelines
        }]

        history = self.game.history[self.history_start_idx:]
        if len(self.bg_messages) == 0:
            user = 'assistant'
        else:
            user = self.bg_messages[-1]['role']

        if user == 'assistant':
            history_messages = [{
                'role': 'user' if i % 2 == 0 else 'assistant',
                'content': m
            } for i, m in enumerate(history)]
        else:
            history_messages = [{
                'role': 'user' if i % 2 == 1 else 'assistant',
                'content': m
            } for i, m in enumerate(history)]

        messages = system_message + self.bg_messages + history_messages

        return messages
