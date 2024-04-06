from utils import *
import json
import random
import re
import copy
    
class DialogueGame:
    def __init__(self, moderator, agents, max_turn, last_message, history) -> None:
        self.moderator = moderator
        self.agents = agents
        self.max_turn = max_turn
        self.current_turn = 0
        self.history = history
        self.last_message = last_message
        
    def step(self, **kwargs):
        agent = self.agents[self.current_turn % 2]
        self.last_message = agent.chat(**kwargs)
        self.history.append(self.last_message)

        if self.current_turn % 2 == 1:
            finished, res = self.moderator.generate_result(self.history)
        else:
            finished, res = False, 0

        self.current_turn += 1 
        if self.current_turn >= self.max_turn:
            return True, res
        
        if finished:
            return True, res
        
        return False, None
    
    def step_and_state(self, **kwargs):
        finished, res = self.step(**kwargs)
        if finished:    
            reward = self.compute_reward(res)
        else:
            reward = 0
        return finished, reward, self.gamestate()        

    def compute_reward(self, res):
        return self.moderator.compute_result(res)['utility']

    def get_valid_actions(self):
        agent = self.agents[self.current_turn % 2]
        return agent.get_valid_actions()
    
    def predict(self):
        agent = self.agents[self.current_turn % 2]
        return agent.predict()

    def run(self):
        while True:
            finished, res = self.step()
            if finished:
                break
        return res

    @classmethod
    def resume_gamestate(cls, gamestate):
        game = cls(gamestate.moderator, gamestate.agents, gamestate.max_turn, gamestate.history[-1], gamestate.history)
        game.current_turn = gamestate.current_turn

        for agent in game.agents:
            agent.game = game

        new_game = copy.copy(game)
        new_game.history = game.history[:]
        new_game.agents = gamestate.agents

        for a in new_game.agents:
            a.game = new_game

        return new_game
    
    def gamestate(self):
        return DialogueGameState(self.history[:], self.moderator, self.agents, self.max_turn, self.current_turn)

class DialogueGameState:
    def __init__(self, history, moderator, agents, max_turn, current_turn) -> None:
        self.history = history
        self.agents = agents
        self.max_turn = max_turn
        self.current_turn = current_turn
        self.moderator = moderator

    def __str__(self):
        return json.dumps(self.history)
    
    def __hash__(self):
        return hash(str(self))

    def __eq__(self, o: object) -> bool:
        return str(self) == str(o)

    def copy(self):
        gamestates = copy.copy(self)
        gamestates.history = self.history[:]
        return gamestates

class BarginRollout:
    def __init__(self):
        self.agent = None
    
    def simulate(self, state):
        history = state.history
        history = '\n'.join(['{}: {}'.format('Buyer' if i % 2 == 0 else 'Seller', m) for i, m in enumerate(history)])

        messages = [
            {
                "role": "system",
                "content": "You are GPT4 developed by OpenAI. Now please act as a bargining dialogue simulator to simulate a bargining where the seller and buyer are maximizing their profit."
            },
            {
                "role": "user",
                "content": f"Dialogue History:\n{history}\n\nPlease complete the dialogue. Note both the seller and buyer are not require to reach an agreement. After the dialogue is completed, show whether this deal is made in [YES] or [NO]. If the answer is [YES], please also show the price in <$xx>. Your completion should start with Buyer. I will tip you $100 if you complete this task successfully."
            }
        ]

        for i in range(10):
            try:
                dialogue_history = None
                price = 0
                if self.agent.model == 'gpt4':
                    res = query_gpt4(messages)
                elif self.agent.model == 'chatgpt':
                    res = query_chatgpt(messages)
                else:
                    res = query_mixtral(messages)

                if res.startswith('Seller'):
                    continue

                dialogue_history = res.split('[YES]')[0].split('[NO]')[0].strip()
                if '[YES]' in res.upper():
                    price = re.findall('<\$[0-9,\.]+>', res)[0].replace(',', '')
                    # find the number in the price and change it into float
                    price = float(re.findall('[0-9\.]+', price)[0])
                    break
                elif '[NO]' in res.upper():
                    break
            except:
                pass
        if i == 10:
            return '', 0
        
        r = self.agent.game.moderator.compute_result(price)['utility']

        return dialogue_history, r

        
class BarginingModerator():
    def __init__(self, data, mode, panelty) -> None:
        self.data = data
        self.mode = mode
        self.panelty = panelty

    def generate_result(self, history):
        history = history
        history = '\n'.join(['{}: {}'.format('Buyer' if i % 2 == 0 else 'Seller', m) for i, m in enumerate(history)])
        
        messages_2 = [
            {
                "role": "system",
                "content": "You are GPT4, a large language model trained by OpenAI. Now please act as a bargining system to determine whether the seller and buyer has reached an agreement."
            },
            {
                "role": "user",
                "content": f"Dialogue History:\n{history}\n\n\nPlease first summarize the buyer and seller's opinion and determine whether the buyer and seller has reach an agreement in [YES] or [NO]. If an agreement is reached, please also provide the price of the item in <$XX>. You should follow this format:\nReview: ...\nResult: [YES]/[NO] <$...>"
            }
        ]

        res = query_gpt4(messages_2)

        finished = 'YES' in res
        
        if finished:
            try:
                price = re.findall('<\$[0-9,\.]+>', res)[0].replace(',', '')
                # find the number in the price and change it into float
                price = float(re.findall('[0-9\.]+', price)[0])
                return finished, price
            except:
                return False, 0
        return finished, 0
    
    def compute_result(self, price):
        seller_price = self.data['seller-price']
        buyer_price = self.data['buyer-price']

        return {
            'utility': (2*price - seller_price - buyer_price) / (seller_price - buyer_price) if price != 0 else self.panelty,
            'agreement': 1 if price != 0 else 0,
        }