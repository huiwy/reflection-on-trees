import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from core import *
from user_simulators.naive_simulator import *
from user_simulators.strategy_simulator import *
from user_simulators.mcts_simulator import *
import json
import argparse

data = json.load(open('data/CraigslistBargain/dev-luis-post.json'))

stat = {'utility': 0, 'agreement': 0}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--buyer', type=str)
    parser.add_argument('--seller', type=str)
    parser.add_argument('--prompt_path', type=str, default='none')
    parser.add_argument('--stage', type=str, default='evaluate')
    
    args = parser.parse_args()
    
    if args.stage == 'reflect':
        data = data[:2]
    else:
        data = data[2:20]
    
    for i, d in enumerate(data):
        buyer = UserSimulator('You are GPT4, trained by OpenAI. Now enter role-playing mode. In the following conversation, you will play as a buyer in a price bargaining game. Please make your response short and succinct.', [{
            'role': 'user',
            'content': f'You are the buyer who is trying to buy a product at the price of ${d["buyer-price"]}. The product information are described as follows: \n\nTitle:{d["title"]}\nDescription:\n{d["description"]}\n\n\nNow start the game.'
        }], model=args.buyer)
        
        if args.prompt_path == 'none':
            seller = UserSimulator('You are GPT4, trained by OpenAI. Now enter role-playing mode. In the following conversation, you will play as a seller in a price bargaining game. Please make your response short and succinct.', [{
                'role': 'user',
                'content': f'You are the seller who is trying to sell a product at the price of ${d["seller-price"]}. The product information are described as follows: \n\nTitle:{d["title"]}\nDescription:\n{d["description"]}\n\nNow start the game.\nHow much is this product?'
            },], history_start_idx=1, model=args.seller)
            
        else:
            strategy = json.load(open(f'prompt/{args.prompt}'))

            seller = FixedStrategyUserSimulator('You are GPT4, trained by OpenAI. Now enter role-playing mode. In the following conversation, you will play as a seller in a price bargaining game. Please make your response short and succinct.', [{
                'role': 'user',
                'content': f'You are the seller who is trying to sell a product at the price of ${d["seller-price"]}. The product information are described as follows: \n\nTitle:{d["title"]}\nDescription:\n{d["description"]}\n\nNow start the game.\nHow much is this product?'
            }], history_start_idx=1, strategy=strategy, model=args.seller)
        

        res =  f'Hi, its price is ${d["seller-price"]}.'

        history = [
            'How much is this product?',
            f'Hi, its price is ${d["seller-price"]}.'
        ]

        moderator = BarginingModerator(d, mode='seller', panelty=0.0)

        game = DialogueGame(moderator, [buyer, seller], max_turn=8, last_message=res, history=history)
        
        seller.game = game
        buyer.game = game
        
        res = game.run()
        res = moderator.compute_result(res)
        
        import os
        os.makedirs(f'{args.log_path}/{i}', exist_ok=True)

        with open(f'{args.log_path}/{i}/game.json', 'w') as f:
            json.dump({
                'item': d['title'],
                'description': d['description'],
                'buyer-price': d['buyer-price'],
                'seller-price': d['seller-price'],
                'history': game.history,
                'utility': res['utility'],
                'agreement': res['agreement']
            }, f)
            
        stat['utility'] += res['utility']
        stat['agreement'] += res['agreement']
        
    
        print(res)
        
    stat['utility'] = stat['utility'] / stat['agreement'] if stat['agreement'] != 0 else 0
    stat['utility'] = stat['utility'] / len(data)
    stat['agreement'] = stat['agreement'] / len(data)
    
    with open(f'{args.log_path}/stat.json', 'w') as f:
        json.dump(stat, f)