import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from core import *
from user_simulators import *
import json
import argparse

mcts_config = MCTSConfig()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--buyer', type=str)
    parser.add_argument('--seller', type=str)
    parser.add_argument('--n_iters', type=int, default=8)
    parser.add_argument('--panelty', type=float, default=0.0)
    parser.add_argument('--stage', type=str, default='evaluate')
    parser.add_argument('--prompt_path', type=str, default='none')

    args = parser.parse_args()
    
    if args.prompt_path == 'none':
        prompt = ''
    else:
        prompt = json.load(open(args.prompt_path))
        
    mcts_config.threshold_offset = 0.2
    mcts_config.reuse_mcts = True
    mcts_config.num_simulations = args.n_iters
    mcts_config.max_realizations = 1

    data = json.load(open('data/CraigslistBargain/dev-luis-post.json'))
        
    if args.stage == 'reflect':
        data = data[:2]
    else:
        data = data[2:20]

    stat = {'utility': 0, 'agreement': 0}

    for i, d in enumerate(data):
        buyer = UserSimulator('Now enter the role-playing mode. In the following conversation, you will play as a buyer in a price bargaining game.', [{
            'role': 'user',
            'content': f'You are the buyer who is trying to buy a product at the price of ${d["buyer-price"]}. The product information are described as follows: \n\nTitle:{d["title"]}\nDescription:\n{d["description"]}\nPlease make your response short and succinct.\n\nNow start the game.'
        }], model=args.buyer)

        seller = MCTSUserSimulator('Now enter the role-playing mode. In the following conversation, you will play as a seller in a price bargaining game.', [{
            'role': 'user',
            'content': f'You are the seller who is trying to sell a product at the price of ${d["seller-price"]}. The product information are described as follows: \n\nTitle:{d["title"]}\nDescription:\n{d["description"]}\nPlease make your response short and succinct.{prompt}\n\nNow start the game.\nHow much is this product?'
        },], history_start_idx=1, actions=[
            "Strategy 1: Emphasize Exclusivity\nHighlight the uniqueness and limited availability of the product to create a sense of urgency and justify its value.", 
            "Strategy 2: Payment Plans\nOffer flexible payment options or installment plans that enable the customer to purchase at the asking price but spread out the payment over time.", 
            "Strategy 3: Customer Loyalty\nConsider a small discount or added benefit for repeat customers, to reinforce loyalty and encourage future full-price purchases.", 
            "Strategy 4: Price Anchoring\nMention higher-priced comparable items first, making the asking price seem more reasonable by comparison.", 
            "Strategy 5: Create a Win-Win Situation\nFind out what's most valuable to the buyer that doesn't significantly affect your bottom line and leverage that in the negotiation.",             
        ], configs=mcts_config, simulator=BarginRollout(), model=args.seller)

        res =  f'Hi, its price is ${d["seller-price"]}.'

        history = [
            'How much is this product?',
            f'Hi, its price is ${d["seller-price"]}.'
        ]

        moderator = BarginingModerator(d, mode='seller', panelty=args.panelty)

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
            }, f, indent=4)

            seller.mcts.dump(f'{args.log_path}/{i}/mcts.pkl')

        os.system(f'python analyze_mcts_log.py --path {args.log_path}/{i}')
        stat['utility'] += res['utility']
        stat['agreement'] += res['agreement']
        
        print(res)
    
    stat['utility'] = stat['utility'] / stat['agreement'] if stat['agreement'] != 0 else 0
    stat['utility'] = stat['utility'] / len(data)
    stat['agreement'] = stat['agreement'] / len(data)
    
    with open(f'{args.log_path}/stat.json', 'w') as f:
        json.dump(stat, f, indent=4)
        
    print(stat)