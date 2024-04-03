import sys
import os

path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(1, path)

from typing import Type, Callable, Optional

import numpy as np

from reasoners import Reasoner, SearchAlgorithm
from reasoners.benchmark import BWEvaluator, blocksworld
from reasoners.algorithm import MCTS, MCTSAggregation

from world_model import BlocksWorldModel
from search_config import BWConfig
import json
def rap_bw(hf_path: str,
           prompt_path: str,
           search_algo: Type[SearchAlgorithm] = MCTS,
           data_path: str = 'data',
           resume: int = 0,
           depth_limit: int = 6,
           reward_alpha: float = 0.5,
           batch_size = 1,
           goal_reached_reward = 100,
           goal_reward_default = 0.,
           cum_reward: Callable[[list[float]], float] = sum,
           calc_q: Callable[[list[float]], float] = np.mean,
           log_dir: Optional[str] = None,
           disable_log: bool = False,
           domain_file: str = "examples/blocksworld/data/generated_domain.pddl",
           config_file: str = "",
           lm_plan_file: str = 'lm_plan.tmp',
           **search_algo_params):


    from reasoners import VLLMModel
    base_model = VLLMModel(model=hf_path)

    aggregator = MCTSAggregation(lambda x: x.history_actions, weight_policy='edge')
    
    prompt = json.load(open(prompt_path, 'r'))
    print(search_algo)

    search_algo_params |= {'cum_reward': cum_reward, 'calc_q': calc_q, "depth_limit": depth_limit, "disable_tqdm": False, 'aggregator': aggregator}
    world_model = BlocksWorldModel(base_model=base_model, prompt=prompt, batch_size=batch_size, max_steps=depth_limit)
    config = BWConfig(base_model=base_model, prompt=prompt, batch_size=batch_size,
                      reward_alpha=reward_alpha, goal_reached_reward=goal_reached_reward,
                      goal_reward_default=goal_reward_default)
    search_algo = MCTS(**search_algo_params)
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
    evaluator = BWEvaluator(config_file=config_file, domain_file=domain_file, data_path=data_path, init_prompt=prompt, disable_log=disable_log)
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(accuracy)

if __name__ == '__main__':
    import fire

    fire.Fire(rap_bw) # user will need to switch the model in the code
