import json

import utils.env_utils as env_utils
from utils.llm_utils import LLM
from utils.reflexion_utils import Reflexion
import utils.utils as utils

import logger.logger as logger
from babyai.levels import *

def run_reflexion(directory:str, trial:int, is_rewrite:bool):
    with open(f'{directory}/config.json') as f:
        config = json.load(f)
    log_path = f'{directory}/log_trial{trial}.json' 
    with open(log_path) as f:
        log = json.load(f)
    LLM.load(config)
    seed = utils.get_value(config, "env_fixed_seed", None)
    env = env_utils.make(config)
    obs, _ = env.reset(seed=seed)

    reflexion = Reflexion(env, obs, config)
    memory = None
    if trial >= 1:
        pre_trial = trial-1
        with open(f'{directory}/log_trial{pre_trial}.json') as f:
            pre_log = json.load(f)[1]
        memory = pre_log["memory"]
    history = log[1]["history"]
    subgoal_tree = log[1]["subgoal_tree"]
    for i in range(env.agent_num):
        if trial >= 1:
            reflexion.memories[i].set_dict(memory[i])
        reflexion.histories[i].set_dict(history[i])
        reflexion.subgoal_trees[i].set_dict(subgoal_tree[i])

    print("\n[info] running reflexion...")
    is_success = log[0]["steps"][-1]["is_success"]
    reason = log[0]["steps"][-1]["reason"]

    queries = reflexion.run(is_success, reason, config)
    if is_rewrite:
        memory_dict = [memory.get_dict() for memory in reflexion.memories]
        log[1]["reflexion_queries"] = queries
        log[1]["memory"] = memory_dict
        logger.output(log, log_path)
    else:
        print("----------------------------------------")
        #print(queries)
        for memory in reflexion.memories:
            print("----------------------------------------")
            print(memory.contents)
        print("----------------------------------------")

def main(directory, trial, is_rewrite=False):
    path = utils.search_directory_path(directory)
    run_reflexion(path, trial, is_rewrite)

if __name__ == "__main__":
    directory = "20241128212851_BlockedUnlock_simple"
    trial = 0
    is_rewrite = False
    main(directory, trial, is_rewrite)
