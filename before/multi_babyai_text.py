import gym
from babyai.levels import *
from movie_maker import MovieMaker
import before.policy as policy
from logger import Logger
import json
from tqdm import tqdm
from executed_configs import configs
import random

def run(config_name:str):
    print(f"------ execute {config_name} ------")

    with open(f'./config/{config_name}.json') as f:
        config = json.load(f)

    hyperparam = config["hyperparam"]
    policy_option = config["policy_option"]

    logger = Logger("./result_dqn/" + policy_option["policy_name"], "_" + config_name)

    env = gym.make(hyperparam["env_name"], agent_num = hyperparam["agent_num"])
    initial_grid = env.grid.encode().tolist()
    movie_maker = MovieMaker(env, logger.path)
    movie_maker.reset()
    obs, _ = env.reset()

    def rendering(env:gym.Env):
        capture = MovieMaker(env, logger.path)  
        capture.render()
        capture.make("tmp_capture")

    log_steps = []
    done = False
    step = 0

    policy_option["pre_action"] = []
    policy_option["pre_reason"] = []
    policy_option["pre_plan"] = []

    policy.init(hyperparam, policy_option)

    for step in tqdm(range(hyperparam["max_step"])):
        movie_maker.render()
        if hyperparam["realtime_rendering"] : rendering(env)

        action, response = policy.get_action(env, obs, policy_option)
        
        if action[0] < 0 or action[0] > 6 : break
        policy_option["pre_action"].append(action)
        policy_option["pre_reason"].append(response["reasons"])
        obs_pre = obs

        obs, reward, done, truncated, info =  env.step(action)
        reward = [1 if done else -1 for _ in range(hyperparam["agent_num"])]
        #reward = [random.randint(-100,100) / 100 for i in range(hyperparam["agent_num"])]
        policy.train(reward)

        log_steps.append({
            "step":step,
            "obs_pre":obs_pre,
            "obs_new":obs,
            "reward":reward,
            "done":done,
            "info":response
        })

        logger.clear()
        logger.append({
            "env_name" : hyperparam["env_name"],
            "grid" : initial_grid,
            "steps" : log_steps
        })
        logger.output("log")
        movie_maker.make()

        if done:
            print("done!")
            break
        
    policy.save_model(logger.path, policy_option)

    movie_maker.render()
    movie_maker.make()
    # TODO!を潰す

for config in configs:
    run(config)