import gym
from babyai.levels import *
from movie_maker import MovieMaker
import policy
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

    logger = Logger("./result_reflexion2/" + policy_option["policy_name"], "_" + config_name)

    env = gym.make(hyperparam["env_name"], render_mode="rgb_array")
    env.agent_num = 1
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

        obs, reward, done, truncated, info =  env.step(action[0])

        reason = "time up"
        is_success = False

        if reward < -10:
            reason = "falling to the cliff"
            done = True
        elif done:
            is_success = True

        #if reward > -10 and obs_pre == obs:
        #    reward = -10
        #policy.train([reward])

        log_steps.append({
            "step":step,
            "obs_pre":obs_pre,
            "obs_new":obs,
            "reward":reward,
            "done":done,
            "info":response
        })

        memory = []
        if done or (step == hyperparam["max_step"]-1):
            memory = policy.run_reflexion(is_success, reason, policy_option)
            if len(memory) > 0:
                print(memory[-1])

        logger.clear()
        logger.append({
            "env_name" : hyperparam["env_name"],
            "steps" : log_steps,
            "memory" : memory
        })
        logger.output("log")
        movie_maker.make()

        if done:
            break
        
    #policy.save_model(logger.path, policy_option)

    movie_maker.render()
    movie_maker.make()
    # TODO!を潰す

for config in configs:
    run(config)