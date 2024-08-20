import gym
from babyai.levels import *
from movie_maker import MovieMaker
import policy
from logger import Logger
import json
from tqdm import tqdm

def run(config_name:str):
    with open(f'./config/{config_name}.json') as f:
        config = json.load(f)

    hyperparam = config["hyperparam"]
    policy_option = config["policy_option"]

    logger = Logger("./result/" + policy_option["policy_name"])

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
    policy_option["pre_plan"] = []

    for step in tqdm(range(hyperparam["max_step"])):
        movie_maker.render()
        if hyperparam["realtime_rendering"] : rendering(env)

        action, response = policy.get_action(env, obs, policy_option)
        
        if action[0] < 0 or action[0] > 6 : break
        policy_option["pre_action"].append(action)
        obs_pre = obs

        obs, reward, done, truncated, info =  env.step(action)

        log_steps.append({
            "step":step,
            "obs_pre":obs_pre,
            "obs_new":obs,
            "reward":reward,
            "done":done,
            "info":response
        })

        if done:
            print("done!")
            break

        logger.clear()
        logger.append({
            "env_name" : hyperparam["env_name"],
            "grid" : initial_grid,
            "steps" : log_steps
        })
        logger.output("log")
        movie_maker.make()

    movie_maker.render()
    movie_maker.make()
    # TODO!を潰す

configs = [
    #"gpt4",
    #"debug",
    #"llama3_BlockedUnlock_simple",
    #"llama3_BlockedUnlock_message_twoside",
    #"llama3_BlockedUnlock_message_conversation",
    #"llama3_RoomS20_message_conversation",
    "llama3_RoomS20_simple",
    #"llama3_BlockedUnlock_message_conversation_3",
    #"llama3_BlockedUnlock_message_oneside",
    #"llama3_BlockedUnlock_message_twoside",
    #"llama3_BlockedUnlock_message_oneside",
    #"llama3_GoToObj_message_twoside",
    #"llama3_GoToObj_message_twoside",
    #"llama3_GoToObj_message_twoside",
    #"llama3_GoToObj_simple",
    #"llama3_BlockedUnlock_simple",
]

for config in configs:
    print(f"------ execute {config} ------")
    run(config)