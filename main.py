import json
from tqdm import tqdm

import gym
from babyai.levels import *

import policy

from executed_configs import configs
from logger.movie_maker import MovieMaker
from logger.logger import Logger

import utils.env_utils as env_utils
import utils.llm_utils as llm_utils
from utils.reflexion_utils import Reflexion

# ファイル名から設定を読み込む
def load_config(config_name:str):
    with open(f'./config/{config_name}.json') as f:
        config = json.load(f)
    hyperparams = config["hyperparam"]
    policy_options = config["policy_option"]
    params = {**hyperparams, **policy_options}
    params["config_name"] = config_name
    return params

# Reflexionを指定Trial分実行する
def run(config:dict):
    # ログの設定
    logger = Logger("./result_reflexion/" + config["policy_name"], "_" + config["config_name"])
    logger.output("config", config)

    # リアルタイムに環境を描画する処理
    def rendering(env:gym.Env):
        capture = MovieMaker(env, logger.path)  
        capture.render()
        capture.make("capture_realtime")

    # Reflexionに関する色々を初期化
    reflexion = Reflexion(config)

    # 指定された回数分エピソードを実行する
    for trial in tqdm(range(config["trial_count"])):
        reflexion.reset(config)

        # 環境やログなどの初期化
        env = gym.make(config["env_name"], render_mode="rgb_array")
        env.agent_num = config["agent_num"]
        movie_maker = MovieMaker(env, logger.path)
        movie_maker.reset()
        obs, _ = env.reset()
        log_steps = []

        is_success = False
        reason = ""

        # 指定ステップ数まで繰り返す
        for step in tqdm(range(config["max_step"])):
            # 環境の描画
            movie_maker.render()
            if config["realtime_rendering"] : rendering(env)

            # 現在の状態を履歴に追加
            obs_texts = env_utils.obs_to_str(env, obs, config)
            reflexion.add_histories("observation", obs_texts)

            # 行動を行う
            action, response = policy.get_action(env, reflexion, config)
            obs, reward, done, truncated, info =  env.step(action[0])

            # 終了状態と状態の評価を取得
            done, is_success, reason = env_utils.get_achievement_status(reward, done, step, config)

            # ログに関する処理
            log_steps.append({
                "step":step,
                "obs":obs,
                "reward":reward,
                "is_success":is_success,
                "reason":reason,
                "info":response
            })

            logger.clear()
            logger.append({
                "env_name" : config["env_name"],
                "steps" : log_steps,
            })
            logger.output(f"log_trial{trial}")

            logger.output(f"log_history_trial{trial}", [str(reflexion.histories[i]).split('\n') for i in range(env.agent_num)])
            
            movie_maker.make(f"capture_trial{trial}")

            # 終了していたら打ち切る
            if done: break

        # reflexionを実行
        print("run reflexion...")
        memory = reflexion.run(is_success, reason, config)

        # ログなどの処理
        logger.append({
            "memory" : memory,
        })
        logger.output(f"log_trial{trial}")
        movie_maker.render()
        movie_maker.make(f"capture_trial{trial}")

def main():
    # 指定した設定ファイルの数だけ連続で実行する
    for config_name in configs:
        config = load_config(config_name)
        llm_utils.load_llm(config) # 先にLLMをロードしておく
        print(f"------ execute {config_name} ------")
        run(config)

if __name__ == "__main__":
    main()