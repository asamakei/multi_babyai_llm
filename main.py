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
import utils.utils as utils

# ファイル名から設定を読み込む
def load_config(config_name:str):
    with open(f'./config/{config_name}.json') as f:
        config = json.load(f)
    # configを階層的に読み込む
    params = {}
    extends = utils.get_value(config, "extends", [])
    for name in extends:
        p = load_config(name)
        params = {**params, **p}
    hyperparams = utils.get_value(config, "hyperparam", {})
    policy_option = utils.get_value(config, "policy_option", {})
    params = {**params, **hyperparams, **policy_option}
    params["config_name"] = config_name
    return params

# Reflexionを指定Trial分実行する
def run(config:dict):
    # ログの設定
    logger = Logger("./result_reflexion/" + config["env_name"] + "/" + config["policy_name"], "_" + config["config_name"])
    logger.output("config", config)

    def make_history_log(env, reflexion:Reflexion):
        history = [str(reflexion.histories[i]).split('\n') for i in range(env.agent_num)]
        history.append("length: " + str(step+1))
        return history

    # Reflexionに関する色々を初期化
    reflexion = None

    # 乱数に関する初期化
    seed = utils.get_value(config, "env_fixed_seed", None)

    # 指定された回数分エピソードを実行する
    for trial in tqdm(range(config["trial_count"])):

        # 環境やログなどの初期化
        env = env_utils.make(config)
        obs, _ = env.reset(seed=seed)
        if reflexion is None:
            reflexion = Reflexion(env, config)
        else:
            reflexion.reset(env, config)

        movie_maker = MovieMaker(env, logger.path)
        log_steps = []

        is_success = False
        reason = ""

        # 指定ステップ数まで繰り返す
        for step in tqdm(range(config["max_step"])):
            # 環境の描画
            movie_maker.render()
            movie_maker.make(f"capture_trial{trial}")
            if config["realtime_rendering"] :
                movie_maker.make_last_frame(f"capture_realtime")

            # 現在の状態を履歴に追加
            obs_texts = env_utils.obs_to_str(env, obs, config)
            reflexion.add_histories("observation", [f"time {step}:"] * config["agent_num"])
            reflexion.add_histories("observation", obs_texts)

            # 行動を決定し実行する
            actions, response = policy.get_action(env, reflexion, config)
            if utils.get_value(config, "is_use_feedback", False):
                feedbacks = env_utils.get_feedbacks(env, obs, actions, config)
                reflexion.add_histories("observation", feedbacks)
            obs, reward, done, truncated, info =  env.step(actions)

            # 終了状態と状態の評価を取得
            done, is_success, reason = env_utils.get_achievement_status(reward, done, step, config)

            # ログに関する処理
            log_steps.append({
                "step":step,
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

            history = make_history_log(env, reflexion)
            logger.output(f"log_history_trial{trial}", history)

            # 終了していたら打ち切る
            if done: break

        # reflexionを実行
        print("\n[info] running reflexion...")
        memory, queries = reflexion.run(is_success, reason, config)

        # ログなどの処理
        history = make_history_log(env, reflexion)
        logger.output(f"log_history_trial{trial}", history)
        
        logger.append({"reflexion_queries":queries, "memory" : memory})
        logger.output(f"log_trial{trial}")

        logger.output(f"reflexion_backup", {"memory" : memory, "trial": trial})

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