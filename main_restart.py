import os
from main import *
import utils.utils as utils
from utils.llm_utils import LLM
from run_reflexion import main as run_reflexion

def main(directory:str, trial:int=-1, is_run_reflexion:bool=False):
    path = utils.search_directory_path(directory)    
    if path is None: return
    with open(f'{path}/config.json') as f:
        config = json.load(f)

    LLM.load(config) # 先にLLMをロードしておく
    if utils.get_value(config, "is_use_embedding_model", False):
        Embedder.load(config)
        
    config_name = config["config_name"]
    logger = Logger(path, config_name, False)

    seed = utils.get_value(config, "env_fixed_seed", None)
    env = env_utils.make(config)
    obs, _ = env.reset(seed=seed)

    reflexion = Reflexion(env, obs, config)

    print(f"------ execute {config_name} ------")

    # trial番号の計算
    if trial < 0:
        backup = f"{path}/reflexion_backup.json"
        if os.path.exists(backup):
            with open(backup) as f:
                trial = json.load(f)["trial"] + 1
        else:
            trial = 0

    # メモリーの読み込み
    if trial >= 1:
        pre_trial = trial-1
        if is_run_reflexion:
            # 前回のReflexionから始める
            run_reflexion(directory, trial-1, True)
        with open(f'{path}/log_trial{pre_trial}.json') as f:
            pre_log = json.load(f)[1]
        for i in range(env.agent_num):
            reflexion.memories[i].set_dict(pre_log["memory"][i])

    run(logger, reflexion, trial, config)

# 処理を再開する
if __name__ == "__main__":
    directory = "20241204000001_Failed"
    trial = 1
    is_run_reflexion = False
    main(directory, trial, is_run_reflexion)