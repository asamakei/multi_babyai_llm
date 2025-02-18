import utils.env_utils as env_utils
from PIL import Image 
import numpy as np
from babyai.levels import * # 各環境を登録するためにimportがが必要

def generate_init_images():
    seeds = range(0, 100)

    for seed in seeds:
        env = env_utils.make({
            "env_name": "BabyAI-BlockedUnlockPickup-v0",
            "agent_num": 2,
        })
        obs, _ = env.reset(seed=seed)
        rgb = env.render()
        rgb = np.array(Image.fromarray(np.array(rgb).astype(np.uint8)))
        image = Image.fromarray(rgb.astype('uint8')).convert('RGB')
        image.save(f"env_check/seed{seed}.png")

def prompt_test():
    config = {
        "env_name": "BabyAI-BlockedUnlockPickup-v0",
        "agent_num": 1,
    }
    env = env_utils.make(config)
    obs, _ = env.reset(seed=14)
    print(env_utils.get_init_subgoal_instr(env, "test", 0, config))

if __name__ == "__main__":
    generate_init_images()