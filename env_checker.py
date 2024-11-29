import utils.env_utils as env_utils
from PIL import Image 
import numpy as np
from babyai.levels import * # 各環境を登録するためにimportがが必要

seeds = range(51, 100)

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