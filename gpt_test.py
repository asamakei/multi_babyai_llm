import os
import re
import json
import datetime
import random

from tqdm import trange
import gymnasium as gym
from openai import OpenAI
from matplotlib import pyplot as plt

from movie_maker import MovieMaker
import KEY

# 各種設定
hyperparams = {
    "directory":"./log/gpt/",
    "model": "gpt-4",#"gpt-3.5-turbo",
    "api_key": KEY.openai_api_key,
    "env": "CliffWalking-v0",
    "episodes": 1,
    "max_steps": 20,
}

# 新たなフォルダを作成
dt_now = datetime.datetime.now()
dt_str = dt_now.strftime('%Y%m%d%H%M%S')
hyperparams["directory"] = hyperparams["directory"] + dt_str + "/"
os.mkdir(hyperparams["directory"])

# プロンプトに関する文字列処理
actions = ["Action0", "Action1", "Action2", "Action3"]
system_prompt = """You are an excellent strategist. You are successful if you reach goal by moving on grid without falling off cliff. You cannot move to wall."""
format_prompt = f"""You can act "{actions[0]}" to move up, "{actions[1]}" to move right, "{actions[2]}" to move down, or "{actions[3]}" to move left. Which is the best action in "{actions[0]}", "{actions[1]}", "{actions[2]}" or "{actions[3]}"? Output only result."""

def obs_to_text(obs:int):
    object_name = ["wall", "floor", "cliff", "goal"]
    field = [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,2,2,2,2,2,2,2,2,2,2,3,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ]
    r = obs // 12 + 1
    c = obs % 12 + 1
    up = f"- {object_name[field[r-1][c]]} in 1 step your up"
    down = f"- {object_name[field[r+1][c]]} in 1 step your down"
    right = f"- {object_name[field[r][c+1]]} in 1 step your right"
    left = f"- {object_name[field[r][c-1]]} in 1 step your left"
    if c == 12 and r == 3:
        goal = ""
    else:
        s1 = "s" if 12 - c >=2 else ""
        s2 = "s" if 4 - r >=2 else ""
        if 12 - c == 0: goal = f"- goal in {4 - r} step{s2} down"
        elif 4 - r == 0: goal = f"- goal in {12 - c} step{s1} right"
        else: goal = f"- goal in {12 - c} step{s1} right and {4 - r} step{s2} down"
    return f"{up}\n{down}\n{right}\n{left}\n{goal}"

# 行動の取得
client = OpenAI(api_key = hyperparams["api_key"])
def llm(messages:list):
    messages_format = [{"role": message[0], "content": message[1]} for message in messages]
    response = client.chat.completions.create(
        model = hyperparams["model"],
        messages = messages_format
    )
    text = response.choices[0].message.content
    return text, response

def get_random_action():
    a = actions[random.randint(0,len(actions)-1)]
    return a, ""

def text_to_action(text:str):
    for i, action in enumerate(actions):
        match = re.compile(action).search(text)
        if match: return i
    return -1

# メイン処理

# 初期化
env = gym.make(hyperparams["env"], render_mode="rgb_array")
movie_maker = MovieMaker(env, hyperparams["directory"])
rewards = []
log = []

# 指定数のエピソードを実行
for episode in trange(hyperparams["episodes"]):
    # 環境や記録の初期化
    obs, info = env.reset()
    done = False
    sum_reward = 0
    step = 0
    movie_maker.reset()

    log_steps = []

    while not done and step < hyperparams["max_steps"]:
        # クエリの生成
        sight_prompt = obs_to_text(obs)
        messages = [
            ("system", system_prompt),
            ("system", sight_prompt),
            ("system", format_prompt),
        ]

        # 行動の取得
        action_text, response = llm(messages)
        action = text_to_action(action_text)
        if action == -1: continue

        # 環境に反映
        obs_pre = obs
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 記録処理
        log_steps.append({"step":step, "obs_pre":obs_pre, "messages":messages, "response":str(response), "action":action, "obs_new":obs, "reward":reward, "done":done})
        sum_reward += float(reward)
        step += 1
        movie_maker.render()
        
    log.append({"episode":episode, "steps":log_steps})
    rewards.append(sum_reward)
    movie_maker.make(f"cliff_{episode}")

    with open(hyperparams["directory"] + "log.json", 'w') as f:
        json.dump(log, f, indent=1)

plt.plot(range(len(rewards)), rewards)
plt.savefig(hyperparams["directory"] + "rewards.png")