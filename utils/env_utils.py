import re
import gym
from utils.utils import get_value
from gym_minigrid.minigrid import (
    IDX_TO_OBJECT,
    IDX_TO_COLOR,
)
from utils.llm_utils import get_image_token

CLIFF = "Cliff"
BABY = "Baby"

# 環境を作成する
def make(params:dict):
    env_name = params["env_name"]
    if CLIFF in env_name:
        env = gym.make(env_name, render_mode="rgb_array")
    elif BABY in env_name:
        env = gym.make(env_name, agent_num = params["agent_num"])

    env.agent_num = params["agent_num"]
    return env

# エージェントの名前を返す
def get_agent_name(agent_id:int, params = {}):
    return f"agent{agent_id}" if agent_id >= 0 else ""

# 行動の名前の取得
def get_actions_str(env_name):
    if CLIFF in env_name:
        return ["move north","move east","move south","move west"]
    elif BABY in env_name:
        return ["turn left","turn right","go forward","pick up the item in forward","drop the carrying item to forward","open the door in forward"]

# 行動の名前の略称を取得
def get_brief_actions_str(env_name):
    if CLIFF in env_name:
        return ["north","east","south","west"]
    elif BABY in env_name:
        return ["left","right","forward","pick up","drop","open"]

# 行動の名前をカンマ区切りで連結したものを取得(プロンプト用)
def get_actions_joined_str(env_name, last_word=""):
    actions_str = get_actions_str(env_name)
    actions_str = [f"'{action}'" for action in actions_str]
    if last_word == "" or len(actions_str) <= 1:
        actions_joined = ", ".join(actions_str)
    else:
        actions_joined = ", ".join(actions_str[:-1]) + f" {last_word} " + actions_str[-1]
    return actions_joined

# プロンプトに使用する問題とタスクの説明文を返す
def get_base_task_prompt(env, params):
    env_name = params["env_name"]
    actions_joined = get_actions_joined_str(env_name, "or")
    action_prompt = f"Each step, You must select actions in {actions_joined}."
    if CLIFF in env_name:
        sentences = []
        sentences.append("Interact with an grid world environment to solve a task.")
        sentences.append(action_prompt)
        base_prompt = " ".join(sentences)
        task_prompt = "Reach the goal as soon as possible."
    elif BABY in env_name:
        right = env.width - 1
        bottom = env.height - 1
        sentences = []
        sentences.append("Interact with an grid world environment to solve a task.")

        # additional
        sentences.append(f"The grid (0,0) is at the northwest end, ({right}, 0) is northeast, (0, {bottom}) is southwest, ({right}, {bottom}) is southeast.")
        sentences.append(f"You cannot go through objects.")

        sentences.append(action_prompt)
        base_prompt = " ".join(sentences)
        task_prompt = 'Achieve the "mission" as soon as possible.'
    return base_prompt, task_prompt

def get_image_explain(agent_id:int, params):
    is_use_vision = get_value(params,"is_use_vision",False)
    if not is_use_vision: return ""

    image_token = get_image_token()
    env_name = params["env_name"]
    if CLIFF in env_name:
        return f"{image_token} This image is your view of grid world. You are the character with the hat."
    elif BABY in env_name:
        color = IDX_TO_COLOR[agent_id]
        return f"{image_token} This image is your view of grid world. You can observe only your forward. You are {color} triangle. "
    return f"{image_token}"

# Reflexionを行う時のプロンプトに記述する指示文を返す
def get_reflexion_prompt(reason:str, agent_id:int, params = {}):
    agent_name = get_agent_name(agent_id)
    profile = f"You are {agent_name}. " if agent_id >= 0 else ""
    image_explain = get_image_explain(agent_id, params)
    prompt = profile + image_explain + f"You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task because of {reason}. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Output only your plan in 3 to 5 sentences. \nGive your plan:"
    #prompt = profile + image_explain + f"You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task because of {reason}. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan:"
    return prompt

# 行動を生成する時のプロンプトに記述する指示文を返す
def get_action_prompt(env_name:str, agent_id:int, params = {}):
    agent_name = get_agent_name(agent_id)
    profile = f"You are {agent_name}. " if agent_id >= 0 else ""
    image_explain = get_image_explain(agent_id, params)
    actions_joined = get_actions_joined_str(env_name, "or")
    prompt = profile + image_explain + f"What is the best action to achieve your task in {actions_joined}? Output only result."
    return prompt

# メッセージを生成する時のプロンプトに記述する指示文を返す
def get_communication_prompt(label:str, agent_id:int, targets:str, params={}):
    targets_str = " and ".join(targets)
    agent_name = get_agent_name(agent_id)
    profile = f"You are {agent_name}. " if agent_id >= 0 else ""
    image_explain = get_image_explain(agent_id, params)
    if label == "message":
        prompt = profile + image_explain + f'You are in a cooperative relationship with {targets_str}. The mission has not yet been accomplished. To achieve mission, Output only message to {targets_str} in 2 to 5 sentences.'
        #prompt = f'You are in a cooperative relationship with {targets_str}. To achieve mission, Output only message to {targets_str}.'
    elif label == "conversation":
        prompt = profile + image_explain + f'You are in a cooperative relationship with {targets_str}, and discussing to decide next action. The mission has not yet been accomplished. To achieve mission, output only reply message to them in 1 to 3 sentences.'
        #prompt = f'You are in a cooperative relationship with {targets_str}, and discussing to decide next action. To achieve mission, output only reply message to them.'
    return prompt

# 終了判定や状態の評価を返す
def get_achievement_status(reward, done, step, params):
    env_name = params["env_name"]
    max_step = params["max_step"]
    
    reason = ""
    is_success = False
    if CLIFF in env_name:
        if reward < -10:
            reason = "falling to the cliff"
            done = True
        elif done:
            reason = "reaching the goal"
            is_success = True
        elif step == max_step-1:
            reason = "running out of time"
            done = True
        return done, is_success, reason
    elif BABY in env_name:
        if done:
            reason = "achieve the mission"
            is_success = True
        elif step == max_step-1:
            reason = "running out of time"
            done = True
            is_success = False
        return done, is_success, reason

    return False, False, ""

# 観測情報を文字列に変換する
def obs_to_str(env, observation, params) -> list[str]:
    env_name = params["env_name"]
    if CLIFF in env_name:
        return obs_to_str_cliff(observation)
    elif BABY in env_name:
        return obs_to_str_baby(env, observation, params)

# CliffWalkingの変換処理
def obs_to_str_cliff(obs:int) -> list[str]:
    object_name = ["Wall", "Nothing", "Cliff", "Goal"]
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
    north = f"Your north is {object_name[field[r-1][c]]}."
    south = f"Your south is {object_name[field[r+1][c]]}."
    east = f"Your east is {object_name[field[r][c+1]]}."
    west = f"Your west is {object_name[field[r][c-1]]}."
    row_s = "s" if 4 - r > 1 else ""
    col_s = "s" if 12 - c > 1 else ""
    if 12 - c == 0:
        goal = f"Goal is in {4 - r} step{row_s} to your south."
    elif 4 - r == 0:
        goal = f"Goal is in {12 - c} step{col_s} to your east."
    else:
        goal = f"Goal is in {12 - c} step{col_s} to your east and {4 - r} step{row_s} to your south."

    return [f"{north} {south} {east} {west} {goal}"]

# BabyAIの変換処理
def obs_to_str_baby(env, observations, options:dict={}) -> list[str]:
    directions_str = ["east", "south", "west", "north"]
    def direction_to_str(direction:int):
        return directions_str[direction]
    def object_to_text(obj):
        obj_id = obj[0]
        obj_name = IDX_TO_OBJECT[obj[0]]
        obj_color = IDX_TO_COLOR[obj[1]]
        state = obj[2]
        result = ""
        if obj_id == 2: # Wall
            result = obj_name
        elif obj_id == 4: # Door
            if state == 0:
                result = f"{obj_color} open {obj_name}"
            elif state == 1:
                result = f"{obj_color} closed {obj_name}"
            elif state == 2:
                result = f"{obj_color} locked {obj_name}"
        elif obj_id == 5: # Key
            result = f"{obj_color} {obj_name}"
        elif obj_id == 6: # Ball
            result = f"{obj_color} {obj_name}"
        elif obj_id == 7: # Box
            result = f"{obj_color} {obj_name}"
        elif obj_id == 8: # Goal
            result = obj_name
        elif obj_id == 9: # Lava
            result = obj_name
        elif obj_id >= 10: # Agent_n
            if get_value(options, "sight_include_agents", False): result = ""
            else:
                target_agent_id = obj_id - 10
                direction = observations[target_agent_id]["direction"]
                direction_str = direction_to_str(direction)
                result = f"{obj_name} facing {direction_str}"
        return result
    
    def position_to_text_euler(p:tuple[int,int]):
        return f"({p[0]}, {p[1]})"
    
    def position_to_text(p:tuple[int,int], agent_dir:int):
        dirs = directions_str[agent_dir:] + directions_str[:agent_dir]

        positions = []
        if p[0] != 0:
            is_one = p[0] == 1 or p[0] == -1
            is_forward = p[0] > 0
            s = "" if is_one else "s"
            dir = dirs[0] if is_forward else dirs[2]
            positions.append(f"{abs(p[0])} step{s} {dir}")
        if p[1] != 0:
            is_one = p[1] == 1 or p[1] == -1
            is_right = p[1] > 0
            s = "" if is_one else "s"
            dir = dirs[1] if is_right else dirs[3]
            positions.append(f"{abs(p[1])} step{s} {dir}")
        result = " and ".join(positions)
        return result
    
    def local_to_world_pos(pos, dir, pivot) -> tuple[int,int]:
        pos = (-pos[1], pos[0])
        if dir == 0: # east
            pos = (pos[1], -pos[0])
        elif dir == 1: # south
            pos = (pos[0], pos[1])
        elif dir == 2: # west
            pos = (-pos[1], pos[0])
        elif dir == 3: # north
            pos = (-pos[0], -pos[1])
        return (pos[0]+pivot[0], pos[1]+pivot[1])
    
    def one_agent(obs, agent_id):
        obj_infos = []
        carrying_info = ""
        image = obs["image"]
        shape = image.shape
        height, width = shape[0], shape[1]
        direction = obs['direction']
        self_pos = env.agents_pos[agent_id]

        for r in range(height):
            for c in range(width):
                local_pos = (height - r - 1, c - width // 2)

                obj = image[c][r]
                if obj[0] in (0,1,3): continue # 不可視, 通常床, 空白 の場合は飛ばす
                obj_name = object_to_text(obj)
                if obj_name == "": continue

                if local_pos == (0, 0) and len(obj_name) > 0:
                    carrying_info = obj_name
                    continue

                # 相対位置を返す
                #obj_pos = position_to_text(local_pos, direction)

                # 絶対位置を返す
                world_pos = local_to_world_pos(local_pos, direction, self_pos)
                obj_pos = position_to_text_euler(world_pos)

                if local_pos == (1, 0):# 正面の時の例外
                    obj_info = f"{obj_name} is in your forward, at {obj_pos}."
                else:
                    obj_info = f"{obj_name} is at {obj_pos}."
                obj_info = obj_info[:1].upper() + obj_info[1:]
                obj_infos.append(obj_info)

        if get_value(options, "sight_include_agents", False):
            for target_id in range(env.agent_num):
                if target_id == agent_id: continue
                target_pos = env.agents_pos[target_id]

                # 相対位置
                # diff = self_pos[1] - target_pos[1], target_pos[0] - self_pos[0] # y, x
                # dir = obs['direction']
                # if dir == 0: diff = (diff[1], -diff[0]) # east
                # elif dir == 1: diff = (-diff[0], -diff[1]) # south
                # elif dir == 2: diff = (-diff[1], diff[0]) # west
                # elif dir == 3: diff = (diff[0], diff[1]) # north

                # obj_pos = position_to_text(diff, direction)
                
                # 絶対位置
                obj_pos = target_pos

                target_dir = observations[target_id]["direction"]
                target_dir_str = direction_to_str(target_dir)

                obj_name = f"agent{target_id} facing {target_dir_str}"
                obj_info = f"{obj_name} is at {obj_pos}."
                obj_info = obj_info[:1].upper() + obj_info[1:]
                obj_infos.append(obj_info)

        direction = direction_to_str(obs['direction'])
        self_pos_str = position_to_text_euler(self_pos)
        mission = obs['mission']

        sentences = []
        sentences.append(f'Your mission is "{mission}".') # 目標
        sentences.append(f'You are facing {direction}.') # 方角
        sentences.append(f'You are at {self_pos_str}.') # 絶対位置
        if len(carrying_info) > 0:
            sentences.append(f'You are carrying {carrying_info}.')
        else:
            sentences.append(f'You have no item.')
        sentences.extend(obj_infos)
        result = ' '.join(sentences)
        return result

    return [one_agent(obs, agent_id) for agent_id, obs in enumerate(observations)]

# 文字列を行動IDに変換する
def str_to_action(text:str, params):
    env_name = params["env_name"]
    text = text.lower()
    # 正式な行動名とマッチさせる
    actions_str = get_actions_str(env_name)
    action_to_idx = {actions_str[i]:i for i in range(len(actions_str))}
    for action, value in action_to_idx.items():
        match = re.compile(action.lower()).search(text)
        if match: return value

    # 簡易的な行動名とマッチさせる
    brief_actions_str = get_actions_str(env_name)
    brief_action_to_idx = {actions_str[i]:i for i in range(len(brief_actions_str))}
    for action, value in brief_action_to_idx.items():
        match = re.compile(action.lower()).search(text)
        if match: return value

    return 0
      
# 行動IDを文字列に変換する
def action_to_str(action_idx:int, params):
    env_name = params["env_name"]
    actions = get_actions_str(env_name)
    return actions[action_idx]