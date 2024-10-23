import re
from gym_minigrid.minigrid import (
    IDX_TO_OBJECT,
    IDX_TO_COLOR,
)

CLIFF = "Cliff"
BABY = "Baby"

# エージェントの名前を返す
def get_agent_name(agent_id:int, params = {}):
    return f"agent{agent_id}"

# プロンプトに使用する問題とタスクの説明文を返す
def get_base_task_prompt(params):
    env_name = params["env_name"]
    if CLIFF in env_name:
        base_prompt = "Interact with an grid world environment to solve a task."
        task_prompt = "Reach the goal as soon as possible."
    elif BABY in env_name:
        base_prompt = "Interact with an grid world environment to solve a task."
        task_prompt = "Achieve the 'mission' as soon as possible."
    return base_prompt, task_prompt

# Reflexionを行う時のプロンプトに記述する指示文を返す
def get_reflexion_prompt(reason, params):
    env_name = params["env_name"]
    if CLIFF in env_name:
        prompt = f"You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task because of {reason}. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan:"
    elif BABY in env_name:
        prompt = f"You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task because of {reason}. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan:"
    return prompt

# 行動を生成する時のプロンプトに記述する指示文を返す
def get_action_prompt(env_name:str):
    if CLIFF in env_name:
            prompt = "What is the best action to achieve your task in moving 'north', 'east', 'south' or 'west'? Output only result."
    elif BABY in env_name:
        prompt = "aaa"
    return prompt

# メッセージを生成する時のプロンプトに記述する指示文を返す
def get_communication_prompt(label:str, targets:str):
    targets_str = " and ".join(targets)
    if label == "message":
        prompt = f'You are in a cooperative relationship with {targets_str}. To achieve mission, Output only message to {targets_str}.'
    elif label == "conversation":
        prompt = f'You are in a cooperative relationship with {targets_str}, and discussing to decide next action. To achieve mission, output only reply message to them.'
    return prompt

# 終了判定や状態の評価を返す
def get_achievement_status(reward, done, step, params):
    env_name = params["env_name"]
    max_step = params["max_step"]

    if CLIFF in env_name:
        is_success = False
        reason = ""
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

    return False, False, ""

# 観測情報を文字列に変換する
def obs_to_str(env, observation, params) -> list[str]:
    env_name = params["env_name"]
    if CLIFF in env_name:
        return obs_to_str_cliff(observation)
    elif "Baby" in env_name:
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
    def direction_to_str(direction:int):
        return ["east", "south", "west", "north"][direction]
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
                result = f"opened {obj_name}"
            elif state == 1:
                result = f"locked {obj_name}"
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
            if "sight_include_agents" in options.keys(): result = ""
            else:
                target_agent_id = obj_id - 10
                direction = observations[target_agent_id]["direction"]
                direction_str = direction_to_str(direction)
                result = f"{obj_name} facing {direction_str}"
        return result
    
    def position_to_text(p:tuple[int,int]):
        positions = []
        if p[0] != 0:
            is_one = p[0] == 1 or p[0] == -1
            is_forward = p[0] > 0
            s = "" if is_one else "s"
            dir = "forward" if is_forward else "back"
            positions.append(f"{abs(p[0])} step{s} {dir}")
        if p[1] != 0:
            is_one = p[1] == 1 or p[1] == -1
            is_right = p[1] > 0
            s = "" if is_one else "s"
            dir = "right" if is_right else "left"
            positions.append(f"{abs(p[1])} step{s} {dir}")
        result = " and ".join(positions)
        return result
    
    def one_agent(obs, agent_id):
        obj_infos = []
        carrying_info = ""
        image = obs["image"]
        shape = image.shape
        height, width = shape[0], shape[1]
        for r in range(height):
            for c in range(width):
                pos = (height - r - 1, c - width // 2)
                obj = image[c][r]
                if obj[0] in (0,1,3): # 不可視, 通常床, 空白 の場合は飛ばす
                    continue
                obj_name = object_to_text(obj)

                if obj_name == "":
                    continue
                if pos == (0, 0) and len(obj_name) > 0:
                    carrying_info = obj_name
                    continue

                obj_pos = position_to_text(pos)
                obj_info = f"- {obj_name} in {obj_pos}"
                obj_infos.append(obj_info)
        if "sight_include_agents" in options.keys():
            for target_id in range(env.agent_num):
                if target_id == agent_id: continue
                target_pos = env.agents_pos[target_id]
                self_pos = env.agents_pos[agent_id]
                diff = self_pos[1] - target_pos[1], target_pos[0] - self_pos[0] # y, x
                dir = obs['direction']
                if dir == 0: diff = (diff[1], -diff[0]) # east
                elif dir == 1: diff = (-diff[0], -diff[1]) # south
                elif dir == 2: diff = (-diff[1], diff[0]) # west
                elif dir == 3: diff = (diff[0], diff[1]) # north

                target_dir = observations[target_id]["direction"]
                target_dir_str = direction_to_str(target_dir)

                obj_pos = position_to_text(diff)
                obj_name = f"agent{target_id} facing {target_dir_str}"
                obj_info = f"- {obj_name} in {obj_pos}"
                obj_infos.append(obj_info)

        direction = direction_to_str(obs['direction'])
        mission = obs['mission']

        result = f'Your mission is "{mission}". You are facing {direction}.'
        if len(carrying_info) > 0:
            result = f"{result} You are carrying {carrying_info}, so you can't pick up other objects without dropping it elsewhere."
        result = f"{result} The followings is in your sight."
        result = result + "\n" + '\n'.join(obj_infos)

        return result

    return [one_agent(obs) for obs in observations]

# 文字列を行動IDに変換する
def str_to_action(text:str, params):
    env_name = params["env_name"]

    if CLIFF in env_name:
        action_to_idx = {
            "north":0,
            "east":1,
            "south":2,
            "west":3,
        }
    elif BABY in env_name:
        action_to_idx = {
            "left":0,
            "right":1,
            "forward":2,
            "pick up":3,
            "drop":4,
            "toggle":5,
            "done":6
        }

    for action, value in action_to_idx.items():
        match = re.compile(action).search(text)
        if match: return value
    return 0

# 行動IDを文字列に変換する
def action_to_str(action_idx:int, params):
    env_name = params["env_name"]
    if CLIFF in env_name:
        actions = ["north","east","south","west"]
    elif BABY in env_name:
        actions = ["left","right","forward","pick up","drop","toggle","done"]
    return actions[action_idx]