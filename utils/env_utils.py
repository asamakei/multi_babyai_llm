import re
import gym
from utils.utils import get_value

from utils.llm_utils import get_image_token
import utils.utils as utils
import utils.babyai_utils as babyai_utils

from gym_minigrid.minigrid import (
    IDX_TO_COLOR,
)

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
    env_name = params["env_name"]
    if CLIFF in env_name:
        return f"agent{agent_id}" if agent_id >= 0 else ""
    elif BABY in env_name:
        return babyai_utils.get_agent_name(agent_id)

# 行動の名前の取得
def get_actions_str(env_name):
    if CLIFF in env_name:
        return ["move north","move east","move south","move west"]
    elif BABY in env_name:
        return ["turn left","turn right","go forward","pick up the item in forward","drop the carrying item forward","open the door in forward"]

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
def get_explain(env, params):
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
        
        if env.agent_num >= 2:
            sentences.append(f"Work with other agents to accomplish tasks.")

        base_prompt = utils.join_sentences(sentences)
        task_prompt = 'achieve the "mission" as soon as possible'

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

# 通常のReflexionを行う時のプロンプトに記述する指示文を返す
def get_general_reflexion_instr(reason:str, agent_id:int, params = {}):
    agent_name = get_agent_name(agent_id, params)
    profile = f"You are {agent_name}. " if agent_id >= 0 else ""
    image_explain = get_image_explain(agent_id, params)
    prompt = profile + image_explain
    prompt += f"You are given the history of a past experience in which you were placed in an environment and given a task to complete. In this trial, you were unsuccessful in completing the task because of {reason}."
    prompt += " " + f"Do not summarize your environment, but rather think about the strategy to complete the task, considering actions that are useful and those that should be improved. Output a detailed new plan from the start of the task to its accomplishment in 3 to 6 sentences \nGive only your plan:"
    return prompt

# Subgoalに関するReflexionを行う時のプロンプトに記述する指示文を返す
def get_subgoal_reflexion_instr(reason:str, agent_id:int, subgoal_tree, params = {}):
    achieved, failed = subgoal_tree.get_all_sequence()
    agent_name = get_agent_name(agent_id, params)
    profile = f"You are {agent_name}. " if agent_id >= 0 else ""
    image_explain = get_image_explain(agent_id, params)
    next_trial = params["reflexion_memory_size"]+1
    prompt = profile + image_explain
    prompt += f"You are given the history of a past experience in which you were placed in an environment and given a task to complete. In this trial, you were unsuccessful in completing the task because of {reason}."
    prompt += " " + f"You planned and accomplished the following subgoal list {achieved}."
    prompt += " " + f"However, You were not able to achieve the following subgoal list {failed}."
    #prompt += " " + f"Do not summarize your environment, but rather think about the subgoal planning strategy to complete the task, considering concretely subgoals that are useful and those that should be improved. You will need this later when you are solving the same task. Output only your plan in 3 to 6 sentences. \nGive your plan:"
    #prompt += " " + f"Do not summarize your environment, but rather think about the subgoal planning strategy to complete the task, considering concretely subgoals that are useful and those that should be improved. You will need this as your memory 'Trial {next_trial}' later when you are solving the same task. Output without repeating memory of previous trials. \nGive your plan:"
    #prompt += " " + f"Do not summarize your environment, but rather think about the subgoal planning strategy to complete the task, considering concretely subgoals that are useful and those that should be improved. For each subgoal that was not achieved, consider how it could have been achieved. You will need this later when you are solving the same task. Output only your plan in 3 to 6 sentences. \nGive your plan:"
    #prompt += " " + f"Do not summarize your environment, but rather think about the subgoal planning strategy to complete the task, considering concretely subgoals that are useful and those that should be improved. For each subgoal that was not achieved, consider how it could have been achieved. You will need this later when you are solving the same task. Output your plan in 3 to 6 sentences and output new subgoal list briefly in the format ['A', 'B', 'C', ...] for achievement the task. \nGive your plan and list:"
    prompt += " " + f"Do not summarize your environment, but rather think about the subgoal planning strategy to complete the task, considering subgoals that are useful and those that should be improved. Output a detailed new plan from the start of the task to its accomplishment in 3 to 6 sentences \nGive only your plan:"
    return prompt

def get_subgoal_reflexion_achievement(subgoal_tree, params={}):
    achieved, failed = subgoal_tree.get_all_sequence()
    result = f"I planned and accomplished the following subgoal list {achieved}, "
    result += f"but was not able to achieve the following subgoal list {failed}."
    return result

# 行動を生成する時のプロンプトに記述する指示文を返す
def get_action_instr(env_name:str, agent_id:int, params = {}):
    agent_name = get_agent_name(agent_id, params)
    profile = f"You are {agent_name}. " if agent_id >= 0 else ""
    image_explain = get_image_explain(agent_id, params)
    actions_joined = get_actions_joined_str(env_name, "or")
    prompt = profile + image_explain + f"What is the best action to achieve your task in {actions_joined}? Output only result.\n>"
    return prompt

# メッセージを生成する時のプロンプトに記述する指示文を返す
def get_communication_instr(label:str, agent_id:int, targets:str, is_last:bool, params={}):
    targets_str = " and ".join(targets)
    agent_name = get_agent_name(agent_id, params)
    profile = f"You are {agent_name}. " if agent_id >= 0 else ""
    image_explain = get_image_explain(agent_id, params)
    if label == "message":
        prompt = profile + image_explain + f'You are in a cooperative relationship with {targets_str}. The mission has not yet been accomplished. To two agents split up and accomplish tasks efficiently, Output only message to {targets_str} in 2 to 5 sentences, do not give instructions.'
        #prompt = f'You are in a cooperative relationship with {targets_str}. To achieve mission, Output only message to {targets_str}.'
    elif label == "conversation":
        last_message = "This is the last turn in the conversation." if is_last else ""
        prompt = profile + image_explain + f'You are in a cooperative relationship with {targets_str}, and discussing to decide next action. The mission has not yet been accomplished.{last_message} To two agents split up and accomplish tasks efficiently, output only reply message to them in 1 to 3 sentences, do not give instructions.'
        #prompt = f'You are in a cooperative relationship with {targets_str}, and discussing to decide next action. To achieve mission, output only reply message to them.'
    prompt += f"\n{agent_name}:"
    return prompt

# 思考を生成する時のプロンプトに記述する指示文を返す
def get_consideration_instr(env_name:str, agent_id:int, params = {}):
    agent_name = get_agent_name(agent_id, params)
    profile = f"You are {agent_name}. " if agent_id >= 0 else ""
    image_explain = get_image_explain(agent_id, params)
    prompt = profile + image_explain + f"Output what you think would accomplish the task in the current situation in 2 to 3 sentences.\nYour think:"
    return prompt

# サブゴールを生成する時のプロンプトに記述する指示文を返す
def get_subgoal_instr(env_name:str, agent_id:int, achieved:list[str], not_achieved:list[str], params = {}):
    agent_name = get_agent_name(agent_id, params)
    profile = f"You are {agent_name}. " if agent_id >= 0 else ""
    image_explain = get_image_explain(agent_id, params)
    subgoal = not_achieved[0]
    actions = get_actions_joined_str(params["env_name"])
    # prompt = profile + image_explain + f"You think you should achieve subgoals {not_achieved}. Output only one subgoal for achieving '{subgoal}' in a few words, abstractly at first, then gradually more concretely. The most concrete subgoal is your action. Do not output subgoals with the same meaning.\nYour subgoal:"
    # prompt = profile + image_explain + f"You think you should achieve subgoals {not_achieved}. Output only one subgoal for achieving it in a few words, abstractly. Do not output with the same meaning.\nYour subgoal:"
    # prompt = profile + image_explain + f"You think you should achieve subgoals {not_achieved}. Output only one subgoal for achieving it in a few words, abstractly at first, then gradually more concretely. Do not output with the same meaning.\nYour subgoal:"
    # prompt = profile + image_explain + f"You think you should achieve subgoals {not_achieved}. Output only one subgoal for achieving it in a few words, abstractly at first, then gradually more concretely. The most concrete subgoal is {actions}. Do not output with the same meaning.\nYour subgoal:"
    # prompt = profile + image_explain + f"You think you should achieve subgoals {not_achieved}. Output only one subgoal for achieving it in a few words, abstractly at first, then gradually more concretely. Do not output with the same meaning.\nYour subgoal:"
    prompt = profile + image_explain + f"You achieved subgoals {achieved}. You think you should achieve subgoals {not_achieved}. Output only one subgoal for achieving it in a few words, abstractly at first, then gradually more concretely. Do not output with the same meaning.\nYour subgoal:"
    return prompt

# サブゴールを行動に変換する時のプロンプトに記述する指示文を返す
def get_subgoal_to_action_instr(env_name:str, agent_id:int, subgoals:list[str], params = {}):
    agent_name = get_agent_name(agent_id, params)
    profile = f"You are {agent_name}. " if agent_id >= 0 else ""
    image_explain = get_image_explain(agent_id, params)
    actions_joined = get_actions_joined_str(env_name, "or")
    prompt = profile + image_explain + f"You think you should achieve {subgoals}. To achieve '{subgoals[0]}', what is the best action in {actions_joined}? Output only action name.\nYour action:"
    return prompt

# サブゴールを行動に変換する時のプロンプトに記述する指示文を返す
def get_subgoal_achieved_instr(env_name:str, agent_id:int, subgoals:list[str], params = {}):
    agent_name = get_agent_name(agent_id, params)
    profile = f"You are {agent_name}. " if agent_id >= 0 else ""
    image_explain = get_image_explain(agent_id, params)
    prompt = profile + image_explain + f"In the previous step, you thought you should achieve {subgoals}. Was the subgoal '{subgoals[0]}' achieved? Output only 'Yes' or 'No'.\nYes or No:" #  by the previous your action も要るかと思ったがマルチエージェントを考慮して一旦消す
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

def get_imgs(env, params):
    is_use_vision = utils.get_value(params,"is_use_vision",False)
    imgs = env.render_masked() if is_use_vision else [None] * env.agent_num
    return imgs

# 観測情報を文字列に変換する
def obs_to_str(env, observation, params) -> list[str]:
    env_name = params["env_name"]
    if CLIFF in env_name:
        return obs_to_str_cliff(observation)
    elif BABY in env_name:
        return babyai_utils.obs_to_str_baby(env, observation, params)

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

def get_feedbacks(env, observations, actions:list[int], params:dict={}):
    env_name = params["env_name"]
    if CLIFF in env_name:
        return [""] * env.agent_num
    elif BABY in env_name:
        return babyai_utils.get_feedbacks(env, observations, actions, params)

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