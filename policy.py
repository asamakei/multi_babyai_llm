import gym
from utils.reflexion_utils import Reflexion
import utils.env_utils as env_utils
import utils.llm_utils as llm_utils
import utils.utils as utils

# 行動を取得
def get_action(env:gym.Env, reflexion:Reflexion, params:dict={}) -> tuple[list[int], dict]:
    policy = policies[params["policy_name"]]
    return policy(env, reflexion, params)

# ユーザがコマンドラインで行動を決定する方策
def command_policy(env:gym.Env, reflexion, params:dict={}):
    return [int(input())] * env.agent_num, {}

# ランダムに行動を決定する方策
def random_policy(env:gym.Env, reflexion, params:dict={}):
    return env.action_space.sample(), {}

# LLMで現在の状況について考えさせる
def consideration(env:gym.Env, reflexion:Reflexion, params:dict={}):
    info = {
        "queries":[],
        "responses":[],
        "considerations":[],
    }

    is_use_vision = utils.get_value(params,"is_use_vision",False)
    imgs = env.render_masked() if is_use_vision else []

    # 各エージェントが状況について考える
    for agent_id in range(env.agent_num):
        # 履歴をもとに状態(文字列)を生成
        prompt = reflexion.get_consideration_prompt(agent_id, params)
        image = imgs[agent_id] if is_use_vision else None
        text, response = llm_utils.llm(prompt, image)
        text = "Your think:" + text
        info["queries"].append(prompt)
        info["responses"].append(response)
        info["considerations"].append(text)

    # 履歴に思考情報を追加
    reflexion.add_histories("observation", info["considerations"])
    return info

# LLMで行動を決定する
def act_by_llm(env:gym.Env, reflexion:Reflexion, params:dict={}):
    info = {
        "queries":[],
        "responses":[],
        "actions":[],
    }

    is_use_vision = utils.get_value(params,"is_use_vision",False)
    imgs = env.render_masked() if is_use_vision else []

    # 各エージェントが行動を決定する
    actions = []
    for agent_id in range(env.agent_num):
        # 履歴をもとに状態(文字列)を生成
        prompt = reflexion.get_action_prompt(agent_id, params)
        image = imgs[agent_id] if is_use_vision else None
        action_str, response = llm_utils.llm(prompt, image)
        action = env_utils.str_to_action(action_str, params)
        actions.append(action)

        info["queries"].append(prompt)
        info["responses"].append(response)
        info["actions"].append([action, action_str])

    # 履歴に行動情報を追加
    actions_str = [env_utils.action_to_str(action, params) for action in actions]
    reflexion.add_histories("action", actions_str)

    return actions, info

# エージェントがそれぞれ独立して行動する方策
def simple_llm_policy(env:gym.Env, reflexion:Reflexion, params:dict={}):
    info = {}
    if utils.get_value(params, "is_use_consideration", False):
        info = consideration(env, reflexion, params)
    actions, info_act = act_by_llm(env, reflexion, params)
    utils.dict_of_lists_extend(info, info_act)
    return actions, info

# エージェントが互いにメッセージを交換したあとに行動する方策
def message_llm_policy(env:gym.Env, reflexion:Reflexion, params:dict={}):
    info = {
        "queries":[],
        "responses":[],
        "messages":[],
        "actions":[],
    }

    if utils.get_value(params, "is_use_consideration", False):
        info_consideration = consideration(env, reflexion, params)
        utils.dict_of_lists_extend(info, info_consideration)

    is_use_vision = utils.get_value(params,"is_use_vision",False)
    imgs = env.render_masked() if is_use_vision else []

    # エージェント間のメッセージの生成
    tree:list[list[int]] = params["message_graph"]
    for agent_id in range(env.agent_num):
        agent_name = env_utils.get_agent_name(agent_id, params)
        for target_id in tree[agent_id]:
            # メッセージを生成し履歴に追加
            target_name = env_utils.get_agent_name(target_id, params)
            prompt = reflexion.get_message_prompt(agent_id, [target_name], params)
            image = imgs[agent_id] if is_use_vision else None
            text, response = llm_utils.llm(prompt, image)

            if text[:len(agent_name)+1].lower() == f"{agent_name}:".lower():
                text = text[len(agent_name)+1:]
            reflexion.add_message(target_id, agent_name, text)

            info["queries"].append(prompt)
            info["responses"].append(str(response))
            info["messages"].append(text)

    # 行動を生成
    actions, info_act = act_by_llm(env, reflexion, params)
    utils.dict_of_lists_extend(info, info_act)

    return actions, info

# エージェントが互いに会話をしたあとに行動する方策
def conversation_llm_policy(env:gym.Env, reflexion:Reflexion, params:dict={}):
    info = {
        "queries":[],
        "responses":[],
        "messages":[],
        "actions":[],
    }

    if utils.get_value(params, "is_use_consideration", False):
        info_consideration = consideration(env, reflexion, params)
        utils.dict_of_lists_extend(info, info_consideration)

    is_use_vision = utils.get_value(params,"is_use_vision",False)
    imgs = env.render_masked() if is_use_vision else []

    # エージェントのグループで一周会話を行う処理
    def conversation_one_round(groups):
        for agent_id in groups:
            agent_name = env_utils.get_agent_name(agent_id, params)
            targets_id = [target_id for target_id in groups if target_id != agent_id]
            targets_str = [env_utils.get_agent_name(target_id, params) for target_id in targets_id]

            # メッセージを生成
            prompt = reflexion.get_conversation_prompt(agent_id, targets_str, params)
            image = imgs[agent_id] if is_use_vision else None
            text, response = llm_utils.llm(prompt, image)
            if text[:len(agent_name)+1].lower() == f"{agent_name}:".lower():
                text = text[len(agent_name)+1:]

            # 会話に参加したエージェントの履歴にメッセージを追加
            for id in groups:
                reflexion.add_message(id, agent_name, text)

            info["responses"].append(str(response))
            info["queries"].append(prompt)
            info["messages"].append(text)

    # 全てのグループで指定ラウンドの会話を行う
    for pair in params["conversation_pairs"]:
        for _ in range(params["conversation_count"]):
            conversation_one_round(pair)

    # 行動を生成
    actions, info_act = act_by_llm(env, reflexion, params)
    utils.dict_of_lists_extend(info, info_act)

    return actions, info

# サブゴールを階層的に生成して行動を決定する方策
def subgoal_llm_policy(env:gym.Env, reflexion:Reflexion, params:dict={}):
    info = {
        "queries":[],
        "responses":[],
        "subgoals":[],
        "actions":[],
    }

    if utils.get_value(params, "is_use_consideration", False):
        info_consideration = consideration(env, reflexion, params)
        utils.dict_of_lists_extend(info, info_consideration)

    is_use_vision = utils.get_value(params,"is_use_vision",False)
    imgs = env.render_masked() if is_use_vision else []

    def one_agent(agent_id:int):
        # サブゴールを階層的に生成する
        subgoal_max_generation = params["subgoal_max_generation"]
        subgoals = ["achieve the mission"]
        for i in range(subgoal_max_generation):
            prompt = reflexion.get_subgoal_prompt(agent_id, subgoals, params)
            image = imgs[agent_id] if is_use_vision else None
            subgoal, _ = llm_utils.llm(prompt, image)
            info["queries"].append(prompt)

            if subgoal.lower() in env_utils.get_actions_str(params["env_name"]):
                is_subgoal_atomic = True
            elif subgoal in subgoals or i == subgoal_max_generation - 1:
                prompt = reflexion.get_subgoal_to_action_prompt(agent_id, subgoals, params)
                image = imgs[agent_id] if is_use_vision else None
                subgoal, _ = llm_utils.llm(prompt, image)
                info["queries"].append(prompt)
                is_subgoal_atomic = True
            else:
                is_subgoal_atomic = False

            subgoals.append(subgoal)
            if is_subgoal_atomic: break

        # 今はとりあえずLLMにに変換させる

        action = env_utils.str_to_action(subgoals[-1], params)

        info["actions"].append([action, subgoals[-1]])
        info["subgoals"].append(subgoals)
        return action

    # 各エージェントが行動を決定する
    actions = []
    for agent_id in range(env.agent_num):
        action = one_agent(agent_id)
        actions.append(action)

    # サブゴールを履歴に追加
    subgoals_str = [f"Subgoals you planned:{l}" for l in info["subgoals"]]
    reflexion.add_histories("observation", subgoals_str)

    # 履歴に行動情報を追加
    actions_str = [env_utils.action_to_str(action, params) for action in actions]
    reflexion.add_histories("action", actions_str)

    return actions, info

# ポリシー名と関数の紐付け
policies = {
    "command" : command_policy,
    "simple_llm" : simple_llm_policy,
    "message_llm" : message_llm_policy,
    "conversation_llm" : conversation_llm_policy,
    "subgoal_llm" : subgoal_llm_policy,
    "random" : random_policy
}