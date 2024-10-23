import gym
from utils.reflexion_utils import Reflexion
import utils.env_utils as env_utils
import utils.llm_utils as llm_utils

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

# エージェントがそれぞれ独立して行動する方策
def simple_llm_policy(env:gym.Env, reflexion:Reflexion, params:dict={}):
    info = {
        "queries":[],
        "responses":[],
        "actions":[],
    }

    # 各エージェントが行動を決定する
    actions = []
    for agent_id in range(env.agent_num):
        # 履歴をもとに状態(文字列)を生成
        prompt = reflexion.get_action_prompt(agent_id)
        action_str, response = llm_utils.llm(prompt)
        action = env_utils.str_to_action(action_str, params)
        actions.append(action)

        info["queries"].append(prompt)
        info["responses"].append(response)
        info["actions"].append([action, action_str])

    # 履歴に行動情報を追加
    actions_str = [env_utils.action_to_str(action, params) for action in actions]
    reflexion.add_histories("action", actions_str)

    return actions, info

# エージェントが互いにメッセージを交換したあとに行動する方策
def message_llm_policy(env:gym.Env, reflexion:Reflexion, params:dict={}):
    info = {
        "queries":[],
        "responses":[],
        "messages":[],
        "actions":[],
    }

    # エージェント間のメッセージの生成
    tree:list[list[int]] = params["message_graph"]
    for agent_id in range(env.agent_num):
        agent_name = env_utils.get_agent_name(agent_id)
        for target_id in tree[agent_id]:
            # メッセージを生成し履歴に追加
            target_name = env_utils.get_agent_name(target_id)
            prompt = reflexion.get_message_prompt(agent_id, [target_name])
            text, response = llm_utils.llm(prompt)
            reflexion.add_message(target_id, agent_name, text)

            info["queries"].append(prompt)
            info["responses"].append(str(response))
            info["messages"].append(text)

    # 行動を生成
    actions, info_simple = simple_llm_policy(env, reflexion, params)
    info["queries"].join(info_simple["queries"])
    info["responses"].join(info_simple["responses"])
    info["actions"].join(info_simple["actions"])

    return actions, info

# エージェントが互いに会話をしたあとに行動する方策
def conversation_llm_policy(env:gym.Env, obs, reflexion:Reflexion, params:dict={}):
    info = {
        "queries":[],
        "responses":[],
        "messages":[],
        "actions":[],
    }

    # エージェントのグループで一周会話を行う処理
    def conversation_one_round(groups):
        for agent_id in groups:
            agent_name = env_utils.get_agent_name(agent_id)
            targets_id = [target_id for target_id in groups if target_id != agent_id]
            targets_str = [env_utils.get_agent_name(target_id) for target_id in targets_id]

            # メッセージを生成
            prompt = reflexion.get_conversation_prompt(agent_id, targets_str)
            text, response = llm_utils.llm(prompt)

            # 会話に参加したエージェントの履歴にメッセージを追加
            for id in groups:
                reflexion.add_message(id, agent_name, text)

            info["responses"].append(str(response))
            info["queries"].append(prompt)
            info["sended_messages"].append(text)

    # 全てのグループで指定ラウンドの会話を行う
    for pair in params["conversation_pairs"]:
        for _ in params["conversation_count"]:
            conversation_one_round(pair)

    # 行動を生成
    actions, info_simple = simple_llm_policy(env, reflexion, params)
    info["queries"].join(info_simple["queries"])
    info["responses"].join(info_simple["responses"])
    info["actions"].join(info_simple["actions"])

    return actions, info

# ポリシー名と関数の紐付け
policies = {
    "command" : command_policy,
    "simple_llm" : simple_llm_policy,
    "message_llm" : message_llm_policy,
    "conversation_llm" : conversation_llm_policy,
    "random" : random_policy
}