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
def command_policy(env:gym.Env, reflexion:Reflexion, params:dict={}):
    return [int(input())] * env.agent_num, {}

# ランダムに行動を決定する方策
def random_policy(env:gym.Env, reflexion:Reflexion, params:dict={}):
    return env.action_space.sample(), {}

# LLMで現在の状況について考えさせる
def consideration(env:gym.Env, reflexion:Reflexion, params:dict={}):
    if not utils.get_value(params, "is_use_consideration", False):
        return {}

    info = {
        "queries":[],
        "responses":[],
        "considerations":[],
    }

    imgs = env_utils.get_imgs(env, params)

    # 各エージェントが状況について考える
    for agent_id in range(env.agent_num):
        # 履歴をもとに状態(文字列)を生成
        prompt = reflexion.get_consideration_prompt(agent_id, params)
        text, response = llm_utils.llm(prompt, imgs[agent_id])
        text = "Your think:" + text
        info["queries"].append(prompt)
        info["responses"].append(response)
        info["considerations"].append(text)

    # 履歴に思考情報を追加
    reflexion.add_histories("consideration", info["considerations"])
    return info

# LLMで行動を決定する
def act_by_llm(env:gym.Env, reflexion:Reflexion, params:dict={}):
    info = {
        "queries":[],
        "responses":[],
        "actions":[],
    }

    imgs = env_utils.get_imgs(env, params)

    # 各エージェントが行動を決定する
    actions = []
    for agent_id in range(env.agent_num):
        # 履歴をもとに状態(文字列)を生成
        prompt = reflexion.get_action_prompt(agent_id, params)
        action_str, response = llm_utils.llm(prompt, imgs[agent_id])
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

    info_consideration = consideration(env, reflexion, params)
    utils.dict_of_lists_extend(info, info_consideration)

    imgs = env_utils.get_imgs(env, params)

    # エージェント間のメッセージの生成
    tree:list[list[int]] = params["message_graph"]
    for agent_id in range(env.agent_num):
        agent_name = env_utils.get_agent_name(agent_id, params)
        for target_id in tree[agent_id]:
            # メッセージを生成し履歴に追加
            target_name = env_utils.get_agent_name(target_id, params)
            prompt = reflexion.get_message_prompt(agent_id, [target_name], params)
            text, response = llm_utils.llm(prompt, imgs[agent_id])

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

    info_consideration = consideration(env, reflexion, params)
    utils.dict_of_lists_extend(info, info_consideration)

    imgs = env_utils.get_imgs(env, params)

    # エージェントのグループで一周会話を行う処理
    def conversation_one_round(groups:list, is_last:bool):
        for agent_id in groups:
            agent_name = env_utils.get_agent_name(agent_id, params)
            targets_id = [target_id for target_id in groups if target_id != agent_id]
            targets_str = [env_utils.get_agent_name(target_id, params) for target_id in targets_id]

            # メッセージを生成
            prompt = reflexion.get_conversation_prompt(agent_id, targets_str, is_last, params)
            text, response = llm_utils.llm(prompt, imgs[agent_id])
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
        for i in range(params["conversation_count"]):
            is_last = i == params["conversation_count"] - 1
            conversation_one_round(pair, is_last)

    # 行動を生成
    actions, info_act = act_by_llm(env, reflexion, params)
    utils.dict_of_lists_extend(info, info_act)

    return actions, info

# サブゴールを階層的に生成して行動を決定する方策
def subgoal_llm_policy(env:gym.Env, reflexion:Reflexion, params:dict={}):
    info = {
        "queries":[],
        "responses":[],
        "achieved":[],
        "subgoals":[],
        "actions":[],
    }

    info_consideration = consideration(env, reflexion, params)
    utils.dict_of_lists_extend(info, info_consideration)

    imgs = env_utils.get_imgs(env, params)

    def subgoal_format(text:str) -> str:
        symbols = ["'", '"', '.', ',']
        for symbol in symbols:
            text = utils.remove_edge_symbol(text, symbol)
        text = text.lower()
        text = text.replace(' the ', ' ')
        return text

    def get_max_similarity(text:str, targets:list[str]) -> tuple[int, float]:
        max_similarity = 0
        max_index = 0
        text_format = subgoal_format(text)
        for i, target in enumerate(targets):
            target_format = subgoal_format(target)
            similarity = llm_utils.get_similarity(text_format, target_format)
            if similarity > max_similarity:
                max_index = i
                max_similarity = similarity
        return max_index, max_similarity

    def one_agent(agent_id:int):
        actions_str = env_utils.get_actions_str(params["env_name"])

        subgoal_max_generation = params["subgoal_max_generation"]
        subgoal_tree = reflexion.subgoal_trees[agent_id]

        # 各サブゴールを達成したかどうかを判定させる
        subgoals = subgoal_tree.get_subgoals()
        is_achieved = [False] * len(subgoals)
        log_outputs = ["No"] * len(subgoals)
        for i in range(len(subgoals)-1):
            prompt = reflexion.get_subgoal_achieved_prompt(agent_id, subgoals[i:], params)
            judge, _ = llm_utils.llm(prompt, imgs[agent_id])
            is_achieved[i] = "yes" in judge.lower()
            log_outputs[i] = judge
            info["queries"].append(prompt)

        log_achieved = [(subgoals[i], log_outputs[i]) for i in range(len(subgoals))]
        info["achieved"].append(log_achieved)

        # サブゴールリストを整理する
        for i in range(len(subgoals)):
            if is_achieved[i]:
                subgoal_tree.move_up()
            elif any(is_achieved[i:]):
                # 不要なサブゴールは削除する
                subgoal_tree.delete()

        achieved_subgoals, subgoals = subgoal_tree.get_separated_sequence(subgoal_tree.now_node)

        # 続きのサブゴールを生成する
        is_subgoal_atomic = False
        for _ in range(len(subgoals), subgoal_max_generation):
            prompt = reflexion.get_subgoal_prompt(agent_id, achieved_subgoals, subgoals, params)
            subgoal, _ = llm_utils.llm(prompt, imgs[agent_id])
            subgoal = subgoal_format(subgoal)
            info["queries"].append(prompt)

            # 行動に相当するサブゴールかどうかをチェックする
            index, similarity = get_max_similarity(subgoal, actions_str)
            if similarity > 0.925:
                subgoal = actions_str[index]
                is_subgoal_atomic = True
                break

            # すでに存在するサブゴールかどうかをチェックする
            _, similarity = get_max_similarity(subgoal, subgoals)
            if similarity > 0.9:#0.85?
                break # continue?

            subgoals = [subgoal] + subgoals
            subgoal_tree.append(subgoal)

        # サブゴールを行動に変換する
        if not is_subgoal_atomic:
            prompt = reflexion.get_subgoal_to_action_prompt(agent_id, subgoals, params)
            subgoal, _ = llm_utils.llm(prompt, imgs[agent_id])
            info["queries"].append(prompt)
            action, _ = get_max_similarity(subgoal, actions_str)
        else:
            action = env_utils.str_to_action(subgoal, params)

        info["actions"].append([action, subgoal])
        info["subgoals"].append(subgoals)
        return action

    # 各エージェントが行動を決定する
    actions = []
    for agent_id in range(env.agent_num):
        action = one_agent(agent_id)
        actions.append(action)

    # サブゴールを履歴に追加
    subgoals_str = [str(l) for l in info["subgoals"]]
    reflexion.add_histories("subgoal", subgoals_str)

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