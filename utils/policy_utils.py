import gym
from utils.reflexion_utils import Reflexion
import utils.env_utils as env_utils
from utils.llm_utils import LLM
import utils.utils as utils
from utils.embedding_utils import Embedder

# LLMで行動を決定する
def act_by_llm(env:gym.Env, reflexion:Reflexion, pre_info:dict, params:dict={}) -> list[int]:
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
        action_str, response = LLM.generate(prompt, imgs[agent_id])
        action = env_utils.str_to_action(action_str, params)
        actions.append(action)

        info["queries"].append(prompt)
        info["responses"].append(response)
        info["actions"].append([action, action_str])

    # 履歴に行動情報を追加
    actions_str = [env_utils.action_to_str(action, params) for action in actions]
    reflexion.add_histories("action", actions_str)

    utils.dict_of_lists_extend(pre_info, info)
    return actions

def act_by_hierarchical_subgoal(env:gym.Env, reflexion:Reflexion, pre_info:dict, params:dict={}) -> list[int]:
    info = {
        "queries":[],
        "responses":[],
        "achieved":[],
        "subgoals":[],
        "actions":[],
    }

    imgs = env_utils.get_imgs(env, params)

    def subgoal_format(text:str) -> str:
        if ":" in text:
            text = text.split(":")[1]
            text = text.strip()
        symbols = ["'", '"', '.', ',']
        for symbol in symbols:
            text = utils.remove_edge_symbol(text, symbol)
        text = text.lower()
        text = text.replace(' the ', ' ')
        text = text.replace('coordinate (', '(')
        return text

    def get_max_similarity(text:str, targets:list[str]) -> tuple[int, float]:
        max_similarity = 0
        max_index = 0
        text_format = subgoal_format(text)
        for i, target in enumerate(targets):
            target_format = subgoal_format(target)
            #similarity = LLM.get_similarity(text_format, target_format)
            similarity = Embedder.get_similarity(text_format, target_format)
            if similarity > max_similarity:
                max_index = i
                max_similarity = similarity
        return max_index, max_similarity

    actions = []
    actions_str = env_utils.get_actions_str(params["env_name"])
    subgoal_max_generation = params["subgoal_max_generation"]
    threshold_subgoal = utils.get_value(params, "subgoal_equal_threshold", 0.9)
    threshold_action = utils.get_value(params, "action_equal_threshold", 0.9)

    for agent_id in range(env.agent_num):

        subgoal_tree = reflexion.subgoal_trees[agent_id]
        # すでに続きのサブゴールがある場合は読んでおく
        subgoal_tree.move_to_leaf()

        subgoals = subgoal_tree.get_subgoals()
        achieved_subgoals, _ = subgoal_tree.get_separated_sequence(subgoal_tree.now_node)

        # 続きのサブゴールを生成する
        is_subgoal_atomic = False
        if subgoal_tree.is_after_halfway_node() or True:
            for _ in range(len(subgoals), subgoal_max_generation):
                prompt = reflexion.get_subgoal_prompt(agent_id, achieved_subgoals, subgoals, params)
                subgoal, _ = LLM.generate(prompt, imgs[agent_id])
                subgoal = subgoal_format(subgoal)
                info["queries"].append(prompt)
                info["responses"].append(subgoal)

                # 行動に相当するサブゴールかどうかをチェックする
                index, similarity = get_max_similarity(subgoal, actions_str)
                if similarity > threshold_action:
                    subgoal = actions_str[index]
                    is_subgoal_atomic = True
                    break

                # すでに存在するサブゴールかどうかをチェックする
                _, similarity = get_max_similarity(subgoal, subgoals)
                if similarity > threshold_subgoal:#0.85?
                    break

                subgoals = [subgoal] + subgoals
                subgoal_tree.append(subgoal)

        # サブゴールを行動に変換する
        if not is_subgoal_atomic:
            prompt = reflexion.get_subgoal_to_action_prompt(agent_id, subgoals, params)
            subgoal, _ = LLM.generate(prompt, imgs[agent_id])
            info["queries"].append(prompt)
            info["responses"].append(subgoal)
            action, _ = get_max_similarity(subgoal, actions_str)
        else:
            action = env_utils.str_to_action(subgoal, params)

        info["actions"].append([action, subgoal])
        info["subgoals"].append(subgoals)
        actions.append(action)

    # サブゴールを履歴に追加
    subgoals_str = [str(l) for l in info["subgoals"]]
    reflexion.add_histories("subgoal", subgoals_str)

    # 履歴に行動情報を追加
    actions_str = [env_utils.action_to_str(action, params) for action in actions]
    reflexion.add_histories("action", actions_str)

    utils.dict_of_lists_extend(pre_info, info)
    return actions

# LLMで現在の状況について考えさせる
def consideration(env:gym.Env, reflexion:Reflexion, pre_info:dict, params:dict={}):
    if not utils.get_value(params, "is_use_consideration", False): return

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
        text, response = LLM.generate(prompt, imgs[agent_id])
        text = "You think:" + text

        info["queries"].append(prompt)
        info["responses"].append(response)
        info["considerations"].append(text)

    # 履歴に思考情報を追加
    reflexion.add_histories("consideration", info["considerations"])

    utils.dict_of_lists_extend(pre_info, info)

# エージェント間のメッセージ交換
def message(env:gym.Env, reflexion:Reflexion, pre_info:dict, params:dict={}):
    info = {
        "queries":[],
        "responses":[],
        "messages":[],
    }
    imgs = env_utils.get_imgs(env, params)

    tree:list[list[int]] = params["message_graph"]
    for agent_id in range(env.agent_num):
        agent_name = env_utils.get_agent_name(agent_id, params)
        for target_id in tree[agent_id]:
            # メッセージを生成し履歴に追加
            target_name = env_utils.get_agent_name(target_id, params)
            prompt = reflexion.get_message_prompt(agent_id, [target_name], params)
            text, response = LLM.generate(prompt, imgs[agent_id])

            if text[:len(agent_name)+1].lower() == f"{agent_name}:".lower():
                text = text[len(agent_name)+1:]
            reflexion.add_message(target_id, agent_name, text)

            info["queries"].append(prompt)
            info["responses"].append(str(response))
            info["messages"].append(text)
    utils.dict_of_lists_extend(pre_info, info)

# エージェント間の対話
def conversation(env:gym.Env, reflexion:Reflexion, pre_info:dict, params:dict={}):
    info = {
        "queries":[],
        "responses":[],
        "messages":[],
    }
    imgs = env_utils.get_imgs(env, params)

    # 全てのグループで指定ラウンドの会話を行う
    for group in params["conversation_pairs"]:
        messages = []
        for i in range(params["conversation_count"]):
            for j, agent_id in enumerate(group):
                is_last = i == (params["conversation_count"] - 1) and (j == len(group) - 1)
                agent_name = env_utils.get_agent_name(agent_id, params)
                targets_id = [target_id for target_id in group if target_id != agent_id]
                targets_str = [env_utils.get_agent_name(target_id, params) for target_id in targets_id]

                # メッセージを生成
                prompt = reflexion.get_conversation_prompt(agent_id, targets_str, messages, is_last, params)
                text, response = LLM.generate(prompt, imgs[agent_id])
                if text[:len(agent_name)+1].lower() == f"{agent_name}:".lower():
                    text = text[len(agent_name)+1:]

                messages.append((agent_name, text))

                info["responses"].append(str(response))
                info["queries"].append(prompt)
                info["messages"].append(text)

        # 会話に参加したエージェントの履歴にメッセージを追加
        for id in group:
            for name, text in messages:
                reflexion.add_message(id, name, text)

    utils.dict_of_lists_extend(pre_info, info)

def judge_subgoal_achievement(env:gym.Env, reflexion:Reflexion, pre_info:dict, is_clear:bool, params:dict={}):
    if env.now_step <= 0: return
    info = {
        "queries":[],
        "responses":[],
        "achieved":[],
    }
    imgs = env_utils.get_imgs(env, params)
    for agent_id in range(env.agent_num):
        # 各サブゴールを達成したかどうかを判定させる
        subgoal_tree = reflexion.subgoal_trees[agent_id]
        subgoals = subgoal_tree.get_subgoals()
        is_achieved = [False] * len(subgoals)
        log_outputs = ["No"] * len(subgoals)
        if is_clear:
            is_achieved[-1] = True
            log_outputs[-1] = "Yes"

        # 一括で判定したかった(精度が悪い)        
        # prompt = reflexion.get_all_subgoals_achieved_prompt(agent_id, subgoals, params)
        # judge, _ = LLM.generate(prompt, imgs[agent_id])
        # if judge[0] != '[':
        #     judge = '[' + judge
        # if judge[-1] != ']':
        #     judge = judge + ']'
        # judge_list = utils.text_to_str_list(judge)
        # judge_failed = len(judge_list) != len(subgoals)
        # info["queries"].append(prompt)
        # info["achieved"].append(judge)
        judge_failed = True

        if judge_failed:
            for i in range(len(subgoals)-1):
                prompt = reflexion.get_subgoal_achieved_prompt(agent_id, subgoals[i:], params)
                judge, _ = LLM.generate(prompt, imgs[agent_id])
                is_achieved[i] = "yes" in judge.lower()
                log_outputs[i] = judge
                info["queries"].append(prompt)
        # else:
        #     for i in range(len(subgoals)-1):
        #         is_achieved[i] = "yes" in judge_list[i].lower()
        #         log_outputs[i] = judge_list[i]

        log_achieved = [(subgoals[i], log_outputs[i]) for i in range(len(subgoals))]
        info["achieved"].append(log_achieved)

        # サブゴールリストを整理する
        for i in range(len(subgoals)):
            if is_achieved[i]:
                subgoal_tree.move_up()
            elif any(is_achieved[i:]):
                # 不要なサブゴールは削除する
                subgoal_tree.delete()

        # 続きのサブゴールがあるなら読み込んでおく
        subgoal_tree.move_to_leaf()

    utils.dict_of_lists_extend(pre_info, info)
