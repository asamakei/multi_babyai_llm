import gym
from utils.reflexion_utils import Reflexion
from utils.policy_utils import *

# 行動を取得
def get_action(env:gym.Env, reflexion:Reflexion, params:dict={}) -> tuple[list[int], dict]:
    info:dict = {}
    policy = policies[params["policy_name"]]
    actions = policy(env, reflexion, info, params)
    return actions, info

# ユーザがコマンドラインで行動を決定する方策
def command_policy(env:gym.Env, reflexion:Reflexion, info:dict, params:dict={}):
    commands = [int(x) for x in input().split()]
    return commands

# ランダムに行動を決定する方策
def random_policy(env:gym.Env, reflexion:Reflexion, info:dict, params:dict={}):
    return env.action_space.sample()

# エージェントがそれぞれ独立して行動する方策
def simple_policy(env:gym.Env, reflexion:Reflexion, info:dict, params:dict={}):
    consideration(env, reflexion, info, params)
    actions = act_by_llm(env, reflexion, info, params)
    return actions

# エージェントが互いにメッセージを交換したあとに行動する方策
def message_policy(env:gym.Env, reflexion:Reflexion, info:dict, params:dict={}):
    message(env, reflexion, info, params)
    consideration(env, reflexion, info, params)
    actions = act_by_llm(env, reflexion, info, params)
    return actions

# エージェントが互いに会話をしたあとに行動する方策
def conversation_policy(env:gym.Env, reflexion:Reflexion, info:dict, params:dict={}):
    conversation(env, reflexion, info, params)
    consideration(env, reflexion, info, params)
    actions = act_by_llm(env, reflexion, info, params)
    return actions

# サブゴールを階層的に生成して行動を決定する方策
def subgoal_policy(env:gym.Env, reflexion:Reflexion, info:dict, params:dict={}):
    judge_subgoal_achievement(env, reflexion, info, False, params)
    consideration(env, reflexion, info, params)
    actions = act_by_hierarchical_subgoal(env, reflexion, info, params)
    return actions

# サブゴールを初回のみ生成して行動を決定する方策
def simple_subgoal_policy(env:gym.Env, reflexion:Reflexion, info:dict, params:dict={}):
    # サブゴールを履歴に一時的に載せる 処理が汚いので直したい
    judge_subgoal_achievement(env, reflexion, info, False, params)
    reflexion.remove_label("subgoal")
    reflexion.add_now_subgoal()
    
    consideration(env, reflexion, info, params)
    actions = act_by_llm(env, reflexion, info, params)
    return actions

# ポリシー名と関数の紐付け
policies = {
    "command" : command_policy,
    "simple" : simple_policy,
    "message" : message_policy,
    "conversation" : conversation_policy,
    "subgoal" : subgoal_policy,
    "simple_subgoal" : simple_subgoal_policy,
    "random" : random_policy
}