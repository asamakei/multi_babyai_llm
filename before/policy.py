import os
import re
import gym

from openai import OpenAI

from flan_rl_agent import FlanAgent
from trl import AutoModelForSeq2SeqLMWithValueHead
from transformers import T5Tokenizer
from peft import LoraConfig

import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import KEY

from before.dqn_agent import DQNAgent

from reflexion.env_history import EnvironmentHistory
from env_utils import obs_to_str

from gym_minigrid.minigrid import (
    IDX_TO_OBJECT,
    IDX_TO_COLOR,
)

hyperparams = {
    "api_key": KEY.openai_api_key,
}

gpt = {
    "client": OpenAI(api_key = hyperparams["api_key"]),
}

llama = {
    "model": None,
    "tokenizer": None,
    "pipeline": None,
    "terminators": None,
    "agents": None
}

flan = {
    "agent": None
}

reflexion = {
    "history": None,
    "memory": []
}

def init(params, options):
    model_name = options["llm_model"]

    if "CliffWalking" in params["env_name"]:
        base_prompt = "Interact with an grid world environment to solve a task."
        reflexion["history"] = EnvironmentHistory(
            base_prompt,
            "Reach the goal as soon as possible without falling off the cliff.",
            reflexion["memory"],
            []
        )

    if "llama" in model_name:
        load_llama(model_name)
        if llama["agents"] == None:
            llama["agents"] = [
                #DQNAgent(128256, 6, "cuda") for _ in range(params["agent_num"])
                DQNAgent(4+48, 4, "cuda") for _ in range(params["agent_num"])
            ]
    elif "flan" in model_name:
        load_flan(model_name)

def run_reflexion(is_success: bool, reason: str, options):
    history:EnvironmentHistory = reflexion["history"]
    result_str = "SUCCESS" if is_success else f"FALIED"
    history.add("result", result_str)
    if is_success: return reflexion["memory"]

    query = f"You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task because of {reason}. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan:"
    prompt = [("system", str(history) + "\n" + query)]
    text, response = llm(prompt, options["llm_model"])
    reflexion["memory"].append(text)
    if len(reflexion["memory"]) > 3:
        reflexion["memory"] = reflexion["memory"][-3:]
    return reflexion["memory"]

def save_model(path:str, options):
    if llama["agents"] is None: return
    for i in range(len(llama["agents"])):
        llama["agents"][i].save_model(path, f"agent{i}")

def get_action(env:gym.Env, obs, options:dict={}) -> tuple[list[int], dict]:
    policy = policies[options["policy_name"]]
    return policy(env, obs, options)

def command_policy(env:gym.Env, obs, options:dict={}):
    return [int(input())] * env.agent_num, {}

def random_policy(env:gym.Env, obs, options:dict={}):
    return env.action_space.sample(), {}

def simple_llm_policy(env:gym.Env, obs, options:dict={}):
    options["message_graph"] = [[] for _ in range(env.agent_num)]
    return message_llm_policy(env, obs, options)

def conversation_llm_policy(env:gym.Env, obs, options:dict={}):
    info = {
        "queries":[],
        "responses":[],
        "sended_messages":[],
        "plans":[],
        "actions":[],
        "actions_str":[],
        "reasons":[],
    }

    # エージェント間のメッセージの生成
    pairs:list[list[int]] = options["conversation_pairs"]
    conversation_count = options["conversation_count"]
    messages:list[list[tuple[str, str]]] = [[] for _ in range(env.agent_num)]
    for pair in pairs:
        for _ in range(conversation_count):
            for agent_id in pair:
                targets_str = []
                for target_id in pair:
                    if target_id == agent_id: continue
                    targets_str.append(f"agent{target_id}")
                
                options["message_target"] = ", ".join(targets_str)
                queries = get_llm_query(env, obs, agent_id, "conversation", messages[agent_id], options=options)
                if options["free_mode"]:
                    message, response = "Move freely.", {}
                else:
                    message, response = llm(queries, options['llm_model'])
                for target_id in pair:
                    messages[target_id].append((f"agent{agent_id}", message))

                info["responses"].append(str(response))
                info["queries"].append(queries)
                info["sended_messages"].append(message)
    # 行動を生成
    actions = generate_actions_llm(env, obs, messages, info, options)
    # 現在の方針を生成
    if options["is_add_pre_plan"]:
        generate_plans_llm(env, obs, messages, info, options)

    return actions, info

def message_llm_policy(env:gym.Env, observation, options:dict={}):
    info = {
        "queries":[],
        "responses":[],
        "actions":[],
        "actions_str":[],
        "reasons":[]
    }
    obs_text = obs_to_str(env, observation, 0, options)
    reflexion["history"].add("observation", obs_text)
    query = str(reflexion["history"]) + " What is the best action to achieve your task in moving 'north', 'east', 'south' or 'west'? Output only result.\n" + ">"
    prompt = [("system", query)]
    text, response = llm(prompt, options["llm_model"])
    #print(query)
    print(text)
    actions = [str_to_action_cliff(text)]

    reflexion["history"].add("action", text)
    info["queries"].append(query)
    info["actions"].append(actions)
    info["actions_str"].append(text)
    info["responses"].append(response)
    return actions, info

def message_llm_policy_cliff_agent(env:gym.Env, observation, options:dict={}):
    info = {
        "queries":[],
        "responses":[],
        "sended_messages":[],
        "plans":[],
        "actions":[],
        "actions_str":[],
        "reasons":[],
    }

    # 行動を生成
    action_str_list = ["up", "right", "down", "left"]
    obs_str = obs_to_str(env, observation, 0, options)
    query = [{"role":"system", "content":f'You are excellent strategist. You must reach goal moving on grid and avoiding cliff. {obs_str} You are select next action in "up", "down", "left", "right". Which action is the best to reach goal? Output only your selection.'}]
    #actions = [generate_action_from_llama_agent_cliff(query, observation)]
    actions = [flan["agent"].act(query)]
    info["actions"].append(actions)
    info["actions_str"].append([action_str_list[actions[0]]])
    return actions, info

def message_llm_policy_baby(env:gym.Env, obs, options:dict={}):
    info = {
        "queries":[],
        "responses":[],
        "sended_messages":[],
        "plans":[],
        "actions":[],
        "actions_str":[],
        "reasons":[],
    }

    # エージェント間のメッセージの生成
    tree:list[list[int]] = options["message_graph"]
    messages:list[list[tuple[str, str]]] = [[] for _ in range(env.agent_num)]
    for agent_id in range(env.agent_num):
        for target_id in tree[agent_id]:
            options["message_target"] = f"agent{target_id}"
            queries = get_llm_query(env, obs, agent_id, "message", options=options)
            if options["free_mode"]:
                message, response = "Move freely.", {}
            else:
                message, response = llm(queries, options['llm_model'])

            if "gpt" in options["llm_model"]:
                role_content = ("system", f"agent{agent_id} say, " + message)
            else:
                role_content = (f"agent{agent_id}", message)
            messages[target_id].append(role_content)

            info["responses"].append(str(response))
            info["queries"].append(queries)
            info["sended_messages"].append(message)

    # 行動を生成
    actions = generate_actions_llm(env, obs, messages, info, options)
    # 現在の方針を生成
    if options["is_add_pre_plan"]:
        generate_plans_llm(env, obs, messages, info, options)

    return actions, info

def generate_plans_llm(env:gym.Env, obs, messages, info, options:dict={}):
    pre_plan = [""] * env.agent_num
    for agent_id in range(env.agent_num):
        queries = get_llm_query(env, obs, agent_id, "plan", messages[agent_id], options)
        if options["free_mode"]:
            plan, response = "I'll achieve my mission.", {}
        else:
            plan, response = llm(queries, options['llm_model']) 
        pre_plan[agent_id] = plan
        info["plans"].append(plan) 
        info["responses"].append(str(response))
        info["queries"].append(queries)
    options["pre_plan"] = pre_plan

def generate_actions_llm(env:gym.Env, obs, messages, info, options:dict={}):
    for agent_id in range(env.agent_num):
        queries = get_llm_query(env, obs, agent_id, "action", messages[agent_id], options)
        
        if "llama" in options["llm_model"]:
            action = generate_action_from_llama_agent(queries, agent_id)
            reason_str = ""
            action_str = action_to_str(action)
            response = ""
        else:
            if options["free_mode"]:
                action_str, response = action_to_str(env.action_space.sample()[0]), {}
            else:
                action_str, response = llm(queries, options['llm_model'])
            
            is_add_reason = False
            if "is_add_reason" in options.keys():
                is_add_reason = options["is_add_reason"]
            
            if is_add_reason:
                action_reason = action_str.splitlines()
                action_str = action_reason[0]
                reason_str = "\n".join(action_reason[1:])
            else:
                reason_str = ""    
            action = str_to_action(action_str)

        info["actions"].append(action)
        info["reasons"].append(reason_str)
        info["actions_str"].append(action_str)
        info["responses"].append(str(response))
        info["queries"].append(queries)
    return info["actions"]

def get_llm_query(env:gym.Env, obs, agent_id:int, query_type:str, messages:list[tuple[str,str]] = [], options:dict={}):
    result = []

    # 設定の説明
    system_prompt = ("system", f'You are an excellent strategist named agent{agent_id}. You are successful if achieve "mission" as soon as possible.')
    result.append(system_prompt)

    # 状況の説明
    obs_prompt = ("system", obs_to_str(env, obs, agent_id, options))
    result.append(obs_prompt)

    # メッセージ
    messages_prompt = [message for message in messages]
    # 会話でない場合はここでメッセージを提示
    if query_type != "conversation":
        result.extend(messages_prompt)

    # 過去の行動
    if "pre_action_count" in options.keys():
        count = options["pre_action_count"]
        actions = options["pre_action"]
        reasons = options["pre_reason"]

        is_add_reason = False
        if "is_add_reason" in options.keys():
            is_add_reason = options["is_add_reason"]

        if is_add_reason:
            count = 1

        count = min(count, len(actions))

        if count > 0:
            actions = actions[-count:]
            actions_str = [action_to_str(actions[i][agent_id]) for i in range(len(actions))]
            joined = ", ".join(actions_str)
            if count == 1:
                reason = reasons[-1][agent_id]
                reason_prompt = f' because of "{reason}"' if is_add_reason else ''
                past_prompt = ('system', f'One step ago you selected "{joined}"{reason_prompt}.')
            else:
                past_prompt = ('system', f'Your past actions are, {joined}, in order from the past.')
            result.append(past_prompt)

    # 方針
    if "is_add_pre_plan" in options.keys() and options["is_add_pre_plan"] and len(options["pre_plan"]) == env.agent_num:
        plan = options["pre_plan"][agent_id]
        result.append(("system", f"One step ago you say, {plan}"))

    # 行動の説明
    action_prompt = ("system", 'You can select an action in followings.\n- "left" to turn left\n- "right" to turn right\n- "forward" to go 1 step forward\n- "pick up" to pick up object in front\n- "drop" to drop the object you are carrying to front\n- "toggle" to open door in front')
    result.append(action_prompt)
    
    # 指示
    if query_type == "action":
        if is_add_reason:
            instruction = 'Which is the best action? Output only your selection and the reason in 2 different lines.'
        else:
            instruction = 'Which is the best action? Output only action name.'
    elif query_type == "message":
        target = options["message_target"]
        instruction = f'You are in a cooperative relationship with {target}. To achieve mission, Output only message to {target} in one or two sentence.'
    elif query_type == "conversation":
        target = options["message_target"]
        instruction = f'You are in a cooperative relationship with {target} and discussing to decide next action. To achieve mission, output only reply message to them in one or two sentence.'
    elif query_type == "plan":
        instruction = f'Output your current status and future plan in one or two sentences.'

    instruction_prompt = ("system", instruction)
    
    result.append(instruction_prompt)

    # 会話の場合はメッセージを最後につける
    if query_type == "conversation":
        result.extend(messages_prompt)

    # 同じroleはまとめる？
    result_arrange = []
    for message in result:
        if len(result_arrange) > 0 and message[0] == result_arrange[-1][0]:
            result_arrange[-1] = (result_arrange[-1][0], result_arrange[-1][1] + "\n" + message[1])
        else:
            result_arrange.append(message)

    return result_arrange

def load_llama(model_name:str):
    if not llama["model"] is None:
        return
    llama["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    llama["model"] = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        cache_dir=KEY.model_dir,
        low_cpu_mem_usage=True
    )
    llama["pipeline"] = transformers.pipeline(
        "text-generation",
        model=llama["model"],
        tokenizer=llama["tokenizer"],
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    llama["terminators"] = [
        llama["pipeline"].tokenizer.eos_token_id,
        llama["pipeline"].tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

def llama_create(messages:list):
    pipeline = llama["pipeline"]
    terminators = llama["terminators"]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    response = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        pad_token_id=pipeline.tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    text = response[0]["generated_text"][len(prompt):]
    return text, response

def generate_action_from_llama_agent_cliff(messages:list, obs:int):
    messages_format = [{"role": message[0], "content": message[1]} for message in messages]
    pipeline = llama["pipeline"]
    tokenizer = llama["tokenizer"]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages_format,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
    eos_token_id = tokenizer.eos_token_id

    output = llama["model"].generate(
        input_ids=inputs,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        max_new_tokens=64,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
        return_dict_in_generate=True,
        output_logits=True,
    )
    logits = output.logits[0][0].tolist()
    logits = [logits[455], logits[1315], logits[2996], logits[2414]]
    observation = [0.0] * 48
    observation[obs] = 1.0
    logits += observation
    action = llama["agents"][0].get_action(logits)
    return action

def generate_action_from_llama_agent(messages:list, agent_id):
    messages_format = [{"role": message[0], "content": message[1]} for message in messages]
    pipeline = llama["pipeline"]
    tokenizer = llama["tokenizer"]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages_format,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
    eos_token_id = tokenizer.eos_token_id

    output = llama["model"].generate(
        input_ids=inputs,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        max_new_tokens=64,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
        return_dict_in_generate=True,
        output_logits=True,
    )
    logits = output.logits[0][0].tolist()

    action = llama["agents"][agent_id].get_action(logits)
    return action

def load_flan(model_name):
    hyperparams = {
        "model_name": model_name,
        "env": "Blackjack-v1",
        "lora/r": 16,
        "lora/lora_alpha": 32,
        "lora/lora_dropout": 0.05,
        "lora/bias": "none",
        "lora/task_type": "CAUSAL_LM",
        "load_in_8bit": True,
        "batch_size": 8,
        "seed": 42069,
        "episodes": 5000,
        "generate/max_new_tokens": 32,
        "generate/do_sample": True,
        "generate/top_p": 0.6,
        "generate/top_k": 0,
        "generate/temperature": 0.9,
    }
    device = "cuda"
    HF_TOKEN = os.environ.get("HF_TOKEN")

    lora_config = LoraConfig(
        **{
            key.split("/")[-1]: value
            for key, value in hyperparams.items()
            if key.startswith("lora/")
        }
    )
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        pretrained_model_name_or_path=hyperparams["model_name"],
        peft_config=lora_config,
        load_in_8bit=hyperparams["load_in_8bit"],
        token=HF_TOKEN,
    ).to(device)
    tokenizer = T5Tokenizer.from_pretrained(hyperparams["model_name"], device_map=device, token=HF_TOKEN)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.pretrained_model.resize_token_embeddings(len(tokenizer))

    flan["agent"] = FlanAgent(
        model,
        tokenizer,
        device,
        {
            key: value
            for key, value in hyperparams.items()
            if key.startswith("generate/")
        },
        {
            "batch_size": hyperparams["batch_size"],
            "mini_batch_size": hyperparams["batch_size"],
        },
    )
def flan_create(messages):
    return flan["agent"].act(messages), {}

def train(reward):
    if flan["agent"] is not None:
        flan["agent"].assign_reward(reward)
        flan["agent"].terminate_episode()
    if llama["agents"] is not None:
        for i in range(len(llama["agents"])):
            llama["agents"][i].train_brief(reward[i])

def llm(messages:list, model_name:str):
    messages_format = [{"role": message[0], "content": message[1]} for message in messages]
    
    if "llama" in model_name:
        text, response = llama_create(
            messages = messages_format,
        )
    elif "gpt" in model_name:
        response = gpt["client"].chat.completions.create(
            model = model_name,
            messages = messages_format
        )
        text = response.choices[0].message.content
    elif "flan" in model_name:
        text, response = flan_create(
            messages = messages_format
        )
    return text, response

def str_to_action(action_str:str):
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
        match = re.compile(action).search(action_str)
        if match: return value
    return 0

def str_to_action_cliff(action_str:str):
    action_to_idx = {
        "north":0,
        "east":1,
        "south":2,
        "west":3,
    }
    for action, value in action_to_idx.items():
        match = re.compile(action).search(action_str)
        if match: return value
    return 0

def action_to_str(action_idx:int):
    actions = ["left","right","forward","pick up","drop","toggle","done"]
    return actions[action_idx]

policies = {
    "command" : command_policy,
    "simple_llm" : simple_llm_policy,
    "message_llm" : message_llm_policy,
    "conversation_llm" : conversation_llm_policy,
    "random" : random_policy
}