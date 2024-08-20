import re
import gym

from openai import OpenAI

import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import KEY


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
}

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
    generate_plans_llm(env, obs, messages, info, options)

    return actions, info

def message_llm_policy(env:gym.Env, obs, options:dict={}):
    info = {
        "queries":[],
        "responses":[],
        "sended_messages":[],
        "plans":[],
        "actions":[],
        "actions_str":[],
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
        if options["free_mode"]:
            action_str, response = action_to_str(env.action_space.sample()[0]), {}
        else:
            action_str, response = llm(queries, options['llm_model'])
            
        action = str_to_action(action_str)
        info["actions"].append(action)
        info["actions_str"].append(action_str)
        info["responses"].append(str(response))
        info["queries"].append(queries)
    return info["actions"]

def get_llm_query(env:gym.Env, obs, agent_id:int, query_type:str, messages:list[tuple[str,str]] = [], options:dict={}):
    result = []

    # 設定の説明
    system_prompt = ("system", f'You are an excellent strategist named agent{agent_id}. You are successful if achieve "mission" as soon as possible selecting actions each turn.')
    result.append(system_prompt)

    # 状況の説明
    obs_prompt = ("system", obs_to_text(env, obs, agent_id, options))
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
        count = min(count, len(actions))
        if count > 0:
            actions = actions[-count:]
            actions_str = [action_to_str(actions[i][agent_id]) for i in range(len(actions))]
            joined = ", ".join(actions_str)
            if count == 1:
                past_prompt = ('system', f'One step ago you selected "{joined}".')
            else:
                past_prompt = ('system', f'Your past actions are, {joined}, in order from the past.')
            result.append(past_prompt)

    # 方針
    if "is_add_pre_plan" in options.keys() and options["is_add_pre_plan"] and len(options["pre_plan"]) == env.agent_num:
        plan = options["pre_plan"][agent_id]
        result.append(("system", f"One step ago you say, {plan}"))

    # 行動の説明
    action_prompt = ("system", 'You can select an action in followings.\n- "left" to turn left\n- "right" to turn right\n- "forward" to go 1 step forward\n- "pick up" to pick up object in front\n- "drop" to drop the object you are carrying to front\n- "toggle" to open door in front.')
    result.append(action_prompt)
    
    # 指示
    if query_type == "action":
        instruction = 'Which is the best action? Output only result.'
    elif query_type == "message":
        target = options["message_target"]
        instruction = f'You are in a cooperative relationship with {target}. To achieve mission efficiently, Output only message to {target} in one or two sentence.'
    elif query_type == "conversation":
        target = options["message_target"]
        instruction = f'You are in a cooperative relationship with {target} and discussing. To achieve mission efficiently, output only reply message to them in one or two sentence.'
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


def obs_to_text(env:gym.Env, observations, agent_id:int, options:dict={}) -> list[str]:
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
    
    def one_agent(obs):
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

    return one_agent(observations[agent_id])

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
        cache_dir="/home/data/asama",
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

def llama_create(model:str, messages:list):
    load_llama(model)
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

def llm(messages:list, model_name:str):
    messages_format = [{"role": message[0], "content": message[1]} for message in messages]
    
    if "llama" in model_name:
        text, response = llama_create(
            model = model_name,
            messages = messages_format
        )
    elif "gpt" in model_name:
        response = gpt["client"].chat.completions.create(
            model = model_name,
            messages = messages_format
        )
        text = response.choices[0].message.content
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