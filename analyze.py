import os
import json
import matplotlib.pyplot as plt
import japanize_matplotlib

from gym_minigrid.minigrid import COLOR_TO_IDX

def get_directories(path:str):
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

def get_log_pathes(target_directories, start_date):
    log_pathes = {}
    for directory in target_directories:
        dates = get_directories(f"./{directory}")
        for date in dates:
            log_path = f"./{directory}/{date}/log.json"

            if not os.path.isfile(log_path): continue # ファイルが存在しない場合は除外
            if date <= start_date: continue # 古いデータは除外

            with open(log_path) as f:
                log = json.load(f)[0]
            
            if len(log["steps"]) > 101: continue # 長すぎるデータは除外
            if len(log["steps"]) == 50: continue # 旧データは除外

            env_name = log["env_name"]
            message_count = len(log["steps"][0]["info"]["sended_messages"])
            policy_name = {1:"oneside", 2:"twoside", 6:"conversation"}[message_count]
            if env_name not in log_pathes: log_pathes[env_name] = {}
            if policy_name not in log_pathes[env_name]: log_pathes[env_name][policy_name] = []
            log_pathes[env_name][policy_name].append(log_path)

    return log_pathes

def get_step(pathes:list[str]):
    result = []
    for path in pathes:
        with open(path) as f:
            log = json.load(f)[0]
        step_len = len(log["steps"])
        if step_len == 99: step_len = 100
        result.append(step_len)
    return result

def get_ballkey_step(pathes:list[str]):
    colors = COLOR_TO_IDX.keys()
    result = []
    for path in pathes:
        with open(path) as f:
            log = json.load(f)[0]

        is_ok = False
        step = len(log["steps"])
        if step == 99: step = 100

        for step_data in log["steps"]:
            is_ball_ok = False
            is_key_ok = False
            queries = step_data["info"]["queries"]
            for query in queries:
                for message in query:
                    role = message[0]
                    content = message[1]
                    if not role == "system": continue
                    for color in colors:
                        if f"You are carrying {color} key" in content:
                            is_key_ok = True
                        if f"You are carrying {color} ball" in content:
                            is_ball_ok = True

                    if is_ball_ok and is_key_ok:
                        step = step_data["step"] - 1
                        is_ok = True
                        break

                if is_ok: break
            if is_ok: break
        result.append(step)
    return result

def get_door_step(pathes:list[str]):
    result = []
    for path in pathes:
        with open(path) as f:
            log = json.load(f)[0]

        is_ok = False
        step = len(log["steps"])
        if step == 99: step = 100

        for step_data in log["steps"]:
            queries = step_data["info"]["queries"]
            for query in queries:
                for message in query:
                    role = message[0]
                    content = message[1]
                    if role == "system" and "- opened door in" in content:
                        step = step_data["step"] - 1
                        is_ok = True
                        break
                if is_ok: break
            if is_ok: break
        result.append(step)
    return result

log_pathes = get_log_pathes(
    ["result/conversation_llm", "result/message_llm", 
     "result_00/conversation_llm", "result_00/message_llm", 
     "result_reason/conversation_llm", "result_reason/message_llm"],
    "20240822000000"
)

GOTO = "BabyAI-1RoomS20-v0"
BLOCK = "BabyAI-BlockedUnlockPickup-v0"

ONE = "oneside"
TWO = "twoside"
CON = "conversation"

steps_goto_one = get_step(log_pathes[GOTO][ONE])
steps_goto_two = get_step(log_pathes[GOTO][TWO])
steps_goto_con = get_step(log_pathes[GOTO][CON])

steps_block_one = get_step(log_pathes[BLOCK][ONE])
steps_block_two = get_step(log_pathes[BLOCK][TWO])
steps_block_con = get_step(log_pathes[BLOCK][CON])

ballkey_step_one = get_ballkey_step(log_pathes[BLOCK][ONE])
ballkey_step_two = get_ballkey_step(log_pathes[BLOCK][TWO])
ballkey_step_con = get_ballkey_step(log_pathes[BLOCK][CON])

door_step_one = get_door_step(log_pathes[BLOCK][ONE])
door_step_two = get_door_step(log_pathes[BLOCK][TWO])
door_step_con = get_door_step(log_pathes[BLOCK][CON])

steps_goto_list = [steps_goto_one, steps_goto_two, steps_goto_con]
steps_block_list = [steps_block_one, steps_block_two, steps_block_con]
ballkey_list = [ballkey_step_one, ballkey_step_two, ballkey_step_con]
door_list = [door_step_one, door_step_two, door_step_con]


def save_figure(data, name):
    fig, ax = plt.subplots()
    bp = ax.boxplot(data, whis=[0, 100])
    ax.set_xticklabels(['baseline', '提案手法1', '提案手法2'])
    plt.grid()
    plt.savefig(f"./analyze_result/{name}")

save_figure(steps_goto_list, "steps_goto")
save_figure(steps_block_list, "steps_block")
save_figure(ballkey_list, "steps_ballkey")
save_figure(door_list, "steps_door")        
