from gym_minigrid.minigrid import (
    IDX_TO_OBJECT,
    IDX_TO_COLOR,
)
import utils.utils as utils

DIRECTIONS_STR = ["east", "south", "west", "north"]

# エージェントの名前を返す
def get_agent_name(agent_id:int):
    return f"agent{agent_id}" if agent_id >= 0 else ""

def object_to_text(obj, params):
    id, color_id, state = obj 
    name = IDX_TO_OBJECT[id]
    color = IDX_TO_COLOR[color_id]
    result = ""
    if id in (0, 1, 3): # Invisible, Floor, None
        result = ""
    elif id in (2, 8, 9): # Wall, Goal, Lava
        result = name
    elif id == 4: # Door
        state_str = ["open", "closed", "locked"][state]
        result = f"{color} {state_str} {name}"
    elif id in (5, 6, 7): # Key, Ball, Box
        result = f"{color} {name}"
    return result

def local_to_world_pos(pos, dir, pivot) -> tuple[int,int]:
    if dir == 0: # east
        pos = (pos[1], -pos[0])
    elif dir == 1: # south
        pos = (pos[0], pos[1])
    elif dir == 2: # west
        pos = (-pos[1], pos[0])
    elif dir == 3: # north
        pos = (-pos[0], -pos[1])
    return (pos[0]+pivot[0], pos[1]+pivot[1])

def local_to_obj(obs, pos):
    grid = obs["image"]
    height, width = grid.shape[0:2]
    x, y = (width // 2 - pos[0], height - pos[1] - 1)
    obj_id, color_id, state_id = grid[x][y]
    return obj_id, color_id, state_id

# 観測をテキストに変換する処理
def obs_to_str_baby(env, observations, params:dict={}) -> list[str]:
    
    def object_to_detail_text(obj, params):
        result = ""
        if obj[0] >= 10:
            if not utils.get_value(params, "sight_include_agents", False):
                result = get_agent_info_str(id - 10)
        else:
            result = object_to_text(obj, params)
        return result

    def get_agent_info_str(target_id:str):
        target_name = get_agent_name(target_id)
        direction = observations[target_id]["direction"]
        direction_str = DIRECTIONS_STR[direction]
        result = f"{target_name} facing {direction_str}"
        return result

    def one_agent(obs, agent_id):
        grid = obs["image"]
        height, width = grid.shape[0:2]

        self_dir = obs['direction']
        self_pos = env.agents_pos[agent_id]
        is_carrying = False
        carrying_info = "no item"
        forward_obj_name = "no object"
        obj_texts = []

        for y in range(height):
            for x in range(width):
                obj_name = object_to_detail_text(grid[x][y], params)
                local_pos = (width // 2 - x, height - y - 1)
                world_pos = local_to_world_pos(local_pos, self_dir, self_pos)
                
                if obj_name == "":
                    continue
                elif local_pos == (0, 0):
                    is_carrying = True
                    carrying_info = obj_name
                    continue

                if local_pos == (0, 1):# 正面の時の例外
                    forward_obj_name = obj_name

                obj_text = f"{obj_name} is at {world_pos}."
                obj_texts.append(obj_text)

        if utils.get_value(params, "sight_include_agents", False):
            for target_id in range(env.agent_num):
                if target_id == agent_id: continue
                obj_name = get_agent_info_str(target_id)
                target_pos = env.agents_pos[target_id]
                obj_text = f"{obj_name} is at {target_pos}."
                obj_texts.append(obj_text)

        forward_pos = local_to_world_pos((0, 1), self_dir, self_pos)
        right_pos = local_to_world_pos((-1, 0), self_dir, self_pos)
        left_pos = local_to_world_pos((1, 0), self_dir, self_pos)

        self_dir_str = DIRECTIONS_STR[self_dir]
        mission = obs['mission']

        sentences = []
        sentences.append(f'Your mission is "{mission}".') # 目標
        #sentences.append(f'You are looking at {self_dir_str}.') # 方角
        sentences.append(f'You are at {self_pos}.') # 絶対位置
        sentences.append(f'Your forward is {forward_pos}.') # 正面
        sentences.append(f'Your right is {right_pos}') # 右
        sentences.append(f'Your left is {left_pos}.') # 左
        sentences.append(f'You have {carrying_info}.') # 所持品
        if is_carrying: sentences.append(f'So you cannot pick up any more.')
        sentences.extend(obj_texts) # 視界にあるオブジェクト
        sentences.append(f"There is {forward_obj_name} in your forward.") # 目の前のオブジェクト
        result = utils.join_sentences(sentences)
        return result

    return [one_agent(obs, agent_id) for agent_id, obs in enumerate(observations)]

def get_feedbacks(env, observations, actions:list[int], params:dict={}):

    def one_agent(agent_id) -> str:
        observation = observations[agent_id]
        action = actions[agent_id]
        forward_obj = local_to_obj(observation, (0, 1))
        forward_id, forward_color, forward_state = forward_obj
        forward_name = object_to_text(forward_obj, params)
        is_forward_empty = forward_id in (1, 3) or (forward_id == 4 and forward_state == 0) # Floor, None, Open door
        carrying_obj = local_to_obj(observation, (0, 0))
        carrying_id, carrying_color, _ = carrying_obj
        carrying_name = object_to_text(carrying_obj, params)
        is_carrying = carrying_id in (5, 6, 7)
        if action == 0:
            return "You turned left."
        elif action == 1:
            return "You turned right."
        elif action == 2:
            if is_forward_empty:
                return "You took a step forward."
            else:
                return "You couldn't took a step forward."
        elif action == 3:
            if is_carrying:
                return f"You failed to picked up."
            elif forward_id in (5, 6, 7):
                return f"You picked up {forward_name}."
            else:
                return f"You failed to picked up."
        elif action == 4:
            if is_carrying and is_forward_empty:
                return f"You dropped {carrying_name}."
            else:
                return f"You failed to drop."
        elif action == 5:
            if forward_id == 4 and forward_state == 1:
                return f"You opened {forward_name}"
            elif forward_id == 4 and forward_state == 2 and carrying_id == 5 and forward_color == carrying_color:
                return f"You opened {forward_name}"
            else:
                return f"You failed to open."
        return f"Nothing happened."
    

    return [one_agent(agent_id) for agent_id in range(env.agent_num)]
