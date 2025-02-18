from gym_minigrid.minigrid import (
    IDX_TO_OBJECT,
    IDX_TO_COLOR,
)
import utils.utils as utils

DIRECTIONS_STR = ["east", "south", "west", "north"]

# エージェントの名前を返す
def get_agent_name(agent_id:int):
    return f"agent{agent_id}" if agent_id >= 0 else ""

def object_to_text(obj):
    id, color_id, state = obj 
    name = IDX_TO_OBJECT[id]
    color = IDX_TO_COLOR[color_id]
    result = ""
    if id in (0, 1, 3): # Invisible, Floor, None
        result = ""
    elif id in (2, 8, 9): # Wall, Goal, Lava
        result = name
    elif id == 4: # Door
        state_str = ["open ", "", "locked "][state]
        result = f"{color} {state_str}{name}"
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
def obs_to_str_baby(env, observations, params:dict={}) -> list:
    
    def object_to_detail_text(obj, params):
        if obj[0] >= 10:
            result = get_agent_info_str(obj[0] - 10)
        else:
            result = object_to_text(obj)
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
        carrying_obj_name = "no item"
        forward_obj_name = "no object"
        right_obj_name = "no object"
        left_obj_name = "no object"
        obj_texts = []
        agent_texts = []
        is_agent_observable = utils.get_value(params, "sight_include_agents", False)

        for y in range(height):
            for x in range(width):
                obj_name = object_to_detail_text(grid[x][y], params)
                if obj_name == "": continue

                local_pos = (width // 2 - x, height - y - 1)
                if local_pos == (0, 0): # 運んでいるオブジェクト名
                    carrying_obj_name = obj_name
                    continue
                if local_pos == (0, 1): # 正面にあるオブジェクト名
                    forward_obj_name = obj_name
                    continue
                elif local_pos == (-1, 0): # 右にあるオブジェクト名
                    right_obj_name = obj_name
                    continue
                elif local_pos == (1, 0): # 左にあるオブジェクト名
                    left_obj_name = obj_name
                    continue

                # ここではエージェント情報は飛ばす
                obj_id = grid[x][y][0]
                if is_agent_observable and obj_id >= 10:
                    continue

                world_pos = local_to_world_pos(local_pos, self_dir, self_pos)
                obj_text = f"There is {obj_name} at {world_pos}."
                obj_texts.append(obj_text)

        if is_agent_observable:
            for target_id in range(env.agent_num):
                if target_id == agent_id: continue
                agent_name = get_agent_info_str(target_id)
                target_pos = env.agents_pos[target_id]
                agent_text = f"{agent_name} is at {target_pos}."
                agent_texts.append(agent_text)

        forward_pos = local_to_world_pos((0, 1), self_dir, self_pos)
        right_pos = local_to_world_pos((-1, 0), self_dir, self_pos)
        left_pos = local_to_world_pos((1, 0), self_dir, self_pos)

        self_dir_str = DIRECTIONS_STR[self_dir]
        forward_dir = DIRECTIONS_STR[self_dir]
        right_dir = DIRECTIONS_STR[(self_dir+1)%len(DIRECTIONS_STR)]
        left_dir = DIRECTIONS_STR[self_dir-1]
        #mission = obs['mission']

        sentences = []

        # sentences.append(f'Your forward is coordinate {forward_pos}, which is {forward_dir}, there is {forward_obj_name}.') # 正面
        # sentences.append(f'Your right is cooridnate {right_pos}, which is {right_dir}, there is {right_obj_name}.') # 右
        # sentences.append(f'Your left is coordinate {left_pos}, which is {left_dir}, there is {left_obj_name}.') # 左
        sentences.append(f'Your forward is coordinate {forward_pos}, there is {forward_obj_name}.') # 正面
        sentences.append(f'Your right is cooridnate {right_pos}, there is {right_obj_name}.') # 右
        sentences.append(f'Your left is coordinate {left_pos}, there is {left_obj_name}.') # 左

        relative = " ".join(sentences)
        sentences = []

        sentences.append(f'Your position is coordinate {self_pos}. You are facing {forward_dir}') # 絶対位置
        sentences.append(f'You have {carrying_obj_name}.') # 所持品
        sentences.append(f"There is a passable floor in the area.") # 通行可能であることを説明
        sentences.extend(obj_texts) # 視界にあるオブジェクト
        sentences.extend(agent_texts) # 他エージェントの位置情報
        #sentences.append(f"There is {forward_obj_name} in your forward coordinate {forward_pos}.") # 目の前のオブジェクト
        #sentences.append(f"There is {right_obj_name} in your right coordinate {right_pos}.") # 右のオブジェクト
        #sentences.append(f"There is {left_obj_name} in your left coordinate {left_pos}.") # 左のオブジェクト
        absolute = utils.join_sentences(sentences)
        return (relative, absolute)

    return [one_agent(obs, agent_id) for agent_id, obs in enumerate(observations)]

def world_to_str_baby(env, is_add_position:bool, params:dict={}) -> str:
    grid = env.grid.encode()
    width = env.width
    height = env.height
    obj_texts = []
    for y in range(height):
        for x in range(width):
            obj = grid[x][y]
            if obj[0] not in (4, 5, 6, 7): continue
            obj_name = object_to_text(grid[x][y])
            world_pos = (x, y)
            if is_add_position:
                obj_name += f" at {world_pos}"
            obj_text = f"There is {obj_name}."
            obj_texts.append(obj_text)
    return " ".join(obj_texts)


def get_feedbacks(env, observations, actions:list[int], params:dict={}):

    def one_agent(agent_id) -> str:
        observation = observations[agent_id]
        action = actions[agent_id]
        forward_obj = local_to_obj(observation, (0, 1))
        forward_id, forward_color, forward_state = forward_obj
        forward_name = object_to_text(forward_obj)
        is_forward_empty = forward_id in (1, 3) or (forward_id == 4 and forward_state == 0) # Floor, None, Open door
        carrying_obj = local_to_obj(observation, (0, 0))
        carrying_id, carrying_color, _ = carrying_obj
        carrying_name = object_to_text(carrying_obj)
        is_carrying = carrying_id in (5, 6, 7)
        if action == 0:
            return "You turned left."
        elif action == 1:
            return "You turned right."
        elif action == 2:
            if is_forward_empty:
                return "You took a step forward."
            else:
                return "You couldn't took a step forward because there is an object in your forward coordinate."
        elif action == 3:
            if forward_id not in (5, 6, 7):
                return f"You failed to picked up because there is no item in your forward coordinate."                
            if is_carrying:
                return f"You failed to picked up because you already have other item."
            else:
                return f"You picked up {forward_name}."
        elif action == 4:
            if is_carrying and is_forward_empty:
                return f"You dropped {carrying_name}."
            elif is_forward_empty:
                return f"You failed to drop item because you have no item."
            else:
                return f"You failed to drop item because there is an object in your forward coordinate."
        elif action == 5:
            if forward_id == 4 and forward_state == 1:
                return f"You opened {forward_name}."
            elif forward_id == 4 and forward_state == 2 and carrying_id == 5 and forward_color == carrying_color:
                return f"You opened {forward_name} using the key."
            elif forward_id == 4 and forward_state == 2 and not (carrying_id == 5 and forward_color == carrying_color):
                color = IDX_TO_COLOR[forward_color]
                return f"You failed to open because you don't have {color} key."
            else:
                return f"You failed to open."
        return f"Nothing happened."

    return [one_agent(agent_id) for agent_id in range(env.agent_num)]
