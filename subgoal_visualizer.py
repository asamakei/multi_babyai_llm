import os
import json

from PIL import Image, ImageDraw, ImageFont

import utils.utils as utils
from utils.reflexion_utils import SubgoalTree

def save_image(directory_path:str, trial:int, image:Image.Image):
    image_dir_path = f"{directory_path}/subgoal_tree"
    if not os.path.exists(image_dir_path):
        os.mkdir(image_dir_path)
    image_path = f"{image_dir_path}/trial{trial}_0.png"
    image.save(image_path)

def draw_subgoaltree(tree:SubgoalTree) -> Image:
    margin = (60, 30)
    node_size = (240, 50)

    vertical_count = tree.get_leaf_count()
    horizontal_count = tree.get_max_depth()

    img_width = horizontal_count * node_size[0] + (horizontal_count + 1) * margin[0]
    img_height = vertical_count * node_size[1] + (vertical_count + 1) * margin[1]

    img = Image.new('RGB', (img_width, img_height), (255, 255, 255))

    positions:list[tuple[int, int]] = [None] * len(tree.content)

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 20)

    def draw_node(node:int, content:str, is_achieved:bool):
        color = (127, 255, 127) if is_achieved else (255, 127, 127)
        position_center = positions[node]
        position = (
            position_center[0] - node_size[0] / 2,
            position_center[1] - node_size[1] / 2,
            position_center[0] + node_size[0] / 2,
            position_center[1] + node_size[1] / 2,
        )

        font_size = font.getbbox(content)
        font_holizontal_size = font_size[2] - font_size[0]
        if font_holizontal_size > node_size[0]:
            words = content.split(" ")
            center = len(words) // 2
            content = " ".join(words[:center]) + "\n" + " ".join(words[center:])

        draw.rectangle(position, fill=color, outline=(0, 0, 0))
        draw.text(position_center, content, (0, 0, 0), font=font, anchor='mm')

    def draw_edge(parent:int, child:int):
        if parent < 0 or child < 0: return
        position_parent = positions[parent]
        position_child = positions[child]
        begin = (position_parent[0] + node_size[0] / 2, position_parent[1])
        end = (position_child[0] - node_size[0] / 2, position_child[1])
        draw.line((begin, end), fill=(0, 0, 0), width = 2)

    def draw_by_dfs(node:int=0, row:int=0, col:int=0, is_achieved:bool=True):
        positions[node] = (
            col * node_size[0] + (col + 1) * margin[0] + node_size[0] / 2,
            row * node_size[1] + (row + 1) * margin[1] + node_size[1] / 2,
        )
        
        is_leaf_node = len(tree.childrens[node]) == 0
        is_failed_node = node == tree.failed_node
        child_count = 0

        if is_leaf_node:
            if not is_failed_node:
                child_count = 1
            is_achieved &= not is_failed_node
        else:
            for child in tree.childrens[node]:
                count, achieved = draw_by_dfs(child, row+child_count, col+1, is_achieved)
                child_count += count
                is_achieved &= achieved

        if not is_failed_node:
            draw_edge(tree.parent[node], node)
            draw_node(node, tree.content[node], is_achieved)

        return child_count, is_achieved

    draw_by_dfs()

    return img

def load_subgoaltree(log_path:str) -> SubgoalTree:
    if not os.path.exists(log_path): return None
    with open(log_path) as f:
        log = json.load(f)
    if len(log) <= 1 or not "subgoal_tree" in log[1].keys():
        return None
    tree_dict = log[1]["subgoal_tree"][0]
    tree = SubgoalTree()
    tree.set_dict(tree_dict)
    return tree

def main(directory_name:str, trials:list[int]):
    if directory_name[0] == ".":
        if directory_name[-1] == "/":
            directory_name = directory_name[:-1]
        directory_name = directory_name.split("/")[-1]
    directory_path = utils.search_directory_path(directory_name)
    if directory_path is None: return
    for trial in trials:
        log_path = f"{directory_path}/log_trial{trial}.json"
        tree = load_subgoaltree(log_path)
        if tree is None:
            print(f"[WARNING] Trial{trial} log don't exists or isn't valid.")
            break
        tree.reduction()
        image = draw_subgoaltree(tree)
        save_image(directory_path, trial, image)
    return

if __name__ == "__main__":
    directory_name = "20241202202051_Debug"
    trials = [1]#list(range(100))
    main(directory_name, trials)
    print(f"[INFO] The process is terminated.")