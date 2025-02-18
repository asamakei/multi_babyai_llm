import utils.utils as utils
from utils.embedding_utils import Embedder

# エピソードの履歴を保持するクラス
class History(utils.Jsonable):
    def __init__(self, base_query: str, task_info: str, memory: list[str]) -> None:
        super().__init__()
        self.base_query: str = self.get_base_query(base_query, task_info, memory)
        self.history: list[dict[str, str, str]] = []
        self.labels = ['action', 'observation', 'relative_observation', 'feedback', 'message', 'subgoal', 'consideration', 'count', 'result']
        self.indexes: dict[int, int] = {}

    # 履歴をラベル付けして追加
    def add(self, label: str, value) -> None:
        assert label in self.labels
        if label == 'count':
            step = value
        else:
            step = self.history[-1]['step']
        self.history.append({
            'label': label,
            'value': value,
            'step': step
        })

    # エピソードが終わった時の履歴消去
    def reset(self) -> None:
        self.history = []
        self.indexes = {}

    # 履歴から特定ラベルの内容を消去
    def remove(self, label) -> None:
        self.history = [h for h in self.history if h['label'] != label]

    def __str__(self) -> str:
        return self.get_str()

    def get_str(self, length=-1, select=[], counts={}):
        contents: list[str] = []
        contents.append(self.base_query)
        contents.append("")
        contents.append('The following is your history:')

        last_step = self.history[-1]['step'] if len(self.history[-1]) > 0 else 0
        if length < 0: length = last_step + 1

        for item in self.history:

            label, value, step = item['label'], item['value'], item['step']
            length_for_label = utils.get_value(counts, label, length)

            if step + length <= last_step: continue
            if step + length_for_label <= last_step: continue
            if len(select) > 0 and label not in select: continue

            if label == 'action':
                contents.append(f'Your action: {value}')
            elif label == 'result':
                contents.append(f'result: {value}')
            elif label == 'subgoal':
                contents.append(f'subgoals: {value}')
            elif label == 'count':
                contents.append(f'time {value}:')
            else:
                contents.append(str(value))

        return "\n".join(contents)

    # プロンプトの最初に記述する説明文を生成
    def get_base_query(self, base_query: str, task_info: str, memories: list[str]) -> str:
        contents:list[str] = []
        contents.append(base_query)
        contents.append("")

        # 反省文ががあれば追加
        if len(memories) > 0:
            contents.append("Your memory for the task below:")
            for i, memory in enumerate(memories):
                contents.append(f"Trial {i}:")
                contents.append(str(memory.strip()))
            contents.append("")

        contents.append("Here is the task:")
        contents.append(task_info)
        
        return "\n".join(contents)
    
# サブゴールを木構造で保持するクラス
class SubgoalTree(utils.Jsonable):
    def __init__(self, root_content:str=""):
        super().__init__()
        self.reset(root_content)
    
    def reset(self, root_content:str=""):
        self.node_count:int = 0
        self.content:list[str] = []
        self.parent:list[int] = []
        self.childrens:list[list[int]] = []
        self.now_node:int = -1
        self.failed_node:int = -1
        self.access:list[int] = []
        self.halfway_node = -1
        if len(root_content) > 0:
            self.append(root_content)
    
    def reset_access(self):
        self.access = [0] * len(self.access)
        self.now_node = 0

    def append(self, content:str, is_move:bool=True) -> int:
        new_node_id = self.node_count
        self.node_count += 1
        self.content.append(content)
        self.parent.append(self.now_node)
        self.childrens.append([])
        self.access.append(0)
        if self.now_node >= 0:
            self.childrens[self.now_node].append(new_node_id)
            if is_move: self.access[self.now_node] += 1
        if is_move:
            self.now_node = new_node_id
        return new_node_id
    
    def move_to_leaf(self):
        while True:
            children = self.childrens[self.now_node]
            access = self.access[self.now_node]
            if len(children) <= access:
                break
            target = children[access]
            if target == self.failed_node:
                break
            self.access[self.now_node] += 1
            self.now_node = target

    def append_failed_node(self):
        if self.failed_node >= 0: return
        self.failed_node = self.append("")

    def delete_one(self, node:int, apply_access:bool=False):
        parent = self.parent[node]
        index = self.childrens[parent].index(node)
        first_half = self.childrens[parent][:index]
        second_half = self.childrens[parent][index+1:]
        children = self.childrens[node]
        if apply_access:
            children = children[:self.access[node]]
        for child in children:
            self.parent[child] = parent
            first_half.append(child)
        self.childrens[parent] = first_half + second_half
        self.parent[node] = -1
        if apply_access:
            self.access[parent] -= 1
            self.access[parent] += self.access[node]

    def delete(self):
        parent = self.parent[self.now_node]
        self.delete_one(self.now_node, True)
        self.now_node = parent

    def delete_subtree(self, node:int):
        parent = self.parent[node]
        self.parent[node] = -1
        subtree = self.dfs(node)
        if self.failed_node in subtree:
            index = self.childrens[parent].index(node)
            self.childrens[parent][index] = self.failed_node
            self.parent[self.failed_node] = parent
        else:
            self.childrens[parent].remove(node)

    def move_up(self) -> int:
        # 不要だったものを削除
        self.childrens[self.now_node] = self.childrens[self.now_node][:self.access[self.now_node]]
        parent = self.parent[self.now_node]
        if parent >= 0:
            self.now_node = parent
        return self.now_node
    
    def get_now_subgoal(self) -> str:
        return self.content[self.now_node]

    def get_subgoals(self) -> list[str]:
        node = self.now_node
        result = []
        while node >= 0:
            result.append(self.content[node])
            node = self.parent[node]
        return result

    def get_separated_sequence(self, separate_node:int, is_separate_in_front:bool=True) -> tuple[list[str], list[str]]:
        first_half, second_half = self.get_separated_sequence_ids(separate_node, is_separate_in_front)
        first_half_str = [self.content[n] for n in first_half]
        second_half_str = [self.content[n] for n in second_half]
        return first_half_str, second_half_str

    def get_separated_sequence_ids(self, separate_node:int, is_separate_in_front:bool=True) -> tuple[list[int], list[int]]:
        nodes:list[int] = self.dfs(0)
        if separate_node not in nodes:
            return nodes, []
        index = nodes.index(separate_node)
        if not is_separate_in_front: index += 1
        first_half = nodes[:index]
        second_half = nodes[index:]
        if self.failed_node in first_half:
            first_half.remove(self.failed_node)
        if self.failed_node in second_half:
            second_half.remove(self.failed_node)
        return first_half, second_half

    def get_all_sequence(self) -> tuple[list[str], list[str]]:
        return self.get_separated_sequence(self.failed_node)
    
    def get_achieved_not_achieved(self) -> tuple[list[str], list[str]]:
        return self.get_separated_sequence(self.now_node)

    def dfs(self, node:int) -> list[int]:
        result:list[int] = []
        for next in self.childrens[node]:
            result.extend(self.dfs(next))
        result.append(node)
        return result
    
    def get_leaf_count(self, node:int=0) -> int:
        children = self.childrens[node].copy()
        if self.failed_node in children:
            children.remove(self.failed_node)
        result = 0
        if len(children) == 0: return 1
        for n in children:
            result += self.get_leaf_count(n)
        return result

    def get_max_depth(self, node:int=0) -> int:
        if node == self.failed_node: return 0
        result = 1
        for n in self.childrens[node]:
            result = max(self.get_max_depth(n)+1, result)
        return result

    def remove_failed_node(self):
        self.now_node = 0
        if self.failed_node < 0: return
        parent = self.parent[self.failed_node]
        self.childrens[parent].remove(self.failed_node)
        self.failed_node = -1

    def reduction(self):
        def merge(first_node:int, second_node:int):
            for node in self.childrens[second_node]:
                self.childrens[first_node].append(node)
                self.parent[node] = first_node
            parent = self.parent[second_node]
            self.childrens[parent].remove(second_node)
            self.parent[second_node] = -1

        def is_nodes_similar(node1:int, node2:int):
            content1 = self.content[node1]
            content2 = self.content[node2]
            #is_similar = content1 == content2
            is_similar = Embedder.get_similarity(content1, content2) > 0.9
            return is_similar

        if not Embedder.is_loaded:
            Embedder.load()

        while True:
            nodes = self.dfs(0)
            is_changed = False
            for i in range(1, len(nodes)):
                pre_node = nodes[i-1]
                node = nodes[i]
                if is_nodes_similar(node, pre_node):
                    if self.parent[pre_node] == node:
                        self.delete_one(node)
                    else:
                        self.delete_subtree(node)
                    is_changed = True
                    break
            if is_changed: continue

            for node in nodes:
                children = self.childrens[node]
                for j in range(1, len(children)):
                    first_node = children[j-1]
                    second_node = children[j]
                    if is_nodes_similar(first_node, second_node):
                        merge(first_node, second_node)
                        is_changed = True
                        break
                # if is_changed: break
                # for child in children:
                #     if is_nodes_similar(node, child):
                #         self.delete_one(child)
                #         is_changed = True
                #         break
                if is_changed: break
            if not is_changed: break
    
    def is_after_halfway_node(self):
        if self.halfway_node == -1: return True
        achieved, not_achieved = self.get_separated_sequence_ids(self.now_node)
        if self.halfway_node in achieved: return True
        if self.halfway_node not in achieved and self.halfway_node not in not_achieved: return True
        return False

    def extract(self):
        node = 0
        achieved, _ = self.get_separated_sequence_ids(self.failed_node)
        self.remove_failed_node()
        self.halfway_node = -1 # 達成した後はサブゴールを生成しても良いというノード
        while True:
            if node in achieved:
                self.halfway_node = node
                break
            children = self.childrens[node]
            if len(children) == 0:
                self.halfway_node = -1
                break
            for child in children[1:]:
                if child not in achieved:
                    self.delete_subtree(child)
            if self.childrens[node][0] not in achieved:
                node = children[0]
            else:
                self.halfway_node = self.childrens[node][-1]
                break
        print(self.halfway_node)

class Memory(utils.Jsonable):
    def __init__(self, memory_size:int):
        super().__init__()
        self.size = memory_size
        self.contents:list[str] = []
        self.trim_memory()

    def add_memory(self, content:str):
        self.contents.append(content)
        self.trim_memory()

    def trim_memory(self):
        if len(self.contents) > self.size:
            self.contents = self.contents[-self.size:]