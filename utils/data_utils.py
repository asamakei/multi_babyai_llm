import utils.utils as utils

# エピソードの履歴を保持するクラス
class History(utils.Jsonable):
    def __init__(self, base_query: str, task_info: str, memory: list[str]) -> None:
        super().__init__()
        self.base_query: str = self.get_base_query(base_query, task_info, memory)
        self.history: list[dict[str, str, str]] = []
        self.labels = ['action', 'observation', 'feedback', 'message', 'subgoal', 'consideration', 'count', 'result']
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
        if len(root_content) > 0:
            self.append(root_content)

    def append(self, content:str, is_move:bool=True) -> int:
        new_node_id = self.node_count
        self.node_count += 1
        self.content.append(content)
        self.parent.append(self.now_node)
        self.childrens.append([])
        if self.now_node >= 0:
            self.childrens[self.now_node].append(new_node_id)
        if is_move:
            self.now_node = new_node_id
        return new_node_id

    def append_failed_node(self):
        if self.failed_node >= 0: return
        self.failed_node = self.append("")

    def delete(self):
        parent = self.parent[self.now_node]
        index = self.childrens[parent].index(self.now_node)
        first_half = self.childrens[parent][:index]
        second_half = self.childrens[parent][index+1:]
        for child in self.childrens[self.now_node]:
            self.parent[child] = parent
            first_half.append(child)
        self.childrens[parent] = first_half + second_half
        self.parent[self.now_node] = -1
        self.now_node = parent

    def move_up(self) -> int:
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
        nodes:list[int] = self.dfs(0)
        index = nodes.index(separate_node)
        if not is_separate_in_front: index += 1
        first_half = nodes[:index]
        second_half = nodes[index:]
        if self.failed_node in first_half:
            first_half.remove(self.failed_node)
        if self.failed_node in second_half:
            second_half.remove(self.failed_node)
        first_half_str = [self.content[n] for n in first_half]
        second_half_str = [self.content[n] for n in second_half]
        return first_half_str, second_half_str

    def get_all_sequence(self) -> tuple[list[str], list[str]]:
        return self.get_separated_sequence(self.failed_node)

    def dfs(self, node:int) -> list[int]:
        result:list[int] = []
        for next in self.childrens[node]:
            result.extend(self.dfs(next))
        result.append(node)
        return result
    
    def get_leaf_count(self, node:int=0) -> int:
        if node == self.failed_node and len(self.childrens[self.parent[node]]) > 1:
            return 0
        if len(self.childrens[node]) == 0:
            return 1
        result = 0
        for n in self.childrens[node]:
            result += self.get_leaf_count(n)
        return result

    def get_max_depth(self, node:int=0) -> int:
        if node == self.failed_node: return 0
        result = 1
        for n in self.childrens[node]:
            result = max(self.get_max_depth(n)+1, result)
        return result

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