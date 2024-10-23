import utils.env_utils as env_utils
import utils.llm_utils as llm_utils

# エピソードの履歴を保持するクラス
class History:
    def __init__(self, base_query: str, task_info: str, memory: list[str]) -> None:
        self.base_query: str = f'{self.get_base_query(base_query, task_info, memory)}'
        self.history: list[dict[str, str]] = []

    # 履歴をラベル付けして追加
    def add(self, label: str, value: str) -> None:
        assert label in ['action', 'observation', 'message', 'result']
        self.history += [{
            'label': label,
            'value': value,
        }]

    # エピソードが終わった時の履歴消去
    def reset(self) -> None:
        self.history = []

    def __str__(self) -> str:
        s: str = self.base_query + '\n'
        for i, item in enumerate(self.history):
            label = item['label']
            value = item['value']
            if label == 'action':
                s += f'> {value}'
            elif label == 'observation':
                s += value
            elif label == 'message':
                s += value
            elif label == 'result':
                s += f'result: {value}'
            if i != len(self.history) - 1:
                s += '\n'
        return s

    # プロンプトの最初に記述する説明文を生成
    def get_base_query(self, base_query: str, task_info: str, memory: list[str]) -> str:
        query = base_query

        # 反省文ががあれば追加
        if len(memory) > 0:
            query += '\n\nYour memory for the task below:'
            for i, m in enumerate(memory):
                query += f'\nTrial {i}:\n{m.strip()}'

        query += f"\nHere is the task:\n{task_info}"
        return query

# Reflexionに関する全般処理を行うクラス
# エピソードを超えてメモリを保持する
class Reflexion:
    def __init__(self, params:dict):
        self.env_name = params["env_name"]
        self.agent_num = params["agent_num"]
        self.memory_size = params["reflexion_memory_size"]
        self.memories = [[] for _ in range(self.agent_num)]

        base, task = env_utils.get_base_task_prompt(params)
        
        self.histories = [History(
            base,
            task,
            [],
        ) for _ in range(self.agent_num)]
    
    # 履歴を追加
    def add_histories(self, label:str, contents:list[str]):
        for i in range(self.agent_num):
            self.histories[i].add(label, contents[i])

    # メッセージを履歴に追加
    def add_message(self, agent_id:int, sender_name:str, content:str):
        self.histories[agent_id].add("message", f"{sender_name}:{content}")

    # Reflexionのメイン処理
    def run(self, is_success: bool, reason: str, params:dict):
        # タスクの達成状況を履歴に追加
        result = "SUCCESS" if is_success else f"FALIED"
        self.add_histories("result", [result] * self.agent_num)

        # 達成していたらReflexionを行わない
        if is_success: return self.memories

        # 指示文を取得
        inst = env_utils.get_reflexion_prompt(reason, params)

        # エージェントの数だけ実行
        for i in range(self.agent_num):
            # 反省文を出力
            prompt = str(self.histories[i]) + "\n" + inst
            text, _ = llm_utils.llm(prompt)

            # メモリに追加
            self.memories[i].append(text)
            if len(self.memories[i]) > self.memory_size:
                self.memories[i] = self.memories[i][-self.memory_size:]

        return self.memories
    
    # エピソードが終わった時に反省文だけ引き継いで履歴をリセットする処理
    def reset(self, params):
        base, task = env_utils.get_base_task_prompt(params)
        self.histories = [History(
            base,
            task,
            self.memories[i]
        ) for i in range(self.agent_num)]

    # メッセージを生成する時のプロンプトを返す
    def get_message_prompt(self, agent_id:int, targets:list[str]):
        self.get_communication_prompt(agent_id, targets, "message")

    # 会話文を生成する時のプロンプトを返す
    def get_conversation_prompt(self, agent_id:int, targets:list[str]):
        self.get_communication_prompt(agent_id, targets, "conversation")
    
    # 会話文またはメッセージを生成する時のプロンプトを返す
    def get_communication_prompt(self, agent_id:int, targets:list[str], label:str):
        history = str(self.histories[agent_id])
        instr = env_utils.get_communication_prompt(label, targets)
        prompt = history + " " + instr + "\n>"
        return prompt
    
    # 行動を生成する際のプロンプトを返す
    def get_action_prompt(self, agent_id:int):
        history = str(self.histories[agent_id])
        instr = env_utils.get_action_prompt(self.env_name)
        prompt = history + " " + instr + "\n>"
        return prompt