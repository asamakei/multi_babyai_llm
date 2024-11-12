import utils.env_utils as env_utils
import utils.llm_utils as llm_utils
import utils.utils as utils

# エピソードの履歴を保持するクラス
class History:
    def __init__(self, base_query: str, task_info: str, memory: list[str], history_size: int) -> None:
        self.base_query: str = f'{self.get_base_query(base_query, task_info, memory)}'
        self.history: list[dict[str, str, str]] = []
        self.history_size = history_size

    # 履歴をラベル付けして追加
    def add(self, label: str, value: str) -> None:
        assert label in ['action', 'observation', 'message', 'result']
        step = 0
        if len(self.history) > 0:
            step = self.history[-1]['step']
            if self.history[-1]['label'] == 'action':
                step += 1

        self.history += [{
            'label': label,
            'value': value,
            'step': step
        }]

        self.trim()

    # historyを一定の長さに保持する
    def trim(self):
        if len(self.history) == 0: return
        if self.history_size <= 0: return
        last_step = self.history[-1]["step"]
        first_step = last_step - self.history_size
        index = 0
        while self.history[index]["step"] <= first_step:
            index += 1
        self.history = self.history[index:]

    # エピソードが終わった時の履歴消去
    def reset(self) -> None:
        self.history = []

    def __str__(self) -> str:
        return self.get_history_str()
    
    def get_history_str(self, is_without_past_messages=False):
        s: str = self.base_query + '\n'
        last_step = self.history[-1]['step']
        for i, item in enumerate(self.history):
            label = item['label']
            value = item['value']
            step = item['step']
            if label == 'action':
                s += f'> {value}'
            elif label == 'observation':
                s += value
            elif label == 'message':
                if is_without_past_messages and step < last_step:
                    continue
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
    def __init__(self, env, params:dict):
        self.env = env
        self.env_name = params["env_name"]
        self.agent_num = params["agent_num"]
        self.memory_size = params["reflexion_memory_size"]
        self.memories = [[] for _ in range(self.agent_num)]
        history_size = utils.get_value(params, "reflexion_history_size", 0)

        base, task = env_utils.get_base_task_prompt(env, params)
        
        self.histories = [History(
            base,
            task,
            [],
            history_size
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

        is_use_vision = utils.get_value(params,"is_use_vision", False)
        imgs = self.env.render_masked() if is_use_vision else []

        # エージェントの数だけ実行
        for agent_id in range(self.agent_num):
            # 指示文を取得
            instr = env_utils.get_reflexion_prompt(reason, agent_id, params)
            # 反省文を出力
            prompt = str(self.histories[agent_id]) + "\n" + instr
            if is_use_vision:
                text, _ = llm_utils.vlm(prompt, imgs[agent_id])
            else:
                text, _ = llm_utils.llm(prompt)

            # メモリに追加
            self.memories[agent_id].append(text)
            if len(self.memories[agent_id]) > self.memory_size:
                self.memories[agent_id] = self.memories[agent_id][-self.memory_size:]

        return self.memories
    
    # エピソードが終わった時に反省文だけ引き継いで履歴をリセットする処理
    def reset(self, env, params):
        self.env = env
        base, task = env_utils.get_base_task_prompt(env, params)
        history_size = utils.get_value(params, "reflexion_history_size", 0)
        self.histories = [History(
            base,
            task,
            self.memories[i],
            history_size
        ) for i in range(self.agent_num)]

    # メッセージを生成する時のプロンプトを返す
    def get_message_prompt(self, agent_id:int, targets:list[str], params:dict):
        return self.get_communication_prompt(agent_id, targets, "message", params)

    # 会話文を生成する時のプロンプトを返す
    def get_conversation_prompt(self, agent_id:int, targets:list[str], params:dict):
        return self.get_communication_prompt(agent_id, targets, "conversation", params)
    
    # 会話文またはメッセージを生成する時のプロンプトを返す
    def get_communication_prompt(self, agent_id:int, targets:list[str], label:str, params:dict):
        if utils.get_value(params, "without_past_messages", False):
            history = self.histories[agent_id].get_history_str(True)
        else:
            history = str(self.histories[agent_id])
        instr = env_utils.get_communication_prompt(label, agent_id, targets, params)
        agent_name = env_utils.get_agent_name(agent_id)
        prompt = f"{history}\n\n{instr}\n{agent_name}:"
        return prompt
    
    # 行動を生成する際のプロンプトを返す
    def get_action_prompt(self, agent_id:int, params:dict):
        if utils.get_value(params, "without_past_messages", False):
            history = self.histories[agent_id].get_history_str(True)
        else:
            history = str(self.histories[agent_id])
        instr = env_utils.get_action_prompt(self.env_name, agent_id, params)
        prompt = f"{history}\n\n{instr}\n>"
        return prompt