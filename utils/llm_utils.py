import os
import utils.utils as utils

from openai import OpenAI

import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    T5Tokenizer,
    MllamaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    AutoProcessor,
    LlavaNextProcessor
)

import numpy as np
import torch
import ENV

image_token:str = ""
def get_image_token():
    return image_token

# 初期化とテキスト生成の機能を持ったLLM
class LLM:
    # LLMの初期化処理
    def __init__(self, model_name):
        self.model_name = model_name
        self.internal_representations = {}

    # プロンプトをChat形式に変換
    def prompt_format(self, prompt):
        return [{"role": "system", "content": prompt}]

    # 画像付きプロンプトをChat形式に変換
    def prompt_format_vision(self, prompt, image):
        return [{"role": "system", "content": prompt}]

    # プロンプトをもとに応答を生成
    def generate_text(self, prompt):
        return "I don't know.", {}

    # プロンプトと画像をもとに応答を生成
    def generate_text_with_vision(self, prompt, image):
        return self.generate_text(prompt)
    
    # 入力の潜在表現を取得
    def generate_internal_representation(self, prompt):
        return np.array([0])
    
    def get_similarity(self, text1, text2) -> float:
        v1 = self.generate_internal_representation(text1)
        v2 = self.generate_internal_representation(text2)
        return utils.get_cos_similarity(v1, v2)

class Llama(LLM):
    def __init__(self, model_name):
        super().__init__(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            cache_dir=ENV.model_dir,
            low_cpu_mem_usage=True
        )

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    
    def generate_text(self, prompt):
        message = self.prompt_format(prompt)
        query = self.pipeline.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        response = self.pipeline(
            query,
            max_new_tokens=256,
            eos_token_id=self.terminators,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        text = response[0]["generated_text"][len(query):]
        text = response[0]["generated_text"][len(query):]
        return text, response
    
    def generate_internal_representation(self, prompt):
        if prompt in self.internal_representations.keys():
            return self.internal_representations[prompt]
        message = self.prompt_format(prompt)
        query = self.pipeline.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(query, return_tensors="pt")["input_ids"].to("cuda")
        eos_token_id = self.tokenizer.eos_token_id
        output = self.model.generate(
            input_ids=inputs,
            do_sample=True,
            temperature=0.6, # 0.6
            top_p=0.9, # 0.9
            max_new_tokens=1,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
            return_dict_in_generate=True,
            output_logits=True,
            output_hidden_states=True,
        )
        result_tensor = output.hidden_states[0][-1][0][-1]
        internal_representation = result_tensor.to('cpu').detach().numpy().copy()
        self.internal_representations[prompt] = internal_representation
        return internal_representation

class LlamaVision(LLM):
    def __init__(self, model_name):
        global image_token
        super().__init__(model_name)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            device_map="auto"
            #torch_dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        image_token = "<|image|>"

    def generate_text(self, prompt):
        return "", {}
    
    def generate_text_with_vision(self, prompt, image):
        message = self.prompt_format(prompt)
        input_text = self.processor.apply_chat_template(message, add_generation_prompt=True)
        inputs = self.processor(image, input_text, return_tensors="pt").to(self.model.device)
        response = self.model.generate(**inputs, max_new_tokens=512)
        text = self.processor.decode(response[0][1:-1])[len(input_text):]
        return text, {"output":self.processor.decode(response[0][1:-1])}

class Llava(LLM):
    def __init__(self, model_name):
        global image_token
        super().__init__(model_name)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,  # 消す?
            quantization_config=quantization_config,
        )
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        image_token = "<image>"

    def prompt_format(self, prompt):
        #return f"[INST] {prompt} [/INST]"
        return [
            {"role": "user","content": [{"type": "text", "text": prompt},],},
        ]

    def generate_text(self, prompt):
        return "", {}
    
    def generate_text_with_vision(self, prompt, image):
        message = self.prompt_format(prompt)
        input_text = self.processor.apply_chat_template(message, add_generation_prompt=True)
        inputs = self.processor(input_text, image, return_tensors="pt").to(self.model.device)
        response = self.model.generate(
            **inputs,
            max_new_tokens=512
        )
        text = self.processor.decode(response[0][1:-1])[len(input_text):]
        return text, {"output":self.processor.decode(response[0])}

class Gpt(LLM):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.client = OpenAI(api_key = ENV.openai_api_key)
    
    def prompt_format(self, prompt, image = None):
        if image is None: return super().prompt_format(prompt)

        image_base64 = utils.np_image_to_base64(image)
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url":  f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }]

    def call_api(self, prompt, image = None):
        messages = self.prompt_format(prompt, image)
        response = self.client.chat.completions.create(
            model = self.model_name,
            messages = messages
        )
        text = response.choices[0].message.content
        return text, {}        

    def generate_text(self, prompt):
        return self.call_api(prompt)

    def generate_text_with_vision(self, prompt, image):
        return self.call_api(prompt, image)

class Flan(LLM):
    def __init__(self, model_name):
        super().__init__(model_name)
        hyperparams = {
            "model_name": model_name,
            "env": "Blackjack-v1",
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

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=hyperparams["model_name"],
            load_in_8bit=hyperparams["load_in_8bit"],
            token=HF_TOKEN,
        ).to(device)

        self.tokenizer = T5Tokenizer.from_pretrained(model_name, device_map=device, token=HF_TOKEN)
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.model.pretrained_model.resize_token_embeddings(len(self.tokenizer))

    def generate_text(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            inputs=inputs.input_ids,
            return_dict_in_generate=True,
            output_logits=True,
            **{
                key.split("/")[-1]: value
                for key, value in self.generate_config_dict.items()
            }
        )
        generate_ids = output.sequences
        response = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        text = response[0].split("[/INST]")[-1].strip()

        return text, response

# 何度もロードしなくて良いようにグローバル変数に保持
llm_api:LLM = None

# 事前に読み込んだLLMを使用する関数
def llm(prompt:str, image = None):
    global llm_api
    if image is not None:
        return llm_api.generate_text_with_vision(prompt, image)
    return llm_api.generate_text(prompt)

# テキストの潜在表現を取得する
def get_internal_representation(text:str):
    global llm_api
    return llm_api.generate_internal_representation(text)

# テキスト同士の類似度を取得する
def get_similarity(text1:str, text2:str) -> float:
    global llm_api
    return llm_api.get_similarity(text1, text2)

# LLMを読み込む
def load_llm(params):
    global llm_api

    model_name = params["llm_model"]

    # すでにロードされていたら読み込まない
    if llm_api is not None and model_name == llm_api.model_name:
        return

    # ミスしてGPTを沢山使うと嫌なので念の為
    if params["free_mode"]:
        llm_api = LLM("free")
        return

    # モデル名によってそれぞれのロード処理を呼ぶ
    if "llama" in model_name:
        if "llava" in model_name:
            llm_api = Llava(model_name)
        elif "Vision" in model_name:
            llm_api = LlamaVision(model_name)            
        else:
            llm_api = Llama(model_name)
    elif "gpt" in model_name:
        llm_api = Gpt(model_name)
    elif "flan" in model_name:
        llm_api = Flan(model_name)
    else:
        llm_api = LLM("free")