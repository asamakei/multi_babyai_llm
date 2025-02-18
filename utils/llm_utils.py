import os
import utils.utils as utils

from openai import OpenAI
import tiktoken

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
from logger.logger import output_token_count

# 初期化とテキスト生成の機能を持ったLLM
class LLM:

    __llm = None
    __llm_high = None

    image_token:str = ""

    @staticmethod
    def __make(model_name) -> 'LLM':
        if "llama" in model_name:
            if "llava" in model_name:
                llm_instance = Llava(model_name)
            elif "Vision" in model_name:
                llm_instance = LlamaVision(model_name)            
            else:
                llm_instance = Llama(model_name)
        elif "gpt" in model_name:
            llm_instance = Gpt(model_name)
        elif "flan" in model_name:
            llm_instance = Flan(model_name)
        else:
            llm_instance = LLM("free")
        return llm_instance
    
    @classmethod
    def load(cls, params):
        is_free_mode = utils.get_value(params, "free_mode", False)

        # 通常のモデルを読み込み
        model_name = params["llm_model"] if not is_free_mode else "free"
        if cls.__llm is None or model_name != cls.__llm.model_name:
            cls.__llm = cls.__make(model_name)

        # よりハイレベルなモデル(通常はGPTを想定)を読み込み
        high_model_name = utils.get_value(params, "llm_high_model", "none")
        if high_model_name == "none":
            cls.__llm_high = cls.__llm
        else:
            high_model_name = high_model_name if not is_free_mode else "free"
            if cls.__llm_high is None or high_model_name != cls.__llm_high.model_name:
                cls.__llm_high = cls.__make(high_model_name)

    @staticmethod
    def __generate(llm:'LLM', prompt:str, image):
        if image is not None:
            return llm._generate_text_with_vision(prompt, image)
        return llm._generate_text(prompt)

    @classmethod
    def generate(cls, prompt:str, image = None) -> str:
        return cls.__generate(cls.__llm, prompt, image)

    @classmethod
    def generate_high(cls, prompt:str, image = None) -> str:
        return cls.__generate(cls.__llm_high, prompt, image)

    @classmethod
    def get_internal_representation(cls, text:str):
        return cls.__llm._generate_internal_representation(text)

    # テキスト同士の類似度を取得する
    @classmethod
    def get_similarity(cls, text1:str, text2:str) -> float:
        return cls.__llm._get_similarity(text1, text2)

    # LLMの初期化処理
    def __init__(self, model_name):
        self.model_name = model_name
        self.internal_representations = {}

    # プロンプトをChat形式に変換
    def _prompt_format(self, prompt):
        return [{"role": "system", "content": prompt}]

    # 画像付きプロンプトをChat形式に変換
    def _prompt_format_vision(self, prompt, image):
        return [{"role": "system", "content": prompt}]

    # プロンプトをもとに応答を生成
    def _generate_text(self, prompt):
        return "I don't know.", {}

    # プロンプトと画像をもとに応答を生成
    def _generate_text_with_vision(self, prompt, image):
        return self._generate_text(prompt)
    
    # 入力の潜在表現を取得
    def _generate_internal_representation(self, prompt):
        return np.array([0])
    
    def _get_similarity(self, text1, text2) -> float:
        v1 = self._generate_internal_representation(text1)
        v2 = self._generate_internal_representation(text2)
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
    
    def _generate_text(self, prompt):
        message = self._prompt_format(prompt)
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
    
    def _generate_internal_representation(self, prompt):
        if prompt in self.internal_representations.keys():
            return self.internal_representations[prompt]
        message = self._prompt_format(prompt)
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
        LLM.image_token = "<|image|>"

    def _generate_text(self, prompt):
        return "", {}
    
    def _generate_text_with_vision(self, prompt, image):
        message = self._prompt_format(prompt)
        input_text = self.processor.apply_chat_template(message, add_generation_prompt=True)
        inputs = self.processor(image, input_text, return_tensors="pt").to(self.model.device)
        response = self.model.generate(**inputs, max_new_tokens=512)
        text = self.processor.decode(response[0][1:-1])[len(input_text):]
        return text, {"output":self.processor.decode(response[0][1:-1])}

class Llava(LLM):
    def __init__(self, model_name):
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
        LLM.image_token = "<image>"

    def _prompt_format(self, prompt):
        #return f"[INST] {prompt} [/INST]"
        return [
            {"role": "user","content": [{"type": "text", "text": prompt},],},
        ]

    def _generate_text(self, prompt):
        return "", {}
    
    def _generate_text_with_vision(self, prompt, image):
        message = self._prompt_format(prompt)
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
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.input_token = 0
        self.output_token = 0
    
    def _prompt_format(self, prompt, image = None):
        if image is None: return super()._prompt_format(prompt)

        image_base64 = utils.np_image_to_base64(image)
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url":  f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }]

    def _token_checker(self, input, output):
        input_token_len = self._calc_token(input)
        output_token_len = self._calc_token(output)
        self.input_token += input_token_len
        self.output_token += output_token_len
        output_token_count(input_token_len, output_token_len)

        cost = self.input_token / 1000000 * 0.15 + self.output_token / 1000000 * 0.6
        if cost > 0.45:
            print(f"input:{self.input_token}")
            print(f"output:{self.output_token}")
            print(f"cost:{cost}")
            import sys
            a = 1
            b = a[1]
            print(b)
            sys.exit()

    def _call_api(self, prompt, image = None):
        messages = self._prompt_format(prompt, image)
        response = self.client.chat.completions.create(
            model = self.model_name,
            messages = messages
        )
        text = response.choices[0].message.content

        self._token_checker(prompt, text)
        return text, {}        

    def _calc_token(self, text):
        encoding = tiktoken.get_encoding(self.encoding.name)
        num_tokens = len(encoding.encode(text))
        return num_tokens

    def _generate_text(self, prompt):
        return self._call_api(prompt)

    def _generate_text_with_vision(self, prompt, image):
        return self._call_api(prompt, image)

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

    def _generate_text(self, prompt):
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