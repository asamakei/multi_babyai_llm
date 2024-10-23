import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import KEY
import json

from before.dqn_agent import DQNAgent

model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    cache_dir=KEY.model_dir,
    low_cpu_mem_usage=True
)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
)

agent = DQNAgent(128256, 6, "cuda")

pre_obs = None

while(True):
    print("-----------input-------------")
    q = input()
    if q == "quit" or q == "q" or q == "quit()":
        break
    
    with open(f'./llama_test_prompt.json') as f:
        messages = json.load(f)

    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
    print("-----------output-------------")
    #eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_token_id = tokenizer.eos_token_id
    output = model.generate(
        input_ids=inputs,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        max_new_tokens=64,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
        return_dict_in_generate=True,
        output_logits=True,
        output_hidden_states=True,
    )
    logits = output.logits[0][0]
    input_text = tokenizer.batch_decode(inputs, skip_special_tokens = True)[0]
    text = tokenizer.batch_decode(output.sequences, skip_special_tokens = True)[0]
    print(logits)
    print(logits.shape)
    print(text[len(input_text):])
    print(output.hidden_states[0])
    print(output.hidden_states[0][-1])
    print(len(output.hidden_states))
    print(len(output.hidden_states[0]))
    print(output.hidden_states[0][-1].shape) # エンコーダ？ torch.Size([1, 344, 8192]) [_, inputのトークン数, 潜在表現の大きさ？]
    print(output.hidden_states[1][-1].shape) # デコーダ？　torch.Size([1, 1, 8192]) [_, outputのトークン数, 潜在表現の大きさ？]
    print("----------")
    print(output.hidden_states[0][-1][0][-1])

    # obs = logits.tolist()
    # action = agent.get_action(logits.tolist())
    # print(f"action:{action}")

    # if pre_obs is not None:
    #     agent.train(obs, action, 1, pre_obs)
    # pre_obs = obs
