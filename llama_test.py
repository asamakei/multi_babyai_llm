import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import KEY

#transformers.set_seed(42) # シード固定

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    #device_map="auto",
    cache_dir=KEY.model_dir,
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
)

messages = [
    {"role": "system",
     "content": "You are an excellent strategist. You are successful if achieve \"mission\" as soon as possible selecting actions each turn."},
    {"role": "system",
     "content": """Your mission is \"go to the green key\". You are facing east. The followings is in your sight.
- wall in 4 steps forward and 1 step left
- wall in 4 steps forward
- wall in 4 steps forward and 1 step right
- wall in 4 steps forward and 2 steps right
- wall in 4 steps forward and 3 steps right
- wall in 3 steps forward and 1 step left
- wall in 2 steps forward and 1 step left
- green key in 2 steps forward and 2 steps right
- wall in 1 step forward and 1 step left
- wall in 1 step left"""},
    {"role": "system",
     "content": "You must select \"left\" to turn left, \"right\" to turn right, \"forward\" to go forward 1 step, \"pickup\" to pick up object is in front of you, \"drop\" to drop the object you are carrying to the front or \"toggle\" to open/close box or door in front of you. Which is the best action? Output only result."},
]

prompt = pipeline.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print(f"prompt:{prompt}")

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=64,
    eos_token_id=terminators,
    pad_token_id=pipeline.tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])