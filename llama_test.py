import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import KEY
import json

#transformers.set_seed(42) # シード固定

model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

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

    print(f"prompt:{prompt}")

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    print("-----------output-------------")

    outputs = pipeline(
        prompt,
        max_new_tokens=64,
        eos_token_id=terminators,
        pad_token_id=pipeline.tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        #return_dict_in_generate=True,
        #output_logits=True,
    )
    print(outputs)

    continue
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            input_ids=inputs["input_ids"].to("cuda"),
            max_length= 512,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            return_dict_in_generate=True,
            output_logits=True,
        )
        output_text = tokenizer.batch_decode(output.sequences, skip_special_tokens = False)[0]
        logits = output.logits
        print(output.sequences)
        print(output_text)
        print(logits)
        print(len(logits))
        print(logits[0][0])
        print(logits[0][0].shape)
    #print(outputs[0]["generated_text"][len(prompt):])
