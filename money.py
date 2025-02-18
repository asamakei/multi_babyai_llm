import json
from datetime import datetime

gpt_cost = {
    "4o-mini":{
        "input" : 0.15,
        "output" : 0.60
    },
    "4o":{
        "input" : 2.5,
        "output" : 10
    }
}

model = "4o"

start = datetime.strptime("2024-12-28 15:35:00", '%Y-%m-%d %H:%M:%S')
end = datetime.strptime("2024-12-28 17:00:00", '%Y-%m-%d %H:%M:%S')

path = "./token_count.json"
with open(path) as f:
    content = list(json.load(f))

input_tokens = 0
output_tokens = 0
for date_str, input, output in content:
    date = datetime.strptime(date_str[:19], '%Y-%m-%d %H:%M:%S')
    if not (start <= date and date <= end): continue
    input_tokens += input
    output_tokens += output

input_cost = input_tokens / 1000000 * gpt_cost[model]["input"]
output_cost = output_tokens / 1000000 * gpt_cost[model]["output"]
cost = input_cost + output_cost
cost = int(cost * 1000) / 1000

print(f"input: {input_tokens} tokens")
print(f"output: {output_tokens} tokens")
print(f"cost: {cost} $")