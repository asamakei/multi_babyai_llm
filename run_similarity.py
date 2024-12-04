from utils.llm_utils import LLM
import utils.utils as utils

def main():
    config = {
        "llm_model" : "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "free_mode" : False
    }
    LLM.load(config)

    v = []
    # > 0.9
    v.append(LLM.get_internal_representation("go to green key"))
    v.append(LLM.get_internal_representation("go to purple box"))
    v.append(LLM.get_internal_representation("get the green key"))
    v.append(LLM.get_internal_representation("find the purple box"))

    for i in range(len(v)):
        for j in range(i, len(v)):
            similarity = utils.get_cos_similarity(v[i], v[j])
            print(f"v{i}, v{j} : {similarity}")

if __name__ == "__main__":
    main()