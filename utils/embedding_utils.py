from sentence_transformers import SentenceTransformer, util
from utils.utils import get_value
import re

model : SentenceTransformer = None
name : str = ""
cache : dict = {}

def is_loaded():
    return model != None

def load(config={}):
    global model, name
    model_name = get_value(config, "embedding_model", "paraphrase-MiniLM-L6-v2")
    if model is not None and model_name == name: return
    model = SentenceTransformer(model_name, device="cpu")
    name = model_name

def get_embeddings(text):
    global cache
    if text in cache.keys():
        return cache[text]
    embeddings = model.encode(text, convert_to_tensor=True)
    cache[text] = embeddings
    return embeddings

def get_similarity(text1, text2):
    emb1 = get_embeddings(text1)
    emb2 = get_embeddings(text2)

    # 数字が違ったら異なるものとして判定する
    if extraction_numbers(text1) != extraction_numbers(text2):
        return 0
    
    cosine_score = util.pytorch_cos_sim(emb1, emb2)[0][0]
    return float(cosine_score)

def extraction_numbers(text):
    match_list = re.findall(r'\d+', text)
    result = ''.join(match_list)
    return result

if __name__ == "__main__":
    load()

    while True:
        print("text1:")
        text1 = input()
        print("text2:")
        text2 = input()
        
        similarity = get_similarity(text1, text2)
        print(similarity)