import base64
from io import BytesIO
from PIL import Image
import numpy as np
import glob

class Jsonable:
    def get_dict(self) -> dict:
        return vars(self)

    def set_dict(self, data:dict):
        for key, value in data.items():
            setattr(self, key, value)

def dict_of_lists_extend(a:dict, b:dict):
    for key, item in b.items():
        if not isinstance(item, list):continue
        if key in a.keys():
            if isinstance(a[key], list):
                a[key].extend(item)
        else:
            a[key] = item

def get_value(dict:dict, key:str, default):
    if key in dict.keys():
        return dict[key]
    return default

def np_image_to_base64(img, format="jpeg") -> str:
    pil_image = Image.fromarray(img)
    buffer = BytesIO()
    pil_image.save(buffer, format)
    img_str = base64.b64encode(buffer.getvalue()).decode("ascii")
    return img_str

def initial_to_upper(text:str) -> str:
    return text[:1].upper() + text[1:]

def join_sentences(sentences:list[str]) -> str:
    uppers = list(map(initial_to_upper, sentences))
    result = ' '.join(uppers)
    return result

def get_cos_similarity(v1, v2) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def remove_edge_symbol(text:str, symbol:str) -> str:
    if text[0] == symbol:
        text = text[1:]
    if text[-1] == symbol:
        text = text[:-1]
    return text
    
def search_directory_path(directory_name:str) -> str:
    path_list = glob.glob(f"./result/*/*/{directory_name}", recursive=True) 
    result = None
    if len(path_list) == 0:
        print(f"[ERROR] Directory '{directory_name}' don't exists.")
    elif len(path_list) > 1:
        print(f"[ERROR] There is more than one directory '{directory_name}'")
    else:
        result = path_list[0]
    return result