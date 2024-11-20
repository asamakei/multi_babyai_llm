import base64
from io import BytesIO
from PIL import Image

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
