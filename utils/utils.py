def get_value(dict:dict, key:str, default):
    if key in dict.keys():
        return dict["key"]
    return default