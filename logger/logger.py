import json
import os
import datetime
import numpy

from json import JSONEncoder

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, numpy.int64):
            return int(o)
        elif isinstance(o, numpy.uint8):
            return int(o)
        elif isinstance(o, numpy.bool_):
            return bool(o)
        elif isinstance(o, numpy.ndarray):
            return list(o)
        return JSONEncoder.default(self, o)

def output(content, path):
    with open(path, 'w') as f:
        json.dump(content, f, indent=1, cls=NumpyJSONEncoder)

def output_token_count(input:int, output:int):
    path = "./token_count.json"
    with open(path) as f:
        content = list(json.load(f))
    dt_now = str(datetime.datetime.now())
    content.append([dt_now, input, output])
    with open(path, "w") as f:
        json.dump(content, f, indent=1, cls=NumpyJSONEncoder)

class Logger:
    def __init__(self, path, name = "", is_create = True):
        self.log = []
        self.path = path

        if not is_create:
            self.path = self.path + "/"
            return

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if self.path[-1] != '/': self.path += '/'
        dt_now = datetime.datetime.now()
        dt_str = dt_now.strftime('%Y%m%d%H%M%S')
        self.path = self.path + dt_str + name

        if not os.path.exists(self.path):
            os.mkdir(self.path)

        self.path = self.path + "/"

    def append(self, obj):
        self.log.append(obj)

    def clear(self):
        self.log = []

    def output(self, name, content = None):
        if content == None:
            content = self.log
        path = f"{self.path}{name}.json"
        output(content, path)
    
    def make_path(self, filename):
        return self.path + filename