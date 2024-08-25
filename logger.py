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

class Logger:
    def __init__(self, path, name = ""):
        self.log = []
        self.path = path

        if not os.path.exists(self.path):
            os.mkdir(self.path)

        if self.path[-1] != '/': self.path += '/'
        dt_now = datetime.datetime.now()
        dt_str = dt_now.strftime('%Y%m%d%H%M%S')
        self.path = self.path + dt_str + name + "/"
        os.mkdir(self.path[:-1])

    def append(self, obj):
        self.log.append(obj)

    def clear(self):
        self.log = []

    def output(self, name="tmp"):
        with open(f"{self.path}{name}.json", 'w') as f:
            json.dump(self.log, f, indent=1, cls=NumpyJSONEncoder)
    
    def make_path(self, filename):
        return self.path + filename