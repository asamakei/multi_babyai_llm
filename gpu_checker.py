import subprocess
import time
from datetime import datetime

# GPUぶん取りプログラム
# バックグラウンドで実行 → GPUを常に監視して指定容量が空いた瞬間に指定のコマンドを自動で実行する
# 使うと嫌われます

# user settings
require = 42 #GB
cmd = "python main.py"
# ------------

DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)

def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]

print(str(datetime.now()) + " start checking", flush=True)
while True:
    info = get_gpu_info()
    memory_free = int(info[0]["memory.free"])
    if memory_free > require * 1000:
        print(str(datetime.now()) + " get memory", flush=True)
        subprocess.check_output(cmd, shell=True)
        break
    time.sleep(15) # 15秒に一度チェック
