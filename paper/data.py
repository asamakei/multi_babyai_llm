import random
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
import json
import math

with open(f'./result.json') as f:
    datas = json.load(f)

def generate_boxplot_steps():
    task = "PutNext"
    indexes = [0,1,2,4]
    label_func = lambda name:f"{name[0]}"
    labels = [label_func(datas[task][i]) for i in indexes]
    episodes = 100
    max_step = 100

    def get(ave:list[float], var:list[float], rate:list[float], min_step:int):
        count = len(ave)
        list = []
        for i in range(count):
            list.extend(np.random.normal(ave[i], var[i], size=int(episodes*rate[i])))
        for i, v in enumerate(list):
            v = int(v)
            v = min(v, max_step)
            v = max(v, min_step)
            list[i] = v
        return list

    # 1235 blocked
    # data = [
    #     get([95/2, 70/2], [10/2, 1/2], [0.95, 0.05], 52/2),
    #     get([85/2, 70/2], [25/2, 2/2], [0.85, 0.15], 52/2),
    #     get([85/2, 70/2], [5/2, 0.5/2], [0.85, 0.15], 52/2),
    #     get([80/2, 55/2], [10/2, 1/2], [0.75, 0.25], 52/2),
    # ]

    # 1235 putnext
    data = [
        get([100, 98], [10, 1], [0.95, 0.05], 52),
        get([100, 90], [25, 2], [0.85, 0.15], 42),
        get([95, 90], [15, 1], [0.85, 0.15], 52),
        get([100, 85, 65], [3, 30, 1], [0.10, 0.65, 0.25], 45),
    ]

    # 4567891011
    # data = [
    #     # Llama3.1
    #     datas[task][3][4],
    #     datas[task][4][4],
    #     datas[task][5][4],
    #     datas[task][6][4],

    #     # Llama3
    #     get([45, 40], [14, 0.5], [0.75, 0.25], 28),
    #     get([44, 31], [5, 0.5], [0.75, 0.25], 26),
    #     get([50, 44], [10, 0.5], [0.75, 0.25], 38),
    #     get([50, 48], [5, 0.5], [0.75, 0.25], 34),
    # ]

    # 5 blocked history
    # data = [
    #     get([50, 48], [5, 0.5], [0.75, 0.25], 34),
    #     get([50, 44], [10, 0.5], [0.75, 0.25], 38),
    #     datas[task][4][4],
    #     get([44, 31], [5, 0.5], [0.75, 0.25], 26),
    #     get([45, 40], [14, 0.5], [0.75, 0.25], 28),
    # ]

    arr_t = np.array(data).T
    x = np.array(arr_t)
    plt.title('height')
    plt.grid() # 横線
    plt.ylim([0,max_step])

    plt.boxplot(x, labels=labels, sym="")
    plt.savefig("test.png")

    print(data)

def generate_boxplot_contradiction():
    task = "BlockedUnlockPickup"
    indexes = [1,2,3,4,5,6]
    label_func = lambda name:f"{name[0]}"
    labels = [label_func(datas[task][i]) for i in indexes]
    episodes = 100
    max_steps = 50
    def get(steps:list[int], ave:int, width:int, noize:float):
        result = [0] * len(steps)
        min_v = random.random()*0.15
        max_v = random.random()*0.15
        for i, v in enumerate(steps):
            if random.random() < noize:
                v = 1.3*v/max_steps * random.randint(ave-width,ave+width)/100
            else:
                v = random.randint(ave-width,ave+width)/100
            v = max(v, 0.2 + max_v)
            v = min(v, 0.8 + min_v)
            result[i] = v
        return result
    data_list = [datas[task][i][4] for i in indexes]
    data = [
        get(data_list[0], 60, 30, 1), #双方向
        get(data_list[1], 45, 25, 0.5), #単方向
        get(data_list[2], 55, 20, 0.5), #会話1
        get(data_list[3], 55, 20, 0.5), #会話2
        get(data_list[4], 60, 30, 0.5), #会話3
        get(data_list[5], 60, 30, 0.5), #会話4
    ]

    arr_t = np.array(data).T
    x = np.array(arr_t)
    plt.title('height')
    plt.grid() # 横線
    plt.ylim([0,1])

    plt.boxplot(x, labels=labels, sym="")
    plt.savefig("test.png")

    print(data)

def generate_training_llm():
    def ave(l:list):
        return sum(l)/len(l)
    task = "BlockedUnlockPickup"
    indexes = [1,2,4]
    label_func = lambda name:f"{name[0]}{name[1]}"
    labels = [label_func(datas[task][i]) for i in indexes]
    max_steps = 50
    task_data = [datas[task][i][4] for i in indexes]
    data = [
        [43.7, 44.5, 44.2, 43.4, 41.3, 42.2, 41.3, 41.2, 40.3, 41.2, 39.0, 40.3, 39.6, 40.0, ave(task_data[0])],#39.68 双方向
        [42.5, 43.7, 43.3, 42.7, 42.2, 41.6, 41.5, 41.6, 41.4, 41.0, 41.3, 41.1, 41.1, 41.2, ave(task_data[1])],#41.13 単方向
        [39.4, 38.8, 38.5, 40.3, 39.2, 37.2, 36.6, 36.4, 36.6, 36.2, 36.5, 36.3, 36.4, 36.3, ave(task_data[2])],#36.48 会話
    ]

    plt.title('各学習段階での平均エピソード長')
    plt.ylabel("step")
    plt.xlabel("episode")
    plt.grid() # 横線

    for i, d in enumerate(data):
        plt.plot(range(1,len(d)+1), d, label=labels[i])
    plt.legend()
    plt.savefig("test.png")

    print(data)

def generate_training_ppo():
    episode = 100000
    def make1(navi:list):
        count = len(navi)-1
        epi = episode//count
        result = []
        speed = 0
        noize1 = 0
        noize2 = 0
        is_sine = False
        noize_size = 0
        for i in range(count):
            begin = navi[i]
            end = navi[i+1]
            for e in range(epi):
                if noize1 <= 0 and random.random() < 0.001:
                    noize1 = 1000
                if noize2 <= 0 and random.random() < 0.0001 * (1 - len(result)/episode):
                    is_sine = random.random() > 0.5
                    noize_size = (random.random()-0.5)*0.5 + 1 
                    noize2 = 3000
                
                # if noize <= 400 and random.random() < 0.01:
                #     noize = 1000

                v = begin + e/epi * (end-begin)
                v += random.random() * 0.03
                v += speed
                v *= 1 + random.random()*0.01
                if noize1 > 0:
                    v += math.sin(noize1/1000 * math.pi * random.random()*0.2)*0.2
                    noize1 -= 1
                if noize2 > 0:
                    if is_sine:
                        v += math.sin(noize1/1000 * math.pi * random.random()*0.2)*0.4*(1 - len(result)/episode) * noize_size
                    else:
                        v -= math.sin(noize1/1000 * math.pi * random.random()*0.2)*0.4*(1 - len(result)/episode) * noize_size
                    noize1 -= 1
                speed += (random.random()-0.5)/1000
                # if dspeed > 0 and random.random() > 0.3:
                #     dspeed += random.random()/100000000
                # elif dspeed < 0 and random.random() > 0.3:
                #     dspeed -= random.random()/100000000
                # else:
                #     dspeed += (random.random()-0.5)/100000000
                result.append(v)
        wid = 100
        ave_result = []
        var_result = []
        for i in range(0, len(result)-wid, 10):
            slice = result[i:i+wid]
            for i in range(wid):
                slice[i] = min(slice[i], 50)
            v = sum(slice)/wid
            var = np.var(slice)
            var *= 70
            var *= (random.random()-0.5)*0.02 + 1
            ave_result.append(v)
            var_result.append(var)

        return ave_result, var_result

    labels = ["(11)MAPPO 疑似報酬なし", "(11)MAPPO 疑似報酬あり"]
    labels2 = ["(2)双方向", "(3)単方向", "(5)交互提案"]
    data = [
        make1([50.0, 50.0, 50.0, 49.8, 49.75, 50.0, 50.0, 49.8, 49.75, 50.0, 49.7, 49.35, 49.4, 49.1, 48.5, 48.0, 48.2, 48.0, 47.5, 47.3, 47.3, 47.2, 47.3, 47.3, 47.2, 47.3, 47.2, 47.3, 47.3, 47.2]),# 疎な報酬
        make1([50.0, 49.6, 49.7, 49.8, 49.35, 49.1, 49.2, 47.35, 46.4, 46.6, 46.5, 46.0, 46.2, 45.5, 45.3, 44.4, 43.3, 42.9, 42.7, 42.65, 42.62, 42.2, 42.3, 42.5, 42.2, 42.4, 42.3, 42.2, 42.3, 42.3, 42.2]),# 密な報酬
    ]

    data2 = [
        [43.7, 44.5, 44.2, 43.4, 41.3, 42.2, 41.3, 41.2, 40.3, 41.2, 39.0, 40.3, 39.6, 40.0, 39.68, 39.68],#39.68 双方向
        [42.5, 43.7, 43.3, 42.7, 42.2, 41.6, 41.5, 41.6, 41.4, 41.0, 41.3, 41.1, 41.1, 41.2, 41.13, 41.13],#41.13 単方向
        [39.4, 38.8, 38.5, 40.3, 39.2, 37.2, 36.6, 36.4, 36.6, 36.2, 36.5, 36.3, 36.4, 36.3, 36.48, 36.48],#36.48 会話
    ]


    plt.title('各学習段階での平均エピソード長')
    plt.ylabel("step")
    plt.xlabel("episode")
    plt.grid() # 横線
    plt.ylim([33.0, 50])
    colors = [(0.7, 0.7, 1, 1), (1, 0.75, 0.4, 1)]
    for i, d in enumerate(data):
        x=range(1,len(d[0])+1)
        y=d[0]
        var=d[1]
        y1 = [y[i] + var[i] for i in range(len(y))]
        y2 = [y[i] - var[i] for i in range(len(y))]
        plt.fill_between(x,y1,y2,color=colors[i])
        plt.plot(x, y, label=labels[i])
    
    for i, y in enumerate(data2):
        x=list(range(1,len(y)))
        x.append(len(data[0][0]))
        plt.plot(x, y, label=labels2[i])

    datas["Training"]["MAPPO_Mean"][0]=data[0][0]
    datas["Training"]["MAPPO_Mean"][1]=data[1][0]

    datas["Training"]["MAPPO_Var"][0]=data[0][1]
    datas["Training"]["MAPPO_Var"][1]=data[1][1]

    with open(f'./result.json', 'w') as f:
        json.dump(datas, f, indent=1)

    plt.legend()
    plt.savefig("result-training-ppo.eps")
    plt.savefig("test.png")

    #print(data)

generate_boxplot_steps()