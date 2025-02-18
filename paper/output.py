import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
import json
import matplotlib.gridspec as gridspec

with open(f'./result.json') as f:
    datas = json.load(f)

def output_boxplot(task:str, indexes:list[int], max_step:int, title:str, label_maker, filename:str):
    plt.figure(figsize=(4, 6))
    task_result = datas[task]
    data = [task_result[i][4] for i in indexes]
    #labels = [f"{task_result[i][0]}{task_result[i][1]}" for i in indexes]
    labels = [label_maker(task_result[i]) for i in indexes]
    x = np.array(np.array(data).T)
    plt.title(title)
    plt.ylabel("step")
    # plt.xlabel("ラウンド")
    plt.ylim([0, max_step])
    plt.grid()
    plt.boxplot(x, labels=labels, sym="", whis=[0, 100])
    plt.savefig(f"{filename}.eps")
    plt.savefig(f"{filename}.png")
    plt.clf()

def output_boxplot_contradiction(task:str, indexes:list[int], title:str, label_maker, filename:str):
    plt.figure(figsize=(3, 6))
    task_result = datas[task]
    data = [datas["Contradiction"][task][i] for i in indexes]
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] *= 100
    #labels = [f"{task_result[i][0]}{task_result[i][1]}" for i in indexes]
    labels = [label_maker(task_result[i]) for i in indexes]
    x = np.array(np.array(data).T)
    plt.ylabel("矛盾の割合(%)")
    plt.title(title)
    plt.ylim([0, 100])
    plt.grid()
    plt.boxplot(x, labels=labels, sym="", whis=[0, 100])
    plt.savefig(f"{filename}.eps")
    plt.savefig(f"{filename}.png")
    plt.clf()

def output_plot_contradiction(task:str, max_step:int, indexes:list[int], title:str, label_maker, filename:str):
    task_result = datas[task]
    for i in range(len(indexes)):
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(title)
        #fig.subplots_adjust(hspace=0.4, wspace=0.2)
        gs = gridspec.GridSpec(1, 1, figure=fig)
        x = np.array(datas["Contradiction"][task][indexes[i]])
        y = np.array(datas[task][indexes[i]][4])
        label = label_maker(task_result[indexes[i]])

        a, b = np.polyfit(x, y, 1)
        y2 = a * x + b
        #fig=plt.figure()
        r,c = i//1, i%3
        ax=fig.add_subplot(111)
        #ax = fig.add_subplot(gs)
        ax.scatter(x,y)
        ax.set_title(label)
        ax.plot(x, y2,color='black')
        ax.set_ylabel("ステップ数")
        ax.set_xlabel("矛盾の割合")
        #ax[r,c].set_xlim([0,1])
        #ax[r,c].set_ylim([0,max_step])
        #ax.text(0.1,a*0.1+b, 'y='+ str(round(a,4)) +'x+'+str(round(b,4)))
    #ax[1,1] = fig.add_subplot(gs[1,1])
        plt.savefig(f"{filename}{i}.eps")
        plt.savefig(f"{filename}{i}.png")
        plt.clf()

def output_training_llm(task:str, indexes:list[int], title:str, label_maker, filename:str):
    task_result = datas[task]
    data = [datas["Training"][task][i] for i in indexes]
    labels = [label_maker(task_result[i]) for i in indexes]
    # labels = ["(2) bidirectional","(3) unidirectional","(5) alternation proposal"]
    # plt.figure(figsize=(4.5*0.8, 6*0.8))
    plt.figure(figsize=(6*0.9, 4*0.9))
    # plt.figure(figsize=(6, 4.5))

    plt.title(title)
    plt.ylabel("step")
    plt.xlabel("episode")
    plt.grid() # 横線
    # plt.ylim([25.0, 50])

    for i, d in enumerate(data):
        plt.plot(range(1,len(d)+1), d, label=labels[i])

    # y=[27,27]
    # x=[1,15]
    # plt.plot(x, y, label="Optimal solution (average)")
    # plt.plot(x, y, label="最適解 (平均)")
    # plt.subplots_adjust(left=0.1, right=0.985, bottom=0.12, top=0.88)
    plt.legend()
    plt.savefig(f"{filename}.eps")
    plt.savefig(f"{filename}.png")
    plt.clf()

def output_training_ppo(task:str, indexes:list[int], title:str, label_maker, filename:str):
    # plt.figure(figsize=(8, 6))
    task_result = datas[task]
    # labels = ["MAPPO 疑似報酬なし", "MAPPO 疑似報酬あり"]
    labels = ["(8)MAPPO 疑似報酬なし", "(9)MAPPO 疑似報酬あり"]
    # labels = ["(8)MAPPO no pesudo-reward", "(9)MAPPO with pesudo-reward"]
    labels_llm = [label_maker(task_result[i]) for i in indexes]
    # labels_llm = ["(2) bidirectional","(3) unidirectional","(5) alternation proposal"]
    # labels_llm[1] += " (提案手法1)"
    # labels_llm[2] += " (提案手法2)"
    data = [
        (datas["Training"]["MAPPO_Mean"][0],datas["Training"]["MAPPO_Var"][0]),
        (datas["Training"]["MAPPO_Mean"][1],datas["Training"]["MAPPO_Var"][1])
    ]
    data_llm = [datas["Training"][task][i] for i in indexes]
    plt.title(title)
    plt.ylabel("step")
    plt.xlabel("episode")
    plt.grid() # 横線
    plt.ylim([25.0, 50])
    colors = [(1, 0.7, 0.7, 1), (0.8, 0.75, 1, 1)]

    for i, y in enumerate(data_llm):
        y.append(y[-1])
        x=list(range(1,len(y)))
        x.append(len(data[0][0])*10)
        plt.plot(x, y, label=labels_llm[i])

    for i, d in enumerate(data):
        x=list(range(1,len(d[0])+1))
        x = [i*10 for i in x]
        y=d[0]
        var=d[1]
        y1 = [y[i] + var[i] for i in range(len(y))]
        y2 = [y[i] - var[i] for i in range(len(y))]
        plt.fill_between(x,y1,y2,color=colors[i])
        plt.plot(x, y, "--", label=labels[i])

    y=[27,27]
    x=list(range(1,len(y)))
    x.append(len(data[0][0])*10)
    plt.plot(x, y, label="最適解 (平均)")
    # plt.plot(x, y, label="Optimal solution (average)")
    plt.subplots_adjust(left=0.1, right=0.985, bottom=0.12, top=0.88)
    # plt.legend()
    plt.legend(loc='lower left', borderaxespad=2)
    plt.savefig(f"{filename}.eps")
    plt.savefig(f"{filename}.png")
    plt.clf()

output_boxplot(
    "PutNext",
    [0, 1, 2, 4],
    100,
    "PutNextの達成ステップ数",
    lambda name: f"{name[1]}",
    "result-putnext"
)

# output_boxplot(
#     "PutNext",
#     [0, 1, 2, 4],
#     100,
#     "PutNext-v0における通信方法ごとのエピソード長(step)",
#     lambda name: f"{name[0]}{name[1]}",
#     "result-putnext"
# )

output_boxplot(
    "PutNext",
    [12, 13, 14, 15],
    100,
    "3エージェントでのPutNext-v0における通信方法ごとのエピソード長(step)",
    lambda name: f"{name[0]}{name[1]}",
    "result-putnext_3"
)

output_boxplot(
    "BlockedUnlockPickup",
    [0, 1, 2, 4],
    50,
    "BlockedUnlockPickupの達成ステップ数",
    lambda name: f"{name[1]}",
    "result-blocked"
)

# output_boxplot(
#     "BlockedUnlockPickup",
#     [0, 1, 2, 4],
#     50,
#     "BlockedUnlockPickup-v0における通信方法ごとのエピソード長(step)",
#     lambda name: f"{name[0]}{name[1]}",
#     "result-blocked"
# )

output_boxplot(
    "BlockedUnlockPickup",
    [3,4,5,6],
    50,
    "ラウンドごとの達成ステップ数",
    lambda name: f"{name[2][0]}",
    "result-round"
)

# output_boxplot(
#     "BlockedUnlockPickup",
#     [3,4,5,6,7,8,9,10], # [3,4,5,6]
#     50,
#     "BlockedUnlockPickup-v0におけるラウンド数ごとのエピソード長(step)",
#     lambda name: f"{name[0]} {name[2]}\n{name[3]}", # lambda name: f"{name[2]}"
#     "result-round"
# )

output_boxplot(
    "BlockedUnlockPickup",
    [11,12,13,14,15],
    50,
    "BlockedUnlockPickup-v0における入力する履歴の長さごとのエピソード長(step)",
    lambda name: f"{name[0]} step",
    "result-history"
)

output_boxplot_contradiction(
    "BlockedUnlockPickup",
    [1,2,4],
    "BlockedUnlockPickupでの矛盾の割合",
    lambda name: f"{name[1]}",
    "result-contradiction-boxplot"
)

# output_boxplot_contradiction(
#     "BlockedUnlockPickup",
#     [1,2,3,4,5,6],
#     #[1,2,4],
#     "BlockedUnlockPickup-v0における手法ごとの矛盾の割合",
#     lambda name: f"{name[0]}{name[1]}\n{name[2]}",
#     #lambda name: f"{name[0]}{name[1]}",
#     "result-contradiction-boxplot"
# )

output_plot_contradiction(
    "BlockedUnlockPickup",
    50,
    [1,2,4],
    "手法ごとのエピソード長と矛盾の割合の関係",
    lambda name: f"{name[0]}{name[1]}",
    "result-contradiction-plot"
)

# output_training_llm(
#     "BlockedUnlockPickup",
#     [1,2,4],
#     "Average number of steps achieved \nin each learning phase of Reflexion",
#     lambda name: f"{name[1]}",
#     "result-training-llm"
# )

output_training_llm(
    "BlockedUnlockPickup",
    [1,2,4],
    "Reflexionの各学習段階での平均達成ステップ",
    # lambda name: f"{name[1]} {name[2]}",
    lambda name: f"{name[0]}{name[1]}",
    # lambda name: f"{name[1]}",
    "result-training-llm"
)

# output_training_ppo(
#     "BlockedUnlockPickup",
#     [1,2,4],
#     'Average number of steps achieved \nin each learning phase of MAPPO',
#     lambda name:f"{name[1]}",
#     "result-training-ppo"
# )

output_training_ppo(
    "BlockedUnlockPickup",
    [1,2,4],
    'MAPPOの各学習段階での平均達成ステップ数',
    # lambda name:f"{name[1]} {name[2]}",
    lambda name:f"{name[0]}{name[1]}",
    "result-training-ppo"
)