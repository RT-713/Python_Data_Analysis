# %% [markdown]
# ## 消費者情報のネットワーク
# %%
import pandas as pd

df_links = pd.read_csv('./data/data6/links.csv')
df_links.head()
# %% [markdown]
# node同士のつながりを表すdf_linksを使用してネットワークを可視化
# %%
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

# 頂点の設定
num = len(df_links.index)
for i in range(1, num + 1):
    node_no = df_links.columns[i].strip('Node')
    # print(node_no)
    G.add_node(str(node_no))

# 辺の設定
for i in range(num):
    for j in range(num):
        # print(i, j)
        node_name = 'Node' + str(j)
        if df_links[node_name].iloc[i] == 1:
            G.add_edge(str(i), str(j))

# リンクが多いものが中心にくるようにネットワークを作成
# ただし，そのほかのnodeは実行するたびにバラける
nx.draw_networkx(G, node_color='k', edge_color='k', font_color='w')
plt.show()
# %% [markdown]
# ## クチコミによる情報拡散をシュミレート
# - 10%の確率でクチコミが拡散すると仮定
# %%
import numpy as np

# 確率に基づき拡散させるか否かを決定
def determine_link(percent):
    rand_val = np.random.rand()
    if rand_val <= percent:
        return 1
    else:
        return 0

# 拡散のシミュレーション
def simulate_perlocation(num, list_active, persent_percolation):
    for i in range(num):
        if list_active[i] == 1:
            for j in range(num):
                node_name = 'Node' + str(j)
                if df_links[node_name].iloc[i] == 1:
                    if determine_link(persent_percolation) == 1:
                        list_active[j] = 1
    return list_active


T_NUM = 100
NUM = len(df_links.index)
list_active = np.zeros(NUM)
list_active[0] = 1
# 10%と仮定
persent_percolation = 0.1

list_timeSeries = []
for t in range(T_NUM):
    list_active = simulate_perlocation(NUM, list_active, persent_percolation)
    list_timeSeries.append(list_active.copy())
# %%
def active_node_coloring(list_active):
    #print(list_timeSeries[t])
    list_color = []
    for i in range(len(list_timeSeries[t])):
        if list_timeSeries[t][i] == 1:
            list_color.append('r')
        else:
            list_color.append('k')
    # print(len(list_color))
    return list_color
# %%
t = 0
nx.draw_networkx(G, font_color='w', node_color = active_node_coloring(list_timeSeries[t]))
plt.show()
# %%
t = 10
nx.draw_networkx(G, font_color='w', node_color = active_node_coloring(list_timeSeries[t]))
plt.show()
# %%
t = 99
nx.draw_networkx(G, font_color='w', node_color = active_node_coloring(list_timeSeries[t]))
plt.show()
# %% [markdown]
# ## 拡散の時系列変化を可視化
# %%
list_timeSeries_num = []
for i in range(len(list_timeSeries)):
    list_timeSeries_num.append(sum(list_timeSeries[i]))

plt.plot(list_timeSeries_num)
plt.show()
# %% [markdown]
# ## 時系列変化（会員数）
# %%
def simulate_population(num, list_active, percent_percolation, percent_disapparence,df_links):
    # 拡散のパターン
    for i in range(num):
        if list_active[i]==1:
            for j in range(num):
                if df_links.iloc[i][j]==1:
                    if determine_link(percent_percolation)==1:
                        list_active[j] = 1
    # 消滅のパターン
    for i in range(num):
        if determine_link(percent_disapparence)==1:
            list_active[i] = 0
    return list_active
# %%
percent_percolation = 0.1
percent_disapparence = 0.05
T_NUM = 100
NUM = len(df_links.index)
list_active = np.zeros(NUM)
list_active[0] = 1

list_timeSeries = []
for t in range(T_NUM):
    list_active = simulate_population(NUM, list_active, percent_percolation, percent_disapparence,df_links)
    list_timeSeries.append(list_active.copy())
# %%
# 時系列グラフ
list_timeSeries_num = []
for i in range(len(list_timeSeries)):
    list_timeSeries_num.append(sum(list_timeSeries[i]))

plt.plot(list_timeSeries_num)
plt.show()
# %%
# 利用者が減少していく過程の確認
percent_disapparence = 0.3
list_active = np.zeros(NUM)
list_active[0] = 1
list_timeSeries = []
for t in range(T_NUM):
    list_active = simulate_population(NUM, list_active, percent_percolation, percent_disapparence,df_links)
    list_timeSeries.append(list_active.copy())

# 時系列グラフ
list_timeSeries_num = []
for i in range(len(list_timeSeries)):
    list_timeSeries_num.append(sum(list_timeSeries[i]))

plt.plot(list_timeSeries_num)
plt.show()
# %%
# 相図計算
print("相図計算開始")
T_NUM = 100
NUM_PhaseDiagram = 20
phaseDiagram = np.zeros((NUM_PhaseDiagram,NUM_PhaseDiagram))
for i_p in range(NUM_PhaseDiagram):
    for i_d in range(NUM_PhaseDiagram):
        percent_percolation = 0.05*i_p
        percent_disapparence = 0.05*i_d
        list_active = np.zeros(NUM)
        list_active[0] = 1
        for t in range(T_NUM):
            list_active = simulate_population(NUM, list_active, percent_percolation, percent_disapparence,df_links)
        phaseDiagram[i_p][i_d] = sum(list_active)
print(phaseDiagram)

# 図示
plt.matshow(phaseDiagram)
plt.colorbar(shrink=0.8)
plt.xlabel('percent_disapparence')
plt.ylabel('percent_percolation')
plt.xticks(np.arange(0.0, 20.0,5), np.arange(0.0, 1.0, 0.25))
plt.yticks(np.arange(0.0, 20.0,5), np.arange(0.0, 1.0, 0.25))
plt.tick_params(bottom=False,
                left=False,
                right=False,
                top=False)
plt.show()
# %%
