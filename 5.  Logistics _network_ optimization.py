# %% [markdown]
# ## 輸送の最適化問題
# %%
import numpy as np
import pandas as pd
from itertools import product
from pulp import LpVariable, lpSum, value
from ortoolpy import model_min, addvars, addvals
# %%
# データ読み込み
df_tc = pd.read_csv('./data4/trans_cost.csv', index_col='工場')
df_demand = pd.read_csv('./data4/demand.csv')
df_supply = pd.read_csv('./data4/supply.csv')

# 初期設定 #
np.random.seed(1)
nw = len(df_tc.index)
nf = len(df_tc.columns)
pr = list(product(range(nw), range(nf)))

# 数理モデル作成 #
m1 = model_min()
v1 = {(i,j):LpVariable('v%d_%d'%(i,j),lowBound=0) for i,j in pr}

m1 += lpSum(df_tc.iloc[i][j]*v1[i,j] for i,j in pr)
for i in range(nw):
    m1 += lpSum(v1[i,j] for j in range(nf)) <= df_supply.iloc[0][i]
for j in range(nf):
    m1 += lpSum(v1[i,j] for i in range(nw)) >= df_demand.iloc[0][j]
m1.solve()

# 総輸送コスト計算 #
df_tr_sol = df_tc.copy()
total_cost = 0
for k,x in v1.items():
    i,j = k[0],k[1]
    df_tr_sol.iloc[i][j] = value(x)
    total_cost += df_tc.iloc[i][j]*value(x)

print(df_tr_sol)
print(f'総輸送コスト：{total_cost}')
# %% [markdown]
# ## 最適化ネットワークの可視化
# %%
import matplotlib.pyplot as plt
import networkx as nx

# データ読み込み
df_tr = df_tr_sol.copy()
df_pos = pd.read_csv('./data4/trans_route_pos.csv')

# グラフオブジェクトの作成
G = nx.Graph()

# 頂点の設定
for i in range(len(df_pos.columns)):
    G.add_node(df_pos.columns[i])

# 辺の設定&エッジの重みのリスト化
num_pre = 0
edge_weights = []
size = 0.1
for i in range(len(df_pos.columns)):
    for j in range(len(df_pos.columns)):
        if not (i == j):
            # 辺の追加
            G.add_edge(df_pos.columns[i],df_pos.columns[j])
            # エッジの重みの追加
            if num_pre<len(G.edges):
                num_pre = len(G.edges)
                weight = 0
                if (df_pos.columns[i] in df_tr.columns)and(df_pos.columns[j] in df_tr.index):
                    if df_tr[df_pos.columns[i]][df_pos.columns[j]]:
                        weight = df_tr[df_pos.columns[i]][df_pos.columns[j]]*size
                elif(df_pos.columns[j] in df_tr.columns)and(df_pos.columns[i] in df_tr.index):
                    if df_tr[df_pos.columns[j]][df_pos.columns[i]]:
                        weight = df_tr[df_pos.columns[j]][df_pos.columns[i]]*size
                edge_weights.append(weight)
                

# 座標の設定
pos = {}
for i in range(len(df_pos.columns)):
    node = df_pos.columns[i]
    pos[node] = (df_pos[node][0],df_pos[node][1])

# 描画
nx.draw(G, pos, with_labels=True,font_size=16, node_size = 1000, node_color='k', font_color='w', width=edge_weights)

# 表示
plt.show()
# %% [markdown]
# ## 制約条件内に入っているかを確認
# - 1が帰ってくればOK
# %%
# データ読み込み
df_demand = pd.read_csv('./data4/demand.csv')
df_supply = pd.read_csv('./data4/supply.csv')

# 制約条件計算関数
# 需要側
def condition_demand(df_tr,df_demand):
    flag = np.zeros(len(df_demand.columns))
    for i in range(len(df_demand.columns)):
        temp_sum = sum(df_tr[df_demand.columns[i]])
        if (temp_sum>=df_demand.iloc[0][i]):
            flag[i] = 1
    return flag
            
# 供給側
def condition_supply(df_tr,df_supply):
    flag = np.zeros(len(df_supply.columns))
    for i in range(len(df_supply.columns)):
        temp_sum = sum(df_tr.loc[df_supply.columns[i]])
        if temp_sum<=df_supply.iloc[0][i]:
            flag[i] = 1
    return flag

print('需要条件計算結果:'+str(condition_demand(df_tr_sol,df_demand)))
print('供給条件計算結果:'+str(condition_supply(df_tr_sol,df_supply)))
# %% [markdown]
# ## 生産計画において利益を計算する
# %%
# データの読み込み
df_material = pd.read_csv('./data5/product_plan_material.csv', index_col='製品')
df_material
# %%
df_profit = pd.read_csv('./data5/product_plan_profit.csv', index_col='製品')
df_profit
# %%
df_stock = pd.read_csv('./data5/product_plan_stock.csv', index_col='項目')
df_stock
# %%
df_plan = pd.read_csv('./data5/product_plan.csv', index_col='製品')
df_plan
# %%
# 利益を計算する関数＝目的関数
def product_plan(df_profit, df_plan):
    profit = 0
    for i in range(len(df_profit.index)):
        for j in range(len(df_plan.columns)):
            profit += df_profit.iloc[i][j] * df_plan.iloc[i][j]
    return profit

print(f'総利益：{product_plan(df_profit, df_plan)}')
# %% [markdown]
# ## 生産最適化問題を考える
# %%
# 目的関数を最大化する計算を実装
from pulp import LpVariable, lpSum, value
from ortoolpy import model_max, addvars, addvals

df = df_material.copy()
inv = df_stock

m = model_max()
v1 = {(i):LpVariable('v%d'%(i),lowBound=0) for i in range(len(df_profit))}
m += lpSum(df_profit.iloc[i]*v1[i] for i in range(len(df_profit)))
for i in range(len(df_material.columns)):
    m += lpSum(df_material.iloc[j,i]*v1[j] for j in range(len(df_profit)) ) <= df_stock.iloc[:,i]
m.solve()

df_plan_sol = df_plan.copy()
for k,x in v1.items():
    df_plan_sol.iloc[k] = value(x)

print(df_plan_sol)
print(f'総利益：{value(m.objective)}')
# %% [markdown]
# ## 最適化された内容を制約条件を踏まえて検討する
# %%
# 制約条件を計算する関数
def condition_stock(df_plan, df_material, df_stock):
    flag = np.zeros(len(df_material.columns))
    for i in range(len(df_material.columns)):  
        temp_sum = 0
        for j in range(len(df_material.index)):  
            temp_sum = temp_sum + df_material.iloc[j][i] * float(df_plan.iloc[j])
        if (temp_sum <= float(df_stock.iloc[0][i])):
            flag[i] = 1
        print(df_material.columns[i]+'  使用量:'+str(temp_sum)+', 在庫:'+str(float(df_stock.iloc[0][i])))
    return flag

print(f'制約条件の計算結果：{condition_stock(df_plan_sol,df_material,df_stock)}')
# %%
