# %% [markdown]
# ## 物流データの読み込み
# %%
import pandas as pd
# %% [markdown]
# ## 各データテーブルの読み込み
# * コード：行の見出しとなる列を指定（index_col = 0）して読み込み
# %%
# 工場のデータ
factories = pd.read_csv('./data4/tbl_factory.csv', index_col = 0) 
factories.head()
# %%
# 倉庫のデータ
warehouses = pd.read_csv('./data4/tbl_warehouse.csv', index_col = 0)
warehouses
# %%
# コストのデータ
cost = pd.read_csv('./data4/rel_cost.csv', index_col = 0)
cost.head() 
# %%
# 輸送トランザクションのデータ
trans = pd.read_csv('./data4/tbl_transaction.csv', index_col = 0)
trans.head()
# %% [markdown]
# ## 各テーブルの結合
# %%
# transとcost
join_data = pd.merge(trans, cost, left_on = ['ToFC', 'FromWH'], right_on = ['FCID', 'WHID'], how = 'left')
join_data.head()
# %%
# join_dataとfactories（工場データの付与）
join_data = pd.merge(join_data, factories, left_on = 'ToFC', right_on = 'FCID', how = 'left')
join_data.head()
# %%
# join_dataとwarehouses（倉庫データの付与）
join_data = pd.merge(join_data, warehouses, left_on = 'FromWH', right_on = 'WHID', how = 'left')

# 列の並び替え+余分な列の削除
join_data = join_data[['TransactionDate', 'Quantity', 'Cost', 'ToFC', 'FCName', 'FCDemand', 'FromWH', 'WHName', 'WHSupply', 'WHRegion']]
join_data.head()
# %%
# 関東と東北データを個別に抽出
kanto = join_data.loc[join_data['WHRegion'] == '関東']
kanto.head()
# %%
tohoku = join_data.loc[join_data['WHRegion'] == '東北']
tohoku.head()
# %% [markdown]
# ## コストの集計
# %%
# 支社ごとのコストを算出
print('関東支社のコスト：' + str(kanto['Cost'].sum()) + '万円')
print('東北支社のコスト：' + str(tohoku['Cost'].sum()) + '万円')

# %%
# 支社の総輸送個数
print('関東支社の総部品輸送個数: ' + str(kanto['Quantity'].sum()) + '個')
print('東北支社の総部品輸送個数: ' + str(tohoku['Quantity'].sum()) + '個')
# %%
# 部品一つ当たりの輸送コスト
tmp = (kanto['Cost'].sum() / kanto['Quantity'].sum()) * 10000
print('関東支社の部品１つ当たりの輸送コスト: ' + str(int(tmp)) + '円')
tmp = (tohoku['Cost'].sum() / tohoku['Quantity'].sum()) * 10000
print('東北支社の部品１つ当たりの輸送コスト: ' + str(int(tmp)) + '円')
# %%
# コストテーブルを支社ごとに集計
cost_chk = pd.merge(cost, factories, on = 'FCID', how = 'left')
# 平均
print('東京支社の平均輸送コスト：' + str(cost_chk['Cost'].loc[cost_chk['FCRegion'] == '関東'].mean()) + '万円')
print('東北支社の平均輸送コスト：' + str(cost_chk['Cost'].loc[cost_chk['FCRegion'] == '東北'].mean()) + '万円')
# %% [markdown]
# ## 結論
# 東北支社の方が輸送を効率的に実施できている．
# - 輸送のコスト：関東 > 東北
# - 輸送部品の数：東北 > 関東
# - 平均輸送コスト：関東 > 東北
# %% [markdown]
# ## ネットワークの可視化
# %%
# ネットワーク可視化のライブラリをインポート
import networkx as nx
import matplotlib.pyplot as plt
# %%
# ネットワークの作成

# インスタンス作成
G = nx.Graph()

# 頂点の設定
G.add_node('nodeA')
G.add_node('nodeB')
G.add_node('nodeC')

# 辺の設定
G.add_edge('nodeA', 'nodeB')
G.add_edge('nodeA', 'nodeC')
G.add_edge('nodeB', 'nodeC')

# 座標の設定
pos = {}
pos['nodeA'] = (0, 0)
pos['nodeB'] = (1, 1)
pos['nodeC'] = (0, 1)

# 描写
nx.draw(G, pos)
plt.show()
# %%
G.add_node('nodeD')
G.add_edge('nodeA', 'nodeD')
pos['nodeD'] = (1, 0)

# with_labels引数を指定することで各ノードにラベルを付与できる
nx.draw(G, pos, with_labels = True) 
# %% [markdown]
# ## ルートの重みづけ（ノード間の調整）
# %%
# ノードの太さを重みに応じて付与

# データ読み込み
df_w = pd.read_csv('./data4/network_weight.csv')
df_p = pd.read_csv('./data4/network_pos.csv')

# インスタンス化
G = nx.Graph()

# エッジの重みのリスト化
size = 10
edge_weights = []
for i in range(len(df_w)):
    for j in range(len(df_w.columns)):
        edge_weights.append(df_w.iloc[i][j] * size)

# 頂点の設定
for i in range(len(df_w.columns)):
    G.add_node(df_w.columns[i])

# 辺の設定
for i in range(len(df_w.columns)):
    for j in range(len(df_w.columns)):
        G.add_edge(df_w.columns[i],df_w.columns[j])

# 座標の設定
pos = {}
for i in range(len(df_w.columns)):
    node = df_w.columns[i]
    pos[node] = (df_p[node][0],df_p[node][1])

# 引数を指定して表示
nx.draw(G, pos, with_labels = True, font_size = 16, node_size = 1000, node_color = 'k', font_color = 'w', width = edge_weights)
plt.show()
# %% [markdown]
# ## 輸送ルートの情報をもとにネットワークを可視化
# %%
df_tr = pd.read_csv('./data4/trans_route.csv', index_col = '工場')
df_tr
# %%
df_pos = pd.read_csv('./data4/trans_route_pos.csv')
df_pos
# %%
# データの可視化
# インスタンス化
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
        if not (i==j):
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
# ## 輸送ルートの最適化のために関数を作成
# %%
df_tc = pd.read_csv('./data4/trans_cost.csv', index_col = '工場')
df_tc
# %%
def trans_cost(df_tr, df_tc):
    cost = 0
    for i in range(len(df_tc.index)):
        for j in range(len(df_tr.columns)):
            cost += df_tr.iloc[i][j] * df_tc.iloc[i][j]
    return cost

print(f'総輸送コスト：{trans_cost(df_tr, df_tc)}')
# %% [markdown]
# ## 総輸送コストを下げるために制約条件を作成する．
# %%
df_demand = pd.read_csv('./data4/demand.csv')
df_demand
# %%
df_supply = pd.read_csv('./data4/supply.csv')
df_supply
# %%
# 需要側の制約条件
for i in range(len(df_demand.columns)):
    temp_sum = sum(df_tr[df_demand.columns[i]])
    print(str(df_demand.columns[i])+"への輸送量:"+str(temp_sum)+" (需要量:"+str(df_demand.iloc[0][i])+")")
    if temp_sum>=df_demand.iloc[0][i]:
        print("需要量を満たしています．")
    else:
        print("需要量を満たしていません．輸送ルートを再計算して下さい．")

# 供給側の制約条件
for i in range(len(df_supply.columns)):
    temp_sum = sum(df_tr.loc[df_supply.columns[i]])
    print(str(df_supply.columns[i])+"からの輸送量:"+str(temp_sum)+" (供給限界:"+str(df_supply.iloc[0][i])+")")
    if temp_sum<=df_supply.iloc[0][i]:
        print("供給限界の範囲内です．")
    else:
        print("供給限界を超過しています．輸送ルートを再計算して下さい．")
# %% [markdown]
# ## 輸送ルートの変更で関数の変化を確認する．
# %%
# 新しく設定した輸送ルートの読み込み
df_tr_new = pd.read_csv('./data4/trans_route_new.csv', index_col="工場")
df_tr_new
# %%
import numpy as np

# 総輸送コスト再計算 
print("総輸送コスト(変更後):"+str(trans_cost(df_tr_new,df_tc)))

# 制約条件を満たす場合は1，そうでない場合は0を付与する
# 需要側
def condition_demand(df_tr,df_demand):
    flag = np.zeros(len(df_demand.columns))
    for i in range(len(df_demand.columns)):
        temp_sum = sum(df_tr[df_demand.columns[i]])
        if (temp_sum >= df_demand.iloc[0][i]):
            flag[i] = 1
    return flag
            
# 供給側
def condition_supply(df_tr,df_supply):
    flag = np.zeros(len(df_supply.columns))
    for i in range(len(df_supply.columns)):
        temp_sum = sum(df_tr.loc[df_supply.columns[i]])
        if temp_sum <= df_supply.iloc[0][i]:
            flag[i] = 1
    return flag

print("需要条件計算結果:"+str(condition_demand(df_tr_new,df_demand)))
print("供給条件計算結果:"+str(condition_supply(df_tr_new,df_supply)))
# %% [markdown]
# - 総輸送コストはわずかに削減できたが，2番目の供給条件が満たされていない．
# 最適化の計算は試行錯誤する必要あり．
# %%
