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