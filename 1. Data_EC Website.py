# %% [markdown]
# ## ウェブからの注文数を分析
# ### 1. データの読み込み
# %%
from numpy.core.defchararray import join
import pandas as pd
# %%
# csvファイルの読み込みとデータの先頭5列の表示
customer_master = pd.read_csv('./data/data1/customer_master.csv')
customer_master.head()
# %%
# csvファイルの読み込み
item_master = pd.read_csv('./data/data1/item_master.csv')
item_master.head()
# %%
# csvファイルの読み込み
transaction_1 = pd.read_csv('./data/data1/transaction_1.csv')
transaction_1.head()
# %%
# csvファイルの読み込み
transaction_detail_1 = pd.read_csv('./data/data1/transaction_detail_1.csv')
transaction_1.head()
# %% [markdown]
# ### 2. データの結合（ユニオン）
# 今回のデータ結合ではデータを縦方向に結合（行を追加）
# %%
# csvファイルの読み込み
transaction_2 = pd.read_csv('./data/data1/transaction_2.csv')
transaction_2.head()
# %%
# concat関数で2つのデータフレームを結合（インデックスは含めず結合）
transaction = pd.concat([transaction_1, transaction_2], ignore_index = True)
# %%
# 結合の確認
print(f'{len(transaction_1)} + {len(transaction_2)} = {len(transaction)}')
# %%
# transaction_detail_1と2のデータを結合
transaction_detail_2 = pd.read_csv('./data/data1/transaction_detail_2.csv')
transaction_detail_2.head()
# %%
transaction_detail = pd.concat([transaction_detail_1, transaction_detail_2], ignore_index = True)
print(f'transaction_detail_1：{transaction_detail_1.shape} \ntransaction_detail_2：{transaction_detail_2.shape}')
transaction_detail.shape
# %% [markdown]
# ## 売上データ同士を結合（ジョイン）
# %%
join_data = pd.merge(transaction_detail, transaction[['transaction_id', 'payment_date', 'customer_id']], on = 'transaction_id', how = 'left')
join_data.head()
# %% [markdown]
# ## マスターデータの結合（ジョイン）
# %%
join_data = pd.merge(join_data, customer_master, on = 'customer_id', how = 'left')
join_data = pd.merge(join_data, item_master, on = 'item_id', how = 'left')
join_data.head()
# %%
# joinデータにデータを追加
join_data['price'] = join_data['quantity'] * join_data['item_price']
join_data[['price']].tail()
# %%
# 確認
print(join_data['price'].sum() == transaction['price'].sum())
# %% [markdown]
# ## 記述統計を行う
# %%
join_data.describe()
print(join_data['payment_date'].min())
print(join_data['payment_date'].max())
# %%
# 各列における欠損値の合計
join_data.isnull().sum()
# %%
# データフレームにひとつでも欠損値が含まれるか確認
join_data.isna().any().any()
# %% [markdown]
# ## データフレームの各データ型を確認・変換する 
# %%
# 全体の一覧
join_data.dtypes
# %%
# 個別で確認
join_data.dtypes[['payment_date']]
# %%
# データ型の変換
join_data['payment_date'] = pd.to_datetime(join_data['payment_date'])
join_data['payment_month'] = join_data['payment_date'].dt.strftime('%Y%m') # dt以下の部分で列全体に任意の時刻表示（文字型）を適用
join_data[['payment_date', 'payment_month']]
# %% [markdown]
# ## 月別の集計（groupby）
# %%
# 月別の販売数と価格
join1 = join_data.groupby('payment_month').sum()[['quantity', 'price']]
# %%
join_data.groupby(['payment_month', 'item_name']).sum()[['price', 'quantity']]
# %% [markdown]
# ## ピボットテーブルによる集計
# %%
pd.pivot_table(join_data, index = 'item_name', columns = 'payment_month', values = ['price', 'quantity'], aggfunc = 'sum')
# %% [markdown]
# ## データの可視化
# %%
import matplotlib.pyplot as plt

# データの整形
graph_data = pd.pivot_table(join_data, index = 'payment_month', columns = 'item_name', values = 'price', aggfunc = 'sum')

# 図の表示
plt.plot(list(graph_data.index), graph_data['PC-A'], label = 'PC-A')
plt.plot(list(graph_data.index), graph_data['PC-B'], label = 'PC-B')
plt.plot(list(graph_data.index), graph_data['PC-C'], label = 'PC-C')
plt.plot(list(graph_data.index), graph_data['PC-D'], label = 'PC-D')
plt.plot(list(graph_data.index), graph_data['PC-E'], label = 'PC-E')

plt.legend()
plt.show()