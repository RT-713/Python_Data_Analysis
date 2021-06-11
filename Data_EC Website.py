# %% [markdown]
# ## ウェブからの注文数を分析
# ### 1. データの読み込み
# %%
import pandas as pd
# %%
# csvファイルの読み込みとデータの先頭5列の表示
customer_master = pd.read_csv('data/customer_master.csv')
customer_master.head()
# %%
# 各csvファイルの読み込み
item_master = pd.read_csv('data/item_master.csv')
item_master.head()
# %%
transaction_1 = pd.read_csv('data/transaction_1.csv')
transaction_1.head()
# %%
transaction_detail_1 = pd.read_csv('data/transaction_detail_1.csv')
transaction_1.head()
# %% [markdown]
# ### 2. データの結合（ユニオン）
# 今回のデータ結合ではデータを縦方向に結合（行を追加）
# %%
transaction_2 = pd.read_csv('data/transaction_2.csv')
transaction_2.head()
# %%
# concat関数で2つのデータフレームを結合（インデックスは含めず結合）
transaction = pd.concat([transaction_1, transaction_2], ignore_index = True)
# %%
# 結合の確認
print(f'{len(transaction_1)} + {len(transaction_2)} = {len(transaction)}')
# %%
# transaction_detail_1と2のデータを結合
transaction_detail_2 = pd.read_csv('data/transaction_detail_2.csv')
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
