# %% [markdown]
# ## GYMデータの分析
# %%
import numpy as np
import pandas as pd
# %%
uselog = pd.read_csv('./data3/use_log.csv')
print(uselog.shape)
uselog.head()
# %%
customer = pd.read_csv('./data3/customer_master.csv')
print(customer.shape)
customer.head()
# %%
class_master = pd.read_csv('./data3/class_master.csv')
print(class_master.shape)
class_master
# %%
campaign_master = pd.read_csv('./data3/campaign_master.csv')
print(campaign_master.shape)
campaign_master
# %% [markdown]
# ## customerとclass_master・campaign_masterを結合
# %%
# customerとclass_masterを結合
customer_join = pd.merge(customer, class_master, on = 'class', how = 'left')
customer_join.head()
# %%
# 上記の結合データフレームとcampaign_masterを結合
customer_join = pd.merge(customer_join, campaign_master, on = 'campaign_id', how = 'left')
customer_join.head()
# %% [markdown]
# ## 結合後のデータ確認
# %%
print(len(customer) == len(customer_join)) # 結合後の確認（列数の一致）
# %%
customer_join.isnull().any() # 欠損値を含むか否か
# %%
print(f'全体の列数：{len(customer_join)} \n')
customer_join.info() # end_dateと全体の比較
# %%
customer_join.isnull().sum() # null数のカウント
# %%
# end_dateのうち，nullの行だけを抽出
customer_join[customer_join['end_date'].isnull()]
# %% [markdown]
# ## 顧客データの集計
# %%
# classごとの会員数
customer_join.groupby('class_name').count()['customer_id']
# %%
# キャンペーンごとの会員数
customer_join.groupby('campaign_name').count()['customer_id']
# %%
# 男女の会員数
customer_join.groupby('gender').count()['customer_id']
# %%
# 在籍会員（0）と退会会員（1）
customer_join.groupby('is_deleted').count()['customer_id']
# %%
# 2018年4月1日〜2019年3月31日まで入会した人数
customer_join['start_date'] = pd.to_datetime(customer_join['start_date'])
customer_start = customer_join.loc[customer_join['start_date'] > pd.to_datetime('20180401')]
len(customer_start)
# %% [markdown]
# ## 最新月の顧客情報を集計
# %%
customer_join['end_date'] = pd.to_datetime(customer_join['end_date'])
customer_newer = customer_join.loc[(customer_join['end_date'] >= pd.to_datetime('20190331')) | (customer_join['end_date'].isna())]
print(len(customer_newer))
customer_newer['end_date'].unique()
# %%
# classごとの会員数
customer_newer.groupby('class_name').count()['customer_id']
# %%
# キャンペーンごとの会員数
customer_newer.groupby('campaign_name').count()['customer_id']
# %%
# 男女の会員数
customer_newer.groupby('gender').count()['customer_id']
# %%
