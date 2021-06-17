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
# %% [markdown]
# ## 利用履歴データの集計
# %%
# uselogデータの日付を
uselog['usedate'] = pd.to_datetime(uselog['usedate']) # 形式の変換
uselog['年月'] = uselog['usedate'].dt.strftime('%Y%m') # 年月列を追加し，指定の形式で値を代入
uselog_month = uselog.groupby(['年月', 'customer_id'], as_index = False).count() # as_indexで各行に年月が入るようにしている
uselog_month
# %%
uselog_month.rename(columns = {'log_id':'count'}, inplace = True) # inplaceでlog_idをcountという名前に置き換え
uselog_month.head()
# %%
del uselog_month['usedate'] #不要なusedateは削除してある
uselog_month.head()
# %%
uselog_customer = uselog_month.groupby('customer_id').agg(['mean', 'median', 'max', 'min'])['count']
uselog_customer
# %%
# groupbyにより追加されたcustomer_idをカラムに変更
uselog_customer = uselog_customer.reset_index(drop = False)
uselog_customer
# %%
# usedateの曜日を計算し，列毎に集計（log_idの個数）
uselog['weekday'] = uselog['usedate'].dt.weekday
uselog_weekday = uselog.groupby(['customer_id', '年月', 'weekday'], as_index = False).count()[['customer_id', '年月', 'weekday', 'log_id']]
uselog_weekday
# %%
# log_idの列名をcountへリネーム
uselog_weekday.rename(columns = {'log_id':'count'}, inplace = True)
uselog_weekday
# %% [markdown]
# * 曜日の集計では0~6が月〜日に相当する．<br>
# AS002855さんは2018/4は土曜（weekday=5）に4回来ている<br>
# 5月は土曜に加え，水曜にも1回来ている<br>
# ★ 4回以上のデータにフラグをつけてみる
# %%
uselog_weekday = uselog_weekday.groupby('customer_id', as_index = False).max()[['customer_id', 'count']]
uselog_weekday['routine_flg'] = 0
uselog_weekday['routine_flg'] = uselog_weekday['routine_flg'].where(uselog_weekday['count'] < 4 , 1)
uselog_weekday
# %% [markdown]
# ## 集計データの結合
# %%
customer_join = pd.merge(customer_join, uselog_customer, on = 'customer_id', how = 'left')
customer_join
# %%
# uselog_weekdayのうち，count列は不要なので結合対象に含めず
customer_join = pd.merge(customer_join, uselog_weekday[['customer_id', 'routine_flg']], on = 'customer_id', how = 'left')
customer_join
# %%
# end_date以外，欠損値はない
customer_join.isnull().sum()
# %% [markdown]
# ## 会員期間を計算して結果を列に追加する
# %%
from dateutil.relativedelta import relativedelta
customer_join['calc_date'] = customer_join['end_date']
customer_join['calc_date'] = customer_join['calc_date'].fillna(pd.to_datetime('20190430')) # 欠損値にdatetime型の日付を代入
customer_join['membership_period'] = 0
for i in range(len(customer_join)):
    delta = relativedelta(customer_join['calc_date'].iloc[i], customer_join['start_date'].iloc[i])
    customer_join['membership_period'].iloc[i] = delta.years*12 + delta.months
customer_join
# %% [markdown]
# ## 会員期間の可視化
# %%
import matplotlib.pyplot as plt
plt.hist(customer_join['membership_period'])
plt.show()
# %%
# 記述統計
customer_join[['mean', 'median', 'max', 'min']].describe()
# %%
# routine_flgごとの集計
customer_join.groupby('routine_flg').count()['customer_id']
# %% [markdown]
# 顧客の違いを確認する（継続or退会の差）
# %%
customer_stay = customer_join.loc[customer_join['is_deleted'] == 0]
customer_stay.describe()
# %%
customer_end = customer_join.loc[customer_join['is_deleted'] == 1]
customer_end.describe()
# %% [markdown]
# ## 集計したデータをpickleで保存
# %%
customer_join.to_csv('./data3/customer_join.csv', index = False)