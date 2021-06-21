# %%
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
# %% [markdown]
# ## 顧客の退会を予測する
# %%
# データの読み込み
customer = pd.read_csv('./data3/customer_join.csv')
uselog_months = pd.read_csv('./data3/use_log_months.csv')
# %%
# データの整形
year_months = list(uselog_months['年月'].unique())
uselog = pd.DataFrame()
for i in range(1, len(year_months)):
    tmp = uselog_months.loc[uselog_months['年月'] == year_months[i]]
    tmp.rename(columns={'count':'count_0'}, inplace = True)
    tmp_before = uselog_months.loc[uselog_months['年月'] == year_months[i-1]]
    del tmp_before['年月']
    tmp_before.rename(columns = {'count':'count_1'}, inplace = True)
    tmp = pd.merge(tmp, tmp_before, on = 'customer_id', how = 'left')
    uselog = pd.concat([uselog, tmp], ignore_index = True)
uselog.head()
# %%
# 退会前月の顧客データの作成
from dateutil.relativedelta import relativedelta
exit_customer = customer.loc[customer['is_deleted'] == 1]
exit_customer['exit_date'] = None
exit_customer['end_date'] = pd.to_datetime(exit_customer['end_date'])
for i in range(len(exit_customer)):
    exit_customer['exit_date'].iloc[i] = exit_customer['end_date'].iloc[i] - relativedelta(months = 1)
exit_customer['年月'] = pd.to_datetime(exit_customer['exit_date']).dt.strftime('%Y%m')
uselog['年月'] = uselog['年月'].astype(str)
exit_uselog = pd.merge(uselog, exit_customer, on = ['customer_id', '年月'], how = 'left')
print(len(uselog))
exit_uselog.head()
# %%
# 欠損値の削除
exit_uselog = exit_uselog.dropna(subset = ['name']) # subsetはリスト型で削除したい行・列を指定する
print(len(exit_uselog))
print(len(exit_uselog['customer_id'].unique()))
exit_uselog.head()
# %%
# 継続している顧客のデータを作成
conti_customer = customer.loc[customer['is_deleted'] == 0]
conti_uselog = pd.merge(uselog, conti_customer, on = 'customer_id', how = 'left')
print(len(conti_uselog))
conti_uselog = conti_uselog.dropna(subset = ['name'])
print(len(conti_uselog))
# %%
# データのシャッフルと重複データの削除
conti_uselog = conti_uselog.sample(frac = 1).reset_index(drop = True)
conti_uselog = conti_uselog.drop_duplicates(subset = 'customer_id')
print(len(conti_uselog))
conti_uselog.head()
# %%
# 作成した継続・退会顧客のデータを結合
predict_data = pd.concat([conti_uselog, exit_uselog],ignore_index = True)
print(len(predict_data))
predict_data.head()
# %%
# 予測する月の在籍期間の作成
predict_data['period'] = None
predict_data['now_date'] = pd.to_datetime(predict_data['年月'], format = '%Y%m')
predict_data['start_date'] = pd.to_datetime(predict_data['start_date'])
for i in range(len(predict_data)):
    delta = relativedelta(predict_data['now_date'][i], predict_data['start_date'][i])
    predict_data['period'][i] = int(delta.years*12 + delta.months)
predict_data.head()
# %%
predict_data.isna().sum()
# %%
# count_1列に欠損値を含む行を削除する
predict_data = predict_data.dropna(subset = ['count_1'])
predict_data.isna().sum()
# %%
target_col = ['campaign_name', 'class_name', 'gender', 'count_1', 'routine_flg', 'period', 'is_deleted']
predict_data = predict_data[target_col]
predict_data.head()
# %%
