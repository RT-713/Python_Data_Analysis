# %% [markdown]
# ## データの加工
# %%
import pandas as pd
# %% [markdown]
# ## 売上および顧客データの読み込み
# %%
uriage_data = pd.read_csv('data2/uriage.csv')
uriage_data.head()
# %%
kokyaku_data = pd.read_excel('data2/kokyaku_daicho.xlsx')
kokyaku_data.head()
# %%
# 表記が統一されていないデータの確認
uriage_data[['item_name']].head()
# %% [markdown]
# ## データの前処理を行う前にデータを確認
# %%
uriage_data['purchase_date'] = pd.to_datetime(uriage_data['purchase_date'])
uriage_data['purchase_month'] = uriage_data['purchase_date'].dt.strftime('%Y%m')
result = uriage_data.pivot_table(index = 'purchase_month', columns = 'item_name', aggfunc = 'size', fill_value = 0)
result
# %% [markdown]
# ## 商品名の統一処理
# %%
len(pd.unique(uriage_data['item_name'])) # 処理前の商品名のユニークデータ数
# %%
uriage_data['item_name'] = uriage_data['item_name'].str.upper() # 小文字を大文字に変換
uriage_data['item_name'] = uriage_data['item_name'].str.replace('　', '') # 全角スペースを削除
uriage_data['item_name'] = uriage_data['item_name'].str.replace(' ', '') # 半角スペースを削除
uriage_data.sort_values(by = ['item_name'], ascending = True) # item_nameで昇順にソート
# %%
# 処理の確認
print(len(pd.unique(uriage_data['item_name']))) # 商品の種類の数
print(pd.unique(uriage_data['item_name'])) # 商品全体のリストを表示
# %% [markdown]
# ## 欠損値の処理
# %%
uriage_data.isnull().any() # いずれかの列で欠損値を含むか否か
# %%
uriage_data['item_price'].isnull().sum() # 欠損値のあった列において欠損値は何個存在するか
# %%
# 欠損値補完
flg_is_null = uriage_data['item_price'].isnull()
for trg in list(uriage_data.loc[flg_is_null, 'item_name'].unique()):
    price = uriage_data.loc[(~flg_is_null) & (uriage_data['item_name'] == trg), 'item_price'].max()
    uriage_data['item_price'].loc[(flg_is_null) & (uriage_data['item_name'] == trg)] = price
uriage_data.head()
# %%
uriage_data['item_price'].isnull().sum() # 欠損値補完の確認
# %%
