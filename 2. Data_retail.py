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
# %% [markdown]
# ## 顧客名の記載方法を統一
# %%
kokyaku_data.head()
# %%
# 全角・半角スペースの削除処理
kokyaku_data['顧客名'] = kokyaku_data['顧客名'].str.replace('　', '')
kokyaku_data['顧客名'] = kokyaku_data['顧客名'].str.replace(' ', '')
kokyaku_data.head()
# %% [markdown]
# ## 登録日の修正
# %%
# 登録日が日付でない（数値となっている）数の算出
flg_is_serial = kokyaku_data['登録日'].astype('str').str.isdigit()
flg_is_serial.sum()
# %%
fromSerial = pd.to_timedelta(kokyaku_data.loc[flg_is_serial, '登録日'].astype(float), unit = 'D') + pd.to_datetime('1900/01/01')  # loc関数はbool型で指定することも可能（Trueを引っ張ってくる）
fromSerial
print(fromSerial.count())
# %%
# 登録日の数値データが正常に変換されたことを確認
flg_is_serial = kokyaku_data['登録日'].astype('str').str.isdigit()
flg_is_serial.sum()
# %%
fromString = pd.to_datetime(kokyaku_data.loc[~flg_is_serial, '登録日']) # Falseのデータをpandasのデータタイム型へ変換
fromString
print(fromString.count())
# %%
# データの型と表記を統一した各登録日のデータをconcatで統合する
kokyaku_data['登録日'] = pd.concat([fromSerial, fromString])
kokyaku_data
# %%
kokyaku_data['登録年月'] = kokyaku_data['登録日'].dt.strftime('%Y%m')
rslt = kokyaku_data.groupby('登録年月').count()['顧客名']
rslt
# %%
len(kokyaku_data) # データの確認
# %% [markdown]
# ## データの結合
# %%
# 顧客名をキーにして結合
join_data = pd.merge(uriage_data, kokyaku_data, left_on = 'customer_name', right_on = '顧客名', how = 'left')
join_data = join_data.drop('customer_name', axis = 1) # 顧客名が重複しているので削除
join_data
# %% [markdown]
# ## データの最終的な整形（列の並びをわかりやすく）
# %%
# 列の並びを指定
dump_data = join_data[['purchase_date', 'purchase_month', 'item_name', 'item_price', '顧客名', 'かな', '地域', 'メールアドレス', '登録日']]
dump_data
# %% [markdown]
# ## 整形したデータの出力（csv）
# %%
dump_data.to_csv('dump_data.csv', index = False)