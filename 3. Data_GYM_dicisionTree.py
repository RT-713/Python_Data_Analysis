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
predict_data['period'] = 0
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
# %% [markdown]
# ## ダミー変数の処理
# %%
# データの絞り込み
target_col = ['campaign_name', 'class_name', 'gender', 'count_1', 'routine_flg', 'period', 'is_deleted']
predict_data = predict_data[target_col]
predict_data.head()
# %%
# ダミー変数に変換する処理
predict_data = pd.get_dummies(predict_data)
predict_data.head()
# %%
predict_data.shape
# %%
# 余分な列を削除する
del predict_data['campaign_name_通常']
del predict_data['class_name_ナイト']
del predict_data['gender_M']
predict_data.shape
# %% [markdown]
# ## 決定木による退会予測モデル
# %%
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection

# データの準備
exit = predict_data.loc[predict_data['is_deleted'] == 1]
conti = predict_data.loc[predict_data['is_deleted'] == 0].sample(len(exit)) # 退会者の数と合わせる処理

X = pd.concat([exit, conti], ignore_index = True)
y = X['is_deleted']
del X['is_deleted']
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
# %%
# モデルの作成と学習
model = DecisionTreeClassifier(random_state = 0)
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
y_test_pred
# %%
# 予測と正解の差異を比較する表
results_test = pd.DataFrame({'y_test':y_test, 'y_pred':y_test_pred})
results_test.head()
# %%
# 正解率を関数を用いずに算出する
correct = len(results_test.loc[results_test['y_test'] == results_test['y_pred']])
data_count = len(results_test)
score_test = correct / data_count
score_test
# %%
# 関数による正解率の評価 => 過学習を起こしている
print(f'訓練用データでの正解率{model.score(X_train, y_train)}')
print(f'評価用データでの正解率{model.score(X_test, y_test)}')
# %%
# モデルの修正
X = pd.concat([exit, conti], ignore_index = True)
y = X['is_deleted']
del X['is_deleted']
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)

# モデルの構築と学習
model_2 = DecisionTreeClassifier(random_state = 0, max_depth = 5) # 深さを指定
model_2.fit(X_train, y_train)
# %%
# 改良したモデルで再評価 => 訓練と評価で同程度の精度が認められた
print(f'訓練用データでの正解率②{model.score(X_train, y_train)}')
print(f'評価用データでの正解率②{model.score(X_test, y_test)}')
# %% [markdown]
# ## 変数の寄与を確認
# %%
importance = pd.DataFrame({'feature_names': X.columns, 'coefficient': model_2.feature_importances_})
importance
# %% [markdown]
# ## 決定木の可視化
# %%
from sklearn import tree
import matplotlib.pyplot as plt
tree.plot_tree(model_2)
plt.show()
# %% [markdown]
# ## 退会の予測
# %%
count_1 = 3
routine_flg = 1
period = 10
campaign_name = "入会費無料"
class_name = "オールタイム"
gender = "M"
# %%
if campaign_name == "入会費半額":
    campaign_name_list = [1, 0]
elif campaign_name == "入会費無料":
    campaign_name_list = [0, 1]
elif campaign_name == "通常":
    campaign_name_list = [0, 0]
if class_name == "オールタイム":
    class_name_list = [1, 0]
elif class_name == "デイタイム":
    class_name_list = [0, 1]
elif class_name == "ナイト":
    class_name_list = [0, 0]
if gender == "F":
    gender_list = [1]
elif gender == "M":
    gender_list = [0]
input_data = [count_1, routine_flg, period]
input_data.extend(campaign_name_list)
input_data.extend(class_name_list)
input_data.extend(gender_list)
# %% [markdown]
# ## 上記で設定した顧客がいた場合の退会確率
# - 退会 or not => 退会
# - その確率：98.57%
# %%
print(model_2.predict([input_data]))
print(model_2.predict_proba([input_data]))
# %%
