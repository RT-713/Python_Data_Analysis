# %% [markdown]
# ## ジムデータの詳細な分析
# %%
import numpy as np
import pandas as pd
# %%
# データの読み込み1
uselog = pd.read_csv('./data3/use_log.csv')
uselog.head()
# %%
# データの読み込み2
customer = pd.read_csv('./data3/customer_join.csv')
customer.head()
# %% [markdown]
# ## クラスタリングによる顧客のグループ分け
# %%
# 説明変数の抽出
customer_clustering = customer[['mean', 'median', 'max', 'min', 'membership_period']]
customer_clustering.head()
# %% [markdown]
# ## k-means法
# %%
# データの標準化処理
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
customer_clustering_std = sc.fit_transform(customer_clustering)
customer_clustering_std
# %%
# K-meansの適用
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 4, random_state = 0)
clusters = kmeans.fit(customer_clustering_std)
customer_clustering['cluster'] = clusters.labels_ # クラリタリングのラベルを列として追加
# %%
# クラリタリングのラベル数
customer_clustering['cluster'].unique()
# %%
customer_clustering.head()
# %% [markdown]
# ## k-means法の結果分析
# %%
# 列名の日本語化
customer_clustering.columns = ['月内平均値', '月内中央値', '月内最大値', '月内最小値', '会員期間', 'cluster']
# クラスターごとにデータ数を集計
customer_clustering.groupby('cluster').count()
# %%
# 各クラスターの特性
customer_clustering.groupby('cluster').mean()
# %% [markdown]
# ## 次元削減（主成分分析）を行い，データを可視化する
# %%
from sklearn.decomposition import PCA
X = customer_clustering_std
pca = PCA(n_components = 2) # 2次元
pca.fit(X)
x_pca = pca.transform(X)
pca_df = pd.DataFrame(x_pca)
pca_df
# %%
pca_df['cluster'] = customer_clustering['cluster']
pca_df
# %%
# データの可視化（for文による色分け）
import matplotlib.pyplot as plt
for i in customer_clustering['cluster'].unique():
    tmp = pca_df.loc[pca_df['cluster'] == i]
    plt.scatter(tmp[0], tmp[1])
plt.legend(customer_clustering['cluster'].unique())
# %%
# データの可視化：その２（引数指定）
plt.scatter(pca_df[0], pca_df[1], c = customer_clustering['cluster'])
# %% [markdown]
# ## クラスタリング結果による分析
# * 退会顧客の傾向分析
# %%
# データフレームの結合
customer_clustering = pd.concat([customer_clustering, customer], axis = 1)
customer_clustering
# %%
# クラスタごとの退会顧客数
customer_clustering.groupby(['cluster', 'is_deleted'], as_index = False).count()[['cluster', 'is_deleted', 'customer_id']]
# %%
# クラスタごとの定期利用数
customer_clustering.groupby(['cluster', 'routine_flg'], as_index = False).count()[['cluster', 'routine_flg', 'customer_id']]
# %% [markdown]
# ## 翌月の利用回数予測
# %%
# データの整形
uselog['usedate'] = pd.to_datetime(uselog['usedate'])
uselog['年月'] = uselog['usedate'].dt.strftime('%Y%m')
uselog_months = uselog.groupby(['年月', 'customer_id'], as_index = False).count()
uselog_months.rename(columns = {'log_id':'count'}, inplace = True)
del uselog_months['usedate']
uselog_months.head()
# %%
# データの整形
year_months = list(uselog_months['年月'].unique())
predict_data = pd.DataFrame()
for i in range(6, len(year_months)):
    tmp = uselog_months.loc[uselog_months['年月'] == year_months[i]]
    tmp.rename(columns={'count':'count_pred'}, inplace = True)
    for j in range(1, 7):
        tmp_before = uselog_months.loc[uselog_months['年月'] == year_months[i-j]]
        del tmp_before['年月']
        tmp_before.rename(columns = {'count':'count_{}'.format(j-1)}, inplace = True)
        tmp = pd.merge(tmp, tmp_before, on = 'customer_id', how = 'left')
    predict_data = pd.concat([predict_data, tmp], ignore_index = True)
predict_data
# %%
# 欠損値の除去
predict_data = predict_data.dropna()
predict_data = predict_data.reset_index(drop = True) # indexの初期化
# %%
# start_date列の追加
predict_data = pd.merge(predict_data, customer[['customer_id', 'start_date']], on = 'customer_id', how = 'left')
predict_data.head()
# %%
# 会員期間の算出
predict_data['now_date'] = pd.to_datetime(predict_data['年月'], format='%Y%m')
predict_data['start_date'] = pd.to_datetime(predict_data['start_date'])

from dateutil.relativedelta import relativedelta
predict_data['period'] = None
for i in range(len(predict_data)):
    delta = relativedelta(predict_data['now_date'][i], predict_data['start_date'][i])
    predict_data['period'][i] = delta.years*12 + delta.months
predict_data.head()
# %%
# 対象のデータを2018/4以降に絞る．
predict_data = predict_data.loc[predict_data['start_date'] >= pd.to_datetime('20180401')]
# %% [markdown]
# ## 線形回帰の実装
# %%
from sklearn import linear_model
import sklearn.model_selection
model = linear_model.LinearRegression()
X = predict_data[['count_0', 'count_1', 'count_2', 'count_3', 'count_4', 'count_5', 'period']]
y = predict_data['count_pred']

# 訓練・評価用データの作成
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
# %%
# 学習
model.fit(X_train, y_train)
# %%
# モデルの精度
print(round(model.score(X_train, y_train), 2))
print(round(model.score(X_test, y_test), 2))
# %%
# 変数の寄与を確認
coef = pd.DataFrame({'特徴量':X.columns, '係数':model.coef_})
coef
# %%
# サンプルとなる顧客データの作成
x1 = [3, 4, 4, 6, 8, 7, 8]
x2 = [2, 2, 3, 3, 4, 6, 8]
x_pred = [x1, x2]
# %%
# 作成したモデルと顧客データを用いて予測
model.predict(x_pred)
# %%
# データの出力
uselog_months.to_csv('data3/use_log_months.csv', index = False)