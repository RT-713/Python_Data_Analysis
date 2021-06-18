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
