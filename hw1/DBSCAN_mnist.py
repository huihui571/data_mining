import numpy as np
from time import time
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
from sklearn import manifold

# #############################################################################
# Generate sample data
# centers = [[1, 1], [-1, -1], [1, -1]]
# X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                             random_state=0)

# X = StandardScaler().fit_transform(X)

digits = load_digits()        # 加载数据集
#data = scale(digits.data)     # 标准化，使每一维数据变成均值0，标准差1
data = StandardScaler().fit_transform(digits.data)
labels_true = digits.target
reducde_data = PCA(n_components=2).fit_transform(data)
X = reducde_data
# reducde_data = manifold.SpectralEmbedding(n_components=2).fit_transform(data) # 不好用
# X = reducde_data
# tsne = TSNE(n_components=2)
# reducde_data = tsne.fit_transform(data)
# X = reducde_data、
# X= data
# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.4, min_samples=4).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
# 一致性
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# 完整性
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# 标准化互信息，[0,1],值越大意味聚类结果与真实情况越吻合
print("Normal Mutual Information: %0.3f"
      % metrics.normalized_mutual_info_score(labels_true, labels,
                                           average_method='arithmetic'))
# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()