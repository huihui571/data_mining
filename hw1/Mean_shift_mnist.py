import numpy as np
from time import time
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
# #############################################################################
# Generate sample data
#centers = [[1, 1], [-1, -1], [1, -1]]
#X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)


digits = load_digits()        # 加载数据集
data = scale(digits.data)     # 标准化，使每一维数据变成均值0，标准差1
#data = digits.data
_, labels_true = data, digits.target

#reducde_data = PCA(n_components=10).fit_transform(data)
#X = reducde_data
tsne = TSNE(n_components=2)
reducde_data = tsne.fit_transform(data)
X = reducde_data
# #############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
t0 = time()
bandwidth = estimate_bandwidth(X, quantile=0.07, n_samples=500)  # 窗口尺寸
ms = MeanShift(bandwidth=bandwidth, bin_seeding=False)
ms.fit(X)
print('Run time: %.2f' % (time() - t0))
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)
# 一致性
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# 完整性
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# 上两者的调和平均
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# 调整兰德指数取值范围[-1.1],值越大意味聚类结果与真实情况越吻合
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
# 调整互信息，[-1,1],值越大意味聚类结果与真实情况越吻合
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels,
                                           average_method='arithmetic'))
# 轮廓系数，综合了聚集度和分离度，取值为[-1, 1]，其值越大越好，且当值为负时，表明 ai<bi，样本被分配到错误的簇中，聚类结果不可接受。对于接近0的结果，则表明聚类结果有重叠的情况。
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))
# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle


plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
print(colors)
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=12)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()