import numpy as np
from time import time
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE, SpectralEmbedding
# #############################################################################
# Generate sample data
#centers = [[1, 1], [-1, -1], [1, -1]]
#X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)


digits = load_digits()        # 加载数据集
data = scale(digits.data)     # 标准化，使每一维数据变成均值0，标准差1
#data = digits.data
_, labels_true = data, digits.target

# reducde_data = PCA(n_components=10).fit_transform(data)
# X = reducde_data
# X = data
# tsne = TSNE(n_components=2)
# reducde_data = tsne.fit_transform(data)
# X = reducde_data
reducde_data = SpectralEmbedding(n_components = 10).fit_transform(data)
X = reducde_data
# #############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
t0 = time()
sp = SpectralClustering(n_clusters=10, eigen_solver='arpack') 
sp.fit(X)
print('Run time: %.2f' % (time() - t0))
labels = sp.labels_
#cluster_centers = sp.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

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
from itertools import cycle


plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
print(colors)
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    #cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    #plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
    #         markeredgecolor='k', markersize=12)
plt.title('spectral clustering')
plt.show()