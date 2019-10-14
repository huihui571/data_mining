from time import time
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# #############################################################################
# Generate sample data
#centers = [[1, 1], [-1, -1], [1, -1]]
#X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
#                            random_state=0)

digits = load_digits()        # 加载数据集
data = scale(digits.data)     # 标准化，使每一维数据变成均值0，标准差1
#data = digits.data
X, labels_true = data, digits.target

reducde_data = PCA(n_components=10).fit_transform(data)
X = reducde_data
# #############################################################################
# Compute Affinity Propagation
t0 = time()
af = AffinityPropagation(preference=-1900).fit(X)
print('Run time: %.2f' % (time() - t0))
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
# 一致性
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# 完整性
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# 上两者的调和平均
#print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# 调整兰德指数取值范围[-1.1],值越大意味聚类结果与真实情况越吻合
#print("Adjusted Rand Index: %0.3f"
#      % metrics.adjusted_rand_score(labels_true, labels))
# 标准化互信息，[0,1],值越大意味聚类结果与真实情况越吻合
print("Normal Mutual Information: %0.3f"
      % metrics.normalized_mutual_info_score(labels_true, labels,
                                           average_method='arithmetic'))
# 轮廓系数，综合了聚集度和分离度，取值为[-1, 1]，其值越大越好，且当值为负时，表明 ai<bi，样本被分配到错误的簇中，聚类结果不可接受。对于接近0的结果，则表明聚类结果有重叠的情况。
#print("Silhouette Coefficient: %0.3f"
#      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))
# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

# reducde_data = PCA(n_components=2).fit_transform(data)        # 数据降到2维
# t0 = time()
# af = AffinityPropagation(preference=-2650).fit(reducde_data)
# print('Run time: %.2f' % (time() - t0))
# cluster_centers_indices = af.cluster_centers_indices_
# labels = af.labels_
# #print(labels)
# n_clusters_ = len(cluster_centers_indices)

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k # [true,false,...]
    #print(class_members)
    cluster_center = reducde_data[cluster_centers_indices[k]]
    #print(len(class_members))
    plt.plot(reducde_data[class_members, 0], reducde_data[class_members, 1], col+'*')
    #plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
    #         markeredgecolor='k', markersize=9)
    for x in reducde_data[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

