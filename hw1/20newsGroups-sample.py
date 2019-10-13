from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from time import time

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
dataset = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
labels = dataset.target

# Perform an IDF normalization on the output of HashingVectorizer
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
X = vectorizer.fit_transform(dataset.data)

# Vectorizer results are normalized, which makes KMeans behave as
# spherical k-means for better results. Since LSA/SVD results are
# not normalized, we have to redo the normalization.
svd = TruncatedSVD(4)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)

'''
运行聚类算法
'''
print('|算法|time|Homogeneity|Completeness|NMI|')
print('|:- |:-: |:-: | :-: | :-: |')
def run_clusters(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('|%-9s\t|%.2fs\t|%.3f\t|%.3f\t|%.3f|'
          % (name, (time() - t0),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.normalized_mutual_info_score(labels,  estimator.labels_,
                                                average_method='arithmetic')))

run_clusters(KMeans(init='k-means++', n_clusters=4, n_init=10),
				name="kMeans", data=X)
run_clusters(AffinityPropagation(preference=-0.04),
				name="AffinityPropagation", data=X)
bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=500)
run_clusters(MeanShift(bandwidth=bandwidth, bin_seeding=False),
				name="MeanShift", data=X)
run_clusters(SpectralClustering(n_clusters=4, eigen_solver='arpack'),
				name="SpectralClustering", data=X)
run_clusters(AgglomerativeClustering(linkage='ward', n_clusters=4),
				name="Agg-ward", data=X)
run_clusters(AgglomerativeClustering(linkage='average', n_clusters=4),
				name="Agg-average", data=X)
run_clusters(AgglomerativeClustering(linkage='complete', n_clusters=4),
				name="Agg-complete", data=X)
run_clusters(AgglomerativeClustering(linkage='single', n_clusters=4),
				name="Agg-single", data=X)
run_clusters(DBSCAN(eps=0.05, min_samples=4),
 				name="DBSCAN", data=X)
