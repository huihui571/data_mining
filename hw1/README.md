# homework1
首先在mnist数据集上分别运行以下几种聚类算法，得到运行结果和评价。
## Kmeans

数据先进行标准化，然后应用各种评价指标进行比较。如下表，从上到下依次是按照Kmeans++方法初始化、randoma初始化以及PCA降维后的运行结果。可以看出，经过PCA降维操作，选取数据特征空间10个主成分轴作为初始聚类中心，可以大大减少算法运行时间，并在一致性等指标上也有显著提升。

![KMeans-metrics.PNG](https://i.loli.net/2019/10/09/h2CV5K31r4oOR6F.png)  
![K-means.png](https://i.loli.net/2019/10/09/yMuDvOBoIFZafUE.png)   

## Affinity propagation
AP聚类和谱聚类都是基于图的聚类。  

主要思想：将全部样本看作网络的节点，然后通过网络中各条边的消息传递计算出各样本的聚类中心。聚类过程中，共有两种消息在各节点间传递，分别是吸引度(responsibility)和归属度(availability)。AP算法通过迭代过程不断更新每一个点的吸引度和归属度值，直到产生m个高质量的Exemplar（类似于质心），同时将其余的数据点分配到相应的聚类中。  
适合：高维、多类别数据的快速聚类。
  
由于AP算法并没有指定聚类中心k的值，所以每次聚类得到的中心个数是不确定的，与preference参数的取值有关。该参数表示每个点被选举为中心的概率，如果设一个值，则表示所有点被选举为中心的概率相同。该参数恒为负，且值越小生成的聚类中心个数越少。

![AP_mnist-1.png](https://i.loli.net/2019/10/10/BFhPLd1Da2uTeYx.png)   
```
Run time: 4.32
Estimated number of clusters: 10
Homogeneity: 0.590
Completeness: 0.623
V-measure: 0.606
Adjusted Rand Index: 0.474
Adjusted Mutual Information: 0.602
Silhouette Coefficient: 0.379
```

## Mean Shift
MeanShift算法的关键操作是通过感兴趣区域内的数据密度变化计算中心点的漂移向量，从而移动中心点进行下一次迭代，直到到达密度最大处（中心点不变）。从每个数据点出发都可以进行该操作，在这个过程，统计出现在感兴趣区域内的数据的次数。该参数将在最后作为分类的依据。

与K-Means算法不一样的是，MeanShift算法可以自动决定类别的数目。与K-Means算法一样的是，两者都用集合内数据点的均值进行中心点的移动。

![MeanShift原理](https://img-blog.csdn.net/20150327144310549)

由于MeanShift算法同样没有指定聚类中心的个数，所以也不能保证恰好分为10个类，需要调整quantile参数。该参数用于估计窗口大小，越小则窗口越小，生成的类越多，等于0.5则窗口大小取所有采样点对距离的中位数。  

![MeanShift_mnist.png](https://i.loli.net/2019/10/10/K372mJsFzurLlbo.png)  
```
Run time: 3.33
number of estimated clusters : 10
Homogeneity: 0.844
Completeness: 0.851
V-measure: 0.847
Adjusted Rand Index: 0.791
Adjusted Mutual Information: 0.846
Silhouette Coefficient: 0.759 
```
PS:该算法在PCA降维后聚类效果很差，换用另一种降维方法t-SNE后效果明显变好。

## Spectral clustering
