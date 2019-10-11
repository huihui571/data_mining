# homework1
首先在mnist数据集上分别运行以下几种聚类算法，得到运行结果和评价。
## Kmeans

数据先进行标准化，然后应用各种评价指标进行比较。如下表，从上到下依次是按照Kmeans++方法初始化、randoma初始化以及PCA降维后的运行结果。可以看出，经过PCA降维操作，选取数据特征空间10个主成分轴作为初始聚类中心，可以大大减少算法运行时间，并在一致性等指标上也有显著提升。

![KMeans-metrics-2.PNG](https://i.loli.net/2019/10/11/MTx4VrWnZFaNldf.png)    
![K-means.png](https://i.loli.net/2019/10/09/yMuDvOBoIFZafUE.png)   

## Affinity propagation
AP聚类和谱聚类都是基于图的聚类。  

主要思想：将全部样本看作网络的节点，然后通过网络中各条边的消息传递计算出各样本的聚类中心。聚类过程中，共有两种消息在各节点间传递，分别是吸引度(responsibility)和归属度(availability)。AP算法通过迭代过程不断更新每一个点的吸引度和归属度值，直到产生m个高质量的Exemplar（类似于质心），同时将其余的数据点分配到相应的聚类中。  
适合：高维、多类别数据的快速聚类。
  
由于AP算法并没有指定聚类中心k的值，所以每次聚类得到的中心个数是不确定的，与preference参数的取值有关。该参数表示每个点被选举为中心的概率，如果设一个值，则表示所有点被选举为中心的概率相同。该参数恒为负，且值越小生成的聚类中心个数越少。
```
Run time: 4.32
Estimated number of clusters: 10
Homogeneity: 0.590
Completeness: 0.623
```
![AP_mnist-1.png](https://i.loli.net/2019/10/10/BFhPLd1Da2uTeYx.png)   

## Mean Shift
MeanShift算法的关键操作是通过感兴趣区域内的数据密度变化计算中心点的漂移向量，从而移动中心点进行下一次迭代，直到到达密度最大处（中心点不变）。从每个数据点出发都可以进行该操作，在这个过程，统计出现在感兴趣区域内的数据的次数。该参数将在最后作为分类的依据。

与K-Means算法不一样的是，MeanShift算法可以自动决定类别的数目。与K-Means算法一样的是，两者都用集合内数据点的均值进行中心点的移动。

![MeanShift原理](https://img-blog.csdn.net/20150327144310549)

由于MeanShift算法同样没有指定聚类中心的个数，所以也不能保证恰好分为10个类，需要调整quantile参数。该参数用于估计窗口大小，越小则窗口越小，生成的类越多，等于0.5则窗口大小取所有采样点对距离的中位数。  
```
Run time: 3.33
number of estimated clusters : 10
Homogeneity: 0.844
Completeness: 0.851 
```
![MeanShift_mnist.png](https://i.loli.net/2019/10/10/K372mJsFzurLlbo.png)  
PS:该算法在PCA降维后聚类效果很差，换用另一种降维方法t-SNE后效果明显变好。

## Spectral clustering
谱聚类是一种基于图论的聚类方法，通过对样本数据的拉普拉斯矩阵的特征向量进行聚类，从而达到对样本数据聚类的母的。谱聚类可以理解为将高维空间的数据映射到低维，然后在低维空间用其它聚类算法（如KMeans）进行聚类。   
先根据样本点计算相似度矩阵，然后计算度矩阵和拉普拉斯矩阵，接着计算拉普拉斯矩阵前k个特征值对应的特征向量，最后将这k个特征值对应的特征向量组成n*k的矩阵U，U的每一行成为一个新生成的样本点，对这些新生成的样本点进行k-means聚类。   
谱聚类适用于均衡分类问题，即各簇之间点的个数相差不大，对于簇之间点个数相差悬殊的聚类问题，谱聚类则不适用。  
```
Run time: 0.36
Homogeneity: 0.628
Completeness: 0.663
Normal Mutual Information: 0.645
```
![Spectral_mnist.png](https://i.loli.net/2019/10/11/n5EpTebwVtMzf2c.png)   

## Agglomerative Clustering
凝聚的层次聚类：先计算样本之间的距离，每次将距离最近的点合并到同一个类。然后，再计算类与类之间的距离，将距离最近的类合并为一个大类，不停的合并，直到合成了一个类。其中类与类的距离的计算方法有：最短距离法，最长距离法，中间距离法，类平均法等。比如最短距离法，将类与类的距离定义为类与类之间样本的最短距离。
```
ward :	0.33s
Homogeneity: 0.511
Completeness: 0.531
Normal Mutual Information: 0.521
```
![Agg_mnist-1.png](https://i.loli.net/2019/10/11/RnCiDtI7SlFW1Jg.png)  
```
average :	0.27s
Homogeneity: 0.502
Completeness: 0.551
Normal Mutual Information: 0.525
``` 
![Agg_mnist-4.png](https://i.loli.net/2019/10/11/XGCgfwmKd13z6Tv.png) 
```
complete :	0.28s
Homogeneity: 0.477
Completeness: 0.533
Normal Mutual Information: 0.503
```
![Agg_mnist-3.png](https://i.loli.net/2019/10/11/V6HBoar3P4WihYT.png)  
```
single :	0.14s
Homogeneity: 0.121
Completeness: 0.845
Normal Mutual Information: 0.212
```
![Agg_mnist-2.png](https://i.loli.net/2019/10/11/xvjaMO1KIzdegD6.png)  

## DBSCAN
DBSCAN（具有噪声的基于密度的聚类方法）是一种基于密度的空间聚类算法。该算法将具有足够密度的区域划分为簇，并在具有噪声的空间数据库中发现任意形状的簇，它将簇定义为密度相连的点的最大集合。 
```
Estimated number of clusters: 14
Estimated number of noise points: 114
Homogeneity: 0.051
Completeness: 0.232
Normal Mutual Information: 0.083
```
该算法在mnist数据集可视化结果并不好，难道是数据分布和密度无关？
![DBSCAN_mnist.png](https://i.loli.net/2019/10/11/SDt4I8wOfsnHpWx.png)   

## GMM
混合高斯模型，它通过求解两个高斯模型，并通过一定的权重将两个高斯模型融合成一个模型，即最终的混合高斯模型。它的本质就是融合几个单高斯模型，来使得模型更加复杂，从而产生更复杂的样本。理论上，如果某个混合高斯模型融合的高斯模型个数足够多，它们之间的权重设定得足够合理，这个混合模型可以拟合任意分布的样本。
```
spherical: 0.04
Homogeneity: 0.529
Completeness: 0.542
Normal Mutual Information: 0.535
diag: 0.04
Homogeneity: 0.542
Completeness: 0.553
Normal Mutual Information: 0.548
tied: 0.05
Homogeneity: 0.559
Completeness: 0.573
Normal Mutual Information: 0.566
full: 0.06
Homogeneity: 0.556
Completeness: 0.562
Normal Mutual Information: 0.559
```
![GMM_mnist.png](https://i.loli.net/2019/10/11/AZ1csgMXlKtVxTW.png)   


## 20newsGroup数据集

