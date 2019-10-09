# homework1
首先在mnist数据集上分别运行以下几种聚类算法，得到运行结果和评价。
## Kmeans

数据先进行标准化，然后应用各种评价指标进行比较。如下表，从上到下依次是按照Kmeans++方法初始化、randoma初始化以及PCA降维后的运行结果。可以看出，经过PCA降维操作，选取数据特征空间10个主成分轴作为初始聚类中心，可以大大减少算法运行时间，并在一致性等指标上也有显著提升。

![metrices](https://github.com/huihui571/data_mining/blob/master/hw1/assets/KMeans-metrics.PNG)
![PCA-KMeans](https://github.com/huihui571/data_mining/blob/master/hw1/assets/K-means.png)

## Affinity propagation
AP聚类和谱聚类都是基于图的聚类。  
主要思想：将全部样本看作网络的节点，然后通过网络中各条边的消息传递计算出各样本的聚类中心。聚类过程中，共有两种消息在各节点间传递，分别是吸引度(responsibility)和归属度(availability)。AP算法通过迭代过程不断更新每一个点的吸引度和归属度值，直到产生m个高质量的Exemplar（类似于质心），同时将其余的数据点分配到相应的聚类中。  
适合：高维、多类别数据的快速聚类。  
由于AP算法并没有指定聚类中心k的值，所以每次聚类得到的中心个数是不确定的，与preference参数的取值有关。preference参数表示每个点被选举为中心的概率，如果设一个值，则表示所有点被选举为中心的概率相同。该参数恒为负，且值越小生成的聚类中心个数越少。

![AP](https://github.com/huihui571/data_mining/blob/master/hw1/assets/AP_mnist-2.png)

## Mean Shift