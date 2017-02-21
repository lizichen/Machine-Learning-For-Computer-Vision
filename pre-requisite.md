# List of Prerequisites [updating]

### Math Stuff
- Linear Algebra
- Matrix
- Calculus
- Statistics
- https://www.khanacademy.org/math
- http://datascience.ibm.com/blog/the-mathematics-of-machine-learning/

### Knowledgebase:
- Linear Regression
- R-Squared
- Multiple Linear Regression and Polynomial Regression
- L2 Regularization
- Gradient Descent
    - Wiki: https://en.wikipedia.org/wiki/Gradient_descent
- L1 Regularization
- Linear Classification
- Logistic Classifier
- Bayes Classifier
- Cross-Entropy Error Function
    - Very Simple Explainations: http://colah.github.io/posts/2015-09-Visual-Information/ 
- MNIST
    - MNIST Beginner on Tensorflow: https://www.tensorflow.org/tutorials/mnist/beginners/#the_mnist_data 
- Supervised vs Unsupervised
- Random Forest
- K-Means(算法接受参数k,然后将事先输入的n个数据对象划分为k个聚类以便使得所获得的聚类满足聚类中的对象相似度较高，而不同聚类中的对象相似度较小)
  + Time Complexity: O(tkmn)，其中，t为迭代次数，k为簇的数目，m为记录数，n为维数；
  + Space Complexity：O((m+k)n)，其中，k为簇的数目，m为记录数，n为维数。
  + 适用范围:
    * K-menas算法试图找到使平凡误差准则函数最小的簇。当潜在的簇形状是凸面的，簇与簇之间区别较明显，且簇大小相近时，其聚类结果较理想。前面提到，该算法时间复杂度为O(tkmn)，与样本数量线性相关，所以，对于处理大数据集合，该算法非常高效，且伸缩性较好。但该算法除了要事先确定簇数K和对初始聚类中心敏感外，经常以局部最优结束，同时对“噪声”和孤立点敏感，并且该方法不适于发现非凸面形状的簇或大小差别很大的簇。
  + 缺点：
    * 聚类中心的个数K 需要事先给定，但在实际中这个 K 值的选定是非常难以估计的，很多时候，事先并不知道给定的数据集应该分成多少个类别才最合适； 
    * Kmeans需要人为地确定初始聚类中心，不同的初始聚类中心可能导致完全不同的聚类结果。（可以使用K-means++算法来解决）  
- KNN (K-Nearest-Neighbors)
  + instead of finding the single closest image in the training set, we will find the top k closest images, and have them vote on the label of the test image.
  + in more general term, if we have a labeled data set {x_i}, and we need to classify some new item called y, find the k elements in {x_i} data set that are closest to y, and then find the approach to average their labels to get the label of y.
    * What is the distance, or how to define being closest? how to measure the distance between x and y?
    * How to calculate the average of all the labels of the k x items?
  + Using KNN for MNIST: http://andrew.gibiansky.com/blog/machine-learning/k-nearest-neighbors-simplest-machine-learning/  





- Stochastic Gradient Descent (SGD)  
- Types of Data set:
  + Training phase: Use training dataset to train model. Training Dataset has pairs of labeled input data and expected labeled output data.
  + Validation phase: Use validation dataset to tune parameters of a classifier, to find the best weights and bias to determine a stopping point for the back-propagation process.
  + Test dataset: only used to access the trained model, to get the error rate.



### Online Resources:
- Classification Dataset Results: http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#4d4e495354 
- Popular Public Datasets for AI: https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2#.s0y36g3m5


