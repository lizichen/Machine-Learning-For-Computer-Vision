# List of Prerequisites [updating]

### Math Stuff
- Linear Algebra
- Matrix
- Calculus
- Statistics
- https://www.khanacademy.org/math
- https://www.khanacademy.org/math/linear-algebra
- https://www.khanacademy.org/math/multivariable-calculus
- https://www.khanacademy.org/math/statistics-probability
- http://datascience.ibm.com/blog/the-mathematics-of-machine-learning/

[//]: # (#################################### SECTION SEPERATOR ############################################)

### Textbook
- **The Elements of Statistical Learning** - (Hastie, Tibshirani, and Friedman)
  + Main textbook for L1 and L2 regularization, trees, bagging, random forests, and boosting.
- **Understanding Machine Learning: From Theory to Algorithms** - (Shalev-Shwartz and Ben-David)
  + primary reference for kernel methods and multiclass classification. 
- **An Introduction to Statistical Learning** (James, Witten, Hastie, and Tibshirani)
- **Pattern Recognition and Machine Learning** (Christopher Bishop)  
  + Primary reference for probabilistic methods, including bayesian regression, latent variable models, and the EM algorithm.
- **Bayesian Reasoning and Machine Learning** (David Barber)
  + Resource for topics in probabilistic modeling, and a possible substitute for the Bishop book.

[//]: # (#################################### SECTION SEPERATOR ############################################)

### Outstanding Knowledge Base:
#### Part I:
- What's **Machine Learning** in brief: Given training data set, discover underlying pattern so as to apply the pattern to new data. ML includes: Supervised Learning, Unsupervised Learning, etc.
- | Machine Learning      | Statistics                     |
|-----------------------|--------------------------------|
| Supervised Learning   | Regression, Classification     |
| Unsupervised Learning | Density Estimation, Clustering |
| Generalization        | Test set performance           |
| Learning              | Fitting                        |
| Weights               | Parameters                     |
| Network, Graphs       | Model                          |
- **Supervised Learning**: Training input data is labeled. (Supervised Learning includes **Classification** and **Regression**).
  + Classification（分类）: Learn a decision boundary.
    * Logistic Regression | Perceptron-like Algorithm | AdaBoost | SVMs (Linear Kernel for large data, RBF for small data) | Random Forests 
  + Regression（回归）: Learn to predict a continuous value.
    * Linear Regression | Random Forest
- **Unsupervised Learning** - Try to find hidden meaning/structures in the unlabelled data. (Clustering聚类, Outlier/Anomaly Detection异常值检测)
- Feature Engineering: The *process* of transforming raw data into **vectors** that can be used by ML algorithms for training.
- Hyperparameter Tuning: Tuen {Learning Rate, Regularization Constant, etc}
  + Process: Set the parameter values -> Train -> Evaluate Result -> Refine
- Types of Data set:
  + **Training phase**: Use training dataset to train model. Training Dataset has pairs of labeled input data and expected labeled output data.
  + **Validation phase**: Use validation dataset to tune parameters of a classifier, to find the best weights and bias to determine a stopping point for the *back-propagation process*.
  + **Test dataset**: only used to access the trained model, to get the error rate.
- MNIST
    - MNIST Beginner on Tensorflow: https://www.tensorflow.org/tutorials/mnist/beginners/#the_mnist_data 
- Tensor张量: 基于向量和矩阵的推广，可用来表示在一些向量、标量和其他张量之间的线性关系的多线性函数。
  + 通俗一点理解的话，我们可以将标量(Scalar, 只有大小，没有方向的量)视为零阶张量，矢量(Vector, 既有大小，又有方向的量)视为一阶张量，那么**矩阵**就是二阶张量（矩阵是一种表达方式）。

[//]: # (#################################### SECTION SEPERATOR ############################################)

#### Part II:
- Linear Regression
- Linear Classification
- Cross Validation
- 解释度
- R-Squared
- Multiple Linear Regression and Polynomial Regression
- L1 Regularization
- L2 Regularization
- Properties of Activation Functions
  + Non-Linear, Differentiable, Monotone
  + Reading: [How to Choose an Activation Function][how-to-choose-actFunc]
- Gradient Descent
  + Wiki: https://en.wikipedia.org/wiki/Gradient_descent
- **Stochastic Gradient Descent (SGD) 随机梯度下降**    
  + Both Gradient Descent (GD) and Stochastic Gradient Descent (SGD) update a set of parameters in an iterative manner to minimize an Error Function.
  + GD runs through **ALL the samples** in training set to do a single update for a parameter in one iteration. SGD uses **ONLY ONE** or a *SUBSET* of training samples from the training dataset to make one update per each iteration. If we use SUBSET, it's called *Mini-Batch SGD*.
  + If the size of training samples is large, using GD is not effective because in every iteration when updating the parameters, you are running through the complete training set. On the other hand, using SGD will be faster because you only use one training sample and it starts improving itself right away from the first sample. 
  + **SGD** often converges much **faster** compared to GD, but the error function is not as well minimized as in the case of GD. Often in most cases, the close approximation that you get in SGD for the parameter values are enough because they reach the optimal values and keep oscillating there.
- ADAM (**Ada**ptive **M**oment Estimation) optimizer: Computes adaptive learning rates for each parameter. 
- **ReLU** f(x) = max(0, x)
  + Greatly accelerate the convergence of stochastic gradient descent (compared to sigmoid/tanh functions).
    * Sigmoid函数的backpropagation很容易出现梯度消失(SGD die out)，在Sigmoid接近饱和区，变化太缓慢，导数趋近于0，导致信息丢失。
  + Non-saturating form.
  + ReLU is fragile and can 'die' - to fix this: **Leaky ReLU** & **batch normalization**. 
- Maxout
- Softmax
- Logistic Classifier
  + Logistic function: logistic(x) =  1 / (1 + numpy.exp(-x))
  + Derivative of logistic function: logistic_derivative(x) = logistic(x) * (1 - logistic(x))
- Bayes Classifier
- **SVM** (Support Vector Machine 支持向量机)
- Decision Tree
- **Entropy**
  + Why must we live in an universe that disorder is constantly growing and order is so hard to come by?
  + 热力学第二定律 (*Second Law of Thermodynamics*): 不可能把热从低温物体传到高温物体而不产生其他影响，或不可能从 **单一热源** 取热使之完全转换为有用的 **功** 而不产生其他影响，或不可逆热力过程中 **熵** 的微增量总是大于零。又称“熵增定律”，表明了在自然过程中，一个孤立系统的总混乱度（即“熵”）不会减小。
- Cross-Entropy Error Function
    - Very Simple Explainations: http://colah.github.io/posts/2015-09-Visual-Information/ 
- Boltzmann Constant
- Random Forest
- Learning Rate

[//]: # (#################################### SECTION SEPERATOR ############################################)

#### Part III: 
- **Local Response Normalization**: (Lateral Inhibition, in neuro-biology - in order to sharpen/boost the response.)
  + Because ReLU neurons have unbounded activations and we need LRN to normalize that. We want to detect high frequency features with a large response. If we normalize around the local neighborhood of the excited neuron, it becomes even more sensitive as compared to its neighbors.
  + LRN will dampen the responses that are uniformly large in any given local neighborhood. 
  + Discussed in Krizhevsky et al (**AlexNet Paper Sec 3.3**)
  + Two approaches: Normalize within the same channel, Normalize across channels.
- **Batch Normalization** vs **Dropout**: https://arxiv.org/pdf/1502.03167.pdf
  + Dropout - a technique for **regularization**. Dropout introduces noise into a neural network to force the neural network to learn to generalize well enough to deal with noise.
  + Batch normalization is mostly a technique for improving **optimization**. As a side effect, batch normalization happens to introduce some noise into the network, so it can regularize the model a little bit.
- **Local Constrast Normalization** (LCN)


[//]: # (#################################### SECTION SEPERATOR ############################################)




#### Part IV:
- **K-Means** (算法接受参数k,然后将事先输入的n个数据对象划分为k个聚类以便使得所获得的聚类满足聚类中的对象相似度较高，而不同聚类中的对象相似度较小)
  + Time Complexity: O(tkmn)，其中，t为迭代次数，k为簇的数目，m为记录数，n为维数；
  + Space Complexity：O((m+k)n)，其中，k为簇的数目，m为记录数，n为维数。
  + 适用范围:
    * K-menas算法试图找到使平凡误差准则函数最小的簇。当潜在的簇形状是凸面的，簇与簇之间区别较明显，且簇大小相近时，其聚类结果较理想。前面提到，该算法时间复杂度为O(tkmn)，与样本数量线性相关，所以，对于处理大数据集合，该算法非常高效，且伸缩性较好。但该算法除了要事先确定簇数K和对初始聚类中心敏感外，经常以局部最优结束，同时对“噪声”和孤立点敏感，并且该方法不适于发现非凸面形状的簇或大小差别很大的簇。
  + 缺点：
    * 聚类中心的个数K 需要事先给定，但在实际中这个 K 值的选定是非常难以估计的，很多时候，事先并不知道给定的数据集应该分成多少个类别才最合适； 
    * Kmeans需要人为地确定初始聚类中心，不同的初始聚类中心可能导致完全不同的聚类结果。（可以使用K-means++算法来解决）  
- K-Means++  
- KNN (K-Nearest-Neighbors)
  + instead of finding the single closest image in the training set, we will find the top k closest images, and have them vote on the label of the test image.
  + in more general term, if we have a labeled data set {x_i}, and we need to classify some new item called y, find the k elements in {x_i} data set that are closest to y, and then find the approach to average their labels to get the label of y.
    * What is the distance, or how to define being closest? how to measure the distance between x and y?
    * How to calculate the average of all the labels of the k x items?
  + Using KNN for MNIST: http://andrew.gibiansky.com/blog/machine-learning/k-nearest-neighbors-simplest-machine-learning/
  +        
    <table>
      <tr><td colspan="2"><b>Difference between K-Means and KNN</b></td></tr>
      <tr><td><b>KNN</b></td><td><b>K-Means</b></td></tr>
      <tr><td>KNN categorizes data</td><td>K-Means groups data</td></tr>
      <tr><td>Supervised Learning</td><td>Non-Supervised Learning</td></tr>
      <tr><td>Feed correctly labeled data</td><td>Feed non-labeled disordered data</td></tr>
      <tr><td>No training process, memory-based learning</td><td>Need training process</td></tr>
      <tr><td colspan="2"><b>Meaning of K</b></td></tr>  
      <tr>
        <td>Sample data x to be categorized, find the closest K items to the sample data x, where label L has the majority in the K items, label x as L.</td>
        <td>Preset K, assuming the dataset can be splited into K groups. Need analytics beforehand to determine K (But in reality, it's hard to guess the value of K, can use <b>K-Means++ algo</b> to improve).</td>
      </tr>
      <tr><td colspan="2"><b>Similarity</b></td></tr>
      <tr><td colspan="2">Both have such procedure: Given a point, find the closest point(s). Both uses the concept of NN(Nears Neighbor), use <a href="https://en.wikipedia.org/wiki/K-d_tree"><b>KD Tree(K-Dimensional Tree) algorithm</b></a> to implement NN.</td></tr>
    </table>  
- [Restricted Boltzmann Machines][rbm]
- AdaBoost
- t-SNE: http://distill.pub/2016/misread-tsne/ 
- Jonker-Volgenant Algorithm + t-SNE: https://blog.sourced.tech/post/lapjv/?utm_source=Mailing+list&utm_campaign=0c93d571d7-Kaggle_Newsletter_04-11-2017&utm_medium=email&utm_term=0_f42f9df1e1-0c93d571d7-400609233 
- Markov Chain
- Markov Decision Process (MDP)
  + States: S1, S2, ... Sn
  + Actions: a_1, a_2, ... a_m
  + State Transition Matrix: T(s,a,s_prime)  =  P(s1|a, s) (Conditional State Transitional Probability: executing action a on state s -> have s1)
  + To determine the purpose of the MDP: define Reward Function: R(s)
  + Value Function:
  + Value Iteration: 


[//]: # (#################################### SECTION SEPERATOR ############################################)

### Online Resources:
- Classification Dataset Results: http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#4d4e495354 
- Popular Public Datasets for AI: https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2#.s0y36g3m5
- Standford CS229 Machine Learning Course Materials: http://cs229.stanford.edu/materials.html and 2016 Final Projects: http://cs229.stanford.edu/projects2016.html

### OTHER TUTORIALS AND REFERENCES:
- Carlos Fernandez-Granda's lecture notes provide a comprehensive review of the prerequisite material in linear algebra, probability, statistics, and optimization.
- **Step-by-Step Backpropagation:** https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
- Brian Dalessandro's iPython notebooks from DS-GA-1001: Intro to Data Science
- The Matrix Cookbook has lots of facts and identities about matrices and certain probability distributions.
- Stanford CS229: "Review of Probability Theory"
- Stanford CS229: "Linear Algebra Review and Reference"
- Math for Machine Learning by Hal Daumé III
- Deep Learning Courses List: [http://machinelearningmastery.com/deep-learning-courses/][deeplearningcourselist] 
- Code tutorials: [http://deeplearning.net/reading-list/tutorials/][codetutorial]
- 从零开始掌握Python机器学习：十四步教程: [https://zhuanlan.zhihu.com/p/25761248][python14steps]
- Google ML Presentation "Nail Your Next ML Gig" - Natalia Ponomareva
- 谷歌微软等科技巨头数据科学岗位面试题（108道）- http://mp.weixin.qq.com/s/JoSgGHa8-ara2bz2L9ppTw 
- Understanding the backward pass through Batch Normalization Layer: http://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
- Sample Data Normalization Process: http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing#Data_Normalization
  + Simple Rescaling
  + Per-example mean subtraction (a.k.a. remove DC)
  + Feature Standardization 
- https://www.sitepoint.com/simple-ways-to-build-artificial-intelligence/

### 大数据学习网站 https://mp.weixin.qq.com/s/iiANLHqAxVRfy6iU0brhRg 
- 一、如何用R的处理大数据 http://www.xmind.net/m/LKF2/
- 二、R语言的工具包 https://cran.r-project.org/web/views/
- 三、帮你获得python大数据处理工具大全 http://www.xmind.net/m/WvfC
- 四、学习Python语言的 Learn Python the Hard Way https://learnpythonthehardway.org/book/
- 五、SAS图例集 http://robslink.com/SAS/Home.htm
- 六、美国布朗大学概率和统计的可视化导论，一个非常棒的可视化概率及统计的学习网站 http://students.brown.edu/seeing-theory/?vt=4
- 七、教你玩动态的GIF图表 http://lenagroeger.com/
- 八、如何选择机器学习算法 http://blogs.sas.com/content/subconsciousmusings/2017/04/12/machine-learning-algorithm-use/
- 九、一套数据，25种可视化 http://flowingdata.com/2017/01/24/one-dataset-visualized-25-ways
- 十、大数据数据处理资源 http://usefulstuff.io/big-data/
- 十一、里面蕴含编程语言、机器算法、大数据等，内容巨丰富 https://www.analyticsvidhya.com/blog/2017/02/top-28-cheat-sheets-for-machine-learning-data-science-probability-sql-big-data/
- 十三、学习数据挖掘、机器学习的好网站，写的很全面很系统，适合各个级别的高手 https://www.autonlab.org/tutorials

[//]: # (#################################### SECTION SEPERATOR ############################################)
### Knowledge Graphs:
![Alt text](./Resource/ai-full-dev-cycle.jpeg?raw=true "ai-full-dev-cycle")
![Alt text](./Resource/machine-learning.png?raw=true "machine-learning")
![Alt text](./Resource/python-ecosystem.png?raw=true "python-ecosystem.png")
![Alt text](./Resource/ML_Algorithms.jpg?raw=true "ML_Algorithms.jpg")
![Alt text](./Resource/MLKnowledgeGraph.png?raw=true "MLKnowledgeGraph.png")
![Alt text](./Resource/MI-Landscape-3_7.png?raw=true "MI-Landscape-3_7.png")



[//]: # (#################################### SECTION SEPERATOR ############################################)

[python14steps]: https://zhuanlan.zhihu.com/p/25761248
[how-to-choose-actFunc]: http://papers.nips.cc/paper/874-how-to-choose-an-activation-function.pdf
[deeplearningcourselist]:http://machinelearningmastery.com/deep-learning-courses/ 
[codetutorial]:http://deeplearning.net/reading-list/tutorials/
[rbm]:http://deeplearning.net/tutorial/rbm.html#rbm 
