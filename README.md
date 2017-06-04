# Machine Learning for Computer Vision Papers Reading

#### Pre-requisite:
- [pre-requisite.md][pre]
- [tools-frameworks.md][framework]

### Week - 1 [Back Propagation, Gradient Descent]
- Learning representations by back-propagating errors [David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams] [[Link]][backpro]

### Week - 2 [Haar Wavelets and Morlet Wavelets]
- Invariant Scattering Convolution Networks [Joan Bruna, Stephane Mallat] [[Link]][ISCN]

### Week - 3 [Back Propagation, SGD, Chain Rule in Maths]
- ImageNet Classification with Deep Convolutional Neural Networks [Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton] [[Link]][ImageNet]
- Very Deep Convolutional Networks for Large-Scale Image Recognition [Karen Simonyan, Andrew Zisserman] [[arXiv]][VeryDeepCN]
- Max-Pooling
- Overfitting, Saturation and Dropout

### Week - 4 [Intro to ResNet and Implementation]
- Going Deeper with Convolutions [Christian Szegedy, Wei Liu, Andrew Rabinvich] [[arXiv]][GoingDeeper]
- Deep Residual Learning for Image Recognition [Kaiming He, Xiangyu Zhang, Jian Sun] [[arXiv]][ResNet]
- Aggregated Residual Transformations for Deep Neural Networks [Saining Xie, Kaiming He] [[arXiv]][AggregatedResTrans]

### Week - 5 
N/A

### Week - 6 [Image Segmentation Pixel-Level Classification]
- Fully Convolutional Networks for Semantic Segmentation [Jonathan Long] [[Link]][FCNSS]
- Fast Approximate Energy Minimization via Graph Cuts [Yuri Boykov] [[Link]][FAEMGC]
- Exact optimization for Markov random fields with convex priors [Hiroshi Ishikawa]  [[Link]][EOMRFCP]
- “GrabCut” — Interactive Foreground Extraction using Iterated Graph Cuts [Carsten Rother] [[Link]][grabcut]

### Week - 7 
- Synergistic Face Detection and Pose Estimation with Energy-Based Models [Margarita Osadchy, Yann LeCun, Matthew L. Miller] [[Link]][synergface]
- Rapid Object Detection using a Boosted Cascade of Simple Features [Paul Viola, Michael Jones] [[Link]][viola]
- Real-Time Continuous Pose Recovery of Human Hands Using Convolutional Networks [[NYU Hand Pose Dataset]][NYU-Hand]

### Week - 8
- [Hand Video and Images Collection][nyu_hr]
- Additional Articles:
    + SSD: Single Shot MultiBox Detector [[Arxiv]][ssd]
    + R-CNN: Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation [[PDF]][rcnn]
    + YOLO:
        * You Only Look Once: Unified, Real-Time Object Detection [[Arxiv]][yolopaper]
        * [Another YOLO Paper][yolopaper2]
        * [Comparison][yolocomparison]

### Week - 9
- Generative Adversarial Nets (GAN) https://arxiv.org/pdf/1406.2661v1.pdf 
- Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks. Emily L. Denton et. al. Nips 2015. https://arxiv.org/abs/1506.05751 
- GANs with Feature Extraction Function:
    + ALI
    + BiGAN
    + Adversarial Autoencoder
    + Adversarial Variational Bayes.

### Week - 10 [Variational Auto Encoders]
- Tutorial on Variational Autoencoders(VAEs), Carl Doersch. August, 2016. https://arxiv.org/abs/1606.05908
    + Relevance: sampling from distributions inside deep networks, and can be trained with stochastic gradient descent. VAEs have already shown promise in generating many kinds of complicated data, including handwritten digits, faces, house numbers, CIFAR images, physical models of scenes, segmentation, and predicting the future from static images.
- "Variational Convolutional Networks for Human-Centric Annotations," 13th Asian Conference on Computer Vision, 2016. Tsung-Wei Ke, Che-Wei Lin, Tyng-Luh Liu and Davi Geiger, 
    + Relevance: Use of VAEs to annotate automatically images.

### Week - 11 + 12

### Week - 13 [Reinforcement Learning and Markov Decision Process]

### Week - 14 [Recurrent Neural Network]

### Week - 15 [Review Reinforcement Learning and Self-Driving Car]  



### Additional Topic - [Object Detection]
- RCNN: Region-based Convolutional Neural Networks
- Fast and Faster R-CNN: https://github.com/rbgirshick/py-faster-rcnn 
- YOLO
- SSD
- Soft-NMS: https://github.com/bharatsingh430/soft-nms

### Research Topics:
- Synthetic Gradients [[arxiv]][synthetic_gradients]
- One Shot Learning [[Wiki]][one_shot_learning] 
- Improving optimization in GAN

### Additional Resource: [Edit Before Submit:]
机器学习自学者必读的20篇顶级论文导读: https://mp.weixin.qq.com/s/ghMj3OO2yu7IIkQEkvkdVA
常见面试机器学习方法总览: http://www.chinakdd.com/article-oyU85v018dQL0Iu.html


[pre]:https://github.com/lizichen/Machine-Learning-For-Computer-Vision/blob/master/pre-requisite.md
[framework]:https://github.com/lizichen/Machine-Learning-For-Computer-Vision/blob/master/tools-frameworks.md
[backpro]:https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf
[ISCN]:https://www.di.ens.fr/~mallat/papiers/Bruna-Mallat-Pami-Scat.pdf 
[ImageNet]:https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
[VeryDeepCN]:https://arxiv.org/abs/1409.1556

[GoingDeeper]:https://arxiv.org/abs/1409.4842
[ResNet]:https://arxiv.org/abs/1512.03385
[AggregatedResTrans]:https://arxiv.org/abs/1611.05431

[FCNSS]: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf 
[FAEMGC]: http://www.cs.cornell.edu/~rdz/Papers/BVZ-pami01-final.pdf
[EOMRFCP]: http://www.f.waseda.jp/hfs/MRF.pdf 
[grabcut]:https://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf

[NYU-Hand]: http://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm
[synergface]: http://rita.osadchy.net/papers/OsadchyLeCunJMLR.pdf
[viola]: http://www.merl.com/publications/docs/TR2004-043.pdf

[nyu_hr]:https://github.com/lizichen/Machine-Learning-For-Computer-Vision/tree/master/NYU_HR 
[ssd]:https://arxiv.org/abs/1512.02325
[rcnn]:https://arxiv.org/pdf/1311.2524.pdf
[yolopaper]:https://arxiv.org/abs/1506.02640
[yolopaper2]:https://pjreddie.com/media/files/papers/YOLO9000.pdf
[yolocomparison]:https://pjreddie.com/darknet/yolo/ 



[one_shot_learning]: https://en.wikipedia.org/wiki/One-shot_learning
[synthetic_gradients]: https://arxiv.org/abs/1703.00522


