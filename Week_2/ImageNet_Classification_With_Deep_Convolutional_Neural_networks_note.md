#### Note for
##*ImageNet Classification with Deep Convolutional Neural Networks*

Paper link: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf   
GPU implementation of 2D CNN: http://code.google.com/p/cuda-convnet/

##### Numbers and Facts:
> - Task: to classify 1.2 million labeled images (ILSVRC-2010 contest) into 1000 classes. 50k validation images, 150k testing images.
> - Images are down-sampled to a fixed resolution of 256-by-256.
> - Result: Top-1 Err. 37.5% Top-5 Err. 17.0% (ILSVRC-2010), Top-5 Err. 15.3% (ILSVRC-2012)
> - Neural Network: 60 million parameters, 650k neurons, 5 Conv Layers, 3 fully-connected, 1000-way Softmax.

##### Introduction:
To open CNN to such task, the paper says:
> To learn abot thousands of objects from millions of images, we need a model with a large learning capacity.    

Cited from:    
1. Learning methods for generic object recognition with invariance to pose and lighting [Yann LeCun]    
2. What is the best multi stage architecture for object recognition? [Kevin Jarrett]

##### Data Preparation:
- Down-sampled the images to a fixed resolution of 256*256
- Subtract the mean activity over the training set from each pixel. [Image normalization?]
	+ **Image Normalization**: [[wiki]][normalization_wiki]
		* Normalization is a process that changes the range of pixel intensity values.
		* For example: if the intensity range of the image is [50, 180] and the desired range is [0, 255] the process entails subtracting 50 from each of pixel intensity, making the range [0, 130]. Then each pixel intensity is multiplied by 255/130, making the range [0, 255].
	+ Why do we normalize images by **subtracting the dataset's image mean** and not the current image mean in deep learning?: [[stackexchange]][normalize_image_stackexchange] 
		* 1. Subtract the mean per channel calculated over all images
		* 2. Subtract by pixel/channel calculated over all images
		* Related Resources:
			- Color Normalization [[wiki]][color_normalization_wiki]
			- Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift [[arVix]][batch_normal]
			- Local Contrast Normalization (per-image) [[pierre_sermanet]][pierre_sermanet]

### Network Architecture:
![alt text](http://www.nallatech.com/wp-content/uploads/CNN-Figure-02.png)

##### ReLU Nonlinearity (Non-saturating *f(x) = max(0, x)*) [[Original Nair and Hinton's Paper]][relu_paper]
- Saturating nonlinearities: tanh(x) and sigmoid(x) 
- ReLU trains faster than saturating nonlinearities, gradients flow along a few paths instead of all possible paths (as in saturating ones) resulting to fsater convergence. Better performance can be achieved by reducing training time for each epoch and training larger datasets to prevent overfitting. 
- ReLUs have the desirable property that they do not require input normalization to prevent from saturating. However, they find that a **local response normalization** scheme after applying the ReLU nonlinearity can reduce their top-1 and top-5 error rates by 1.4% and 1.2%.



##### Max-Pooling:

##### Softmax:

##### Dropout:

##### Better techniques for preventing overfitting [Secion 4]

##### Improve performance and reduce training time [Section 3]


### Resources to go through:
http://jamiis.me/submodules/presentation-convolutional-neural-nets/#/



[normalization_wiki]: https://en.wikipedia.org/wiki/Normalization_(image_processing)
[normalize_image_stackexchange]: http://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c
[color_normalization_wiki]: https://en.wikipedia.org/wiki/Color_normalization
[batch_normal]: https://arxiv.org/abs/1502.03167
[pierre_sermanet]: https://www.cs.nyu.edu/media/publications/sermanet_pierre.pdf
[relu_paper]: http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_NairH10.pdf

