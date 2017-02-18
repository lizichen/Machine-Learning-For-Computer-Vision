##**Project Ideas for** **Computer Vision and Deep Learning**   
####Classification   
**1. Low signal-noise recognition.**     
Input: Train SVHN/CIFAR-10 with AWGN, Shot noise of different levels.     
Output: Class     
Difficulty: Easy     
**2. Flower recognition**     
Input: Flower image      
Output: Species type     
Refs: [http://www.robots.ox.ac.uk/~vgg/data/flowers/](http://www.robots.ox.ac.uk/~vgg/data/flowers/)   
**3. Tiny Imagenet**      
Input: low-res version of imagenet     
Output: class     
Difficulty: Easy/Medium   
**4. Road sign classification**     
Input: image of road sign     
Output: type      
Refs: http://people.idsia.ch/~juergen/nn2012traffic.pdf      
Difficulty: Easy   
**5. Galaxy Zoo**   
Input: Galaxy image         
Output:        
Refs: [https://www.galaxyzoo.org/](https://www.galaxyzoo.org/)     
[https://arxiv.org/abs/1503.07077](https://arxiv.org/abs/1503.07077)     
Difficulty: Hard      
**6. Signature verification**      
[http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_(SigComp2011)](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_(SigComp2011))         
Difficulty: Medium/Hard   
####Detection      
**7. Satellite imagery**      
Input: Satellite image      
Output: label map of roads/vegetation etc.         
Refs: [http://csc.lsu.edu/~saikat/deepsat/](http://csc.lsu.edu/~saikat/deepsat/)         
[https://www.cs.toronto.edu/~vmnih/docs/](https://www.cs.toronto.edu/~vmnih/docs/road_detection.pdf)**[roa**d](https://www.cs.toronto.edu/~vmnih/docs/road_detection.pdf)[_detection.pdf](https://www.cs.toronto.edu/~vmnih/docs/road_detection.pdf)      
[Google for more]      
Difficulty: medium/hard      
**8. Diabetic retinopathy**      
Input: Retina image      
Output: map of diseased regions      
Refs: [Rob has data]      
Difficulty: Medium      
**9. Face recognition**      
Input: Pair of face images      
Output: Same / not-same      
Refs: [Labelled Faces from Wild], https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf      
Difficulty: Hard      
**10. Breast cancer screening**   
Input: Mammogram image   
Output: highlight suspect regions   
Refs: [http://www.mammoimage.org/databases/](http://www.mammoimage.org/databases/) [Google for more]   
Difficulty: Hard   
**11. Melanoma detection**   
Input: image of skin mole   
Output: classification of mole type   
Refs: [http://homepages.inf.ed.ac.uk/rbf/DERMOFIT/](http://homepages.inf.ed.ac.uk/rbf/DERMOFIT/)   
[http://homepages.inf.ed.ac.uk/rbf/DERMOFIT/datasets.htm](http://homepages.inf.ed.ac.uk/rbf/DERMOFIT/datasets.htm)   
[Google for more]   
Difficulty: medium/hard   
####Image + Text   
**12. Image captioning**   
Input: Image   
Output: short sentence describing scene   
Refs: [http://mscoco.org/](http://mscoco.org/); http://cs.stanford.edu/people/karpathy/sfmltalk.pdf   
[https://people.eecs.berkeley.edu/~sgupta/pdf/captions.pdf](https://people.eecs.berkeley.edu/~sgupta/pdf/captions.pdf)   
Tutorial on RNNs: http://www.coling-2014.org/COLING%202014%20Tutorial-fix%20-%20Tomas%20Mikolov.pdf   
Papers to implement:   
[http://www.cs.cmu.edu/~xinleic/papers/cvpr15_rnn.pdf](http://www.cs.cmu.edu/~xinleic/papers/cvpr15_rnn.pdf)   
* UCLA / Baidu [[Paper]](http://arxiv.org/pdf/1410.1090)   
    * Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, Alan L. Yuille, Explain Images with Multimodal Recurrent Neural Networks, arXiv:1410.1090.   
* Toronto [[Paper]](http://arxiv.org/pdf/1411.2539)   
    * Ryan Kiros, Ruslan Salakhutdinov, Richard S. Zemel, Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models, arXiv:1411.2539.   
* Berkeley [[Paper]](http://arxiv.org/pdf/1411.4389)   
    * Jeff Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, Trevor Darrell, Long-term Recurrent Convolutional Networks for Visual Recognition and Description, arXiv:1411.4389.   
* Google [[Paper]](http://arxiv.org/pdf/1411.4555)   
    * Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan, Show and Tell: A Neural Image Caption Generator, arXiv:1411.4555.   
* Stanford [[Web]](http://cs.stanford.edu/people/karpathy/deepimagesent/) [[Paper]](http://cs.stanford.edu/people/karpathy/cvpr2015.pdf)   
    * Andrej Karpathy, Li Fei-Fei, Deep Visual-Semantic Alignments for Generating Image Description, CVPR, 2015.   
* UML / UT [[Paper]](http://arxiv.org/pdf/1412.4729)   
    * Subhashini Venugopalan, Huijuan Xu, Jeff Donahue, Marcus Rohrbach, Raymond Mooney, Kate Saenko, Translating Videos to Natural Language Using Deep Recurrent Neural Networks, NAACL-HLT, 2015.   
* CMU / Microsoft [[Paper-arXiv]](http://arxiv.org/pdf/1411.5654) [[Paper-CVPR]](http://www.cs.cmu.edu/~xinleic/papers/cvpr15_rnn.pdf)   
    * Xinlei Chen, C. Lawrence Zitnick, Learning a Recurrent Visual Representation for Image Caption Generation, arXiv:1411.5654.   
    * Xinlei Chen, C. Lawrence Zitnick, Mind’s Eye: A Recurrent Visual Representation for Image Caption Generation, CVPR 2015   
* Microsoft [[Paper]](http://arxiv.org/pdf/1411.4952)   
    * Hao Fang, Saurabh Gupta, Forrest Iandola, Rupesh Srivastava, Li Deng, Piotr Dollár, Jianfeng Gao, Xiaodong He, Margaret Mitchell, John C. Platt, C. Lawrence Zitnick, Geoffrey Zweig, From Captions to Visual Concepts and Back, CVPR, 2015.   
* Univ. Montreal / Univ. Toronto [[Web](http://kelvinxu.github.io/projects/capgen.html)] [[Paper](http://www.cs.toronto.edu/~zemel/documents/captionAttn.pdf)]   
    * Kelvin Xu, Jimmy Lei Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard S. Zemel, Yoshua Bengio, Show, Attend, and Tell: Neural Image Caption Generation with Visual Attention, arXiv:1502.03044 / ICML 2015   
* Idiap / EPFL / Facebook [[Paper](http://arxiv.org/pdf/1502.03671)]   
    * Remi Lebret, Pedro O. Pinheiro, Ronan Collobert, Phrase-based Image Captioning, arXiv:1502.03671 / ICML 2015   
* UCLA / Baidu [[Paper](http://arxiv.org/pdf/1504.06692)]   
    * Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, Zhiheng Huang, Alan L. Yuille, Learning like a Child: Fast Novel Visual Concept Learning from Sentence Descriptions of Images, arXiv:1504.06692   
* MS + Berkeley   
    * Jacob Devlin, Saurabh Gupta, Ross Girshick, Margaret Mitchell, C. Lawrence Zitnick, Exploring Nearest Neighbor Approaches for Image Captioning, arXiv:1505.04467 [[Paper](http://arxiv.org/pdf/1505.04467.pdf)]   
    * Jacob Devlin, Hao Cheng, Hao Fang, Saurabh Gupta, Li Deng, Xiaodong He, Geoffrey Zweig, Margaret Mitchell, Language Models for Image Captioning: The Quirks and What Works, arXiv:1505.01809 [[Paper](http://arxiv.org/pdf/1505.01809.pdf)]   
Difficulty: hard  

####Image processing   

**13. Neural style transfer**   
Input: photo   
Output: photo in artistic style   
Ref: [http://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf](http://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)   
Difficulty: Medium   
**14. Denoising**   
Input: image with noise (Additive white Gaussian, or shot noise)   
Output: image with noise removed.   
Ref: http://webdav.is.mpg.de/pixel/files/neural_denoising/paper.pdf   
Difficulty: Easy   
**15. Super-resolution**   
Input: Low-res image   
Output: High-res version   
Ref: [https://arxiv.org/pdf/1501.00092.pdf](https://arxiv.org/pdf/1501.00092.pdf) (google for more)   
Difficulty: Easy   
**16. Colorizing gray-scale images**   
Input: gray-scale image   
Output: color version   
Ref: https://arxiv.org/abs/1603.08511   
Difficulty: Easy   
**17. Tamper detection**   
Input: image with possible alteration (e.g. object added/removed)   
Output: binary flag saying tampered or not   
Difficulty: Medium   
**18. Depth prediction**   
Input: RGB image   
Output: depth map   
Refs: [http://www.cs.nyu.edu/~deigen/dnl/dnl_iccv15.pdf](http://www.cs.nyu.edu/~deigen/dnl/dnl_iccv15.pdf); http://www.cs.nyu.edu/~deigen/depth   
Data: http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html   
Difficulty: Medium   
**19. DCGAN**   
Generative model of images   
Ref: [https://arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434)   
Difficulty: Medium/Hard   
####Video   
**20. Sign language interpretation**   
Input: Image of hand sign   
Output: word   
Refs:http://vlm1.uta.edu/~athitsos/asl_lexicon/   
[http://csr.bu.edu/asl/asllvd/annotate/index-cvpr4hb08-dataset.html](http://csr.bu.edu/asl/asllvd/annotate/index-cvpr4hb08-dataset.html)   
Difficulty: Medium/hard   
####Random   
**21. Playing Go**   
Input: image of Go board   
Output: prediction of next human move   
Refs:     
[http://www.cs.toronto.edu/~cmaddis/pubs/deepgo.pdf](http://www.cs.toronto.edu/~cmaddis/pubs/deepgo.pdf)   
[https://github.com/facebookresearch/darkforestGo](https://github.com/facebookresearch/darkforestGo)   
[arXiv Page](http://arxiv.org/abs/1511.06410)   
Difficulty: Hard   
**22. Higgs Boson**   
Input: Particle tracks   
Output: Classify particle   
[https://www.kaggle.com/c/higgs-boson](https://www.kaggle.com/c/higgs-boson)   
Difficulty: Hard   
