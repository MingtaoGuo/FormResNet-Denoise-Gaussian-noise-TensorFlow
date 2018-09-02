# FormResNet-Denoise-Gaussian-noise-TensorFlow
Simplely implement the paper 'FormResNet: Formatted Residual Learning for Image Restoration' by TensorFlow
## Introduction
This code just simplely implement the paper [FormResNet: Formatted Residual Learning for Image Restoration](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Jiao_FormResNet_Formatted_Residual_CVPR_2017_paper.pdf), but there are some details of the code are different from the paper.
![](https://github.com/MingtaoGuo/FormResNet-Denoise-Gaussian-noise-TensorFlow/blob/master/IMAGES/method.jpg)
## DataSets
The datasets include 400 gray images, but i have croped them into 40x40 patches. the croped datasets can be downloaded from my [BaiduYun](https://pan.baidu.com/s/1Uiq29K2WLvOyeGlnRu8j_A) 

Examples of TrainingSet

|||||||||
|-|-|-|-|-|-|-|-|
|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/TrainingSet/1_17.jpg)|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/TrainingSet/1_18.jpg)|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/TrainingSet/1_19.jpg)|![](https://github.com/MingtaoGuo/FormResNet-Denoise-Gaussian-noise-TensorFlow/blob/master/TrainingSet/1_20.jpg)|![](https://github.com/MingtaoGuo/FormResNet-Denoise-Gaussian-noise-TensorFlow/blob/master/TrainingSet/1_25.jpg)|![](https://github.com/MingtaoGuo/FormResNet-Denoise-Gaussian-noise-TensorFlow/blob/master/TrainingSet/1_26.jpg)|![](https://github.com/MingtaoGuo/FormResNet-Denoise-Gaussian-noise-TensorFlow/blob/master/TrainingSet/1_27.jpg)|![](https://github.com/MingtaoGuo/FormResNet-Denoise-Gaussian-noise-TensorFlow/blob/master/TrainingSet/1_28.jpg)|
## Python packages
====================
1. python3.5
2. tensorflow1.4.0
3. pillow
4. numpy
5. scipy
6. skimage

====================
## Results of the code
Trained about 1 epoch, noise intensity sigma: 25

|Raw|Noised|FormResNet[1]|DnCNN[2]|
|-|-|-|-|
|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/TestingSet/01.png)|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/IMAGES/noised1.jpg)|![](https://github.com/MingtaoGuo/FormResNet-Denoise-Gaussian-noise-TensorFlow/blob/master/IMAGES/denoised1.jpg)|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/IMAGES/denoised1.jpg)|
|-|-|psnr/ssim 26.12/0.82|psnr/ssim 25.01/0.77|
|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/TestingSet/02.png)|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/IMAGES/noised2.jpg)|![](https://github.com/MingtaoGuo/FormResNet-Denoise-Gaussian-noise-TensorFlow/blob/master/IMAGES/denoised2.jpg)|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/IMAGES/denoised2.jpg)|
|-|-|psnr/ssim 32.01/0.83|psnr/ssim 31.24/0.80|
|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/TestingSet/03.png)|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/IMAGES/noised3.jpg)|![](https://github.com/MingtaoGuo/FormResNet-Denoise-Gaussian-noise-TensorFlow/blob/master/IMAGES/denoised3.jpg)|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/IMAGES/denoised3.jpg)|
|-|-|psnr/ssim 25.55/0.86|psnr/ssim 24.53/0.82|
|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/TestingSet/04.png)|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/IMAGES/noised4.jpg)|![](https://github.com/MingtaoGuo/FormResNet-Denoise-Gaussian-noise-TensorFlow/blob/master/IMAGES/denoised4.jpg)|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/IMAGES/denoised4.jpg)|
|-|-|psnr/ssim 27.66/0.85|psnr/ssim 26.30/0.82|
|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/TestingSet/05.png)|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/IMAGES/noised5.jpg)|![](https://github.com/MingtaoGuo/FormResNet-Denoise-Gaussian-noise-TensorFlow/blob/master/IMAGES/denoised5.jpg)|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/IMAGES/denoised5.jpg)|
|-|-|psnr/ssim 29.392/0.90|psnr/ssim 28.85/0.87|
|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/TestingSet/06.png)|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/IMAGES/noised6.jpg)|![](https://github.com/MingtaoGuo/FormResNet-Denoise-Gaussian-noise-TensorFlow/blob/master/IMAGES/denoised6.jpg)|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/IMAGES/denoised6.jpg)|
|-|-|psnr/ssim 28.34/0.84|psnr/ssim 28.00/0.80|
|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/TestingSet/07.png)|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/IMAGES/noised7.jpg)|![](https://github.com/MingtaoGuo/FormResNet-Denoise-Gaussian-noise-TensorFlow/blob/master/IMAGES/denoised7.jpg)|![](https://github.com/MingtaoGuo/DnCNN-TensorFlow/blob/master/IMAGES/denoised7.jpg)|
|-|-|psnr/ssim 22.00/0.83|psnr/ssim 23.83/0.79|

[1] Jiao J, Tu W C, He S, et al. FormResNet: Formatted Residual Learning for Image Restoration[C]// Computer Vision and Pattern Recognition Workshops. IEEE, 2017:1034-1042.

[2] Zhang K, Zuo W, Chen Y, et al. Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising[J]. IEEE Transactions on Image Processing A Publication of the IEEE Signal Processing Society, 2017, 26(7):3142-3155.

