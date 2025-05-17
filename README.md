
# INN-RSIC (Accepted by Remote Sensing in 2025)
### 📖[**Arxiv**](https://arxiv.org/abs/2405.10518) | 🖼️[**PDF**](/figs/INN-RSIC.pdf)

PyTorch codes for "[Enhancing Perception Quality in Remote Sensing Image Compression via Invertible Neural Network](https://arxiv.org/abs/2405.10518)", **Remote Sensing**, 2025.

- Authors: Junhui Li, and Xingsong Hou* <br>


## Abstract
> Despite the impressive performance of existing image compression algorithms,
they struggle to balance perceptual quality and high image fidelity. To address this issue,
we propose a novel invertible neural network-based remote sensing image compression
(INN-RSIC) method. Our approach captures the compression distortion from an existing
image compression algorithm and encodes it as Gaussian-distributed latent variables
using an INN, ensuring that the distortion in the decoded image remains independent
of the ground truth. By using the inverse mapping of the INN, we input the decoded
image with randomly resampled Gaussian variables, generating enhanced images with
improved perceptual quality. We incorporate channel expansion, Haar transformation, and
invertible blocks into the INN to accurately represent compression distortion. Additionally,
a quantization module (QM) is introduced to mitigate format conversion impact, enhancing
generalization and perceptual quality. Extensive experiments show that INN-RSIC achieves
superior perceptual quality and fidelity compared to existing algorithms. As a lightweight
plug-and-play (PnP) method, the proposed INN-based enhancer can be easily integrated
into existing high-fidelity compression algorithms, enabling flexible and simultaneous
decoding of images with enhanced perceptual quality.

## Network
![image](/figs/Graphical_Abstract.png)
 
## 🧩 Install
```
git clone https://github.com/mlkk518/INN-RSIC.git
```


## 🎁 Dataset
Please download the following remote sensing benchmarks:
Experimental Datasets:
  [DOTA-v1.5](https://captain-whu.github.io/DOTA/dataset.html) | [UC-M](http://weegee.vision.ucmerced.edu/datasets/landuse.html) 

Testing set  (Baidu Netdisk) [DOTA:Download](https://pan.baidu.com/s/1R52rO-gxZH1jG-amwUCO-g) Code：ldc1 | [UC_M:Download](https://pan.baidu.com/s/1KJAy2cPVnj6VfqrlR5XPCg)  Code：pvf3 

## 🧩 Test
[Download Pre-trained ELIC Model](https://pan.baidu.com/s/1OsPSjPp34RHasHi9YM5rHg) (Baidu Netdisk) Code：v72j
[Downloading Pretrained INN_enhancer] (https://pan.baidu.com/s/1Q9ubqMHa9afO1piRAoIvag?pwd=bxhn (Baidu Netdisk) Code：bxhn)
- **Step I.**  Change the roots of ./ELIC/scripts/test.sh to your data and Use the pretrained models of [ELIC] to generate the initial decoded images.

- **Step II.**  Refer to option/test_IRN_compress_S1_0.0004.yml to set the data roots and pretrained models of [INN-RSIC] (./weights/), and run sh ./scripts/tetest_comS1.sh LAMBDA. Here lambda belongs to [0.0004, 0.0008, 0.0016, 0.0032,  0.045] 

```

sh ./ELIC/scripts/test.sh 0.0004 

sh ./scripts/tetest_comS1.sh 0.0004 
```

## 🧩 Train
- **Step**  Using INN to compensate the compression distortion of ELIC.   
```
sh ./scripts/train_comS1.sh 0.0004 0

```

### Results 1: Compared with the baseline
 ![image](/figs/Com_base_UC.png)
 
### Quantitative results  on the images of DOTA.
 ![image](/figs/visua_low_bits_DOTA.png)

### Quantitative results  on the images of UC-M.
 ![image](/figs/visua_high_bits_UC.png)
 
#### More details can be found in our paper!

## Contact
If you have any questions or suggestions, feel free to contact me. 😊  
Email: mlkkljh@stu.xjtu.edu.cn



## Citation
If you find our work helpful in your research, please consider citing it. We appreciate your support！😊


## Acknowledgment: 

This work was supported by:  
- [BasicSR](https://github.com/xinntao/BasicSR)
- [DiffIR](https://github.com/Zj-BinXia/DiffIR)
- [HI-Diff](https://github.com/zhengchen1999/HI-Diff)
- [LDM](https://github.com/CompVis/latent-diffusion)
- [ELiC](https://github.com/VincentChandelier/ELiC-ReImplemetation)
- [LDM-RSIC](https://github.com/mlkk518/LDM-RSIC)



```
@article{li2024enhancing,
  title={Enhancing Perception Quality in Remote Sensing Image Compression via Invertible Neural Network},
  author={Li, Junhui and Hou, Xingsong},
  journal={arXiv preprint arXiv:2405.10518},
  year={2024}
}

@ARTICLE{10980206,
  author={Li, Junhui and Li, Jutao and Hou, Xingsong and Wang, Huake},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Exploring Distortion Prior with Latent Diffusion Models for Remote Sensing Image Compression}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Image coding;Entropy;Distortion;Transformers;Decoding;Adaptation models;Remote sensing;Accuracy;Discrete wavelet transforms;Diffusion models;Image compression;latent diffusion models;remote sensing image;image enhancement},
  doi={10.1109/TGRS.2025.3565259}}

```

