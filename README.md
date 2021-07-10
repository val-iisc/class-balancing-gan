## Class Balancing GAN with A Classifier In the Loop

This is code release for our UAI 2021 paper Class Balancing GAN with a Classifier in the Loop. This is built upon the StudioGAN and LDAM-DRW repository.



## 3. Requirements

- Anaconda
- Python > 3.6
- torch > 1.6.0
- torchvision > 0.7.0
- Pillow < 7
- apex 0.1 (for fused optimiers)
- tensorboard
- h5py
- tqdm

You can install the recommended environment setting as follows:

```
conda env create -f environment.yml -n classbalancinggan
```



## 4. Dataset(CIFAR10, Tiny ImageNet, ImageNet possible)
The folder structure of the datasets is shown below:
```
├── data
   └── ILSVRC2012
       ├── train
           ├── n01443537
     	        ├── image1.png
     	        ├── image2.png
		└── ...
           ├── n01629819
           └── ...
       ├── valid
           └── val_folder
	        ├── val1.png
     	        ├── val2.png
		└── ...
```

### Pretrained Classifier

One of the requirments of our framework is the availability of pretrained classifier on the data on the classes you want to train the GAN. For all the results we use the [LDAM-DRW](https://github.com/kaidic/LDAM-DRW) repo to obtain the pretrained models. We provide link for downloading the pretrained models of classifier.

Dataset | 0.01 | 0.1 | 1.0 
--- | --- | --- | ---
CIFAR | [link](https://drive.google.com/file/d/18OPwjIpFYYcNJfuLNcnyEY3e_V5UGScf/view?usp=sharing) | [link](https://drive.google.com/file/d/1o-5f0b2Fr7LwxK0lgThZ3yLcZ2VTpZiI/view?usp=sharing) | [link](https://drive.google.com/file/d/1o-5f0b2Fr7LwxK0lgThZ3yLcZ2VTpZiI/view?usp=sharing)
LSUN | [link](https://drive.google.com/file/d/1vvNVQLFFmpv1qxX_28V-sVDdSHwFM58X/view?usp=sharing) | [link](https://drive.google.com/file/d/1OouiaShrUiwn48EtYasRQKmRxrq74rSE/view?usp=sharing) | [link](https://drive.google.com/file/d/1dSTuv2IFEeYshyr0MQCjnGVSkj1_lI1w/view?usp=sharing)

Please download these files before you start to run experiments. Update the path of pretrained models in the ```pretrained_model_path``` field in the configurations in ```./configs``` folder.


## 5. How to run
For each of the imbalance factors (i.e. 0.01, 0.1 and 1) there is seperate configuration file in the config folder.

For CIFAR10 image generation training:

```
CUDA_VISIBLE_DEVICES=1 python3 main.py  -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_0.01_no.json" -mpc --eval
```

For LSUN image generation training:

```
CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_dcgan_lsun_rel_weightReg_0.1_no.json" -mpc --eval
```
Most experiments were run on an Nvidia 12GB RTX 2080ti gpu.
## 6. References

**PyTorch-StudioGAN** : https://github.com/POSTECH-CVLab/PyTorch-StudioGAN

**LDAM-DRW**: https://github.com/kaidic/LDAM-DRW

We thank them for open sourcing their code which has been immensely helpful.

## 7. Citation
Please email <harshr@iisc.ac.in> in case of any queries. In case you find our work useful please consider citing the following paper:

```
@article{rangwani2021class,
  title={Class Balancing GAN with a Classifier in the Loop},
  author={Rangwani, Harsh and Mopuri, Konda Reddy and Babu, R Venkatesh},
  journal={arXiv preprint arXiv:2106.09402},
  year={2021}
}
```

