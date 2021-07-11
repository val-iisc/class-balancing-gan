## Class Balancing GAN with A Classifier In the Loop ([Paper](https://arxiv.org/abs/2106.09402))


This is code release for our UAI 2021 paper Class Balancing GAN with a Classifier in the Loop. 
![approach](https://user-images.githubusercontent.com/15148765/125190714-1f9a3300-e25c-11eb-9933-e13e91c79ea6.jpg)




## 1. Requirements

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



## 2. Dataset (CIFAR10, LSUN)
CIFAR-10 dataset will be downloaded automatically in ```./data``` folder in the project directory. For LSUN dataset download please follow the instructions [here](https://github.com/fyu/lsun) on how to download, then update the config file with the dataset path.


## 3. Pretrained Classifier

One of the requirments of our framework is the availability of pretrained classifier on the data on the classes you want to train the GAN. For all the results we use the [LDAM-DRW](https://github.com/kaidic/LDAM-DRW) repo to obtain the pretrained models. We provide link for downloading the pretrained models of classifier.

Dataset | 0.01 | 0.1 | 1.0 
--- | --- | --- | ---
CIFAR | [link](https://drive.google.com/file/d/18OPwjIpFYYcNJfuLNcnyEY3e_V5UGScf/view?usp=sharing) | [link](https://drive.google.com/file/d/1o-5f0b2Fr7LwxK0lgThZ3yLcZ2VTpZiI/view?usp=sharing) | [link](https://drive.google.com/file/d/1o-5f0b2Fr7LwxK0lgThZ3yLcZ2VTpZiI/view?usp=sharing)
LSUN | [link](https://drive.google.com/file/d/1vvNVQLFFmpv1qxX_28V-sVDdSHwFM58X/view?usp=sharing) | [link](https://drive.google.com/file/d/1OouiaShrUiwn48EtYasRQKmRxrq74rSE/view?usp=sharing) | [link](https://drive.google.com/file/d/1dSTuv2IFEeYshyr0MQCjnGVSkj1_lI1w/view?usp=sharing)

Please download these files before you start to run experiments. Update the path of pretrained models in the ```pretrained_model_path``` field in the configurations in ```./configs``` folder.


## 4. How to run
For each of the imbalance factors (i.e. 0.01, 0.1 and 1) there is seperate configuration file in the config folder.

For CIFAR10 image generation training:

```
python3 main.py  -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_0.01_no.json" -mpc --eval
```

For LSUN image generation training:

```
python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_dcgan_lsun_rel_weightReg_0.1_no.json" -mpc --eval
```
Most experiments were run on an Nvidia 12GB RTX 2080ti gpu.
## 5. References

**PyTorch-StudioGAN** : https://github.com/POSTECH-CVLab/PyTorch-StudioGAN

**LDAM-DRW**: https://github.com/kaidic/LDAM-DRW

We thank them for open sourcing their code which has been immensely helpful.

## 6. Citation
Please email <harshr@iisc.ac.in> in case of any queries. In case you find our work useful please consider citing the following paper:

```
@article{rangwani2021class,
  title={Class Balancing GAN with a Classifier in the Loop},
  author={Rangwani, Harsh and Mopuri, Konda Reddy and Babu, R Venkatesh},
  journal={arXiv preprint arXiv:2106.09402},
  year={2021}
}
```

