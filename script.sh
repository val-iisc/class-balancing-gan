#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Table2/proj_sngan_lsun_rel_no_weightReg_0.1_512_no.json" -mpc --eval
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Table2/proj_sngan_lsun_rel_no_weightReg_0.01_512_no.json" -mpc --eval
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Table2/proj_sngan_lsun_rel_no.json" -mpc --eval


#configs/Table2/proj_sngan_cifar32_rel_no_weightReg_0.01_512_no.json

#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Table2/proj_sngan_cifar32_rel_no.json" -mpc --eval
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Table2/proj_sngan_cifar32_rel_no_weightReg_0.01_512_no.json" -mpc --eval

#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_resgan_lsun_rel_no_weightReg_0.1_no.json" -mpc --eval
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_resgan_lsun_rel_no_weightReg_no.json" -mpc --eval

# Eval Scripts
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -c "./configs/Table2/proj_sngan_lsun_rel_no_weightReg_0.1_512_no.json" -mpc --eval --checkpoint_folder  checkpoints/proj_sngan_lsun_rel_no_weightReg_0.1_512_no-train-2020_09_05_15_47_44
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -c "./configs/Table2/proj_sngan_lsun_rel_no_weightReg_0.01_512_no.json" -mpc --eval --checkpoint_folder checkpoints/proj_sngan_lsun_rel_no_weightReg_0.01_512_no-train-2020_09_06_15_36_12
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -c "./configs/Table2/proj_sngan_lsun_rel_no.json" -mpc --eval --checkpoint_folder checkpoints/proj_sngan_lsun_rel_no-train-2020_09_06_23_34_24


# Eval scripts for CIFAR 10 acgan
#CUDA_VISIBLE_DEVICES=0 python3 main.py  -c "./configs/Table2/acgan_sngan_cifar32_rel_no_weightReg_0.01_no.json" -mpc --eval --checkpoint_folder checkpoints/acgan_sngan_cifar32_rel_no_weightReg_0.01_no-train-2020_08_27_16_14_12 --seed 0
#CUDA_VISIBLE_DEVICES=0 python3 main.py  -c "./configs/Table2/acgan_sngan_cifar32_rel_no_weightReg_0.1_no.json" -mpc --eval --checkpoint_folder checkpoints/acgan_sngan_cifar32_rel_no_weightReg_0.1_no-train-2020_08_28_00_07_47 --seed 0
#CUDA_VISIBLE_DEVICES=0 python3 main.py  -c "./configs/Table2/acgan_sngan_cifar32_rel_no.json" -mpc --eval --checkpoint_folder checkpoints/acgan_sngan_cifar32_rel_no-train-2020_08_28_02_33_52 --seed 0

# Eval scripts for CIFAR 10 sngan

#CUDA_VISIBLE_DEVICES=1 python3 main.py   -c "./configs/Table2/proj_sngan_cifar32_rel_no.json" -mpc --eval --checkpoint_folder checkpoints/proj_sngan_cifar32_rel_no-train-2020_09_07_13_04_54
#CUDA_VISIBLE_DEVICES=1 python3 main.py   -c "./configs/Table2/proj_sngan_cifar32_rel_no_weightReg_0.01_512_no.json" --checkpoint_folder checkpoints/proj_sngan_cifar32_rel_no_weightReg_0.01_512_no-train-2020_09_07_19_08_55 --eval
#CUDA_VISIBLE_DEVICES=1 python3 main.py   -c "./configs/Table2/proj_sngan_cifar32_rel_no_weightReg_0.1_512_no.json" -mpc --eval --checkpoint_folder checkpoints/proj_sngan_cifar32_rel_no_weightReg_0.1_512_no-train-2020_09_07_15_36_32

#CUDA_VISIBLE_DEVICES=1 python3 main.py -l -t -c "./configs/Table2/proj_sngan_inaturalist2019_rel_no.json" -mpc --eval 


#CUDA_VISIBLE_DEVICES=2,3 python3 main.py -l -t -c "./configs/Unconditional_img_synthesis/no_resgan_inaturalist2019_rel_no_weightReg_no.json" -mpc --eval --checkpoint_folder checkpoints/no_resgan_inaturalist2019_rel_no_weightReg_no-train-2021_02_07_15_24_49 --load_current

# configs/Unconditional_img_synthesis/no_resgan_inaturalist2019_rel_no_weightReg_no.json
#CUDA_VISIBLE_DEVICES=2,3 python main.py  -l -c "./configs/Unconditional_img_synthesis/no_resgan_inaturalist2019_rel_weightReg_no.json" -mpc --eval --num_workers 40 --update_every 1 --checkpoint_folder checkpoints/no_resgan_inaturalist2019_rel_weightReg_no-train-2021_02_02_15_03_01 --type4eval_dataset train

#CUDA_VISIBLE_DEVICES=1 python3 main.py -l -c "./configs/Table2/proj_sngan_inaturalist2019_rel_no.json" -mpc --eval --checkpoint_folder checkpoints/proj_sngan_inaturalist2019_rel_no-train-2021_02_11_00_11_06 --type4eval_dataset train
# --checkpoint_folder checkpoints/no_resgan_inaturalist2019_rel_weightReg_no-train-2021_02_02_15_03_01 --type4eval_dataset train


# Best model to reproduce results
# --checkpoint_folder checkpoints/no_resgan_inaturalist2019_rel_weightReg_no-train-2021_02_02_15_03_01 --type4eval_dataset train

# Eval for the baseline model
#CUDA_VISIBLE_DEVICES=0,1 python3 main.py -l -c "./configs/Unconditional_img_synthesis/no_resgan_inaturalist2019_rel_no_weightReg_no.json" -mpc --eval --checkpoint_folder checkpoints/no_resgan_inaturalist2019_rel_no_weightReg_no-train-2021_02_07_15_24_49 --type4eval_dataset train
##--checkpoint_folder checkpoints/no_resgan_inaturalist2019_rel_weightReg_no-train-2020_12_11_01_12_06
# Eval script for CIFAR 10 our model


#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Table2/proj_sngan_cifar32_rel_no.json" -mpc --eval --seed 0
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Table2/proj_sngan_cifar32_rel_no_weightReg_0.01_512_no.json" -mpc --eval --seed 0

#CUDA_VISIBLE_DEVICES=1 python3 main.py  -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_0.01_finetune.json" -mpc --seed 0 --eval --checkpoint_folder checkpoints/no_dcgan_cifar32_rel_weightReg_0.01_finetune-train-2020_11_12_02_17_44
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_0.1_finetune_.json" -mpc --seed 0 --eval --checkpoint_folder  checkpoints/no_dcgan_cifar32_rel_weightReg_0.1_finetune_-train-2020_11_14_13_08_10

#CUDA_VISIBLE_DEVICES=1 python3 main.py  -c "./configs/Table2/proj_sngan_cifar32_rel_no_weightReg_0.01_512_no.json" -mpc --eval --checkpoint_folder checkpoints/proj_sngan_cifar32_rel_no_weightReg_0.01_512_no-train-2020_11_14_02_14_14 --seed 0 --checkpoint_folder checkpoints/acgan_sngan_inaturalist2019_rel_no-train-2021_02_12_16_19_17 --type4eval_dataset train
#CUDA_VISIBLE_DEVICES=1 python3 main.py -l  -c "./configs/Table2/acgan_sngan_inaturalist2019_rel_no.json" -mpc --eval   --checkpoint_folder checkpoints/acgan_sngan_inaturalist2019_rel_no-train-2021_02_12_16_19_17 --type4eval_dataset train


#CUDA_VISIBLE_DEVICES=0,1 python3 main.py  -l -c "./configs/Unconditional_img_synthesis/no_resgan_inaturalist2019_rel_weightReg_no.json" -mpc --eval --num_workers 40 --update_every 1 --checkpoint_folder checkpoints/no_resgan_inaturalist2019_rel_weightReg_no-train-2021_02_12_23_58_59

#CUDA_VISIBLE_DEVICES=1 python3 main.py -t -c "./configs/Table1/proj_biggan_cifar32_hinge_no.json" -mpc --eval
