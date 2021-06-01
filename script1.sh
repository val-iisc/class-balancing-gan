#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_dcgan_lsun_rel_weightReg_0.1_no.json" -mpc --eval
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_dcgan_lsun_rel_weightReg_0.01_no.json" -mpc --eval
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_dcgan_lsun_rel_weightReg_no.json" -mpc --eval

#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Table2/proj_sngan_lsun_rel_no.json" -mpc --eval
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Table2/proj_sngan_lsun_rel_no_weightReg_0.01_512_no.json" -mpc --eval


#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_resgan_lsun_rel_no_weightReg_0.01_no.json" -mpc --eval

#Our GAN Eval

#CUDA_VISIBLE_DEVICES=1 python3 main.py  -c "./configs/Unconditional_img_synthesis/no_dcgan_lsun_rel_weightReg_0.1_no.json" -mpc --eval --checkpoint_folder checkpoints/no_dcgan_lsun_rel_weightReg_0.1_no-train-2020_09_04_02_14_38
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -c "./configs/Unconditional_img_synthesis/no_dcgan_lsun_rel_weightReg_0.01_no.json" -mpc --eval --checkpoint_folder checkpoints/no_dcgan_lsun_rel_weightReg_0.01_no-train-2020_09_02_14_12_56
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -c "./configs/Unconditional_img_synthesis/no_dcgan_lsun_rel_weightReg_no.json" -mpc --eval --checkpoint_folder checkpoints/no_dcgan_lsun_rel_weightReg_no-train-2020_09_10_20_17_28


# ACGAN
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Table2/acgan_sngan_lsun_rel_no.json" -mpc --eval
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Table2/acgan_sngan_lsun_rel_no_weightReg_0.1_no.json" -mpc --eval
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Table2/acgan_sngan_lsun_rel_no_weightReg_0.01_no.json" -mpc --eval

#ACGAN Eval
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -c "./configs/Table2/acgan_sngan_lsun_rel_no.json" -mpc --eval --checkpoint_folder checkpoints/acgan_sngan_lsun_rel_no-train-2020_09_09_12_32_07
# CUDA_VISIBLE_DEVICES=0 python3 main.py  -c "./configs/Table2/acgan_sngan_lsun_rel_no_weightReg_0.1_no.json" -mpc --eval --checkpoint_folder checkpoints/acgan_sngan_lsun_rel_no_weightReg_0.1_no-train-2020_09_10_03_40_29
# CUDA_VISIBLE_DEVICES=0 python3 main.py  -c "./configs/Table2/acgan_sngan_lsun_rel_no_weightReg_0.01_no.json" -mpc --eval --checkpoint_folder checkpoints/acgan_sngan_lsun_rel_no_weightReg_0.01_no-train-2020_09_09_07_03_50

# Do an eval for our GAN

#CUDA_VISIBLE_DEVICES=1 python3 main.py  -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_0.01_no.json" -mpc --eval --checkpoint_folder checkpoints/no_dcgan_cifar32_rel_weightReg_0.01_no-train-2020_08_27_00_56_43 --seed 0
#CUDA_VISIBLE_DEVICES=0 python3 main.py  -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_0.1_no.json" -mpc --eval --checkpoint_folder checkpoints/no_dcgan_cifar32_rel_weightReg_0.1_no-train-2020_08_29_00_01_24 --seed 0
#CUDA_VISIBLE_DEVICES=0 python3 main.py  -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_no.json" -mpc --eval --checkpoint_folder checkpoints/no_dcgan_cifar32_rel_weightReg_no-train-2020_08_27_18_59_05 --seed 0

CUDA_VISIBLE_DEVICES=1 python3 main.py -t  -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_no_weightReg_no.json" -mpc --eval
# Resgan Eval LSUN
# CUDA_VISIBLE_DEVICES=0 python3 main.py  -c "./configs/Unconditional_img_synthesis/no_resgan_lsun_rel_no_weightReg_0.01_no.json" -mpc --eval --checkpoint_folder checkpoints/no_resgan_lsun_rel_no_weightReg_0.01_no-train-2020_09_08_13_25_35
# CUDA_VISIBLE_DEVICES=0 python3 main.py  -c "./configs/Unconditional_img_synthesis/no_resgan_lsun_rel_no_weightReg_0.1_no.json" -mpc --eval --checkpoint_folder checkpoints/no_resgan_lsun_rel_no_weightReg_0.1_no-train-2020_09_08_02_59_47
#CUDA_VISIBLE_DEVICES=0 python3 main.py  -c "./configs/Unconditional_img_synthesis/no_resgan_lsun_rel_no_weightReg_no.json" -mpc --eval --checkpoint_folder checkpoints/acgan_sngan_lsun_rel_no_weightReg_0.01_no-train-2020_09_09_07_03_50


# Baseline Eval

#CUDA_VISIBLE_DEVICES=1 python3 main.py  -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_no_weightReg_0.01_no.json" -mpc --eval --checkpoint_folder checkpoints/no_dcgan_cifar32_rel_no_weightReg_0.01_no-train-2020_08_27_13_22_14 --seed 0
#CUDA_VISIBLE_DEVICES=0 python3 main.py  -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_no_weightReg_0.1_no.json" -mpc --eval --checkpoint_folder checkpoints/no_dcgan_cifar32_rel_no_weightReg_0.1_no-train-2020_08_28_00_06_20 --seed 0
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_no_weightReg_no.json" -mpc --eval --checkpoint_folder  checkpoints/no_dcgan_cifar32_rel_no_weightReg_no-train-2020_08_29_02_50_07 --seed 0

#CUDA_VISIBLE_DEVICES=1 python3 main.py -t -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_no.json" -mpc --eval --seed 0


# Ablations
# N = 1e8
#CUDA_VISIBLE_DEVICES=1 python3 main.py -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_no.json" -mpc --eval --seed 0 --checkpoint_folder checkpoints/no_dcgan_cifar32_rel_weightReg_no-train-2020_09_27_01_42_53
# N_0 = 1
#CUDA_VISIBLE_DEVICES=1 python3 main.py -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_no.json" -mpc --eval --seed 0 --checkpoint_folder checkpoints/no_dcgan_cifar32_rel_weightReg_no-train-2020_09_26_22_16_41


# Ablations
# Update Every: 28-09-2020
# 1000
#CUDA_VISIBLE_DEVICES=1 python3 main.py -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_0.1_no.json" -mpc --eval --seed 0 --save_every 1000 --checkpoint_folder checkpoints/no_dcgan_cifar32_rel_weightReg_0.1_no-train-2020_09_28_02_34_47
# 4000
#CUDA_VISIBLE_DEVICES=1 python3 main.py -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_0.1_no.json" -mpc --eval --seed 0 --save_every 4000 --checkpoint_folder checkpoints/no_dcgan_cifar32_rel_weightReg_0.1_no-train-2020_09_28_02_41_09





#CUDA_VISIBLE_DEVICES=1 python3 main.py -t -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_no_cr.json" -mpc --eval --seed 0 

#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_0.01_no.json" -mpc --seed 0 --eval
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_0.1_no.json" -mpc --seed 0 --eval

#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_0.01_finetune.json" -mpc --seed 0 --eval
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_0.1_finetune_.json" -mpc --seed 0 --eval

#CUDA_VISIBLE_DEVICES=0 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_dcgan_lsun_rel_weightReg_0.01_no_few_shot.json" -mpc --eval --seed 0

#CUDA_VISIBLE_DEVICES=0 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar100_rel_weightReg_0.1_no.json" -mpc --seed 0 --eval --update_every 4000
# Imbalance
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Table2/proj_sngan_cifar100_rel_no_weightReg_0.1_512_no.json" -mpc --seed 0 --eval
#Imbalance GAN
#CUDA_VISIBLE_DEVICES=0 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar100_rel_no_weightReg_0.1_no.json" -mpc --seed 0 --eval 
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Table2/acgan_sngan_cifar100_rel_no_weightReg_0.1_512_no.json" -mpc --seed 0 --eval

# Evaluation for CIFAR-100

#CUDA_VISIBLE_DEVICES=1 python3 main.py  -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar100_rel_weightReg_0.1_no.json" -mpc --eval --checkpoint_folder checkpoints/no_dcgan_cifar100_rel_weightReg_0.1_no-train-2020_11_18_16_45_09 --seed 0

# CUDA_VISIBLE_DEVICES=1 python3 main.py   -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar100_rel_no_weightReg_0.1_no.json" -mpc --seed 0 --eval --checkpoint_folder checkpoints/no_dcgan_cifar100_rel_no_weightReg_0.1_no-train-2020_11_18_21_58_57

#CUDA_VISIBLE_DEVICES=0 python3 main.py  -c "./configs/Table2/proj_sngan_cifar100_rel_no_weightReg_0.1_512_no.json" -mpc --seed 0 --eval --checkpoint_folder checkpoints/proj_sngan_cifar100_rel_no_weightReg_0.1_512_no-train-2020_11_13_12_47_49

#CUDA_VISIBLE_DEVICES=0,1 python3 main.py  -c "./configs/Table2/acgan_sngan_cifar100_rel_no_weightReg_0.1_512_no.json" -mpc --seed 0 --eval --checkpoint_folder checkpoints/acgan_sngan_cifar100_rel_no_weightReg_0.1_512_no-train-2020_11_15_16_18_10

# Evaluation for few shot experiments
# LSUN
#CUDA_VISIBLE_DEVICES=0 python3 main.py  -c "./configs/Unconditional_img_synthesis/no_dcgan_lsun_rel_weightReg_0.01_no_few_shot.json" -mpc --eval --seed 0 --checkpoint_folder checkpoints/no_dcgan_lsun_rel_weightReg_0.01_no_few_shot-train-2020_11_15_22_55_01

#CUDA_VISIBLE_DEVICES=0 python3 main.py  -c "./configs/Unconditional_img_synthesis/no_dcgan_lsun_rel_weightReg_0.1_no_few_shot.json" -mpc --eval  --checkpoint_folder checkpoints/no_dcgan_lsun_rel_weightReg_0.1_no_few_shot-train-2020_11_15_02_26_37
#CIFAR - 10
# CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_0.01_finetune.json" -mpc --seed 0 --eval --checkpoint_folder checkpoints/no_dcgan_cifar32_rel_weightReg_0.01_finetune-train-2020_11_12_02_17_44
# CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_0.1_finetune_.json" -mpc --seed 0 --eval -- checkpoint_folder  checkpoints/no_dcgan_cifar32_rel_weightReg_0.1_finetune_-train-2020_11_14_13_08_10

#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_0.01_no_diff_augm.json" -mpc --eval 

#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Table2/proj_sngan_cifar32_rel_no_weightReg_0.01_no_diff.json" -mpc --eval 
# CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Table2/acgan_sngan_cifar32_rel_no_weightReg_0.01_no_diff.json" -mpc --eval 
# CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_no_weightReg_0.01_no_diff.json" -mpc --eval 
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_0.1_no_diff_augm.json" -mpc --eval 

#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Table2/proj_sngan_cifar32_rel_no_weightReg_0.1_no_diff.json" -mpc --eval 
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Table2/acgan_sngan_cifar32_rel_no_weightReg_0.1_no_diff.json" -mpc --eval 
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_no_weightReg_0.1_no_diff.json" -mpc --eval 
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_no_weightReg_no.json" -mpc --eval --checkpoint_folder checkpoints/no_dcgan_cifar32_rel_no_weightReg_no-train-2021_02_04_18_33_32 --current

# Activation Maximization Baseline
# CUDA_VISIBLE_DEVICES=0 python3 main.py  -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_0.01_no.json" -mpc --eval --checkpoint_folder checkpoints/no_dcgan_cifar32_rel_weightReg_0.01_no-train-2021_02_15_12_00_49 &
#CUDA_VISIBLE_DEVICES=1 python3 main.py  -t -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_0.01_no.json" -mpc --eval #--checkpoint_folder checkpoints/no_dcgan_cifar32_rel_weightReg_0.01_no-train-2021_02_15_12_02_26 &
#CUDA_VISIBLE_DEVICES=3 python3 main.py -t -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_0.01_no.json" -mpc --eval #--checkpoint_folder checkpoints/no_dcgan_cifar32_rel_weightReg_0.01_no-train-2021_02_15_12_07_42 &
#CUDA_VISIBLE_DEVICES=3 python3 main.py -c "./configs/Unconditional_img_synthesis/no_dcgan_cifar32_rel_weightReg_0.1_no.json" -mpc --eval --seed 0 --checkpoint_folder checkpoints/no_dcgan_cifar32_rel_weightReg_0.1_no-train-2021_02_19_22_19_14  #
