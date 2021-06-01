exp_name=$1
gen_name=$2
# Regulatized gans
#CUDA_VISIBLE_DEVICES=1 python3 cifar10_degan.py --dataroot ../../../../lsun_dataset/ --imageSize 64 --cuda  --manualSeed 108 --niter 500 --batchSize 512 --outf ../../../../models/lsun_imb_0.01_dcgan_dl_5_${exp_name} --pretrained_model ../train_teacher/LDAM-DRW/checkpoint/lsun_resnet32_Focal_DRW_exp_0.01_0/ckpt.best.pth.tar --tbp ../../../../tensorboard_paths/dcgan_exp_0.01_dl_5_${exp_name} --dl 10 --imb_factor 0.01 --loss_type RASGAN --dataset lsun


CUDA_VISIBLE_DEVICES=1 python3 cifar10_degan.py --dataroot ../../../../lsun_dataset/ --imageSize 64  --cuda  --manualSeed 108 --niter 500 --batchSize  512 --outf ../../../../models/lsun_imb_0.1_dcgan_dl_5_${exp_name} --pretrained_model ../train_teacher/LDAM-DRW/checkpoint/lsun_resnet32_Focal_DRW_exp_0.1_0/ckpt.best.pth.tar --tbp ../../../../tensorboard_paths/dcgan_exp_0.1_dl_5_${exp_name} --dl 7.5 --imb_factor 0.1 --loss_type RASGAN --dataset lsun

#CUDA_VISIBLE_DEVICES=1 python3 cifar10_degan.py --dataroot ../../../../lsun_dataset/ --imageSize 64  --cuda  --manualSeed 108 --niter 500 --batchSize 512 --outf ../../../../models/lsun_imb_1.0_dcgan_dl_5_${exp_name} --pretrained_model ../train_teacher/LDAM-DRW/checkpoint/lsun_resnet32_Focal_DRW_exp_1.0_0/ckpt.best.pth.tar --tbp ../../../../tensorboard_paths/dcgan_exp_1.0_dl_5_${exp_name} --dl 5 --imb_factor 1.0 --loss_type RASGAN --dataset lsun

# # # Normal Gans
# CUDA_VISIBLE_DEVICES=1 python3 cifar10_degan.py --dataroot ../../../../lsun_dataset/ --imageSize 64  --cuda --manualSeed 108 --niter 500 --batchSize 512 --outf ../../../../models/lsun_imb_0.01_dcgan_${exp_name} --pretrained_model ../train_teacher/LDAM-DRW/checkpoint/lsun_resnet32_Focal_DRW_exp_0.01_0/ckpt.best.pth.tar --tbp ../../../../tensorboard_paths/dcgan_exp_0.01_${exp_name} --imb_factor 0.01 --loss_type RASGAN --dataset lsun


# CUDA_VISIBLE_DEVICES=1 python3 cifar10_degan.py --dataroot ../../../../lsun_dataset/ --imageSize 64  --cuda --manualSeed 108 --niter 500 --batchSize 512 --outf ../../../../models/lsun_imb_0.1_dcgan_${exp_name} --pretrained_model ../train_teacher/LDAM-DRW/checkpoint/lsun_resnet32_Focal_DRW_exp_0.1_0/ckpt.best.pth.tar --tbp ../../../../tensorboard_paths/dcgan_exp_0.1_${exp_name} --imb_factor 0.1 --loss_type RASGAN --dataset lsun

# CUDA_VISIBLE_DEVICES=1 python3 cifar10_degan.py --dataroot ../../../../lsun_dataset/ --imageSize 64  --cuda --manualSeed 108 --niter 500 --batchSize 512 --outf ../../../../models/lsun_imb_1.0_dcgan_${exp_name} --pretrained_model ../train_teacher/LDAM-DRW/checkpoint/lsun_resnet32_Focal_DRW_exp_1.0_0/ckpt.best.pth.tar --tbp ../../../../tensorboard_paths/dcgan_exp_1.0_${exp_name} --imb_factor 1.0 --loss_type RASGAN --dataset lsun


#CUDA_VISIBLE_DEVICES=1 ./generate_script_lsun.sh 490 499 ../../../../models/lsun_imb_0.01_dcgan_dl_5_${exp_name}/ ../../../../generated_cifar/lsun_imb_0.01_dcgan_dl_5_${exp_name}_${gen_name}/ 50
CUDA_VISIBLE_DEVICES=1 ./generate_script_lsun.sh 490 499 ../../../../models/lsun_imb_0.1_dcgan_dl_5_${exp_name}/ ../../../../generated_cifar/lsun_imb_0.1_dcgan_dl_5_${exp_name}_${gen_name}/ 50
#CUDA_VISIBLE_DEVICES=1 ./generate_script_lsun.sh 490 499 ../../../../models/lsun_imb_1.0_dcgan_dl_5_${exp_name}/ ../../../../generated_cifar/lsun_imb_1.0_dcgan_dl_5_${exp_name}_${gen_name}/ 50

# CUDA_VISIBLE_DEVICES=1 ./generate_script_lsun.sh 490 499  ../../../../models/lsun_imb_0.01_dcgan_${exp_name}/ ../../../../generated_cifar/lsun_imb_0.01_dcgan_${exp_name}_${gen_name}/ 50
# CUDA_VISIBLE_DEVICES=1 ./generate_script_lsun.sh 490 499  ../../../../models/lsun_imb_0.1_dcgan_${exp_name}/ ../../../../generated_cifar/lsun_imb_0.1_dcgan_${exp_name}_${gen_name}/ 50
# CUDA_VISIBLE_DEVICES=1 ./generate_script_lsun.sh 490 499  ../../../../models/lsun_imb_1.0_dcgan_${exp_name}/ ../../../../generated_cifar/lsun_imb_1.0_dcgan_${exp_name}_${gen_name}/ 50


# python3 pytorch-fid/fid_score.py  ../../../../generated_cifar/lsun_imb_0.01_dcgan_dl_5_${exp_name}_${gen_name}/generated_dataset/  ./pytorch-fid/fid_stats_cifar10_train.npz -c 1
# python3 pytorch-fid/fid_score.py  ../../../../generated_cifar/lsun_imb_0.1_dcgan_dl_5_${exp_name}_${gen_name}/generated_dataset/  ./pytorch-fid/fid_stats_cifar10_train.npz -c 1
# python3 pytorch-fid/fid_score.py  ../../../../generated_cifar/lsun_imb_1.0_dcgan_dl_5_${exp_name}_${gen_name}/generated_dataset/  ./pytorch-fid/fid_stats_cifar10_train.npz -c 1

# python3 pytorch-fid/fid_score.py  ../../../../generated_cifar/lsun_imb_0.01_dcgan_${exp_name}_${gen_name}/generated_dataset/  ./pytorch-fid/fid_stats_cifar10_train.npz -c 1
# python3 pytorch-fid/fid_score.py  ../../../../generated_cifar/lsun_imb_0.1_dcgan_${exp_name}_${gen_name}/generated_dataset/  ./pytorch-fid/fid_stats_cifar10_train.npz -c 1
# python3 pytorch-fid/fid_score.py  ../../../../generated_cifar/lsun_imb_1.0_dcgan_${exp_name}_${gen_name}/generated_dataset/  ./pytorch-fid/fid_stats_cifar10_train.npz -c 1


#python3 calculate_dataset_stats.py ../../../../generated_cifar/lsun_imb_0.01_dcgan_dl_5_${exp_name}_${gen_name}/generated_dataset/ lsun_imb_0.01_dcgan_dl_5_${exp_name}_${gen_name}

python3 calculate_dataset_stats.py ../../../../generated_cifar/lsun_imb_0.1_dcgan_dl_5_${exp_name}_${gen_name}/generated_dataset/ lsun_imb_0.1_dcgan_dl_5_${exp_name}_${gen_name}
#python3 calculate_dataset_stats.py ../../../../generated_cifar/lsun_imb_1.0_dcgan_dl_5_${exp_name}_${gen_name}/generated_dataset/ lsun_imb_1.0_dcgan_dl_5_${exp_name}_${gen_name}

# python3 calculate_dataset_stats.py ../../../../generated_cifar/lsun_imb_0.01_dcgan_${exp_name}_${gen_name}/generated_dataset/ lsun_imb_0.01_dcgan_${exp_name}_${gen_name}

# python3 calculate_dataset_stats.py ../../../../generated_cifar/lsun_imb_0.1_dcgan_${exp_name}_${gen_name}/generated_dataset/ lsun_imb_0.1_dcgan_${exp_name}_${gen_name}
# python3 calculate_dataset_stats.py ../../../../generated_cifar/lsun_imb_1.0_dcgan_${exp_name}_${gen_name}/generated_dataset/ lsun_imb_1.0_dcgan_${exp_name}_${gen_name}

