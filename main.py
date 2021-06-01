# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# main.py


from argparse import ArgumentParser
import json
import os

from make_hdf5 import make_hdf5
from train import train_framework



def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-c', '--config_path', type=str, default='./configs/Table1/proj_biggan_cifar32_hinge_no.json')
    parser.add_argument('--checkpoint_folder', type=str, default=None)
    parser.add_argument('-current', '--load_current', action='store_true', help='choose whether you load current or     est weights')
    parser.add_argument('--log_output_path', type=str, default=None)

    parser.add_argument('--seed', type=int, default=82624, help='seed for generating random number')
    parser.add_argument('--num_workers', type=int, default=16, help='')
    parser.add_argument('-sync_bn', '--synchronized_bn', action='store_true', help='select whether turn on synchronized batchnorm')
    parser.add_argument('-mpc', '--mixed_precision', action='store_true', help='select whether turn on mixed precision training')
    parser.add_argument('-rm_API', '--disable_debugging_API', action='store_true', help='whether disable pytorch autograd debugging mode')
    parser.add_argument('-fz_op', '--fused_optimization', action='store_true', help='using fused optimization for faster training')

    parser.add_argument('--reduce_train_dataset', type=float, default=1.0, help='control the number of train dataset')
    parser.add_argument('-l', '--load_all_data_in_memory', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--eval', action='store_true')
    parser.add_argument('-knn', '--k_nearest_neighbor', action='store_true', help='select whether conduct k-nearest neighbor analysis')
    parser.add_argument('-itp', '--interpolation', action='store_true', help='select whether conduct interpolation analysis')
    parser.add_argument('-le', '--linear_evaluation', action='store_true', help='select whether conduct linear classification on the feature space')
    parser.add_argument('--nrow', type=int, default=10, help='number of rows to plot image canvas')
    parser.add_argument('--ncol', type=int, default=8, help='number of cols to plot image canvas')
    parser.add_argument('--step_linear_eval', type=int, default=10000, help='number of steps for the optimization')

    parser.add_argument('--print_every', type=int, default=100, help='control log interval')
    parser.add_argument('--save_every', type=int, default=2000, help='control evaluation and save interval')
    parser.add_argument('--update_every', type=int, default=2000, help='Update the batch size every')
    parser.add_argument('--type4eval_dataset', type=str, default='test', help='[train/valid/test]')

    

    args = parser.parse_args()

    if args.config_path is not None:
        with open(args.config_path) as f:
            model_config = json.load(f)
        train_config = vars(args)
    else:
        raise NotImplementedError

    dataset = model_config['data_processing']['dataset_name']
    if dataset == 'cifar10':
        assert args.type4eval_dataset == 'train' or args.type4eval_dataset == 'test', "cifar10 does not contain dataset for validation"
    elif dataset == 'imagenet' or dataset == 'tiny_imagenet':
        assert args.type4eval_dataset == 'train' or args.type4eval_dataset == 'valid',\
             "we do not support the evaluation mode using test images in tiny_imagenet/imagenet dataset"

    hdf5_path_train = make_hdf5(mode='train',**model_config['data_processing'], **train_config) if args.load_all_data_in_memory else None

    train_framework(**train_config,
                    **model_config['data_processing'],
                    **model_config['train']['model'],
                    **model_config['train']['optimization'],
                    **model_config['train']['loss_function'],
                    **model_config['train']['initialization'],
                    **model_config['train']['training_and_sampling_setting'],
                    train_config=train_config, model_config=model_config['train'], hdf5_path_train=hdf5_path_train)

if __name__ == '__main__':
    main()
