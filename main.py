import warnings
import argparse
warnings.filterwarnings('ignore')
from experiments import exp_with_fixed_seed_in_nb201, exp_with_rand_seed_in_nb201, exp_with_fixed_seed_in_meta_predictor, exp_with_rand_seed_in_meta_predictor

def main(args):
    """ Runnning main programme by the following commands in Windows Command Prompt:

    cd /d D:/OneDrive/ZHOU_BINGYE/2024-2025学年上/NAS/EvoDiff/DiffEvo-NAS/EvoDiff/NAS-Bench-201_SearchSpace
    conda activate base
    cls

    To reproduce the results:
    python ./main.py --dataset cifar10
    python ./main.py --dataset cifar100
    python ./main.py --dataset imagenet
    python ./main.py --dataset aircraft
    python ./main.py --dataset pets

    To run the random experiments:
    python ./main.py --exp_type random --dataset cifar10
    python ./main.py --exp_type random --dataset cifar100
    python ./main.py --exp_type random --dataset imagenet
    python ./main.py --exp_type random --dataset aircraft
    python ./main.py --exp_type random --dataset pets

    nvidia-smi -l 3
    """
    dataset_name = {'cifar10': 'cifar10', 'cifar100': 'cifar100', 'imagenet': 'ImageNet16-120', 'aircraft': 'aircraft', 'pets': 'pets'}
    assert args.dataset.lower() in list(dataset_name.keys()), f'ERROR: invalid dataset {args.dataset}'
    assert args.exp_type.lower() in ['reproduce', 'random'], f'ERROR: invalid exp_type {args.exp_type}'

    if args.exp_type.lower() == 'reproduce':
        if args.dataset.lower() in ['cifar10', 'cifar100', 'imagenet']:
            exp_with_fixed_seed_in_nb201(dataset=dataset_name[args.dataset.lower()])
        else:
            exp_with_fixed_seed_in_meta_predictor(dataset=dataset_name[args.dataset.lower()])
    else:
        if args.dataset.lower() in ['cifar10', 'cifar100', 'imagenet']:
            exp_with_rand_seed_in_nb201(dataset=dataset_name[args.dataset.lower()])
        else:
            exp_with_rand_seed_in_meta_predictor(dataset=dataset_name[args.dataset.lower()])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EvoDiff-NAS')
    parser.add_argument('--exp_type', type=str, default='reproduce', help='reproduce, random')
    parser.add_argument('--dataset', type=str, help='cifar10, cifar100, imagenet, aircraft, pets')
    args = parser.parse_args()
    main(args)
    # import torch
    # x=torch.load('./rand_result.pth')
    # print(x)
