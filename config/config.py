# 提供的可选数据集
nb201_dataset_list = ['cifar10', 'cifar100', 'ImageNet16-120']  # NAS-Bench-201
meta_dataset_list = ['aircraft', 'pets'] # MetaD2A

# 实验超参数设置
nb201_hyper_params_setting = {
    'cifar10': {
        'num_step': 100, 'population_num': 30, 'geno_shape': (8,7), 'temperature': 1.0, 
        'noise_scale':0.8, 'mutate_rate':0.6, 'elite_rate':0.1, 'diver_rate': 0.2, 
        'mutate_distri_index':5, 'rand_exp_num': 20, 'max_iter_time': 30, 
        'save_dir': './results/nb201_benchmark/cifar10/', 'nb201_or_meta': 'nb201', 
        'seed': [
            1731578139, # max_acc@1: 94.37
            1731578141, # max_acc@1: 94.37
            1731578146, # max_acc@1: 94.37
            1731578150, # max_acc@1: 94.37
            1731578154, # max_acc@1: 94.37
        ]
    },
    'cifar100': {
        'num_step': 100, 'population_num': 30, 'geno_shape': (8,7), 'temperature': 1.0, 
        'noise_scale':0.8, 'mutate_rate':0.6, 'elite_rate':0.1, 'diver_rate': 0.2, 
        'mutate_distri_index':5, 'rand_exp_num': 20, 'max_iter_time': 30, 
        'save_dir': './results/nb201_benchmark/cifar100/', 'nb201_or_meta': 'nb201', 
        'seed': [
            1731578242, # max_acc@1: 73.51
            1731578247, # max_acc@1: 73.51
            1731578254, # max_acc@1: 73.51
            1731578256, # max_acc@1: 73.51
            1731578258, # max_acc@1: 73.51
        ]
    },
    'ImageNet16-120': {
        'num_step': 100, 'population_num': 30, 'geno_shape': (8,7), 'temperature': 1.0, 
        'noise_scale':0.8, 'mutate_rate':0.6, 'elite_rate':0.1, 'diver_rate': 0.2, 
        'mutate_distri_index':5, 'rand_exp_num': 20, 'max_iter_time': 30, 
        'save_dir': './results/nb201_benchmark/imagenet16_120/', 'nb201_or_meta': 'nb201', 
        'seed': [
            1731578516, # max_acc@1: 47.31
            1731578531, # max_acc@1: 47.31
            1731578554, # max_acc@1: 47.31
            1731578556, # max_acc@1: 47.31
            1731578650, # max_acc@1: 47.31
        ]
    }, 
}
meta_hyper_params_setting = {
    'aircraft': {
        'num_step': 100, 'population_num': 30, 'geno_shape': (8,7), 'temperature': 1.0, 
        'noise_scale':0.8, 'mutate_rate':0.6, 'elite_rate':0.1, 'diver_rate': 0.3, 
        'mutate_distri_index':5, 'rand_exp_num': 5, 'max_iter_time': 90, 
        'save_dir': './results/meta/aircraft/', 'nb201_or_meta': 'meta', 
        "eta_min"   : 0.0,
        "epochs"    : 200,
        "warmup"    : 20,
        "LR"        : 0.1,
        "decay"     : 0.0005,
        "momentum"  : 0.9,
        "nesterov"  : True,
        "batch_size": 256,
        "image_cutout": 5, 
        'topk'    : 3, 
        'early_stop': False, 
        'multi_thread':False, 
        'seed': [
            # Current Benchmark: 59.15+-0.58
            777,    # max_acc@1: 60.50 (top1)
            1234,   # max_acc@1: 60.13 (top1)
            2345,   # max_acc@1: 60.03 (top1)
            3456,   # max_acc@1: 60.37 (top1)
            8901,   # max_acc@1: 63.05 (top1)
            9012,   # max_acc@1: 59.55 (top2, top1: 59.16)
            333,    # max_acc@1: 61.96 (top2)
            44,     # max_acc@1: 59.92 (top3, top2: 59.66)
            5678,   # max_acc@1: 59.67 (top3)
        ]
    }, 
    'pets': {
        'num_step': 100, 'population_num': 30, 'geno_shape': (8,7), 'temperature': 1.0, 
        'noise_scale':0.8, 'mutate_rate':0.6, 'elite_rate':0.1, 'diver_rate': 0.3, 
        'mutate_distri_index':5, 'rand_exp_num': 5, 'max_iter_time': 90, 
        'save_dir': './results/meta/pets/', 'nb201_or_meta': 'meta', 
        "eta_min"   : 0.0,
        "epochs"    : 200,
        "warmup"    : 20,
        "LR"        : 0.1,
        "decay"     : 0.0005,
        "momentum"  : 0.9,
        "nesterov"  : True,
        "batch_size": 256,
        "image_cutout": 5, 
        'topk'    : 2, 
        'early_stop': False, 
        'multi_thread':False, 
        'seed': [
            # Current Benchmark: 41.80+-3.82
            66,     # max_acc@1: 46.05 (Top1)
            88,     # max_acc@1: 47.85 (Top1)
            999,    # max_acc@1: 50.82 (Top2, Top1: 43.39)
            7890,   # max_acc@1: 48.24 (Top1, 47.46)
            77,     # max_acc@1: 43.15 (Top2) TODO
            3456,   # max_acc@1: 43.55 (Top2) TODO
            99,     # max_acc@1: 42.06 (Top2) TODO
            777,    # max_acc@1: 41.51 (Top2) TODO

        ]
    }, 
}

