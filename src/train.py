#!/usr/bin/env python3
import argparse
import os
import shutil
import random
import ast
from os.path import basename, exists

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import yaml

from core.trainers import EstimatorTrainer, ClassifierTrainer, SolverTrainer, ConditionalTrainer, OnestepTrainer
from utils.misc import OutPathGetter, Logger, register


def read_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    return cfg or {}


def parse_config(cfg_name, cfg):
    # Note that no type check is yet done here!
    # Parse the name of config file
    sp = cfg_name.split('.')[0].split('_')
    if len(sp) >= 2:
        cfg.setdefault('tag', sp[1])
        cfg.setdefault('suffix', '_'.join(sp[2:]))
    
    # Parse metric configs
    if 'metric_configs' in cfg:
        cfg['metric_configs'] = tuple((dict() if c is None else c) for c in cfg['metric_configs'])

    return cfg


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['train', 'val'])
    parser.add_argument('task', choices=['E', 'C', 'S', 'O', 'N'])

    # tensorboard
    # parser.add_argument('--tensorboard_dir', type=str, default=None,
    #                     help="if None is given, the default dir will be './runs/CURRENT_DATETIME_HOSTNAME/' ")

    # Data
    # Common
    group_data = parser.add_argument_group('data')
    group_data.add_argument('-d', '--dataset', type=str, default='NTIRE2020')
    group_data.add_argument('-p', '--crop-size', type=int, default=256, metavar='P', 
                        help='patch size (default: %(default)s)')
    group_data.add_argument('--num-workers', type=int, default=8)
    group_data.add_argument('--repeats', type=int, default=100)
    group_data.add_argument('--mode', type=int, choices=[1, 2, 3], default=2)
    # For NTIRE2020
    group_data.add_argument('--track', type=int, default=1, choices=[1, 2])

    # Optimizer
    group_optim = parser.add_argument_group('optimizer')
    group_optim.add_argument('--optimizer', type=str, default='Adam')
    group_optim.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: %(default)s)')
    group_optim.add_argument('--lr-mode', type=str, default='const')
    group_optim.add_argument('--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: %(default)s)')
    group_optim.add_argument('--step', type=int, default=200)

    # Training related
    group_train = parser.add_argument_group('training related')
    group_train.add_argument('--batch-size', type=int, default=8, metavar='B',
                        help='input batch size for training (default: %(default)s)')
    group_train.add_argument('--num-epochs', type=int, default=1000, metavar='NE',
                        help='number of epochs to train (default: %(default)s)')
    group_train.add_argument('--load-optim', action='store_true')
    group_train.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    group_train.add_argument('--anew', action='store_true',
                        help='clear history and start from epoch 0 with the checkpoint loaded')
    group_train.add_argument('--trace-freq', type=int, default=50)
    group_train.add_argument('--device', type=str, default='cpu')
    group_train.add_argument('--chop', action='store_true')
    group_train.add_argument('--ckps', nargs='*',
                             help="The path to pre-trained weights for the joint model. Not for the resume")

    # Experiment
    group_exp = parser.add_argument_group('experiment related')
    group_exp.add_argument('--exp-dir', default='../exp/')
    group_exp.add_argument('-o', '--out-dir', default='')
    group_exp.add_argument('--tag', type=str, default='')
    group_exp.add_argument('--suffix', type=str, default='')
    group_exp.add_argument('--exp-config', type=str, default='')
    group_exp.add_argument('--save-on', action='store_true')
    group_exp.add_argument('--log-off', action='store_true')
    group_exp.add_argument('--suffix-off', action='store_true')

    # Criterion
    group_critn = parser.add_argument_group('criterion related')
    group_critn.add_argument('--criterion', type=str, default='L1')
    group_critn.add_argument('--ce-weights', type=str, default=(1.0,)*40)

    # Metrics
    group_metric = parser.add_argument_group('metric related')
    group_metric.add_argument('--metrics', type=str, default='RMSE+PSNR+SSIM+MRAE')
    group_metric.add_argument('--metric-configs', type=str, nargs='*', default=("{}", "{}", "{}", "{}"))

    # Model
    group_model = parser.add_argument_group('model')
    group_model.add_argument('--sens-type', type=str, default='C', choices=['C', 'D'])
    group_model.add_argument('--num-resblocks', type=int, default=2)
    group_model.add_argument('--num-feats-in', type=int, default=3)
    group_model.add_argument('--num-feats-out', type=int, default=31)

    args = parser.parse_args()
    cfg = None

    if exists(args.exp_config):
        cfg = read_config(args.exp_config)
        cfg = parse_config(basename(args.exp_config), cfg)
        # Settings from cfg file overwrite those in args
        # Note that the non-default values will not be affected
        parser.set_defaults(**cfg)  # Reset part of the default values
        args = parser.parse_args()  # Parse again

    # Handle cross entropy weights
    if isinstance(args.ce_weights, str):
        args.ce_weights = ast.literal_eval(args.ce_weights)
    args.ce_weights = tuple(args.ce_weights)

    # Handle metric-configs
    if not cfg or 'metric_configs' not in cfg:
        args.metric_configs = tuple(ast.literal_eval(config) for config in args.metric_configs)

    return args


def set_gpc_and_logger(args):
    gpc = OutPathGetter(
            root=os.path.join(args.exp_dir, args.tag), 
            suffix=args.suffix)

    log_dir = '' if args.log_off else gpc.get_dir('log')
    logger = Logger(
        scrn=True,
        log_dir=log_dir,
        phase=args.cmd
    )

    register('GPC', gpc)
    register('LOGGER', logger)

    return gpc, logger
    

def main():
    args = parse_args()
    gpc, logger = set_gpc_and_logger(args)

    if exists(args.exp_config):
        # Make a copy of the config file
        cfg_path = gpc.get_path('root', basename(args.exp_config), suffix=False)
        shutil.copy(args.exp_config, cfg_path)

    # Set random seed
    RNG_SEED = 1
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)

    cudnn.deterministic = True
    cudnn.benchmark = False

    try:
        if args.task == 'E':
            trainer_type = EstimatorTrainer
        elif args.task == 'C':
            trainer_type = ClassifierTrainer
        elif args.task == 'S':
            trainer_type = SolverTrainer
        elif args.task == 'O':
            trainer_type = ConditionalTrainer
        elif args.task == 'N':
            trainer_type = OnestepTrainer
        else:
            trainer_type = None
        trainer = trainer_type(args.dataset, args.optimizer, args)
        trainer.run()
    except BaseException as e:
        import traceback
        # Catch ALL kinds of exceptions
        logger.error(traceback.format_exc())
        exit(1)

if __name__ == '__main__':
    main()