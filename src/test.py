import argparse
import yaml
import json
from core.predictors import Predictor
from models.conditional import Conditional
from models.estimator import Estimator
from models.Onestep import Onestep
# from models.classifier import Classifier
# from models.common import BasicConv
# from models.estimator import Estimator
import torch
import os
from core.factories import model_factory


def read_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    return cfg or {}


def read_file_paths(json_path):
    with open(json_path, 'r') as test:
        rgb_list = json.load(test)

    return rgb_list


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
#
#
# def parse_ckp(ckp_list):
#     assert ckp_list
#     ckps = []
#     for ckp in ckp_list:
#         checkpoint = torch.load(ckp)
#         ckp_dict = checkpoint.get('state_dict', checkpoint)
#         ckps.append(ckp_dict)
#
#     return ckps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('solution', choices=['C', 'G', 'S', 'E', 'O'],
                        help="C: estimator+solver; G: generic; S: classifier+solver; E: Estimator; O: Onestep")
    parser.add_argument('--exp-config', type=str, default='')
    parser.add_argument('--checkpoints', nargs='+',  help="The path of pre-trained weights for the model")
    parser.add_argument('--json_path', default=None)
    parser.add_argument('--save_dir', help='The directory for saving results')
    parser.add_argument('--cuda_off', action='store_true', default=False)
    parser.add_argument('--num_feats_in', default=3)
    parser.add_argument('--num_feats_out', default=31)
    parser.add_argument('--num_resblocks', default=2)

    args = parser.parse_args()

    if os.path.exists(args.exp_config):
        cfg = read_config(args.exp_config)
        # print(cfg)
        cfg = parse_config(os.path.basename(args.exp_config), cfg)
        # Settings from cfg file overwrite those in args
        # Note that the non-default values will not be affected
        parser.set_defaults(**cfg)  # Reset part of the default values
        args = parser.parse_args()  # Parse again

    print(args)

    if args.solution == 'C':
        # model = model_factory('Conditional', args)
        if len(args.checkpoints) == 1:
            model = Conditional(args.num_feats_in, args.num_feats_out, args.num_resblocks)
            checkpoint = torch.load(*args.checkpoints)
            ckp_dict = checkpoint.get('state_dict', checkpoint)
            model.load_state_dict(ckp_dict)
        else:
            model = Conditional(args.num_feats_in, args.num_feats_out, args.num_resblocks, *args.checkpoints)
    elif args.solution == 'G':
        pass
    elif args.solution == 'S':
        pass
    elif args.solution == 'E':
        # model = Estimator(args.num_feats_in, args.num_feats_out)
        if len(args.checkpoints) == 1:
            model = Estimator(args.num_feats_in, args.num_feats_out)
            checkpoint = torch.load(*args.checkpoints)
            ckp_dict = checkpoint.get('state_dict', checkpoint)
            model.load_state_dict(ckp_dict)
    elif args.solution == 'O':
        model = Onestep(args.num_feates_in, args.num_feats_out, args.num_resblocks)
        checkpoint = torch.load(*args.ckeckpoints)
        ckp_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(ckp_dict)
    else:
        raise NotImplementedError


    file_list = read_file_paths(args.json_path)

    predictor = Predictor(model=model, mode='list', cuda_off=args.cuda_off, save_dir=args.save_dir)
    predictor(file_list)



if __name__ == '__main__':
    main()
