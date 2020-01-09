from functools import wraps
from inspect import isfunction, isgeneratorfunction, getmembers
from collections.abc import Iterable
from itertools import chain
from importlib import import_module

import torch
import torch.nn as nn
import torch.utils.data as data

import constants
import utils.metrics as metrics
from utils.misc import R
from data.augmentation import *


class _Desc:
    def __init__(self, key):
        self.key = key
    def __get__(self, instance, owner):
        return tuple(getattr(instance[_],self.key) for _ in range(len(instance)))
    def __set__(self, instance, values):
        if not (isinstance(values, Iterable) and len(values)==len(instance)):
            raise TypeError("incorrect type or number of values")
        for i, v in zip(range(len(instance)), values):
            setattr(instance[i], self.key, v)


def _func_deco(func_name):
    def _wrapper(self, *args):
        # TODO: Add key argument support
        # XXX: This does not look any good
        try:
            # Dispatch type 1
            ret = tuple(getattr(ins, func_name)(*args) for ins in self)
        except Exception:
            # Dispatch type 2
            if len(args) > 1 or (len(args[0]) != len(self)): raise
            ret = tuple(getattr(i, func_name)(a) for i, a in zip(self, args[0]))
        return ret
    return _wrapper


def _generator_deco(func_name):
    def _wrapper(self, *args, **kwargs):
        for ins in self:
            yield from getattr(ins, func_name)(*args, **kwargs)
    return _wrapper


# Duck typing
class Duck(tuple):
    __ducktype__ = object
    def __new__(cls, *args):
        if any(not isinstance(a, cls.__ducktype__) for a in args):
            raise TypeError("please check the input type")
        return tuple.__new__(cls, args)


class DuckMeta(type):
    def __new__(cls, name, bases, attrs):
        assert len(bases) == 1
        for k, v in getmembers(bases[0]):
            if k.startswith('__'):
                continue
            if isgeneratorfunction(v):
                attrs[k] = _generator_deco(k)
            elif isfunction(v):
                attrs[k] = _func_deco(k)
            else:
                attrs[k] = _Desc(k)
        attrs['__ducktype__'] = bases[0]
        return super().__new__(cls, name, (Duck,), attrs)


class DuckModel(nn.Module, metaclass=DuckMeta):
    pass


class DuckOptimizer(torch.optim.Optimizer, metaclass=DuckMeta):
    @property
    def param_groups(self):
        return list(chain.from_iterable(ins.param_groups for ins in self))


class DuckCriterion(nn.Module, metaclass=DuckMeta):
    pass


class DuckDataset(data.Dataset, metaclass=DuckMeta):
    pass


def _import_module(pkg: str, mod: str, rel=False):
    if not rel:
        # Use absolute import
        return import_module('.'.join([pkg, mod]), package=None)
    else:
        return import_module('.'+mod, package=pkg)


def single_model_factory(model_name, C):
    import utils.utils
    module = _import_module('models', model_name.strip())
    model = getattr(module, module.__ENTRANCE__)     # Find the entrance
    if model.__init__.__code__.co_argcount == (1+2):
        return model(C.num_feats_in, C.num_feats_out)
    elif model.__init__.__code__.co_argcount == (1+3):
        # Give an extra num_resblocks
        return model(C.num_feats_in, C.num_feats_out, C.num_resblocks)
    else:
        raise NotImplementedError('cannot determine arguments for {}'.format(model_name.strip()))


def single_optim_factory(optim_name, params, C):
    optim_name = optim_name.strip()
    name = optim_name.upper()
    if name == 'ADAM':
        return torch.optim.Adam(
            params, 
            betas=(0.9, 0.999),
            lr=C.lr,
            weight_decay=C.weight_decay
        )
    else:
        raise NotImplementedError("{} is not a supported optimizer type".format(optim_name))


def single_critn_factory(critn_name, C):
    import losses
    critn_name = critn_name.strip()
    try:
        criterion, params = {
            'L1': (nn.L1Loss, ()),
            'MSE': (nn.MSELoss, ()),
            'CE': (nn.CrossEntropyLoss, (torch.Tensor(C.ce_weights),)),
            'NLL': (nn.NLLLoss, (torch.Tensor(C.ce_weights),)),
            'BCE': (nn.BCELoss, ()),
            'SMOOTH': (losses.SmoothLoss, ()),
        }[critn_name.upper()]
        return criterion(*params)
    except KeyError:
        raise NotImplementedError("{} is not a supported criterion type".format(critn_name))


def _get_basic_configs(ds_name, C):
    if ds_name == 'BSDS500':
        return dict(
            root = constants.IMDB_BSDS500,
            n_seg_cats = C.max_num_cats
        )
    elif ds_name == 'DFC2019':
        return dict(
            root = constants.IMDB_DFC2019,
            json_dir = constants.IMDB_DFC2019_JSON
        )
    elif ds_name == 'OSCD':
        return dict(
            root = constants.IMDB_OSCD
        )
    elif ds_name.startswith('AC'):
        return dict(
            root = constants.IMDB_AIRCHANGE
        )
    elif ds_name.startswith('Lebedev'):
        return dict(
            root = constants.IMDB_LEBEDEV
        )
    else:
        return dict()


def single_train_ds_factory(ds_name, C):
    ds_name = ds_name.strip()
    module = _import_module('data', ds_name)
    dataset = getattr(module, ds_name+'Dataset')
    if C.task == 'seg':
        configs = dict(
            phase='train', 
            transforms=(Compose(Scale([1.0, 2.0]), Crop(C.crop_size), Flip()), None, None),
            repeats=C.repeats
        )
    else:
        configs = dict(
            phase='train', 
            transforms=(Compose(Crop(C.crop_size), Flip()), None, None),
            repeats=C.repeats
        )

    # Update some common configurations
    configs.update(_get_basic_configs(ds_name, C))

    # Set phase-specific ones
    if ds_name == 'BSDS500':
        configs.update(
            dict(
                sub_id = C.sub_id
            )
        )
    elif ds_name == 'Lebedev':
        _tf = (
            Choose(
                HorizontalFlip(True), VerticalFlip(True), 
                Rotate('90'), Rotate('180'), Rotate('270'),
                Scale([1.0, 3.0]),
                Shift()
            )
        )
        configs.update(
            dict(
                transforms=(Compose(_tf, Crop(C.crop_size)), None, None),
                subsets=('real',)
            )
        )
    else:
        pass

    dataset_obj = dataset(**configs)

    return data.DataLoader(
        dataset_obj,
        batch_size=C.batch_size,
        shuffle=True,
        num_workers=C.num_workers,
        pin_memory=not (C.device == 'cpu'), drop_last=True
    )


def single_val_ds_factory(ds_name, C):
    ds_name = ds_name.strip()
    module = _import_module('data', ds_name)
    dataset = getattr(module, ds_name+'Dataset')
    configs = dict(
        phase='val', 
        transforms=(None, None, None)
    )

    # Update some common configurations
    configs.update(_get_basic_configs(ds_name, C))

    # Set phase-specific ones
    if ds_name == 'BSDS500':
        configs.update(
            dict(
                sub_id = 0  # Fix sub_id during val phase
            )
        )
    elif ds_name == 'Lebedev':
        configs.update(
            dict(
                subsets=('real',)
            )
        )
    else:
        pass

    dataset_obj = dataset(**configs)  

    # Create eval set
    return data.DataLoader(
        dataset_obj,
        batch_size=1,
        shuffle=False,
        num_workers=C.num_workers,
        pin_memory=False, drop_last=False
    )


def _parse_input_names(name_str):
    return name_str.split('+')


def model_factory(model_names, C):
    name_list = _parse_input_names(model_names)
    if len(name_list) > 1:
        return DuckModel(*(single_model_factory(name, C) for name in name_list))
    else:
        return single_model_factory(model_names, C)


def optim_factory(optim_names, models, C):
    name_list = _parse_input_names(optim_names)
    num_models = len(models) if isinstance(models, DuckModel) else 1
    if len(name_list) != num_models:
        raise ValueError("the number of optimizers does not match the number of models")
    
    if num_models > 1:
        optims = []
        for name, model in zip(name_list, models):
            param_groups = [{'params': module.parameters(), 'name': module_name} for module_name, module in model.named_children()]
            optims.append(single_optim_factory(name, param_groups, C))
        return DuckOptimizer(*optims)
    else:
        return single_optim_factory(
            optim_names, 
            [{'params': module.parameters(), 'name': module_name} for module_name, module in models.named_children()], 
            C
        )


def critn_factory(critn_names, C):
    name_list = _parse_input_names(critn_names)
    if len(name_list) > 1:
        return DuckCriterion(*(single_critn_factory(name, C) for name in name_list))
    else:
        return single_critn_factory(critn_names, C)


def data_factory(dataset_names, phase, C):
    name_list = _parse_input_names(dataset_names)
    if phase not in ('train', 'val'):
        raise ValueError("phase should be either 'train' or 'val'")
    fact = globals()['single_'+phase+'_ds_factory']
    if len(name_list) > 1:
        return DuckDataset(*(fact(name, C) for name in name_list))
    else:
        return fact(dataset_names, C)


def metric_factory(metric_names, C):
    from utils import metrics
    name_list = _parse_input_names(metric_names)
    return [getattr(metrics, name.strip())() for name in name_list]


if __name__ == '__main__':
    fake_model = DuckModel(nn.Conv2d(3,3,3), nn.Conv2d(3,3,3))
    fake_model.load_state_dict(fake_model.state_dict())
    optim = torch.optim.Adam(fake_model.parameters(), 
                            betas=(0.9,0.999), 
                            lr=1E-4, 
                            weight_decay=1e-4
                            )
    fake_optim = DuckOptimizer(optim, optim)

    fake_criterion = DuckCriterion(nn.L1Loss(), nn.CrossEntropyLoss())

    from data.BSDS500 import BSDS500Dataset
    fake_dataset = DuckDataset(BSDS500Dataset('~/Datasets/BSR/'), BSDS500Dataset('~/Datasets/BSR/'))
    loader = torch.utils.data.DataLoader(fake_dataset)

    for param_group in fake_optim.param_groups:
        param_group['lr'] = 0.1
    
    fake_criterion.to('cpu')
    print(fake_model._version)
    fake_model._version = (0.5, 0.5)
    print(fake_model._version)
    fake_model._version = {0.5, 0.9}
