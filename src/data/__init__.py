from os.path import join, expanduser, basename
from inspect import currentframe

import torch
import torch.utils.data as data
import numpy as np

from utils.data_utils import (default_loader, mat_loader, to_tensor)


class SEDataset(data.Dataset):
    def __init__(
        self, 
        root, phase,
        transforms,
        repeats
    ):
        super().__init__()
        # Get additional arguments from the outer stack frame
        outer_namespace = {k:v for k, v in currentframe().f_back.f_locals.items() if k not in locals()}
        self._set_attributes({**outer_namespace, **locals()})
        self.rgb_list, self.hsi_list = self._read_file_paths()
        self.len = len(self.rgb_list)

    def __len__(self):
        return self.len * self.repeats

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        index = index % self.len
        
        rgb = self.fetch_rgb(self.rgb_list[index])
        hsi = self.fetch_hsi(self.hsi_list[index])
        rgb, hsi = self.preprocess(rgb, hsi)
        if self.phase == 'train':
            return rgb, hsi
        else:
            return basename(self.rgb_list[index]), rgb, hsi

    def _set_attributes(self, ctx):
        self.root = expanduser(ctx['root'])
        self.phase = ctx['phase']
        self.transforms = ctx['transforms']
        self.repeats = ctx['repeats']

    def _read_file_paths(self):
        raise NotImplementedError
        
    def fetch_hsi(self, hsi_path):
        return default_loader(hsi_path)

    def fetch_rgb(self, rgb_path):
        return default_loader(rgb_path)

    def preprocess(self, rgb, hsi):

        if self.transforms[0] is not None:
            # Applied on both
            rgb, hsi = self.transforms[0](rgb, hsi)
        if self.transforms[1] is not None:
            # For RGB images solely
            rgb = self.transforms[1](rgb)
        if self.transforms[2] is not None:
            # For HSI images solely
            hsi = self.transforms[2](hsi)

        rgb, hsi = to_tensor(rgb).float(), to_tensor(hsi).float()
        return rgb, hsi
