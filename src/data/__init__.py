from os.path import join, expanduser, basename, exists
from inspect import currentframe
from types import MethodType

import torch
import torch.utils.data as data
import numpy as np

from utils.data_utils import (default_loader, mat_loader, to_tensor)


class SEDataset(data.Dataset):
    PH = 0  # Placeholder
    def __init__(
        self, 
        root, phase,
        transforms,
        repeats,
        mode
    ):
        super().__init__()
        self.root = expanduser(root)
        if not exists(self.root):
            raise FileNotFoundError
        self.phase = str(phase)
        self.transforms = list(transforms)
        self.transforms += [None]*(3-len(self.transforms))
        self.repeats = int(repeats)
        assert int(mode) in (1, 2, 3)
        self._set_mode(int(mode))
        self.rgb_list, self.hsi_list = self._read_file_paths()
        self.len = len(self.rgb_list)

    def __len__(self):
        return self.len * self.repeats

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        index = index % self.len

        data = self.fetch(index)
        if self.phase == 'train':
            return self.preprocess(*data)
        else:
            return (self.get_name(index), *self.preprocess(*data))

    def _read_file_paths(self):
        raise NotImplementedError

    def _set_mode(self, mode):
        self.mode = mode

        def _fetch_rgb_only(self, index):
            return self.fetch_rgb(self.rgb_list[index]), self.PH

        def _fetch_hsi_only(self, index):
            return self.PH, self.fetch_hsi(self.hsi_list[index])

        def _fetch_rgb_hsi(self, index):
            return self.fetch_rgb(self.rgb_list[index]), self.fetch_hsi(self.hsi_list[index])

        def _proprocess_rgb_only(self, rgb, _):
            if self.transforms[1] is not None:
                rgb = self.transforms[1](rgb)
            return to_tensor(rgb).float(), self.PH

        def _preprocess_hsi_only(self, _, hsi):
            if self.transforms[2] is not None:
                hsi = self.transforms[2](hsi)
            return self.PH, to_tensor(hsi).float()

        def _preprocess_rgb_hsi(self, rgb, hsi):
            if self.transforms[0] is not None:
                # Applied on both
                rgb, hsi = self.transforms[0](rgb, hsi)
            if self.transforms[1] is not None:
                # For RGB images solely
                rgb = self.transforms[1](rgb)
            if self.transforms[2] is not None:
                # For HSI images solely
                hsi = self.transforms[2](hsi)
            return to_tensor(rgb).float(), to_tensor(hsi).float()

        # Patching
        if mode == 1:
            # RGB only mode
            assert (self.transforms[0] is None) and (self.transforms[2] is None)
            self.fetch = MethodType(_fetch_rgb_only, self)
            self.preprocess = MethodType(_proprocess_rgb_only, self)
        elif mode == 2:
            # HSI only mode
            assert (self.transforms[0] is None) and (self.transforms[1] is None)
            self.fetch = MethodType(_fetch_hsi_only, self)
            self.preprocess = MethodType(_preprocess_hsi_only, self)
        else:
            # RGB and HSI mode
            self.fetch = MethodType(_fetch_rgb_hsi, self)         
            self.preprocess = MethodType(_preprocess_rgb_hsi, self)

    @property
    def _rgb_on(self):
        # If mode==1, RGB only; mode==3, RGB and HSI
        return self.mode & 1

    @property
    def _hsi_on(self):
        # If mode==2, HSI only; mode==3, RGB and HSI
        return self.mode & 2
        
    def fetch_hsi(self, hsi_path):
        return default_loader(hsi_path)

    def fetch_rgb(self, rgb_path):
        return default_loader(rgb_path)

    def get_name(self, index):
        return basename(self.rgb_list[index])

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
