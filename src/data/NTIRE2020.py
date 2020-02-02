import json
from os.path import join, basename

from . import SEDataset
from utils.data_utils import mat_loader

class NTIRE2020Dataset(SEDataset):
    JSON_PATTERN1 = 'very_clean_{}_data.json'
    JSON_PATTERN2 = 'very_real_{}_data.json'
    def __init__(
        self, 
        root, phase='train', 
        transforms=(None, None, None), 
        repeats=1,
        mode=3, 
        track=1
    ):
        self.track = str(track)
        super().__init__(root, phase, transforms, repeats, mode)

    def _read_file_paths(self):
        with open(join(self.root, getattr(self, 'JSON_PATTERN'+self.track).format(self.phase)), 'r') as f:
            pair_list = sorted(json.load(f))
        rgb_list, hsi_list = zip(*pair_list)
        return rgb_list, hsi_list

    def fetch_hsi(self, hsi_path):
        return mat_loader(hsi_path)['cube']