import torch
import torchvision
import os
from torchvision import transforms
from utils.metrics import AverageMeter
from utils.misc import Logger
import time
import glob
from skimage import io
import numpy as np
from tqdm import tqdm
from functools import partial
from utils.data_utils import to_tensor, to_array, normalize
import hdf5storage as hdf5
"""
输入：
可以是dataloader,可以是文件夹， 可以是装有文件路径的列表， 可以是单张文件的路径，可以是张量

"""

class Predictor:
    modes = ['dataloader', 'folder', 'list', 'file', 'data']
    def __init__(self, model=None, mode='folder', save_dir=None, scrn=True, log_dir=None, cuda_off=False):

        self.save_dir = save_dir
        self.output = None

        if not cuda_off and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        assert  model is not None, "The model must be assigned"
        self.model = self._model_init(model)


        if mode not in Predictor.modes:
            raise NotImplementedError

        self.logger = Logger(scrn=scrn, log_dir=log_dir, phase='predict')

        if mode == 'dataloader':
            self._predict = partial(self._predict_dataloader, dataloader=None, save_dir=save_dir)
        elif mode == 'folder':
            # self.suffix = ['.jpg', '.png', '.bmp', '.gif', '.npy']  # 支持的图像格式
            self._predict = partial(self._predict_folder, save_dir=save_dir)
        elif mode == 'list':
            self._predict = partial(self._predict_list, save_dir=save_dir)
        elif mode == 'file':
            self._predict = partial(self._predict_file, save_dir=save_dir)
        elif mode == 'data':
            self._predict = partial(self._predict_data, save_dir=save_dir)
        else:
            raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self._predict(*args, **kwargs)


    def _model_init(self, model):
        model.to(self.device)
        model.eval()
        return model

    def _load_data(self, path):
        return io.imread(path)

    def _to_tensor(self, arr):
        return to_tensor(arr)

    def _to_array(self, tensor):
        return  to_array(tensor)

    def _normalize(self, tensor):
        return normalize(tensor)

    def _np2tensor(self, arr):
        nor_tensor = self._normalize(self._to_tensor(arr))
        assert isinstance(nor_tensor, torch.Tensor)
        return nor_tensor

    def _save_data_NTIRE2020(self, data, path):
        s_dir = os.path.dirname(path)
        if not os.path.exists(s_dir):
            os.mkdir(s_dir)
        path = path.replace('_clean.png', '.mat').replace('_RealWorld.png', '.mat')
        if isinstance(data, torch.Tensor):
            data = self._to_array(data).squeeze()

        content = {}
        content['cube'] = data
        content['bands'] = np.array([[400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520,
        530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650,
        660, 670, 680, 690, 700]])
        # content['norm_factor'] =
        hdf5.write(data=content, filename=path, store_python_metadata=True, matlab_compatible=True)


    def _save_data(self, data, path):
        s_dir = os.path.dirname(path)
        if not os.path.exists(s_dir):
            os.mkdir(s_dir)

        torchvision.utils.save_image(data, path)

    def predict_base(self, model, data, path=None):
        start = time.time()
        with torch.no_grad():
            output = model(data)
        torch.cuda.synchronize()
        su_time = time.time() - start
        if path:
            self._save_data_NTIRE2020(output, path)

        self.output = output
        return output, su_time


    def _predict_dataloader(self, dataloader, save_dir=None):
        assert dataloader is not None, \
            "In 'dataloader' mode the input must be a valid dataloader!"
        consume_time = AverageMeter()
        pb = tqdm(dataloader)
        for idx, (name, data) in enumerate(pb):
            assert isinstance(data, torch.Tensor) and data.dim() == 4,\
            "input data must be 4-dimention tensor"
            data = data.to(self.device) # 4-d tensor
            save_path = os.path.join(save_dir, name) if save_dir else None
            _, su_time = self.predict_base(self.model, data, path=save_path)
            consume_time.update(su_time, n=1)

            # logger
            description = ("[{}/{}] speed: {time.val:.4f}s({time.avg:.4f}s)".
                           format(idx+1, len(dataloader.dataset), time=consume_time))
            pb.set_description(description)
            self.logger.dump(description)


    def _predict_folder(self, folder, save_dir=None):
        assert folder is not None and os.path.isdir(folder),\
        "In 'folder' mode the input must be a valid path of a folder!"
        consume_time = AverageMeter()
        file_list = glob.glob(os.path.join(folder, '*'))

        assert not len(file_list) == 0, "The input folder is empty"

        pb = tqdm(file_list)    # processbar

        for idx, file in enumerate(pb):
            img = self._load_data(file)
            name = os.path.basename(file)
            img = self._np2tensor(img).unsqueeze(0).to(self.device)
            save_path = os.path.join(save_dir, name) if save_dir else None
            _, su_time = self.predict_base(model=self.model, data=img, path=save_path)
            consume_time.update(su_time)

            # logger
            description = ("[{}/{}] speed: {time.val:.4f}s({time.avg:.4f}s)".
                           format(idx + 1, len(file_list), time=consume_time))
            pb.set_description(description)
            self.logger.dump(description)


    def _predict_list(self, file_list, save_dir=None):
        assert isinstance(file_list, list),\
        "In 'list' mode the input must be a valid file_path list!"
        consume_time = AverageMeter()

        assert not len(file_list) == 0, "The input file list is empty!"

        pb = tqdm(file_list)    # processbar

        for idx, path in enumerate(pb):
            data = self._load_data(path)
            name = os.path.basename(path)
            data = self._np2tensor(data).unsqueeze(0).to(self.device)
            path = os.path.join(save_dir, name) if save_dir else None
            _, su_time = self.predict_base(model=self.model, data=data, path=path)
            consume_time.update(su_time, n=1)

            # logger
            description = ("[{}/{}] speed: {time.val:.4f}s({time.avg:.4f}s)".
                           format(idx + 1, len(file_list), time=consume_time))
            pb.set_description(description)
            self.logger.dump(description)


    def _predict_file(self, file_path, save_dir=None):
        assert isinstance(file_path, str) and os.path.isfile(file_path), \
        "In 'file' mode the input must a valid path of a file!"

        consume_time = AverageMeter()
        data = self._load_data(file_path)
        name = os.path.basename(file_path)
        data = self._np2tensor(data).unsqueeze(0).to(self.device)
        path = os.path.join(save_dir, name) if save_dir else None

        _, su_time = self.predict_base(model=self.model, data=data, path=path)
        consume_time.update(su_time)

        # logger
        description = ("file: {}  speed: {time.val:.4f}s".
                       format(name, time=consume_time))

        self.logger.show(description)


    def _predict_data(self, data):
        """
        :return: tensor
        """

        assert isinstance(data, torch.Tensor) and data.dim() == 4, \
        "In 'data' mode the input must be a 4-d tensor"

        consume_time = AverageMeter()
        output, su_time = self.predict_base(model=self.model, data=data)

        consume_time.update(su_time)

        # logger
        description = ("speed: {time.val:.4f}s".format(time=consume_time))

        self.logger.dump(description)

        return output




if __name__ == '__main__':
    pass
    # from models.ddnet import  DDnet
    #
    # model = DDnet(n_channels=3, bilinear=False)
    # state_dict = torch.load(r'snapshots/3-epoch.pt')
    # model.load_state_dict(state_dict['model_state_dict'])
    # img2tensor = transforms.ToTensor()
    #
    # img = io.imread(r'../datasets/DDnet/test/haze/SOTS-indoor_1400_1.png')
    # # img = img2tensor(img)
    # predictor = Predictor(input=r'../datasets/DDnet/test/haze/', mode='folder', model=model, save_dir='../datasets/DDnet/test/result', scrn=True)
    # data = predictor.predict()
    # print(data)




