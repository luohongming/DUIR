import os

import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from data.transforms import augment, paired_random_crop
from utils import FileClient, imfrombytes, img2tensor, modcrop
from pathlib import Path
import numpy as np
import random
import scipy.io as scio


from utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class GIRTrainDataset(data.Dataset):

    def __init__(self, opt):
        self.opt = opt

        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        # self.gray = True if opt['util_gray'] else False

        self.feat_folder = opt['feat_folder'] if 'feat_folder' in opt else None

        with open(opt['meta_info_file'], 'r') as fin:
            self.lq_keys = []
            self.gt_keys = []
            self.folder = []
            for line in fin:
                a = line.split(' ')
                self.lq_keys.append(a[0])
                self.gt_keys.append(a[1].replace(',', ''))
                self.folder.append(os.path.split(a[0])[0])

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

    def __getitem__(self, idx):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        lq_key = self.lq_keys[idx]
        gt_key = self.gt_keys[idx]
        scale = self.opt['scale']
        if self.feat_folder is not None:
            degrade_feat_name = lq_key[:-4] + '.mat'
            degrade_feat_path = os.path.join(self.feat_folder, degrade_feat_name)
            degrade_feat = scio.loadmat(degrade_feat_path)['gfeat']
            degrade_feat = torch.from_numpy(degrade_feat).float()

        if self.is_lmdb:
            img_gt_path = gt_key.split('.')[0]
            img_lq_path = lq_key.split('.')[0]
        else:
            img_gt_path = self.gt_root / gt_key
            img_lq_path = self.lq_root / lq_key

        img_byte = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_byte, float32=True)

        img_byte = self.file_client.get(img_lq_path, 'lq')
        img_lq = imfrombytes(img_byte, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop

            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, img_gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt[0], img_lq[0]])

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        if self.feat_folder is not None:
            return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_key, 'gt_path': gt_key, 'feats': degrade_feat}
        else:
            return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_key, 'gt_path': gt_key}

    def __len__(self):
        return len(self.gt_keys)

#
@DATASET_REGISTRY.register()
class GIRTrainDataset_new(data.Dataset):

    def __init__(self, opt):
        self.opt = opt

        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.batch_samples = opt['batch_size']

        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        if opt['degrade_type'] == 'all':
            self.degrade_type = ['blur1_2', 'blur2_4', 'blur3_6',
                                 'GN15', 'GN25', 'GN50',
                                 'jpeg10', 'jpeg20', 'jpeg40']
        elif opt['degrade_type'] == 'all_wHQ':
            self.degrade_type = ['blur1_2', 'blur2_4', 'blur3_6',
                                 'GN15', 'GN25', 'GN50',
                                 'jpeg10', 'jpeg20', 'jpeg40', 'high_quality']
        else:
            self.degrade_type = opt['degrade_type'].split(',')

        # self.gray = True if opt['util_gray'] else False
        assert isinstance(self.degrade_type, list), 'Degradation type error'

        with open(opt['meta_info_file'], 'r') as fin:
            self.lq_keys = []
            self.gt_keys = []
            self.folder = []
            for line in fin:
                line = line[:-1]

                # self.lq_keys.append(a[0])
                self.gt_keys.append(line)
                self.lq_keys.append(line[:-3]+'png')
                # self.folder.append(os.path.split(a[0])[0])

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

    def __getitem__(self, idx):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        degrade_type = random.choice(self.degrade_type)
        scale = self.opt['scale']

        idx_list = []
        for i in range(self.batch_samples):
            idx_ = (idx + i) % len(self.gt_keys)
            idx_list.append(idx_)

        lq_imgs = []
        gt_imgs = []
        for idx_ in idx_list:
            lq_key = os.path.join(degrade_type, self.lq_keys[idx_])
            gt_key = os.path.join('high_quality', self.gt_keys[idx_])

            if self.is_lmdb:
                img_gt_path = gt_key.split('.')[0]
                img_lq_path = lq_key.split('.')[0]
            else:
                img_gt_path = self.gt_root / gt_key
                img_lq_path = self.lq_root / lq_key

            img_byte = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_byte, float32=True)

            img_byte = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_byte, float32=True)

            # augmentation for training
            if self.opt['phase'] == 'train':
                gt_size = self.opt['gt_size']
                # random crop

                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, img_gt_path)
                # flip, rotatio
                img_gt, img_lq = augment([img_gt[0], img_lq[0]])

            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True)

            # normalize
            if self.mean is not None or self.std is not None:
                normalize(img_lq, self.mean, self.std, inplace=True)
                normalize(img_gt, self.mean, self.std, inplace=True)

            lq_imgs.append(img_lq)
            gt_imgs.append(img_gt)

        lq_imgs = torch.stack(lq_imgs, dim=0)
        gt_imgs = torch.stack(gt_imgs, dim=0)

        return {'lq': lq_imgs, 'gt': gt_imgs, 'degrade_type': degrade_type}

    def __len__(self):
        return len(self.gt_keys)


if __name__ == '__main__':
    import time
    opt = {'io_backend': {'type': 'disk'},
           'dataroot_gt': r'D:\datasets\degraded_imgs\blur1_2\BSD500',
           'dataroot_lq': r'D:\datasets\degraded_imgs\blur1_2\BSD500',
           'meta_info_file': 'D:\pycharm-workspace\hongming\DUIR\data\meta_info\meta_info_WED_DIV2K_BSD_Flickr2k_train_woHQ.txt',
           'gt_size': 128,
           'phase': 'train',
           'use_flip': False, 'use_rot': False, 'scale': 1}

    # opt1 = {'io_backend': {'type': 'lmdb'},
    #        'dataroot_gt': '/home/lhm/data/data/degraded_imgs.lmdb',
    #        'dataroot_lq': '/home/lhm/data/data/degraded_imgs.lmdb',
    #        'meta_info_file': '/home/lhm/PycharmProject/GIR/data/meta_info/meta_info_WED_DIV2K_BSD_Flickr2k_train_all.txt',
    #        'gt_size': 128,
    #        'phase': 'train',
    #        'use_flip': False, 'use_rot': False, 'scale': 1, 'batch_size': 32, 'degrade_type': 'all'}

    dataset = GIRTrainDataset(opt)
    data = dataset.__getitem__(3)
    print(data)
