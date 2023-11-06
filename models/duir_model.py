
import torch
from .sr_model import SRModel

from utils.registry import MODEL_REGISTRY
from utils.dist_util import get_dist_info
from utils import get_root_logger, imwrite, tensor2img
from collections import OrderedDict
from collections import Counter
from torch import distributed as dist

from metrics import calculate_metric
from tqdm import tqdm
import os.path as osp
import scipy.io as scio

import numpy as np

lq_list = {'blur1_2': [0.2, 0., 0.], 'blur2_4': [0.5, 0., 0.], 'blur3_6': [0.8, 0., 0.], 'blur2_0': [0.4, 0., 0.], 'blur3_0': [0.65, 0., 0.],
           'GN15': [0., 0.2, 0.], 'GN25': [0., 0.4, 0.], 'GN50': [0., 0.8, 0.], 'GN35': [0., 0.6, 0.],
           'jpeg40': [0., 0., 0.2], 'jpeg20': [0., 0., 0.6], 'jpeg10': [0., 0., 0.8], 'jpeg30': [0., 0., 0.3],
           'high_quality': [0., 0., 0.]}


@MODEL_REGISTRY.register()
class DUIRModel(SRModel):
    def __init__(self, opt):

        super(DUIRModel, self).__init__(opt)

        self.mix = opt.get('mix_degrade', False)

    def feed_data(self, data):    # 数据准备
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if 'feats' in data:
            self.cond = data['feats'].to(self.device)
            self.cond = self.cond.squeeze(1)
        else:

            # for GIRTrainDataset
            if self.mix:
                # cond_list = []
                # for lq_key in data['lq_path']:
                #     cond = lq_key.split('/')[0]
                #     cond_list.append(torch.tensor(lq_list[cond]))
                # self.cond = torch.stack(cond_list, dim=0).to(self.device)
                #
            # for RealBlurDataset
                cond_list = []
                for lq_key in data['lq_path']:
                    cond_list.append(torch.tensor(lq_list['blur1_2']))
                self.cond = torch.stack(cond_list, dim=0).to(self.device)

            else:
            # for GIRTrainDataset_new
                    self.lq = data['lq'][0].to(self.device)
                    if 'gt' in data:
                        self.gt = data['gt'][0].to(self.device)
                    self.cond = torch.tensor(lq_list[data['degrade_type'][0]]).view(1, -1).to(self.device)
            #       self.cond = self.cond.repeat(self.lq.size(0), 1)


        # # for GIRTrainDataset_new
        # self.lq = data['lq'][0].to(self.device)
        # if 'gt' in data:
        #     self.gt = data['gt'][0].to(self.device)
        # self.cond = torch.tensor(lq_list[data['degrade_type'][0]]).view(1, -1).to(self.device)

    def feed_data_test(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        # cond = data['lq_path'].split('/')[0]
        # self.cond = torch.tensor(lq_list[cond]).view(1, -1).to(self.device)  # use provided cond

        self.cond = torch.tensor(lq_list['blur1_2']).view(1, -1).to(self.device)   # do not use cond

    def test(self):
        self.net_g.eval()

        with torch.no_grad():
            self.output = self.net_g(self.lq, self.cond)

        self.net_g.train()
        # if self


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.cond)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            # l_pix.backward()
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None

        if with_metrics and not hasattr(self, 'metrics_results'):
            self.metric_results = {}
            # metric_key = self.opt['val']['metrics'].keys()
            num_frame_each_folder = Counter(dataset.folder)
            for folder, num_frame in num_frame_each_folder.items():
                self.metric_results[folder] = torch.zeros(
                    num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')

        rank, world_size = get_dist_info()
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()

        # record all frames (border and center frames)
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='image')

        idx_count = {}
        for idx in range(rank, len(dataset), world_size):
            val_data = dataset[idx]
            val_data['lq'].unsqueeze_(0)
            val_data['gt'].unsqueeze_(0)
            a = osp.split(val_data['lq_path'])
            folder, img_name = a[0], a[1][:-4]
            if folder not in idx_count.keys():
                idx_count.update({folder: 0})
            else:
                idx_count[folder] += 1

            # self.feed_data(val_data)
            self.feed_data_test(val_data)
            self.test()

            visuals = self.get_current_visuals()

            sr_img = tensor2img([visuals['result']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            # if 'feats' in visuals:
            #     gfeats = visuals['feats']
            #     del self.feats
            #     feat_dir = '/home/lhm/PycharmProject/VISOR_feature/feats2/'
            #     save_name = f'{folder}/{img_name}.mat'
            #     print(save_name)
            #     scio.savemat(feat_dir + save_name, {'gfeat': np.array(gfeats.squeeze(0).cpu())})

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()


            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             f'{folder}/{img_name}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{folder}/{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{folder}/{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:

                # calculate metrics
                for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                    metric_data = dict(img1=sr_img, img2=gt_img)
                    result = calculate_metric(metric_data, opt_)
                    self.metric_results[folder][idx_count[folder], metric_idx] += result

            # progress bar
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {folder}/{img_name}')
        if rank == 0:
            pbar.close()

        if with_metrics:

            if self.opt['dist']:
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                dist.barrier()
            else:
                pass

            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger=None):
        # average all frames for each sub-folder
        # metric_results_avg is a dict:{
        #    'folder1': tensor (len(metrics)),
        #    'folder2': tensor (len(metrics))
        # }

        metric_results_avg = {
            folder: torch.mean(tensor, dim=0).cpu()
            for (folder, tensor) in self.metric_results.items()
        }

        # total_avg_results is a dict: {
        #    'metric1': float,
        #    'metric2': float
        # }
        total_avg_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        for folder, tensor in metric_results_avg.items():
            for idx, metric in enumerate(total_avg_results.keys()):
                total_avg_results[metric] += metric_results_avg[folder][idx].item()
        # average among folders
        for metric in total_avg_results.keys():
            total_avg_results[metric] /= len(metric_results_avg)

        log_str = f'Validation {dataset_name}\n'
        for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
            log_str += f'\t # {metric}: {value:.4f}'
            for folder, tensor in metric_results_avg.items():
                log_str += f'\t # {folder}: {tensor[metric_idx].item():.4f}'
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
                for folder, tensor in metric_results_avg.items():
                    tb_logger.add_scalar(f'metrics/{metric}/{folder}', tensor[metric_idx].item(), current_iter)
