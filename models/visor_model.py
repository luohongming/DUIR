
import torch
import torch.nn as nn

from .base_model import BaseModel

from utils.registry import MODEL_REGISTRY
from utils.dist_util import get_dist_info
from utils import get_root_logger, imwrite, tensor2img
from collections import OrderedDict
import scipy.io as scio
from losses import build_loss
import lpips

import os
import numpy as np
from tqdm import tqdm
import os.path as osp

from archs.Net_Big import Diff_Enc, Cont_Enc, Dist_G

@MODEL_REGISTRY.register()
class VISORModel(BaseModel):
    def __init__(self, opt):
        super(VISORModel, self).__init__(opt)

        self.net_contencode = Cont_Enc()
        self.net_diffencode = Diff_Enc()
        self.net_recon = Dist_G()

        # load_net_contencode = torch.load('/home/lhm/PycharmProject/GIR/experiments/cont_enc.pth', map_location=lambda storage, loc: storage)
        # load_net_diffencode = torch.load('/home/lhm/PycharmProject/GIR/experiments/enc_diff.pth', map_location=lambda storage, loc: storage)
        # load_net_recon = torch.load('/home/lhm/PycharmProject/GIR/experiments/dec_dist.pth', map_location=lambda storage, loc: storage)
        # self.net_contencode.load_state_dict(load_net_contencode)
        # self.net_diffencode.load_state_dict(load_net_diffencode)
        # self.net_recon.load_state_dict(load_net_recon)

        self.net_contencode = self.model_to_device(self.net_contencode)
        self.net_diffencode = self.model_to_device(self.net_diffencode)
        self.net_recon = self.model_to_device(self.net_recon)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_recon.train()
        self.net_diffencode.train()
        self.net_contencode.eval()

        train_opt = self.opt['train']
        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        self.cri_perceptual = lpips.LPIPS(net='alex', spatial=True, verbose=False).to(self.device)
        self.cri_perceptual.requires_grad = False

        self.setup_schedulers()
        self.setup_optimizers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_diffencode.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # optim_params = []
        for k, v in self.net_recon.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])

        self.optimizers.append(self.optimizer_g)

    def test(self):
        self.net_diffencode.eval()
        self.net_recon.eval()

        with torch.no_grad():
            code_cont = self.net_contencode(self.gt)
            self.feats, mask = self.net_diffencode(self.lq)
            self.output = self.net_recon(code_cont, self.feats, mask)

        self.net_diffencode.train()
        self.net_recon.train()


    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        code_cont = self.net_contencode(self.gt)
        code_diff, mask = self.net_diffencode(self.lq)
        # print(self.net_diffencode.module.conv1[0].weight)
        self.output = self.net_recon(code_cont, code_diff, mask)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.lq)
            # l_pix.backward()
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_perceptual:
            l_percep = self.cri_perceptual(self.output, self.lq)
            l_percep = l_percep.mean()
            l_total += l_percep
            loss_dict['l_percep'] = l_percep

        l_total.backward()
        self.optimizer_g.step()

        # training mask
        self.optimizer_g.zero_grad()
        for k, v in self.net_diffencode.named_parameters():
            if 'module.mask1.' not in k:
                v.requires_grad = False

        _, maskc = self.net_diffencode(self.lq)
        imgg = (self.lq[:, 0, :, :] * 0.299 + self.lq[:, 1, :, :] * 0.587 + self.lq[:, 2, :, :] * 0.114).unsqueeze(1)
        orgg = (self.gt[:, 0, :, :] * 0.299 + self.gt[:, 1, :, :] * 0.587 + self.gt[:, 2, :, :] * 0.114).unsqueeze(1)

        errc = nn.functional.interpolate(torch.abs(imgg - orgg), size=[maskc.shape[2], maskc.shape[3]],
                                         mode='bilinear')
        loss_m = self.cri_pix(maskc, errc)
        loss_m.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        if hasattr(self, 'feats'):
            out_dict['feats'] = self.feats.detach().cpu()

        return out_dict


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        # with_metrics = self.opt['val']['metrics'] is not None

        # if with_metrics and not hasattr(self, 'metrics_results'):
        #     self.metric_results = {}
        #     # metric_key = self.opt['val']['metrics'].keys()
        #     num_frame_each_folder = Counter(dataset.folder)
        #     for folder, num_frame in num_frame_each_folder.items():
        #         self.metric_results[folder] = torch.zeros(
        #             num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')

        rank, world_size = get_dist_info()
        # if with_metrics:
        #     for _, tensor in self.metric_results.items():
        #         tensor.zero_()

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
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()

            sr_img = tensor2img([visuals['result']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            if 'feats' in visuals:
                gfeats = visuals['feats']
                del self.feats
                feat_dir = osp.join(self.opt['path']['visualization'], 'embedding',
                                         f'{folder}/{img_name}.mat')
                dir_name = os.path.abspath(os.path.dirname(feat_dir))
                os.makedirs(dir_name, exist_ok=True)

                scio.savemat(feat_dir, {'gfeat': np.array(gfeats.squeeze(0).cpu())})

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

            # if with_metrics:
            #
            #     # calculate metrics
            #     for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
            #         metric_data = dict(img1=sr_img, img2=gt_img)
            #         result = calculate_metric(metric_data, opt_)
            #         self.metric_results[folder][idx_count[folder], metric_idx] += result

            # progress bar
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {folder}/{img_name}')
        if rank == 0:
            pbar.close()

    def save(self, epoch, current_iter):
        self.save_network(self.net_diffencode, 'net_diff', current_iter)
        self.save_network(self.net_recon, 'net_recon', current_iter)
        self.save_training_state(epoch, current_iter)