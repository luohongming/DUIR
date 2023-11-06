

import random
import os
import cv2
import glob
import numpy as np


path = '/home/lhm/data/data/SIDD_medium/SIDD_Medium_Srgb/Data'
save_path = '/home/lhm/data/data/SIDD_medium/SIDD_Medium_Srgb/Data_patches'
meta_info = '/home/lhm/data/data/SIDD_medium/SIDD_Medium_Srgb/meta_info_SIDD_medium_all.txt'


img_instance_list = sorted(os.listdir(path))
print(img_instance_list)

with open(meta_info, 'w') as fin:
    for img_path_ in img_instance_list:

        img_path = os.path.join(path, img_path_)
        save_img_path = os.path.join(save_path, img_path_)
        if not os.path.exists(save_img_path):
            os.mkdir(save_img_path)

        gt_imgs = sorted(glob.glob(os.path.join(img_path, '*GT*')))
        noise_imgs = sorted(glob.glob(os.path.join(img_path, '*NOISY*')))

        for gt_img, noise_img in zip(gt_imgs, noise_imgs):
            noise_img_ = cv2.imread(noise_img)
            gt_img_ = cv2.imread(gt_img)
            gt_name = os.path.split(gt_img)[-1]
            noise_name = os.path.split(noise_img)[-1]
            a = gt_img.split('/')
            a_ = a[-2]

            gt_name = gt_name[:-4]
            noise_name = noise_name[:-4]
            H, W, C = noise_img_.shape
            print(noise_img_.shape)
            max_len_H = H // 256
            max_len_W = W // 256
            patch_list = []
            k = 0
            for i in range(max_len_H-1):
                for j in range(max_len_W-1):
                    noise_patch = noise_img_[i*256:(i+1)*256, j*256:(j+1)*256, :]
                    gt_patch = gt_img_[i*256:(i+1)*256, j*256:(j+1)*256, :]

                    save_noise = os.path.join(save_img_path, f'{noise_name}_{k+1}.png')
                    save_gt = os.path.join(save_img_path, f'{gt_name}_{k+1}.png')

                    cv2.imwrite(save_noise, noise_patch)
                    cv2.imwrite(save_gt, gt_patch)
                    out = f'RealNoise/{a_}/{noise_name}_{k+1}.png RealNoise/{a_}/{gt_name}_{k+1}.png {gt_patch.shape}\n'
                    fin.write(out)
                    k += 1


