
import os
import glob
import cv2
from data.transforms import add_jpg_noise, add_gaussian_noise, add_gaussian_blur
path = '/home/lhm/data/data/degraded_imgs_x2'
LQ_paths = ['blur1_2', 'blur2_4', 'blur3_6', 'GN15', 'GN25', 'GN50', 'jpeg10', 'jpeg20', 'jpeg40']
# LQ_paths = ['jpeg10', 'jpeg20', 'jpeg40']
# LQ_paths = ['blur2_0', 'blur3_0', 'GN35', 'jpeg30']
from utils.img_util import crop_border

hq_path = os.path.join(path, 'high_quality')
# datasets = os.listdir(hq_path)

# datasets = ['CBSD68', 'Kodak24', 'McMaster', 'urban100']
# datasets = ['classic5', 'LIVE1',]
# datasets = ['urban100', 'Kodak24']
# datasets = ['Set5', 'BSDS100', 'Set14', 'urban100']
datasets = ['BSD500', 'DIV2K_x2', 'WED', 'Flickr2K_x2']

for LQ_path in LQ_paths:
    lq_path = os.path.join(path, LQ_path)
    for dataset in datasets:
        lq_path_ = os.path.join(lq_path, dataset)
        if not os.path.exists(lq_path_):
            os.makedirs(lq_path_)

        hq_img_paths = glob.glob(os.path.join(hq_path, dataset, '*'))
        for hq_img_path in hq_img_paths:
            hq_img = cv2.imread(hq_img_path, cv2.IMREAD_UNCHANGED)
            img_name = os.path.splitext(os.path.basename(hq_img_path))[0]
            if '+' in LQ_path:
                degrade_list = LQ_path.split('+')
                degrade_dict = {}
                for degrade in degrade_list:
                    if 'blur' in degrade:
                        sigma = degrade.replace('blur', '')
                        sigma = float(sigma.replace('_', '.'))
                        degrade_dict.update({'blur': sigma})
                    if 'GN' in degrade:
                        sigma = int(degrade.replace('GN', ''))
                        degrade_dict.update({'GN': sigma})
                    if 'jpeg' in degrade:
                        qp = int(degrade.replace('jpeg', ''))
                        degrade_dict.update({'jpeg': qp})
            else:
                degrade_dict = {}
                if 'blur' in LQ_path:
                    sigma = LQ_path.replace('blur', '')
                    sigma = float(sigma.replace('_', '.'))
                    degrade_dict.update({'blur': sigma})
                if 'GN' in LQ_path:
                    sigma = int(LQ_path.replace('GN', ''))
                    degrade_dict.update({'GN': sigma})
                if 'jpeg' in LQ_path:
                    qp = int(LQ_path.replace('jpeg', ''))
                    degrade_dict.update({'jpeg': qp})

            print(degrade_dict)
            out_img = hq_img
            for k, v in degrade_dict.items():
                if k == 'blur':
                    out_img = add_gaussian_blur(out_img, 21, v)
                if k == 'GN':
                    out_img = add_gaussian_noise(out_img, v)
                if k == 'jpeg':
                    out_img = add_jpg_noise(out_img, v)

            save_name = os.path.join(lq_path_, img_name + '.png')
            print(save_name)
            cv2.imwrite(save_name, out_img)

            # if 'blur' in LQ_path:
            #     sigma = LQ_path.replace('blur', '')
            #     sigma = float(sigma.replace('_', '.'))
            #     lq_img = add_gaussian_blur(hq_img, 21, sigma)
            #
            # if 'GN' in LQ_path:
            #     sigma = int(LQ_path.replace('GN', ''))
            #     lq_img = add_gaussian_noise(hq_img, sigma)
            #     # save_name = os.path.join(lq_path, img_name + '.png')
            #     # cv2.imwrite(save_name, lq_img)
            #
            # if 'jpeg' in LQ_path:
            #     qp = int(LQ_path.replace('jpeg', ''))
            #     lq_img = add_jpg_noise(hq_img, qp)
            #     # save_name = os.path.join(lq_path, img_name + '.png')
            #     # cv2.imwrite(save_name, lq_img)
            #
            # save_name = os.path.join(lq_path_, img_name + '.png')
            # print(save_name)
            # cv2.imwrite(save_name, lq_img)