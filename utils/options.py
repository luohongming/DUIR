
import os.path as osp
import yaml

import json

def parse(opt_path, root_path, is_train=True):
    """Parse option file.

     Args:
         opt_path (str): Option file path.
         root_path (str): Root path
         is_train (bool): Indicate whether in training or not. Default: True.

     Returns:
         (dict): Options.
     """
    opt_path = osp.abspath(opt_path)
    # print(opt_path)
    fileext = osp.splitext(opt_path)[1]
    print(fileext)
    with open(opt_path, mode='r') as f:
        if fileext in ('.yml', '.yaml'):
            opt = yaml.safe_load(f)

        elif fileext in ('.json'):
            opt = json.load(f)

        else:
            raise TypeError(f'Only json or yaml file is supported now, but got {fileext}')

    opt['is_train'] = is_train

    # datasets  ###add phase and scale into datasets options
    # dataset: {'train': ..., 'val': ...}
    for phase, dataset in opt['datasets'].items():
        # for several datasets, e.g., test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths used for resume or load pretrained models
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)

    if is_train:
        experiments_root = osp.join(root_path, 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')

    else:
        results_root = osp.join(root_path, 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt

def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg

# if __name__ == '__main__':
#     path = '../options/train_DSM_x2_SR_VISTAWeChat.json'
#     parse(path, None)
#
#     path = '../options/train_DSM_x2_SR_VISTAWeChat.yml'
#     parse(path, None)
