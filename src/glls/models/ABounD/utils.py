import torchvision.transforms as transforms
from VVCLIP_lib.transform import image_transform
from VVCLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os
import logging
import sys 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import cv2
import sys
from skimage.morphology import disk
from skimage.filters import median
from typing import Union, List
from pkg_resources import packaging 
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
from skimage import measure
class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=0.25, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        # logit: [B, 2, 224, 224]
        # target:[B, 1, 224, 224]
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        # 2
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            # [B, 2, 224*224]
            logit = logit.view(logit.size(0), logit.size(1), -1)
            # [B, 224*224, 2]
            logit = logit.permute(0, 2, 1).contiguous()
            # [B*224*224, 2]
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        # [B*224*224, 1]
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        # [B*224*224, 1]
        idx = target.cpu().long()

        # [B*224*224, 2]
        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()

        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        loss = 1 - N_dice_eff.sum() / N
        return loss
def get_logger(save_path, logger_name='test'):
    log_dir = save_path 
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, 'log.txt')

    logger = logging.getLogger(logger_name)


    if not logger.hasHandlers(): 
        logger.setLevel(logging.INFO) 

        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
            datefmt='%y-%m-%d %H:%M:%S'
        )

        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)



    return logger
def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == 'mpdd':
        obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    elif dataset_name == 'btad':
        obj_list = ['01', '02', '03']
    elif dataset_name == 'DAGM_KaggleUpload':
        obj_list = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
    elif dataset_name == 'SDD':
        obj_list = ['electrical commutators']
    elif dataset_name == 'DTD':
        obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
    elif dataset_name == 'colon':
        obj_list = ['colon']
    elif dataset_name == 'ISBI':
        obj_list = ['skin']
    elif dataset_name == 'Chest':
        obj_list = ['chest']
    elif dataset_name == 'thyroid':
        obj_list = ['thyroid']
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id

class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        self.dataset_name = dataset_name
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)

        self.cls_name_list = [data['cls_name'] for data in self.data_all]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                # just for classification not report error
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask_np = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask_np.astype(np.uint8) * 255, mode='L')
        # transforms
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(   
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        
        # Ensure img_mask is always a Tensor or an empty list (as per original code)
        # If it's a tensor, ensure it has a channel dimension for consistency if needed, 
        # but original code likely wants [H, W]. Let's stick to original intent for now.
        img_mask = torch.empty(0) if img_mask is None else img_mask # Return empty tensor if None

        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 'specie_name': data['specie_name'],
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}
def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def get_transform(args):
    preprocess = image_transform(args.image_size, is_train=False, mean = OPENAI_DATASET_MEAN, std = OPENAI_DATASET_STD)
    target_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()
    ])
    preprocess.transforms[0] = transforms.Resize(size=(args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC,
                                                    max_size=None, antialias=None)
    preprocess.transforms[1] = transforms.CenterCrop(size=(args.image_size, args.image_size))
    return preprocess, target_transform

import torch
# from dataset import VisaDataset, MVTecDataset
import torch.nn.functional as F
import kornia as K
import torch
import numpy as np
def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])

def get_translation_mat(a, b):
    return torch.tensor([[1, 0, a],
                         [0, 1, b]])

def rot_img(x, theta):
    dtype =  torch.FloatTensor
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size(),align_corners=True).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection",align_corners=True)
    return x

def translation_img(x, a, b):
    dtype =  torch.FloatTensor
    rot_mat = get_translation_mat(a, b)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size(),align_corners=True).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection",align_corners=True)
    return x

def hflip_img(x):
    x = K.geometry.transform.hflip(x)
    return x


def rot90_img(x,k):
    # k is 0,1,2,3
    degreesarr = [0., 90., 180., 270., 360]
    degrees = torch.tensor(degreesarr[k])
    x = K.geometry.transform.rotate(x, angle = degrees, padding_mode='reflection')
    return x

def grey_img(x):
    x = K.color.rgb_to_grayscale(x)
    x = x.repeat(1, 3, 1,1)
    return x

def aug(support_img):
    augment_support_img = support_img
    # rotate img with small angle
    for angle in [-np.pi / 4, -3 * np.pi / 16, -np.pi / 8, -np.pi / 16, np.pi / 16, np.pi / 8, 3 * np.pi / 16,
                  np.pi / 4]:
        rotate_img = rot_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate_img], dim=0)
    # translate img
    for a, b in [(0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2), (0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1),
                 (0.1, -0.1)]:
        trans_img = translation_img(support_img, a, b)
        augment_support_img = torch.cat([augment_support_img, trans_img], dim=0)
    # hflip img
    flipped_img = hflip_img(support_img)
    augment_support_img = torch.cat([augment_support_img, flipped_img], dim=0)
    # rgb to grey img
    greyed_img = grey_img(support_img)
    augment_support_img = torch.cat([augment_support_img, greyed_img], dim=0)
    # rotate img in 90 degree
    for angle in [1, 2, 3]:
        rotate90_img = rot90_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate90_img], dim=0)
    augment_support_img = augment_support_img[torch.randperm(augment_support_img.size(0))]
    return augment_support_img

WIDTH_BOUNDS_PCT = {'bottle':((0.31, 0.31), (0.31, 0.31)), 'cable':((0.14, 0.14), (0.14, 0.14)), 'capsule':((0.14, 0.14), (0.14, 0.14)), 
                    'hazelnut':((0.15, 0.15), (0.15, 0.15)), 'metal_nut':((0.32, 0.32), (0.32, 0.32)), 'pill':((0.15, 0.15), (0.15, 0.15)), 
                    'screw':((0.04, 0.045), (0.04, 0.045)), 'toothbrush':((0.17, 0.17), (0.17, 0.17)), 'transistor':((0.1, 0.1), (0.1, 0.1)), 
                    'zipper':((0.19, 0.19), (0.19, 0.19)), 
                    'carpet':((0.03, 0.4), (0.03, 0.4)), 'grid':((0.11, 0.11), (0.11, 0.11)), 
                    'leather':((0.03, 0.4), (0.03, 0.4)), 'tile':((0.12, 0.12), (0.12, 0.12)), 'wood':((0.09, 0.09), (0.09, 0.09)),
                    'candle':((0.16, 0.16), (0.16, 0.16)),'capsules':((0.065, 0.065), (0.065, 0.065)), 'cashew':((0.14, 0.15), (0.14, 0.15)),
                    'chewinggum':((0.11, 0.11), (0.11, 0.11)),'fryum':((0.11, 0.11), (0.11, 0.11)),'macaroni2':((0.1, 0.1), (0.1, 0.1)),
                    'pcb4':((0.07, 0.07), (0.07, 0.07)),'pcb3':((0.09, 0.09), (0.09, 0.09)),'pcb2':((0.22, 0.22), (0.22, 0.22)),
                    'pcb1':((0.09, 0.09), (0.09, 0.09)),'macaroni1':((0.13, 0.13), (0.13, 0.13)),'pipe_fryum':((0.1, 0.1), (0.1, 0.1))
                   }

MIN_OVERLAP_PCT = {'bottle': 0.25,  'capsule':0.25, 
                   'hazelnut':0.25, 'metal_nut':0.25, 'pill':0.25, 
                   'screw':0.25, 'toothbrush':0.25, 
                   'zipper':0.25}
MIN_OBJECT_PCT = {'bottle': 0.7,  'capsule':0.7, 
                  'hazelnut':0.7, 'metal_nut':0.5, 'pill':0.7, 
                  'screw':.5, 'toothbrush':0.25, 
                  'zipper':0.75}
NUM_PATCHES = {
     'screw': 4,  'zipper': 4,
    'carpet': 4, 'grid': 4, 'leather': 4, 'tile': 4, 'wood': 4,'transistor':3,'bottle':1,'cable':3,'screw':3
}

INTENSITY_LOGISTIC_PARAMS = {
    'bottle': (1/12, 7), 'cable': (1/12, 14), 'capsule': (1/2, 20), 
    'hazelnut': (1/12, 24), 'metal_nut': (1/3, 1), 
    'pill': (1/3, 7), 'screw': (1, 3), 'toothbrush': (1/6, 15), 
    'transistor': (1/6, 10), 'zipper': (1/6, 17),
    'carpet': (1/3, 7), 'grid': (1/3, 9), 'leather': (1/3, 7), 
    'tile': (1/3, 7), 'wood': (1/6, 17),
    'candle':(1/12, 1/12),'capsules':(1/12, 1),'cashew':(1/3, 5),'chewinggum':(1/3,11),'fryum':(1/3,2),
    'macaroni2':(1/3, 11),'pcb4':(1/3, 15),'pcb3':(1/3, 25),'pcb2':(1/3, 12),'pcb1':(1/3, 5),'pipe_fryum':(1/3, 23)
}

UNALIGNED_OBJECTS = ['bottle', 'hazelnut', 'metal_nut', 'screw']

BACKGROUND = {
    'bottle': (200, 60), 'screw': (200, 60), 'capsule': (200, 60), 
    'zipper': (200, 60), 'hazelnut': (20, 20), 'pill': (20, 20), 
    'toothbrush': (20, 20), 'metal_nut': (20, 20)
}

OBJECTS = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 
           'pill', 'screw', 'toothbrush', 'transistor', 'zipper','pcb4']
TEXTURES = ['carpet', 'grid', 'leather', 'tile', 'wood']



def load_foreground_mask(obj_name, img_filename, dataset_name, data_path):
    if dataset_name == 'mvtec':
        mask_path = os.path.join(data_path, 'fg_mask', obj_name, img_filename)
    elif dataset_name == 'visa':
        mask_path = os.path.join(data_path, 'fg_mask', obj_name, img_filename)
    else:
        return None
    
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = (mask > 127).astype(np.uint8)
            return mask
    
    return None


def get_anomaly_synthesis_params(dataset_name, class_name=None):

    if dataset_name.lower() in ['mvtec', 'visa']:
        params = {
            'width_bounds_pct'        : WIDTH_BOUNDS_PCT.get(class_name, ((0.1, 0.2), (0.1, 0.2))),
            'intensity_logistic_params': INTENSITY_LOGISTIC_PARAMS.get(class_name, (1/12, 24)),
            'num_patches'             : NUM_PATCHES.get(class_name, 3),
            'min_object_pct'          : 0,
            'min_overlap_pct'         : 0.25,
            'gamma_params'            : (2, 0.05, 0.03),
            'resize'                  : True,
            'shift'                   : True,
            'same'                    : False,
            'mode'                    : cv2.NORMAL_CLONE,
            'label_mode'              : 'logistic-intensity',
            'skip_background'         : BACKGROUND.get(class_name), 
        }

        if class_name in TEXTURES:
            params['resize_bounds'] = (0.5, 2)

    # --- 这里保留了您代码中为其他数据集准备的逻辑 ---
    else:  # 其余数据集保持原默认
        params = {
            'width_bounds_pct': ((0.05, 0.15), (0.05, 0.15)),
            'num_patches'     : 10,
            'shift'           : True,
            'resize'          : True,
            'mode'            : cv2.NORMAL_CLONE,
            'resize_bounds'   : (0.7, 1.3),
            'skip_background' : None,
            'min_object_pct'  : 0.25,
            'min_overlap_pct' : 0.25,
            'label_mode'      : 'binary',
            'gamma_params'    : None,
            'verbose'         : True
        }
    return params
def patch_ex(ima_dest, ima_src=None, same=False, num_patches=1,
             mode=cv2.NORMAL_CLONE, width_bounds_pct=((0.05,0.2),(0.05,0.2)), min_object_pct=0.25, 
             min_overlap_pct=0.25, shift=True, label_mode='binary', skip_background=None, tol=1, resize=True,
             gamma_params=None, intensity_logistic_params=(1/6, 20),
             resize_bounds=(0.7, 1.3), num_ellipses=None, verbose=True, cutpaste_patch_generation=False,
             foreground_mask=None):
    """
    Create a synthetic training example from the given images by pasting/blending random patches.
    Args:
        ima_dest (uint8 numpy array): image with shape (W,H,3) or (W,H,1) where patch should be changed
        ima_src (uint8 numpy array): optional, otherwise use ima_dest as source
        same (bool): use ima_dest as source even if ima_src given
        mode: 'uniform', 'swap', 'mix', cv2.NORMAL_CLONE, or cv2.MIXED_CLONE what blending method to use
             ('mix' is flip a coin between normal and mixed clone)
        num_patches (int): how many patches to add. the method will always attempt to add the first patch,
                    for each subsequent patch it flips a coin
        width_bounds_pct ((float, float), (float, float)): min half-width of patch ((min_dim1, max_dim1), (min_dim2, max_dim2))
        shift (bool): if false, patches in src and dest image have same coords. otherwise random shift
        resize (bool): if true, patch is resampled at random size (within bounds and keeping aspect ratio the same) before blending  
        skip_background (int, int) or [(int, int),]: optional, assume background color is first and only interpolate patches
                    in areas where dest or src patch has pixelwise MAD < second from background.
        tol (int): mean abs intensity change required to get positive label
        gamma_params (float, float, float): optional, (shape, scale, left offset) of gamma dist to sample half-width of patch from,
                    otherwise use uniform dist between 0.05 and 0.95
        intensity_logistic_params (float, float): k, x0 of logitistc map for intensity based label
        num_ellipses (int): optional, if set, the rectangular patch mask is filled with random ellipses
        label_mode: 'binary', 
                    'continuous' -- use interpolation factor as label (only when mode is 'uniform'),
                    'intensity' -- use median filtered mean absolute pixelwise intensity difference as label,
                    'logistic-intensity' -- use logistic median filtered of mean absolute pixelwise intensity difference as label,
        cutpaste_patch_generation (bool): optional, if set, width_bounds_pct, resize, skip_background, min_overlap_pct, min_object_pct, 
                    num_patches and gamma_params are ignored. A single patch is sampled as in the CutPaste paper: 
                        1. sampling the area ratio between the patch and the full image from (0.02, 0.15)
                        2. determine the aspect ratio by sampling from (0.3, 1) union (1, 3.3)
                        3. sample location such that patch is contained entirely within the image
        foreground_mask (uint8 numpy array): optional, foreground mask with shape (H,W), values 0 (background) or 1 (foreground)
    """
    if mode == 'mix':
        mode = (cv2.NORMAL_CLONE, cv2.MIXED_CLONE)[np.random.randint(2)]

    if cutpaste_patch_generation:
        width_bounds_pct = None 
        resize = False 
        skip_background = None
        min_overlap_pct = None 
        min_object_pct = None
        gamma_params = None
        num_patches = 1

    ima_src = ima_dest.copy() if same or (ima_src is None) else ima_src

    if foreground_mask is not None:
        if len(foreground_mask.shape) == 2:
            foreground_mask = foreground_mask[..., None]
        src_object_mask = foreground_mask.astype(np.uint8)
        dest_object_mask = foreground_mask.astype(np.uint8)
    elif skip_background is not None and not cutpaste_patch_generation:
        if isinstance(skip_background, tuple):
            skip_background = [skip_background]
        src_object_mask = np.ones_like(ima_src[...,0:1])
        dest_object_mask = np.ones_like(ima_dest[...,0:1])
        for background, threshold in skip_background:
            src_object_mask &= np.uint8(np.abs(ima_src.mean(axis=-1, keepdims=True) - background) > threshold)
            dest_object_mask &= np.uint8(np.abs(ima_dest.mean(axis=-1, keepdims=True) - background) > threshold)
        src_object_mask[...,0] = cv2.medianBlur(src_object_mask[...,0], 7)  # remove grain from threshold choice
        dest_object_mask[...,0] = cv2.medianBlur(dest_object_mask[...,0], 7)  # remove grain from threshold choice
    else:
        src_object_mask = None
        dest_object_mask = None
    
    # add patches
    mask = np.zeros_like(ima_dest[..., 0:1])  # single channel
    patchex = ima_dest.copy()
    coor_min_dim1, coor_max_dim1, coor_min_dim2, coor_max_dim2 = mask.shape[0] - 1, 0, mask.shape[1] - 1, 0 
    if label_mode == 'continuous':
        factor = np.random.uniform(0.05, 0.95)
    else:
        factor = 1
    
    patches_added = 0
    for i in range(num_patches):
        if i == 0 or np.random.randint(2) > 0:  # at least one patch
            max_attempts = 10
            for attempt in range(max_attempts):
                result = _patch_ex(
                    patchex, ima_src, dest_object_mask, src_object_mask, mode, label_mode, shift, resize, width_bounds_pct, 
                    gamma_params, min_object_pct, min_overlap_pct, factor, resize_bounds, num_ellipses, verbose, cutpaste_patch_generation)
                
                new_patchex, ((_coor_min_dim1, _coor_max_dim1), (_coor_min_dim2, _coor_max_dim2)), patch_mask = result
                
                if patch_mask is not None:
                    patchex = new_patchex
                    mask[_coor_min_dim1:_coor_max_dim1,_coor_min_dim2:_coor_max_dim2] = patch_mask
                    coor_min_dim1 = min(coor_min_dim1, _coor_min_dim1) 
                    coor_max_dim1 = max(coor_max_dim1, _coor_max_dim1) 
                    coor_min_dim2 = min(coor_min_dim2, _coor_min_dim2)
                    coor_max_dim2 = max(coor_max_dim2, _coor_max_dim2)
                    patches_added += 1
                    break
                elif attempt == max_attempts - 1:
                    if verbose:
                        print(f'Failed to generate patch {i+1} after {max_attempts} attempts.')
    
    if patches_added == 0:
        if verbose:
            print('No patches were successfully added.')
        return ima_dest, np.zeros_like(ima_dest[..., 0:1])

    # create label
    label_mask = np.uint8(np.mean(np.abs(1.0 * mask*ima_dest - 1.0 * mask*patchex), axis=-1, keepdims=True) > tol)
    label_mask[...,0] = cv2.medianBlur(label_mask[...,0], 5)  # remove grain from threshold choice

    if label_mode == 'continuous':
        label = label_mask * factor
    elif label_mode in ['logistic-intensity', 'intensity']:
        k, x0 = intensity_logistic_params
        label = np.mean(np.abs(label_mask * ima_dest * 1.0 - label_mask * patchex * 1.0), axis=-1, keepdims=True)
        label[...,0] = median(label[...,0], disk(5))
        if label_mode == 'logistic-intensity':
            label = label_mask / (1 + np.exp(-k * (label - x0)))
    elif label_mode == 'binary':
        label = label_mask  
    else:
        raise ValueError("label_mode not supported" + str(label_mode))

    return patchex, label


def _patch_ex(ima_dest, ima_src, dest_object_mask, src_object_mask, mode, label_mode, shift, resize, width_bounds_pct, 
              gamma_params, min_object_pct, min_overlap_pct, factor, resize_bounds, num_ellipses, verbose, cutpaste_patch_generation):
    
    if dest_object_mask is not None:
        foreground_pixels = np.sum(dest_object_mask)
        total_pixels = dest_object_mask.shape[0] * dest_object_mask.shape[1]
        if foreground_pixels / total_pixels < 0.01:
            if verbose:
                print('Foreground area too small for patch generation.')
            return ima_dest.copy(), ((0,0),(0,0)), None
    
    if cutpaste_patch_generation:
        skip_background = False
        dims = np.array(ima_dest.shape)
        if dims[0] != dims[1]:
            raise ValueError("CutPaste patch generation only works for square images")
        # 1. sampling the area ratio between the patch and the full image from (0.02, 0.15)
        # (divide by 4 as patch-widths below are actually half-widths)
        area_ratio = np.random.uniform(0.02, 0.15) / 4.0
        #  2. determine the aspect ratio by sampling from (0.3, 1) union (1, 3.3)
        if np.random.randint(2) > 0:
            aspect_ratio = np.random.uniform(0.3, 1)
        else:
            aspect_ratio = np.random.uniform(1, 3.3)
        
        patch_width_dim1 = int(np.rint(np.clip(np.sqrt(area_ratio * aspect_ratio * dims[0]**2), 0, dims[0])))
        patch_width_dim2 = int(np.rint(np.clip(area_ratio * dims[0]**2 / patch_width_dim1, 0, dims[1])))
        #  3. sample location such that patch is contained entirely within the image
        center_dim1 = np.random.randint(patch_width_dim1, dims[0] - patch_width_dim1)
        center_dim2 = np.random.randint(patch_width_dim2, dims[1] - patch_width_dim2)

        coor_min_dim1 = np.clip(center_dim1 - patch_width_dim1, 0, dims[0])
        coor_min_dim2 = np.clip(center_dim2 - patch_width_dim2, 0, dims[1])
        coor_max_dim1 = np.clip(center_dim1 + patch_width_dim1, 0, dims[0])
        coor_max_dim2 = np.clip(center_dim2 + patch_width_dim2, 0, dims[1])

        patch_mask = np.ones((coor_max_dim1 - coor_min_dim1, coor_max_dim2 - coor_min_dim2, 1), dtype=np.uint8)
    else:
        skip_background = (src_object_mask is not None) and (dest_object_mask is not None)
        dims = np.array(ima_dest.shape)
        min_width_dim1 = (width_bounds_pct[0][0]*dims[0]).round().astype(int)
        max_width_dim1 = (width_bounds_pct[0][1]*dims[0]).round().astype(int)
        min_width_dim2 = (width_bounds_pct[1][0]*dims[1]).round().astype(int)
        max_width_dim2 = (width_bounds_pct[1][1]*dims[1]).round().astype(int)

        if gamma_params is not None:
            shape, scale, lower_bound = gamma_params
            patch_width_dim1 = int(np.clip((lower_bound + np.random.gamma(shape, scale)) * dims[0], min_width_dim1, max_width_dim1))
            patch_width_dim2 = int(np.clip((lower_bound + np.random.gamma(shape, scale)) * dims[1], min_width_dim2, max_width_dim2))
        else:
            patch_width_dim1 = np.random.randint(min_width_dim1, max_width_dim1)
            patch_width_dim2 = np.random.randint(min_width_dim2, max_width_dim2)

        found_patch = False
        attempts = 0
        while not found_patch:
            valid_min_dim1 = max(min_width_dim1, patch_width_dim1)
            valid_max_dim1 = min(dims[0] - min_width_dim1, dims[0] - patch_width_dim1)
            valid_min_dim2 = max(min_width_dim2, patch_width_dim2)
            valid_max_dim2 = min(dims[1] - min_width_dim2, dims[1] - patch_width_dim2)
            
            if valid_min_dim1 >= valid_max_dim1 or valid_min_dim2 >= valid_max_dim2:
                patch_width_dim1 = max(min_width_dim1, patch_width_dim1 // 2)
                patch_width_dim2 = max(min_width_dim2, patch_width_dim2 // 2)
                attempts += 1
                if attempts > 100:
                    if verbose:
                        print('No suitable patch size found.')
                    return ima_dest.copy(), ((0,0),(0,0)), None
                continue
            
            if dest_object_mask is not None:
                center_attempts = 0
                center_found = False
                while center_attempts < 50:
                    center_dim1 = np.random.randint(valid_min_dim1, valid_max_dim1)
                    center_dim2 = np.random.randint(valid_min_dim2, valid_max_dim2)
                    
                    if dest_object_mask[center_dim1, center_dim2, 0] > 0:
                        center_found = True
                        break
                    center_attempts += 1
                
                if not center_found:
                    patch_width_dim1 = max(min_width_dim1, patch_width_dim1 // 2)
                    patch_width_dim2 = max(min_width_dim2, patch_width_dim2 // 2)
                    attempts += 1
                    if attempts > 100:
                        if verbose:
                            print('No center point found in foreground region.')
                        return ima_dest.copy(), ((0,0),(0,0)), None
                    continue
            else:
                center_dim1 = np.random.randint(valid_min_dim1, valid_max_dim1)
                center_dim2 = np.random.randint(valid_min_dim2, valid_max_dim2)

            coor_min_dim1 = np.clip(center_dim1 - patch_width_dim1, 0, dims[0])
            coor_min_dim2 = np.clip(center_dim2 - patch_width_dim2, 0, dims[1])
            coor_max_dim1 = np.clip(center_dim1 + patch_width_dim1, 0, dims[0])
            coor_max_dim2 = np.clip(center_dim2 + patch_width_dim2, 0, dims[1])

            if num_ellipses is not None:
                ellipse_min_dim1 = min_width_dim1
                ellipse_min_dim2 = min_width_dim2
                ellipse_max_dim1 = max(min_width_dim1 + 1, patch_width_dim1 // 2)
                ellipse_max_dim2 = max(min_width_dim2 + 1, patch_width_dim2 // 2)
                patch_mask = np.zeros((coor_max_dim1 - coor_min_dim1, coor_max_dim2 - coor_min_dim2), dtype=np.uint8) 
                x = np.arange(patch_mask.shape[0]).reshape(-1, 1)
                y = np.arange(patch_mask.shape[1]).reshape(1, -1)
                for _ in range(num_ellipses):
                    theta = np.random.uniform(0, np.pi)
                    x0 = np.random.randint(0, patch_mask.shape[0])
                    y0 = np.random.randint(0, patch_mask.shape[1])
                    a = np.random.randint(ellipse_min_dim1, ellipse_max_dim1)
                    b = np.random.randint(ellipse_min_dim2, ellipse_max_dim2)
                    ellipse = (((x-x0)*np.cos(theta) + (y-y0)*np.sin(theta))/a)**2 + (((x-x0)*np.sin(theta) + (y-y0)*np.cos(theta))/b)**2 <= 1  # True for points inside the ellipse
                    patch_mask |= ellipse
                patch_mask = patch_mask[...,None]
            else:
                patch_mask = np.ones((coor_max_dim1 - coor_min_dim1, coor_max_dim2 - coor_min_dim2, 1), dtype=np.uint8) 

            if skip_background:
                background_area = np.sum(patch_mask & src_object_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2])
                if num_ellipses is not None:
                    patch_area = np.sum(patch_mask)
                else:
                    patch_area = patch_mask.shape[0] * patch_mask.shape[1]
                found_patch = (background_area / patch_area > min_object_pct) 
            else:
                found_patch = True
            attempts += 1
            if attempts == 200:
                if verbose:
                    print('No suitable patch found.')
                return ima_dest.copy(), ((0,0),(0,0)), None

    src = ima_src[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2]
    height, width, _ = src.shape
    if resize:
        lb, ub = resize_bounds
        scale = np.clip(np.random.normal(1, 0.5), lb, ub)
        new_height = np.clip(scale * height, min_width_dim1, max_width_dim1)
        new_width = np.clip(int(new_height / height * width), min_width_dim2, max_width_dim2)
        new_height = np.clip(int(new_width / width * height), min_width_dim1, max_width_dim1)  # in case there was clipping
        if src.shape[2] == 1:  # grayscale
            src = cv2.resize(src[..., 0], (new_width, new_height))
            src = src[...,None]
        else:
            src = cv2.resize(src, (new_width, new_height))
        height, width, _ = src.shape
        patch_mask = cv2.resize(patch_mask[...,0], (width, height))
        patch_mask = patch_mask[...,None]

    if skip_background:
        src_object_mask = cv2.resize(src_object_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2, 0], (width, height))
        src_object_mask = src_object_mask[...,None]
    
    # sample destination location and size
    if shift:
        found_center = False
        attempts = 0
        while not found_center:
            valid_min_dim1 = height//2 + 1
            valid_max_dim1 = ima_dest.shape[0] - height//2 - 1
            valid_min_dim2 = width//2 + 1
            valid_max_dim2 = ima_dest.shape[1] - width//2 - 1
            
            if valid_min_dim1 >= valid_max_dim1 or valid_min_dim2 >= valid_max_dim2:
                found_center = True
                coor_min_dim1, coor_max_dim1 = center_dim1 - height//2, center_dim1 + (height+1)//2
                coor_min_dim2, coor_max_dim2 = center_dim2 - width//2, center_dim2 + (width+1)//2
                break
            
            if dest_object_mask is not None:
                shift_center_attempts = 0
                shift_center_found = False
                while shift_center_attempts < 50:
                    center_dim1 = np.random.randint(valid_min_dim1, valid_max_dim1)
                    center_dim2 = np.random.randint(valid_min_dim2, valid_max_dim2)
                    
                    if dest_object_mask[center_dim1, center_dim2, 0] > 0:
                        shift_center_found = True
                        break
                    shift_center_attempts += 1
                
                if not shift_center_found:
                    found_center = True
                    coor_min_dim1, coor_max_dim1 = center_dim1 - height//2, center_dim1 + (height+1)//2
                    coor_min_dim2, coor_max_dim2 = center_dim2 - width//2, center_dim2 + (width+1)//2
                    break
            else:
                center_dim1 = np.random.randint(valid_min_dim1, valid_max_dim1)
                center_dim2 = np.random.randint(valid_min_dim2, valid_max_dim2)
            
            coor_min_dim1, coor_max_dim1 = center_dim1 - height//2, center_dim1 + (height+1)//2
            coor_min_dim2, coor_max_dim2 = center_dim2 - width//2, center_dim2 + (width+1)//2

            if skip_background: 
                src_and_dest = dest_object_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] & src_object_mask & patch_mask
                src_or_dest = (dest_object_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] | src_object_mask) & patch_mask
                found_center = (np.sum(src_object_mask) / (patch_mask.shape[0] * patch_mask.shape[1]) > min_object_pct and    # contains object
                            np.sum(src_and_dest) / np.sum(src_object_mask) > min_overlap_pct)                    # object overlaps src object
            else:
                found_center = True
            attempts += 1
            if attempts == 200:
                if verbose:
                    print('No suitable center found. Dims were:', width, height)
                return ima_dest.copy(), ((0,0),(0,0)), None
            
    # blend
    if skip_background:
        dest_region_mask = dest_object_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2]
        patch_mask = patch_mask & dest_region_mask
        
        if np.sum(patch_mask) < 50: 
            if verbose:
                print('Patch area too small after foreground intersection.')
            return ima_dest.copy(), ((0,0),(0,0)), None

    if mode == 'swap':
        patchex = ima_dest.copy()
        before = patchex[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2]
        patchex[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] -= patch_mask * before
        patchex[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] += patch_mask * src
    elif mode == 'uniform':
        patchex = 1.0 * ima_dest
        before = patchex[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2]
        patchex[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] -= factor * patch_mask * before
        patchex[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] += factor * patch_mask * src
        patchex = np.uint8(np.floor(patchex))
    elif mode in [cv2.NORMAL_CLONE, cv2.MIXED_CLONE]:  # poisson interpolation
        int_factor = np.uint8(np.ceil(factor * 255))
        # add background to patchmask to avoid artefacts
        if skip_background:
            patch_mask_scaled = int_factor * (patch_mask | ((1 - src_object_mask) & (1 - dest_object_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2])))
        else:
            patch_mask_scaled = int_factor * patch_mask
        patch_mask_scaled[0], patch_mask_scaled[-1], patch_mask_scaled[:,0], patch_mask_scaled[:,-1] = 0, 0, 0, 0  # zero border to avoid artefacts
        center = (coor_max_dim2 - (coor_max_dim2 - coor_min_dim2) // 2, coor_min_dim1 + (coor_max_dim1 - coor_min_dim1) // 2)  # height dim first
        if np.sum(patch_mask_scaled > 0) < 50:  # cv2 seamlessClone will fail if positive mask area is too small
            return ima_dest.copy(), ((0,0),(0,0)), None
        try:
            if ima_dest.shape[2] == 1:  # grayscale
                # pad to 3 channels as that's what OpenCV expects
                src_3 = np.concatenate((src, np.zeros_like(src), np.zeros_like(src)), axis=2)
                ima_dest_3 = np.concatenate((ima_dest, np.zeros_like(ima_dest), np.zeros_like(ima_dest)), axis=2)
                patchex = cv2.seamlessClone(src_3, ima_dest_3, patch_mask_scaled, center, mode)
                patchex = patchex[...,0:1]  # extract first channel
            else:  # RGB
                patchex = cv2.seamlessClone(src, ima_dest, patch_mask_scaled, center, mode)
        except cv2.error as e:
            print('WARNING, tried bad interpolation mask and got:', e)
            return ima_dest.copy(), ((0,0),(0,0)), None
    else:
        raise ValueError("mode not supported" + str(mode))

    return patchex, ((coor_min_dim1, coor_max_dim1), (coor_min_dim2, coor_max_dim2)), patch_mask
texture_list = ['carpet', 'leather', 'grid',
                'tile', 'wood']

# Class mapping from original names to display names used in prompts.
class_mapping = {
    "macaroni1": "curved macaroni pasta",
    "macaroni2": "elbow macaroni",
    "pipe_fryum": "pipe fryum",
    "chewinggum": "chewing gum",
    "metal_nut": "metal nut",
    # Example for PCBs if needed
    "pcb1": "printed circuit board",
    "pcb2": "printed circuit board",
    "pcb3": "printed circuit board",
    "pcb4": "PCB with mixed electronic components",
}

# --- Base State Descriptors ---
state_normal_descriptors = [
    "normal {}",
    "flawless {}",
    "perfect {}",
    "unblemished {}",
]

state_anomaly_descriptors = [
    "damaged {}",
    "abnormal {}",
    "imperfect {}",
    "blemished {}",

]

# --- Class-Specific State Descriptors ---
# This dictionary now uses the ORIGINAL class names as keys.
class_state_specific_abnormal = {
    # MVTec
    'bottle': ['broken {}', 'cracked {}', 'contaminated {}', 'deformed {}', 'label defect on {}', 'leaking {}'],
    'cable': ['bent {}', 'cut {}', 'poked {}', 'missing insulation on {}', 'exposed wire in {}', 'twisted {}'],
    'capsule': ['cracked {}', 'faulty print on {}', 'scratched {}', 'dented {}', 'poked {}', 'discolored {}', 'faded print on {}'],
    'carpet': ['cut in {}', 'stain on {}', 'loose thread on {}', 'hole in {}', 'discolored patch on {}'],
    'grid': ['broken {}', 'bent {}', 'glue on {}', 'metal stain on {}', 'thread on {}', 'disjointed {}'],
    'hazelnut': ['cracked {}', 'chipped {}', 'hole in {}', 'discoloration on {}', 'cut on {}', 'mold on {}'],
    'leather': ['cut in {}', 'stain on {}', 'fold on {}', 'poked hole in {}', 'glue on {}', 'discolored {}', 'tear on {}'],
    'metal_nut': ['bent {}', 'scratched {}', 'flipped {}', 'color defect on {}', 'stain on {}'],
    'pill': ['cracked {}', 'stain on {}', 'faulty print on {}', 'scratched {}', 'chipped {}', 'discolored {}', 'torn {} wrapper'],
    'screw': ['damage to {} head', 'scratch on {}', 'damaged {} thread', 'bent {}', 'uneven {} head'],
    'tile': ['cracked {}', 'glue on {}', 'rough patch on {}', 'oil stain on {}', 'large scratch on {}', 'chipped {} edge'],
    'toothbrush': ['bent bristles on {}', 'deformed head of {}', 'cracked handle on {}'],
    'transistor': ['bent lead on {}', 'cut lead on {}', 'missing lead on {}', 'damaged top of {}', 'misplaced {}'],
    'wood': ['stain on {}', 'hole in {}', 'scratched {}', 'liquid stain on {}', 'color defect on {}', 'cracked {}'],
    'zipper': ['broken teeth on {}', 'fabric defect on {}', 'split {}', 'squeezed teeth on {}', 'discolored {}', 'teeth misaligned on {}'],
    # VisA
    'candle': ['melt {}', 'extra wax on {}', 'missing wax on {}', 'weird wick on {}', 'stain {}', 'damage {} corner'],
    'capsules': ['scratch {}', 'discolor {}', 'deform {}', 'leak {}', 'bubble on {}'],
    'cashew': ['break {}', 'scratch {}', 'burn {}', 'stick {}', 'spot {}'],
    'chewinggum': ['break {}', 'scratch {}', 'missing corner on {}', 'spot {}', 'crack {}'],
    'fryum': ['break {}', 'scratch {}', 'burn {}', 'stick {}', 'spot {}'],
    'macaroni1': ['spot {}', 'chip {} edge', 'scratch {}', 'break {}', 'crack {}'],
    'macaroni2': ['spot {}', 'chip {} edge', 'scratch {}', 'break {}', 'crack {}'],
    'pcb1': ['bend {}', 'scratch {}', 'missing component on {}', 'melt {}'],
    'pcb2': ['bend {}', 'scratch {}', 'missing component on {}', 'melt {}'],
    'pcb3': ['bend {}', 'scratch {}', 'missing component on {}', 'melt {}'],
    'pcb4': ['scratch {}', 'add extra component on {}', 'miss component on {}', 'misplace component on {}', 'burn {}', 'dirt on {}'],
    'pipe_fryum': ['break {}', 'scratch {}', 'burn {}', 'stick {}', 'spot {}', 'crack {}'],
}
MAX_PROMPTS_PER_STATE = 66666
inds_temp = ["an industrial photo of a {} for visual inspection"]
img_temp = ["an industrial image of the {} for anomaly detection"]
text_temp = ["a textural photo of a {} for visual inspection"]
surf_temp = ["a surface photo of the {} for anomaly detection"]


def encode_text_with_prompt_ensemble(model, objs, tokenizer, device,
                                     dataset='mvtec'):  # dataset param not used in current logic but kept for signature
    text_prompts_output = {}
    text_prompts_list_output = {}  # Returned but not populated, as per your code

    for obj_name in objs:
        # Get the display name for formatting prompts, fallback to the original name if not in mapping
        obj_display_name = class_mapping.get(obj_name, obj_name)

        # Select prompt templates based on the object type.
        if obj_name in texture_list:
            current_prompt_templates = inds_temp + text_temp + surf_temp
        else:
            current_prompt_templates = inds_temp + img_temp

        current_obj_states_for_prompting = []

        # 1. Normal State Prompts
        normal_phrases_formatted = [desc.format(obj_display_name) for desc in state_normal_descriptors]
        current_obj_states_for_prompting.append(normal_phrases_formatted)

        # 2. Abnormal State Prompts
        abnormal_phrases_base = [desc.format(obj_display_name) for desc in state_anomaly_descriptors]

        specific_abnormal_phrases = [desc.format(obj_display_name) for desc in
                                     class_state_specific_abnormal.get(obj_name, [])]

        all_abnormal_phrases = list(
            set(abnormal_phrases_base + specific_abnormal_phrases))
        current_obj_states_for_prompting.append(all_abnormal_phrases)

        text_features_for_current_obj = []
        for i in range(len(current_obj_states_for_prompting)):  # i=0 for normal, i=1 for abnormal
            current_formatted_state_descriptions = current_obj_states_for_prompting[i]

            final_prompt_sentences_for_state = []
            if not current_formatted_state_descriptions:
                print(f"Warning: No state descriptions for object '{obj_display_name}', state index {i}.")
            else:
                if not current_prompt_templates:
                    print(
                        f"Warning: No prompt templates found for object '{obj_display_name}' (texture_list check might be an issue or templates not defined). Using descriptions directly.")
                    final_prompt_sentences_for_state.extend(current_formatted_state_descriptions)
                else:
                    for template in current_prompt_templates:
                        for desc_phrase in current_formatted_state_descriptions:
                            final_prompt_sentences_for_state.append(
                                template.format(desc_phrase))

            if len(final_prompt_sentences_for_state) > MAX_PROMPTS_PER_STATE:
                final_prompt_sentences_for_state = final_prompt_sentences_for_state[:MAX_PROMPTS_PER_STATE]

            if not final_prompt_sentences_for_state:
                projection_dim = model.text_projection.shape[0] if hasattr(model,
                                                                           'text_projection') and model.text_projection is not None else 768
                class_embedding_for_state = torch.zeros(projection_dim, device=device)
                print(
                    f"Warning: No final prompts generated for object '{obj_display_name}', state index {i}. Using zero vector.")
            else:
                with torch.no_grad():
                    tokenized_sentences = tokenizer(final_prompt_sentences_for_state).to(device)
                    class_embeddings_all_sentences = model.encode_text(tokenized_sentences)
                class_embedding_for_state = class_embeddings_all_sentences.mean(dim=0)

            class_embedding_for_state /= (class_embedding_for_state.norm(p=2, dim=-1, keepdim=True) + 1e-8)  # Normalize
            text_features_for_current_obj.append(class_embedding_for_state)

        stacked_text_features = torch.stack(text_features_for_current_obj, dim=1).to(device)
        text_prompts_output[obj_name] = stacked_text_features

    return text_prompts_output, text_prompts_list_output
def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


def image_level_metrics(results, obj, metric):
    gt = results[obj]['gt_sp']
    pr = results[obj]['pr_sp']
    gt = np.array(gt)
    pr = np.array(pr)
    if metric == 'image-auroc':
        performance = roc_auc_score(gt, pr)
    elif metric == 'image-ap':
        performance = average_precision_score(gt, pr)

    return performance
    # table.append(str(np.round(performance * 100, decimals=1)))


def pixel_level_metrics(results, obj, metric):
    gt = results[obj]['imgs_masks']
    pr = results[obj]['anomaly_maps']
    gt = np.array(gt)
    pr = np.array(pr)
    if metric == 'pixel-auroc':
        performance = roc_auc_score(gt.ravel(), pr.ravel())
    elif metric == 'pixel-aupro':
        if len(gt.shape) == 4:
            gt = gt.squeeze(1)
        if len(pr.shape) == 4:
            pr = pr.squeeze(1)
        performance = cal_pro_score(gt, pr)
    return performance
