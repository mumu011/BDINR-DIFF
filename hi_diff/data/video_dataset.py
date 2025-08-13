import os
from os.path import join as opj
from torch.utils import data as data
import numpy as np
import cv2
import torch

from basicsr.utils.registry import DATASET_REGISTRY

def vid_transform(vid, prob=0.5, tform_op=['all']):
    """
    video data vid_transform (data augment) with a $op chance

    Args:
        vid ([ndarray]): [shape: N*H*W*C]
        prob (float, optional): [probility]. Defaults to 0.5.
        op (list, optional): ['flip' | 'rotate' | 'reverse']. Defaults to ['all'].
    """
    if 'flip' in tform_op or 'all' in tform_op:
        # flip left-right or flip up-down
        if np.random.rand() < prob:
            vid = vid[:, :, ::-1, :]
        if np.random.rand() < prob:
            vid = vid[:, ::-1, :, :]
    if 'rotate' in tform_op or 'all' in tform_op:
        # rotate 90 / -90 degrees
        if prob/4 < np.random.rand() <= prob/2:
            np.transpose(vid, axes=(0, 2, 1, 3))[:, ::-1, ...]  # -90
        elif prob/2 < np.random.rand() <= prob:
            vid = np.transpose(
                vid[:, ::-1, :, :][:, :, ::-1, :], axes=(0, 2, 1, 3))[:, ::-1, ...]  # 90

    if 'reverse' in tform_op or 'all' in tform_op:
        if np.random.rand() < prob:
            vid = vid[::-1, ...]
    
    return vid.copy()

@DATASET_REGISTRY.register()
class VideoFrame_Dataset(data.Dataset):
    def __init__(self, opt):
        super(VideoFrame_Dataset, self).__init__()
        self.opt = opt
        self.data_dir = opt['dataroot']
        self.sigma_range = opt['sigma_range']
        self.patch_sz = [opt['patch_sz']] * 2 if isinstance(opt['patch_sz'], int) else opt['patch_sz']
        self.tform_op = opt['tform_op']
        self.vid_length = opt['frame_num']
        self.img_paths = []
        self.vid_idx = []
        self.stride = opt['stride']

        # get image paths
        img_nums = []
        vid_paths = []
        if isinstance(self.data_dir, str):
            # single dataset
            vid_names = sorted(os.listdir(self.data_dir))
            vid_paths = [opj(self.data_dir, vid_name) for vid_name in vid_names]
            if all(os.path.isfile(vid_path) for vid_path in vid_paths):
                # data_dir is an image dir rather than a vid dir
                vid_paths = [self.data_dir]
        else:
            # multiple dataset
            for data_dir_n in sorted(self.data_dir):
                vid_names_n = sorted(os.listdir(data_dir_n))
                vid_paths_n = [opj(data_dir_n, vid_name_n)
                               for vid_name_n in vid_names_n]
                vid_paths.extend(vid_paths_n)

        for vid_path in vid_paths:
            img_names = sorted(os.listdir(vid_path))
            img_nums.append(len(img_names))
            self.img_paths.extend(
                [opj(vid_path, img_name) for img_name in img_names])

        counter = 0
        for img_num in img_nums:
            self.vid_idx.extend(
                list(range(counter, counter+img_num-self.vid_length+1, self.stride)))
            counter = counter+img_num

    def __getitem__(self, idx):
        # load video frames
        vid = []
        for k in range(self.vid_idx[idx], self.vid_idx[idx]+self.vid_length):
            # read image
            img = cv2.imread(self.img_paths[k])
            assert img is not None, 'Image read falied'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.patch_sz:
                if k == self.vid_idx[idx]:
                    # set the random crop point
                    img_sz = img.shape
                    assert (img_sz[0] >= self.patch_sz[0]) and (img_sz[1] >= self.patch_sz[1]
                                                                ), 'error PATCH_SZ larger than image size'
                    xmin = np.random.randint(0, img_sz[1]-self.patch_sz[1])
                    ymin = np.random.randint(0, img_sz[0]-self.patch_sz[0])

                # crop to patch size
                img_crop = img[ymin:ymin+self.patch_sz[0],
                               xmin:xmin+self.patch_sz[1], :]
            else:
                img_crop = img

            vid.append(img_crop)

        # list2ndarray, shape [vid_num, h, w, c], value 0-255
        vid = np.array(vid)  

        # data augment
        if self.tform_op:
            vid = vid_transform(vid, tform_op=self.tform_op)

        # add noise
        if isinstance(self.sigma_range, (int, float)):
            noise_level = self.sigma_range
        else:
            noise_level = np.random.uniform(*self.sigma_range)
        assert 0 <= noise_level <= 1, f'noise level (sigma_range) should be within 0-1, but get {self.sigma_range}'
        if noise_level>0:
            image_dtype = vid.dtype
            image_maxv = np.iinfo(image_dtype).max  # 8/16 bit image -> 255/65535
            vid = vid + np.random.normal(0, image_maxv*noise_level, vid.shape)
            vid = vid.clip(0, image_maxv).astype(image_dtype)

        vid = vid.transpose(0, 3, 1, 2)

        return vid

    def __len__(self):
        return len(self.vid_idx)

