import torch
import numpy as np
import skimage.io as io
import skimage.transform as trans
import os
from torch.utils.data import Dataset

class CelebA(Dataset):
    def __init__(self, gt_root, g_root, mask_root, files, img_size=(128, 128), augmentation=False):
        self.files = files
        self.img_size = img_size
        self.gt_root = gt_root
        self.g_root = g_root
        self.mask_root = mask_root

        self.augmentation = augmentation

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """
        The output image should including gt, g, and mask
        
        """
        # print("11")
        filename = self.files[index]
        filename = os.path.basename(filename)
        img_path = os.path.join(self.g_root, filename)
        gt_path = os.path.join(self.gt_root, filename.split('.')[0]+'.jpg')
        mask_path = os.path.join(self.mask_root, filename)
        
        gt_data = io.imread(gt_path)
        img_data = io.imread(img_path)
        mask_data = io.imread(mask_path, as_gray=True)
        mask_data = np.expand_dims(mask_data, axis=2)

        # Normalize data 
        mask_data = mask_data / 255
        gt_data  = gt_data  / 127.5 - 1.0
        img_data = img_data / 127.5 - 1.0
        
        # mask_data = (gt_data != img_data) * 1.0  # generate the mask. 
        # mask_data = np.sum(mask_data, axis=-1, keepdims=True) / 3
        # io.imsave("./Debug/gt_data_before_resize.png", gt_data)
        # io.imsave("./Debug/img_data_before_resize.png", img_data)
        # io.imsave("./Debug/mask_data_before_resize.png", mask_data)

        # resize
        comp_data = np.concatenate([img_data, mask_data, gt_data], axis=-1)
        comp_data = trans.resize(comp_data, self.img_size, order=0)

        # img_data_, mask_data_, gt_data_ = np.split(comp_data, [3, 4], axis=-1)
        # io.imsave("./Debug/gt_data_after_resize.png", gt_data_)
        # io.imsave("./Debug/img_data_after_resize.png", img_data_)
        # io.imsave("./Debug/mask_data_after_resize.png", mask_data_)

        # transform:
        if self.augmentation:
            # rotation
            degree = 90 * np.random.choice([0, 1, 2, 3], 1)[0]
            comp_data = trans.rotate(comp_data, degree)

        img_data_, mask_data_, gt_data_ = np.split(comp_data, [3, 4], axis=-1)
        # io.imsave("./Debug/gt_data_after_augmentation.png", gt_data_)
        # io.imsave("./Debug/img_data_after_augmentation.png", img_data_)
        # io.imsave("./Debug/mask_data_after_augmentation.png", mask_data_)
        g_in, gt_data = np.split(comp_data, [4], axis=-1)
        # io.imsave("./Debug/g_in_after_augmentation_RGBA.png", img_data_ * mask_data_)
        
        return g_in, gt_data, mask_data_

    
