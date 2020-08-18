import torch
import random
import numpy as np
from torchvision import transforms
import scipy.misc as m
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils import data
from glob import glob

class CYDataset_MSLesion(data.Dataset):
    def __init__(self, root, split, img_size):
        self.root = root
        self.split = split
        self.num_classes = 2
        self.img_size = img_size
        self.mean = np.array([122.67892, 116.66877, 104.00699])  # BGR # Imagenet-pretrained

        self.base_glob_path = self.root + '/' + self.split + '/*/{}/*.png'

        self.modality_names = ['t1_brain_mni', 'flair_brain_mni', 'lesion_mni']
        glob_paths = {modality_name : self.base_glob_path.format(modality_name) for modality_name in self.modality_names}

        temp_paths_dict = {modality_name : glob(glob_paths[modality_name]) for modality_name in self.modality_names}

        paths_dict = {}
        for modality_name, paths in temp_paths_dict.items():
            for path in paths:
                tokens = path.split('/')[-1].split('.')[0].split('-')
                case_index = tokens[0]
                slice_index = tokens[-1].replace('.png', '')
                final_index = case_index + '-' + slice_index
                if final_index not in paths_dict:
                    paths_dict[final_index] = {}
                paths_dict[final_index][modality_name] = path

        self.dataset_paths_info_list = []
        for final_index, modality_paths in paths_dict.items():
            element = [final_index, modality_paths]
            self.dataset_paths_info_list.append(element)

        print('[CYDataset_MSLesion] {} dataset constructed: {} slices from {} cases'.format(self.split,
                                                                                             len(self.dataset_paths_info_list),
                                                                                             len(set([final_index.split('-')[0] for final_index in paths_dict.keys()]))))
        print('[CYDataset_MSLesion] glob_base_path: {}'.format(self.base_glob_path))
        print('[CYDataset_MSLesion] modality_names: {}'.format(self.modality_names))

    def __len__(self):
        return len(self.dataset_paths_info_list)

    def joint_augmentation(self, imgs):
        if random.random() < 0.5:
            imgs = {modality_name: img.transpose(Image.FLIP_LEFT_RIGHT) for modality_name, img in imgs.items()}
        if random.random() < 0.5:
            imgs = {modality_name: img.transpose(Image.FLIP_TOP_BOTTOM) for modality_name, img in imgs.items()}
        if random.random() < 0.5:
            imgs = {modality_name: img.transpose(Image.TRANSPOSE) for modality_name, img in imgs.items()}
        temp = random.random()
        if temp < 0.25:
            imgs = {modality_name: img.transpose(Image.ROTATE_90) for modality_name, img in imgs.items()}
        elif temp < 0.5:
            imgs = {modality_name: img.transpose(Image.ROTATE_180) for modality_name, img in imgs.items()}
        elif temp < 0.75:
            imgs = {modality_name: img.transpose(Image.ROTATE_270) for modality_name, img in imgs.items()}
        return imgs

    def __getitem__(self, index):
        dataset_paths_info = self.dataset_paths_info_list[index]
        final_index = dataset_paths_info[0]
        paths_dict = dataset_paths_info[1]

        imgs = {modality_name: Image.open(path) for modality_name, path in paths_dict.items()}
        
        if self.split == 'train':
            imgs = self.joint_augmentation(imgs)

        np_arrays = {modality_name: np.array(img) for modality_name, img in imgs.items()}
        np_arrays[self.modality_names[0]] = np_arrays[self.modality_names[0]].astype(np.uint8)
        np_arrays[self.modality_names[1]] = np_arrays[self.modality_names[1]].astype(np.uint8)
        np_arrays[self.modality_names[2]] = np_arrays[self.modality_names[2]].astype(np.int32)
        np_arrays[self.modality_names[2]] = np.where(np_arrays[self.modality_names[2]] > 127, 255, 0)

        transformed_tensors = self.transform(np_arrays)

        X = torch.cat((transformed_tensors[self.modality_names[0]], transformed_tensors[self.modality_names[1]]), 0)
        Y = transformed_tensors[self.modality_names[2]]

        return X, Y, final_index


    def transform(self, imgs):
        
        currH, currW = imgs['t1_brain_mni'].shape[0], imgs['t1_brain_mni'].shape[1]
        currH = (self.img_size-currH)//2
        currW = (self.img_size-currW)//2
        padded_imgs = {}
        padded_imgs[self.modality_names[0]] = np.pad(imgs[self.modality_names[0]], ((currH, currH), (currW, currW), (0, 0)), mode='constant', constant_values=0)
        padded_imgs[self.modality_names[1]] = np.pad(imgs[self.modality_names[1]], ((currH, currH), (currW, currW), (0, 0)), mode='constant', constant_values=0)
        padded_imgs[self.modality_names[2]] = np.pad(imgs[self.modality_names[2]], ((currH, currH), (currW, currW)), mode='constant', constant_values=0)
        
        #resized_imgs = {modality_name: m.imresize(img, (self.img_size, self.img_size), 'nearest') for modality_name, img in padded_imgs.items()}
        resized_imgs = padded_imgs

        resized_imgs[self.modality_names[0]] = (2 * (resized_imgs[self.modality_names[0]].astype(np.float64)
                                                     - self.mean)).transpose(2, 0, 1)
        resized_imgs[self.modality_names[1]] = (2 * (resized_imgs[self.modality_names[1]].astype(np.float64)
                                                     - self.mean)).transpose(2, 0, 1)
        normalized_imgs = {modality_name: img / 255 for modality_name, img in resized_imgs.items()}

        tensors = {modality_name: torch.from_numpy(img).float() for modality_name, img in normalized_imgs.items()}

        return tensors
