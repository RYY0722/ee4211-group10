# ----------------------------------------
# Written by Xiaoqing GUO
# ----------------------------------------

from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import multiprocessing
from skimage import io
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from datasets.transform import *

class BirdDataset(Dataset):
    def __init__(self, dataset_name, cfg, period, aug=False):
        self.dataset_name = dataset_name
        self.root_dir = os.path.join(cfg.ROOT_DIR,'data')
        self.dataset_dir = os.path.join(self.root_dir, dataset_name)
        self.rst_dir = os.path.join(self.root_dir,'results',dataset_name,'Segmentation')
        self.eval_dir = os.path.join(self.root_dir,'eval_result',dataset_name,'Segmentation')
        self.period = period
        if self.period != 'test':
            self.img_dir = os.path.join(self.dataset_dir, 'Train', 'images')
            self.ann_dir = os.path.join(self.dataset_dir, 'Train', 'segmentations')
            self.seg_dir = os.path.join(self.dataset_dir, 'Train', 'segmentations')
        else:
            self.img_dir = os.path.join(self.dataset_dir, 'Test', 'images')
            self.ann_dir = os.path.join(self.dataset_dir, 'Test', 'segmentations')
            self.seg_dir = os.path.join(self.dataset_dir, 'Test', 'segmentations')
        self.set_dir = os.path.join(self.root_dir, dataset_name)
        file_name = None
        if aug:
            file_name = self.set_dir+'/'+period+'aug.txt'
        else:
            file_name = self.set_dir+'/'+period+'.txt'
        df = pd.read_csv(file_name, names=['filename'])
        self.name_list = df['filename'].values
        self.rescale = None
        self.centerlize = None
        self.randomcrop = None
        self.randomflip = None
        self.randomrotation = None
        self.randomscale = None
        self.randomhsv = None
        self.multiscale = None
        self.totensor = ToTensor()
        self.cfg = cfg
	
        if dataset_name == 'BirdDataset':
            self.categories = ['Bird'] 
            
            self.num_categories = len(self.categories)
            assert(self.num_categories+1 == self.cfg.MODEL_NUM_CLASSES)
            self.cmap = self.__colormap(len(self.categories)+1)

        if cfg.DATA_RESCALE > 0:
            self.rescale = Rescale(cfg.DATA_RESCALE,fix=False)
            #self.centerlize = Centerlize(cfg.DATA_RESCALE)
        if 'train' in self.period:        
            if cfg.DATA_RANDOMCROP > 0:
                self.randomcrop = RandomCrop(cfg.DATA_RANDOMCROP)
            if cfg.DATA_RANDOMROTATION > 0:
                self.randomrotation = RandomRotation(cfg.DATA_RANDOMROTATION)
            if cfg.DATA_RANDOMSCALE != 1:
                self.randomscale = RandomScale(cfg.DATA_RANDOMSCALE)
            if cfg.DATA_RANDOMFLIP > 0:
                self.randomflip = RandomFlip(cfg.DATA_RANDOMFLIP)
            if cfg.DATA_RANDOM_H > 0 or cfg.DATA_RANDOM_S > 0 or cfg.DATA_RANDOM_V > 0:
                self.randomhsv = RandomHSV(cfg.DATA_RANDOM_H, cfg.DATA_RANDOM_S, cfg.DATA_RANDOM_V)
        else:
            self.multiscale = Multiscale(self.cfg.TEST_MULTISCALE)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx].split()[0]
        img_file = self.img_dir + '/' + name.split('.png')[0] + '.jpg'
        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.array(io.imread(img_file),dtype=np.uint8)
        r,c,_ = image.shape
        sample = {'image': image, 'name': name, 'row': r, 'col': c}
        
        if 'train' in self.period:
            seg_file = self.seg_dir + '/' + self.name_list[idx].split()[0]
            segmentation = np.array(Image.open(seg_file))
            (T, segmentation) = cv2.threshold(segmentation, 0, 255, cv2.THRESH_BINARY)
            sample['segmentation'] = segmentation[:,:,0]/255.
           
            if self.cfg.DATA_RANDOM_H>0 or self.cfg.DATA_RANDOM_S>0 or self.cfg.DATA_RANDOM_V>0:
                sample = self.randomhsv(sample)
            if self.cfg.DATA_RANDOMFLIP > 0:
                sample = self.randomflip(sample)
            if self.cfg.DATA_RANDOMROTATION > 0:
                sample = self.randomrotation(sample)
            if self.cfg.DATA_RANDOMSCALE != 1:
                sample = self.randomscale(sample)
            if self.cfg.DATA_RANDOMCROP > 0:
                sample = self.randomcrop(sample)
            if self.cfg.DATA_RESCALE > 0:
                #sample = self.centerlize(sample)
                sample = self.rescale(sample)
        elif 'valid' in self.period:
            seg_file = self.seg_dir + '/' + self.name_list[idx].split()[0]
            segmentation = np.array(Image.open(seg_file))
            (T, segmentation) = cv2.threshold(segmentation, 0, 255, cv2.THRESH_BINARY)
            sample['segmentation'] = segmentation[:,:,0]/255.
            if self.cfg.DATA_RESCALE > 0:
                sample = self.rescale(sample)
            sample = self.multiscale(sample)
        else:
            seg_file = self.seg_dir + '/' + self.name_list[idx].split()[0]
            if self.cfg.DATA_RESCALE > 0:
                sample = self.rescale(sample)
            sample = self.multiscale(sample)

        if 'segmentation' in sample.keys():
            sample['mask'] = sample['segmentation'] < self.cfg.MODEL_NUM_CLASSES
            #print(sample['segmentation'].max(),sample['segmentation'].shape)
            t = sample['segmentation']
            t[t >= self.cfg.MODEL_NUM_CLASSES] = 0
            #print(t.max(),t.shape)
            #print(onehot(np.int32(t),self.cfg.MODEL_NUM_CLASSES))
            sample['segmentation_onehot']=onehot(np.int32(t),self.cfg.MODEL_NUM_CLASSES)
        sample = self.totensor(sample)

        return sample

    def __colormap(self, N):
        """Get the map from label index to color

        Args:
            N: number of class

            return: a Nx3 matrix

        """
        cmap = np.zeros((N, 3), dtype = np.uint8)

        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

        for i in range(N):
            r = 0
            g = 0
            b = 0
            idx = i
            for j in range(7):
                str_id = uint82bin(idx)
                r = r ^ ( np.uint8(str_id[-1]) << (7-j))
                g = g ^ ( np.uint8(str_id[-2]) << (7-j))
                b = b ^ ( np.uint8(str_id[-3]) << (7-j))
                idx = idx >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        return cmap
    
    def label2colormap(self, label):
        m = label.astype(np.uint8)
        r,c = m.shape
        cmap = np.zeros((r,c,3), dtype=np.uint8)
        cmap[:,:,0] = (m&1)<<7 | (m&8)<<3
        cmap[:,:,1] = (m&2)<<6 | (m&16)<<2
        cmap[:,:,2] = (m&4)<<5
        return cmap
    
    def save_result(self, result_list, model_id):
        """Save test results

        Args:
            result_list(list of dict): [{'name':name1, 'predict':predict_seg1},{...},...]

        """
        i = 1
        folder_path = os.path.join(self.rst_dir,'%s_%s_cls'%(model_id,self.period))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for sample in result_list:
            file_path = os.path.join(folder_path, '%s'%sample['name'])
            cv2.imwrite(file_path, sample['predict'])
            # print('[%d/%d] %s saved'%(i,len(result_list),file_path))
            i+=1
        return folder_path