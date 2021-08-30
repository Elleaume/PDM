import random
import itertools
from pathlib import Path
import time
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import nibabel as nib
from torch.utils.data import DataLoader
from tqdm import tqdm

from decoder_pretrain import DecoderPretrainNet
from encoder_pretrain import EncoderPretrainNet
from gloss_dminus import GlobalLossDminus
from seg_unet import UNet_pretrained, UNet
import torch.nn.functional as F

import json
import statistics
from sklearn.metrics import f1_score, precision_score, recall_score
import pickle


class TrainDataset(torch.utils.data.Dataset) :
    def __init__(self, config, volumes, masks, dataset = None):
        self.path_volumes = volumes
        self.path_masks = masks
        #self.n_vols = 1
        
        for i in range(len(self.path_volumes)):
            vol_file = self.path_volumes[i]
            mask_file = self.path_masks[i]

            volume = nib.load(vol_file).get_fdata()
            mask = nib.load(mask_file).get_fdata()
            
            assert volume.shape == mask.shape

            new_img = volume.transpose(2, 0, 1)
            new_mask = mask.transpose(2, 0, 1)
            
            if i == 0: 
                self.imgs = new_img
                self.masks = new_mask
            else :
                self.imgs = np.concatenate((self.imgs, new_img), axis=0)
                self.masks = np.concatenate((self.masks, new_mask), axis=0)
                
        
        if dataset == 'USZ' :
            # Make a balance set of tumors and healthy images
            indexes_tumors_slices = []
            for i in range(self.imgs.shape[0]) :
                if 1 in self.masks[i] :
                    indexes_tumors_slices.append(i)
            
            self.imgs_tumors = self.imgs[indexes_tumors_slices,:,:]
            self.masks_tumors = self.masks[indexes_tumors_slices,:,:]
            
            indexes = np.arange(0,self.imgs.shape[0], dtype=np.int32)
            indexes_healthy = np.delete(indexes,indexes_tumors_slices)
            
            self.imgs_healthy = self.imgs[indexes_healthy,:,:]
            self.masks_healthy = self.masks[indexes_healthy,:,:]
            
            indexes_balance_healthy = random.sample(range(0, self.imgs_healthy.shape[0]), len(indexes_tumors_slices))
            
            self.imgs_balance_healthy = self.imgs_healthy[indexes_balance_healthy,:,:]
            self.masks_balance_healthy = self.masks_healthy[indexes_balance_healthy,:,:]
            
            self.imgs = np.concatenate((self.imgs_balance_healthy , self.imgs_tumors ), axis=0)
            self.masks = np.concatenate((self.masks_balance_healthy, self.masks_tumors), axis=0)
            
            
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.masks[idx]
    
def initialize_dataset(config_datasets, config_seg, dataset, total_n_volumes=24, n_volumes=2, 
                       split_set = 'train',shuffle = False, idx_vols_val = None, return_idx = False) :
            
    img_dataset = []
    mask_dataset = []
        
    idx_vols = select_random_volumes(total_n_volumes, n_volumes) 
    if idx_vols_val != None :
        assert len(idx_vols_val) + n_volumes <= total_n_volumes
        while any(item in idx_vols for item in idx_vols_val) :
            idx_vols = select_random_volumes(total_n_volumes, n_volumes) 
            
    count = -1
    data = None
    for config_dataset in config_datasets :
        if config_dataset['Data'] == dataset :  
            for path in Path(config_dataset['savedir']+ split_set +'/').rglob('subject_*/'):
                
                if dataset == 'USZ' :
                    if "nii.gz" in str(path) or ".png" in str(path) :
                        continue
                    #count += 1   
                    #if count >= n_volumes :
                    #    break
                        
                    # Add the image and the corresponding mask to the datasets
                    for path_image in path.rglob("img.nii.gz") :
                        img_dataset.append(path_image)
                        print(path_image)
                    for path_mask in path.rglob("mask.nii.gz") :
                        mask_dataset.append(path_mask)
                    
                    if split_set == 'train' or split_set == 'validation': 
                        data = dataset
                    
                else :
                    # We want total path not individual path of images
                    if "nii.gz" in str(path) or ".png" in str(path) :
                        continue
                    count += 1   

                    # Different criterion to stop adding train or test volumes
                    if split_set == 'test':
                        if count >= n_volumes :
                            break
                    else :
                        if count not in idx_vols :
                            continue

                    # Add the image and the corresponding mask to the datasets
                    for path_image in path.rglob("img.nii.gz") :
                        img_dataset.append(path_image)
                        print(path_image)
                    for path_mask in path.rglob("mask.nii.gz") :
                        mask_dataset.append(path_mask)
    
    # initalize the dataset
    dataset = TrainDataset(config_seg, img_dataset, mask_dataset, data)
    dataset_loader = DataLoader(dataset,
                                num_workers=1,
                                batch_size=config_seg['batch_size'],
                                shuffle=shuffle,
                                pin_memory=True,
                                drop_last=False)
    if return_idx :
        return dataset_loader, idx_vols
    else :
        return dataset_loader
            

def compute_f1(y_true, y_pred, lab):

    y_pred= y_pred.flatten()
    y_true= y_true.flatten()
    f1_val= f1_score(y_true, y_pred, labels=lab,zero_division=0,average=None)

    return f1_val

def compute_recall(y_true, y_pred, lab):

    y_pred= y_pred.flatten()
    y_true= y_true.flatten()
    f1_val= recall_score(y_true, y_pred, labels=lab,zero_division=0,average=None)

    return f1_val

def compute_precision(y_true, y_pred, lab):

    y_pred= y_pred.flatten()
    y_true= y_true.flatten()
    f1_val= precision_score(y_true, y_pred, labels=lab,zero_division=0,average=None)

    return f1_val

def random_gen(low, high):
    while True:
        yield random.randrange(low, high)
        
def select_random_volumes(n_total, n_train) :
    assert n_total >= n_train
    gen = random_gen(0, n_total)
    items = []
    # Try to add elem to set until set length is less than 10
    for x in itertools.takewhile(lambda x: len(items) < n_train, gen):
        if x not in items :
            items.append(x)
    
    return items

def double_std(array):
    return np.std(array) * 2

def plot_losses(save_directory):
    
    save_models = save_directory + '/save_models/'
    infile = open(save_directory + '/losses.pkl','rb')
    losses = pickle.load(infile)
    
    losses_min = losses.loc[losses['validation loss'] == losses['validation loss'].min()]

    plt.figure()
    losses.plot(x = 'epoch', y =['train loss','validation loss'],mark_right=False)
    plt.savefig(save_directory + '/loss.png')

    plt.figure()