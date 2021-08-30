import random
from pathlib import Path
from collections import OrderedDict
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
from gloss_d import GlobalLossD
from dice_loss import DiceLoss
from seg_unet import UNet_pretrained, UNet
import torch.nn.functional as F

import json
import statistics
from sklearn.metrics import f1_score
import pickle
import copy

from training_utils import *
from data_augmentation_utils import DataAugmentation

torch.cuda.empty_cache()

with open('configs/preprocessing_datasets.json') as config_file:
    config_datasets = json.load(config_file)
with open('configs/seg_unet_Met.json') as config_file:
    config_seg = json.load(config_file)
with open('configs/config_encoder.json') as config_file:
    config_encoder = json.load(config_file)

#n_vol_train = config_seg['n_vol_train']
n_vol_val = config_seg['n_vol_val']

lambda_ = 0.6

n_vol_trains = [35]
seeds = [0, 10, 20, 30, 40, 50] 
pretrainings= ['baseline',
               'pretrained with USZ (global_d)',
               'pretrained with USZ (global_dminus)',
               'pretrained with HCP (global_d)',
               'pretrained with HCP (global_dminus)',
               'pretrained with USZ-HCP (global_d)',
               'pretrained with USZ-HCP (global_dminus)',
               'pretrained with HCP-Abide (global_d)',
               'pretrained with HCP-Abide (global_dminus)',
               'pretrained with USZ-MMWHS-Chaos-MedDecath Prostate (global_d)',
               'pretrained with USZ-MMWHS-Chaos-MedDecath Prostate (global_dminus)',
                'pretrained with MMWHS-HCP-Chaos-MedDecath Prostate (global_d)',
                'pretrained with MMWHS-HCP-Chaos-MedDecath Prostate (global_dminus)',
                'pretrained with HCP-ACDC-Chaos-MedDecath Prostate (global_d)',
                'pretrained with HCP-ACDC-Chaos-MedDecath Prostate (global_dminus)',
               'pretrained with USZ-HCP-Abide (global_d)',
               'pretrained with USZ-HCP-Abide (global_dminus)'
]
loss_unet = config_seg['loss_unet']

dataset = config_seg['dataset']
resize_size = config_seg['resize_size']
n_channels = config_seg['n_channels']
max_epochs = config_seg['max_epochs']
max_steps = config_seg['max_steps']
n_classes = config_seg['n_classes']
batch_size = config_seg['batch_size']
lr = config_seg['lr']
weight_pretrained = config_seg['weight_pretrained']

save_global_path = config_seg['save_global_path']


if dataset == 'Abide':
    n_classes = 15
    lab = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    weights = torch.tensor([0.025, 0.075, 0.075, 0.035, 0.035, 0.075, 0.075, 0.075, 
                                   0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075], dtype=torch.float32)
    
    total_n_volumes = 24
    n_vol_test = 12
    
elif dataset == 'CIMAS' or dataset == 'ACDC' :
    n_classes = 4
    lab = [1,2,3]
    weights = torch.tensor([0.1, 0.3, 0.3, 0.3], dtype=torch.float32)
    
    if dataset == 'CIMAS' :
        total_n_volumes = 13
        n_vol_test = 7
    else :
        total_n_volumes = 69
        n_vol_test = 30
        
if dataset == 'USZ':
    n_classes = 2
    lab = [1]
    weights = torch.tensor([0.1, 0.9], dtype=torch.float32)
    
    total_n_volumes = 56
    n_vol_test = 21
    n_vol_val = 21
        
def test_loading(config_datasets, config_seg, dataset, total_n_volumes=24, n_volumes=2, 
                       split_set = 'train',shuffle = False, idx_vols_val = None, return_idx = False):
    img_dataset = []
    mask_dataset = []
        
    idx_vols = select_random_volumes(total_n_volumes, n_volumes) 
    if idx_vols_val != None :
        assert len(idx_vols_val) + n_volumes <= total_n_volumes
        while any(item in idx_vols for item in idx_vols_val) :
            idx_vols = select_random_volumes(total_n_volumes, n_volumes) 
            
    count = -1
    
    for config_dataset in config_datasets :
        if config_dataset['Data'] == dataset :  
            for path in Path(config_dataset['savedir']+ split_set +'/').rglob('subject_*/'):
                
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
    
    return img_dataset, mask_dataset

print('Test set')
img_dataset, mask_dataset = test_loading(config_datasets, config_seg, dataset, total_n_volumes=n_vol_test, 
                                        n_volumes=n_vol_test, split_set = 'test', 
                                        shuffle = False)


for pretraining in pretrainings :
    print('pretraining : ',pretraining)
    run = -1
    for seed in seeds :
        run+= 1 
        
        print('run : ', run)
        for n_vol_train in n_vol_trains :
            print('n train : ', n_vol_train)
            # Load best trained model
            save_models_path = save_global_path + dataset + '/' + pretraining + '/batch_size_' + str(batch_size)  + \
                    '/'+ loss_unet +'_lr_'+ str(lr) + '_' + str(n_vol_train) + '_vol_in_train_'  + str(n_vol_val)  + \
                     '_vol_in_val' + '/run_' + str(run)
            best_model = UNet(config_seg)
            for path in Path(str(save_models_path) + '/save_models/').rglob('checkpoints_best_model*.pt'):
                print(path)
                checkpoint = torch.load(path, map_location=torch.device('cpu') )
                best_model.load_state_dict(checkpoint['model_state_dict'])
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            best_model.to(device)
            print("Running model %s" % config_seg["model"])
            
#             print('Using : ')
#             print(pretraining)
#             print('dataset : ',dataset)
#             print('max_epochs : ', max_epochs)
#             print('max_steps : ', max_steps)
#             print('batch_size : ', batch_size)
#             print('n_vol_train : ', n_vol_train)
#             print('n_vol_val : ', n_vol_val)
#             print('n_vol_test : ', n_vol_test)
#             print('loss unet : ', loss_unet)
#             print('lr : ', lr)
#             print('save_models_path : ', save_models_path)
#             print('current run : ', str(run) , ' with seed : ',  str(seed))
#             print("Using device: ", device)
            
            with torch.no_grad():

                best_model.eval()

                #print("Epoch {:03d}".format(best_epoch))
                batch_test_loss = []
                f1_mean = []
                recall_mean = []
                precision_mean = []
                f1_arr = []
                for i in range(len(img_dataset)) :
                    #if str(img_dataset[i])  == '../img_cropped/abide/test/subject_'+str(subj_label)+'/img.nii.gz':
                    #print(img_dataset[i])
                    vol_file= img_dataset[i]
                    mask_file = mask_dataset[i]

                    volume_data = nib.load(vol_file)
                    mask_data = nib.load(mask_file)

                    affine_volume = volume_data.affine
                    affine_mask = mask_data.affine

                    volume = volume_data.get_fdata()
                    mask = mask_data.get_fdata()

                    assert volume.shape == mask.shape
                    batch_x = torch.from_numpy(volume.transpose(2, 0, 1))
                    batch_y = torch.from_numpy(mask.transpose(2, 0, 1))
                    
                    test_batch_x = (batch_x.float().to(device)).view((-1, n_channels, *resize_size))
                    test_batch_y = (batch_y.long().to(device)).view((-1, n_channels, *resize_size))

                    pred = best_model(test_batch_x)
                    pred_labels_baseline = F.softmax(pred, dim=1) 
                    pred_labels_baseline = torch.argmax(pred_labels_baseline, dim=1)

                    f1_val = compute_f1(test_batch_y.cpu(), pred_labels_baseline.cpu(), lab)
                    recall_val = compute_recall(test_batch_y.cpu(), pred_labels_baseline.cpu(), lab)
                    precision_val = compute_precision(test_batch_y.cpu(), pred_labels_baseline.cpu(), lab)
                    f1_mean.append(np.mean(f1_val))
                    recall_mean.append(np.mean(recall_val))
                    precision_mean.append(np.mean(precision_val))
                    f1_arr.append(f1_val)
                    baseline_mean_score = np.mean(f1_val)
                    #print('f1 scores : ', f1_val)
                    #print('mean dice score : ', baseline_mean_score )

                    if loss_unet == 'crossentropy_loss' :
                        criterion = nn.CrossEntropyLoss(weights.to(device))
                        loss = criterion(pred.float(), test_batch_y.squeeze(1))
                    elif loss_unet == 'dice_loss' :
                        criterion = DiceLoss()
                        loss = criterion(pred, test_batch_y, device)
                    elif loss_unet == 'mixed_dice_crossentropy_loss' :
                        criterion = nn.CrossEntropyLoss(weights.to(device))
                        crossentropy_loss = criterion(pred.float(), test_batch_y.squeeze(1)).cpu()
                        criterion = DiceLoss()
                        dice_loss = criterion(pred, test_batch_y, device).to(device)
                    else :
                        print('ERROOOOR')

                    batch_test_loss.append(loss.item())

                test_loss = statistics.mean(batch_test_loss)
                f1_mean_epoch = np.mean(f1_mean)
                recall_mean_epoch = np.mean(recall_mean)
                precision_mean_epoch = np.mean(precision_mean)
                f1_arr_epoch = np.asarray(f1_arr)
                #print('F1 score on test set : ', f1_mean_epoch)

            infile = open(save_models_path+ '/results.pkl','rb')
            results_BN = pickle.load(infile)
#             print('test F1 before : ', results_BN['test F1'])

            results = pd.DataFrame(columns = ['model', 'n vol train', 'lr', 'loss unet',
                                                              'train F1', 'validation F1', 'test F1', 
                                                              'train loss', 'validation loss', 'test loss',
                                                              'best epoch',
                                                              'recall', 'precision'])

            results = results.append([{'model': pretraining, 'n vol train': n_vol_train, 'lr' : results_BN['lr'], 
                               'loss unet' : loss_unet,
                               'batch_size' : batch_size,
                               'train F1': results_BN['train F1'][0], 
                               'validation F1' : results_BN['validation F1'][0], 
                               'test F1' : f1_mean_epoch, 
                               'train loss' : results_BN['train loss'][0], 
                               'validation loss' : results_BN['validation loss'][0], 
                               'test loss' : test_loss, 
                               'best epoch' : results_BN['best epoch'][0], 
                               'f1_arr' : np.mean(f1_arr_epoch, axis = 0),
                               'recall' : recall_mean_epoch,
                               'precision' : precision_mean_epoch
                               }])
#             print('test F1 after : ', results['test F1'])

            results.to_pickle(save_models_path + "/results_2.pkl")
            
            