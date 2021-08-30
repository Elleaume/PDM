import random
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
with open('configs/seg_unet.json') as config_file:
    config_seg = json.load(config_file)
with open('configs/config_encoder.json') as config_file:
    config_encoder = json.load(config_file)

n_vol_train = config_seg['n_vol_train']
n_vol_val = config_seg['n_vol_val']
#n_vol_test = config_seg['n_vol_test']
lambda_ = 0.6

n_vol_trains = [5]
seeds = [1, 11, 23, 32, 37, 45, 67, 89, 101, 267]# ] # 
options = ['option_7']#'option_3']#, ] 
losses_unet = [config_seg['loss_unet']] #"mixed_dice_crossentropy_loss"] #"dice_loss", 

dataset = config_seg['dataset']
resize_size = config_seg['resize_size']
n_channels = config_seg['n_channels']
max_epochs = config_seg['max_epochs']
n_classes = config_seg['n_classes']
batch_size = config_seg['batch_size']
lr = config_seg['lr']
weight_pretrained = config_seg['weight_pretrained']

save_global_path = config_seg['save_global_path']

if dataset == 'Abide':
    n_classes = 15
    lab = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    weights = torch.tensor([0.025, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 
                                   0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075], dtype=torch.float32)
    
    total_n_volumes = 18
    n_vol_test = 18
    
elif dataset == 'CIMAS' or dataset == 'ACDC' :
    n_classes = 4
    lab = [1,2,3]
    weights = torch.tensor([0.1, 0.3, 0.3, 0.3], dtype=torch.float32)
    
    if dataset == 'CIMAS' :
        total_n_volumes = 10
        n_vol_test = 10
    else :
        total_n_volumes = 24
        n_vol_test = 50

        
print('Test set')
dataset_loader_test = initialize_dataset(config_datasets, config_seg, dataset, total_n_volumes=n_vol_test, 
                                         n_volumes=n_vol_test, split_set = 'test', 
                                         shuffle = False)

data_aug = DataAugmentation()
for loss_unet in losses_unet :
    run = -1
    for seed in seeds :
        run += 1
        torch.manual_seed(seed)
        # Load losses Pretrained encoder
        for n_vol_train in n_vol_trains:

            # Initialization of the datasets
            if dataset == 'CIMAS' or dataset == 'Abide' :
                random.seed(seed)
                print('Validation set')
                random.seed(seed)
                dataset_loader_validation, idx_vols_val = initialize_dataset(config_datasets, config_seg, dataset, 
                                                               total_n_volumes=total_n_volumes, 
                                                               n_volumes=n_vol_val, split_set = 'train', 
                                                               shuffle = False, return_idx = True)
                random.seed(seed)
                print('Training set')
                dataset_loader_train = initialize_dataset(config_datasets, config_seg, dataset, 
                                                          total_n_volumes=total_n_volumes, 
                                                          n_volumes=n_vol_train, split_set = 'train', 
                                                          shuffle = True, idx_vols_val = idx_vols_val)
            else :
                # Initialization of the datasets
                print('Training set')
                random.seed(seed)
                dataset_loader_train = initialize_dataset(config_datasets, config_seg, dataset, 
                                                          total_n_volumes=total_n_volumes, 
                                                          n_volumes=n_vol_train, split_set = 'train', 
                                                          shuffle = True)
                print('Validation set')
                random.seed(seed)
                dataset_loader_validation = initialize_dataset(config_datasets, config_seg, dataset, 
                                                               total_n_volumes=total_n_volumes, 
                                                               n_volumes=n_vol_val, split_set = 'validation', 
                                                               shuffle = False)
            
            for option in options :
                
                # baseline 
                if option == 'baseline' :
                    pretraining = 'baseline'
                    
                # ACDC pretraining
                elif option == 'option_1':
                    save_models_pretrain = save_global_path +  "pretraining_FullTrainSet" + \
                                           "/global_dminus/1datasets_12volumesPerBatch_2transforms_tempfac1/"
                    pretraining = 'pretrained with ACDC FullTrainSet (global_dminus)' 
                elif option == 'option_2':
                    save_models_pretrain = save_global_path +  "pretraining_FullTrainSet" + \
                                           "/global_d/1datasets_12volumesPerBatch_2transforms_tempfac1/"
                    pretraining = 'pretrained with ACDC FullTrainSet (global_d)' 
                elif option == 'option_3':
                    save_models_pretrain = save_global_path +  "pretraining_FullTrainSet" + \
                                           "/representation_loss/1datasets_12volumesPerBatch_1transforms_lb0.5_tempfac1/"
                    pretraining = 'pretrained with ACDC FullTrainSet (representation_loss)' 
                
                # 4 datasets in pretraining
                elif option == 'option_4':
                    save_models_pretrain = save_global_path + "pretraining_FullTrainSet" + \
                    "/global_dminus/4datasets_3volumesPerBatch_2transforms_tempfac1/"
                    pretraining = 'pretrained with 4 datasets (global_dminus)' 
                elif option == 'option_5':
                    save_models_pretrain = save_global_path + "pretraining_FullTrainSet" + \
                    "/global_d/4datasets_3volumesPerBatch_2transforms_tempfac1/"
                    pretraining = 'pretrained with 4 datasets (global_d)' 
                elif option == 'option_6':
                    save_models_pretrain = save_global_path + "pretraining_FullTrainSet" + \
                    "/representation_loss/4datasets_3volumesPerBatch_1transforms_lb0.5_tempfac1/"
                    pretraining = 'pretrained with 4 datasets (representation_loss)' 
                elif option == 'option_7':
                    save_models_pretrain = save_global_path + "pretraining_FullTrainSet" + \
                    "/2_steps_global_d/4datasets_3volumesPerBatch_2transforms_lb0.5_tempfac1/"
                    pretraining = 'pretrained with 4 datasets (2 steps global_d)' 
                
                else :
                    print('Error : Unvalid option of baseline / pretraining')
                    break
            
                save_directory = save_global_path + dataset + '/' + pretraining + '/'+ loss_unet +'_lr_'+ str(lr) + \
                '_' + str(n_vol_train) + '_vol_in_train_'  + str(n_vol_val) + '_vol_in_val' + '/run_' \
                + str(run)

                save_models = save_directory + '/save_models/'  
                Path(save_models).mkdir(parents=True, exist_ok=True)
                losses = pd.DataFrame(columns = ['epoch', 'train loss', 'validation loss', 'train dice', 'validation dice'])

                print('Using : ')
                print(pretraining)
                print('dataset : ',dataset)
                print('max_epochs : ', max_epochs)
                print('batch_size : ', batch_size)
                print('n_vol_train : ', n_vol_train)
                print('n_vol_val : ', n_vol_val)
                print('n_vol_test : ', n_vol_test)
                print('loss unet : ', loss_unet)
                print('lr : ', lr)
                print('save directory : ', save_directory)

                # Load only running_mean and running_var in batch
                if option == 'baseline' : 
                    model = UNet(config_seg)

                # Re init mean and var Batch norm
                if option != 'baseline':
                    
                    #infile = open(save_models_pretrain + 'losses.pkl','rb')
                    #losses_pretrained = pickle.load(infile)
                    epoch = 499#losses_pretrained.loc[losses_pretrained['validation loss'] == \
                            #                      losses_pretrained['validation loss'].min()]['epoch'].item()
                    print('dir_pretrained_net', save_models)
                    # Initialize encoder
                    model_encoder = EncoderPretrainNet(config_encoder)

                    # Load best pretrained model
                    name = str(save_models_pretrain)+ '/save_models/checkpoints_' + str(epoch) + '.pt'
                    checkpoint = torch.load(name, map_location=torch.device('cpu') )
                    model_encoder.load_state_dict(checkpoint['model_state_dict'])

                    model = UNet_pretrained(config_seg, model_encoder)
                    
                    # Set running_mean to 0 and running_var to 1 (batch norm) in encoder
                    for name, module in model.named_modules():
                        if 'encoder' in name :
                            #print(name)
                            if isinstance(module, nn.BatchNorm2d):
                                module.reset_running_stats()
                                
                # choose specified optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=config_seg["lr"])

                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                print("Using device: ", device)
                model.to(device)
                print("Running model %s" % config_seg["model"])

                # Training

                min_loss = 999999
                best_model = UNet(config_seg)
                best_optimizer= torch.optim.Adam(model.parameters(), lr=config_seg["lr"])

                for epoch in range(max_epochs) :
                    print("Doing Train...")
                    print("Epoch {:03d}".format(epoch))

                    model.train()
                    batch_train_loss = []
                    f1_mean = []
                    f1_arr = []
                    for id_batch, (batch_x, batch_y) in enumerate(tqdm(dataset_loader_train)):

                        optimizer.zero_grad(set_to_none=True)

                        train_batch_x = (batch_x.float().to(device)).view((-1, n_channels, *resize_size)) 
                        train_batch_y = (batch_y.float().to(device)).view((-1, n_channels, *resize_size))
                        #print('before augmentation : ', torch.unique(train_batch_y))
                        # perform data augmentation
                        train_batch_x, train_batch_y = data_aug(train_batch_x, train_batch_y)

                        train_batch_y = train_batch_y.long().to(device)
                        train_batch_y = train_batch_y.to(device)
                        #print('after augmentation : ', torch.unique(train_batch_y))

                        pred = model(train_batch_x).to(device)
                        pred_labels = F.softmax(pred, dim=1) 
                        pred_labels = torch.argmax(pred_labels, dim=1)
                        
                        if loss_unet == 'crossentropy_loss' :
                            criterion = nn.CrossEntropyLoss(weights.to(device))
                            loss = criterion(pred.float(), train_batch_y.squeeze(1)).cpu()
                        elif loss_unet == 'dice_loss' :
                            criterion = DiceLoss()
                            loss = criterion(pred, train_batch_y, device).to(device)
                        elif loss_unet == 'mixed_dice_crossentropy_loss' :
                            criterion = nn.CrossEntropyLoss(weights.to(device))
                            crossentropy_loss = criterion(pred.float(), train_batch_y.squeeze(1)).cpu()
                            criterion = DiceLoss()
                            dice_loss = criterion(pred, train_batch_y, device).to(device)
                            loss = lambda_ * dice_loss + (1-lambda_) * crossentropy_loss
                        else :
                            print('ERROOOOR')
                            break

                        batch_train_loss.append(loss.item())

                        f1_val = compute_f1(train_batch_y.cpu(), pred_labels.cpu(), lab)
                        f1_mean.append(np.mean(f1_val))
                        f1_arr.append(f1_val)

                        loss.backward()
                        optimizer.step()
                    train_loss = statistics.mean(batch_train_loss)
                    #wandb.log({"train loss": train_loss, 'epoch': steps})
                    f1_mean_epoch_train = np.mean(f1_mean)
                    f1_arr_epoch_train = np.asarray(f1_arr)
                    print('F1 score on train set : ', f1_mean_epoch_train)
                    print("Current train loss: %f" % train_loss) 

                    with torch.no_grad():
                        model.eval()
                        batch_val_loss = []
                        f1_mean = []
                        f1_arr = []
                        for id_batch, (batch_x, batch_y) in enumerate(tqdm(dataset_loader_validation)):

                            optimizer.zero_grad(set_to_none=True)

                            val_batch_x = (batch_x.float().to(device)).view((-1, n_channels, *resize_size))
                            val_batch_y = (batch_y.long().to(device)).view((-1, n_channels, *resize_size))

                            pred = model(val_batch_x.to(device))

                            pred_labels = F.softmax(pred, dim=1) 
                            pred_labels = torch.argmax(pred_labels, dim=1)

                            if loss_unet == 'crossentropy_loss' :
                                criterion = nn.CrossEntropyLoss(weights.to(device))
                                loss = criterion(pred.float(), val_batch_y.squeeze(1))
                            elif loss_unet == 'dice_loss' :
                                criterion = DiceLoss()
                                loss = criterion(pred, val_batch_y, device)
                            elif loss_unet == 'mixed_dice_crossentropy_loss' :
                                criterion = nn.CrossEntropyLoss(weights.to(device))
                                crossentropy_loss = criterion(pred.float(), val_batch_y.squeeze(1)).cpu()
                                criterion = DiceLoss()
                                dice_loss = criterion(pred, val_batch_y, device).to(device)
                            else :
                                print('ERROOOOR')
                                break

                            batch_val_loss.append(loss.item())

                            f1_val = compute_f1(val_batch_y.cpu(), pred_labels.cpu(), lab)
                            f1_mean.append(np.mean(f1_val))
                            f1_arr.append(f1_val)


                    validation_loss = statistics.mean(batch_val_loss)
                    #wandb.log({"train loss": train_loss, 'epoch': steps}) 
                    f1_mean_epoch_val = np.mean(f1_mean)
                    f1_arr_epoch_val = np.asarray(f1_arr) 
                    print('F1 score on validation set : ', f1_mean_epoch_val)
                    print("Current validation loss: %f" % validation_loss) 

                    losses = losses.append([{'epoch': epoch, 'train loss' : train_loss, \
                                                 'validation loss' : validation_loss,
                                                 'train dice': f1_mean_epoch_train, \
                                                 'validation dice': f1_mean_epoch_val}])

                    # Save loss and model at each epoch
                    if (epoch+1) % 50 == 0 :
                        torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss,
                                },str(save_models) + "checkpoints_" + str(epoch)+".pt")

                        losses.to_pickle(str(save_directory) + "/losses.pkl")

                    #Keep the best model so far
                    if min_loss > validation_loss : 
                        min_loss = validation_loss
                        best_model = copy.deepcopy(model)
                        best_epoch = epoch
                        best_optimizer.load_state_dict(optimizer.state_dict()) 

                torch.save({
                        'epoch': best_epoch,
                        'model_state_dict': best_model.state_dict(),
                        'optimizer_state_dict': best_optimizer.state_dict(),
                        'loss': min_loss,
                        },str(save_models) + "checkpoints_best_model" + str(best_epoch)+".pt")

                losses.to_pickle(str(save_directory) + "/losses.pkl")

                # Perform Test with the best model according to validation F1
                results = pd.DataFrame(columns = ['model', 'n vol train', 'lr', 'loss unet',
                                                  'train F1', 'validation F1', 'test F1', 
                                                  'train loss', 'validation loss', 'test loss',
                                                  'best epoch'])
                losses.to_pickle(str(save_directory) + "/losses.pkl")

                losses_min = losses.loc[losses['validation loss'] == losses['validation loss'].min()]

                with torch.no_grad():
                    best_model.to(device)
                    best_model.eval()

                    print("Epoch {:03d}".format(best_epoch))
                    batch_test_loss = []
                    f1_mean = []
                    f1_arr = []
                    for id_batch, (batch_x, batch_y) in enumerate(tqdm(dataset_loader_test)):
                        test_batch_x = (batch_x.float().to(device)).view((-1, n_channels, *resize_size))
                        test_batch_y = (batch_y.long().to(device)).view((-1, n_channels, *resize_size))

                        pred = best_model(test_batch_x)
                        pred_labels = F.softmax(pred, dim=1) 
                        pred_labels = torch.argmax(pred_labels, dim=1)


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

                        f1_val = compute_f1(test_batch_y.cpu(), pred_labels.cpu(), lab)
                        f1_mean.append(np.mean(f1_val))
                        f1_arr.append(f1_val)

                    test_loss = statistics.mean(batch_test_loss)
                    f1_mean_epoch = np.mean(f1_mean)
                    f1_arr_epoch = np.asarray(f1_arr)
                    print('F1 score on test set : ', f1_mean_epoch)

                # Save results for the best epoch
                results = results.append([{'model': pretraining, 'n vol train': n_vol_train, 'lr' : lr, 
                                           'loss unet' : loss_unet,
                                           'train F1': losses_min['train dice'].item(), 
                                           'validation F1' : losses_min['validation dice'].item(), 
                                           'test F1' : f1_mean_epoch, 
                                           'train loss' : losses_min['train loss'].item(), 
                                           'validation loss' : losses_min['validation loss'].item(), 
                                           'test loss' : test_loss, 'best epoch' : best_epoch, 
                                           }])

                results.to_pickle(save_directory + "/results.pkl")