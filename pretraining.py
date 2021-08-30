import random
from pathlib import Path
import time
import matplotlib.pyplot as plt
#%matplotlib inline

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import nibabel as nib
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

from decoder_pretrain import DecoderPretrainNet
from encoder_pretrain import EncoderPretrainNet
from gloss_dminus import GlobalLossDminus
from gloss_d import GlobalLossD
from pretraining_utils import *

import argparse
import json
import statistics

torch.cuda.empty_cache()

# arguments for training script
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="string which config to use", type=str, required=True)
#parser.add_argument("--wandb_mode", help="wandb mode", type=str, default="online")
#parser.add_argument('--dataset', type = str, default = 'acdc', choices=['acdc', 'cimas'])
args = parser.parse_args()

with open('configs/preprocessing_datasets.json') as config_file:
    config_datasets = json.load(config_file)
with open('configs/config_encoder.json') as config_file:
    config_encoder = json.load(config_file)

# init W&B
#print("Using W&B in %s mode" % 'online')
#wandb.init(project=config_encoder["model"], mode='online')
seed = config_encoder['seed']
torch.manual_seed(seed)
np.random.seed(seed)

# choose model from config
model = EncoderPretrainNet(config_encoder)

# choose specified optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config_encoder["lr"])
criterion = {"global_dminus": lambda: GlobalLossDminus(config_encoder),
             "global_d": lambda: GlobalLossD(config_encoder),
             "representation_loss" : lambda : GlobalLossD(config_encoder),
             "2_steps_global_d": lambda : GlobalLossD(config_encoder)}[config_encoder['loss']]()
    
if config_encoder['loss'] == '2_steps_global_d' :
    criterion2 = GlobalLossD(config_encoder, within_dataset = True)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)
model.to(device)
print("Running model %s" % config_encoder["model"])

n_parts = config_encoder['n_parts']
n_datasets = config_encoder['n_datasets']
n_volumes = config_encoder['n_volumes']
n_transforms = config_encoder['n_transforms']
resize_size = config_encoder['resize_size']
n_channels = config_encoder['n_channels']
loss_pretraining = config_encoder['loss']
lambda_ = config_encoder['lambda']
batch_size = n_parts * n_datasets * n_volumes * (n_transforms+1)
perp_val = 80
max_epochs = config_encoder['max_epochs']
weight_loss = config_encoder['weight_loss']
save_global_path = config_encoder['save_global_path']
temp_fac = config_encoder["temp_fac"]

date = str(time.strftime("%Y%h%d_%Hh%M"))
if loss_pretraining == 'representation_loss' or loss_pretraining == '2_steps_global_d':
    save_directory = save_global_path + loss_pretraining + '/' + str(n_datasets) +'datasets_' + str(n_volumes) \
                    + 'volumesPerBatch_' + str(n_transforms) + 'transforms'
else :
    save_directory = save_global_path + loss_pretraining + '/' + str(n_datasets) +'datasets_' + str(n_volumes) \
                    + 'volumesPerBatch_' + str(n_transforms) + 'transforms' 
save_models = save_directory + '/save_models/'
Path(save_models).mkdir(parents=True, exist_ok=True)

print('Parameters used : ')
print('loss : ', loss_pretraining)
print('weight_loss : ', weight_loss)
print('n_volumes : ', n_volumes)
print('n_transforms : ', n_transforms)
print('n_datasets : ', n_datasets)
print('max_epochs : ', max_epochs)
print('batch size : ', batch_size)
print('temp_fac : ', temp_fac)
print('save folder : ', save_directory)



count_datasets = 0
for config_dataset in config_datasets:
    if config_dataset['experiment'] == 'pretraining' :
        if count_datasets >= n_datasets :
            break
        count_datasets += 1
assert count_datasets == n_datasets

# Dataset Initialization
datasets_train = []
datasets_validation = []

print('Datasets used for pretraining :')

count_datasets = 0
for config_dataset in config_datasets:
    if config_dataset['experiment'] == 'pretraining' :
        if count_datasets >= n_datasets :
            break
        count_datasets += 1
        
        print(config_dataset['Data'])
        
        current_datasets_train = []
        current_datasets_validation = []

        for path in Path( config_dataset["savedir"]).rglob( "train/*/img.nii.gz"):
            current_datasets_train.append(path)
        for path in Path( config_dataset["savedir"]).rglob( "validation/*/img.nii.gz"):
            current_datasets_validation.append(path)
        datasets_train.append(current_datasets_train)
        datasets_validation.append(current_datasets_validation)


dataset_train = PreTrainDataset(config_encoder, datasets_train)
dataset_loader_train = DataLoader(dataset_train,
                            num_workers=1,
                            batch_size=n_volumes,
                            pin_memory=True,
                            shuffle=True,
                            drop_last=True)

dataset_validation = PreTrainDataset(config_encoder, datasets_validation)
dataset_loader_validation = DataLoader(dataset_validation,
                            num_workers=1,
                            batch_size=n_volumes,
                            shuffle=False,
                            drop_last=True)


trans = custom_transforms(config_encoder)

steps = 0
losses = pd.DataFrame(columns = ['epoch', 'train loss', 'validation loss'])

# Training
min_loss = 999999
best_model = EncoderPretrainNet(config_encoder)
best_optimizer = torch.optim.Adam(model.parameters(), lr=config_encoder["lr"])

for epoch in range(max_epochs) :
    print("Doing Train...")
    print("Epoch {:03d}".format(epoch))

    model.train()
    batch_train_loss = []
    
    for id_batch, batch_x in enumerate(tqdm(dataset_loader_train)):

        optimizer.zero_grad()
        
        batch = batch_x.float().to(device)
        batch = batch.view((-1, n_channels, *resize_size))
        train_batch = batch
        
        if loss_pretraining == 'global_dminus' :
            # don't include similar partitions as positive pairs
            for i in range(n_transforms) :
                train_batch = torch.cat([train_batch, trans(batch)])
            pred = model(train_batch).squeeze()
            loss = criterion(pred)
        
        elif loss_pretraining == 'global_d' :
            # include similar partitions as positive pairs
            for i in range(n_transforms) :
                train_batch = torch.cat([train_batch, trans(batch)])
            
            pred = model(train_batch).squeeze()
            loss = criterion(pred)
        
        elif loss_pretraining == 'representation_loss' :
            # use global_d with one transformation for each loss 
            train_batch_1 = torch.cat([train_batch, trans(batch,dtrans ='option_1')])
            train_batch_2 = torch.cat([train_batch, trans(batch,dtrans ='option_2')])
            
            pred = model(train_batch_1).squeeze()
            pred_2 = model.enc(train_batch_2).squeeze()
            
            # loss computed after last dense layer
            loss_1 = criterion(pred)
            # loss computed before last dense layer
            loss_2 = criterion(pred_2)
            
            loss = lambda_ * loss_1 + (1 - lambda_) * loss_2
        
        elif loss_pretraining == '2_steps_global_d':
            # use global_d with two transformations 
            for i in range(n_transforms) :
                train_batch = torch.cat([train_batch, trans(batch)])
            
            pred = model(train_batch).squeeze()
            
            # contrasting between datasets
            loss_1 = criterion(pred)
            # contrasting inside the dataset between partitions
            loss_2 = criterion2(pred)
            
            loss = lambda_ * loss_1 + (1 - lambda_) * loss_2
            
        if id_batch == 0 :
            pred_train = pred
        else :
            pred_train = torch.cat([pred_train, pred.detach()])
        
        batch_train_loss.append(loss.item())

        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 50 == 0 :
        directory_predictions = str(save_directory) + "/predictions/" + str(epoch) +"/" 
        Path(directory_predictions).mkdir(parents=True, exist_ok=True)
        torch.save(pred_train, directory_predictions+ "pred_train.pt")
        
    train_loss = statistics.mean(batch_train_loss)
    print("Current train loss: %f" % train_loss)  
    
    # Validation 
    batch_val_loss = []

    print("Doing Validation...")
    with torch.no_grad():
        model.eval()
        predictions = pd.DataFrame(columns = ['type', 'prediction'])

        for id_batch, batch_x in enumerate(tqdm(dataset_loader_validation)):
            
            batch = batch_x.float().to(device)
            batch = batch.view((-1, n_channels, *resize_size))
            val_batch = batch
        
            if loss_pretraining == 'global_dminus' :
                # don't include similar partitions as positive pairs
                for i in range(n_transforms) :
                    val_batch = torch.cat([val_batch, trans(batch)])
                
                pred = model(val_batch).squeeze()
                loss = criterion(pred)
            
            elif loss_pretraining == 'global_d' :
                # include similar partitions as positive pairs
                for i in range(n_transforms) :
                    val_batch = torch.cat([val_batch, trans(batch)])
                
                pred = model(val_batch).squeeze()
                loss = criterion(pred)
                
            elif loss_pretraining == 'representation_loss' :
                # use global_d with one transformation for each loss 
                val_batch_1 = torch.cat([val_batch, trans(batch,dtrans ='option_1')])
                val_batch_2 = torch.cat([val_batch, trans(batch,dtrans ='option_2')])

                pred = model(val_batch_1).squeeze()
                pred_2 = model.enc(val_batch_2).squeeze()
                
                # loss computed after last dense layer
                loss_1 = criterion(pred)
                # loss computed before last dense layer
                loss_2 = criterion(pred_2)

                loss = lambda_ * loss_1 + (1- lambda_) * loss_2
                
            elif loss_pretraining == '2_steps_global_d':
                # use global_d with two transformations 
                for i in range(n_transforms) :
                    val_batch = torch.cat([val_batch, trans(batch)])

                pred = model(val_batch).squeeze()
                
                # contrasting between datasets
                loss_1 = criterion(pred)
                # contrasting inside the dataset between partitions
                loss_2 = criterion2(pred)

                loss = lambda_ * loss_1 + (1 - lambda_) * loss_2
            
            if id_batch == 0 :
                pred_validation = pred
            else :
                pred_validation = torch.cat([pred_validation, pred])
            
            batch_val_loss.append(loss.item())
    
    if (epoch+1) % 50 == 0 :
        torch.save(pred_validation, directory_predictions+"pred_validation.pt")
    
    validation_loss = statistics.mean(batch_val_loss)
    
    print("Current validation loss: %f" % validation_loss)  
    
    losses = losses.append([{'epoch': epoch, 'train loss' : train_loss, 'validation loss' : validation_loss}])
    steps += 1
    
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
        pred_train_best_model = pred_train
        pred_validation_best_model = pred_validation
        
        

torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': best_optimizer.state_dict(),
        'loss': min_loss,
        },str(save_models) + "checkpoints_best_model" + str(best_epoch)+".pt")

directory_predictions_best_model = str(save_directory) + "/predictions/best_epoch_" + str(best_epoch) +"/" 
Path(directory_predictions_best_model).mkdir(parents=True, exist_ok=True)
    
torch.save(pred_train_best_model, directory_predictions_best_model  + "pred_train.pt")
torch.save(pred_validation_best_model, directory_predictions_best_model  + "pred_validation.pt")
    
losses.to_pickle(str(save_directory) + "/losses.pkl")

losses_min = losses.loc[losses['validation loss'] == losses['validation loss'].min()]
print('best model found at epoch : ',losses_min['epoch'].item(), 
      ', with train loss : ', losses_min['train loss'].item(),
      ', and validation loss : ', losses_min['validation loss'].item())


           