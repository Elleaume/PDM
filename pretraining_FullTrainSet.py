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
from demeaned_gloss_d import DemeanedGlobalLossD
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
             "representation_loss_global_dminus" : lambda: GlobalLossDminus(config_encoder),
             "2_steps_global_dminus" : lambda: GlobalLossDminus(config_encoder),
             "global_d": lambda: GlobalLossD(config_encoder),
             "representation_loss_global_d" : lambda : GlobalLossD(config_encoder),
             "2_steps_global_d": lambda : GlobalLossD(config_encoder),
             "demeaned_representation_loss": lambda : GlobalLossD(config_encoder),
             "demeaned_representation_loss_2": lambda : GlobalLossD(config_encoder),
             } [config_encoder['loss'] ]()

if config_encoder['loss'] == '2_steps_global_d' :
    criterion2 = GlobalLossD(config_encoder, within_dataset = True)
if config_encoder['loss'] == '2_steps_global_dminus' :
    criterion2 = GlobalLossDminus(config_encoder, within_dataset = True)
if config_encoder['loss'] == 'demeaned_representation_loss' :
    criterion2 = DemeanedGlobalLossD(config_encoder)
    
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)
model.to(device)
print("Running model %s" % config_encoder["model"])

n_parts = config_encoder['n_parts']

name_dataset = config_encoder['name_dataset']
n_datasets = len(name_dataset)
n_volumes = config_encoder['n_volumes']
n_transforms = config_encoder['n_transforms']
resize_size = config_encoder['resize_size']
n_channels = config_encoder['n_channels']
loss_pretraining = config_encoder['loss']
lambda_ = config_encoder['lambda']
batch_size = n_parts * n_datasets * n_volumes * (n_transforms+1)
perp_val = 80
max_epochs = config_encoder['max_epochs']
max_steps = config_encoder['max_steps']
weight_loss = config_encoder['weight_loss']
save_global_path = config_encoder['save_global_path']
temp_fac = config_encoder["temp_fac"]

name_datasets = ""
for dataset in name_dataset :
    name_datasets += dataset + '_'

save_directory = save_global_path + loss_pretraining + '/' + name_datasets + str(n_volumes) \
                        + 'volumesPerBatch'
    
save_models = save_directory + '/save_models/'
Path(save_models).mkdir(parents=True, exist_ok=True)

print('Parameters used : ')
print('Pretraining')
print('loss : ', loss_pretraining)
print('weight_loss : ', weight_loss)
print('n_volumes : ', n_volumes)
print('n_transforms : ', n_transforms)
print('n_datasets : ', n_datasets)
print('max_epochs : ', max_epochs)
print('max_steps : ', max_steps)
print('batch size : ', batch_size)
print('temp_fac : ', temp_fac)
print('save folder : ', save_directory)


# Dataset Initialization
datasets_train = []
datasets_test = []
datasets_loaders_demeaned = []

print('Datasets used for pretraining :')

count_datasets = 0
for config_dataset in config_datasets:

    if config_dataset['Data'] not in name_dataset :
        continue
    count_datasets += 1

    print(config_dataset['Data'])

    current_datasets_train = []
    current_datasets_test = []

    for path in Path( config_dataset["savedir"]).rglob( "train/*/img.nii.gz"):
        current_datasets_train.append(path)
    for path in Path( config_dataset["savedir"]).rglob( "validation/*/img.nii.gz"):
        current_datasets_train.append(path)

    for path in Path( config_dataset["savedir"]).rglob( "test/*/img.nii.gz"):
        current_datasets_test.append(path)
    print('n vol in train : ', len(current_datasets_train))
    print('n vol in test : ', len(current_datasets_test))
    datasets_train.append(current_datasets_train)
    datasets_test.append(current_datasets_test)

    if loss_pretraining == 'demeaned_representation_loss'  or loss_pretraining == 'demeaned_representation_loss_2' :
        dataset_demeaned = PreTrainDatasetDemeaned(current_datasets_train)
        datasets_loader_demeaned = DataLoader(dataset_demeaned,
                            num_workers=1,
                            batch_size = 32,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=False)
        datasets_loaders_demeaned.append(datasets_loader_demeaned)
                


dataset_train = PreTrainDataset(config_encoder, datasets_train)
dataset_loader_train = DataLoader(dataset_train,
                            num_workers=1,
                            batch_size= n_volumes,
                            pin_memory=True,
                            shuffle=True,
                            drop_last=True)

dataset_test = PreTrainDataset(config_encoder, datasets_test)
dataset_loader_test = DataLoader(dataset_test,
                            num_workers=1,
                            batch_size=n_volumes,
                            shuffle=False,
                            drop_last=True)

trans = custom_transforms(config_encoder)

steps = 0
losses = pd.DataFrame(columns = ['epoch', 'train loss', 'test loss'])

# Training
for epoch in range(max_epochs) :
    
    if steps > max_steps :
        break
    print("Doing Train...")
    print("Epoch {:03d}".format(epoch))

    model.train()
    batch_train_loss = []
    
    # Calculate mean representation of each dataset from current model
    if loss_pretraining == 'demeaned_representation_loss' or loss_pretraining == 'demeaned_representation_loss_2':
        with torch.no_grad():
            model.eval()
            mean_representations = []
            for i_dataset in range(n_datasets):
                if epoch < 0 :
                    mean_representation = torch.zeros((128, 6, 6))
                else :
                    for id_batch, batch_x in enumerate(tqdm(datasets_loaders_demeaned[i_dataset])):
                        batch = batch_x.float().to(device)
                        batch = batch.view((-1, n_channels, *resize_size))
                        if id_batch == 0 :
                            pred_representation = model.enc(batch).squeeze()
                        else :
                            pred_representation = torch.cat([pred_representation, model.enc(batch).squeeze()])
                    mean_representation = torch.mean(pred_representation, dim = 0).to(device)
                mean_representations.append(mean_representation)
          
    
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
        
        elif loss_pretraining == 'representation_loss_global_d' :
            # use global_d with one transformation for each loss 
            train_batch_1 = train_batch
            train_batch_2 = train_batch
            for i in range(n_transforms) :
                train_batch_1 = torch.cat([train_batch_1, trans(batch,dtrans ='option_1')])
                train_batch_2 = torch.cat([train_batch_2, trans(batch,dtrans ='option_2')])

            pred = model(train_batch_1).squeeze()
            pred_2 = model.enc(train_batch_2).squeeze()
            
            # loss computed after last dense layer
            loss_1 = criterion(pred)
            # loss computed before last dense layer
            loss_2 = criterion(pred_2)
            
            loss = lambda_ * loss_1 + (1 - lambda_) * loss_2
        
        elif loss_pretraining == 'representation_loss_global_dminus' :
            
            # use global_dminus with one transformation for each loss 
            train_batch_1 = train_batch
            train_batch_2 = train_batch
            
            for i in range(n_transforms) :
                train_batch_1 = torch.cat([train_batch_1, trans(batch,dtrans ='option_1')])
                train_batch_2 = torch.cat([train_batch_2, trans(batch,dtrans ='option_2')])

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
            
        elif loss_pretraining == '2_steps_global_dminus':
            # use global_dminus with two transformations 
            for i in range(n_transforms) :
                train_batch = torch.cat([train_batch, trans(batch)])
            
            pred = model(train_batch).squeeze()
            
            # contrasting between datasets
            loss_1 = criterion(pred)
            # contrasting inside the dataset between partitions
            loss_2 = criterion2(pred)
            
            loss = lambda_ * loss_1 + (1 - lambda_) * loss_2
            
        elif loss_pretraining == 'demeaned_representation_loss' :
            # similar as representation_loss but with demeaned version loss
            # subtract the mean of the representation of the repectif dataset of last epoch to compute the loss
            for i in range(n_transforms) :
                train_batch = torch.cat([train_batch, trans(batch, dtrans ='option_2')])
            pred = model(train_batch).squeeze()
            pred_representation = model.enc(train_batch).squeeze()
            
            loss_1 = criterion(pred)
            loss_2 = criterion2(pred_representation, mean_representations)
            
            loss = lambda_ * loss_1 + (1 - lambda_) * loss_2
            
        elif loss_pretraining == 'demeaned_representation_loss_2' :
            for i in range(n_transforms) :
                train_batch = torch.cat([train_batch, trans(batch, dtrans ='option_2')])
            pred_representation = model.enc(train_batch).squeeze()
            
            # Demeaned reg_pred with respective mean of the dataset 
            pred_demeaned_representation = pred_representation.clone()
            for i in range(pred_representation.shape[0]) :
                pred_demeaned_representation[i,:,:,:] -=  mean_representations[((i // n_parts) % n_datasets)] 
                #print((i // n_parts) % n_datasets)
            pred = model.g1(pred_demeaned_representation).squeeze()
            
            loss = criterion(pred)
            
        if id_batch == 0 :
            pred_train = pred
        else :
            pred_train = torch.cat([pred_train, pred.detach()])
        
        batch_train_loss.append(loss.item())

        loss.backward()
        optimizer.step()
        
        steps += 1
        
    if (epoch+1) % 50 == 0 :
        directory_predictions = str(save_directory) + "/predictions/" + str(epoch) +"/" 
        Path(directory_predictions).mkdir(parents=True, exist_ok=True)
        torch.save(pred_train, directory_predictions+ "pred_train.pt")
        
    train_loss = statistics.mean(batch_train_loss)
    print("Current train loss: %f" % train_loss)  
    
    if (epoch+1) % 50 == 0 or steps > max_steps :
        if steps > max_steps : 
            directory_predictions = str(save_directory) + "/predictions/final_encoder/" 
            Path(directory_predictions).mkdir(parents=True, exist_ok=True)
            torch.save(pred_train, directory_predictions+ "pred_train.pt")
        # test
        batch_val_loss = []

        print("Doing Test...")
        with torch.no_grad():
            model.eval()
            predictions = pd.DataFrame(columns = ['type', 'prediction'])

            for id_batch, batch_x in enumerate(tqdm(dataset_loader_test)):

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

                elif loss_pretraining == 'representation_loss_global_d' :
                    val_batch_1 = val_batch
                    val_batch_2 = val_batch

                    for i in range(n_transforms) :
                        val_batch_1 = torch.cat([val_batch_1, trans(batch,dtrans ='option_1')])
                        val_batch_2 = torch.cat([val_batch_2, trans(batch,dtrans ='option_2')])

                    pred = model(val_batch_1).squeeze()
                    pred_2 = model.enc(val_batch_2).squeeze()

                    # loss computed after last dense layer
                    loss_1 = criterion(pred)
                    # loss computed before last dense layer
                    loss_2 = criterion(pred_2)

                    loss = lambda_ * loss_1 + (1- lambda_) * loss_2
                    
                elif loss_pretraining == 'representation_loss_global_dminus' :
                    val_batch_1 = val_batch
                    val_batch_2 = val_batch

                    for i in range(n_transforms) :
                        val_batch_1 = torch.cat([val_batch_1, trans(batch,dtrans ='option_1')])
                        val_batch_2 = torch.cat([val_batch_2, trans(batch,dtrans ='option_2')])

                    pred = model(val_batch_1).squeeze()
                    pred_2 = model.enc(val_batch_2).squeeze()

                    # loss computed after last dense layer
                    loss_1 = criterion(pred)
                    # loss computed before last dense layer
                    loss_2 = criterion(pred_2)

                    loss = lambda_ * loss_1 + (1 - lambda_) * loss_2

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
                    
                elif loss_pretraining == '2_steps_global_dminus':
                    # use global_d with two transformations 
                    for i in range(n_transforms) :
                        val_batch = torch.cat([val_batch, trans(batch)])

                    pred = model(val_batch).squeeze()

                    # contrasting between datasets
                    loss_1 = criterion(pred)
                    # contrasting inside the dataset between partitions
                    loss_2 = criterion2(pred)

                    loss = lambda_ * loss_1 + (1 - lambda_) * loss_2
                
                elif loss_pretraining == 'demeaned_representation_loss' :
                    for i in range(n_transforms) :
                        val_batch = torch.cat([val_batch, trans(batch, dtrans ='option_2')])
                        
                    #val_batch = torch.cat([val_batch, trans(batch, dtrans ='option_2')])
                    pred_representation = model.enc(val_batch).squeeze()
                    pred = model(val_batch).squeeze()
                    
                    loss_1 = criterion(pred)
                    loss_2 = criterion2(pred_representation, mean_representations)
                    
                    loss = lambda_ * loss_1 + (1 - lambda_) * loss_2
                    
                elif loss_pretraining == 'demeaned_representation_loss_2' :
                    for i in range(n_transforms) :
                        val_batch = torch.cat([val_batch, trans(batch, dtrans ='option_2')])
                    pred_representation = model.enc(val_batch).squeeze()

                    # Demeaned reg_pred with respective mean of the dataset 
                    pred_demeaned_representation = pred_representation.clone().to(device)
                    for i in range(pred_representation.shape[0]) :
                        pred_demeaned_representation[i,:,:,:] -=  mean_representations[((i // n_parts) % n_datasets)] 
                        #print((i // n_parts) % n_datasets)
                    pred = model.g1(pred_demeaned_representation).squeeze()

                    loss = criterion(pred)
                    
                if id_batch == 0 :
                    pred_test = pred
                else :
                    pred_test = torch.cat([pred_test, pred])

                batch_val_loss.append(loss.item())

        torch.save(pred_test, directory_predictions+"pred_test.pt")
    
        test_loss = statistics.mean(batch_val_loss)

        print("Current test loss: %f" % test_loss)  
    
    if (epoch+1) % 50 == 0 :
        losses = losses.append([{'epoch': epoch, 'train loss' : train_loss, 'test loss' : test_loss}])
        
    else : 
        losses = losses.append([{'epoch': epoch, 'train loss' : train_loss}])
        
        
    # Save loss and model at each 50 epoch
    if (epoch+1) % 50 == 0 :
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                },str(save_models) + "checkpoints_" + str(epoch)+".pt")

        losses.to_pickle(str(save_directory) + "/losses.pkl")
        
    if steps > max_steps :
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                },str(save_models) + "final_encoder.pt")

        losses.to_pickle(str(save_directory) + "/losses.pkl")
        
        

    
losses.to_pickle(str(save_directory) + "/losses.pkl")
print('Done!')
print('epoch : ', epoch)
print('steps : ', steps) 
