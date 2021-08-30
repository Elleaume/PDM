import numpy as np
import pandas as pd
import plotly.express as px
import torch
import nibabel as nib
import random

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing

from torchvision import transforms
import torchvision.transforms.functional as TF


class PreTrainDataset(torch.utils.data.Dataset):
    def __init__(self, cfg_encoder, datasets):
        
        self.filenames = datasets
        #for dataset in self.filenames :
            #print(dataset)
            
        self.n_datasets = len(self.filenames)
        
        max_n_volumes = 0
        for i in range(len(self.filenames)) :
            self.n_volumes = max( max_n_volumes, len(self.filenames[i]) )
            
        self.n_parts = cfg_encoder['n_parts'] 
    
    def get_images_from_partitions(idx) :
        
        self.vol_indices = np.zeros((self.n_datasets, self.n_volumes, self.n_parts),dtype =int)
        for i in range(self.n_datasets) :
            for j in range(len(self.filenames[i])) :
                self.vol_indices[i][j] = get_index_partitions(self.filenames,i,j, self.n_parts)
                
    def sample_minibatch_for_global_loss(self,dataset, idx): 
        
        vol_indice = get_index_partitions(self.filenames, dataset, idx, self.n_parts)
        #vol_indice = self.vol_indices[dataset][idx]
        vol_file = self.filenames[dataset][idx]
        volume = nib.load(vol_file).get_fdata()
        #print(vol_indice)
        return volume[:, :, vol_indice].transpose(2, 0, 1)

    def __len__(self):
        return self.n_volumes

    def __getitem__(self, idx):
        
        for dataset in range(len(self.filenames)) :
            if idx >= len(self.filenames[dataset]):
                    lower_idx = random.randrange(len(self.filenames[dataset]))
            else : lower_idx = idx
            
            new_slices = self.sample_minibatch_for_global_loss(dataset, lower_idx)
            
            i = random.randint(0, 10)
            
            if dataset == 0 :
                slices = new_slices                
            else :
                slices = np.concatenate((slices, new_slices), axis=0)  
        return slices
    
    
class PreTrainDatasetDemeaned(torch.utils.data.Dataset) :
    def __init__(self, volumes):
        self.path_volumes = volumes
        
        for i in range(len(self.path_volumes)):
            vol_file = self.path_volumes[i]
            volume = nib.load(vol_file).get_fdata()
            new_img = volume.transpose(2, 0, 1)
            
            if i == 0: 
                self.imgs = new_img
            else :
                self.imgs = np.concatenate((self.imgs, new_img), axis=0)
                
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]

    
def get_index_partitions(filenames, i, j, n_parts): 
    
    vol_file = filenames[i][j]
    volume = nib.load(vol_file).get_fdata()

    n_slices = volume.shape[2]
    # starting index of first partition of any chosen volume
    partition_lengths = [0]
    # find the starting and last index of each partition in a volume based on input image size. 
    # shape[0] indicates total no. of slices in axial direction of the input image.
    for k in range(1, n_parts + 1):
        partition_lengths.append(k * int(n_slices / n_parts))
    # Now sample 1 image from each partition randomly. Overall, n_parts images for each chosen volume id.
    idces = []
    for k in range(0, len(partition_lengths) - 1):
        # sample image from each partition randomly
        i_sel = random.sample(range(partition_lengths[k], partition_lengths[k + 1]), 1)
        idces.append(i_sel[0])
    
    return idces
    
    
    
class custom_transforms(object):

    def __init__(self, cfg_encoder) :
        #assert isinstance(output_size, (int,tuple))
        self.output_size = cfg_encoder['resize_size']
        
    def __call__(self, image, dtype ='train', dtrans = 'option_1'):
        img_initial = image
        resize = transforms.Resize(size = self.output_size)
        image = resize(image)
        
        brightness_factor = np.random.uniform(0, 0.3,image.shape[0])
        contrast_factor = np.random.uniform(0.7, 1.3,image.shape[0])
        #Random crop
        #mask = TF = crop(mask,i,j,h,w)

        return_images = image
        for idx in range(image.shape[0]):
            #define a crop size for the current image
            crop_size = [int(random.uniform(3*image.shape[2]/4,image.shape[2])), \
                         int(random.uniform(3*image.shape[3]/4, image.shape[3]))]
            i,j,h,w = transforms.RandomCrop.get_params(image[idx,:,:,:], output_size=crop_size)
            # to crop from the center adjust start of the crop:
            #i , j = int((self.output_size[0] - h)/2), int((self.output_#ize [1] - w)/2)
            
            # For ACDC dataset 
            i = 0
            j = int(random.uniform(0,30))
            
            if dtrans == 'option_1' :
                cropped_img = TF.crop(image[idx],i,j,h,w)
                return_images [idx,:,:,:]= resize(cropped_img)
                
                # adjust contrast and brightness
                return_images[idx,:,:,:] = return_images[idx,:,:,:] * contrast_factor[idx] + brightness_factor[idx]        

                # Keep all values between 0 and 1.5
                return_images[idx,:,:,:] = torch.clamp(return_images[idx,:,:,:], 0, 1.5) 
               
            if dtrans == 'option_2' :
                # adjust contrast and brightness
                return_images[idx,:,:,:] = image[idx] * contrast_factor[idx] + brightness_factor[idx]        

                # Keep all values between 0 and 1.5
                return_images[idx,:,:,:] = torch.clamp(return_images[idx,:,:,:], 0, 1.5) 
                
            
        
        # Show image and its transform
        #volume[:,:,idx_image] = image
        #for i in range(5):
        #    fig, axs = plt.subplots(1, 2)
        #    axs[0].imshow(img_initial[i,0,:,:])
        #    axs[1].imshow(return_images[i,0,:,:])
    
        return return_images# volume #, mask
    