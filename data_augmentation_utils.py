import numpy as np
import torch
import random
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import scipy.ndimage.interpolation
from skimage import transform
import torchvision.transforms.functional as TF
from preprocessing_utils import crop_or_pad_slice_to_size
import matplotlib.pyplot as plt

def elastic_transform_image_and_label(image, # 2d
                                      label,
                                      sigma,
                                      alpha,
                                      random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    
    # applying the gaussian filter with a relatively large std deviation (~20) makes this a relatively smooth deformation field, but with very small deformation values (~1e-3)
    # multiplying it with alpha (500) scales this up to a reasonable deformation (max-min:+-10 pixels)
    # multiplying it with alpha (1000) scales this up to a reasonable deformation (max-min:+-25 pixels)
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    distored_label = map_coordinates(label, indices, order=0, mode='reflect').reshape(shape)
    
    return distored_image, distored_label

class DataAugmentation(object):

    def __init__(self) :
        self.data_aug_ratio = 0.25
        self.sigma = 20
        self.alpha = 1000
        self.trans_min,self.trans_max = -10, 10
        self.rot_min, self.rot_max = -10, 10
        self.scale_min, self.scale_max = 0.90, 1.10
        self.contrast_min, self.contrast_max  = 0.7, 1.3
        self.brightness_min, self.brightness_max = 0, 0.3
            
    def __call__(self, images, masks, dtype ='train'):
        
        initial_images = images.clone()
        initial_masks = masks.clone()
        
        return_images = images
        return_masks = masks

        all_applied_transforms = []
        
        for idx in range(images.shape[0]):
            
            tf_image = images[idx,0,:,:].cpu()
            tf_mask = masks[idx,0,:,:].cpu()
            applied_transforms = []
            
            # Random deformation
            if np.random.rand() < self.data_aug_ratio:
                image, mask = elastic_transform_image_and_label(tf_image, tf_mask, 
                                                                      sigma = self.sigma,alpha = self.alpha)
                tf_image, tf_mask = torch.tensor(image), torch.tensor(mask)
                applied_transforms.append('Elastic')
                
                
            # Translation
            if np.random.rand() < self.data_aug_ratio:
                random_shift_x = np.random.uniform(self.trans_min, self.trans_max)
                random_shift_y = np.random.uniform(self.trans_min, self.trans_max)

                tf_image = torch.tensor(scipy.ndimage.interpolation.shift(tf_image,
                                                            shift = (random_shift_x, random_shift_y),
                                                            order = 1))

                tf_mask = torch.tensor(scipy.ndimage.interpolation.shift(tf_mask,
                                                            shift = (random_shift_x, random_shift_y),
                                                            order = 0))
                applied_transforms.append('Translation')
                
            # Random Rotation
            if np.random.rand() < self.data_aug_ratio:
                rand_angle = np.random.uniform(self.rot_min, self.rot_max)
                tf_image = TF.rotate(tf_image.unsqueeze(0), rand_angle)[0,:,:]
                tf_mask = TF.rotate(tf_mask.unsqueeze(0), rand_angle)[0,:,:]
                applied_transforms.append('Rot')
                
            # Random Scaling
            if np.random.rand() < self.data_aug_ratio:
                n_x, n_y = tf_image.shape
                scale_val = round(random.uniform(self.scale_min,self.scale_max), 2)
                
                image = transform.rescale(tf_image, scale_val, order=1, 
                                          preserve_range=True, mode = 'constant')
                tf_image = torch.tensor(crop_or_pad_slice_to_size(image, n_x, n_y))
                
                mask = transform.rescale(tf_mask, scale_val, order=0, 
                                         preserve_range=True, mode = 'constant')
                tf_mask = torch.tensor(crop_or_pad_slice_to_size(mask, n_x, n_y))
                applied_transforms.append('Scale')
                
            # Random horizontal flipping
            if np.random.rand() < self.data_aug_ratio:
                tf_image = TF.hflip(tf_image)
                tf_mask = TF.hflip(tf_mask)
                applied_transforms.append('HF')
                
            # Random vertical flipping
            if np.random.rand() < self.data_aug_ratio:
                tf_image = TF.vflip(tf_image)
                tf_mask = TF.vflip(tf_mask)
                applied_transforms.append('VF')
                
            # Random intensity / contrast changes
            if np.random.rand() < self.data_aug_ratio:
                contrast_factor =  np.random.uniform(self.contrast_min, self.contrast_max)
                brightness_factor =  np.random.uniform(self.brightness_min, self.brightness_max)
                tf_image = tf_image * contrast_factor + brightness_factor
                applied_transforms.append('intensity')
            
            all_applied_transforms.append(applied_transforms)
            return_images[idx,0,:,:] = tf_image    
            return_masks[idx,0,:,:] = tf_mask                 
        
        # Show image and its transform
        
        #for i in range(0,return_images.shape[0]):
        #    fig, axs = plt.subplots(2, 2)
        #    fig.suptitle(all_applied_transforms[i])
        #    axs[0,0].imshow(initial_images[i,0,:,:], cmap = 'gray')
        #    axs[0,1].imshow(return_images[i,0,:,:], cmap = 'gray')
        #    axs[1,0].imshow(initial_masks[i,0,:,:], cmap = 'gray')
        #    axs[1,1].imshow(return_masks[i,0,:,:], cmap = 'gray')   
        #plt.show()
            
        return return_images, return_masks