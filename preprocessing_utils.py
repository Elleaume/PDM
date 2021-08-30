import matplotlib.pyplot as plt
import nibabel as nib
import random
import numpy as np
from skimage import transform
from collections import OrderedDict
import json
from pathlib import Path
from torchvision import transforms
from sklearn.model_selection import train_test_split
import skimage.transform as skTrans

def crop_or_pad_slice_to_size(img_slice, nx, ny):
    slice_cropped = np.zeros((nx,ny))
    x, y = img_slice.shape
    
    x_s = (x-nx) // 2
    y_s = (y-ny) // 2
    x_c = (nx-x) // 2
    y_c = (ny-y) // 2
    
    if x > nx and y > ny:
        slice_cropped = img_slice[x_s:x_s+nx, y_s:y_s+ny]
    else :
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny :
            slice_cropped[x_c:x_c+x, :] = img_slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = img_slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = img_slice[:, :]

    return slice_cropped

def visualize_data(data_dir1, data_dir2, data_dir3, data_dir4) :    
    
    data = nib.load(data_dir1)
    image_data1 = data.get_fdata()
    
    data = nib.load(data_dir2)
    image_data2 = data.get_fdata()

    data = nib.load(data_dir3)
    image_data3 = data.get_fdata()
    
    data = nib.load(data_dir4)
    image_data4 = data.get_fdata()
    
    fig, axs = plt.subplots(2, 2)
    idx1 = int(image_data1.shape[2]/2)
    axs[0, 0].imshow(image_data1[:,:,idx1], cmap = "gray")
    axs[0, 0].set_title(image_data1.shape)
    idx2 = int(image_data2.shape[2]/2)
    axs[0, 1].imshow(image_data2[:,:,idx2], cmap = "gray")
    axs[0, 1].set_title(image_data2.shape)
    idx3 = int(image_data3.shape[2]/2)
    axs[1, 0].imshow(image_data3[:,:,idx3], cmap = "gray")
    axs[1, 0].set_title(image_data3.shape)
    idx4 = int(image_data4.shape[2]/2)
    axs[1, 1].imshow(image_data4[:,:,idx4], cmap = "gray")
    axs[1, 1].set_title(image_data4.shape)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

def visualize_data_2_channels(data_dir1, data_dir2, data_dir3, data_dir4, channel) :    
    data = nib.load(data_dir1)
    image_data1 = data.get_fdata()
    
    data = nib.load(data_dir2)
    image_data2 = data.get_fdata()

    data = nib.load(data_dir3)
    image_data3 = data.get_fdata()
    
    data = nib.load(data_dir4)
    image_data4 = data.get_fdata()
    
    print(image_data1.shape)
    fig, axs = plt.subplots(2, 2)
    idx1 = int(image_data1.shape[2]/2)
    axs[0, 0].imshow(image_data1[:,:,idx1,channel], cmap = "gray")
    axs[0, 0].set_title(image_data1.shape)
    idx2 = int(image_data2.shape[2]/2)
    axs[0, 1].imshow(image_data2[:,:,idx2,channel], cmap = "gray")
    axs[0, 1].set_title(image_data2.shape)
    idx3 = int(image_data3.shape[2]/2)
    axs[1, 0].imshow(image_data3[:,:,idx3,channel], cmap = "gray")
    axs[1, 0].set_title(image_data3.shape)
    idx4 = int(image_data4.shape[2]/2)
    axs[1, 1].imshow(image_data4[:,:,idx4,channel], cmap = "gray")
    axs[1, 1].set_title(image_data4.shape)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
        
def preprocess_data(img, config, pixel_size_target, pixel_size) :

    nx, ny = config["resize_size"]
    crop_size = config['crop_size']
    resize_size = config['resize_size']
    
    
    scale_vector = [pixel_size[0] / pixel_size_target[0], pixel_size[1] / pixel_size_target[1]]
    
    for slice_no in range(img.shape[2]) :
        slice_img = np.squeeze(img[:, :, slice_no])
        
        slice_rescaled = transform.rescale(slice_img,
                                          scale_vector,
                                          order= 1,
                                          preserve_range = True,
                                          mode = 'constant')
        
        slice_cropped = crop_or_pad_slice_to_size(slice_rescaled, crop_size[0], crop_size[1])
        #slice_cropped = crop_or_pad_slice_to_size(slice_rescaled, nx, ny)
        #print(type(slice_cropped))
        
        slice_resized =  skTrans.resize(slice_cropped, resize_size, order=1, preserve_range=True)
        #print(type(slice_resized)) 
        if(slice_no == 0) :
            #cropped_img = np.reshape(slice_cropped, (nx,ny,1))
            cropped_img = np.reshape(slice_resized, (nx,ny,1))
        else :
            #slice_cropped_tmp = np.reshape(slice_cropped, (nx, ny, 1))
            slice_cropped_tmp = np.reshape(slice_resized, (nx,ny,1))
            cropped_img = np.concatenate((cropped_img, slice_cropped_tmp), axis=2)
            
    return cropped_img

def find_pix_dim(vol_img) :
    
    pix_dim = vol_img.header['pixdim']
    dim = vol_img.header['dim']
    max_indx = np.argmax(dim)
    pixdimX = pix_dim[max_indx]
    dim = np.delete(pix_dim, max_indx)
    max_indy = np.argmax(dim)
    pixdimY = pix_dim[max_indy]
    return [pixdimX, pixdimY]

def normalize_minmax_data(image_data,min_val=1,max_val=99):
    """
    # 3D MRI scan is normalized to range between 0 and 1 using min-max normalization.
    Here, the minimum and maximum values are used as 1st and 99th percentiles respectively from the 3D MRI scan.
    We expect the outliers to be away from the range of [0,1].
    input params :
        image_data : 3D MRI scan to be normalized using min-max normalization
        min_val : minimum value percentile
        max_val : maximum value percentile
    returns:
        final_image_data : Normalized 3D MRI scan obtained via min-max normalization.
    """
    min_val_1p=np.percentile(image_data,min_val)
    max_val_99p=np.percentile(image_data,max_val)
    final_image_data=np.zeros((image_data.shape[0],image_data.shape[1],image_data.shape[2]), dtype=np.float64)
    # min-max norm on total 3D volume
    final_image_data=(image_data-min_val_1p)/(max_val_99p-min_val_1p)
    
    return final_image_data

