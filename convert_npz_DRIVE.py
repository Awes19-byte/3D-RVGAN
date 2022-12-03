import numpy as np
import tensorflow as tf
from numpy import asarray,savez_compressed
import argparse
import os
import glob
import nibabel as nib
from scipy import ndimage, misc
from nibabel.processing import conform


def resize_volume(img, x, y, z):
    desired_depth = z
    desired_width = y
    desired_height = x

    current_depth = img.shape[0]
    current_width = img.shape[1]
    current_height = img.shape[2]

    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height

    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    img = ndimage.zoom(img, (depth_factor, width_factor, height_factor, 1), order=1)
    return img

#load all images in a directory into memory
def load_images(imgpath,maskpath, list_id, size=(64,64,64)):
    src_list, mask_list, label_list = list(), list(), list()
    for i,id in enumerate(list_id):
            # load and resize the image
            
            img = nib.load(imgpath + list_id[i]).get_data()
            fundus_img = resize_volume(img,size[0],size[1],size[2])

            mask = conform(nib.load(os.path.join(maskpath + list_id[i])),
                            out_shape=size)
            mask_img = mask.get_data()

            
            # split into satellite and map
            src_list.append(fundus_img)
            mask_list.append(mask_img)
            print(id)
    return [asarray(src_list), asarray(mask_list)]
 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=(64,64,64))
    parser.add_argument('--n_crops', type=int, default=210)
    parser.add_argument('--outfile_name', type=str, default='DRIVE')
    args = parser.parse_args()

    # dataset path
    imgpath = 'image/'
    maskpath = 'label/'

    partition = {}
    images = glob.glob(os.path.join(imgpath, "*.nii.gz"))
    images_IDs = [name.split("/")[-1] for name in images]
    print(images_IDs)
    partition['train'] = images_IDs

    # load dataset
    [src_images, mask_images] = load_images(imgpath,maskpath,
                                                          partition['train'], args.input_dim)
    print('Loaded: ', src_images.shape, mask_images.shape)
    # save as compressed numpy array
    filename = args.outfile_name+'.npz'
    savez_compressed(filename, src_images, mask_images)
    print('Saved dataset: ', filename)
