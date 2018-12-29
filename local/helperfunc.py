import numpy as np
import os
import tensorflow as tf

from batchgenerators.augmentations.resample_augmentations import augment_linear_downsampling_scipy
from batchgenerators.augmentations.spatial_transformations \
    import augment_spatial
from batchgenerators.augmentations.noise_augmentations \
    import augment_gaussian_noise

######## TFRecordDataset Funcs #######
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_json(fpath):
    with open(fpath, 'r') as fp:
        return(json.load(fp))

def write_json(f, fpath):
    with open(fpath, 'w') as fp:
        return(json.dump(f, fp))

def transforms(volume, segmentation, patch_size_, fraction_, spacing_,
               variance_):
    # BEWARE SPACING: dset.spacing = (x,y,z) but volume is shaped (z,x,y))
    volume, augmentation, rval = augment_spatial(
                                    volume,
                                    segmentation,
                                    patch_size=patch_size_,
                                    patch_center_dist_from_border=np.array(
                                        patch_size_) // 2,
                                    border_mode_data='constant',
                                    border_mode_seg='constant',
                                    border_cval_data=np.min(volume),
                                    border_cval_seg=0,
                                    alpha=(0, 750),
                                    sigma=(10, 13),
                                    scale=(0.8, 1.2),
                                    do_elastic_deform=True,
                                    do_scale=True,
                                    do_rotation=True,
                                    angle_x=(0, 2*np.pi),
                                    angle_y=(0, 0),
                                    angle_z=(0, 0),
                                    fraction=fraction_,
                                    spacing=spacing_)
    if np.any(variance_ != 0):
        volume = augment_gaussian_noise(volume, noise_variance=variance_)
    return volume, augmentation, rval

######## Utility ########

def normalize(x, scale = False):
    '''
    x: input array/img [4 dimensional with tf NWHC ordering
    scale: boolean on what type of normalization you want
    '''
    if scale:
    # scales between 0 and 1
        x_min = x.min(axis=(1, 2), keepdims=True)
        x_max = x.max(axis=(1, 2), keepdims=True)
        return (x - x_min)/(x_max-x_min)

    if not scale:
    # mean 0 and std 1
        return (x - np.mean(x))/np.std(x)

#prints important metadata for data
def metadata(arrays):
    '''
    Arguments:
        arrays: takes list
    Returns:
        print statement of metadata
    '''
    for i in arrays:
        print(i.shape, "\nmax: ",np.amax(i), "\nmin: ",np.amin(i), "\nmean: ",np.mean(i), "\nstd: ",np.std(i))
