import numpy as np
import os
import tensorflow as tf
import pickle
import collections
import logging

from keras_med_io.batchgenerators.augmentations.resample_augmentations import augment_linear_downsampling_scipy
from keras_med_io.batchgenerators.augmentations.spatial_transformations \
    import augment_spatial_nocrop
from keras_med_io.batchgenerators.augmentations.noise_augmentations \
    import augment_gaussian_noise

######## TFRecordDataset Funcs #######

def read_json(fpath):
    with open(fpath, 'r') as fp:
        return(json.load(fp))

def write_json(f, fpath):
    with open(fpath, 'w') as fp:
        return(json.dump(f, fp))

def transforms(volume, segmentation, n_dim, fraction_,
               variance_, data_format_in = "channels_last"):
    '''
    Does data aug
    Random elastic deformations, random scaling, random rotations, gaussian noise
    '''
    # converts data to channels_first
    if data_format_in == "channels_last": # assumes no batch_size dim
        if n_dim == 2:
            to_channels_first = [-1,0,1]
        elif n_dim == 3:
            to_channels_first = [-1,0,1,2]
        volume = np.transpose(volume, to_channels_first)
        segmentation = np.transpose(segmentation, to_channels_first)

    volume, segmentation = augment_spatial_nocrop(
                                    volume,
                                    segmentation,
                                    n_dim,
                                    border_mode_data='constant',
                                    alpha=(0, 750),
                                    sigma=(10, 13),
                                    scale=(0.8, 1.2),
                                    do_elastic_deform=True,
                                    do_scale=True,
                                    do_rotation=True,
                                    angle_x=(0, 2*np.pi),
                                    angle_y=(0, 0),
                                    angle_z=(0, 0),
                                    fraction=fraction_)
                                    #spacing=spacing_)
    if np.any(variance_ != 0):
        volume = augment_gaussian_noise(volume, noise_variance=variance_)
    # converts data to channels_last
    if n_dim == 2:
        to_channels_last = [1,2,0]
    elif n_dim == 3:
        to_channels_last = [1,2,3,0]
    volume = np.transpose(volume, to_channels_last)
    segmentation = np.transpose(segmentation, to_channels_last)
    return volume, segmentation

def distribution_strategy(num_gpus):
    if num_gpus == 1:
        return tf.contrib.distribute.OneDeviceStrategy(device='/gpu:0')
    elif num_gpus > 1:
        return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)
    else:
        return None

def start_logger(logfile, level=logging.INFO):
    """Start logger
    Parameters
    ----------
    logfile : string (optional)
        file to which the log is saved
    level : int, default: logging.INFO
        logging level as int or logging.DEBUG, logging.ERROR
    """
    f = '%(asctime)s %(name)-35s %(levelname)-8s %(message)s'
    logging.basicConfig(level=level,
                        format=f,
                        datefmt='%d.%m. %H:%M:%S',
                        filename=logfile,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(level)
    # set a format which is simpler for console use
    formatter = logging.Formatter(
        '%(asctime)s [%(name)s %(levelname)s] %(message)s')
    formatter.datefmt = '%d.%m. %H:%M:%S'
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

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
