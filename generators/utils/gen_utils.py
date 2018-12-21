# from CapsNetsMRI.io.data_aug import *
# from CapsNetsMRI.io.io_func import normalize_clip, resample_img, whitening, normalize
import keras
import SimpleITK as sitk
from SimpleITK import GetArrayFromImage, ReadImage
from random import randint
import numpy as np
from glob import glob
import os

class BaseGenerator(keras.utils.Sequence):
    '''
    For generating 2D thread-safe data in keras. (no preprocessing and channels_last)
    Attributes:
      list_IDs: filenames (.nii files); must be same for training and labels
      data_dirs: list of [training_dir, labels_dir]
      batch_size: int of desired number images per epoch
      n_channels: <-
    '''
    # https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    def __init__(self, list_IDs, data_dirs, batch_size, shuffle = True):
        # lists of paths to images
        self.list_IDs = list_IDs
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_IDs))

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / float(self.batch_size)))

    def __getitem__(self, idx):
        '''
        Defines the fetching and on-the-fly preprocessing of data.
        '''
        # file names
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.data_gen(list_IDs_temp)
        return (X, y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # self.img_idx = np.arange(len(self.x))
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_gen(self, list_IDs_temp):
        '''
        Preprocesses the data
        Args:
            batch_x, batch_y
        Returns
            x, y
        '''
        raise NotImplementedError

def docs():
    """
    For generator utilies
    * base keras.utils.Sequence to inherit from
        * need to add num_images (number of images to sample from)
            * diff from batch size
                * getting random indices for this
    * basic utility functions for different Sequence archetypes
        * balanced sampling
        * random sampling
    * MAKE A FUNCTION TO GET THE INDICES FOR POSITIVE SLICES
        * then only do computations based on those positive slices
            * random select index and should be proportional to batch size
            * threshold; max 2 batches per image? <- make as parameter
                * so index batch_size/num_slices slices from each scan
        ******but when random cropping, it won't be positive sooo....
    def get_positive_idx(label):
        # gets all the slice indices for positive slices
        pos_idx = np.nonzero(label)
        return np.unique(pos_idx[0])

    """
    pass
