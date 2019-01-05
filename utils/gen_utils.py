import keras
import numpy as np
import os
# from keras_med_io.utils.io_func import normalize_clip, whiten, normalize

class BaseGenerator(keras.utils.Sequence):
    '''
    Basic framework for generating 2D thread-safe data in keras. (no preprocessing and channels_last)
    Based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

    Attributes:
      list_IDs: filenames (.nii files); must be same for training and labels
      data_dirs: list of [training_dir, labels_dir]
      batch_size: int of desired number images per epoch
      n_channels: <-
      n_classes: <-
      normalize_mode: representing the type of normalization of either
          "normalize": squeezes between the specified range
          "whiten": mean var standardizes the data
          "normalize_clip": mean-var standardizes the data, then clips between [-5, 5], and squeezes the pixel values between the specified norm range
      norm_range: the specified range for normalization
      shuffle: boolean on whether or not to shuffle
    '''
    def __init__(self, list_IDs, data_dirs, batch_size, n_channels, n_classes,
                 normalize_mode = "normalize", norm_range = [0,1], shuffle = True):
        # lists of paths to images
        self.list_IDs = list_IDs
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.normalize_mode = normalize_mode
        self.norm_range = norm_range
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

        X, Y = self.data_gen(list_IDs_temp)
        return (X, Y)

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
            list_IDs_temp: temporary batched list of ids (filenames)
        Returns
            x, y
        '''
        raise NotImplementedError
