import numpy as np
from keras_med_io.utils.gen_utils import BaseGenerator
from keras_med_io.utils.patch_utils import PatchExtractor
from keras_med_io.utils.io_func import normalize_clip, resample_img, whitening, normalize, sanity_checks, add_channel
# from keras_med_io.utils.data_aug_deprecated import *

from random import randint
import os
import nibabel as nib

class RandomPatchGenerator(BaseGenerator, PatchExtractor):
    '''
    Generating randomly cropped patches
    New Args:
        * normalize_mode: either 'whitening', 'normalize', 'normalize_clip' for output preprocessing
    Need to make compatible with multiple channels
    '''
    def __init__(self, list_IDs, data_dirs, batch_size, n_windows, patch_shape,
                 normalize_mode = 'whitening', range = [0,1], overlap = 0, n_channels = 1, shuffle = True):
        # lists of paths to images
        self.list_IDs = list_IDs
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.n_windows = n_windows
        self.patch_shape = patch_shape
        self.normalize_mode = normalize_mode
        self.range = range
        self.overlap = overlap
        self.n_channels = n_channels
        self.shuffle = shuffle

        self.ndim = len(patch_shape)
        self.indexes = np.arange(len(self.list_IDs))

    def data_gen(self, list_IDs_temp):
        '''
        Generates a batch of data.
        '''
        patches_x = []
        patches_y = []
        for id in list_IDs_temp:
            file_x, file_y = os.path.join(self.data_dirs[0] + id), os.path.join(self.data_dirs[1] + id)
            # loads data as a numpy arr and then changes the type to float32
            x_train = nib.load(file_x).get_fdata().astype(np.float32)
            y_train = nib.load(file_y).get_fdata().astype(np.float32)
            # (#slices, h,w)
            if self.ndim == 2:
                # extracting a 2D Random Slice
                slice_idx = randint(1, x_train.shape[0]-1)
                x_train = x_train[slice_idx]
                y_train = y_train[slice_idx]
                # print("slice_idx: ", slice_idx)
            if self.n_channels == 1:
                x_train, y_train = add_channel(x_train), add_channel(y_train)
            # print("x_train: ", x_train.shape, "y_train: ", y_train.shape)
            for window in self.n_windows:
                # choose random patch
                patch_x, patch_y = self.extract_random_patches(x_train, y_train, self.patch_shape)
                # print("patch_x: ", patch_x.shape, "patch_y: ", patch_y.shape)
                patch_x = self.normalization(patch_x)
                assert sanity_checks(patch_x, patch_y)
                patches_x.append(patch_x), patches_y.append(patch_y)

        return (np.stack(patches_x), np.stack(patches_y))

    def extract_random_patches(self, image, label, patch_shape):
        '''
        Takes both image and label and gets cropped random patches

        param image: 2D/3D single arr with no batch_size dim
        param label: 2D/3D single arr with no batch_size dim
        param patch_shape: 2D/3D tuple of patch shape without batch_size or n_channels
        Returns: a tuple of (cropped_image, cropped_label)

        '''
        both = np.concatenate([image, label], axis = -1)
        image_shape = image.shape[:self.ndim:]
        n_channels = image.shape[-1]

        # getting random patch index
        patch_indices = self.compute_patch_indices(image_shape, patch_shape, self.overlap)
        patch_idx = patch_indices[np.random.randint(0, patch_indices.shape[0]-1),:]
        both_crop = self.extract_patch(both, self.patch_shape, patch_idx)
        # print("both_crop: ", both_crop.shape)
        if self.ndim == 2:
            x, y = both_crop[:,:, :n_channels], both_crop[:, :, n_channels:]
        elif self.ndim == 3:
            x, y = both_crop[:,:, :, :n_channels], both_crop[:, :,:,  n_channels:]
        return x,y

    def normalization(self, patch_x):
        '''
        Normalizes the image based on the specified mode and range
        '''
        # reiniating the batch_size dimension
        if self.normalize_mode == 'whitening':
            return whitening(patch_x)
        elif self.normalize_mode == 'normalize_clip':
            return normalize_clip(patch_x, range = self.range)
        elif self.normalize_mode == 'normalize':
            return normalize(patch_x, range = self.range)
