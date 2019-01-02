import numpy as np
from keras_med_io.utils.gen_utils import BaseGenerator
from keras_med_io.utils.patch_utils import PatchExtractor
from keras_med_io.utils.io_func import normalize_clip, resample_img, whitening, normalize, sanity_checks, add_channel
# from keras_med_io.utils.data_aug_deprecated import *

from keras_med_io.contrib.without_resampling.pos_gen import PositivePatchGenerator
from keras_med_io.contrib.without_resampling.random_gen import RandomPatchGenerator

import nibabel as nib
from random import randint
import os

class BalancedPatchGenerator(PositivePatchGenerator, RandomPatchGenerator):
    '''
    Adds the option to balance your data based on a specified number of positive patches you want to have in each batch
    New params:
        * n_pos: the number of images in batch to contain a positive class
    ** CHANNELS_LAST
    ** LOADS DATA WITH Nibabel

    Attributes:
        list_IDs: list of filenames
        data_dirs: list of paths to both the input dir and labels dir
        batch_size: The number of images you want in a single batch
        patch_shape: tuple of patch shape without the number of channels
        normalize_mode: representing the type of normalization of either
            "normalize": squeezes between the specified range
            "whitening": mean var standardizes the data
            "normalize_clip": mean-var standardizes the data, then clips between [-5, 5], and squeezes the pixel values between the specified norm range
        range: the specified range for normalization
        n_pos: number of positive samples in a batch
        overlap: number of pixel overlap desired for patch overlapping
        n_channels: number of channels
        shuffle: boolean
    '''
    def __init__(self, list_IDs, data_dirs, batch_size, patch_shape = (64,64),
                 normalize_mode = 'whitening', range = [0,1], n_pos = 1, overlap = 0, n_channels = 1, shuffle = True):
        # lists of paths to images
        self.list_IDs = list_IDs
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.patch_shape = patch_shape
        self.normalize_mode = normalize_mode
        self.range = range
        self.n_pos = n_pos
        self.overlap = overlap
        self.n_channels = n_channels
        self.shuffle = shuffle

        self.indexes = np.arange(len(self.list_IDs))
        self.ndim = len(patch_shape)
        if self.ndim == 2:
            self.pos_slice_dict = self.get_pos_slice_dict()

    def __getitem__(self, idx):
        '''
        Defines the fetching and on-the-fly preprocessing of data.
        Returns a batch of data (x,y)
        '''
        # file names
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # generating data for both positive and randomly sampled data
        X_pos, y_pos = self.data_gen(list_IDs_temp[:self.n_pos], pos_sample = True)
        assert not X_pos[0].shape[0] == 0, "There must be at least 1 positive image, or else use the RandomPatchGenerator."

        X, y = self.data_gen(list_IDs_temp[self.n_pos:], pos_sample = False)
        # concatenating all the corresponding data
        input_data = np.concatenate([X_pos, X])
        seg_labels = np.concatenate([y_pos, y])

        out_rand_indices = np.arange(0, input_data.shape[0])
        np.random.shuffle(out_rand_indices)
        # print(out_rand_indices)
        # return (input_data[out_rand_indices].squeeze(axis = 0), seg_labels[out_rand_indices].squeeze(axis = 0))
        return (input_data[out_rand_indices], seg_labels[out_rand_indices])

    def data_gen(self, list_IDs_temp, pos_sample):
        '''
        generates the data [2D]
        param list_IDs_temp: batched list IDs; usually done by __getitem__
        param pos_sample: boolean on if you want to sample a positive image or not
        Returns:
            tuple of two lists: x, y
        '''
        patches_x = []
        patches_y = []
        for id in list_IDs_temp:
            # loading images
            file_x, file_y = os.path.join(self.data_dirs[0] + id), os.path.join(self.data_dirs[1] + id)
            # loads data as a numpy arr and then changes the type to float32
            x_train = nib.load(file_x).get_fdata().astype(np.float32)
            y_train = nib.load(file_y).get_fdata().astype(np.float32)
            # (h,w, n_slices)
            if self.ndim == 2:
                # extracting a 2D Slice to be resampled
                if pos_sample:
                    ## picks random positive slice
                    pos_slice = self.pos_slice_dict[id]
                    # getting random positive slice idx
                    slice_idx = int(np.random.choice(pos_slice))
                elif not pos_sample:
                    # random slice
                    slice_idx = randint(1, x_train.shape[0]-1)
                x_train = x_train[slice_idx]
                y_train = y_train[slice_idx]
                # print("slice_idx: ", slice_idx)
            if self.n_channels == 1:
                x_train, y_train = add_channel(x_train), add_channel(y_train)
            # print("before extraction: ", x_train.shape, y_train.shape)
            if pos_sample:
                patch_x, patch_y = self.extract_pos_patches(x_train, y_train, self.patch_shape)
            elif not pos_sample:
                patch_x, patch_y = self.extract_random_patches(x_train, y_train, self.patch_shape)

            patch_x = self.normalization(patch_x)
            assert sanity_checks(patch_x, patch_y)

            patches_x.append(patch_x), patches_y.append(patch_y)

        input_data = np.stack(patches_x)
        seg_masks = np.stack(patches_y)
        # pos_masks = seg_masks * input_data
        return (input_data, seg_masks)

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
