import numpy as np
from utils.gen_utils import BaseGenerator
from utils.patch_utils import PatchExtractor
from utils.io_func import normalize_clip, resample_img, whitening, normalize
from utils.data_aug import *

from pos_gen import PositivePatchGenerator
from random_gen import RandomPatchGenerator
import keras
import SimpleITK as sitk
from SimpleITK import GetArrayFromImage, ReadImage
from random import randint
import os

class BalancedPatchGenerator(PositivePatchGenerator, RandomPatchGenerator):
    '''
    Adds the option to balance your data based on a specified number of positive patches you want to have in each batch
    New params:
        * n_pos: the number of images in batch to contain a positive class
    ** CHANNELS_LAST
        * Super hacked up to return channels_last data. Need to refactor
    '''
    def __init__(self, list_IDs, data_dirs, batch_size, patch_shape = (64,64),
                 normalize_mode = 'whitening', range = [0,1], n_pos = 1, overlap = 0, shuffle = True):
        # lists of paths to images
        self.list_IDs = list_IDs
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.patch_shape = patch_shape
        self.normalize_mode = normalize_mode
        self.range = range
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_IDs))
        self.n_pos = n_pos
        self.overlap = overlap
        # self.batch_ratio = n_pos/batch_size
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
        # pos_masks = np.concatenate([y_pos[1], y[1]])

        return (input_data, seg_labels)

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
            # loads the .nii images
            file_x = os.path.join(self.data_dirs[0] + id)
            file_y = os.path.join(self.data_dirs[1] + id)
            sitk_image, sitk_label = sitk.ReadImage(file_x), sitk.ReadImage(file_y)
            # (h,w, n_slices)

            if self.ndim == 2:
                # extracting a 2D Slice to be resampled
                if pos_sample:
                    ## picks random positive slice
                    pos_slice = self.pos_slice_dict[id]
                    # getting random positive slice idx
                    slice_idx = int(np.random.choice(pos_slice))
                    sitk_x_slice = sitk_image[:,:,slice_idx: slice_idx+1]
                    sitk_y_slice = sitk_label[:,:,slice_idx: slice_idx+1]
                elif not pos_sample:
                    # random slice
                    slice_idx = randint(1, sitk_image.GetSize()[-1]-1) # need to test
                    sitk_x_slice = sitk_image[:,:,slice_idx-1:slice_idx] #(320,320,1)
                    sitk_y_slice = sitk_label[:,:,slice_idx-1:slice_idx]
                #resample; defaults to 1mm isotropic spacing
                x_resample, y_resample = resample_img(sitk_x_slice), resample_img(sitk_y_slice, is_label = True)
                channels_last = [1,2,0]
                x_train = np.transpose(GetArrayFromImage(x_resample), channels_last).astype(np.float32)
                y_train = np.transpose(GetArrayFromImage(y_resample), channels_last).astype(np.float32)
            elif self.ndim == 3:
                x_resample, y_resample = resample_img(sitk_image), resample_img(sitk_label, is_label = True)
                x_train = np.expand_dims(GetArrayFromImage(x_resample), -1).astype(np.float32)
                y_train = np.expand_dims(GetArrayFromImage(y_resample), -1).astype(np.float32)
            # print("before extraction: ", x_train.shape, y_train.shape)
            if pos_sample:
                patch_x, patch_y = self.extract_pos_patches(x_train, y_train, self.patch_shape)
            elif not pos_sample:
                patch_x, patch_y = self.extract_random_patches(x_train, y_train, self.patch_shape)

            # reiniating the batch_size dimension
            if self.normalize_mode == 'whitening':
                patch_x = whitening(np.expand_dims(patch_x, axis = 0))
            elif self.normalize_mode == 'normalize_clip':
                patch_x = normalize_clip(np.expand_dims(patch_x, axis = 0), range = self.range)
            elif self.normalize_mode == 'normalize':
                patch_x = normalize(np.expand_dims(patch_x, axis = 0), range = self.range)
            patch_y = np.expand_dims(patch_y, axis = 0)

            # sanity checks
            assert not np.any(np.isnan(patch_x)) and not np.any(np.isnan(patch_y))
            assert np.array_equal(np.unique(patch_y), np.array([0,1])) or np.array_equal(np.unique(patch_y), np.array([0]))

            patches_x.append(patch_x), patches_y.append(patch_y)

        input_data = np.vstack(patches_x)
        seg_masks = np.vstack(patches_y)
        # pos_masks = seg_masks * input_data
        # return np.vstack(patches_x), np.vstack(patches_y)
        return (input_data, seg_masks)
