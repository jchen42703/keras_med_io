import numpy as np
from utils.gen_utils import BaseGenerator
from utils.patch_utils import PatchExtractor
from utils.io_func import normalize_clip, resample_img, whitening, normalize
from utils.data_aug import *

import keras
import SimpleITK as sitk
from SimpleITK import GetArrayFromImage, ReadImage
from random import randint
import os

class PositivePatchGenerator(BaseGenerator, PatchExtractor):
    """
    Generating patches with only the positive class.
    Attributes:
    * list_IDs
    * data_dirs: list of training/label dirs
    * batch_size:
    * patch_shape: tuple of shape without channels
    * overlap:
    * start:
    * shuffle:
    Assumes that scans are single channel image and that they are channels_last
    * no support for overlap
    """
    def __init__(self,  list_IDs, data_dirs, batch_size, patch_shape,
                 normalize_mode = 'whitening', range = [0,1], start = None, shuffle = True):
        self.list_IDs = list_IDs
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.patch_shape = patch_shape
        self.normalize_mode = normalize_mode
        self.range = range
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_IDs))

        self.start = start
        self.ndim = len(patch_shape)

        if self.ndim == 2:
            self.pos_slice_dict = self.get_pos_slice_dict()

    def data_gen(self, list_IDs_temp):
        '''
        generates the data [2D]
        '''
        patches_x = []
        patches_y = []
        for id in list_IDs_temp:
            file_x = os.path.join(self.data_dirs[0] + id)
            file_y = os.path.join(self.data_dirs[1] + id)
            sitk_image, sitk_label = sitk.ReadImage(file_x), sitk.ReadImage(file_y)
            # (h,w, n_slices)
            if self.ndim == 2:
                ## picks random positive slice
                pos_slice = self.pos_slice_dict[id]
                # getting random positive slice idx
                slice_idx = int(np.random.choice(pos_slice))
                sitk_x_slice = sitk_image[:,:,slice_idx: slice_idx+1] #(320,320,1)
                sitk_y_slice = sitk_label[:,:,slice_idx: slice_idx+1]
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
            patch_x, patch_y = self.extract_pos_patches(x_train, y_train, self.patch_shape)
            # print("after extraction: ", patch_x.shape, patch_y.shape)
            patch_x = self.normalization(patch_x)
            assert self.sanity_checks(patch_x, patch_y)
            patches_x.append(patch_x), patches_y.append(patch_y)
        # return np.vstack(patches_x), np.vstack(patches_y)
        return (np.stack(patches_x), np.stack(patches_y))

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

    def sanity_checks(self, patch_x, patch_y):
        '''
        Checks for NaNs, and makes sure that the labels are one-hot encoded
        '''
        # sanity checks
        assert not np.any(np.isnan(patch_x)) and not np.any(np.isnan(patch_y))
        assert np.array_equal(np.unique(patch_y), np.array([0,1])) or np.array_equal(np.unique(patch_y), np.array([0]))
        return True

    def extract_pos_patches(self, image, label, patch_shape):
        '''
        Extracts a random positive patch from a 2D/3D image/label pair
        '''
        both = np.concatenate([image, label], axis = -1)
        image_shape = image.shape[:self.ndim:]
        n_channels = image.shape[-1]

        # get random positive index on the fly
        pos_idx = np.dstack(self.get_positive_idx(label.squeeze())).squeeze()
        patch_idx = pos_idx[np.random.randint(0, pos_idx.shape[0]-1),:]

        both_crop = self.extract_patch(both, self.patch_shape, patch_idx)
        # print("both_crop: ", both_crop.shape, "image: ", image.shape, 'ndim', self.ndim)
        if self.ndim == 2:
            x, y = both_crop[:,:, :n_channels], both_crop[:, :, n_channels:]
        elif self.ndim == 3:
            x, y = both_crop[:,:, :, :n_channels], both_crop[:, :,:,  n_channels:]
        return x,y

    def get_positive_idx(self, label):
        '''
        Returns "n" numpy arrays of all possible positive pixel indices for a specific label, where n is the number of dimensions
        '''
        pos_idx = np.nonzero(label)
        return pos_idx

    def get_pos_slice_dict(self):
        '''
        Returns a dictionary of all positive class slice indices for pre-resampled images corresponding to their ID
        '''
        pos_slice_dict = {}
        for id in self.list_IDs:
            # for file_x, file_y in zip(batch_x, batch_y):
            file_y = sitk.ReadImage(os.path.join(self.data_dirs[1] + id))
            #resample; defaults to 1mm isotropic spacing
            pos_slice_dict[id] = self.get_positive_idx(GetArrayFromImage(file_y))[0]
        return pos_slice_dict
