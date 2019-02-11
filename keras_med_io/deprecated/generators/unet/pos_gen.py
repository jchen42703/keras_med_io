import numpy as np
from keras_med_io.utils.gen_utils import BaseGenerator
from keras_med_io.utils.patch_utils import PosRandomPatchExtractor
from keras_med_io.utils.io_func import normalization, resample_img, get_multi_class_labels
# from keras_med_io.utils.data_aug_deprecated import *

import keras
import SimpleITK as sitk
from SimpleITK import GetArrayFromImage, ReadImage
from random import randint
import os

class PositivePatchGenerator(BaseGenerator, RandomPosPatchExtractor):
    """
    Generating patches with only the positive class.
    Attributes:
        list_IDs: list of filenames
        data_dirs: list of paths to both the input dir and labels dir
        batch_size: The number of images you want in a single batch
        patch_shape: tuple of patch shape without the number of channels
        normalize_mode: representing the type of normalization of either
            "normalize": squeezes between the specified range
            "whiten": mean var standardizes the data
            "normalize_clip": mean-var standardizes the data, then clips between [-5, 5], and squeezes the pixel values between the specified norm range
        start: int
        norm_range: the specified range for normalization
        shuffle: boolean
    """
    def __init__(self,  list_IDs, data_dirs, batch_size, patch_shape,
                 normalize_mode = 'whiten', norm_range = [0,1], start = None, shuffle = True):
        BaseGenerator.__init__(self, list_IDs = list_IDs, data_dirs = data_dirs, batch_size = batch_size,
                               n_channels = n_channels, n_classes = n_classes, normalize_mode = normalize_mode,
                               norm_range = norm_range, shuffle = shuffle)
        self.patch_shape = patch_shape
        self.ndim = len(patch_shape)
        self.indexes = np.arange(len(self.list_IDs))

        if self.ndim == 2: # to make sure that it only intializes when necessary
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
            patch_x, patch_y = self.extract_posrandom_patches(x_train, y_train, self.patch_shape, pos_sample = True)
            # print("after extraction: ", patch_x.shape, patch_y.shape)
            patch_x = normalization(patch_x, self.normalize_mode, self.norm_range)
            if self.n_classes > 2: # no point to run this when binary (foreground/background)
                patch_y = get_multi_class_labels(patch_y, n_labels = self.n_classes, remove_background = True)
            assert sanity_checks(patch_x, patch_y)
            patches_x.append(patch_x), patches_y.append(patch_y)
        return (np.stack(patches_x), np.stack(patches_y))

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
