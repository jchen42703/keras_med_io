import numpy as np
# from gen_utils import BaseGenerator
# from patch_utils import PatchExtractor
# from io_func import normalize_clip, resample_img, whitening, normalize
# from data_aug import *

# from CapsNetsMRI.gen_utils import BaseGenerator
# from CapsNetsMRI.patch_utils import PatchExtractor
# from CapsNetsMRI.io_func import normalize_clip, resample_img, whitening, normalize
# from CapsNetsMRI.data_aug import *

import keras
import SimpleITK as sitk
from SimpleITK import GetArrayFromImage, ReadImage
from random import randint
import os

class RandomPatchGenerator(BaseGenerator, PatchExtractor):
    '''
    Generating randomly cropped patches
    New Args:
        * normalize_mode: either 'whitening', 'normalize', 'normalize_clip' for output preprocessing
    '''
    def __init__(self, list_IDs, data_dirs, batch_size, image_format = "channels_last", patch_shape = (64,64),
                 normalize_mode = 'whitening', range = [0,1], shuffle = True):
        # lists of paths to images
        self.list_IDs = list_IDs
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.image_format = image_format
        self.ndim = len(patch_shape)

        self.patch_shape = patch_shape
        self.normalize_mode = normalize_mode
        self.range = range
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_IDs))
        self.overlap = 0

    def data_gen(self, list_IDs_temp):
        '''generates the data'''
        patches_x = []
        patches_y = []
        for id in list_IDs_temp:
            # for file_x, file_y in zip(batch_x, batch_y):
            file_x = os.path.join(self.data_dirs[0] + id)
            file_y = os.path.join(self.data_dirs[1] + id)
            sitk_image, sitk_label = sitk.ReadImage(file_x), sitk.ReadImage(file_y)
            # (h,w, n_slices)
            if self.ndim == 2:
                slice_idx = randint(1, sitk_image.GetSize()[-1]-1) # need to test
                sitk_x_slice = sitk_image[:,:,slice_idx-1:slice_idx] #(320,320,1)
                sitk_y_slice = sitk_label[:,:,slice_idx-1:slice_idx]
                #resample; defaults to 1mm isotropic spacing
                x_resample, y_resample = resample_img(sitk_x_slice), resample_img(sitk_y_slice, is_label = True)
                # converting to numpy arrays
                if self.image_format == "channels_last":
                    x_train = np.expand_dims(GetArrayFromImage(x_resample), -1).astype(np.float32).squeeze(axis = 0)
                    y_train = np.expand_dims(GetArrayFromImage(y_resample), -1).astype(np.float32).squeeze(axis = 0)
                elif self.image_format == "channels_first":
                    x_train = GetArrayFromImage(x_resample)#np.expand_dims(GetArrayFromImage(x_resample), 0).astype(np.float32).squeeze(axis = 0)
                    y_train = GetArrayFromImage(y_resample)#np.expand_dims(GetArrayFromImage(y_resample), 0).astype(np.float32).squeeze(axis = 0)
                assert x_train.shape == y_train.shape

            elif self.ndim == 3:
                x_resample, y_resample = resample_img(sitk_image), resample_img(sitk_label, is_label = True)
                # converting to numpy arrays
                if self.image_format == "channels_last":
                    x_train = np.expand_dims(GetArrayFromImage(x_resample), -1).astype(np.float32)#.squeeze(axis = 0)
                    y_train = np.expand_dims(GetArrayFromImage(y_resample), -1).astype(np.float32)#.squeeze(axis = 0)
                elif self.image_format == "channels_first":
                    x_train = np.expand_dims(GetArrayFromImage(x_resample), 0).astype(np.float32)#.squeeze(axis = -1)
                    y_train = np.expand_dims(GetArrayFromImage(y_resample), 0).astype(np.float32)#.squeeze(axis = -1)
                assert x_train.shape == y_train.shape

            # print("x_train: ", x_train.shape, "y_train: ", y_train.shape)
            # choose random index
            patch_x, patch_y = self.extract_random_patch(x_train, y_train, self.patch_shape)
            # print("patch_x: ", patch_x.shape, "patch_y: ", patch_y.shape)
            # reiniating the batch_size dimension
            if self.normalize_mode == 'whitening':
                patch_x = whitening(np.expand_dims(patch_x, axis = 0))
            elif self.normalize_mode == 'normalize_clip':
                patch_x = normalize_clip(np.expand_dims(patch_x, axis = 0), range = self.range)
            elif self.normalize_mode == 'normalize':
                patch_x = normalize(np.expand_dims(patch_x, axis = 0), range = self.range)
            patch_y = np.expand_dims(patch_y, axis = 0)
            # tests
            assert not np.any(np.isnan(patch_x)) and not np.any(np.isnan(patch_y))
            assert np.array_equal(np.unique(patch_y), np.array([0,1])) or np.array_equal(np.unique(patch_y), np.array([0]))

            patches_x.append(patch_x), patches_y.append(patch_y)

        return (np.vstack(patches_x), np.vstack(patches_y))

    def extract_random_patch(self, image, label, patch_shape):
        '''
        Takes both image and label and gets cropped random patches

        param image: 2D/3D single arr with no batch_size dim
        param label: 2D/3D single arr with no batch_size dim
        param patch_shape: 2D/3D tuple of patch shape without batch_size or n_channels
        Returns: a tuple of (cropped_image, cropped_label)

        '''
        if self.image_format == "channels_last":
            both = np.concatenate([image, label], axis = -1)
            image_shape = image.shape[:self.ndim:]
            n_channels = image.shape[-1]

        elif self.image_format == "channels_first":
            both = np.concatenate([image, label], axis = 0)
            image_shape = image.shape[-self.ndim:]
            n_channels = image.shape[0]

        # getting random patch index
        patch_indices = self.compute_patch_indices(image_shape, patch_shape, self.overlap, ndim = self.ndim)
        patch_idx = patch_indices[np.random.randint(0, patch_indices.shape[0]-1),:]
        both_crop = self.extract_patch(both, self.patch_shape, patch_idx)
        # print("both_crop: ", both_crop.shape)
        if self.ndim == 2:
            if self.image_format == "channels_last":
                x, y = both_crop[:,:, :n_channels], both_crop[:, :, n_channels:]
            elif self.image_format == "channels_first":
                x, y = both_crop[:n_channels, :,:, ], both_crop[n_channels:, :, :]

        elif self.ndim == 3:
            if self.image_format == "channels_last":
                x, y = both_crop[:,:, :, :n_channels], both_crop[:, :,:,  n_channels:]
            elif self.image_format == "channels_first":
                x, y = both_crop[:n_channels, :,:,:], both_crop[n_channels:, :, :, :]
        return x,y
