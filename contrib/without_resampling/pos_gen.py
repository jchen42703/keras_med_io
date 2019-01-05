import numpy as np
from keras_med_io.utils.gen_utils import BaseGenerator
from keras_med_io.utils.patch_utils import PatchExtractor
from keras_med_io.utils.io_func import normalization, resample_img, sanity_checks, add_channel
# from keras_med_io.utils.data_aug_deprecated import *

from random import randint
import os
import nibabel as nib

class PositivePatchGenerator(BaseGenerator, PatchExtractor):
    """
    Generating patches with only the positive class.
    Attributes:
    * list_IDs
    * data_dirs: list of training/label dirs
    * batch_size:
    * patch_shape: tuple of shape without channels
    * shuffle:
    Assumes that scans are single channel image and that they are channels_last
    * no support for overlap
    """
    def __init__(self,  list_IDs, data_dirs, batch_size, patch_shape,
                 normalize_mode = 'whitening', norm_range = [0,1], n_channels = 1, shuffle = True):
        self.list_IDs = list_IDs
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.patch_shape = patch_shape
        self.normalize_mode = normalize_mode
        self.norm_range = norm_range
        self.n_channels = n_channels
        self.shuffle = shuffle

        self.indexes = np.arange(len(self.list_IDs))
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
            # loading images
            file_x, file_y = os.path.join(self.data_dirs[0] + id), os.path.join(self.data_dirs[1] + id)
            # loads data as a numpy arr and then changes the type to float32
            x_train = nib.load(file_x).get_fdata().astype(np.float32)
            y_train = nib.load(file_y).get_fdata().astype(np.float32)
            # (#slices, h,w)
            if self.ndim == 2:
                ## picks random positive slice
                pos_slice = self.pos_slice_dict[id]
                # getting random positive slice idx
                slice_idx = int(np.random.choice(pos_slice))
                x_train = x_train[slice_idx]
                y_train = y_train[slice_idx]
                assert np.array_equal(np.unique(y_train), np.array([0,1]))

            if self.n_channels == 1:
                x_train, y_train = add_channel(x_train), add_channel(y_train)
            # print("before extraction: ", x_train.shape, y_train.shape)
            patch_x, patch_y = self.extract_pos_patches(x_train, y_train, self.patch_shape)
            # print("after extraction: ", patch_x.shape, patch_y.shape)
            patch_x = normalization(patch_x, self.normalize_mode, self.norm_range)
            assert sanity_checks(patch_x, patch_y)
            patches_x.append(patch_x), patches_y.append(patch_y)
        # return np.vstack(patches_x), np.vstack(patches_y)
        return (np.stack(patches_x), np.stack(patches_y))

    def extract_pos_patches(self, image, label, patch_shape):
        '''
        Extracts a random positive patch from a 2D/3D image/label pair
        '''
        both = np.concatenate([image, label], axis = -1)
        image_shape = image.shape[:self.ndim:]
        n_channels = image.shape[-1]

        # get random positive index on the fly
        pos_idx = np.dstack(self.get_positive_idx(label.squeeze())).squeeze()
        # print("pos_idx before dstack: ", self.get_positive_idx(label.squeeze()))
        # print("extract_pos, pos_idx: ", pos_idx, "\npos_idx shape: ", pos_idx.shape)
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
        Returns a dictionary of all positive class slice indices for images corresponding to their ID
        '''
        pos_slice_dict = {}
        for id in self.list_IDs:
            # for file_x, file_y in zip(batch_x, batch_y):
            file_y = nib.load(os.path.join(self.data_dirs[1] + id)).get_fdata().squeeze()
            pos_slice_dict[id] = self.get_positive_idx(file_y)[0]
        return pos_slice_dict
