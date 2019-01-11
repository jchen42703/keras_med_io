import numpy as np
from keras_med_io.utils.gen_utils import BaseGenerator
from keras_med_io.utils.patch_utils import PosRandomPatchExtractor
from keras_med_io.utils.io_func import normalization, sanity_checks, add_channel
# from keras_med_io.utils.data_aug_deprecated import *

from random import randint
import os
import nibabel as nib

class PositivePatchGenerator(BaseGenerator, PosRandomPatchExtractor):
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
    def __init__(self,  list_IDs, data_dirs, batch_size, patch_shape, n_channels, n_classes,
                 normalize_mode = 'whiten', norm_range = [0,1], shuffle = True):
        BaseGenerator.__init__(self, list_IDs = list_IDs, data_dirs = data_dirs, batch_size = batch_size,
                               n_channels = n_channels, n_classes = n_classes, normalize_mode = normalize_mode,
                               norm_range = norm_range, shuffle = shuffle)
        self.patch_shape = patch_shape
        self.ndim = len(patch_shape)
        self.indexes = np.arange(len(self.list_IDs))

        if self.ndim == 2: # to make sure that it only intializes when necessary
            self.pos_slice_dict = self.get_pos_slice_dict()

    def data_gen(self, list_IDs_temp):
        """
        Generates a batch of data.
        Args:
            list_IDs_temp: batched list IDs; usually done by __getitem__
        Returns:
            tuple of two numpy arrays: x, y
        """
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
                # picks random positive slice
                pos_slice = self.pos_slice_dict[id]
                slice_idx = int(np.random.choice(pos_slice))
                x_train = x_train[slice_idx]
                y_train = y_train[slice_idx]
                assert np.array_equal(np.unique(y_train), np.array([0,1]))

            if not x_train.shape[-1] == self.n_channels:
                # Adds channel in case there is no channel dimension
                x_train, y_train = add_channel(x_train), add_channel(y_train)
                assert len(x_train.shape) == self.ndim + 1, "Input shape must be the \
                                                            shape (x,y, n_channels) or (x, y, z, n_channels)"
            # print("before extraction: ", x_train.shape, y_train.shape)
            patch_x, patch_y = self.extract_posrandom_patches(x_train, y_train, self.patch_shape, pos_sample = True)
            # print("after extraction: ", patch_x.shape, patch_y.shape)
            patch_x = normalization(patch_x, self.normalize_mode, self.norm_range)
            assert sanity_checks(patch_x, patch_y)
            patches_x.append(patch_x), patches_y.append(patch_y)
        # return np.vstack(patches_x), np.vstack(patches_y)
        return (np.stack(patches_x), np.stack(patches_y))

    def get_pos_slice_dict(self):
        """
        Returns a dictionary of all positive class slice indices for images corresponding to their ID
        """
        pos_slice_dict = {}
        for id in self.list_IDs:
            label = nib.load(os.path.join(self.data_dirs[1] + id)).get_fdata().squeeze()
            pos_slice_dict[id] = self.get_positive_idx(label, dstack = False)[0] #producing slice indices for each id
        return pos_slice_dict
