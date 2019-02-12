import numpy as np
from keras_med_io.utils.gen_utils import BaseGenerator
from keras_med_io.utils.patch_utils import PosRandomPatchExtractor
from keras_med_io.utils.io_func import sanity_checks, add_channel, \
                                       get_multi_class_labels, load_data
import nibabel as nib
from random import randint
import os

class BaseTransformGenerator(BaseGenerator):
    """
    Loads data and applies data augmentation with `batchgenerators.transforms`.
    * Supports both channels_last and channels_first
    * Loads data WITH nibabel instead of SimpleITK
        * .nii files should not have the batch_size dimension

    Attributes:
        list_IDs: list of filenames
        data_dirs: list of paths to both the input dir and labels dir
        batch_size: The number of images you want in a single batch
        n_channels: number of channels
        n_classes: number of classes for one hot encoding (>=2)
        ndim: number of dimensions of the input (excluding the batch_size and n_channels axes)
        transform (Transform instance): If you want to use multiple Transforms, use the Compose Transform.
        shuffle: boolean
    """
    def __init__(self, list_IDs, data_dirs, batch_size, n_channels, n_classes, ndim,
                transform, shuffle = True):

        BaseGenerator.__init__(self, list_IDs = list_IDs, data_dirs = data_dirs, batch_size = batch_size,
                               n_channels = n_channels, n_classes = n_classes, shuffle = shuffle)
        self.transform = transform
        self.ndim = ndim
        self.indexes = np.arange(len(self.list_IDs))

    def __getitem__(self, idx):
        """
        Defines the fetching and on-the-fly preprocessing of data.
        Returns a batch of data (x,y)
        """
        # file names
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        # Fetches batched IDs for a thread
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, Y = self.data_gen(list_IDs_temp)
        return self._apply_transform(X, Y)

    def _apply_transform(self, X, Y):
        data_dict = {}
        # batchgenerator transforms only accept channels_first
        data_dict["data"] = self._convert_dformat(X, convert_to = "channels_first")
        data_dict["seg"] = self._convert_dformat(Y, convert_to = "channels_first")
        data_dict = self.transform(**data_dict)
        # Desired target models accept channels_last dformat data (change this for your needs as you'd like)
        return (self._convert_dformat(data_dict["data"], convert_to = "channels_last"),
                self._convert_dformat(data_dict["seg"], convert_to = "channels_last"))

    def data_gen(self, list_IDs_temp):
        """
        Generates a batch of data.
        Args:
            list_IDs_temp: batched list IDs; usually done by __getitem__
            pos_sample: boolean on if you want to sample a positive image or not
        Returns:
            tuple of two numpy arrays: x, y
        """
        patches_x = []
        patches_y = []
        for id in list_IDs_temp:
            # loads data as a numpy arr and then changes the type to float32
            x_train = load_data(os.path.join(self.data_dirs[0], id))
            y_train = load_data(os.path.join(self.data_dirs[1], id))
            if not x_train.shape[-1] == self.n_channels:
                # Adds channel in case there is no channel dimension
                x_train, y_train = add_channel(x_train), add_channel(y_train)
                assert len(x_train.shape) == self.ndim + 1, "Input shape must be the \
                                                            shape (x,y, n_channels) or (x, y, z, n_channels)"
            if self.n_classes > 2: # no point to run this when binary (foreground/background)
                patch_y = get_multi_class_labels(patch_y, n_labels = self.n_classes, remove_background = True)
            assert sanity_checks(patch_x, patch_y)
            patches_x.append(patch_x), patches_y.append(patch_y)

        input_data, seg_masks = np.stack(patches_x), np.stack(patches_y)
        return (input_data, seg_masks)

    def _convert_dformat(self, arr, convert_to = "channels_last"):
        """
        Args:
            arr: numpy array of shape (batch_size, x, y(,z), n_channels) (could be 4D or 5D)
            convert_to: desired data format to convert `arr` to; either "channels_last" or "channels_first"
        """
        # converting to channels_first
        if convert_to == "channels_first":
            if self.ndim == 2:
                axes_list = [0, -1, 1,2]
            elif self.ndim == 3:
                axes_list = [0, -1, 1, 2, 3]
        # converting to channels_last
        elif convert_to == "channels_last":
            if self.ndim == 2:
                axes_list = [0, 2, 3, 1]
            elif self.ndim == 3:
                axes_list = [0, 2, 3, 4, 1]
        else:
            raise Exception("Please choose a compatible data format: 'channels_last' or 'channels_first'")
        return np.transpose(arr, axes_list)
