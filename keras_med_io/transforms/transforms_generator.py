import numpy as np
from keras_med_io.utils.gen_utils import BaseGenerator
from keras_med_io.utils.patch_utils import PosRandomPatchExtractor
from keras_med_io.utils.io_func import sanity_checks, add_channel, \
                                       get_multi_class_labels, load_data, reshape
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
        n_classes: number of unique labels excluding the background class (i.e. binary; n_classes = 1)
        ndim: number of dimensions of the input (excluding the batch_size and n_channels axes)
        transform (Transform instance): If you want to use multiple Transforms, use the Compose Transform.
        max_patient_shape: a tuple representing the maximum patient shape in a dataset; i.e. (x,y, (z,))
        shuffle: boolean
    """
    def __init__(self, list_IDs, data_dirs, batch_size, n_channels, n_classes, ndim,
                transform = None, max_patient_shape = (144, 255, 319), shuffle = True):

        BaseGenerator.__init__(self, list_IDs = list_IDs, data_dirs = data_dirs, batch_size = batch_size,
                               n_channels = n_channels, n_classes = n_classes, shuffle = shuffle)
        self.ndim = ndim
        self.transform = transform
        self.max_patient_shape = max_patient_shape
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
        if self.transform is not None:
            X, Y = self._apply_transform(X, Y)
        return (X, Y)

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
        images_x = []
        images_y = []
        for id in list_IDs_temp:
            # loads data as a numpy arr and then changes the type to float32
            x_train = load_data(os.path.join(self.data_dirs[0], id))
            y_train = load_data(os.path.join(self.data_dirs[1], id))
            if not x_train.shape[-1] == self.n_channels:
                # Adds channel in case there is no channel dimension
                x_train = add_channel(x_train)
                assert len(x_train.shape) == self.ndim + 1, "Input shape must be the \
                                                            shape (x,y, n_channels) or (x, y, z, n_channels)"
            if not y_train.shape[-1] == self.n_channels:
                # Adds channel in case there is no channel dimension
                y_train = add_channel(y_train)
                assert len(y_train.shape) == self.ndim + 1, "Input labels must be the \
                                                            shape (x,y, n_channels) or (x, y, z, n_channels)"
            if self.n_classes > 1: # no point to run this when binary (foreground/background)
                y_train = get_multi_class_labels(y_train, n_labels = self.n_classes, remove_background = True)
            # Padding to the max patient shape (so the arrays can be stacked)
            x_train = reshape(x_train, x_train.min(), self.max_patient_shape + (self.n_channels, ))
            y_train = reshape(y_train, 0, self.max_patient_shape + (self.n_classes, ))

                # x_train.resize(max_patient_shape + (self.n_channels, )), y_train.resize(max_patient_shape + (self.n_classes, ))
            assert sanity_checks(x_train, y_train)
            images_x.append(x_train), images_y.append(y_train)

        input_data, seg_masks = np.stack(images_x), np.stack(images_y)
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
