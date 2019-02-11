import numpy as np
from keras_med_io.utils.gen_utils import BaseGenerator
from keras_med_io.utils.patch_utils import PosRandomPatchExtractor
from keras_med_io.utils.io_func import normalization, sanity_checks, add_channel, get_multi_class_labels, transforms
import nibabel as nib
from random import randint
import os

class PosRandomPatchGenerator(PosRandomPatchExtractor, BaseGenerator):
    """
    Adds the option to balance your data based on a specified number of positive patches you want to have in each batch
    New params:
        * n_pos: the number of images in batch to contain a positive class
    ** CHANNELS_LAST
    ** LOADS DATA WITH Nibabel instead of SimpleITK
        * .nii files should not have the batch_size dimension

    Attributes:
        list_IDs: list of filenames
        data_dirs: list of paths to both the input dir and labels dir
        batch_size: The number of images you want in a single batch
        patch_shape: tuple of patch shape without the number of channels
        n_channels: number of channels
        n_classes: number of classes for one hot encoding (>=2)

        mode: specifying which type of data generation (optional, default: "bal")
            "bal": generating a mixed batch of random/pos patches based on n_pos; n_pos > 0
            "rand": generating only random patches
            "pos": generating only positive patches
        n_pos: int representing the number of positive samples in a batch (optional, default: 1)
        overlap: integer representing the number of pixel overlap desired for patch overlapping (optional, default: 0)
        transforms_args: a dictionary mapping the "keras_med_io.utils.io_func.transforms" arguments to the desired values
            It should not include volume, segmentation, and ndim as arguments. (optional, default: None)
        shuffle: boolean
    """
    def __init__(self, list_IDs, data_dirs, batch_size, patch_shape, n_channels, n_classes,
                 mode = "bal", n_pos = 1, overlap = 0, transforms_args = None, shuffle = True):

        BaseGenerator.__init__(self, list_IDs = list_IDs, data_dirs = data_dirs, batch_size = batch_size,
                               n_channels = n_channels, n_classes = n_classes, shuffle = shuffle)
        self.patch_shape = patch_shape
        self.ndim = len(patch_shape)
        self.mode = mode.lower()
        self.n_pos = n_pos
        self.overlap = overlap
        self.indexes = np.arange(len(self.list_IDs))
        self.transforms_args = transforms_args
        if self.ndim == 2 and not self.mode == "rand": # to make sure that it only intializes when necessary
            self.pos_slice_dict = self.get_pos_slice_dict()
        if self.mode == "bal" and self.n_pos < 1:
            raise Exception("There must a least one positive image in the balanced patch mode (bal). ")

    def __getitem__(self, idx):
        """
        Defines the fetching and on-the-fly preprocessing of data.
        Returns a batch of data (x,y)
        """
        # file names
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        # Fetches batched IDs for a thread
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        if self.mode == "bal":
            # generating data for both positive and randomly sampled data
            X_pos, Y_pos = self.data_gen(list_IDs_temp[:self.n_pos], pos_sample = True)
            X_rand, Y_rand = self.data_gen(list_IDs_temp[self.n_pos:], pos_sample = False)
            # concatenating all the corresponding data
            X, Y = np.concatenate([X_pos, X_rand], axis = 0), np.concatenate([Y_pos, Y_rand], axis = 0)
            # shuffling the order of the positive/random patches
            out_rand_indices = np.arange(0, X.shape[0])
            np.random.shuffle(out_rand_indices)
            X, Y = X[out_rand_indices], Y[out_rand_indices]
        # generating a batch of random patches
        elif self.mode == "rand":
            X, Y = self.data_gen(list_IDs_temp, pos_sample = False)
        # generating a batch of positive only patches
        elif self.mode == "pos":
            X, Y = self.data_gen(list_IDs_temp, pos_sample = True)
        # data augmentation
        if self.transforms_args is not None:
            X, Y = transforms(X, Y, ndim = self.ndim, **self.transforms_args)
        return (X,Y)

    def data_gen(self, list_IDs_temp, pos_sample):
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
            x_train = nib.load(os.path.join(self.data_dirs[0] + id)).get_fdata().astype(np.float32)
            y_train = nib.load(os.path.join(self.data_dirs[1] + id)).get_fdata().astype(np.float32)
            # (...., n_channels)
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

            if not x_train.shape[-1] == self.n_channels:
                # Adds channel in case there is no channel dimension
                x_train, y_train = add_channel(x_train), add_channel(y_train)
                assert len(x_train.shape) == self.ndim + 1, "Input shape must be the \
                                                            shape (x,y, n_channels) or (x, y, z, n_channels)"
            patch_x, patch_y = self.extract_posrandom_patches(x_train, y_train, self.patch_shape, pos_sample)
            if self.n_classes > 2: # no point to run this when binary (foreground/background)
                patch_y = get_multi_class_labels(patch_y, n_labels = self.n_classes, remove_background = True)
            assert sanity_checks(patch_x, patch_y)
            # data augmentation
            patches_x.append(patch_x), patches_y.append(patch_y)

        input_data, seg_masks = np.stack(patches_x), np.stack(patches_y)
        return (input_data, seg_masks)

    def get_pos_slice_dict(self):
        """
        Returns a dictionary of all positive class slice indices for images corresponding to their ID
        """
        pos_slice_dict = {}
        for id in self.list_IDs:
            label = nib.load(os.path.join(self.data_dirs[1] + id)).get_fdata().squeeze()
            pos_slice_dict[id] = self.get_positive_idx(label, dstack = False)[0] #producing slice indices for each id
        return pos_slice_dict
