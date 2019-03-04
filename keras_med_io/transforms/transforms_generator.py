import numpy as np
from keras_med_io.utils.gen_utils import BaseGenerator
# from keras_med_io.utils.patch_utils import PosRandomPatchExtractor
from keras_med_io.transforms.custom_augmentations import get_positive_idx, get_random_slice_idx
from keras_med_io.utils.io_func import sanity_checks, add_channel, \
                                       get_multi_class_labels, load_data, reshape
import nibabel as nib
from random import randint
import os

class BaseTransformGenerator(BaseGenerator):
    """
    Loads data and applies data augmentation with `batchgenerators.transforms`.
    * Supports channels_last
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
                transform = None, max_patient_shape = None, n_workers = 1, shuffle = True):

        BaseGenerator.__init__(self, list_IDs = list_IDs, data_dirs = data_dirs, batch_size = batch_size,
                               n_channels = n_channels, n_classes = n_classes, shuffle = shuffle)
        self.ndim = ndim
        self.transform = transform
        self.max_patient_shape = max_patient_shape
        if max_patient_shape is None:
            self.max_patient_shape = self.compute_max_patient_shape()
        n_samples = len(self.list_IDs)
        self.indexes = np.arange(n_samples)
        max_n_idx = batch_size * (n_workers + 2) # default for now

        # Handles cases where the dataset is small and the batch size is high
        if max_n_idx > n_samples:
            print("Adjusting the indexes since the batch size [adjusted with the n_workers] is greater than the number of images.")
            self.adjust_indexes(max_n_idx)
            print("Done!")

    def adjust_indexes(self, max_n_idx):
        """
        Adjusts the indexes when the batch size [adjusted with the n_workers] is greater than the number of images.
        This is primarily used to make sure that there will always be enough indices for the generator to use during multiprocessing.
        Args:
            max_n_idx: The current maximum size of self.indexes, which is based on the current max `idx` from `__getitem__`, so it
            should be `(idx + 1) * batch_size` or some variant of that.
        Returns:
            Nothing
        """
        if max_n_idx < self.indexes.size:
            print("WARNING! The max_n_idx should be larger than the current number of indexes or else there's no point in using this \n\
            function. It has been automatically adjusted for you.")
            max_n_idx = self.batch_size * max_n_idx
        # expanding the indexes until it passes the threshold: max_n_idx (extra will be removed later)
        while max_n_idx > self.indexes.size:
            self.indexes = np.repeat(self.indexes, 2)

        # ensuring that batch_size is divisible into the number of indices
        if not len(self.indexes) % self.batch_size == 0:
          # Making sure that the batch_size is divisible into the number of indices
          self.indexes = self.indexes[:-(len(self.indexes) % (max_n_idx))]
          assert max_n_idx == self.indexes.size

    def __len__(self):
        return int(np.ceil(len(self.indexes) / float(self.batch_size)))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # self.img_idx = np.arange(len(self.x))
        # self.indexes = np.arange(len(self.indexes))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

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
            X, Y = self.apply_transform(X, Y)
        return (X, Y)

    def apply_transform(self, X, Y):
        data_dict = {}
        # batchgenerator transforms only accept channels_first
        data_dict["data"] = self.convert_dformat(X, convert_to = "channels_first")
        data_dict["seg"] = self.convert_dformat(Y, convert_to = "channels_first")
        data_dict = self.transform(**data_dict)
        # Desired target models accept channels_last dformat data (change this for your needs as you'd like)
        return (self.convert_dformat(data_dict["data"], convert_to = "channels_last"),
                self.convert_dformat(data_dict["seg"], convert_to = "channels_last"))

    def data_gen(self, list_IDs_temp):
        """
        Generates a batch of data.
        Args:
            list_IDs_temp: batched list IDs; usually done by __getitem__
            pos_sample: boolean on if you want to sample a positive image or not
        Returns:
            tuple of two numpy arrays: x, y
        """
        raise NotImplementedError

    def convert_dformat(self, arr, convert_to = "channels_last"):
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

    def compute_max_patient_shape(self):
        """
        Computes various shape statistics (min, max, and mean) and ONLY returns the max_patient_shape
        Args:
            ...
        Returns:
            max_patient_shape: tuple representing the maximum patient shape
        """
        print("Computing shape statistics...")
        # iterating through entire dataset
        shape_list = []
        for id in self.list_IDs:
            x_train = load_data(os.path.join(self.data_dirs[0], id))
            shape_list.append(np.asarray(x_train.shape))
        shapes = np.stack(shape_list)
        # computing stats
        max_patient_shape = tuple(np.max(shapes, axis = 0))
        mean_patient_shape = tuple(np.mean(shapes, axis = 0))
        min_patient_shape = tuple(np.min(shapes, axis = 0))
        print("Max Patient Shape: ", max_patient_shape, "\nMean Patient Shape: ", mean_patient_shape,
        "\nMin Patient Shape: ", min_patient_shape)
        # Running a quick check on a possible fail case
        try:
            assert len(max_patient_shape) == self.ndim
        except AssertionError:
            print("Excluding the channels dimension (axis = -1) for the maximum patient shape.")
            max_patient_shape = max_patient_shape[:-1]
        return max_patient_shape

class Transformed3DGenerator(BaseTransformGenerator):
    """
    Loads data and applies data augmentation with `batchgenerators.transforms`.
    * Supports channels_last
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
                transform = None, max_patient_shape = None, shuffle = True):

        BaseTransformGenerator.__init__(self, list_IDs = list_IDs, data_dirs = data_dirs, batch_size = batch_size,
                               n_channels = n_channels, n_classes = n_classes, ndim = ndim,
                               transform = transform, max_patient_shape = max_patient_shape, shuffle = shuffle)

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

class Transformed2DGenerator(BaseTransformGenerator):
    """
    Loads data, slices them based on the number of positive slice indices and applies data augmentation with `batchgenerators.transforms`.
    * Supports channels_last
    * Loads data WITH nibabel instead of SimpleITK
        * .nii files should not have the batch_size dimension

    Attributes:
        list_IDs: list of filenames
        data_dirs: list of paths to both the input dir and labels dir
        batch_size: The number of images you want in a single batch
        n_channels: number of channels
        n_classes: number of unique labels excluding the background class (i.e. binary; n_classes = 1)
        ndim: number of dimensions of the input (excluding the batch_size and n_channels axes)
        n_pos: The number of positive class 2D images to include in a batch
        transform (Transform instance): If you want to use multiple Transforms, use the Compose Transform.
        max_patient_shape: a tuple representing the maximum patient shape in a dataset; i.e. ((z,)x,y)
            * Note: If you have 3D medical images and want 2D slices and don't want to overpad the slice dimension (z),
            provide a shape that is only 2D (x,y).
        shuffle: boolean
    """
    def __init__(self, list_IDs, data_dirs, batch_size, n_channels, n_classes, ndim,
                n_pos, transform = None, max_patient_shape = None, n_workers = 1, shuffle = True):

        BaseTransformGenerator.__init__(self, list_IDs = list_IDs, data_dirs = data_dirs, batch_size = batch_size,
                               n_channels = n_channels, n_classes = n_classes, ndim = ndim,
                               transform = transform, max_patient_shape = max_patient_shape,
                               n_workers = n_workers, shuffle = shuffle)
        self.n_pos = n_pos
        if n_pos == 0:
            print("WARNING! Your data is going to be randomly sliced.")
            self.mode = "rand"
        elif n_pos == batch_size:
            print("WARNING! Your entire batch is going to be positively sampled.")
            self.mode = "pos"
        else:
            self.mode = "bal"

        if len(self.max_patient_shape) == 2:
            self.dynamic_padding_z = True # no need to pad the slice dimension

    def __getitem__(self, idx):
        """
        Defines the fetching and on-the-fly preprocessing of data.
        Args:
            idx: the id assigned to each worker
        Returns:
            (X,Y): a batch of transformed data/labels based on the n_pos attribute.
        """
        # file names
        max_n_idx = (idx + 1) * self.batch_size
        if max_n_idx > self.indexes.size:
            print("Adjusting for idx: ", idx)
            self.adjust_indexes(max_n_idx)

        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        # Fetches batched IDs for a thread
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # balanced sampling
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
        # random sampling
        elif self.mode == "rand":
            X, Y = self.data_gen(list_IDs_temp, pos_sample = False)
        elif self.mode == "pos":
            X, Y = self.data_gen(list_IDs_temp, pos_sample = True)
        # data augmentation
        if self.transform is not None:
            X, Y = self.apply_transform(X, Y)
        # print("Getting item of size: ", indexes.size, "out of ", self.indexes.size, "with idx: ", idx, "\nX shape: ", X.shape)
        assert X.shape[0] == self.batch_size, "The outputted batch doesn't match the batch size."
        return (X, Y)

    def data_gen(self, list_IDs_temp, pos_sample):
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
            if not y_train.shape[-1] == self.n_channels:
                # Adds channel in case there is no channel dimension
                y_train = add_channel(y_train)

            if self.n_classes > 1: # no point to run this when binary (foreground/background)
                y_train = get_multi_class_labels(y_train, n_labels = self.n_classes, remove_background = True)

            # Padding to the max patient shape (so the arrays can be stacked)
            if self.dynamic_padding_z: # for when you don't want to pad the slice dimension (bc that usually changes in images)
                pad_shape = (x_train.shape[0], ) + self.max_patient_shape
            elif not self.dynamic_padding_z:
                pad_shape = self.max_patient_shape
            x_train = reshape(x_train, x_train.min(), pad_shape + (self.n_channels, ))
            y_train = reshape(y_train, 0, pad_shape + (self.n_classes, ))
            assert sanity_checks(x_train, y_train)
            # extracting slice:
            if pos_sample:
                slice_idx = get_positive_idx(y_train)[0]
            elif not pos_sample:
                slice_idx = get_random_slice_idx(x_train)

            images_x.append(x_train[slice_idx]), images_y.append(y_train[slice_idx])

        input_data, seg_masks = np.stack(images_x), np.stack(images_y)
        return (input_data, seg_masks)

    def load_data(self, batchwise = True):
        """
        Loads all data from given files and directory in two numpy arrays representing the inputs and labels
        (Mainly used for evaluation)
        * Note: Basically the same as self.data_gen except there is no slice extraction, just loading padded full images
        Args:
            batchwise: boolean on whether to return batched outputs or vertically stacked outputs depending on how you want to evaluate your inputs
                * Note: batchwise evaluation would be done if you calculate the metrics for each batch and get the mean (2D/3D)
                        samplewise evaluation would be done if you calculate the metrics for each individual 2D slice (2D only)
        Returns:
            inputs: with shape (n_batches, x, y, z, n_channels) or (n_samples, x, y, n_channels)
            labels: same shape as inputs
        """
        images_x = []
        images_y = []
        for id in self.list_IDs:
            # loads data as a numpy arr and then changes the type to float32
            x_train = load_data(os.path.join(self.data_dirs[0], id))
            y_train = load_data(os.path.join(self.data_dirs[1], id))
            if not x_train.shape[-1] == self.n_channels:
                # Adds channel in case there is no channel dimension
                x_train = add_channel(x_train)
            if not y_train.shape[-1] == self.n_channels:
                # Adds channel in case there is no channel dimension
                y_train = add_channel(y_train)

            if self.n_classes > 1: # no point to run this when binary (foreground/background)
                y_train = get_multi_class_labels(y_train, n_labels = self.n_classes, remove_background = True)

            # Padding to the max patient shape (so the arrays can be stacked)
            if self.dynamic_padding_z: # for when you don't want to pad the slice dimension (bc that usually changes in images)
                pad_shape = (x_train.shape[0], ) + self.max_patient_shape
            elif not self.dynamic_padding_z:
                pad_shape = self.max_patient_shape
            x_train = reshape(x_train, x_train.min(), pad_shape + (self.n_channels, ))
            y_train = reshape(y_train, 0, pad_shape + (self.n_classes, ))
            assert sanity_checks(x_train, y_train)

            images_x.append(x_train), images_y.append(y_train)

        if batchwise:
            input_data, seg_masks = np.stack(images_x), np.stack(images_y)
        elif not batchwise:
            input_data, seg_masks = np.vstack(images_x), np.vstack(images_y)

        return (input_data, seg_masks)
