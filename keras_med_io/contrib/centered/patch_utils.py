import numpy as np

class PatchExtractor(object):
    '''
    Lean patch extractor class.
    Channels_last.

    Main Methods:
        .extract_patch(): Extracting either 2D/3D patch
        .get_centered_patch(): Extracting centered patch around index
    '''
    def __init__(self, ndim, image_format = "channels_first"):
        self.ndim = ndim
        self.image_format = image_format

    def extract_patch(self, data, patch_shape, patch_index):
        '''
        Extracting both 2D and 3D patches depending on patch shape dimensions
        '''
        patch_index = np.asarray(patch_index, dtype=np.int16)
        patch_shape = np.asarray(patch_shape, dtype = np.int16)

        if self.image_format == "channels_first":
            image_shape = data.shape[-self.ndim:]
            # print("image_shape: from extract_patch: ", image_shape)
            if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape):
                data, patch_index = self.fix_out_of_bound_patch_attempt(data, patch_shape, patch_index, ndim = self.ndim)

            if self.ndim == 2:
                return data[..., patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1]]
            elif self.ndim == 3:
                return data[..., patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1],
                            patch_index[2]:patch_index[2]+patch_shape[2]]

        elif self.image_format == "channels_last":
            image_shape = data.shape[:self.ndim]
            if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape):
                data, patch_index = self.fix_out_of_bound_patch_attempt(data, patch_shape, patch_index, ndim = self.ndim)

            if self.ndim == 2:
                return data[patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1], ...]#, ...]
            elif self.ndim == 3:
                print("This ran.", "data.shape: ", data.shape)
                return data[patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1],
                            patch_index[2]:patch_index[2]+patch_shape[2], ...]

    def get_set_of_patch_indices(self, start, stop, step, ndim = 3):
        '''
        getting set of all possible indices with the start, stop and step.
        '''
        if ndim == 2:
            return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1]].reshape(2, -1).T,
                             dtype=np.int)
        elif ndim == 3:
            return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                                           start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)

    def compute_patch_indices(self, image_shape, patch_shape, overlap, start=None, ndim = 3):
        '''
        image_shape: ndarray of dimensions
        patch_shape: ndarray of patch dimensions
        returns: a np array of coordinates & step
        '''
        if isinstance(overlap, int):
            overlap = np.asarray([overlap] * len(image_shape))
        if start is None:
            n_patches = np.ceil(image_shape / (patch_shape - overlap))
            overflow = (patch_shape - overlap) * n_patches - image_shape + overlap
            start = -np.ceil(overflow/2)
        elif isinstance(start, int):
            start = np.asarray([start] * len(image_shape))
        stop = image_shape + start
        step = patch_shape - overlap
        return self.get_set_of_patch_indices(start, stop, step, ndim = self.ndim)

    def fix_out_of_bound_patch_attempt(self, data, patch_shape, patch_index, ndim=3):
        """
        Pads the data and alters the corner patch index so that the patch will be correct.
        :param data:
        :param patch_shape:
        :param patch_index:
        :return: padded data, fixed patch index
        """
        if self.image_format == "channels_first":
            image_shape = data.shape[-ndim:]
        elif self.image_format == "channels_last":
            image_shape = data.shape[:ndim]
        # figures out which indices need to be padded; if they're < 0
        pad_before = np.abs((patch_index < 0 ) * patch_index) # also need to check if idx-patch_shape < 0
        # checking for out of bounds if doing idx+patch shape by replacing the afflicted indices with a kinda random replacement
        pad_after = np.abs(((patch_index + patch_shape) > image_shape) * ((patch_index + patch_shape) - image_shape))
        if self.image_format == "channels_first":
            pad_args = np.stack([pad_before, pad_after], axis=1)
        elif self.image_format == "channels_last":
            pad_args = np.stack([pad_before, pad_after], axis=-1)
        # print('pad_args.tolist: ', pad_args.tolist(), '\nchecking: ', (len(data.shape) - pad_args.shape[0]) + pad_args.tolist())
        # print('pad_before: ', pad_before, '\npad_after: ', pad_after, '\npad_args: ', pad_args)
        # print('data_shape: ', data.shape)
        if pad_args.shape[0] < len(data.shape):
            # adding channels dimension to padding ([0,0] so that it's ignored)
            if self.image_format == "channels_last":
                pad_args = pad_args.tolist() + [[0, 0]] * (len(data.shape) - pad_args.shape[0])
            elif self.image_format == "channels_first":
                pad_args = [[0, 0]] * (len(data.shape) - pad_args.shape[0]) + pad_args.tolist()

        data = np.pad(data, pad_args, mode="edge")
        patch_index += pad_before
        return data, patch_index

    def reconstruct_from_patches(self, patches, patch_indices, data_shape, default_value=0):
        """
        Reconstructs an array of the original shape from the lists of patches and corresponding patch indices. Overlapping
        patches are averaged.
        :param patches: List of numpy array patches.
        :param patch_indices: List of indices that corresponds to the list of patches.
        :param data_shape: Shape of the array from which the patches were extracted.
        :param default_value: The default value of the resulting data. if the patch coverage is complete, this value will
        be overwritten.
        :return: numpy array containing the data reconstructed by the patches.
        """
        data = np.ones(data_shape) * default_value
        image_shape = data_shape[-3:]
        count = np.zeros(data_shape, dtype=np.int)
        for patch, index in zip(patches, patch_indices):
            image_patch_shape = patch.shape[-3:]
            if np.any(index < 0):
                fix_patch = np.asarray((index < 0) * np.abs(index), dtype=np.int)
                patch = patch[..., fix_patch[0]:, fix_patch[1]:, fix_patch[2]:]
                index[index < 0] = 0
            if np.any((index + image_patch_shape) >= image_shape):
                fix_patch = np.asarray(image_patch_shape - (((index + image_patch_shape) >= image_shape)
                                                            * ((index + image_patch_shape) - image_shape)), dtype=np.int)
                patch = patch[..., :fix_patch[0], :fix_patch[1], :fix_patch[2]]
            patch_index = np.zeros(data_shape, dtype=np.bool)
            patch_index[...,
                        index[0]:index[0]+patch.shape[-3],
                        index[1]:index[1]+patch.shape[-2],
                        index[2]:index[2]+patch.shape[-1]] = True
            patch_data = np.zeros(data_shape)
            patch_data[patch_index] = patch.flatten()

            new_data_index = np.logical_and(patch_index, np.logical_not(count > 0))
            data[new_data_index] = patch_data[new_data_index]

            averaged_data_index = np.logical_and(patch_index, count > 0)
            if np.any(averaged_data_index):
                data[averaged_data_index] = (data[averaged_data_index] * count[averaged_data_index] + patch_data[averaged_data_index]) / (count[averaged_data_index] + 1)
            count[patch_index] += 1
        return data
