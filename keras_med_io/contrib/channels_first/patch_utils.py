import numpy as np

class PatchExtractor(object):
    '''
    Lean patch extractor class.

    Main Methods:
        .extract_patch(): Extracting either 2D/3D patch
        .get_centered_patch(): Extracting centered patch around index
    '''
    def __init__(self, ndim):
        self.ndim = ndim

    def get_centered_patch(self, data, patch_shape, patch_index):
        """
        Returns a positive patch centered around a positive pixel
        :param data: numpy array from which to get the patch.
        :param patch_shape: shape/size of the patch.
        :param patch_index: corner index of the patch.
        :return: numpy array take from the data with the patch shape specified.
        """
        # assert self.ndim == 2
        patch_index = np.asarray(patch_index, dtype = np.int16)
        patch_shape = np.asarray(patch_shape, dtype = np.int16)
        image_shape = data.shape[-self.ndim:]
        if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape) or np.any(patch_index-patch_shape < 0):
            data, patch_index = self.fix_out_of_bound_patch_attempt(data, patch_shape, patch_index, ndim = self.ndim)
        # padding all of the missing elements      AA1
        start_x, end_x = patch_index[0]-patch_shape[0]//2, patch_index[0]+patch_shape[0]//2
        start_y, end_y = patch_index[1]-patch_shape[1]//2, patch_index[1]+patch_shape[1]//2

        # deals with padded cases ## NEEDS TO BE MADE MORE EFFICIENT
        if (start_y+end_y) == 0:
            start_y, end_y = (0,128)
        if (start_x+end_x) == 0:
            start_x, end_x = (0,128)
        start_end = np.asarray([start_x, end_x, start_y, end_y])
        # print('start_end before: ', start_end)
        # # start_end[start_end <= 0] += patch_shape[0] # index for patch_shape doesn't matter, assuming square patch
        # print('start_end after: ', start_end, '\npatch_index: ', patch_index)
        if self.ndim == 2:
            # centered crop
            return data[..., start_end[0]:start_end[1], start_end[2]:start_end[3]]
        elif self.ndim == 3:
            start_z, end_z = patch_index[2]-patch_shape[2]//2, patch_index[2]+patch_shape[2]//2
            return data[..., start_x:end_x, start_y:end_y, start_z:end_z]

    def extract_patch(self, data, patch_shape, patch_index):
        '''
        Extracting both 2D and 3D patches depending on patch shape dimensions
        '''
        patch_index = np.asarray(patch_index, dtype = np.int16)
        patch_shape = np.asarray(patch_shape, dtype = np.int16)

        image_shape = data.shape[-self.ndim:]
        # print("image_shape: from extract_patch: ", image_shape)
        if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape):
            data, patch_index = self.fix_out_of_bound_patch_attempt(data, patch_shape, patch_index, ndim = self.ndim)

        if self.ndim == 2:
            return data[..., patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1]]
        elif self.ndim == 3:
            return data[..., patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1],
                        patch_index[2]:patch_index[2]+patch_shape[2]]

    def get_set_of_patch_indices(self, start, stop, step):
        '''
        getting set of all possible indices with the start, stop and step.
        '''
        if self.ndim == 2:
            return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1]].reshape(2, -1).T,
                             dtype=np.int)
        elif self.ndim == 3:
            return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                                           start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)

    def compute_patch_indices(self, image_shape, patch_shape, overlap, start=None):
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
        return self.get_set_of_patch_indices(start, stop, step)

    def fix_out_of_bound_patch_attempt(self, data, patch_shape, patch_index, ndim=3):
        """
        Pads the data and alters the corner patch index so that the patch will be correct.
        :param data:
        :param patch_shape:
        :param patch_index:
        :return: padded data, fixed patch index
        """
        image_shape = data.shape[-ndim:]
        # figures out which indices need to be padded; if they're < 0
        pad_before = np.abs((patch_index < 0 ) * patch_index)
        # checking for out of bounds if doing idx+patch shape by replacing the afflicted indices with a kinda random replacement
        pad_after = np.abs(((patch_index + patch_shape) > image_shape) * ((patch_index + patch_shape) - image_shape))
        pad_args = np.stack([pad_before, pad_after], axis=1)

        if pad_args.shape[0] < len(data.shape):
            # adding channels dimension to padding ([0,0] so that it's ignored)
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