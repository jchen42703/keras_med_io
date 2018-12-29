from keras_med_io.utils.patch_utils import PatchExtractor
from keras_med_io.utils.io_func import sanity_checks, resample_img, get_multi_class_labels, whitening, normalize, normalize_clip
import numpy as np
from glob import glob
import os
import nibabel as nib

class LocalFileGenerator(PatchExtractor):
    '''
    Goal:
        To extracct all possible patches(for both 2D and 3D) and save each patch locally as a .npy file
    **ASSUMES THAT YOU ALREADY RESAMPLED IMGS
    (add option for resample when using the nibabel/skimage resampling version [need to test it first])
    **ASSUMES Channels_last
    in_dirs/out_dirs: sequence in the format (input_dir, label_dir)
    '''
    def __init__(self, patch_shape, n_labels, overlap, in_dirs, out_dirs,  \
                normalize_mode = "normalize", norm_range = [0,1], resample = False, save_indices = True):
        self.patch_shape = patch_shape
        self.n_labels = n_labels # includes the background 0 label
        self.overlap = overlap
        self.ndim = len(patch_shape)

        # self.in_input = glob(in_dirs[0] + '*.nii.gz', recursive = True)
        # self.in_labels = glob(in_dirs[1] +'*.nii.gz', recursive = True)
        # assumes all of the inputs and labels have the same corresponding names and that there are no other files besides the imgs
        self.in_dirs = in_dirs
        self.ids = os.listdir(in_dirs[0])
        self.out_dirs = out_dirs

        self.normalize_mode = normalize_mode
        self.norm_range = norm_range
        self.resample = resample
        self.save_indices = save_indices
        if self.save_indices: # saving indices to a dictionary by id (for reconstruction)
            self.indices = {}

    def gen_and_save(self):
        '''
        Generates all the prepocessed patches and saves them individually.
        Options
        * Can save the indices for reconstruction.
        '''
        n_patch_pairs = 0
        for id in self.ids:
            print("Processing: ", id)
            x, y = self.preprocess(os.path.join(self.in_dirs[0], id), os.path.join(self.in_dirs[1], id))
            # assumes x will be (n_slices, x, y, n_channels)
            image_shape = x.shape[:3]
            patch_indices = self.compute_patch_indices(image_shape, self.patch_shape, self.overlap)
            if self.save_indices:
                self.indices[id] = patch_indices
            # patch extraction
            x_patches = [self.extract_patch(x, self.patch_shape, index) for index in patch_indices]
            y_patches = [self.extract_patch(y, self.patch_shape, index) for index in patch_indices]
            n_patch_pairs += len(x_patches)
            # saving each patch individually
            for idx, patch_pair in enumerate(zip(x_patches, y_patches)):
                new_x = os.path.join(self.out_dirs[0], id.split('.')[0] + '_' + str(idx))
                new_y = os.path.join(self.out_dirs[1], id.split('.')[0] + '_' + str(idx))
                np.save(new_x, patch_pair[0]), np.save(new_y, patch_pair[1])
                print("Saving: ", new_x)
        results = self.save_dict(self.indices, 'indices.json')
        print("Completed! Saved ", str(n_patch_pairs), "patch pairs.")
        return n_patch_pairs

    def gen_and_save_one(self, in_dir, out_dir, is_label = False):
        '''
        Generates all patch pairs for one of the specified directories (to manage memory issues)
        '''
        n_patch_pairs = 0
        for id in self.ids:
            print("Processing: ", id)
            x = self.preprocess_one(os.path.join(in_dir, id), is_label)
            # assumes x will be (n_slices, x, y, n_channels)
            image_shape = x.shape[:3]
            patch_indices = self.compute_patch_indices(image_shape, self.patch_shape, self.overlap)
            if self.save_indices:
                self.indices[id] = patch_indices
            # patch extraction
            x_patches = [self.extract_patch(x, self.patch_shape, index) for index in patch_indices]
            n_patch_pairs += len(x_patches)
            # saving each patch individually
            for idx, patch in enumerate(x_patches):
                new_x = os.path.join(out_dir, id.split('.')[0] + '_' + str(idx))
                np.save(new_x, patch)
                print("Saving: ", new_x)
        results = self.save_dict(self.indices, 'indices.json')
        print("Completed! Saved ", str(n_patch_pairs), "patch pairs.")
        return n_patch_pairs

    def preprocess(self, x, y):
        '''
        Does all of the necessary prerpcessing
        : params x,y: paths to the their .nii.gz locations
        '''
        x = self.normalization(nib.load(x).get_fdata())
        y = self.normalization(nib.load(y).get_fdata())
        if self.resample:
            raise NotImplementedError
        if self.n_labels > 2:
            y = get_multi_class_labels(y, self.n_labels, remove_background = True)
        return x, y

    def preprocess_one(self, x, is_label = False):
        '''
        Does all of the necessary prerpcessing for one of em
        : params x,y: paths to the their .nii.gz locations
        '''
        x = self.normalization(nib.load(x).get_fdata())
        if self.resample:
            raise NotImplementedError
        if is_label:
            if self.n_labels > 2:
                x = get_multi_class_labels(x, self.n_labels, remove_background = True)
        return x

    def normalization(self, patch_x):
        '''
        Normalizes the image based on the specified mode and range
        '''
        # reiniating the batch_size dimension
        if self.normalize_mode == 'whitening':
            return whitening(patch_x)
        elif self.normalize_mode == 'normalize_clip':
            return normalize_clip(patch_x, range = self.norm_range)
        elif self.normalize_mode == 'normalize':
            return normalize(patch_x, range = self.norm_range)

    def save_dict(self, dict, fname):
        '''
        Saves a dictionary; i.e. Used for saving the indices dictionary.
        '''
        import pickle
        save_path = os.path.join(self.out_dirs[0], fname)
        with open(save_path, 'wb') as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(fname, 'download complete.')
        return save_path
