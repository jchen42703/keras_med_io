from io_func import resample_img
import numpy as np
import os
from shutil import copyfile
from SimpleITK import GetImageFromArray, GetArrayFromImage, ReadImage
import SimpleITK as sitk
from glob import glob
from functools import partial
import time
import nibabel as nib
# Grabbing all the functions for reusability purposes

class ResampleLocal(object):
    '''
    Preprocesses 3D nifti images and saves it locally compressed for the Medical Segmentation Decathlon Challenge
    * CURRENTLY ONLY SUPPORTING TEST SET EXTRACTION
    * order -> resample -> extract patches -> [0,1] normalization
    Attributes:
    * data_dirs: paths to the folders with all of .nii files
    * output_path: paths to the folders where you want to save all of the resampled images
    [DON'T FORGET THE DASHES AT THE ENDS OF THE PATHS]
    * num_classes: int
        * when num_classes > 1 CURRENTLY NOT SUPPORTED
    '''
    def __init__(self, data_dirs, output_paths, num_classes=1):
        self.data_dirs = data_dirs
        self.output_paths = output_paths
        if num_classes == 2:
            self.num_classes = 1
        else:
            self.num_classes = num_classes
        self.id_list = self.get_list_IDs_nosplit(data_dirs[0])

    def data_gen_all(self):
        for id in self.id_list:
            x, y = self.data_gen_one(id)
            affine = np.zeros((4,4))
            np.fill_diagonal(affine, val = 1)
            x = nib.Nifti1Image(x, affine = affine)
            y = nib.Nifti1Image(y, affine = affine)

            nib.save(x, self.output_paths[0] + id)
            nib.save(y, self.output_paths[1] + id)
            # save with the original id

    def data_gen_one(self, id): #, set = 'test'):
        '''
        Resamples all possible patches
        param pair: tuple of image/label file path pair
        '''
        # for file_x, file_y in zip(batch_x, batch_y):
        sitk_image, sitk_label = sitk.ReadImage(self.data_dirs[0] + id), sitk.ReadImage(self.data_dirs[1] + id)
        # (h,w, n_slices)
        #resample; defaults to 1mm isotropic spacing
        x_resample, y_resample = resample_img(sitk_image), resample_img(sitk_label, is_label = True)
        # converting to numpy arrays
        x_train = np.expand_dims(GetArrayFromImage(x_resample).astype(np.float32), -1)
        y_train = np.expand_dims(GetArrayFromImage(y_resample).astype(np.float32), -1)
        assert x_train.shape == y_train.shape
        # def _transpose(arr):
        #     return np.transpose(arr, [1,2,0])
        # x_train, y_train = _transpose(x_train), _transpose(y_train)
        print(x_train.shape)
        assert x_train.shape[-1] == self.num_classes
        # extract_2D_patches is done based on channels_last data
        return (x_train, y_train)

    def get_list_IDs_nosplit(self, data_dir):
        """
        param data_dir: data directory to the training or label folder
        * assumes labels and training images have same names
        """
        id_list = os.listdir(data_dir)
        return id_list

# train_dir = 'C:\\Users\\jchen\\Downloads\\Datasets\\Task02_Heart\\imagesTr\\'
# label_dir = 'C:\\Users\\jchen\\Downloads\\Datasets\\Task02_Heart\\labelsTr\\'
# output_train =  'C:\\Users\\jchen\\Downloads\\Datasets\\Heart_Resampled\\imagesTr\\'
# output_label =  'C:\\Users\\jchen\\Downloads\\Datasets\\Heart_Resampled\\labelsTr\\'
# resamp = ResampleLocal([train_dir, label_dir], [output_train, output_label])
# resamp.data_gen_all()
