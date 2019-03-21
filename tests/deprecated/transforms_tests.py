# from keras_med_io.generators.posrandom_generator_noresamp import PosRandomPatchGenerator
"""
Only for example generator
"""
from keras_med_io.generators.general import *
import unittest
import os
import keras_med_io
from keras_med_io.transforms.transforms_generator import *
import os
import numpy as np
import batchgenerators
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.abstract_transforms import Compose
from keras_med_io.utils.transforms_utils import compute_pad_value

class TransformGeneratorsTest(unittest.TestCase):
    """
    Testing generators from `keras_med_io.transforms.transforms_generator.py` to make sure they run and produce
    outputs with the correct shapes.
    """
    def setUp(self):
        task_dir = 'C:\\Users\\jchen\\Desktop\\Datasets\\Heart_Isensee\\'
        input_dir = task_dir + 'imagesTr\\'
        labels_dir = task_dir + 'labelsTr\\'

        # settings
        self.data_dirs = [input_dir, labels_dir]
        self.list_IDs = os.listdir(input_dir)
        self.batch_size_2D = 33
        self.batch_size_3D = 2
        self.border_cval = compute_pad_value(input_dir, self.list_IDs)

        self.patch_shape_2D = (256, 320)
        self.patch_shape_3D = (80, 192, 128)
        self.output_shape_2D = (self.batch_size_2D, ) + self.patch_shape_2D + (1,)
        self.output_shape_3D = (self.batch_size_3D, ) + self.patch_shape_3D + (1,)
        self.n_channels = self.output_shape_2D[-1]

    def test_regular_2D_data(self):
        """
        Tests shapes for 2D data generation with no data augmentation.
        """
        # instantiating generators
        max_patient_shape = (256, 320)
        output_shape = (self.batch_size_2D, ) + max_patient_shape + (1,)
        n_workers = 5
        gen = Transformed2DGenerator(self.list_IDs, self.data_dirs, batch_size = self.batch_size_2D, n_channels = self.n_channels, \
                                     n_classes = 1, ndim = 2, n_pos = 1, transform = None, max_patient_shape = max_patient_shape, \
                                     n_workers = n_workers)
        x, y = gen.__getitem__(n_workers)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.shape, output_shape)

    def test_regular_3D_data(self):
        """
        Tests shapes for generated 3D data with no data augmentation
        """
        # instantiating generators
        # purposely excluding max_patient_shape; it's automatically found
        n_workers = 2
        gen = Transformed3DGenerator(self.list_IDs, self.data_dirs, batch_size = self.batch_size_3D, n_channels = self.n_channels, \
                                     n_classes = 1, ndim = 3, transform = None,)
        x, y = gen.__getitem__(n_workers)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.shape[0], self.batch_size_3D)

    def test_transformed_2D_data(self):
        """
        Tests shapes for generated 2D data with data augmentation
        """
        # instantiating generators
        # purposely excluding max_patient_shape; it's automatically found
        spatial_transform = SpatialTransform(self.patch_shape_2D, #mean_patient_shape[2:],
                         do_elastic_deform=True, alpha=(0., 1500.), sigma=(30., 50.),
                         do_rotation=True, angle_z=(0, 2 * np.pi),
                         do_scale=True, scale=(0.8, 2.),
                         border_mode_data='constant', border_cval_data=self.border_cval,
                         order_data=1, random_crop=True)

        mirror_transform = MirrorTransform(axes=(1, 2))
        transforms_list = [spatial_transform, mirror_transform]
        composed = Compose(transforms_list)
        n_workers = 5
        gen = Transformed2DGenerator(self.list_IDs, self.data_dirs, batch_size = self.batch_size_2D, n_channels = self.n_channels, \
                                     n_classes = 1, ndim = 2, n_pos = 2, transform = composed, max_patient_shape = (256,320),
                                     n_workers = n_workers)
        x, y = gen.__getitem__(n_workers)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.shape, self.output_shape_2D)

    def test_transformed_3D_data(self):
        """
        Tests shapes for generated 3D data with data augmentation
        """
        # instantiating generators
        # purposely excluding max_patient_shape; it's automatically found
        spatial_transform = SpatialTransform(self.patch_shape_3D, #mean_patient_shape[2:],
                         do_elastic_deform=True, alpha=(0., 1500.), sigma=(30., 50.),
                         do_rotation=True, angle_z=(0, 2 * np.pi),
                         do_scale=True, scale=(0.8, 2.),
                         border_mode_data='constant', border_cval_data=self.border_cval,
                         order_data=1, random_crop=True)

        mirror_transform = MirrorTransform(axes=(1, 2))
        transforms_list = [spatial_transform, mirror_transform]
        composed = Compose(transforms_list)
        n_workers = 2
        gen = Transformed3DGenerator(self.list_IDs, self.data_dirs, batch_size = self.batch_size_3D, n_channels = self.n_channels, \
                                     n_classes = 1, ndim = 3, transform = composed)
        x, y = gen.__getitem__(n_workers)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.shape, self.output_shape_3D)



unittest.main(argv=[''], verbosity=2, exit=False)
