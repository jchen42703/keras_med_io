import unittest
import numpy as np
from patch_utils import *

class PatchTest(unittest.TestCase):
    def setUp(self):
        self.n_channels = 4
        self.image_shape_2D = (self.n_channels, 408, 408)
        self.patch_shape_2D = np.asarray((128, 128))
        self.train_image_2D = np.zeros(self.image_shape_2D)
        self.label_image_2D = np.ones(self.image_shape_2D)

        self.image_shape_3D = (self.n_channels, 155, 240, 240)
        self.patch_shape_3D = np.asarray((128,128,128))
        self.train_image_3D = np.zeros(self.image_shape_3D)
        self.label_image_3D = np.ones(self.image_shape_3D)


        self.overlap = 0
        self.extractor_2D = PatchExtractor(ndim = 2, image_format = "channels_first")
        self.extractor_2D_last = PatchExtractor(ndim = 2, image_format = "channels_last")

        self.extractor_3D = PatchExtractor(ndim = 3, image_format = "channels_first")
        self.extractor_3D_last = PatchExtractor(ndim = 3, image_format = "channels_last")

    def test_compute_patch_indices(self):
        '''
        Tests the utility function to make sure it returns the correct dimensions
        '''
        patch_indices = self.extractor_2D.compute_patch_indices(self.image_shape_2D[-2:], self.patch_shape_2D, self.overlap, ndim = 2)
        self.assertEqual(patch_indices.shape[1], 2)

    def test_2D_patch_extraction(self):
        '''
        Tests to make sure that the outputs are the correct shapes for both channels_first and channels_last for 2D patch extraction
        '''
        # testing channels_first extraction
        output_shape = tuple([self.n_channels] + list(self.patch_shape_2D))
        patch_indices = self.extractor_2D.compute_patch_indices(self.image_shape_2D[-2:], self.patch_shape_2D, self.overlap, ndim = 2)
        x = self.extractor_2D.extract_patch(self.train_image_2D, self.patch_shape_2D, patch_indices[0])#, image_format = "channels_first")
        self.assertEqual(x.shape, output_shape)

        # testing channels_last extraction
        output_shape = tuple(list(self.patch_shape_2D) + [self.n_channels])
        image = np.transpose(self.train_image_2D, [1,2,0])
        patch_indices = self.extractor_2D_last.compute_patch_indices(self.image_shape_2D[-2:], self.patch_shape_2D, self.overlap, ndim = 2)
        x = self.extractor_2D_last.extract_patch(image, self.patch_shape_2D, patch_indices[0])#, image_format = "channels_last")
        self.assertEqual(x.shape, output_shape)

    def test_3D_patch_extraction(self):
        '''
        Tests to make sure that the outputs are the correct shapes for both channels_first and channels_last for 3D patch extraction
        '''
        # x, y = self.extractor_3D.run(one = True)
        # self.assertEqual(x.shape, y.shape)
        # self.assertEqual(x.shape, tuple(self.patch_shape_3D))
        # testing channels_first
        output_shape = tuple([self.n_channels] + list(self.patch_shape_3D))
        patch_indices = self.extractor_3D.compute_patch_indices(self.image_shape_3D[-3:], self.patch_shape_3D, self.overlap, ndim = 3)
        x = self.extractor_3D.extract_patch(self.train_image_3D, self.patch_shape_3D, patch_indices[0])
        self.assertEqual(x.shape, output_shape)

        # testing channels_last extraction
        output_shape = tuple(list(self.patch_shape_3D) + [self.n_channels])
        image = np.transpose(self.train_image_3D, [1,2,3,0])
        patch_indices = self.extractor_3D_last.compute_patch_indices(self.image_shape_3D[-3:], self.patch_shape_3D, self.overlap, ndim = 3)
        x = self.extractor_3D_last.extract_patch(image, self.patch_shape_3D, patch_indices[0])#, image_format = "channels_last")
        self.assertEqual(x.shape, output_shape)

unittest.main(argv=[''], verbosity=2, exit=False)
