import unittest
import numpy as np
# from generators import *
from patch_utils import PatchExtractor

class GeneratorTest(unittest.TestCase):
    '''
    Testing Generator Backend functions for channels_first applications
    '''
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
        self.extractor_2D = PatchExtractor(self.train_image_2D, self.label_image_2D, self.image_shape_2D[-2:], self.patch_shape_2D, self.overlap)
        self.extractor_3D = PatchExtractor(self.train_image_3D, self.label_image_3D, self.image_shape_3D[-3:], self.patch_shape_3D, self.overlap)

    # @unittest.skip('Not working yet')
    def test_random_centered_crop_2D(self):
        '''
        Testing random centered crop.
        * Using self.compute_patch_indices from PatchExtractor and then choosing a random index from that for self.get_centered_patch
        * TEMPORARY FIX: If patch_idx-patch_shape < 0, we just convert that specific index into a corner; [0:patch_shape]
        '''
        # grabbing all indexes for image and patch shape
        patch_indexes = self.extractor_2D.compute_patch_indices(self.image_shape_2D[-2:], self.patch_shape_2D,
                                                                self.overlap, ndim = 2)
        self.assertEqual(patch_indexes.shape, (16,2))
        # fetching random patch index
        random_idx = patch_indexes[np.random.randint(0, patch_indexes.shape[0]-1)]
        # print(random_idx)
        self.assertEqual(random_idx.shape, (2,))
        patch_x = self.extractor_2D.get_centered_patch(self.train_image_2D, self.patch_shape_2D, random_idx)
        # has 0 dimension for one of em becuase of patch_index - patch_shape < 0
        self.assertEqual(patch_x.shape, (4,128,128))

    # @unittest.skip('It is good')
    def test_padding_2D(self):
        '''
        Testing if padding can handle both 'out of bounds patches'; i.e. idx-shape or idx+shape
        '''
        # setting it up temporarily
        idx = np.asarray([-52, 76])
        patch_shape = self.patch_shape_2D
        print(idx)
        # gets padded data and padded index
        data, patch_index = self.extractor_2D.fix_out_of_bound_patch_attempt(self.train_image_2D,
                                                                           self.patch_shape_2D, idx, ndim=2)
        print('padded shape: ', data.shape)
        self.assertFalse(np.any(data.shape) == 0)
        self.assertFalse(np.any(patch_index) < 0)
        start_x, end_x = patch_index[0]-patch_shape[0]//2, patch_index[0]+patch_shape[0]//2
        start_y, end_y = patch_index[1]-patch_shape[1]//2, patch_index[1]+patch_shape[1]//2
        start_end = np.asarray([start_x, end_x, start_y, end_y])

        # tests that it pads data to deal for when idx-shape < 0
        self.assertFalse(np.any(start_end < 0))
        # tests to make sure that it's still the right patch_shape
        self.assertTrue((end_x-start_x) == patch_shape[0] and (end_y - start_y == patch_shape[1]))
        print(padded_idx)


unittest.main(argv=[''], verbosity=2, exit=False)
