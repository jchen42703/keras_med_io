from CapsNetMRI.io import generators
from CapsNetMRI.contrib import generators_unet
# import generators_unet
# import generators

import unittest
import os

class GeneratorTest(unittest.TestCase):
    '''
    Testing generators for both the U-Net and CapsNets Versions
        For each generator, the directories may vary; Fork and change them for individual testing
    * RandomPatchGenerator
    * PositivePatchGenerator
    * BalancedPatchGenerator
    '''
    def setUp(self):
        task_dir = 'C:\\Users\\jchen\\Desktop\\Datasets\\Task02_Heart\\'
        input_dir = task_dir + 'imagesTr\\'
        labels_dir = task_dir + 'labelsTr\\'

        # settings
        self.batch_size = 2
        self.normalize_mode = 'normalize'
        self.norm_range = [0,1]
        self.centered = False
        self.data_dirs = [input_dir, labels_dir]
        self.list_IDs = os.listdir(input_dir)
        # capsnet generators
        # pos_gen_caps = generators.PositivePatchGenerator(list_IDs, data_dirs, batch_size = batch_size,
        #                                 patch_shape = (128,128), overlap = 0, normalize_mode = normalize_mode,
        #                                 range = norm_range, centered = centered
        #                                  )
        # rand_gen_caps = generators.RandomPatchGenerator(list_IDs, data_dirs, batch_size = batch_size,
        #                                 patch_shape = (128,128), normalize_mode = normalize_mode,
        #                                 range = norm_range
        #                                  )
        #
        # bal_gen_caps = generators.BalancedPatchGenerator(list_IDs, data_dirs, batch_size = batch_size,
        #                                 patch_shape = (128,128), normalize_mode = normalize_mode,
        #                                 range = norm_range, n_pos = 1, centered = centered
        #                                  )

    @unittest.skip("Works.")
    def test_random_patch_last(self):
        '''
        Only tests that it crops properly for channels_last; assumes everything else is done correctly
        * works 100% okay for channels_last
        '''
        patch_shape_2d = (128, 128)
        patch_shape_3d = (128,128,128)
        output_shape_2d = (self.batch_size, 128, 128, 1)
        output_shape_3d = (self.batch_size, 128, 128, 128, 1)
        image_format = "channels_last"
        # instantiating generators
        rand_gen_caps = generators.RandomPatchGenerator(self.list_IDs, self.data_dirs, batch_size = self.batch_size,
                                        patch_shape = patch_shape_2d, normalize_mode = self.normalize_mode,
                                        range = self.norm_range
                                         )
        rand_gen_unet2d = generators_unet.RandomPatchGenerator(self.list_IDs, self.data_dirs, image_format = image_format,
                                        batch_size = self.batch_size, patch_shape = patch_shape_2d, normalize_mode = self.normalize_mode,
                                        range = self.norm_range
                                         )
        rand_gen_unet3d = generators_unet.RandomPatchGenerator(self.list_IDs, self.data_dirs, image_format = image_format,
                                        batch_size = self.batch_size, patch_shape = patch_shape_3d, normalize_mode = self.normalize_mode,
                                        range = self.norm_range
                                         )
        # testing their output shapes
        x, y = rand_gen_caps.__getitem__(1)
        assert x[0].shape == x[1].shape == y[0].shape == y[1].shape
        self.assertEqual(x[0].shape, output_shape_2d)

        x_u2, y_u2 = rand_gen_unet2d.__getitem__(1)
        assert x_u2.shape == y_u2.shape
        self.assertEqual(x_u2.shape, output_shape_2d)

        x_u3, y_u3 = rand_gen_unet3d.__getitem__(1)
        assert x_u3.shape == y_u3.shape
        self.assertEqual(x_u3.shape, output_shape_3d)

    def test_random_patch_first(self):
        '''
        Only tests that it crops properly for channels_last; assumes everything else is done correctly
        * works 100% okay for channels_last
        '''
        patch_shape_2d = (128, 128)
        patch_shape_3d = (128,128,128)
        output_shape_2d = (self.batch_size, 1, 128, 128)
        output_shape_3d = (self.batch_size, 1, 128, 128, 128)
        image_format = "channels_first"
        # instantiating generators
        rand_gen_caps = generators.RandomPatchGenerator(self.list_IDs, self.data_dirs, batch_size = self.batch_size,
                                        patch_shape = patch_shape_2d, normalize_mode = self.normalize_mode,
                                        range = self.norm_range
                                         )
        rand_gen_unet2d = generators_unet.RandomPatchGenerator(self.list_IDs, self.data_dirs, image_format = image_format,
                                        batch_size = self.batch_size, patch_shape = patch_shape_2d, normalize_mode = self.normalize_mode,
                                        range = self.norm_range
                                         )
        rand_gen_unet3d = generators_unet.RandomPatchGenerator(self.list_IDs, self.data_dirs, image_format = image_format,
                                        batch_size = self.batch_size, patch_shape = patch_shape_3d, normalize_mode = self.normalize_mode,
                                        range = self.norm_range
                                         )
        # testing their output shapes
        # x, y = rand_gen_caps.__getitem__(1)
        # assert x[0].shape == x[1].shape == y[0].shape == y[1].shape
        # self.assertEqual(x[0].shape, output_shape_2d)

        x_u2, y_u2 = rand_gen_unet2d.__getitem__(1)
        assert x_u2.shape == y_u2.shape
        self.assertEqual(x_u2.shape, output_shape_2d)

        x_u3, y_u3 = rand_gen_unet3d.__getitem__(1)
        assert x_u3.shape == y_u3.shape
        self.assertEqual(x_u3.shape, output_shape_3d)

unittest.main(argv=[''], verbosity=2, exit=False)
