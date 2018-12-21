import posbal_gen
import unittest
import os

class PosBalGenTest(unittest.TestCase):
    '''
    Testing generators for both the U-Net and CapsNets Versions
        For each generator, the directories may vary; Fork and change them for individual testing
    * RandomPatchGenerator
    '''
    def setUp(self):
        task_dir = 'C:\\Users\\jchen\\Desktop\\Datasets\\Task02_Heart\\'
        input_dir = task_dir + 'imagesTr\\'
        labels_dir = task_dir + 'labelsTr\\'

        # settings
        self.batch_size = 2
        self.n_channels = 1
        self.normalize_mode = 'normalize'
        self.norm_range = [0,1]
        self.centered = False
        self.data_dirs = [input_dir, labels_dir]
        self.list_IDs = os.listdir(input_dir)

    # @unittest.skip("Works.")
    def test_random_patch_last(self):
        '''
        Only tests that it crops properly for channels_last; assumes everything else is done correctly
        * works 100% okay for channels_last
        '''
        patch_shape_2d = (128, 128)
        patch_shape_3d = (128,128,128)
        output_shape_2d = (self.batch_size, 128, 128, self.n_channels)
        output_shape_3d = (self.batch_size, 128, 128, 128, self.n_channels)
        # instantiating generators
        # rand_gen_caps = generators.RandomPatchGenerator(self.list_IDs, self.data_dirs, batch_size = self.batch_size,
        #                                 patch_shape = patch_shape_2d, normalize_mode = self.normalize_mode,
        #                                 range = self.norm_range
        #                                  )
        rand_gen_unet2d = posbal_gen.PositivePatchGenerator(self.list_IDs, self.data_dirs,
                                        batch_size = self.batch_size, patch_shape = patch_shape_2d, normalize_mode = self.normalize_mode,
                                        range = self.norm_range
                                         )
        rand_gen_unet3d = posbal_gen.PositivePatchGenerator(self.list_IDs, self.data_dirs,
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
