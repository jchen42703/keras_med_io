from keras_med_io.generators.posrandom_generator_noresamp import PosRandomPatchGenerator

import unittest
import os

class GeneratorTest(unittest.TestCase):
    """
    Testing generators.
    """
    def setUp(self):
        task_dir = 'C:\\Users\\jchen\\Desktop\\Datasets\\Heart_Resampled\\'
        input_dir = task_dir + 'imagesTr\\'
        labels_dir = task_dir + 'labelsTr\\'

        # settings
        self.batch_size = 2
        self.normalize_mode = "normalize"
        self.norm_range = [0,1]
        self.data_dirs = [input_dir, labels_dir]
        self.list_IDs = os.listdir(input_dir)

        self.patch_shape_2D = (128, 128)
        self.patch_shape_3D = (128,128,128)
        self.output_shape_2D = (self.batch_size, 128, 128, 1)
        self.output_shape_3D = (self.batch_size, 128, 128, 128, 1)
        self.n_channels = self.output_shape_2D[-1]
        self.n_classes = 2

    def test_random_patch_2D(self):
        """
        Tests shapes for 2D random patch generation from the PosRandomPatchGenerator.
        """
        # instantiating generators
        mode = "rand"
        rand_gen_2D = PosRandomPatchGenerator(self.list_IDs, self.data_dirs, self.batch_size, self.patch_shape_2D,
                                               self.n_channels, self.n_classes, mode = mode,)
        # testing their output shapes
        x_2, y_2 = rand_gen_2D.__getitem__(1)
        assert x_2.shape == y_2.shape
        self.assertEqual(x_2.shape, self.output_shape_2D)

    def test_random_patch_3D(self):
        """
        Tests shapes for 3D random patch generation from the PosRandomPatchGenerator.
        """
        # instantiating generators
        mode = "rand"
        rand_gen_3D = PosRandomPatchGenerator(self.list_IDs, self.data_dirs, self.batch_size, self.patch_shape_3D,
                                               self.n_channels, self.n_classes, mode = mode,)
        # testing their output shapes
        x_3, y_3 = rand_gen_3D.__getitem__(1)
        assert x_3.shape == y_3.shape
        self.assertEqual(x_3.shape, self.output_shape_3D)

    def test_pos_patch_2D(self):
        """
        Tests shapes for 2D random positive patch generation from the PosRandomPatchGenerator.
        """
        # instantiating generators
        mode = "pos"
        pos_gen_2D = PosRandomPatchGenerator(self.list_IDs, self.data_dirs, self.batch_size, self.patch_shape_2D,
                                               self.n_channels, self.n_classes, mode = mode,)
        #testing their output shapes
        x_2, y_2 = pos_gen_2D.__getitem__(1)
        assert x_2.shape == y_2.shape
        self.assertEqual(x_2.shape, self.output_shape_2D)

    def test_pos_patch_3D(self):
        """
        Tests shapes for 3D random positive patch generation from the PosRandomPatchGenerator.
        """
        mode = "pos"
        pos_gen_3D = PosRandomPatchGenerator(self.list_IDs, self.data_dirs, self.batch_size, self.patch_shape_3D,
                                               self.n_channels, self.n_classes, mode = mode,)
        x_3, y_3 = pos_gen_3D.__getitem__(1)
        assert x_3.shape == y_3.shape
        self.assertEqual(x_3.shape, self.output_shape_3D)

    def test_bal_patch_2D(self):
        # instantiating generators
        mode = "bal"
        bal_gen_2D = PosRandomPatchGenerator(self.list_IDs, self.data_dirs, self.batch_size, self.patch_shape_2D,
                                               self.n_channels, self.n_classes, mode = mode,)
        #testing their output shapes
        x_2, y_2 = bal_gen_2D.__getitem__(1)
        assert x_2.shape == y_2.shape
        self.assertEqual(x_2.shape, self.output_shape_2D)

    def test_bal_patch_3D(self):
        mode = "bal"
        bal_gen_3D = PosRandomPatchGenerator(self.list_IDs, self.data_dirs, self.batch_size, self.patch_shape_3D,
                                               self.n_channels, self.n_classes, mode = mode,)
        x_3, y_3 = bal_gen_3D.__getitem__(1)
        assert x_3.shape == y_3.shape
        self.assertEqual(x_3.shape, self.output_shape_3D)

unittest.main(argv=[''], verbosity=2, exit=False)
