from keras_med_io.utils.io_func import *
import unittest
import os

class IOTest(unittest.TestCase):
    """
    Testing IO.
    Functions to test:
        normalization
    Can't really test:
        get_multi_class_labels:
        resample_img
        get_list_IDs
    """
    def setUp(self):
        """
        Initializing the parameters:
            n_channels:
            shapes: tuples with channels_first shapes
            images & labels: numpy arrays
            extractors: 2D and 3D versions for all the patch extractor classes.
        """
        self.n_channels = 4
        self.image_shape_2D = (408, 408, self.n_channels)
        self.patch_shape_2D = np.asarray((128, 128))
        self.train_image_2D = np.arange(0, self.image_shape_2D[0]*self.image_shape_2D[1]*self.n_channels \
                                ).reshape(self.image_shape_2D)
        self.label_image_2D = np.ones(self.image_shape_2D)

        self.image_shape_3D = (155, 240, 240, self.n_channels)
        self.patch_shape_3D = np.asarray((128,128,128))
        self.train_image_3D = np.arange(0, self.image_shape_3D[0]*self.image_shape_3D[1]*self.image_shape_3D[2]*self.n_channels \
                                ).reshape(self.image_shape_3D)
        self.label_image_3D = np.ones(self.image_shape_3D)
        self.overlap = 0

    def test_normalize(self):
        """
        Tests the pixel value range for normalize.
        """
        norm_range = [0,1]
        norm_img = normalization(self.train_image_2D, normalize_mode = "normalize", norm_range = norm_range)
        self.assertEqual(norm_img.min(), norm_range[0])
        self.assertEqual(norm_img.max(), norm_range[1])

    def test_normalize_clip(self):
        """
        Tests the pixel value range for normalize_clip
        """
        norm_range = [0,1]
        norm_img = normalization(self.train_image_2D, normalize_mode = "normalize_clip", norm_range = norm_range)
        self.assertEqual(norm_img.min(), norm_range[0])
        self.assertEqual(norm_img.max(), norm_range[1])

    def test_whiten(self):
        """
        Tests that a whitened image has a standard deviation of 1 and a mean of 0.
        """
        # print(self.train_image_2D.std())
        norm_img = normalization(self.train_image_2D, normalize_mode = "whiten")
        self.assertEqual(np.round(norm_img.std(), decimals = 2), 1.00)
        self.assertEqual(np.round(norm_img.mean(), decimals = 2), 0.00)

    def test_flatten_and_reshape(self):
        """
        Tests that flattening an array and reshaping it back to the original shape will produce the same array
        """
        flatten = self.train_image_2D.flatten()
        reshaped = flatten.reshape(self.image_shape_2D)
        self.assertTrue(np.array_equal(self.train_image_2D, reshaped))

unittest.main(argv=[''], verbosity=2, exit=False)
