# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from warnings import warn

from keras_med_io.batchgenerators.transforms.abstract_transforms import AbstractTransform
from keras_med_io.batchgenerators.augmentations.spatial_transformations import augment_spatial, augment_channel_translation, \
    augment_mirroring, augment_transpose_axes, augment_zoom, augment_resize, flip_vector_axis
import numpy as np



class ZoomTransform(AbstractTransform):
    def __init__(self, zoom_factors=1, order=3, order_seg=1, cval_seg=0, data_key="data", label_key="seg"):
        """
        Zooms 'data' (and 'seg') by zoom_factors
        :param zoom_factors: int or list/tuple of int
        :param order: interpolation order for data (see skimage.transform.resize)
        :param order_seg: interpolation order for seg (see skimage.transform.resize)
        :param cval_seg: cval for segmentation (see skimage.transform.resize)
        :param seg: can be None, if not None then it will also be zoomed by zoom_factors. Can also be list/tuple of
        np.ndarray (just like data). Must also be (b, c, x, y(, z))
        :param concatenate_list: if you give list/tuple of data/seg and set concatenate_list=True then the result will be
        concatenated into one large ndarray (once again b, c, x, y(, z))
        :param data_key:
        :param label_key:

        """
        self.cval_seg = cval_seg
        self.order_seg = order_seg
        self.data_key = data_key
        self.label_key = label_key
        self.order = order
        self.zoom_factors = zoom_factors

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        ret_val = augment_zoom(data=data, seg=seg, zoom_factors=self.zoom_factors, order=self.order, order_seg=self.order_seg, cval_seg=self.cval_seg)

        data_dict[self.data_key] = ret_val[0]
        if seg is not None:
            data_dict[self.label_key] = ret_val[1]
        return data_dict


class ResizeTransform(AbstractTransform):

    def __init__(self, target_size, order=3, order_seg=1, cval_seg=0, concatenate_list=False, data_key="data", label_key="seg"):
        """
        Reshapes 'data' (and 'seg') to target_size
        :param target_size: int or list/tuple of int
        :param order: interpolation order for data (see skimage.transform.resize)
        :param order_seg: interpolation order for seg (see skimage.transform.resize)
        :param cval_seg: cval for segmentation (see skimage.transform.resize)
        :param seg: can be None, if not None then it will also be resampled to target_size. Can also be list/tuple of
        np.ndarray (just like data). Must also be (b, c, x, y(, z))
        :param concatenate_list: if you give list/tuple of data/seg and set concatenate_list=True then the result will be
        concatenated into one large ndarray (once again b, c, x, y(, z))
        :param data_key:
        :param label_key:

        """
        self.concatenate_list = concatenate_list
        self.cval_seg = cval_seg
        self.order_seg = order_seg
        self.data_key = data_key
        self.label_key = label_key
        self.order = order
        self.target_size = target_size

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        ret_val = augment_resize(data=data, seg=seg, target_size=self.target_size, order=self.order,
                                 order_seg=self.order_seg, cval_seg=self.cval_seg,
                                 concatenate_list=self.concatenate_list)

        data_dict[self.data_key] = ret_val[0]
        if seg is not None:
            data_dict[self.label_key] = ret_val[1]
        return data_dict


class MirrorTransform(AbstractTransform):
    """ Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5

    Args:
        axes (tuple of int): axes along which to mirror

    """
    def __init__(self, axes=(2, 3, 4), data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key
        self.axes = axes
        warnings.simplefilter("once", DeprecationWarning)
        warn("The axes in MirrorTransform will soon change! Currently for mirroring along any axes of a 5d tensor "
             "you would set axes=(2, 3, 4). These correspond to the actual axes of a 5d tensor.\n"
             "The way axes are done in the future is to access the spatial dimensions directly, disregarding b and c. "
             "For the same 5d tensor you will have to set axes=(0, 1, 2)!!\n"
             "(don't to anything for now, this is just a warning)")

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        ret_val = augment_mirroring(data=data, seg=seg, axes=self.axes)

        data_dict[self.data_key] = ret_val[0]
        if seg is not None:
            data_dict[self.label_key] = ret_val[1]

        return data_dict


class ChannelTranslation(AbstractTransform):
    """Simulates badly aligned color channels/modalities by shifting them against each other

    Args:
        const_channel: Which color channel is constant? The others are shifted

        max_shifts (dict {'x':2, 'y':2, 'z':2}): How many pixels should be shifted for each channel?

    """
    def __init__(self, const_channel=0, max_shifts=None, data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key
        self.max_shift = max_shifts
        self.const_channel = const_channel

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)

        ret_val = augment_channel_translation(data=data, const_channel=self.const_channel, max_shifts=self.max_shift)

        data_dict[self.data_key] = ret_val[0]

        return data_dict


class SpatialTransform(AbstractTransform):
    """The ultimate spatial transform generator. Rotation, deformation, scaling, cropping: It has all you ever dreamed
    of. Computational time scales only with patch_size, not with input patch size or type of augmentations used.
    Internally, this transform will use a coordinate grid of shape patch_size to which the transformations are
    applied (very fast). Interpolation on the image data will only be done at the very end

    Args:
        patch_size (tuple/list/ndarray of int): Output patch size

        patch_center_dist_from_border (tuple/list/ndarray of int, or int): How far should the center pixel of the
        extracted patch be from the image border? Recommended to use patch_size//2.
        This only applies when random_crop=True

        do_elastic_deform (bool): Whether or not to apply elastic deformation

        alpha (tuple of float): magnitude of the elastic deformation; randomly sampled from interval

        sigma (tuple of float): scale of the elastic deformation (small = local, large = global); randomly sampled
        from interval

        do_rotation (bool): Whether or not to apply rotation

        angle_x, angle_y, angle_z (tuple of float): angle in rad; randomly sampled from interval. Always double check
        whether axes are correct!

        do_scale (bool): Whether or not to apply scaling

        scale (tuple of float): scale range ; scale is randomly sampled from interval

        border_mode_data: How to treat border pixels in data? see scipy.ndimage.map_coordinates

        border_cval_data: If border_mode_data=constant, what value to use?

        order_data: Order of interpolation for data. see scipy.ndimage.map_coordinates

        border_mode_seg: How to treat border pixels in seg? see scipy.ndimage.map_coordinates

        border_cval_seg: If border_mode_seg=constant, what value to use?

        order_seg: Order of interpolation for seg. see scipy.ndimage.map_coordinates. Strongly recommended to use 0!
        If !=0 then you will have to round to int and also beware of interpolation artifacts if you have more then
        labels 0 and 1. (for example if you have [0, 0, 0, 2, 2, 1, 0] the neighboring [0, 0, 2] bay result in [0, 1, 2])

        random_crop: True: do a random crop of size patch_size and minimal distance to border of
        patch_center_dist_from_border. False: do a center crop of size patch_size
    """
    def __init__(self, patch_size, patch_center_dist_from_border=30,
                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                 do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, data_key="data", label_key="seg", p_el_per_sample=1, p_scale_per_sample=1, p_rot_per_sample=1):
        self.p_rot_per_sample = p_rot_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_el_per_sample = p_el_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.patch_size = patch_size
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
        self.random_crop = random_crop

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        if self.patch_size is None:
            if len(data.shape) == 4:
                patch_size = (data.shape[2], data.shape[3])
            elif len(data.shape) == 5:
                patch_size = (data.shape[2], data.shape[3], data.shape[4])
            else:
                raise ValueError("only support 2D/3D batch data.")
        else:
            patch_size = self.patch_size

        ret_val = augment_spatial(data, seg, patch_size=patch_size,
                                  patch_center_dist_from_border=self.patch_center_dist_from_border,
                                  do_elastic_deform=self.do_elastic_deform, alpha=self.alpha, sigma=self.sigma,
                                  do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                  angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                  border_mode_data=self.border_mode_data,
                                  border_cval_data=self.border_cval_data, order_data=self.order_data,
                                  border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                  order_seg=self.order_seg, random_crop=self.random_crop,
                                  p_el_per_sample=self.p_el_per_sample, p_scale_per_sample=self.p_scale_per_sample,
                                  p_rot_per_sample=self.p_rot_per_sample)

        data_dict[self.data_key] = ret_val[0]
        if seg is not None:
            data_dict[self.label_key] = ret_val[1]

        return data_dict


class TransposeAxesTransform(AbstractTransform):
    def __init__(self, transpose_any_of_these=(2, 3, 4), data_key="data", label_key="seg"):
        '''
        This transform will randomly shuffle the axes of transpose_any_of_these.
        :param transpose_any_of_these:
        :param data_key:
        :param label_key:
        '''
        self.data_key = data_key
        self.label_key = label_key
        self.transpose_any_of_these = transpose_any_of_these
        warnings.simplefilter("once", DeprecationWarning)
        warn("The axes in TransposeAxesTransform will soon change! Currently for transposing any axes of a 5d tensor "
             "you would set axes=(2, 3, 4). These correspond to the axes of a 5d tensor.\n"
             "The way axes are done in the future is to access the spatial dimensions directly, disregarding b and c. "
             "For the same 5d tensor you will have to set axes=(0, 1, 2)!!\n"
             "(don't to anything for now, this is just a warning)")

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        ret_val = augment_transpose_axes(data, seg, self.transpose_any_of_these)

        data_dict[self.data_key] = ret_val[0]
        if seg is not None:
            data_dict[self.label_key] = ret_val[1]
        return data_dict


class FlipVectorAxisTransform(AbstractTransform):
    """ Expects as input an image with 3 3D-vectors at each voxels, encoded as a nine-channel image. Will randomly
    flip sign of one dimension of all 3 vectors (x, y or z).
    """
    def __init__(self, axes=(2, 3, 4), data_key="data"):
        self.data_key = data_key
        self.axes = axes

    def __call__(self, **data_dict):
        data_dict[self.data_key] = flip_vector_axis(data=data_dict[self.data_key])
        return data_dict
