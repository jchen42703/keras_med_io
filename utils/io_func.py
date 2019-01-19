
# coding: utf-8
# funcions for quick testing
import os
from glob import glob
import SimpleITK as sitk
import numpy as np

from keras_med_io.batchgenerators.augmentations.spatial_transformations \
    import augment_spatial_nocrop
from keras_med_io.batchgenerators.augmentations.noise_augmentations \
    import augment_gaussian_noise

def to_do():
    """
    * histogram_normalization
    * resample for multiple channels with nibabel
    """
    return ()

def transforms(volume, segmentation, ndim, fraction_,
               variance_, data_format_in = "channels_last"):
    """
    Does data aug
    Random elastic deformations, random scaling, random rotations, gaussian noise
    * Assumes a batch size dimension.
    """
    # converts data to channels_first
    if data_format_in == "channels_last": # assumes no batch_size dim
        if ndim == 2:
            to_channels_first = [0,-1,1,2]
        elif ndim == 3:
            to_channels_first = [0,-1,1,2,3]
        volume = np.transpose(volume, to_channels_first)
        segmentation = np.transpose(segmentation, to_channels_first)

    volume, segmentation = augment_spatial_nocrop(
                                    volume,
                                    segmentation,
                                    ndim,
                                    border_mode_data='constant',
                                    alpha=(0, 750),
                                    sigma=(10, 13),
                                    scale=(0.8, 1.2),
                                    do_elastic_deform=True,
                                    do_scale=True,
                                    do_rotation=True,
                                    angle_x=(0, 2*np.pi),
                                    angle_y=(0, 0),
                                    angle_z=(0, 0),)
                                    # fraction=fraction_)
                                    #spacing=spacing_)
    if np.any(variance_ != 0):
        volume = augment_gaussian_noise(volume, noise_variance=variance_)
    # converts data to channels_last
    if ndim == 2:
        to_channels_last = [0,2,3,1]
    elif ndim == 3:
        to_channels_last = [0,2,3,4,1]
    volume = np.transpose(volume, to_channels_last)
    segmentation = np.transpose(segmentation, to_channels_last)
    return volume, segmentation


# helper functions
def normalization(arr, normalize_mode, norm_range = [0,1]):
    """
    Helper function: Normalizes the image based on the specified mode and range
    Args:
        arr: numpy array
        normalize_mode: either "whiten", "normalize_clip", or "normalize" representing the type of normalization to use
        norm_range: (Optional) Specifies the range for the numpy array values
    Returns:
        A normalized array based on the specifications
    """
    # reiniating the batch_size dimension
    if normalize_mode == "whiten":
        return whiten(arr)
    elif normalize_mode == "normalize_clip":
        return normalize_clip(arr,  norm_range = norm_range)
    elif normalize_mode == "normalize":
        return minmax_normalize(arr, norm_range = norm_range)
    else:
        return NotImplementedError("Please use the supported modes.")

def normalize_clip(arr, norm_range = [0,1]):
    """
    Args:
        arr: numpy array
        norm_range: list of 2 integers specifying normalizing range
            based on https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    Returns:
        Whitened and normalized array with outliers clipped in the specified range
    """
    # whitens -> clips -> scales to [0,1]
    if isinstance(arr,sitk.Image):
        arr = sitk.GetArrayFromImage(arr)
    # whiten
    norm_img = np.clip(whiten(arr), -5, 5)
    norm_img = minmax_normalize(arr, norm_range)
    return norm_img

def whiten(arr):
    """
    Mean-Var Normalization
    * mean of 0 and standard deviation of 1
    Args:
        arr: numpy array
    Returns:
        A numpy array with a mean of 0 and a standard deviation of 1
    """
    shape = arr.shape
    arr = arr.flatten()
    norm_img = (arr-np.mean(arr)) / np.std(arr)
    return norm_img.reshape(shape)

def minmax_normalize(arr, norm_range = [0,1]):
    """
    Args:
        arr: numpy array
        norm_range: list of 2 integers specifying normalizing range
            based on https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    Returns:
        Normalized array with outliers clipped in the specified range
    """
    norm_img = ((norm_range[1]-norm_range[0]) * (arr - np.amin(arr)) / (np.amax(arr) - np.amin(arr))) + norm_range[0]
    return norm_img

def resample_img(itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False):
    """
    Resamples a SimpleITK image to the desired voxel spacing.
    Args:
        itk_image: a 2D/3D SimpleITK image
        out_spacing: a list of the spacing
        is_label: boolean on whether the itk_image is a label
    Returns:
        a resample SimpleITK image
    """
    if isinstance(itk_image, np.ndarray):
        itk_image = sitk.GetImageFromArray(itk_image)
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

# def histogram_normalization():
#     return Exception("Not Implemented")

def N4_bias_correction(img):
    """
    Args:
        img: A SimpleITK image
    Returns:
        A bias corrected SimpleITK image
    """
    # https://github.com/poornasandur/BRATS_N4Bias/blob/master/test3.py
    img=sitk.Cast(img,sitk.sitkFloat32)
    img_mask=sitk.Cast(sitk.BinaryNot(sitk.BinaryThreshold(img, 0, 0)), sitk.sitkUInt8)
    return sitk.N4BiasFieldCorrection(img, img_mask)

def get_multi_class_labels(data, n_labels, labels=None, remove_background = False):
    """
    One-hot encodes a segmentation label.
    Args:
        data: numpy array containing the label map with shape: (n_samples,..., 1).
        n_labels: number of labels
        labels: list of the integer/float values of the labels
        remove_background: option to drop the background mask (first label 0)
    Returns:
        binary numpy array of shape (n_samples,..., n_labels) or (n_samples,..., n_labels-1)
    """

    new_shape = data.shape + (n_labels,)
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            # with the labels specified
            y[:,:,:, label_index][data == labels[label_index]] = 1
        else:
            # automated
            y[:, :, :, label_index][data == (label_index + 1)] = 1
    if remove_background:
        without_background = n_labels - 1
        y = y[-without_background:] # removing the background
    return y

def get_list_IDs(data_dir, splits = [0.6, 0.2, 0.2]):
    """
    Divides filenames into train/val/test sets
    Args:
        data_dir: file path to the directory of all the files; assumes labels and training images have same names
        splits: a list with 3 elements corresponding to the decimal train/val/test splits; [train, val, test]
    Returns:
        a dictionary of file ids for each set
    """
    id_list = os.listdir(data_dir)
    total = len(id_list)
    train = round(total * splits[0])
    val_split = round(total * splits[1]) + train
    return {"train": id_list[:train], "val": id_list[train:val_split], "test": id_list[val_split:]
           }

def sanity_checks(patch_x, patch_y):
    """
    Checks for NaNs, and makes sure that the labels are one-hot encoded.
    Args:
        patch_x: a numpy array
        patch_y: a numpy array (label)
    Returns:
        True (boolean) if all the asserts run.
    """
    # sanity checks
    checks_nan_x, checks_nan_y = np.any(np.isnan(patch_x)), np.any(np.isnan(patch_y))
    assert not checks_nan_x and not checks_nan_y # NaN checks
    assert np.array_equal(np.unique(patch_y), np.array([0,1])) or np.array_equal(np.unique(patch_y), np.array([0]))
    return True

def add_channel(image):
    """
    Adds a single channel dimension to a 3D or 2D monomodal SimpleITK image or numpy array.
    Args:
        image: a monomodal SimpleITK image or numpy array without a channel dimension
    Returns:
        A single channel numpy array (channels_last)
    """
    if isinstance(image, sitk.Image):
        image = GetArrayFromImage(image)
    return np.expand_dims(image.squeeze(), -1).astype(np.float32)
