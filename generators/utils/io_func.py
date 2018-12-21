
# coding: utf-8
# funcions for quick testing
import os
from glob import glob

import SimpleITK as sitk
import numpy as np
def to_do():
    '''
    * histogram_normalization
    * channels_last for get_multi_class_labels
    '''
# helper functions
def whitening(arr):
    '''
    Mean-Var Normalization
    * mean of 0 and standard deviation of 1

    param arr: numpy array to whiten
    '''
    return arr-np.mean(arr) / (np.std(arr))

def normalize(arr, range = [0,1]):
    '''
    Args:
        arr: numpy array
        range: list of 2 integers specifying normalizing range
            based on https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    Returns:
        Normalized array with outliers clipped in the specified range
    '''
    norm_img = (range[1]-range[0]) * (arr - np.amin(arr)) / (np.amax(arr) - np.amin(arr)) + range[0]
    return norm_img

def normalize_clip(arr, range = [0,1]):
    '''
    Args:
        arr: numpy array
        range: list of 2 integers specifying normalizing range
            based on https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    Returns:
        Whitened and normalized array with outliers clipped in the specified range
    '''
    # whitens -> clips -> scales to [0,1]
    if isinstance(arr,sitk.Image):
        arr = sitk.GetArrayFromImage(arr)
    # whiten
    norm_img = np.clip((arr-np.mean(arr)) / (np.std(arr)),
                        -5, 5)
    norm_img = (range[1]-range[0]) * (norm_img - np.amin(norm_img)) / (np.amax(norm_img) - np.amin(norm_img)) + range[0]

    return norm_img

def resample_img(itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False):
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

def histogram_normalization():
    return Exception("Not Implemented")

def N4_bias_correction(img):
    # https://github.com/poornasandur/BRATS_N4Bias/blob/master/test3.py
    img=sitk.Cast(img,sitk.sitkFloat32)
    img_mask=sitk.Cast(sitk.BinaryNot(sitk.BinaryThreshold(img, 0, 0)), sitk.sitkUInt8)
    return sitk.N4BiasFieldCorrection(img, img_mask)

# def get_multi_class_labels(data, n_labels, labels=None):
#     """
#     CHANNELS_FIRST (still not updated to channels_last)
#     Translates a label map into a set of binary labels.
#     :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
#         ** if regular images are 4D, the labels should be expanded to 4D as well
#     :param n_labels: number of labels.
#     :param labels: integer values of the labels.
#     :return: binary numpy array of shape: (n_samples, n_labels, ...)
#     """
#     if isinstance(arr,sitk.Image):
#         arr = sitk.GetArrayFromImage(arr)
#
#     new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
#     y = np.zeros(new_shape, np.int8)
#     for label_index in range(n_labels):
#         if labels is not None:
#             y[:, label_index][data[:, 0] == labels[label_index]] = 1
#         else:
#             y[:, label_index][data[:, 0] == (label_index + 1)] = 1
#     return y

def get_multi_class_labels_first(data, n_labels, labels=None):
    """
    CHANNELS_FIRST
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
        ** if regular images are 4D, the labels should be expanded to 4D as well
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    if isinstance(arr,sitk.Image):
        arr = sitk.GetArrayFromImage(arr)

    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    return y

############### utility
def get_file_lists(data_dir):

    # setting directories
    dataset_train= 'imagesTr'
    dataset_labels= 'labelsTr'
    BASE_IMG_PATH=os.path.join(data_dir, dataset_train)
    BASE_IMG_PATH_LABELS=os.path.join(data_dir, dataset_labels)

    # getting list of paths for files
    trainList = glob(BASE_IMG_PATH + '/**/*.nii', recursive=True)
    labelList = glob(BASE_IMG_PATH_LABELS + '/**/*.nii', recursive=True)

    print ( BASE_IMG_PATH + '\n' + "Number of images: {}". format (len (trainList)))
    print ( BASE_IMG_PATH_LABELS + '\n' + "Number of images: {}". format (len (labelList)))
    return trainList, labelList

def metadata(arrays):
    for arr in arrays:
        print('shape: {}, min: {}, max: {}'.format(arr.shape, np.amin(arr), np.amax(arr)))

def get_list_IDs(data_dir, val_split = 0.8):
    """
    param data_dir: data directory to the training or label folder
    * assumes labels and training images have same names
    """
    id_list = os.listdir(data_dir)
    total = len(id_list)
    train = round(total * val_split)
    return {'train': id_list[:train], 'val': id_list[train:]
           }

def convert_4D_to_3D(arr_4D):
    '''
    arr_4D: sitk image (x,y,z, n_channels)
    returns: list of 3D np arrays (x,y,z)
    '''
    arrays = []
    np_img = sitk.GetArrayFromImage(arr_4D)
    ### OPPORTUNITY TO GET 4D PATCHES HERE

    # getting list of 3D SimpleITK Images for each modality/channel
    for arr_3D in np.split(np_img, arr_4D.GetDimension(), axis= 0):
        arrays.append(arr_3D.squeeze())
    return arrays

def to_channels_first(arr):
    # converts 4D nd.array to channels first; (x,y,z, n_channels) to (x,n_channels,y,z)
    if len(arr.shape) == 4:
        return np.transpose(arr, [3, 0, 1,2])
    elif len(arr.shape) == 3:
        return np.expand_dims(arr, axis = 0)
