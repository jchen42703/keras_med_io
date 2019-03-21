# coding: utf-8
# funcions for quick testing
import numpy as np
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
    # whiten
    norm_img = np.clip(whiten(arr), -5, 5)
    norm_img = minmax_normalize(arr, norm_range)
    return norm_img

def whiten(arr):
    """
    Mean-Var Normalization (Z-score norm)
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

def clip_upper_lower_percentile(image, mask=None, percentile_lower=0.2, percentile_upper=99.8):
    """
    Clipping values for positive class areas.
    Args:
        image:
        mask:
        percentile_lower:
        percentile_upper:
    Return:
        Image with clipped pixel intensities
    """
    # if mask is None:
    #     mask = image != image[0,0,0]
    # finding the percentile values
    cut_off_lower = np.percentile(image[mask != 0].ravel(), percentile_lower)
    cut_off_upper = np.percentile(image[mask != 0].ravel(), percentile_upper)
    # clipping based on percentiles
    res = np.copy(image)
    if mask is not None:
        res[(res < cut_off_lower) & (mask !=0)] = cut_off_lower
        res[(res > cut_off_upper) & (mask !=0)] = cut_off_upper
    elif mask is None:
        res[(res < cut_off_lower)] = cut_off_lower
        res[(res > cut_off_upper)] = cut_off_upper
    return res
