import numpy as np

def pad_nonint_extraction(image, orig_shape, coords):
    """
    Pads the cropped output from `io.io_utils`'s extract_nonint_region function
    Args:
        image: either the mask or the thresholded (= 0.5) segmentation prediction (x, y, z)
        orig_shape: Original shape of the 3D volume no channels
        coords: outputted coordinates from `extract_nonint_region`
    Returns:
        padded: numpy array of shape `orig_shape`
    """
    # trying to reverse the cropping with padding
    assert len(coords) == len(orig_shape), "Please make sure that the length of coords matches orig_shape."
    padding = [[coords[i][0], orig_shape[i] - coords[i][1]] for i in range(len(orig_shape))]
    padded = np.pad(image, padding, mode = 'constant', constant_values = 0)
    return padded

def undo_reshape_padding(image, orig_shape):
    """
    Undoes the padding done by the `reshape` function in `utils.io_func`
    Args:
        image: numpy array that was reshaped to a larger size using reshape
        orig_shape: original shape before padding (do not include the channels)
    Returns:
        the reshaped image
    """
    return image[:orig_img.shape[0], :orig_img.shape[1], :orig_img.shape[2]]

# def load_data_predict():
#     """
#     Loads generator, loads evaluation data, preprocesses (and corresponding coords), then predict, and make up for padding
#     Args:
#
#     Returns:
#         prediction: padded prediction
#         label: corresponding label (same shape as prediction)
#     """
