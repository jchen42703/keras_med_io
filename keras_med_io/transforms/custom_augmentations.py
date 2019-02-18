from builtins import range
import numpy as np

# def random_positive_crop(data, seg, crop_size):
#     """
#     Args:
#         data: numpy array with shape (batch_size, n_channels, x, y, (z,))
#         seg: same shape as data
#         crop_size: (x,y,(z,))
#     """
#     ndim = len(crop_size) # inferring the number of dimensions
#     # concatenating so that cropping is easier
#     both = np.concatenate([data, seg], axis = 1)
#     image_shape = data.shape[:ndim]
#     n_channels = data.shape[1]
#     data_dtype = data[0].dtype
#     seg_dtype = seg[0].dtype
#
#     patch_idx = _get_positive_idx(seg)
#     if ndim == 2:
#         # slicing image into 2D cross sections
#         both = both[:, :, patch_idx[0]]
#     # patch extraction
#     data_return = np.zeros([data_shape[0], data_shape[1]] + list(crop_size), dtype=data_dtype)
#     seg_return = np.zeros([seg_shape[0], seg_shape[1]] + list(crop_size), dtype=seg_dtype)
#     for b in range(data_shape[0]): # for each batch
#         # checks for cases where there may be:
#         need_to_pad = [[0, 0]] + [[abs(min(0, lbs[d])), #negative indices
#                                    abs(min(0, data_shape_here[d + 2] - (lbs[d] + crop_size[d])))] #when cropping -> negative idx
#                                   for d in range(dim)]
#         # Note: ^ If an idx is negative, we're gonna pad it later by abs(idx) bc we're compensating for the distance that's lost
#
#         # we should crop first, then pad -> reduces i/o for memmaps, reduces RAM usage and improves speed
#         # computes the max idx (idx + crop_size) or just the `data shape` if (idx+crop_size) > data_shape
#         ubs = [min(lbs[d] + crop_size[d], data_shape_here[d+2]) for d in range(dim)]
#         # runs checks on the lbs, just making sure that they're not negative
#         lbs = [max(0, lbs[d]) for d in range(dim)]
#
#         # cropping use a list of built-in slice functions
#         slicer_data = [slice(0, data_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
#         data_cropped = data[b][slicer_data]
#
#         if seg_return is not None:
#             slicer_seg = [slice(0, seg_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
#             seg_cropped = seg[b][slicer_seg]
#         # padding
#         if any([i > 0 for j in need_to_pad for i in j]):
#             print("padding is run")
#             data_return[b] = np.pad(data_cropped, need_to_pad, pad_mode, **pad_kwargs)
#             if seg_return is not None:
#                 seg_return[b] = np.pad(seg_cropped, need_to_pad, pad_mode_seg, **pad_kwargs_seg)
#         else:
#             data_return[b] = data_cropped
#             if seg_return is not None:
#                 seg_return[b] = seg_cropped
#
#         both_crop = extract_patch(both, patch_shape, patch_idx)
#     data, seg = both_crop[:,:n_channels], both_crop[:, n_channels:]
#     return data, seg
def get_random_slice_idx(arr):
    slice_dim = arr.shape[0]
    return np.random.choice(slice_dim)

def get_positive_idx(label, channels_format = "channels_last"):
    """
    Gets a random positive patch index that does not include the channels and batch_size dimensions.
    Args:
        label: one-hot encoded numpy array with the dims (n_channels, x,y,z)
    Returns:
        A numpy array representing a 3D random positive patch index
    """
    try:
        assert len(label.shape) == 4
    except AssertionError:
        # adds the channel dim if it doesn't already have it
        if channels_format == "channels_first":
            label = np.expand_dims(label, axis = 0)
        elif channels_format == "channels_last":
            label = np.expand_dims(label, axis = -1)

    # "n_dims" numpy arrays of all possible positive pixel indices for the label
    if channels_format == "channels_first":
        pos_idx_new = np.nonzero(label)[1:]
    elif channels_format == "channels_last":
        pos_idx_new = np.nonzero(label)[:-1]

    # finding random positive class index
    pos_idx = np.dstack(pos_idx_new).squeeze()
    random_coord_idx = np.random.choice(pos_idx.shape[0]) # choosing random coords out of pos_idx
    random_pos_coord = pos_idx[random_coord_idx]
    return random_pos_coord

def extract_patch(data, patch_shape, patch_index):
    """
    Extracting both 2D and 3D patches depending on patch shape dimensions
    Args:
        data: a numpy array of shape (n_channels, x, y(,z)))
        patch_shape: a tuple representing the patch shape without the batch_size or the channels dimensions
        patch_index: 3D coordinate of where to start cropping
    Returns:
        a cropped version of the original image array
    """
    ndim = len(patch_shape) # inferring the number of dimensions
    patch_index = np.asarray(patch_index, dtype = np.int16)
    patch_shape = np.asarray(patch_shape, dtype = np.int16)

    image_shape = data.shape[-ndim:]
    if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape):
        data, patch_index = fix_out_of_bound_patch_attempt(data, patch_shape, patch_index)

    if ndim == 2:
        return data[patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1], ...]#, ...]
    elif ndim == 3:
        return data[patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1],
                    patch_index[2]:patch_index[2]+patch_shape[2], ...]

def fix_out_of_bound_patch_attempt(data, patch_shape, patch_index):
    """
    Pads the data and alters the corner patch index so that the patch will be correct.
    Args:
        data:
        patch_shape:
        patch_index:
    Returns:
        padded data, fixed patch index
    """
    ndim = len(patch_shape)
    image_shape = data.shape[-ndim:]
    # figures out which indices need to be padded; if they're < 0
    pad_before = np.abs((patch_index < 0 ) * patch_index) # also need to check if idx-patch_shape < 0
    # checking for out of bounds if doing idx+patch shape by replacing the afflicted indices with a kinda random replacement
    pad_after = np.abs(((patch_index + patch_shape) > image_shape) * ((patch_index + patch_shape) - image_shape))
    pad_args = np.stack([pad_before, pad_after], axis=-1)
    if pad_args.shape[0] < len(data.shape):
        # adding channels dimension to padding ([0,0] so that it's ignored)
        pad_args = pad_args.tolist() + [[0, 0]] * (len(data.shape) - pad_args.shape[0])

    data = np.pad(data, pad_args, mode="edge")
    patch_index += pad_before
    return data, patch_index

def extract_slice_balanced(data, n_pos):
    """
    Args:
        data: (batch_size, n_channels, x, y(,z))
        n_pos: number of positive slices to extract in a batch (everything else is randomly extracted)
    """
    data_dtype = data.dtype
    seg_dtype = seg.dtype
    # iterating through batch to grab all of the indices
    output_shape = [batch_size] + list(data.shape[2:])
    data_return = np.zeros(output_shape, dtype = data_dtype)
    seg_return = np.zeros(output_shape, dtype = seg_dtype)
    # THIS IS NOT GOING TO WORK BECAUSE WHEN BATCH_SIZE IS BIG (192) THEN THIS IS SLOW
    for batch in range(data.shape[0]):
        _data = data[batch]
        data_shape_here = _data.shape # (n_channels, x, y, z)
        rand_pos_slice_idx = get_positive_idx(seg)[0]
