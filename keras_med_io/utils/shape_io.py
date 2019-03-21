from skimage.transform import resize
import numpy as np

def reshape(orig_img, append_value=-1024, new_shape=(512, 512, 512)):
    """
    Reshapes a numpy array with the specified values where the new shape must have >= to the number of
    voxels in the original image. If # of voxels in `new_shape` is > than the # of voxels in the original image,
    then the extra will be filled in by `append_value`.
    Args:
        orig_img:
        append_value: filler value
        new_shape: must have >= to the number of voxels in the original image.
    """
    reshaped_image = np.zeros(new_shape)
    reshaped_image[...] = append_value
    x_offset = 0
    y_offset = 0  # (new_shape[1] - orig_img.shape[1]) // 2
    z_offset = 0  # (new_shape[2] - orig_img.shape[2]) // 2

    reshaped_image[x_offset:orig_img.shape[0]+x_offset, y_offset:orig_img.shape[1]+y_offset, z_offset:orig_img.shape[2]+z_offset] = orig_img
    # insert temp_img.min() as background value

    return reshaped_image

def extract_nonint_region(image, mask = None, outside_value=0, coords = False):
    """
    Resizing image around a specified region (i.e. nonzero region)
    Args:
        image:
        mask: a segmentation labeled mask that is the same shaped as 'image' (optional; default: None)
        outside_value: (optional; default: 0)
        coords: boolean on whether or not to return boundaries (bounding box coords) (optional; default: False)
    Returns:
        the resized image
        segmentation mask (when mask is not None)
        a nested list of the mins and and maxes of each axis (when coords = True)
    """
    pos_idx = np.where(image != outside_value)
    # fetching all of the min/maxes for each axes
    pos_x, pos_y, pos_z = pos_idx[1], pos_idx[2], pos_idx[0]
    minZidx, maxZidx = int(np.min(pos_z)), int(np.max(pos_z))
    minXidx, maxXidx = int(np.min(pos_x)), int(np.max(pos_x))
    minYidx, maxYidx = int(np.min(pos_y)), int(np.max(pos_y))
    # resize images
    resizer = (slice(minZidx, maxZidx), slice(minXidx, maxXidx), slice(minYidx, maxYidx))
    coord_list = [[minZidx, maxZidx], [minXidx, maxXidx], [minYidx, maxYidx]]
    # returns cropped outputs with the bbox coordinates
    if coords:
        if mask is not None:
            return (image[resizer], mask[resizer], coord_list)
        elif mask is None:
            return (image[resizer], coord_list)
    # returns just cropped outputs
    elif not coords:
        if mask is not None:
            return (image[resizer], mask[resizer])
        elif mask is None:
            return (image[resizer])

def resample_array(src_imgs, src_spacing, target_spacing, is_label = False):
    """
    From: https://github.com/MIC-DKFZ/medicaldetectiontoolkit/
    Resamples a numpy array.
    ** the spacings must be the same dimension as the input numpy array
    Args:
        src_imgs: numpy array
        srs_spacing:
        target_spacing:
    """
    # Calcuating the target shape based on the target spacing
    src_spacing = np.round(src_spacing, decimals = 3)
    target_shape = [int(src_imgs.shape[ix] * src_spacing[ix] / target_spacing[ix]) \
                    for ix in range(len(src_imgs.shape))]
    # Checking that none of the target shape is 0 or negative
    for i in range(len(target_shape)):
        try:
            assert target_shape[i] > 0
        except:
            raise AssertionError("AssertionError:", src_imgs.shape, src_spacing, target_spacing)

    img = src_imgs.astype(float)
    if is_label:
        # nearest neighbor interpolation for label
        resampled_img = resize(img, target_shape, order=0, clip=True, mode='edge').astype('float32')
    elif not is_label:
        # third-order spline interpolation for image
        resampled_img = resize(img, target_shape, order=3, clip=True, mode='edge').astype('float32')

    return resampled_img
