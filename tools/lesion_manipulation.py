import numpy as np
from functools import reduce
from scipy import ndimage as nd
from scipy.ndimage.morphology import binary_erosion as imerode
from skimage.measure import label as bwlabeln

def remove_small_regions(img_vol, min_size=3):
    """
        Function that removes blobs with a size smaller than a minimum from a mask
        volume.
        :param img_vol: Mask volume. It should be a numpy array of type bool.
        :param min_size: Minimum size for the blobs.
        :return: New mask without the small blobs.
    """
    # We will need to first define all blobs in the mask. That is 3D connected
    # regions.
    blobs, _ = nd.measurements.label(
        img_vol,
        nd.morphology.generate_binary_structure(3, 3)
    )
    labels = list(filter(bool, np.unique(blobs)))

    # Finally, for each region we will compute the area and filter those ones
    # that have a smaller area than the threshold.
    areas = [np.count_nonzero(np.equal(blobs, lab)) for lab in labels]
    nu_labels = [lab for lab, a in zip(labels, areas) if a > min_size]
    # We did the filtering in a way that we created a mask for each lesion,
    # so all we need to do is a big union operation of them all.
    nu_mask = reduce(
        lambda x, y: np.logical_or(x, y),
        [np.equal(blobs, lab) for lab in nu_labels]
    ) if nu_labels else np.zeros_like(img_vol)
    return nu_mask


def remove_boundary_regions(img_vol, mask_vol, boundary=2):
    """
    Function to remove regions that are inside the boundary of mask. The
    boundary's thickness can also be defined (default is 2 voxels).
    :param img_vol: Mask volume to process. It should be a numpy array of type
     bool.
    :param mask_vol: Mask from where the boundary area is defined.
    :param boundary: Boundary thickness (default = 2).
    :return: New mask without boundary regions.
    """
    # White matter lesions, should not be on the boundaries of a brain. That
    # region is where the cortex is located which can sometimes present
    # hyperintense artifacts. A way of removing some is to remove all lesions
    # that are on the boundaries (given a certain thickness).

    # First, we'll create a boundary mask by eroding the brain mask and taking
    # only the region of the mask that is not part of the erosion.
    new_mask = np.copy(img_vol)
    im_lab = bwlabeln(img_vol)
    mask_bin = mask_vol.astype(np.bool)
    inner = imerode(
        mask_bin,
        iterations=boundary
    )
    boundary = np.logical_and(mask_bin, np.logical_not(inner))

    # Then it's just a matter of removing any lesion that overlaps with that
    # boundary mask.
    boundary_labs = np.unique(im_lab[boundary])
    boundary_labs = boundary_labs[boundary_labs > 0]
    if len(boundary_labs) > 0:
        new_mask[np.isin(im_lab, boundary_labs)] = 0

    return new_mask
