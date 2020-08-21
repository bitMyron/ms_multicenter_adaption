from scipy.stats import entropy
import numpy as np
from numpy import histogramdd, histogram2d, histogram
from itertools import chain, combinations


def multivariate_mutual_information(images, bins=256):
    # Images should be a list of numpy arrays.

    # In order to compute the mutual information, we will need to compute the
    # entropies for each subset of the images as described in (Bell 2003)
    # Thus, we create an iterable with all the possible combinations.
    nimages = len(images)
    im_idx = range(0, nimages)
    power_range = range(1, nimages+1)
    image_comb_iter = [combinations(im_idx, power) for power in power_range]
    image_comb = chain.from_iterable(image_comb_iter)

    # For convenience, is also better if we vectorise the images and stack
    # them in a single numpy array. This simplifies the process of computing
    # the histogram (joint in multidimensional cases)
    images_vec = [image.flatten() for image in images]
    np_images = np.stack(images_vec, axis=1)
    histograms = [histogramdd(np_images[:, c], bins=bins) for c in image_comb]
    histograms_norm = [(h.flatten() / h.astype(np.float32).sum(), len(h.shape)) for h, e in histograms]
    histograms_non0 = [(h[np.nonzero(h)], s) for h, s in histograms_norm]
    informations = [-((-1) ** ((nimages - s) % 2)) * entropy(h) for h, s in histograms_non0]
    return np.stack(informations).sum()


def entropies(images, bins=256):
    # Images should be a list of numpy arrays.
    histograms = [histogram(image.flatten(), bins=bins) for image in images]
    return [entropy(h[np.nonzero(h)] / h.sum()) for h, s in histograms]


def joint_entropy(images, bins=256):
    # Images should be a list of numpy arrays.
    np_images = np.stack([image.flatten() for image in images], axis=1)
    h, s = histogramdd(np_images, bins=bins)
    return entropy(h[np.nonzero(h)] / h.sum())


def normalized_mutual_information(var_x, var_y, bins=256):
    # Init
    np_x = np.array(var_x)
    np_y = np.array(var_y)
    # We compute the 1d entropies first ...
    hist_x, _ = histogram(np_x.flatten(), bins)
    p_x = hist_x / hist_x.astype(np.float32).sum()
    entr_x = entropy(p_x)

    hist_y, _ = histogram(np_y.flatten(), bins)
    p_y = hist_y / hist_y.astype(np.float32).sum()
    entr_y = entropy(p_y)

    # ... and then the joint one
    hist_xy, _, _ = histogram2d(np_x.flatten(), np_y.flatten(), bins)
    p_xy = hist_xy.flatten() / hist_xy.astype(np.float32).sum()
    entr_xy = entropy(p_xy)

    # This are all the values we need to compute the normalized mutual
    # information.To normalize it, we will use the metric version
    # (H(X) + H(Y) - H(X, Y)) / (H(X))
    mi = (entr_x + entr_y - entr_xy) / (entr_x)

    return mi


def bidirectional_mahalanobis(var_x, var_y):
    # We compute both distribution's Gaussian parameters
    mu_x = np.mean(np.array(var_x))
    sigma_x = np.std(np.array(var_x))

    mu_y = np.mean(np.array(var_y))
    sigma_y = np.std(np.array(var_y))

    mu_diff = mu_x - mu_y

    mahal = (sigma_x + sigma_y) * mu_diff * mu_diff

    return mahal / (sigma_x * sigma_y) if (sigma_x * sigma_y) != 0 else mahal