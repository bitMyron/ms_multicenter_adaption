import time
import os
import re
import sys
import traceback
from subprocess import check_call
import statsmodels.api as sm
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
from nibabel import load as load_nii
from scipy.ndimage.morphology import binary_dilation as imdilate
from scipy.ndimage.morphology import binary_erosion as imerode
from scipy.stats import spearmanr, kendalltau
import torch


"""
Utility functions
"""


def color_codes():
    """
    Function that returns a custom dictionary with ASCII codes related to
    colors.
    :return: Custom dictionary with ASCII codes for terminal colors.
    """
    codes = {
        'nc': '\033[0m',
        'b': '\033[1m',
        'k': '\033[0m',
        '0.25': '\033[30m',
        'dgy': '\033[30m',
        'r': '\033[31m',
        'g': '\033[32m',
        'gc': '\033[32m;0m',
        'bg': '\033[32;1m',
        'y': '\033[33m',
        'c': '\033[36m',
        '0.75': '\033[37m',
        'lgy': '\033[37m',
        'clr': '\033[K',
    }
    return codes


def slicing(center_list, size):
    """

    :param center_list:
    :param size:
    :return:
    """
    half_size = tuple(map(lambda ps: ps/2, size))
    ranges = [
        [
            range(
                np.max([c_idx - p_idx, 0]), c_idx + (s_idx - p_idx)
            ) for c_idx, p_idx, s_idx in zip(center, half_size, size)
        ] for center in center_list
    ]
    slices = np.concatenate(
        map(
            lambda x: np.stack(list(product(*x)), axis=1),
            ranges
        ),
        axis=1
    )
    return slices


def find_file(name, dirname):
    """

    :param name:
    :param dirname:
    :return:
    """
    result = list(filter(
        lambda x: not os.path.isdir(x) and re.search(name, x),
        os.listdir(dirname)
    ))

    return os.path.join(dirname, result[0]) if result else None


def get_dirs(path):
    """
    Function to get the folder name of the patients given a path.
    :param path: Folder where the patients should be located.
    :return: List of patient names.
    """
    # All patients (full path)
    patient_paths = sorted(
        filter(
            lambda d: os.path.isdir(os.path.join(path, d)),
            os.listdir(path)
        )
    )
    # Patients used during training
    return patient_paths


def print_message(message):
    """
    Function to print a message with a custom specification
    :param message: Message to be printed.
    :return: None.
    """
    c = color_codes()
    dashes = ''.join(['-'] * (len(message) + 11))
    print(dashes)
    print(
        '%s[%s]%s %s' %
        (c['c'], time.strftime("%H:%M:%S", time.localtime()), c['nc'], message)
    )
    print(dashes)


def run_command(command, message=None, stdout=None, stderr=None):
    """
    Function to run and time a shell command using the call function from the
    subprocess module.
    :param command: Command that will be run. It has to comply with the call
    function specifications.
    :param message: Message to be printed before running the command. This is
    an optional parameter and by default its
    None.
    :param stdout: File where the stdout will be redirected. By default we use
    the system's stdout.
    :param stderr: File where the stderr will be redirected. By default we use
    the system's stderr.
    :return:
    """
    if message is not None:
        print_message(message)

    time_f(lambda: check_call(command), stdout=stdout, stderr=stderr)


def time_f(f, stdout=None, stderr=None):
    """
    Function to time another function.
    :param f: Function to be run. If the function has any parameters, it should
    be passed using the lambda keyword.
    :param stdout: File where the stdout will be redirected. By default we use
    the system's stdout.
    :param stderr: File where the stderr will be redirected. By default we use
    the system's stderr.
    :return: The result of running f.
    """
    # Init
    stdout_copy = sys.stdout
    if stdout is not None:
        sys.stdout = stdout

    start_t = time.time()
    try:
        ret = f()
    except Exception as e:
        ret = None
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print('{0}: {1}'.format(type(e).__name__, e), file=stderr)
        traceback.print_tb(exc_traceback, file=stderr)
    finally:
        if stdout is not None:
            sys.stdout = stdout_copy

    print(
        time.strftime(
            'Time elapsed = %H hours %M minutes %S seconds',
            time.gmtime(time.time() - start_t)
        )
    )
    return ret


def time_to_string(time_val):
    """
    Function to convert from a time number to a printable string that
     represents time in hours minutes and seconds.
    :param time_val: Time value in seconds (using functions from the time
     package)
    :return: String with a human format for time
    """

    if time_val < 60:
        time_s = '%ds' % time_val
    elif time_val < 3600:
        time_s = '%dm %ds' % (time_val // 60, time_val % 60)
    else:
        time_s = '%dh %dm %ds' % (
            time_val // 3600,
            (time_val % 3600) // 60,
            time_val % 60
        )
    return time_s


"""
Data related functions
"""


def save_bland_altman(x, y, suffix, path):
    f, ax = plt.subplots(1, figsize=(8, 6))
    plt.title('Bland Altman plot')
    sm.graphics.mean_diff_plot(x, y, ax=ax)

    plt.savefig(os.path.join(
        path, 'bland-altman_{:}.png'.format(suffix)
    ))
    plt.close()


def save_correlation(
        x, y, suffix, path, xlabel='Model', ylabel='Manual', verbose=0
):
    results = sm.OLS(y, sm.add_constant(x)).fit()
    spr_coef, spr_p = spearmanr(x, y)
    tau_coef, tau_p = kendalltau(x, y)

    if verbose > 1:
        print(results.summary())

    plt.title(
        'Correlation r-squared = {:5.3f} ({:5.3f}, {:5.3f})'.format(
            results.rsquared, results.pvalues[0], results.pvalues[1]
        )
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.scatter(x, y)
    x_plot = np.linspace(0, np.round(np.max(x)), 1000)
    plt.plot(x_plot, x_plot * results.params[1] + results.params[0], 'k')

    plt.savefig(os.path.join(
        path, 'correlation_r{:5.3f}.{:}.png'.format(results.rsquared, suffix)
    ))
    plt.close()

    z = sm.nonparametric.lowess(x, y)
    plt.title(
        'Spearman = {:5.3f} ({:5.3f})  / '
        'Kendall''s τ = {:3.5f} ({:5.3f})'.format(
            spr_coef, spr_p, tau_coef, tau_p
        )
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.scatter(x, y)
    plt.plot(z[:, 1], z[:, 0], 'k')
    plt.savefig(os.path.join(
        path, 'lowess_S{:5.3f}.t{:5.3f}.{:}.png'.format(
            spr_coef, tau_coef, suffix
        )
    ))

    plt.close()

    return results.rsquared, spr_coef, tau_coef


def save_scatter(
        x, y, suffix, path,
        xmin=None, xmax=None, ymin=None, ymax=None,
        xlabel='Model', ylabel='Manual'
):
    # Init
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)
    if ymin is None:
        ymin = np.min(y)
    if ymax is None:
        ymax = np.max(y)

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title('Scatterplot {:} vs {:}'.format(xlabel, ylabel))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.scatter(x, y)

    ax.set_xlim(xmin=xmin, xmax=xmax)
    ax.set_ylim(ymin=ymin, ymax=ymax)

    plt.savefig(os.path.join(
        path, 'scatter_{:}.png'.format(suffix)
    ))
    plt.close()


def get_mask(mask_name, dilate=0, dtype=np.uint8):
    """
    Function to load a mask image
    :param mask_name: Path to the mask image file
    :param dilate: Dilation radius
    :param dtype: Data type for the final mask
    :return:
    """
    # Lesion mask
    mask_image = (load_nii(mask_name).get_fdata() > 0.5).astype(dtype)
    if dilate > 0:
        mask_d = imdilate(
            mask_image,
            iterations=dilate
        )
        mask_e = imerode(
            mask_image,
            iterations=dilate
        )
        mask_image = np.logical_and(mask_d, np.logical_not(mask_e)).astype(dtype)

    return mask_image


def get_normalised_image(
        image_name, mask=None, dtype=np.float32, masked=False
):
    """
    Function to a load an image and normalised it (0 mean / 1 standard
     deviation)
    :param image_name: Path to the image to be noramlised
    :param mask: Mask defining the region of interest
    :param dtype: Data type for the final image
    :param masked: Whether to mask the image or not
    :return:
    """
    image = load_nii(image_name).get_fdata().astype(dtype)

    # If no mask is provided we use the image as a mask (all non-zero values)
    if mask is None:
        mask_bin = image.astype(np.bool)
    else:
        mask_bin = mask.astype(np.bool)

    # Parameter estimation using the mask provided
    image_mu = np.mean(image[mask_bin])
    image_sigma = np.std(image[mask_bin])
    norm_image = (image - image_mu) / image_sigma

    if masked:
        output = norm_image * mask_bin.astype(dtype)
    else:
        output = norm_image

    return output


def to_torch_var(
        np_array,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        requires_grad=False,
        dtype=torch.float32
):
    """
    Function to convert a numpy array into a torch tensor for a given device
    :param np_array: Original numpy array
    :param device: Device where the tensor will be loaded
    :param requires_grad: Whether it requires autograd or not
    :param dtype: Datatype for the tensor
    :return:
    """
    var = torch.tensor(
        np_array,
        requires_grad=requires_grad,
        device=device,
        dtype=dtype
    )
    return var
