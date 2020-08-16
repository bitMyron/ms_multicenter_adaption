import torch
from torch.nn import functional as F


"""
Tensor functions
"""


def gradient(tensor):
    """
        Function to compute the gradient of a multidimensional tensor. We
         assume that the first two dimensions specify the number of samples and
         channels.
        :param tensor: Input tensor
        :return: The mean gradient tensor
    """

    # Init
    tensor_dims = len(tensor.shape)
    data_dims = tensor_dims - 2

    # Since we want this function to be generic, we need a trick to define
    # the gradient on each dimension.
    all_slices = (slice(0, None),) * (tensor_dims - 1)
    first = slice(0, -2)
    last = slice(2, None)
    slices = [
        (
            all_slices[:i + 2] + (first,) + all_slices[i + 2:],
            all_slices[:i + 2] + (last,) + all_slices[i + 2:],
        )
        for i in range(data_dims)
    ]

    # Remember that gradients moved the image 0.5 pixels while also reducing
    # 1 voxel per dimension. To deal with that we are technically interpolating
    # the gradient in between these positions. These is the equivalent of
    # computing the gradient between voxels separated one space. 1D ex:
    # [a, b, c, d] -> gradient0.5 = [a - b, b - c, c - d]
    # gradient1 = 0.5 * [(a - b) + (b - c), (b - c) + (c - d)] =
    # = 0.5 * [a - c, b - d] ~ [a - c, b - d]
    no_pad = (0, 0)
    pad = (1, 1)
    paddings = [
        no_pad * i + pad + no_pad * (data_dims - i - 1)
        for i in range(data_dims)[::-1]
    ]

    gradients = [
        0.5 * F.pad(tensor[si] - tensor[sf], p)
        for p, (si, sf) in zip(paddings, slices)
    ]

    return torch.cat(gradients, dim=1)


def wasserstein1(fake, real):
    return real.mean() - fake.mean()


"""
Binary losses
"""


def dsc_loss(pred, target, smooth=1e-5):
    """
    Loss function based on a single class DSC metric.
    :param pred: Predicted values. This tensor should have the shape:
     [batch_size, data_shape]
    :param target: Ground truth values. This tensor can have multiple shapes:
     - [batch_size, data_shape]: This is the expected output since
       it matches with the predicted tensor.
     - [batch_size, data_shape]: In this case, the tensor is labeled with
       values ranging from 0 to n_classes. We need to convert it to
       categorical.
    :param smooth: Parameter used to smooth the DSC when there are no positive
     samples.
    :return: The mean DSC for the batch
    """
    # Init
    dims = pred.shape
    # Dimension checks. We want everything to be the same. This a class vs
    # class comparison.
    assert target.shape == pred.shape,\
        'Sizes between predicted and target do not match'
    target = target.type_as(pred)

    # We'll do the sums / means across the 3D axes to have a value per patch.
    # There is only a class here.
    # DSC = 2 * | pred *union* target | / (| pred | + | target |)
    reduce_dims = tuple(range(1, len(dims)))
    num = (2 * torch.sum(pred * target, dim=reduce_dims)) + smooth
    den = torch.sum(pred + target, dim=reduce_dims) + smooth
    dsc_k = num / den
    dsc = 1 - torch.mean(dsc_k)

    return torch.clamp(dsc, 0., 1.)


def focal_loss(pred, target, alpha=0.2, gamma=2.0):
    """
    Function to compute the focal loss based on:
    Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár. "Focal
    Loss for Dense Object Detection".
    https://arxiv.org/abs/1708.02002
    :param pred: Predicted values. The shape of the tensor should be:
     [n_batches, data_shape]
    :param target: Ground truth values. The shape of the tensor should be:
     [n_batches, data_shape]
    :param alpha: Weighting parameter to avoid class imbalance (default 0.2).
    :param gamma: Focusing parameter (default 2.0).
    :return: Focal loss value.
    """
    pt = target.type_as(pred) * pred + (1 - target).type_as(pred) * (1 - pred)
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    focal = alpha * (1 - pt).pow(gamma) * bce
    return focal.mean()


def flip_loss(
        pred, target, q, q_factor=0.5,
        base=F.binary_cross_entropy, regularizer=None
):
    """
    Flip loss function based on:
    Richard McKinley, Michael Rebsamen, Raphael Meier, Mauricio Reyes,
    Christian Rummel and Roland Wiest. "Few-shot brain segmentation from weakly
    labeled data with deep heteroscedastic multi-task network".
    https://arxiv.org/abs/1904.02436
    :param pred: Predicted values. The shape of the tensor should be related
     to the base function.
    :param target: Ground truth values. The shape of the tensor should be
     related to the base function.
    :param q: Uncertainty output from the network. The shape of the tensor
     should be related to the base function.
    :param q_factor: Factor to normalise the value of q.
    :param base: Base function for the flip loss.
    :param regularizer: Regularization function.
    :return: The flip loss given a base loss function
    """
    assert regularizer is None or type(regularizer) is int,\
        'Wrong type for the norm type'
    norm_q = q * q_factor
    flip_0 = (pred >= 0.5).type_as(pred) * (1 - target)
    flip_1 = (pred < 0.5).type_as(pred) * target
    z = flip_0 + flip_1
    q_target = (1 - target) * norm_q + target * (1 - norm_q)
    loss_seg = base(pred, q_target.type_as(pred).detach())
    loss_uncertainty = base(norm_q, z.detach())
    if regularizer is not None:
        loss_regularizer = torch.norm(norm_q, p=regularizer)
        final_loss = loss_seg + loss_uncertainty + loss_regularizer
    else:
        final_loss = loss_seg + loss_uncertainty

    return final_loss


def uncertainty_loss(
        pred, target, q, q_factor=0.5, base=F.binary_cross_entropy
):
    """
    Uncertainty loss function based on:
    Richard McKinley, Michael Rebsamen, Raphael Meier, Mauricio Reyes,
    Christian Rummel and Roland Wiest. "Few-shot brain segmentation from weakly
    labeled data with deep heteroscedastic multi-task network".
    https://arxiv.org/abs/1904.02436
    :param pred: Predicted values. The shape of the tensor should be related
     to the base function.
    :param target: Ground truth values. The shape of the tensor should be
     related to the base function.
    :param q: Uncertainty output from the network. The shape of the tensor
     should be related to the base function.
    :param q_factor: Factor to normalise the value of q.
    :param base: Base function for the flip loss.
    :return: The flip loss given a base loss function
    """
    norm_q = q * q_factor
    flip_0 = (pred >= 0.5).type_as(pred) * (1 - target)
    flip_1 = (pred < 0.5).type_as(pred) * target
    z = flip_0 + flip_1

    return base(norm_q, z.detach())


def lesion_size_loss(pred, target):
    """
    Loss function that compares the number of mask voxels in two masks
    before and after deformation/registration.
    :param pred: Predicted mask (after moving).
    :param target: Reference mask (the mask before moving).
    :return: The L2 norm of the voxel difference per patch.
    """
    # Init
    dims = pred.shape
    reduce_dims = tuple(range(1, len(dims)))
    mask = target.type_as(pred).to(pred.device)

    # Number of voxels in the mask for each patch.
    n_mov_voxels = torch.sum(pred, dim=reduce_dims)
    n_voxels = torch.sum(mask, dim=reduce_dims)
    valid = n_voxels > 1e-5
    ratio = n_mov_voxels[valid] / n_voxels[valid]
    tensor_1 = torch.tensor(1., device=ratio.device)
    tensor_0 = torch.tensor(0., device=ratio.device)
    ratio_loss = torch.abs(tensor_1 - ratio) if len(ratio) > 0 else tensor_0
    return torch.mean(ratio_loss)


def lesion_ppv(pred, target):
    """
    Loss function that computes the positive predictive value between two
    masks.
    :param pred: Predicted mask.
    :param target: Ground truth values.
    :return:
    """
    # Init
    dims = pred.shape
    reduce_dims = tuple(range(1, len(dims)))
    mask = target.type_as(pred).to(pred.device)

    tp = torch.sum(pred * mask, dim=reduce_dims)
    positive = torch.sum(mask, dim=reduce_dims)
    valid = positive > 1e-5
    ppv = tp[valid] / positive[valid]
    tensor_1 = torch.tensor(1., device=ppv.device)
    tensor_0 = torch.tensor(0., device=ppv.device)
    ppv_loss = tensor_1 - ppv if len(ppv) > 0 else tensor_0
    return torch.mean(ppv_loss)


"""
> Regression losses
"""


def normalise_var(var):
    red_dim = tuple(range(2, len(var.shape)))
    mean = torch.mean(var.detach(), dim=red_dim, keepdim=True)
    std = torch.std(var.detach(), dim=red_dim, keepdim=True)
    if (std < 1e-5).any():
        norm = (var[std > 1e-5] - mean[std > 1e-5]) / std[std > 1e-5]
    else:
        norm = (var - mean) / std

    return norm


def normalised_xcor(var_x, var_y):
    """
        Function that computes the normalised cross correlation between two
         tensors.
        :param var_x: First tensor.
        :param var_y: Second tensor.
        :return: A tensor with the normalised cross correlation
    """
    # Init
    var_y = var_y.to(var_x.device)

    # Computation
    var_x_norm = normalise_var(var_x)
    var_y_norm = normalise_var(var_y)

    xcor = [
        F.conv1d(
            torch.unsqueeze(x_i, dim=0).view(1, len(x_i), -1),
            torch.unsqueeze(y_i, dim=0).view(1, len(y_i), -1)
        )
        for x_i, y_i in zip(var_x_norm, var_y_norm)
    ]

    n_elem = var_x.numel() / len(var_x)
    xcor = torch.mean(torch.abs(torch.cat(xcor))) / n_elem

    if torch.isnan(xcor):
        xcor = torch.tensor(1., device=xcor.device)

    return xcor


def normalised_xcor_loss(var_x, var_y):
    """
        Loss function based on the normalised cross correlation between two
         tensors. Since we are using gradient descent, the final value is
         1 - the normalised cross correlation.
        :param var_x: First tensor.
        :param var_y: Second tensor.
        :return: A tensor with the loss
    """
    return 1. - normalised_xcor(var_x, var_y)
