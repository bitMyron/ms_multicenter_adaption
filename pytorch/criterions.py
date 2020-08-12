import torch
from torch.nn import functional as F


"""
Binary losses
"""


def dsc_loss(pred, target, smooth=0.1):
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
    num = (2 * torch.sum(pred * target, dim=reduce_dims))
    den = torch.sum(pred + target, dim=reduce_dims) + smooth
    dsc_k = num / den
    dsc = 1 - torch.mean(dsc_k)

    return torch.clamp(dsc, 0., 1.)


def multidsc_loss(pred, target, smooth=1, averaged=True):
    """
    Loss function based on a multi-class DSC metric.
    :param pred: Predicted values. This tensor should have the shape:
     [batch_size, n_classes, data_shape]
    :param target: Ground truth values. This tensor can have multiple shapes:
     - [batch_size, n_classes, data_shape]: This is the expected output since
       it matches with the predicted tensor.
     - [batch_size, data_shape]: In this case, the tensor is labeled with
       values ranging from 0 to n_classes. We need to convert it to
       categorical.
    :param smooth: Parameter used to smooth the DSC when there are no positive
     samples.
    :param averaged: Parameter to decide whether to return the average DSC or
     a tensor with the different class DSC values.
    :return: The mean DSC for the batch
    """
    # Init
    dims = pred.shape
    n_classes = dims[1]
    # Dimension checks. We want everything to be similar or at least
    # translatable.
    if target.shape != pred.shape:
        assert torch.max(target) <= n_classes, 'Wrong number of classes for GT'
        target = torch.cat([target == i for i in range(n_classes)], dim=1)
        target = target.type_as(pred)

    # We'll do the sums / means across the 3D axes to have a value per label
    # and patch.
    # DSC = 2 * | pred *union* target | / (| pred | + | target |)
    red_dim = tuple(range(2, len(dims)))
    num = (2 * torch.sum(pred * target, dim=red_dim))
    den = torch.sum(pred + target, dim=red_dim) + smooth
    dsc_k = num / den
    if averaged:
        dsc = 1 - torch.mean(dsc_k)
    else:
        dsc = 1 - torch.mean(dsc_k, dim=0)

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
