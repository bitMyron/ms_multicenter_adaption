import torch


def dsc_loss(pred, target, smooth=1e-5):
    """
    Loss function based on a single class DSC metric.
    :param pred: Predicted values. This tensor should have the shape:
     [batch_size, channels, x, y, z] or [batch_size, x, y, z]
     For the first shape, we assume a multiclass DSC, for the second a
     binary one.
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
    reduce_dims = tuple(range(len(dims) - 3, len(dims)))
    num = (2 * torch.sum(pred * target, dim=reduce_dims)) + smooth
    den = torch.sum(pred + target, dim=reduce_dims) + smooth
    dsc_k = num / den
    dsc = 1 - torch.mean(dsc_k)

    return torch.clamp(dsc, 0., 1.)


def class_entropy(prob):
    patch_means = torch.mean(torch.flatten(prob, start_dim=2), dim=-1)
    prob_means = torch.mean(patch_means, dim=0)

    return torch.sum(prob_means * torch.log(prob_means + 1e-2))
