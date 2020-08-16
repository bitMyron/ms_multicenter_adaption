from operator import mul
import torch
import itertools
from functools import reduce
import numpy as np
from torch import nn
from torch.nn import functional as F


class InterpolationLayer(nn.Module):
    """
    Misleading name that might get changed in the future.

    The goal of this layer is perform interpolation given a set of points,
    and their distance to the point we want to estimate.

    Since the SpatialTransformer layer needs to flatten the distances and
    values, a 1D vector is assumed. If that is not the case, using this layer
    on non-1D inputs will most likely cause some kind of CUDA exception or even
    a python one.
    """

    def __init__(
            self,
            n_points, n_features,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Parameters:
            :param n_points: Number of input channels.
            :param n_features: Number of features per point (multiple image
             contrasts might be used, like T1, T2, FLAIR, etc.).
            :param device: pytorch device that will store the layer.
        """
        super().__init__()
        self.device = device
        # The layer is basically a  1D convolution (again the distances and
        # values are assumed to be a flattened vector) with softmax. Why? We
        # want the weights to sum to one (softmax) but we also want the value
        # of the weights of each intensity value to be related to the other
        # points and their distances.
        self.w = nn.Sequential(
            nn.Conv1d(n_points * (n_features + 1), n_features, 1),
            nn.ReLU(),
            nn.BatchNorm1d(n_features),
            nn.Conv1d(n_features, n_points, 1)
        )

    def forward(self, distances, values):
        dist_flat = distances.view(distances.shape[:2] + (-1,))
        val_flat = distances.view(values.shape[:2] + (-1,))
        data = torch.cat([val_flat, dist_flat], dim=1)
        # So, we compute the weights...
        self.w.to(self.device)
        weights = F.softmax(self.w(data), dim=1)
        # And we then interpolate! (we'll leave the reshaping to the
        # transformation layer).
        return (val_flat * weights).view_as(values)


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer pytorch

    The Layer can handle dense transforms that are meant to give a 'shift'
    from the current position. Therefore, a dense transform gives displacements
    (not absolute locations) at each voxel,

    This code is a reimplementation of
    https://github.com/voxelmorph/voxelmorph/tree/master/ext/neuron in
    pytorch with some liberties taken. The goal is to adapt the code to
    some kind of hybrid method to both do dense registration and mask tracking.
    """

    def __init__(
            self,
            interp_method='linear',
            linear_norm=False,
            n_images=2,
            dim=3,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Parameters:
            :param interp_method: 'linear' or 'nearest'.
            :param linear_norm: How to do the interpolation when linear.
            :param device: pytorch device that will store the layer.
        """
        super().__init__()
        self.interp_method = interp_method
        self.device = device
        self.linear_norm = linear_norm
        if self.interp_method == 'learned':
            self.interp_layer = InterpolationLayer(2**dim, n_images)
        else:
            self.interp_layer = None

    def forward(self, vol, df, mesh=None, affine=None):
        """
        Transform (interpolation N-D volumes (features) given shifts at each
        location in pytorch. Essentially interpolates volume vol at locations
        determined by loc_shift.
        This is a spatial transform in the sense that at location [x] we now
        have the data from [x + shift].
        Parameters
            :param vol: Input volume to be warped and deformation field.
            :param df: Deformation field.
            :param mesh: Mesh defining the original positions of each voxel.
            :param affine: Affine transformation.

            :return new interpolated volumes in the same size as df
        """

        # parse shapes
        im_shape = vol.shape[2:] if mesh is None else mesh.shape[2:]
        final_shape = vol.shape[:2] + im_shape
        weights_shape = (vol.shape[0], 1) + im_shape
        nb_dims = len(im_shape)
        max_loc = [s - 1 for s in vol.shape[2:]]

        if mesh is None:
            linvec = [torch.arange(0, s) for s in im_shape]
            mesh = torch.stack([
                m_i.type(dtype=torch.float32)
                for m_i in torch.meshgrid(linvec)
            ]).unsqueeze(dim=0).to(self.device)

        # location should be mesh and delta
        if affine is not None:
            # Reshape all the affine matrices (we are dealing with patches)
            if len(affine.shape) == 3:
                affine = affine.view(
                    vol.shape[:2] + (nb_dims, nb_dims + 1)
                )
            norm_mesh = torch.cat(
                (
                    mesh.view(mesh.shape[:2] + (-1,)),
                    torch.ones(
                        (len(vol), 1, np.prod(im_shape))
                    ).to(self.device)
                ), dim=1
            ).unsqueeze(dim=1)
            aff_mesh = torch.matmul(affine, norm_mesh)
            mesh = aff_mesh.view_as(mesh)

        loc = [
            torch.clamp(mesh[:, d, ...] + df[:, d, ...], 0, m)
            for d, m in zip(range(nb_dims), max_loc)
        ]

        # pre ind2sub setup
        d_size = np.cumprod((1,) + vol.shape[-1:2:-1])[::-1]

        # interpolate
        if self.interp_method == 'nearest':
            # clip values
            roundloc = [
                torch.clamp(l, 0, m).type(torch.long) for l, m in zip(
                    [torch.round(l) for l in loc], max_loc
                )
            ]

            # get values
            loc_list = [s * l for s, l in zip(roundloc, d_size)]
            idx = torch.sum(torch.stack(loc_list, dim=0), dim=0)
            interp_vol_flat = torch.stack(
                [torch.take(vol_i, idx_i) for idx_i, vol_i in zip(idx, vol)],
                dim=0
            )
            interp_vol = torch.reshape(interp_vol_flat, final_shape)

        else:
            # clip values
            loc0 = list(map(torch.floor, loc))
            loc0lst = [
                torch.clamp(l, 0, m) for l, m in zip(loc0, max_loc)
            ]

            # get other end of point cube
            loc1 = [
                torch.clamp(l + 1, 0, m) for l, m in zip(loc0, max_loc)
            ]
            locs = [
                [f.type(torch.long) for f in loc0lst],
                [f.type(torch.long) for f in loc1]
            ]

            # Compute the difference between the upper value and the original
            # value. Differences are basically 1 - (pt - floor(pt)).
            #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) =
            #             = 1 - (pt - floor(pt))
            diff_loc1 = [l1 - l for l1, l in zip(loc1, loc)]
            diff_loc1 = [torch.clamp(l, 0, 1) for l in diff_loc1]
            diff_loc0 = [1 - diff for diff in diff_loc1]
            weights_loc = [diff_loc1, diff_loc0]  # note reverse ordering
            # since weights are inverse of diff.

            # go through all the cube corners, indexed by a ND binary vector
            # e.g. [0, 0] means this "first" corner in a 2-D "cube"
            cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
            norm_factor = nb_dims * len(cube_pts) / 2.0

            def get_point_value(point):
                subs = [locs[cd][i] for i, cd in enumerate(point)]
                loc_list_p = [
                    s.long() * np.long(l) for s, l in zip(subs, d_size)
                ]
                idx_p = torch.sum(
                    torch.stack(loc_list_p, dim=0), dim=0
                )

                vol_val_flat = torch.stack(
                    [torch.stack(
                        [torch.take(vol_ij, idx_i) for vol_ij in vol_i],
                        dim=0
                    ) for vol_i, idx_i in zip(vol, idx_p)],
                    dim=0
                )

                vol_val = torch.reshape(vol_val_flat, final_shape)
                # get the weight of this cube_pt based on the distance
                # if c[d] is 0 --> weight = 1 - (pt - floor(pt)) = diff_loc1
                # if c[d] is 1 --> weight = pt - floor(pt) = diff_loc0
                wts_lst = [weights_loc[cd][i] for i, cd in enumerate(point)]
                if self.linear_norm:
                    wt = sum(wts_lst) / norm_factor
                else:
                    wt = reduce(mul, wts_lst)
                wt = torch.reshape(wt, weights_shape)
                if self.interp_layer is not None:
                    return wt, vol_val
                else:
                    return wt * vol_val

            if self.interp_layer is not None:
                dist_values = [
                    (d, v)
                    for d, v in map(get_point_value, cube_pts)
                ]
                dist, values = zip(*dist_values)
                values = self.interp_layer(
                    torch.stack(dist, dim=1),
                    torch.stack(values, dim=1)
                )
            else:
                values = tuple(map(get_point_value, cube_pts))
                values = torch.stack(values, dim=1)

            interp_vol = torch.sum(values, dim=1)

        return interp_vol


class SmoothingLayer(nn.Module):
    """
    N-D Smoothing layer pytorch

    This layer defines a trainable Gaussian smoothing kernel. While
    convolutional layers might learn such a kernel, the idea is to impose
    smoothing to the activations of the previous layer. The only parameter
    is the sigma value for the Gaussian kernel of a fixed size.
    """
    def __init__(
            self,
            length=5,
            init_sigma=0.5,
            trainable=False
    ):
        super().__init__()
        if trainable:
            self.sigma = nn.Parameter(
                torch.tensor(
                    init_sigma,
                    dtype=torch.float,
                    requires_grad=True
                )
            )
        else:
            self.sigma = torch.tensor(
                    init_sigma,
                    dtype=torch.float
                )
        self.length = length

    def forward(self, x):
        dims = len(x.shape) - 2
        assert dims <= 3, 'Too many dimensions for convolution'

        kernel_shape = (self.length,) * dims
        lims = map(lambda s: (s - 1.) / 2, kernel_shape)
        grid = map(
            lambda g: torch.tensor(g, dtype=torch.float, device=x.device),
            np.ogrid[tuple(map(lambda l: slice(-l, l + 1), lims))]
        )
        sigma_square = self.sigma * self.sigma
        k = torch.exp(
            -sum(map(lambda g: g*g, grid)) / (2. * sigma_square.to(x.device))
        )
        sumk = torch.sum(k)
        if sumk.tolist() > 0:
            k = k / sumk

        kernel = torch.reshape(k, (1,) * 2 + kernel_shape).to(x.device)
        final_kernel = kernel.repeat((x.shape[1],) * 2 + (1,) * dims)
        conv_f = [F.conv1d, F.conv2d, F.conv3d]
        padding = self.length / 2

        smoothed_x = conv_f[dims - 1](x, final_kernel, padding=padding)

        return smoothed_x


class Sine3DLayer(nn.Module):
    """
    Sine activation based on:
    Vincent Sitzmann, Julien NP Martel, Alexander W Bergman, David B Lindell,
    Gordon Wetzstein. "Implicit Neural Representations with Periodic
    Activation Functions".
    https://arxiv.org/abs/2006.09661
    In order to adapt to 3D CNNs, we changed the linear layer with a
    convolutional one of 1 x 1 x 1. That should keep the properties of the
    original layer, while reducing considerably the number of parameters and
    necessary RAM.
    """

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Conv3d(
            in_features, out_features, kernel_size=1, bias=bias
        )

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.in_features,
                    1 / self.in_features
                )
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
