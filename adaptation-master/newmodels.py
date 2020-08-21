from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from data_manipulation.models import BaseModel, Autoencoder
from data_manipulation.models import BaseConv3dBlock, ResConv3dBlock
from data_manipulation.models import Conv3dBlock
from data_manipulation.models import gumbel_softmax
from data_manipulation.utils import to_torch_var
from criterions import dsc_loss, class_entropy
from optimizer import MultipleOptimizer


def norm_f(n_f):
    return nn.GroupNorm(n_f // 8, n_f)


def weighted_softmax(x, weight, dim):
    maxes = torch.max(x, dim, keepdim=True)[0]
    x_exp = torch.exp(weight * (x - maxes))
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)

    return x_exp / x_exp_sum


def gaussian_sample(mu, sigma=None, sample=True):
    if sigma is None:
        noise = torch.randn_like(mu).to(mu.device)
    else:
        noise = sigma * torch.randn_like(mu).to(mu.device)

    if sample:
        return mu + noise
    else:
        return mu


def categorical_sample(logits, sample=True):
    if sample:
        return gumbel_softmax(logits, dim=1)
    else:
        return torch.softmax(logits, dim=1)


def binary_dsc(pred, target):
    dims = pred.shape
    reduce_dims = tuple(range(len(dims) - 3, len(dims)))
    pred_bin = pred > 0.5
    target_bin = target.type(torch.bool)
    bck = torch.sum(target_bin, dim=reduce_dims) == 0

    pred_bin[bck] = torch.logical_not(pred_bin[bck])
    target_bin[bck] = torch.logical_not(target_bin[bck])

    n_pred = torch.sum(pred_bin, dim=reduce_dims).double()
    n_target = torch.sum(target_bin, dim=reduce_dims).double()
    tp = torch.sum(target_bin & pred_bin, dim=reduce_dims).double()

    return torch.mean(2 * tp / (n_pred + n_target))


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        # F.instance_norm(
        #     x, running_mean, running_var, self.weight, self.bias,
        #     True, self.momentum, self.eps
        # )
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class Discriminator(BaseModel):
    def __init__(
            self, filters, kernel_size, device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
    ):
        # Init
        super().__init__()
        self.device = device
        self.block_partial = partial(
            ResConv3dBlock, kernel=kernel_size, norm=norm_f,
            activation=nn.ReLU
        )
        self.block = partial(
            ResConv3dBlock, norm=norm_f, activation=nn.ReLU
        )

        self.conv = nn.ModuleList([
            self.block_partial(f_in, f_out)
            for f_in, f_out in zip([1] + filters[:-1], filters)
        ])
        self.out = nn.Linear(filters[-1], 1)

    def forward(self, data):
        for i, c in enumerate(self.conv):
            c.to(self.device)
            data = F.max_pool3d(c(data), 2)
        data = torch.flatten(data, start_dim=2)
        data, _ = torch.max(data, dim=-1)
        self.out.to(self.device)
        return self.out(data)


class FeatureDiscriminator(Discriminator):
    def __init__(
            self, filters, kernel_size, device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
    ):
        # Init
        super().__init__(
            filters, kernel_size, device
        )
        f_inputs = [filters[0]] + [f * 2 for f in filters[1:-1]]
        self.conv = nn.ModuleList([
            self.block_partial(f_in, f_out)
            for f_in, f_out in zip(f_inputs, filters[1:])
        ])
        self.extra = self.block(filters[-1] * 2, filters[-1], kernel=1)
        self.extra.to(self.device)
        self.out = nn.Linear(filters[-1], 1)
        self.out.to(self.device)

    def forward(self, data):
        data_i = data[0]
        for i, c in enumerate(self.conv):
            c.to(self.device)
            data_i = F.max_pool3d(c(data_i), 2)
            if (i + 1) < len(data):
                data_i = torch.cat([data[i + 1], data_i], dim=1)
        data = self.extra(data_i)
        data = torch.flatten(data, start_dim=2)
        data, _ = torch.max(data, dim=-1)
        return self.out(data)


class LinearDecoder(BaseModel):
    def __init__(
            self, filters, output, device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
    ):
        # Init
        super().__init__()
        self.device = device
        self.linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(f_in, f_out),
                nn.ReLU(),
                nn.GroupNorm(f_out // 8, f_out)
            )
            for f_in, f_out in zip(filters[:-1], filters[1:])
        ])
        for linear in self.linears:
            linear.to(device)
        self.out = nn.Linear(filters[-1], output)
        self.out.to(device)

    def forward(self, data):
        for linear in self.linears:
            data = linear(data)

        return self.out(data)


class DomainAdapter(BaseModel):
    def __init__(
            self,
            filters=None,
            n_images=2,
            n_parts=4,
            style_shape=8,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
    ):
        # Init
        super().__init__()
        self.prefit = False
        self.init = False
        self.device = device
        if filters is None:
            filters = [32, 64, 128, 256, 512]

        # Shape network (n parts - first class should be lesions)
        self.shape_encoder = nn.Sequential(
            Autoencoder(
                # filters, device, n_images, block=ResConv3dBlock,
                filters, device, n_images, block=Conv3dBlock,
                pooling=True, norm=norm_f
                # pooling=True, norm=AdaptiveInstanceNorm
            ),
            nn.Conv3d(filters[0], n_parts, 1)
        )
        self.shape_encoder.to(device)

        # Style network
        self.style_encoder = nn.ModuleList([
            ResConv3dBlock(f_in, f_out, norm=norm_f)
            for f_in, f_out in zip([1] + filters[:-1], filters)
        ] + [nn.Linear(filters[-1], style_shape)])
        for e in self.style_encoder:
            e.to(self.device)

        self.image_decoder = nn.Sequential(
            Autoencoder(
                # filters, device, n_parts, block=ResConv3dBlock,
                filters, device, n_parts, block=Conv3dBlock,
                pooling=True, norm=norm_f
                # pooling=True, norm=AdaptiveInstanceNorm
            ),
            ResConv3dBlock(filters[0], filters[0], norm=AdaptiveInstanceNorm),
            ResConv3dBlock(filters[0], filters[0], norm=AdaptiveInstanceNorm),
            ResConv3dBlock(filters[0], filters[0], norm=AdaptiveInstanceNorm),
            ResConv3dBlock(filters[0], filters[0], norm=AdaptiveInstanceNorm),
            nn.Conv3d(filters[0], 1, 1)
        )
        self.image_decoder.to(device)
        # We'll keep all the Adaptive Normalisation layers handy to modify
        # them at run time as needed.
        # I am not entirely comfortable with breaking access rules by directly
        # accesing the variables of the Autoencoder and residual blocks (mostly
        # because I might change blocks), but it works for now.
        self.decoder_norms = []
        for mi in self.image_decoder.modules():
            if isinstance(mi, BaseConv3dBlock):
                for mj in mi.modules():
                    if isinstance(mj, AdaptiveInstanceNorm):
                        self.decoder_norms.append(mj)

        # style_filters = [style_shape] + [
        #     2 * norm.num_features for norm in self.decoder_norms
        # ]
        # self.norm_decoder = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(f_in, f_out),
        #         nn.ReLU(),
        #         nn.GroupNorm(f_out // 8, f_out)
        #     )
        #     for f_in, f_out in zip(style_filters[:-1], style_filters[1:])
        # ])
        norm_params = sum([
            2 * norm.num_features for norm in self.decoder_norms
        ])
        self.norm_decoder = LinearDecoder(
            [style_shape, 128, 256, 512], norm_params
        )
        self.norm_decoder.to(device)

        # Discriminator
        self.discriminator = Discriminator(filters, 3)

        # Losses and metrics
        # These are exclusive to the GAN model. The losses for each model
        # (generator and discriminator) should be defined in each model
        # separately.
        self.seg_functions = [
            {
                'name': 'dscl',
                'weight': 1,
                'f': lambda p, t: dsc_loss(p, t)
            },
            {
                'name': 'xentr',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(
                    p, t.type_as(p).to(p.device),
                )
            }
        ]
        self.dkl_functions = [
            {
                'name': 'Dkl',
                'weight': 1,
                'f': lambda z: torch.mean(z ** 2)
            },
        ]
        self.style_functions = [
            {
                'name': 'style',
                'weight': 1,
                'f': lambda f, y: F.mse_loss(f, y)
            },
        ]
        self.cycle_functions = [
            {
                'name': 'shape_xentr',
                'weight': 1,
                'f': lambda s, ms: F.binary_cross_entropy(s, ms)
            },
            {
                'name': 'shape_dsc',
                'weight': 1,
                'f': lambda s, ms: dsc_loss(s, ms)
            },
        ]
        self.shape_functions = [
            # {
            #     'name': 'shape',
            #     'weight': 1,
            #     'f': lambda s: torch.mean(
            #         - torch.sum(s * torch.log(s + 1e-8), dim=1)
            #     )
            # },
            {
                'name': 'shape',
                'weight': 1,
                'f': lambda s: class_entropy(s)
            },
            {
                'name': 'lesion',
                'weight': 1,
                'f': lambda s: torch.sum(s[0, ...] > 0.5) / torch.sum(s)
            },
        ]
        self.rec_functions = [
            {
                'name': 'mse',
                'weight': 1,
                'f': lambda f, y: F.mse_loss(f, y)
            },
            {
                'name': 'l1',
                'weight': 1,
                'f': lambda f, y: F.l1_loss(f, y)
            },
        ]
        self.disc_functions = [
            {
                'name': 'disc', 'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy_with_logits(p, t)
            },
        ]
        self.val_functions = self.rec_functions + self.seg_functions
        self.adv_functions = [
            {
                'name': 'adversarial', 'weight': .1,
                'f': lambda p, t: F.binary_cross_entropy_with_logits(p, t)
            },
        ]
        self.acc_functions = [
            {
                'name': 'dsc',
                'f': lambda p, t: binary_dsc(p, t)
            },
        ]

        # Double optimizer (one for each loss)
        disc_params = list(filter(
            lambda p: p.requires_grad, self.discriminator.parameters()
        ))
        da_params = list(filter(
            lambda p: p.requires_grad, self.shape_encoder.parameters()
        )) + list(filter(
            lambda p: p.requires_grad, self.style_encoder.parameters()
        )) + list(filter(
            lambda p: p.requires_grad, self.image_decoder.parameters()
        )) + list(filter(
            lambda p: p.requires_grad, self.norm_decoder.parameters()
        ))

        # self.da_optimizer_alg = torch.optim.SGD(da_params, lr=1e-4)
        self.da_optimizer_alg = torch.optim.Adam(da_params, lr=1e-4)
        # self.disc_optimizer_alg = torch.optim.SGD(disc_params, lr=1e-4)
        self.disc_optimizer_alg = torch.optim.Adam(disc_params, lr=1e-4)
        self.optimizer_alg = MultipleOptimizer(
            disc_params + da_params,
            [self.disc_optimizer_alg, self.da_optimizer_alg]
        )

    def forward(self, x):
        return self.shape(x)

    def update_adain(self, z_style):
        # for d, norm in zip(self.norm_decoder, self.decoder_norms):
        #     params = d(params)
        #     mean = params[:, :norm.num_features]
        #     std = params[:, norm.num_features:]
        #     norm.bias = mean.contiguous().view(-1)
        #     norm.weight = std.contiguous().view(-1)
        params = self.norm_decoder(z_style)
        param_start = 0
        for norm in self.decoder_norms:
            mean_start = param_start
            mean_end = mean_start + norm.num_features
            std_start = mean_end
            std_end = std_start + norm.num_features
            param_start = std_end
            mean = params[:, mean_start:mean_end]
            std = params[:, std_start:std_end]
            norm.bias = mean.contiguous().view(-1)
            norm.weight = std.contiguous().view(-1)

    def style(self, x):
        for e in self.style_encoder[:-1]:
            x = F.max_pool3d(e(x), 2)
        # _, x = self.shape_encoder[0].encode(x)
        x = torch.flatten(x, start_dim=2)
        x, _ = torch.max(x, dim=-1)

        return self.style_encoder[-1](x)

    def shape(self, x, mask=None):

        # return torch.softmax(self.shape_encoder(x), dim=1)
        if mask is not None:
            shape_logit = mask * self.shape_encoder(x)
            shape = mask * weighted_softmax(shape_logit, 10, dim=1)
        else:
            shape_logit = self.shape_encoder(x)
            shape = weighted_softmax(shape_logit, 10, dim=1)
        return shape

    def reconstruct(self, shape, style, mask, freeze=False):
        self.update_adain(style)
        if freeze:
            with torch.no_grad():
                x = self.image_decoder(shape)
        else:
            x = self.image_decoder(shape)
        return x * mask

    def mini_batch_loop(
            self, data, train=True
    ):
        accs = list()
        losses = list()
        mid_losses = list()
        n_batches = len(data)
        for batch_i, data_i in enumerate(data):
            (s, s_m), (t, t_m, t_l) = data_i
            # We move everything to the GPU first
            # Source: Unlabeled data we want to adapt
            # Target: Labeled data we want to learn from
            labels = t_l.type_as(t).to(self.device)
            source_mask = s_m.type_as(s).to(self.device)
            target_mask = t_m.type_as(t).to(self.device)
            source = s.to(self.device)
            target = t.to(self.device)
            source_split = torch.split(source, 1, dim=1)
            target_split = torch.split(target, 1, dim=1)
            reals = source_split + target_split

            # < Direct pass >
            # First, we need to split the images. Each image should have its own
            # style, while the shape should be the same for a given patch.
            source_styles = [self.style(x) for x in source_split]
            target_styles = [self.style(x) for x in target_split]
            # source_samples = [
            #     gaussian_sample(x, sample=self.training) for x in source_styles
            # ]
            # target_samples = [
            #     gaussian_sample(x, sample=self.training) for x in target_styles
            # ]
            all_styles = torch.stack(source_styles + target_styles)

            # Different modalities might give different segmentations.
            # However, we know the images are coregistered and should
            # contain the same "shapes". For example, some lesions might
            # not be visible in a T1w image, while they are in FLAIR.
            # In order to enforce the same shape, we average them for a
            # given patch. This is an implicit way of doing that. To
            # explicitly force similarity between segmentations we use
            # the loss functions.
            # Hopefully, that also makes the model "modality"-agnostic.
            source_shape = self.shape(source, source_mask)
            target_shape = self.shape(target, target_mask)
            # msource_shapes = self.mean_shape(source_shapes)
            # msource_shapes = torch.mean(source_shapes, dim=0)
            # mtarget_shapes = self.mean_shape(source_shapes)
            # mtarget_shapes = torch.mean(target_shapes, dim=0)
            # Lesions will be the first class
            # predlabels = mtarget_shapes[:, 0, ...]
            predlabels = target_shape[:, 0, ...]

            # fsource = [
            #     self.reconstruct(msource_shapes, style)
            #     for style in source_styles
            # ]
            # ftarget = [
            #     self.reconstruct(mtarget_shapes, style)
            #     for style in target_styles
            # ]
            fsource = [
                # self.reconstruct(source_shape, style, source_mask, freeze=True)
                self.reconstruct(source_shape, style, source_mask)
                for style in source_styles
                # for style in source_samples
            ]
            ftarget = [
                self.reconstruct(target_shape, style, target_mask)
                for style in target_styles
                # for style in target_samples
            ]
            fakes = fsource + ftarget

            # < Inverse pass >
            # This is the "data augmentation" pass, where we take the learned
            # shapes and styles and shift them around.
            news_styles = [
                style[torch.randperm(len(style)), ...]
                for style in target_styles
                # for style in target_samples
            ]
            # newsource = [
            #     self.reconstruct(msource_shapes.detach(), style)
            #     for style in news_styles
            # ]
            newsource = [
                # self.reconstruct(
                #     source_shape, style, source_mask, freeze=True
                # )
                self.reconstruct(source_shape, style, source_mask)
                for style in news_styles
            ]
            newt_styles = [
                style[torch.randperm(len(style)), ...]
                for style in source_styles
                # for style in source_samples
            ]
            # newtarget = [
            #     self.reconstruct(mtarget_shapes.detach(), style)
            #     for style in newt_styles
            # ]
            newtarget = [
                self.reconstruct(
                    target_shape, style, target_mask
                )
                for style in newt_styles
            ]

            # After shuffling styles around, we try to recover the styles
            # and shapes back.
            # fs_styles = [self.style(x) for x in newsource]
            ft_styles = [self.style(x) for x in newtarget]

            # fs_shapes = torch.stack([
            #     self.shape(x, style)
            #     for x, style in zip(newsource, fs_styles)
            # ])
            # ft_shapes = torch.stack([
            #     self.shape(x, style)
            #     for x, style in zip(newtarget, ft_styles)
            # ])
            fsource_shape = self.shape(torch.cat(newsource, dim=1))
            ftarget_shape = self.shape(torch.cat(newtarget, dim=1))
            # fms_shapes = torch.mean(fs_shapes, dim=0)
            # fms_shapes = self.mean_shape(fs_shapes)
            # fmt_shapes = torch.mean(ft_shapes, dim=0)
            # fmt_shapes = self.mean_shape(ft_shapes)
            # flabels = fmt_shapes[:, 0, ...]
            flabels = ftarget_shape[:, 0, ...]

            if train:
                # < Discriminator step >
                # First we get the probabilities of the fake and real images to
                # belong to the real class according to the discriminator.
                dr = torch.cat([
                    # self.discriminator(yi.detach())
                    # for yi in torch.split(target_shape, 1, dim=1)
                    # self.discriminator(yi.detach()) for yi in reals
                    self.discriminator(yi.detach()) for yi in target_split
                    # self.discriminator(yi.detach()) for yi in source_split
                ], dim=0)
                df = torch.cat([
                    # self.discriminator(yi.detach())
                    # for yi in torch.split(source_shape, 1, dim=1)
                    # self.discriminator(yi.detach()) for yi in fakes
                    self.discriminator(yi.detach()) for yi in newsource + ftarget
                    # self.discriminator(yi.detach()) for yi in newtarget + fsource
                ], dim=0)
                # Then we generate the labels for real and fake images. Both
                # tensors should have the same shape (there are as many real
                # as fake images since we are using paired data).
                lab_r = torch.ones_like(dr).to(self.device)
                lab_f = torch.zeros_like(df).to(self.device)
                # Finally we create a unique tensor of both real and fake
                # images.
                d = torch.cat((dr, df), dim=0)
                dy = torch.cat((lab_r, lab_f), dim=0)

                #  We start with the discriminator.
                if self.training:
                    self.disc_optimizer_alg.zero_grad()
                disc_losses = [
                    l_f['weight'] * l_f['f'](d, dy)
                    for l_f in self.disc_functions
                ]
                disc_loss = sum(disc_losses)
                if self.training:
                    disc_loss.backward()
                    self.disc_optimizer_alg.step()

                # < Generator step >
                if self.training:
                    self.disc_optimizer_alg.zero_grad()
                    self.da_optimizer_alg.zero_grad()

                gen_losses = [
                    sum([
                        l_f['weight'] * l_f['f'](fy, y)
                        for fy, y in zip(fakes, reals)
                    ])
                    for l_f in self.rec_functions

                ]
                style_losses = [
                    # l_f['weight'] * l_f['f'](
                    #     torch.stack(fs_styles, dim=1),
                    #     torch.stack(news_styles, dim=1).detach()
                    # ) +
                    l_f['weight'] * l_f['f'](
                        torch.stack(ft_styles, dim=1),
                        torch.stack(newt_styles, dim=1).detach()
                    )
                    for l_f in self.style_functions
                ]
                shape_losses = [
                    # l_f['weight'] * l_f['f'](
                    #     torch.cat([source_shapes, fs_shapes], dim=1)
                    # ) +
                    # l_f['weight'] * l_f['f'](
                    #     torch.cat([target_shapes, ft_shapes], dim=1)
                    # )
                    # l_f['weight'] * l_f['f'](
                    #     fs_shapes, source_shapes.detach()
                    # ) +
                    # l_f['weight'] * l_f['f'](
                    #     ft_shapes, target_shapes.detach()
                    # )
                    l_f['weight'] * l_f['f'](
                        fsource_shape, source_shape.detach()
                    ) +
                    l_f['weight'] * l_f['f'](
                        ftarget_shape, target_shape.detach()
                    )
                    for l_f in self.cycle_functions
                ] + [
                    l_f['weight'] * l_f['f'](
                        ftarget_shape
                    ) + l_f['weight'] * l_f['f'](
                        target_shape
                    ) + l_f['weight'] * l_f['f'](
                        source_shape
                    )
                    for l_f in self.shape_functions
                ]
                dkl_losses = [
                    l_f['weight'] * l_f['f'](all_styles)
                    for l_f in self.dkl_functions
                ]
                seg_losses = [
                    l_f['weight'] * l_f['f'](predlabels, labels) +
                    l_f['weight'] * l_f['f'](flabels, labels)
                    for l_f in self.seg_functions
                ]

                gen_loss = sum(gen_losses)
                style_loss = sum(style_losses)
                shape_loss = sum(shape_losses)
                seg_loss = sum(seg_losses)
                dkl_loss = sum(dkl_losses)
                # batch_loss = gen_loss + shape_loss + seg_loss
                batch_loss = gen_loss + style_loss + shape_loss + seg_loss
                # batch_loss = gen_loss + style_loss + seg_loss
                # batch_loss = gen_loss + style_loss + shape_loss + seg_loss + dkl_loss

                # Now we need to make a second pass through the net for the
                # generator loss. Since we want to have a good discriminator,
                # we won't train the generator for the first epochs.
                # First we get the probabilities of the fake images to
                # belong to the real class according to the discriminator.
                df = torch.cat([
                    # self.discriminator(yi)
                    # for yi in torch.split(source_shape, 1, dim=1)
                    # self.discriminator(yi) for yi in fakes
                    self.discriminator(yi) for yi in newsource + ftarget
                    # self.discriminator(yi) for yi in newtarget + fsource
                ], dim=0)
                lab_r = torch.ones_like(df).to(self.device)
                # We will use the "real" labels from before, since we want
                # to fool the discriminator.

                adv_losses = [
                    l_f['weight'] * l_f['f'](df, lab_r)
                    for l_f in self.adv_functions
                ]
                adv_loss = sum(adv_losses)
                if self.training:
                    (batch_loss + adv_loss).backward()
                    self.da_optimizer_alg.step()

            # Validation losses
            else:
                gen_losses = [
                    sum([
                        l_f['f'](fy, y)
                        # for fy, y in zip(fakes, reals)
                        for fy, y in zip(fsource, source_split)
                    ])
                    for l_f in self.rec_functions

                ]
                seg_losses = [
                    # l_f['f'](predlabels, labels)
                    l_f['f'](flabels, labels)
                    for l_f in self.seg_functions
                ]
                batch_loss = sum(gen_losses + seg_losses)
                mid_losses.append([
                    loss.tolist() for loss in gen_losses + seg_losses
                ])
                batch_accs = [
                    # l_f['f'](predlabels, labels)
                    l_f['f'](flabels, labels)
                    for l_f in self.acc_functions
                ]
                accs.append([a.tolist() for a in batch_accs])

            loss_value = batch_loss.tolist()
            losses.append(loss_value)

            # Curriculum dropout / Adaptive dropout
            # Here we could modify dropout to be updated for each batch.
            # (1 - rho) * exp(- gamma * t) + rho, gamma > 0

            self.print_progress(
                batch_i, n_batches, loss_value, np.mean(losses)
            )

        mean_loss = np.mean(losses)
        if train:
            return mean_loss
        else:
            mean_losses = np.mean(list(zip(*mid_losses)), axis=1)
            np_accs = np.array(list(zip(*accs)))
            mean_accs = np.mean(np_accs, axis=1) if np_accs.size > 0 else []
            return mean_loss, mean_losses, mean_accs

    def lesions(
            self,
            data,
            verbose=0
    ):
        # Init
        self.eval()

        data_tensor = to_torch_var(np.expand_dims(data, axis=0))

        with torch.no_grad():
            seg = self(data_tensor)
            torch.cuda.empty_cache()

        if verbose > 1:
            print(
                '\033[K{:}Segmentation finished'.format(' '.join([''] * 12))
            )

        seg = np.squeeze(seg.cpu().numpy())

        return seg

    def transform(self, data, mask, verbose=0):
        # Init
        self.eval()

        data_tensor = to_torch_var(np.expand_dims(data, axis=0))
        mask_tensor = to_torch_var(np.expand_dims(mask, axis=0))

        with torch.no_grad():
            data_split = torch.split(data_tensor, 1, dim=1)
            shape = self.shape(data_tensor)
            styles = [self.style(x) for x in data_split]
            fake = torch.cat([
                self.reconstruct(shape, style, mask_tensor)
                for style in styles
            ], dim=1)

        if verbose > 1:
            print(
                '\033[K{:}Transformation finished'.format(' '.join([''] * 12))
            )

        fake = np.squeeze(fake.cpu().numpy())

        return fake
