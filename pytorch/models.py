import itertools
import time
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from data_manipulation.models import BaseModel, Autoencoder
from data_manipulation.models import Conv3dBlock, ResConv3dBlock
from data_manipulation.models import ResNConv3dBlock
from data_manipulation.models import DoubleConv3dBlock, Gated3dBlock
from data_manipulation.models import gumbel_softmax
from data_manipulation.utils import to_torch_var, time_to_string
from data_manipulation.criterions import dsc_loss, wasserstein1
from optimizer import MultipleOptimizer


def norm_f(n_f):
    return nn.GroupNorm(n_f // 8, n_f)


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


class Discriminator(BaseModel):
    def __init__(
            self, filters, n_images, kernel_size, wgan=False,
            block='conv', norm=None, activation=None,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
    ):
        # Init
        super().__init__()
        block_dict = {
            'conv': Conv3dBlock,
            'double': DoubleConv3dBlock,
            'res': ResConv3dBlock,
            'gate': Gated3dBlock,
        }
        self.device = device
        if norm is None:
            norm = nn.BatchNorm3d
        self.block_partial = partial(
            block_dict[block], kernel=kernel_size, norm=norm,
            activation=activation
        )
        self.block = partial(
            block_dict[block], norm=norm,
            activation=activation
        )

        self.conv = nn.ModuleList([
            self.block_partial(f_in, f_out)
            for f_in, f_out in zip([n_images] + filters[:-1], filters)
        ])
        self.out = nn.Linear(filters[-1], 1)

        if wgan:
            self.train_functions = [
                {
                    'name': 'wloss', 'weight': 1,
                    'f': lambda p, t: wasserstein1(
                        p[len(p) // 2:, ...], p[:len(p) // 2, ...]
                    )
                },
            ]
        else:
            self.train_functions = [
                {
                    'name': 'xentr', 'weight': 1,
                    'f': lambda p, t: F.binary_cross_entropy_with_logits(p, t)
                },
            ]

        self.val_functions = [
            {
                'name': 'xentr', 'weight': 0,
                'f': lambda p, t: F.binary_cross_entropy_with_logits(p, t)
            },
        ]

        disc_params = list(filter(
            lambda p: p.requires_grad, self.parameters()
        ))
        self.optimizer_alg = torch.optim.Adam(disc_params, lr=1e-4)

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
            self, filters, kernel_size, wgan=False,
            block='conv', norm=None, activation=None,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
    ):
        # Init
        super().__init__(
            filters, filters[0], kernel_size, wgan, block,
            norm, activation, device
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


class LesionsUNet(BaseModel):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_images=3,
            dropout=0,
            deep=False,
            verbose=0,
    ):
        super(LesionsUNet, self).__init__()
        self.init = False
        # Init values
        if conv_filters is None:
            conv_filters = [32, 64, 128, 256, 512]
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device
        self.dropout = dropout
        self.deep = deep

        # <Parameter setup>
        self.autoencoder = Autoencoder(
            conv_filters, device, n_images, block=ResConv3dBlock,
            pooling=True, norm=norm_f
        )
        self.autoencoder.dropout = dropout

        # Deep supervision branch.
        # This branch adapts the bottleneck filters to work with the final
        # segmentation block.
        self.deep_seg = nn.Sequential(
            nn.Conv3d(conv_filters[-1], conv_filters[0], 1),
            nn.ReLU(),
            norm_f(conv_filters[0]),
            nn.Conv3d(conv_filters[0], 1, 1)
        )
        self.deep_seg.to(device)

        # Final segmentation block.
        self.seg = nn.Sequential(
            nn.Conv3d(conv_filters[0], conv_filters[0], 1),
            nn.ReLU(),
            norm_f(conv_filters[0]),
            nn.Conv3d(conv_filters[0], 1, 1)
        )
        self.seg.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'dsc',
                'weight': 1,
                'f': lambda p, t: dsc_loss(p[0], t)
            },
            {
                'name': 'xentropy',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(
                    p[0],
                    t.type_as(p[0]).to(p[0].device),
                )
            }
        ]
        if self.deep:
            # DSC loss for the deep supervision branch (bottleneck).
            self.train_functions += [
                {
                    'name': 'dp dsc',
                    'weight': 1,
                    'f': lambda p, t: dsc_loss(
                        p[1],
                        F.max_pool3d(
                            t.type_as(p[1]),
                            2 ** len(self.autoencoder.down)
                        ).to(p[1].device)
                    )
                },
                # Focal loss for the deep supervision branch (bottleneck).
                {
                    'name': 'dp xentropy',
                    'weight': 1,
                    'f': lambda p, t: F.binary_cross_entropy(
                        p[1],
                        F.max_pool3d(
                            t.type_as(p[1]),
                            2 ** len(self.autoencoder.down)
                        ).to(p[1].device)
                    )
                },
            ]

        self.val_functions = [
            {
                'name': 'dsc',
                'weight': 1,
                'f': lambda p, t: dsc_loss(
                    (p[0] > 0.5).type_as(p[0]).to(p[0].device), t
                )
            },
        ]

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def forward(self, data):
        ae_out, features = self.autoencoder(data, keepfeat=True)
        multi_seg = torch.sigmoid(self.seg(ae_out))
        low_seg = torch.sigmoid(self.deep_seg(features[-1]))

        return multi_seg, low_seg

    def dropout_update(self):
        super().dropout_update()
        self.autoencoder.dropout = self.dropout

    def lesions(
            self,
            data,
            verbose=True
    ):
        # Init
        self.eval()

        data_tensor = to_torch_var(np.expand_dims(data, axis=0))

        with torch.no_grad():
            seg, _ = self(data_tensor)
            torch.cuda.empty_cache()

        if verbose > 1:
            print(
                '\033[K{:}Segmentation finished'.format(' '.join([''] * 12))
            )

        seg = np.squeeze(seg.cpu().numpy())

        return seg

    def patch_lesions(
            self, data, patch_size=32, batch_size=512, source=True,
            verbose=1
    ):
        # Init
        self.eval()

        seg = np.zeros(data.shape[1:])
        counts = np.zeros(data.shape[1:])
        # The following lines are just a complicated way of finding all
        # the possible combinations of patch indices.
        limits = tuple(
            list(range(0, lim - patch_size + 1))
            for lim in data.shape[1:]
        )
        limits_product = list(itertools.product(*limits))

        t_in = time.time()
        n_batches = int(np.ceil(len(limits_product) / batch_size))
        # The following code is just a normal test loop with all the
        # previously computed patches.
        for i, batch_i in enumerate(range(0, len(limits_product), batch_size)):
            patches = limits_product[batch_i:batch_i + batch_size]
            patch_list = [
                data[:, xi:xi + patch_size, xj:xj + patch_size, xk:xk + patch_size]
                for xi, xj, xk in patches
            ]
            data_tensor = to_torch_var(np.stack(patch_list, axis=0))
            with torch.no_grad():
                if source:
                    out = self(data_tensor)
                else:
                    _, _, out = self.target_pass(data_tensor)
                torch.cuda.empty_cache()
                for pi, (xi, xj, xk) in enumerate(patches):
                    seg_i = out[pi, ...].cpu().numpy()
                    xslice = slice(xi, xi + patch_size)
                    yslice = slice(xj, xj + patch_size)
                    zslice = slice(xk, xk + patch_size)
                    seg[xslice, yslice, zslice] += np.squeeze(seg_i)
                    counts[xslice, yslice, zslice] += 1

            if verbose > 0:
                percent = 20 * (i + 1) // n_batches
                progress_s = ''.join(['-'] * percent)
                remainder_s = ''.join([' '] * (20 - percent))

                t_out = time.time() - t_in
                time_s = time_to_string(t_out)

                t_eta = (t_out / (i + 1)) * (n_batches - (i + 1))
                eta_s = time_to_string(t_eta)
                batch_s = '\033[KBatch {:03d}/{:03d}  ' \
                          '[{:}>{:}] {:} ETA: {:}'.format(
                    i + 1, n_batches,
                    progress_s, remainder_s, time_s, eta_s
                )
                print(batch_s, end='\r', flush=True)
        if verbose > 1:
            print(
                '\033[K{:}Segmentation finished'.format(' '.join([''] * 12))
            )
        return seg / counts

class DomainAdapter(BaseModel):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_images=3,
            dropout=0,
            deep=False,
            verbose=0,
    ):
        # Init
        super().__init__()
        self.init = False
        self.device = device
        if conv_filters is None:
            conv_filters = [32, 64, 64, 128, 128]

        # Frozen pre-trained network
        self.segmenter = LesionsUNet(
            conv_filters, device, conv_filters[0] // 4, dropout, deep, verbose
        )

        # To latent space (encoding Z)
        self.z_encoder = ResNConv3dBlock(
            conv_filters[0], conv_filters[0] // 4, 3, 1, norm_f,
        )
        self.z_encoder.to(self.device)
        self.target_encoder = nn.Sequential(
            Autoencoder(
                conv_filters, device, n_images, block=ResConv3dBlock,
                pooling=True, norm=norm_f
            ),
            self.z_encoder
        )
        self.target_encoder.dropout = dropout
        self.target_encoder.to(self.device)
        self.source_encoder = nn.Sequential(
            Autoencoder(
                conv_filters, device, n_images, block=ResConv3dBlock,
                pooling=True, norm=norm_f
            ),
            self.z_encoder
        )
        self.source_encoder.dropout = dropout
        self.source_encoder.to(self.device)
        # From latent space (decoding source/target)
        self.z_decoder = ResNConv3dBlock(
                conv_filters[0] // 4, conv_filters[0], 3, 1, norm_f,
        )
        self.z_decoder.to(self.device)
        self.source_decoder = nn.Sequential(
            self.z_decoder,
            Autoencoder(
                conv_filters, device, conv_filters[0], block=ResConv3dBlock,
                pooling=True, norm=norm_f
            ),
            nn.Conv3d(conv_filters[0], n_images, 1)
        )
        self.source_decoder.to(self.device)
        self.target_decoder = nn.Sequential(
            self.z_decoder,
            Autoencoder(
                conv_filters, device, conv_filters[0], block=ResConv3dBlock,
                pooling=True, norm=norm_f
            ),
            nn.Conv3d(conv_filters[0], n_images, 1)
        )
        self.target_decoder.to(self.device)

        # Discriminator
        self.discriminator = nn.ModuleList(
            [
                Discriminator(
                    conv_filters, 1, 3, block='res', norm=norm_f
                ) for _ in range(n_images)
            ]
        )

        # Losses and metrics
        # These are exclusive to the GAN model. The losses for each model
        # (generator and discriminator) should be defined in each model
        # separately.
        self.tgt_functions = [
            {
                'name': 'Dkl',
                'weight': 1,
                'f': lambda f, z, t: torch.mean(z ** 2)
            },
            {
                'name': 'mse',
                'weight': 1,
                'f': lambda f, z, y: F.l1_loss(f, y)
            }
        ]
        self.dkl_functions = [
            {
                'name': 'Dkl',
                'weight': 1,
                'f': lambda f, z, t: torch.mean(
                    torch.cat([z[0], z[1]], dim=1) ** 2
                )
            },
            {
                'name': 'Dkl_s',
                'weight': .1,
                'f': lambda f, z, t: F.mse_loss(z[3], z[0].detach())
            },
            {
                'name': 'Dkl_t',
                'weight': .1,
                'f': lambda f, z, t: F.mse_loss(z[2], z[1].detach())
            },
        ]
        self.rec_functions = [
            {
                'name': 'mse_ss',
                'weight': 2,
                'f': lambda f, z, y: F.l1_loss(f[0], y[0])
            },
            {
                'name': 'cyc_s',
                'weight': 2,
                'f': lambda f, z, y: F.l1_loss(f[1], y[0])
            },
            {
                'name': 'mse_tt',
                'weight': 2,
                'f': lambda f, z, y: F.l1_loss(f[2], y[1])
            },
            {
                'name': 'cyc_t',
                'weight': 2,
                'f': lambda f, z, y: F.l1_loss(f[3], y[1])
            },
        ]
        self.disc_functions = [
            {
                'name': 'xentr', 'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy_with_logits(p, t)
            },
        ]
        self.vae_functions = self.dkl_functions + self.rec_functions
        self.val_functions = self.vae_functions + self.segmenter.val_functions
        self.adv_functions = [
            {
                'name': 'xentr', 'weight': .1,
                'f': lambda p, t: F.binary_cross_entropy_with_logits(p, t)
            },
        ]

        # Double optimizer (one for each loss)
        disc_params = list(filter(
            lambda p: p.requires_grad, self.discriminator.parameters()
        ))
        vae_params = list(filter(
            lambda p: p.requires_grad, self.target_encoder.parameters()
        )) + list(filter(
            lambda p: p.requires_grad, self.target_decoder.parameters()
        )) + list(filter(
            lambda p: p.requires_grad, self.source_encoder.parameters()
        )) + list(filter(
            lambda p: p.requires_grad, self.source_decoder.parameters()
        )) + list(filter(
            lambda p: p.requires_grad, self.segmenter.parameters()
        ))

        self.vae_optimizer_alg = torch.optim.Adam(vae_params, lr=1e-4)
        self.disc_optimizer_alg = torch.optim.Adam(disc_params, lr=1e-4)
        self.optimizer_alg = MultipleOptimizer(
            disc_params + vae_params,
            [self.disc_optimizer_alg, self.vae_optimizer_alg]
        )

    def forward(self, source):
        z_source = self.source_encoder(source)
        seg, _ = self.segmenter(z_source)

        return seg

    def vae_step(self, f_im, z, fy, y, seg, labels, train):
        if train:
            if self.training:
                self.disc_optimizer_alg.zero_grad()
                self.vae_optimizer_alg.zero_grad()
            if f_im is None:
                f_im = fy
                gen_losses = [
                    l_f['weight'] * l_f['f'](fy, z, y)
                    for l_f in self.tgt_functions
                ]
                seg_losses = [
                    l_f['weight'] * l_f['f'](seg, labels)
                    for l_f in self.segmenter.train_functions
                ]

            else:
                gen_losses = [
                    l_f['weight'] * l_f['f'](fy, z, y)
                    for l_f in self.vae_functions
                ]
                seg_losses = [
                    sum([
                        l_f['weight'] * l_f['f'](s, labels)
                        for s in seg
                    ])
                    for l_f in self.segmenter.train_functions
                ]

            gen_loss = sum(gen_losses)
            seg_loss = sum(seg_losses)
            batch_loss = gen_loss + seg_loss

            # Now we need to make a second pass through the net for the
            # generator loss. Since we want to have a good discriminator,
            # we won't train the generator for the first epochs.
            # First we get the probabilities of the fake images to
            # belong to the real class according to the discriminator.
            df = torch.cat(
                [
                    disc(yi)
                    for disc, yi in zip(
                        self.discriminator, torch.split(f_im, 1, dim=1)
                    )
                ], dim=1
            )
            lab_r = torch.ones_like(df).to(self.device)
            # We will use the "real" labels from before, since we want
            # to fool the discriminator.

            adv_losses = [
                l_f['weight'] * l_f['f'](df, lab_r)
                for l_f in self.adv_functions
            ]
            adv_loss = sum(adv_losses)
            if self.training:
                (gen_loss + seg_loss + adv_loss).backward()
                self.vae_optimizer_alg.step()

        # Validation losses
        else:
            if f_im is None:
                rec_losses = [
                    l_f['f'](fy, z, y)
                    for l_f in self.tgt_functions
                ]
                seg_losses = [
                    l_f['f'](seg, labels)
                    for l_f in self.segmenter.val_functions
                ]
                gen_losses = rec_losses + seg_losses
                batch_loss = sum(seg_losses) + rec_losses[1]
            else:
                dkl_losses = [
                    l_f['f'](fy, z, y)
                    for l_f in self.dkl_functions
                ]
                rec_losses = [
                    l_f['f'](fy, z, y)
                    for l_f in self.rec_functions
                ]
                seg_losses = [
                    sum([
                        l_f['f'](s, labels)
                        for s in seg
                    ])
                    for l_f in self.segmenter.val_functions
                ]
                gen_losses = dkl_losses + rec_losses + seg_losses
                batch_loss = sum(rec_losses + seg_losses)

        return gen_losses, batch_loss

    def disc_step(self, fy, y):
        # First we get the probabilities of the fake and real images to
        # belong to the real class according to the discriminator.
        dr = torch.cat(
            [
                disc(yi.detach())
                for disc, yi in zip(
                    self.discriminator, torch.split(y, 1, dim=1)
                )
            ], dim=1
        )
        df = torch.cat(
            [
                disc(yi.detach())
                for disc, yi in zip(
                    self.discriminator, torch.split(fy, 1, dim=1)
                )
            ], dim=1
        )
        # Then we generate the labels for real and fake images. Both
        # tensors should have the same shape (there are as many real
        # as fake images since we are using paired data).
        lab_r = torch.ones_like(dr).to(self.device)
        lab_f = torch.zeros_like(df).to(self.device)
        # Finally we create a unique tensor of both real and fake images.
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

    def target_pass(self, target):
        # Pass to latent space
        z_target = self.target_encoder(target)
        # Sampling
        sample_target = gaussian_sample(z_target, sample=self.training)
        # Fake data
        r_target = self.target_decoder(sample_target)
        # Segmentation in the latest space
        seg = self.segmenter(z_target)

        return z_target, r_target, seg

    def vae_pass(self, source, target):
        # Pass to latent space
        z_source = self.source_encoder(source)
        z_target = self.target_encoder(target)
        # Sampling
        sample_source = gaussian_sample(z_source, sample=self.training)
        sample_target = gaussian_sample(z_target, sample=self.training)

        # Fake data coming from source (recon loss)
        r_source = self.source_decoder(sample_source)
        f_target = self.target_decoder(sample_source)
        # Fake data coming from target (recon loss)
        f_source = self.source_decoder(sample_target)
        r_target = self.target_decoder(sample_target)

        source_out = z_source, sample_source, r_source, f_source
        target_out = z_target, sample_target, r_target, f_target

        return source_out, target_out

    def mini_batch_loop(
            self, data, train=True
    ):
        losses = list()
        mid_losses = list()
        n_batches = len(data)
        for batch_i, data_i in enumerate(data):
            try:
                s, (t, t_l) = data_i
                # We train the model and check the loss
                labels = t_l.unsqueeze(dim=1).type_as(t).to(self.device)
                source_cuda = s.to(self.device)
                target_cuda = t.to(self.device)

                vae_s, vae_t = self.vae_pass(
                    source_cuda, target_cuda
                )
                z_source, _, fss, f_source = vae_s
                z_target, sample_target, ftt, f_target = vae_t
                seg = self.segmenter(sample_target)

                vae_s, vae_t = self.vae_pass(
                    f_source, f_target
                )
                fz_source, sample_source, _, fsts = vae_s
                fz_target, _, _, ftst = vae_t
                fseg = self.segmenter(sample_source)

                # Fake images (adversarial loss)
                f_im = torch.cat([ftt, ftst, f_target], dim=0)

                # First we update the discriminator
                if train:
                    self.disc_step(f_im, target_cuda)
                # We continue with the generator.
                fy = [fss, fsts, ftt, ftst]
                y = [source_cuda, target_cuda]
                z = [z_source, z_target, fz_source, fz_target]
                segs = [seg, fseg]
                gen_losses, batch_loss = self.vae_step(
                    f_im, z, fy, y, segs, labels, train
                )

            except ValueError:
                t, t_l = data_i
                labels = t_l.type_as(t).to(self.device)
                target_cuda = t.to(self.device)
                z_target, r_target, seg = self.target_pass(target_cuda)

                # First we update the discriminator
                if train:
                    self.disc_step(r_target, target_cuda)
                # We continue with the generator.
                gen_losses, batch_loss = self.vae_step(
                    None, z_target, r_target, target_cuda, seg, labels, train
                )

            # Validation losses
            if not train:
                mid_losses.append([loss.tolist() for loss in gen_losses])

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
            return mean_loss, mean_losses, []

    def pre_fit(self, train_loader, val_loader, epochs=10, patience=5):
        self.val_functions = self.tgt_functions + self.segmenter.val_functions
        self.fit(train_loader, val_loader, epochs, patience)
        self.val_functions = self.vae_functions + self.segmenter.val_functions
        self.best_loss_tr = np.inf
        self.best_loss_val = np.inf

    def lesions(
            self,
            data,
            source=True,
            verbose=0
    ):
        # Init
        self.eval()

        data_tensor = to_torch_var(np.expand_dims(data, axis=0))

        with torch.no_grad():
            if source:
                seg = self(data_tensor)
            else:
                _, _, seg = self.target_pass(data_tensor)
            torch.cuda.empty_cache()

        if verbose > 1:
            print(
                '\033[K{:}Segmentation finished'.format(' '.join([''] * 12))
            )

        seg = np.squeeze(seg.cpu().numpy())

        return seg

    def patch_lesions(
            self, data, patch_size=32, batch_size=512, source=True,
            verbose=1
    ):
        # Init
        self.eval()

        seg = np.zeros(data.shape[1:])
        counts = np.zeros(data.shape[1:])
        # The following lines are just a complicated way of finding all
        # the possible combinations of patch indices.
        limits = tuple(
            list(range(0, lim - patch_size + 1))
            for lim in data.shape[1:]
        )
        limits_product = list(itertools.product(*limits))

        t_in = time.time()
        n_batches = int(np.ceil(len(limits_product) / batch_size))
        # The following code is just a normal test loop with all the
        # previously computed patches.
        for i, batch_i in enumerate(range(0, len(limits_product), batch_size)):
            patches = limits_product[batch_i:batch_i + batch_size]
            patch_list = [
                data[:, xi:xi+patch_size, xj:xj+patch_size, xk:xk+patch_size]
                for xi, xj, xk in patches
            ]
            data_tensor = to_torch_var(np.stack(patch_list, axis=0))
            with torch.no_grad():
                if source:
                    out = self(data_tensor)
                else:
                    _, _, out = self.target_pass(data_tensor)
                torch.cuda.empty_cache()
                for pi, (xi, xj, xk) in enumerate(patches):
                    seg_i = out[pi, ...].cpu().numpy()
                    xslice = slice(xi, xi + patch_size)
                    yslice = slice(xj, xj + patch_size)
                    zslice = slice(xk, xk + patch_size)
                    seg[xslice, yslice, zslice] += np.squeeze(seg_i)
                    counts[xslice, yslice, zslice] += 1

            if verbose > 0:
                percent = 20 * (i + 1) // n_batches
                progress_s = ''.join(['-'] * percent)
                remainder_s = ''.join([' '] * (20 - percent))

                t_out = time.time() - t_in
                time_s = time_to_string(t_out)

                t_eta = (t_out / (i + 1)) * (n_batches - (i + 1))
                eta_s = time_to_string(t_eta)
                batch_s = '\033[KBatch {:03d}/{:03d}  ' \
                          '[{:}>{:}] {:} ETA: {:}'.format(
                           i + 1, n_batches,
                           progress_s, remainder_s, time_s, eta_s
                )
                print(batch_s, end='\r', flush=True)
        if verbose > 1:
            print(
                '\033[K{:}Segmentation finished'.format(' '.join([''] * 12))
            )
        return seg / counts

    def transform(self, data, verbose=0):
        # Init
        self.eval()

        data_tensor = to_torch_var(np.expand_dims(data, axis=0))

        with torch.no_grad():
            z = self.source_encoder(data_tensor)
            fake = self.target_decoder(z)

        if verbose > 1:
            print(
                '\033[K{:}Transformation finished'.format(' '.join([''] * 12))
            )

        fake = np.squeeze(fake.cpu().numpy())

        return fake
