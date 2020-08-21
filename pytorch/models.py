import time
import itertools
from functools import partial
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .utils import time_to_string, to_torch_var
from .criterions import dsc_loss, multidsc_loss


class BaseModel(nn.Module):
    """"
    This is the baseline model to be used for any of my networks. The idea
    of this model is to create a basic framework that works similarly to
    keras, but flexible enough.
    For that reason, I have "embedded" the typical pytorch main loop into a
    fit function and I have defined some intermediate functions and callbacks
    to alter the main loop. By itself, this model can train any "normal"
    network with different losses and scores for training and validation.
    It can be easily extended to create adversarial networks (which I have done
    in other repositories) and probably to other more complex problems.
    The network also includes some print functions to check the current status.
    """
    def __init__(self):
        """
        Main init. By default some parameters are set, but they should be
        redefined on networks inheriting that model.
        """
        super().__init__()
        # Init values
        self.init = True
        self.optimizer_alg = None
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.dropout = 0
        self.final_dropout = 0
        self.ann_rate = 0
        self.best_loss_tr = np.inf
        self.best_loss_val = np.inf
        self.best_state = None
        self.best_opt = None
        self.train_functions = [
            {'name': 'train', 'weight': 1, 'f': None},
        ]
        self.val_functions = [
            {'name': 'val', 'weight': 1, 'f': None},
        ]
        self.acc_functions = {}
        self.acc = None

    def forward(self, *inputs):
        """

        :param inputs: Inputs to the forward function. We are passing the
         contents by reference, so if there are more than one input, they
         will be separated.
        :return: Nothing. This has to be reimplemented for any class.
        """
        return None

    def mini_batch_loop(
            self, data, train=True
    ):
        """
        This is the main loop. It's "generic" enough to account for multiple
        types of data (target and input) and it differentiates between
        training and testing. While inherently all networks have a training
        state to check, here the difference is applied to the kind of data
        being used (is it the validation data or the training data?). Why am
        I doing this? Because there might be different metrics for each type
        of data. There is also the fact that for training, I really don't care
        about the values of the losses, since I only want to see how the global
        value updates, while I want both (the losses and the global one) for
        validation.
        :param data: Dataloader for the network.
        :param train: Whether to use the training dataloader or the validation
         one.
        :return:
        """
        losses = list()
        mid_losses = list()
        accs = list()
        n_batches = len(data)
        for batch_i, (x, y) in enumerate(data):
            # In case we are training the the gradient to zero.
            if self.training:
                self.optimizer_alg.zero_grad()

            # First, we do a forward pass through the network.
            torch.cuda.synchronize()
            if isinstance(x, list):
                x_cuda = tuple(x_i.to(self.device) for x_i in x)
                pred_labels = self(*x_cuda)
            else:
                pred_labels = self(x.to(self.device))

            # After that, we can compute the relevant losses.
            if train:
                # Training losses (applied to the training data)
                batch_losses = [
                    l_f['weight'] * l_f['f'](pred_labels, y)
                    for l_f in self.train_functions
                ]
                batch_loss = sum(batch_losses)
                if self.training:
                    batch_loss.backward()
                    self.optimizer_alg.step()

            else:
                # Validation losses (applied to the validation data)
                batch_losses = [
                    l_f['f'](pred_labels, y)
                    for l_f in self.val_functions
                ]
                batch_loss = sum([
                    l_f['weight'] * l
                    for l_f, l in zip(self.val_functions, batch_losses)
                ])
                mid_losses.append([l.tolist() for l in batch_losses])
                batch_accs = [
                    l_f['f'](pred_labels, y)
                    for l_f in self.acc_functions
                ]
                accs.append([a.tolist() for a in batch_accs])

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # It's important to compute the global loss in both cases.
            loss_value = batch_loss.tolist()
            losses.append(loss_value)

            # Curriculum dropout / Adaptive dropout
            # Here we could modify dropout to be updated for each batch.
            # (1 - rho) * exp(- gamma * t) + rho, gamma > 0

            self.print_progress(
                batch_i, n_batches, loss_value, np.mean(losses)
            )

        # Mean loss of the global loss (we don't need the loss for each batch).
        mean_loss = np.mean(losses)

        if train:
            return mean_loss
        else:
            # If using the validation data, we actually need to compute the
            # mean of each different loss.
            mean_losses = np.mean(list(zip(*mid_losses)), axis=1)
            np_accs = np.array(list(zip(*accs)))
            mean_accs = np.mean(np_accs, axis=1) if np_accs.size > 0 else []
            return mean_loss, mean_losses, mean_accs

    def fit(
            self,
            train_loader,
            val_loader,
            epochs=50,
            patience=5,
            verbose=True
    ):
        # Init
        best_e = 0
        no_improv_e = 0
        l_names = ['train', ' val '] + [
            '{:^6s}'.format(l_f['name']) for l_f in self.val_functions
        ]
        acc_names = [
            '{:^6s}'.format(a_f['name']) for a_f in self.acc_functions
        ]
        l_bars = '--|--'.join(
            ['-' * 5] * 2 +
            ['-' * 6] * (len(l_names[2:]) + len(acc_names)) +
            ['-' * 3]
        )
        l_hdr = '  |  '.join(l_names + acc_names + ['drp'])
        # Since we haven't trained the network yet, we'll assume that the
        # initial values are the best ones.
        self.best_state = deepcopy(self.state_dict())
        self.best_opt = deepcopy(self.optimizer_alg.state_dict())
        t_start = time.time()

        # Initial losses
        # This might seem like an unnecessary step (and it actually often is)
        # since it wastes some time checking the output with the initial
        # weights. However, it's good to check that the network doesn't get
        # worse than a random one (which can happen sometimes).
        if self.init:
            # We are looking for the output, without training, so no need to
            # use grad.
            with torch.no_grad():
                self.t_val = time.time()
                # We set the network to eval, for the same reason.
                self.eval()
                # Training losses.
                self.best_loss_tr = self.mini_batch_loop(train_loader)
                # Validation losses.
                self.best_loss_val, best_loss, best_acc = self.mini_batch_loop(
                    val_loader, False
                )
                # Doing this also helps setting an initial best loss for all
                # the necessary losses.
                if verbose:
                    # This is just the print for each epoch, but including the
                    # header.
                    # Mid losses check
                    epoch_s = '\033[32mInit     \033[0m'
                    tr_loss_s = '\033[32m{:7.4f}\033[0m'.format(
                        self.best_loss_tr
                    )
                    loss_s = '\033[32m{:7.4f}\033[0m'.format(
                        self.best_loss_val
                    )
                    losses_s = [
                        '\033[36m{:8.4f}\033[0m'.format(l) for l in best_loss
                    ]
                    # Acc check
                    acc_s = [
                        '\033[36m{:8.4f}\033[0m'.format(a) for a in best_acc
                    ]
                    t_out = time.time() - self.t_val
                    t_s = time_to_string(t_out)

                    drop_s = '{:5.3f}'.format(self.dropout)

                    print('\033[K', end='')
                    whites = ' '.join([''] * 12)
                    print('{:}Epoch num |  {:}  |'.format(whites, l_hdr))
                    print('{:}----------|--{:}--|'.format(whites, l_bars))
                    final_s = whites + ' | '.join(
                        [epoch_s, tr_loss_s, loss_s] +
                        losses_s + acc_s + [drop_s, t_s]
                    )
                    print(final_s)
        else:
            # If we don't initialise the losses, we'll just take the maximum
            # ones (inf, -inf) and print just the header.
            print('\033[K', end='')
            whites = ' '.join([''] * 12)
            print('{:}Epoch num |  {:}  |'.format(whites, l_hdr))
            print('{:}----------|--{:}--|'.format(whites, l_bars))
            best_loss = [np.inf] * len(self.val_functions)
            best_acc = [-np.inf] * len(self.acc_functions)

        for self.epoch in range(epochs):
            # Main epoch loop
            self.t_train = time.time()
            self.train()
            # First we train and check if there has been an improvement.
            loss_tr = self.mini_batch_loop(train_loader)
            improvement_tr = self.best_loss_tr > loss_tr
            if improvement_tr:
                best_loss_tr = loss_tr
                tr_loss_s = '\033[32m{:7.4f}\033[0m'.format(loss_tr)
            else:
                tr_loss_s = '{:7.4f}'.format(loss_tr)

            # Then we validate and check all the losses
            with torch.no_grad():
                self.t_val = time.time()
                self.eval()
                loss_val, mid_losses, acc = self.mini_batch_loop(
                    val_loader, False
                )

            # Mid losses check
            losses_s = [
                '\033[36m{:8.4f}\033[0m'.format(l) if bl > l
                else '{:8.4f}'.format(l) for bl, l in zip(
                    best_loss, mid_losses
                )
            ]
            best_loss = [
                l if bl > l else bl for bl, l in zip(
                    best_loss, mid_losses
                )
            ]
            # Acc check
            acc_s = [
                '\033[36m{:8.4f}\033[0m'.format(a) if ba < a
                else '{:8.4f}'.format(a) for ba, a in zip(
                    best_acc, acc
                )
            ]
            best_acc = [
                a if ba < a else ba for ba, a in zip(
                    best_acc, acc
                )
            ]

            # Patience check
            # We check the patience to stop early if the network is not
            # improving. Otherwise we are wasting resources and time.
            improvement_val = self.best_loss_val > loss_val
            loss_s = '{:7.4f}'.format(loss_val)
            if improvement_val:
                self.best_loss_val = loss_val
                epoch_s = '\033[32mEpoch {:03d}\033[0m'.format(self.epoch)
                loss_s = '\033[32m{:}\033[0m'.format(loss_s)
                best_e = self.epoch
                self.best_state = deepcopy(self.state_dict())
                self.best_opt = deepcopy(self.optimizer_alg.state_dict())
                no_improv_e = 0
            else:
                epoch_s = 'Epoch {:03d}'.format(self.epoch)
                no_improv_e += 1

            t_out = time.time() - self.t_train
            t_s = time_to_string(t_out)

            drop_s = '{:5.3f}'.format(self.dropout)
            self.dropout_update()

            if verbose:
                print('\033[K', end='')
                whites = ' '.join([''] * 12)
                final_s = whites + ' | '.join(
                    [epoch_s, tr_loss_s, loss_s] +
                    losses_s + acc_s + [drop_s, t_s]
                )
                print(final_s)

            if no_improv_e == int(patience / (1 - self.dropout)):
                break

            self.epoch_update(epochs)

        self.epoch = best_e
        self.load_state_dict(self.best_state)
        t_end = time.time() - t_start
        t_end_s = time_to_string(t_end)
        if verbose:
            print(
                    'Training finished in {:} epochs ({:}) '
                    'with minimum loss = {:f} (epoch {:d})'.format(
                        self.epoch + 1, t_end_s, self.best_loss_val, best_e
                    )
            )

    def epoch_update(self, epochs):
        """
        Callback function to update something on the model after the epoch
        is finished. To be reimplemented if necessary.
        :param epochs: Maximum number of epochs
        :return: Nothing.
        """
        return None

    def dropout_update(self):
        """
        Callback function to update the dropout. To be reimplemented
        if necessary. However, the main method already has some basic
        scheduling
        :param epochs: Maximum number of epochs
        :return: Nothing.
        """
        if self.final_dropout <= self.dropout:
            self.dropout = max(
                self.final_dropout, self.dropout - self.ann_rate
            )

    def print_progress(self, batch_i, n_batches, b_loss, mean_loss):
        """
        Function to print the progress of a batch. It takes into account
        whether we are training or validating and uses different colors to
        show that. It's based on Keras arrow progress bar, but it only shows
        the current (and current mean) training loss, elapsed time and ETA.
        :param batch_i: Current batch number.
        :param n_batches: Total number of batches.
        :param b_loss: Current loss.
        :param mean_loss: Current mean loss.
        :return: None.
        """
        init_c = '\033[0m' if self.training else '\033[38;5;238m'
        whites = ' '.join([''] * 12)
        percent = 20 * (batch_i + 1) // n_batches
        progress_s = ''.join(['-'] * percent)
        remainder_s = ''.join([' '] * (20 - percent))
        loss_name = 'train_loss' if self.training else 'val_loss'

        if self.training:
            t_out = time.time() - self.t_train
        else:
            t_out = time.time() - self.t_val
        time_s = time_to_string(t_out)

        t_eta = (t_out / (batch_i + 1)) * (n_batches - (batch_i + 1))
        eta_s = time_to_string(t_eta)
        epoch_hdr = '{:}Epoch {:03} ({:03d}/{:03d}) [{:}>{:}] '
        loss_s = '{:} {:f} ({:f}) {:} / ETA {:}'
        batch_s = (epoch_hdr + loss_s).format(
            init_c + whites, self.epoch, batch_i + 1, n_batches,
            progress_s, remainder_s,
            loss_name, b_loss, mean_loss, time_s, eta_s + '\033[0m'
        )
        print('\033[K', end='', flush=True)
        print(batch_s, end='\r', flush=True)

    def save_model(self, net_name):
        torch.save(self.state_dict(), net_name)

    def load_model(self, net_name):
        self.load_state_dict(torch.load(net_name))


class Autoencoder(BaseModel):
    """
    Main autoencoder class. This class can actually be parameterised on init
    to have different "main blocks", normalisation layers and activation
    functions.
    """
    def __init__(
            self,
            conv_filters,
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
            n_inputs=1,
            kernel=3,
            pooling=False,
            norm=None,
            activation=None,
            block=None,
            dropout=0,
    ):
        """
        Constructor of the class. It's heavily parameterisable to allow for
        different autoencoder setups (residual blocks, double convolutions,
        different normalisation and activations).
        :param conv_filters: Filters for both the encoder and decoder. The
         decoder mirrors the filters of the encoder.
        :param device: Device where the model is stored (default is the first
         cuda device).
        :param n_inputs: Number of input channels.
        :param kernel: Kernel width for the main block.
        :param pooling: Whether to use pooling or not.
        :param norm: Normalisation block (it has to be a pointer to a valid
         normalisation Module).
        :param activation: Activation block (it has to be a pointer to a valid
         activation Module).
        :param block: Main block. It has to be a pointer to a valid block from
         this python file (otherwise it will fail when trying to create a
         partial of it).
        :param dropout: Dropout value.
        """
        super().__init__()
        # Init
        if norm is None:
            norm = partial(lambda ch_in: nn.Sequential())
        if activation is None:
            activation = nn.ReLU
        if block is None:
            block = Conv3dBlock
        block_partial = partial(
            block, kernel=kernel, norm=norm, activation=activation
        )
        self.pooling = pooling
        self.device = device
        self.dropout = dropout

        # Down path
        # We'll use the partial and fill it with the channels for input and
        # output for each level.
        self.down = nn.ModuleList([
            block_partial(f_in, f_out) for f_in, f_out in zip(
                [n_inputs] + conv_filters[:-2], conv_filters[:-1]
            )
        ])

        # Bottleneck
        self.u = block_partial(conv_filters[-2], conv_filters[-1])

        # Up path
        # Now we'll do the same we did on the down path, but mirrored. We also
        # need to account for the skip connections, that's why we sum the
        # channels for both outputs. That basically means that we are
        # concatenating with the skip connection, and not suming.
        down_out = conv_filters[-2::-1]
        up_out = conv_filters[:0:-1]
        deconv_in = map(sum, zip(down_out, up_out))
        self.up = nn.ModuleList([
            block_partial(f_in, f_out, inv=True) for f_in, f_out in zip(
                deconv_in, down_out
            )
        ])

    def forward(self, input_s):
        # We need to keep track of the convolutional outputs, for the skip
        # connections.
        down_inputs = []
        for c in self.down:
            c.to(self.device)
            input_s = F.dropout3d(
                c(input_s), self.dropout, self.training
            )
            down_inputs.append(input_s)
            # Remember that pooling is optional
            if self.pooling:
                input_s = F.max_pool3d(input_s, 2)

        self.u.to(self.device)
        input_s = F.dropout3d(self.u(input_s), self.dropout, self.training)

        for d, i in zip(self.up, down_inputs[::-1]):
            d.to(self.device)
            # Remember that pooling is optional
            if self.pooling:
                input_s = F.dropout3d(
                    d(
                        torch.cat(
                            (F.interpolate(input_s, size=i.size()[2:]), i),
                            dim=1
                        )
                    ),
                    self.dropout,
                    self.training
                )
            else:
                input_s = F.dropout3d(
                    d(torch.cat((input_s, i), dim=1)),
                    self.dropout,
                    self.training
                )

        return input_s


class Conv3dBlock(BaseModel):
    def __init__(
            self, filters_in, filters_out,
            kernel=3, norm=None, activation=None, inv=False
    ):
        super().__init__()
        if not inv:
            conv = nn.Conv3d
        else:
            conv = nn.ConvTranspose3d
        if norm is None:
            norm = partial(lambda ch_in: nn.Sequential())
        if activation is None:
            activation = nn.ReLU

        # Single Conv3d with activation and normalisation.
        self.block = nn.Sequential(
            conv(filters_in, filters_out, kernel, padding=kernel // 2),
            activation(),
            norm(filters_out)
        )

    def forward(self, inputs):
        return self.block(inputs)


class DoubleConv3dBlock(BaseModel):
    def __init__(
            self, filters_in, filters_out,
            kernel=3, norm=None, activation=None, inv=False
    ):
        super().__init__()
        if not inv:
            conv = nn.Conv3d
        else:
            conv = nn.ConvTranspose3d
        if norm is None:
            norm = partial(lambda ch_in: nn.Sequential())
        if activation is None:
            activation = nn.ReLU

        # Two sequential Conv3D  with activation and normalisation.
        self.block = nn.Sequential(
            conv(filters_in, filters_out, kernel, padding=kernel // 2),
            activation(),
            norm(filters_out),
            conv(filters_out, filters_out, kernel, padding=kernel // 2),
            activation(),
            norm(filters_out)
        )

    def forward(self, inputs):
        return self.block(inputs)


class ResConv3dBlock(BaseModel):
    def __init__(
            self, filters_in, filters_out,
            kernel=3, norm=None, activation=None, inv=False
    ):
        super().__init__()
        if not inv:
            conv = nn.Conv3d
        else:
            conv = nn.ConvTranspose3d
        if norm is None:
            norm = partial(lambda ch_in: nn.Sequential())
        if activation is None:
            activation = nn.ReLU

        # Single Conv3d with a convolutional (1x1x1) residual connection,
        # activation and normalisation.
        self.conv = conv(
            filters_in, filters_out, kernel,
            padding=kernel // 2
        )

        self.res = conv(
            filters_in, filters_out, 1,
        )

        self.end_seq = nn.Sequential(
            activation(),
            norm(filters_out)
        )

    def forward(self, inputs):
        res = self.conv(inputs) + self.res(inputs)
        return self.end_seq(res)


class Res3dBlock(BaseModel):
    def __init__(
            self, filters_in, filters_out,
            kernel=3, norm=None, activation=None, inv=False
    ):
        super().__init__()
        if not inv:
            conv = nn.Conv3d
        else:
            conv = nn.ConvTranspose3d
        if norm is None:
            norm = partial(lambda ch_in: nn.Sequential())
        if activation is None:
            activation = nn.ReLU

        # Single Conv3d with a residual connection, activation and
        # normalisation.
        self.conv = conv(
            filters_in, filters_out, kernel,
            padding=kernel // 2
        )

        self.end_seq = nn.Sequential(
            activation(),
            norm(filters_out)
        )

    def forward(self, inputs):
        res = self.conv(inputs) + inputs
        return self.end_seq(res)


class LesionsUNet(BaseModel):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            n_images=1,
            dropout=0,
            verbose=0,
            deep=False,
    ):
        super(LesionsUNet, self).__init__()
        # Init values
        if conv_filters is None:
            conv_filters = [32, 32, 64, 128, 256]
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device
        self.dropout = dropout
        self.deep = deep

        print("image channel %s\n" % str(n_images))
        # <Parameter setup>
        self.autoencoder = Autoencoder(
            conv_filters, device, n_images, pooling=True, norm=nn.BatchNorm3d
        )
        self.autoencoder.dropout = dropout

        self.seg = nn.Sequential(
            nn.Conv3d(conv_filters[0], conv_filters[0], 1),
            nn.ReLU(),
            nn.BatchNorm3d(conv_filters[0]),
            nn.Conv3d(conv_filters[0], 2, 1)
        )
        self.seg.to(device)

        # # <Loss function setup>
        # self.train_functions = [
        #     # {
        #     #     'name': 'dsc',
        #     #     'weight': 1,
        #     #     'f': lambda p, t: multidsc_loss(p, t)
        #     # },
        #     {
        #         'name': 'dsc',
        #         'weight': 1,
        #         'f': lambda p, t:  dsc_loss(
        #             p[:, 0, ...], torch.squeeze((t == 0), dim=1))
        #     },
        #     {
        #         'name': 'xentr',
        #         'weight': 1,
        #         'f': lambda p, t: F.cross_entropy(
        #             p, torch.squeeze(t, dim=1).type(torch.long).to(p.device)
        #         )
        #     },
        # ]

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
                'name': 'bck',
                'weight': 0.5,
                'f': lambda p, t:  dsc_loss(
                    p[:, 0, ...], torch.squeeze((t == 0), dim=1)
                )
            },
            {
                'name': 'les',
                'weight': 0.5,
                'f': lambda p, t:  dsc_loss(
                    p[:, 1, ...], torch.squeeze((t == 1), dim=1)
                )
            },
        ]

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        # self.optimizer_alg = torch.optim.Adadelta(model_params)
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
        input_s = self.autoencoder(data)
        multi_seg = torch.softmax(self.seg(input_s), dim=1)

        return multi_seg

    def dropout_update(self):
        super().dropout_update()
        self.autoencoder.dropout = self.dropout

    def lesions(
            self,
            data,
            verbose=0
    ):
        # Init
        self.eval()

        data_tensor = to_torch_var(
            np.expand_dims(data, axis=0), device=self.device
        )

        with torch.no_grad():
            if self.device == torch.device('cpu'):
                seg = self(data_tensor)
            else:
                torch.cuda.synchronize(self.device)
                seg = self(data_tensor)
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()

        if verbose > 1:
            print(
                '\033[K{:}Segmentation finished'.format(' '.join([''] * 12))
            )

        seg = seg.cpu().numpy()[0]
        return seg

    def patch_lesions(
            self,
            data,
            patch_size,
            verbose=0
    ):
        # Init
        self.eval()

        seg = np.zeros((2,) + data.shape[1:])
        # The following lines are just a complicated way of finding all
        # the possible combinations of patch indices.
        limits = tuple(
            list(range(0, lim, patch_size))[:-1] + [lim - patch_size]
            for lim in data.shape[1:]
        )
        limits_product = list(itertools.product(*limits))

        # The following code is just a normal test loop with all the
        # previously computed patches.
        for patchi, (xi, xj, xk) in enumerate(limits_product):
            # Here we just take the current patch defined by its slice
            # in the x and y axes. Then we convert it into a torch
            # tensor for testing.
            xslice = slice(xi, xi + patch_size)
            yslice = slice(xj, xj + patch_size)
            zslice = slice(xk, xk + patch_size)
            data_tensor = to_torch_var(
                np.expand_dims(
                    data[slice(None), xslice, yslice, zslice],
                    axis=0
                )
            )
            with torch.no_grad():
                torch.cuda.synchronize(self.device)
                seg_i = self(data_tensor)
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
                seg_i = seg_i.cpu().numpy()
                seg[slice(None), xslice, yslice, zslice] = seg_i[0]

        if verbose > 1:
            print(
                '\033[K{:}Segmentation finished'.format(' '.join([''] * 12))
            )
        return seg
