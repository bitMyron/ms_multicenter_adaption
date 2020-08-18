import torch
from chaoyi.utils import ImagePool
from BaseModelSetup import BaseModel
import networks
import itertools
import numpy as np
import torch.nn as nn
import random
import torch.nn.functional as F
from diceloss import *

def dice_loss_bp(input, target):
    smooth = 1.
    iflat = input.view(-1)
    tflat = target.view(-1) # Chaoyi-long()
    intersection = (iflat * tflat).sum()
    diceloss = 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))
    diceloss = diceloss.float()
    return diceloss

class CustomCrossEntropy2D(nn.modules.Module):
    def __init__(self, ratio):
        super(CustomCrossEntropy2D, self).__init__()
        if ratio is None:
            self.weight = None
        else:
            self.weight = self.prep_class_weights(ratio)
    def prep_class_weights(self, ratio):
        weight_foreback = torch.ones(2)
        weight_foreback[0] = 1 / (1 - ratio)
        weight_foreback[1] = 1 / ratio
        weight_foreback = weight_foreback.cuda()
        return weight_foreback
    def forward(self, input, target):
        n, c, h, w = input.size()
        target = target.squeeze()   #Chaoyi

        log_p = F.log_softmax(input, dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[target.view(-1, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        mask = target >= 0
        target = target[mask]
        target = target.long()   # Chaoyi
        loss = F.nll_loss(log_p, target, ignore_index=250,
                          weight=self.weight, size_average=False)
        loss /= mask.data.sum().float()
        return loss


class MSGAN(BaseModel):
    def name(self):
        return 'MSGAN'
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0)
        parser.set_defaults(no_lsgan=True)
        parser.set_defaults(norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(which_model_netG='unet_256')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def initialize(self, opt):
        print(opt)
        BaseModel.initialize(self, opt)
        self.isTrain = opt['isTrain']
        self.pool_size = opt['pool_size']
        self.lr = opt['lr']
        self.beta1 = opt['beta1']
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_CE', 'G_L1', 'dice_LOSS', 'D_real', 'D_fake', 'dice_VALUE', ]
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_T1', 'real_T2', 'fake_SEG_visualize', 'real_SEG']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D_T1', 'D_T2']
            #self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(self.gpu_ids)
        self.custom_m = torch.sigmoid
        if self.isTrain:
            self.netD_T1 = networks.define_D(2, self.gpu_ids)
            self.netD_T2 = networks.define_D(2, self.gpu_ids)
        if self.isTrain:
            self.fake_T1SEG_pool = ImagePool(self.pool_size)
            self.fake_T2SEG_pool = ImagePool(self.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=False).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionDice = dice_loss_bp
            # Chaoyi - Use Weighted CrossEntropy 2D replace L1 loss
            weight = 0.001
            self.criterionCE2D = CustomCrossEntropy2D(weight)
            print('Using Custom Cross_Entropy2d [weight={}]'.format(weight))

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_T1.parameters(),
                                                                self.netD_T2.parameters()),
                                                lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_T1 = input['T1'].to(self.device)
        self.real_T2 = input['T2'].to(self.device)
        self.real_SEG = input['SEG'].to(self.device)
        self.location_infos = input['location_infos']

    def forward(self):
        fake_SEG_2channels = self.netG(self.real_T1, self.real_T2)   # bn x n_classes x W x H

        # Dice - Calculate Dice
        _, pred_map = fake_SEG_2channels.max(1)
        self.loss_dice_VALUE = dice_loss(pred_map, self.real_SEG)

        # Segmentation  [nChannels=1/1/1/2]
        self.fake_SEG_loss = fake_SEG_2channels
        self.fake_SEG_visualize = pred_map.unsqueeze(1).float()
        fake_SEG_lesion = self.custom_m(fake_SEG_2channels[:, 1, :, :].unsqueeze(1))
        fake_SEG_backgd = self.custom_m(fake_SEG_2channels[:, 0, :, :].unsqueeze(1))

        if random.random() < 0.012:
            print('--------------------------------------------------------------------------------')
            print('[vis]', np.unique(self.fake_SEG_visualize.cpu().detach().numpy()))
            print('[lbl]', np.unique(self.real_SEG.cpu().detach().numpy()))
            print('[les]', 'min={} | max={}'.format(np.min(fake_SEG_lesion.cpu().detach().numpy()), np.max(fake_SEG_lesion.cpu().detach().numpy())))
            print('[bgd]', 'min={} | max={}'.format(np.min(fake_SEG_backgd.cpu().detach().numpy()), np.max(fake_SEG_backgd.cpu().detach().numpy())))
            print('[los]', 'min={} | max={}'.format(np.min(self.fake_SEG_loss.cpu().detach().numpy()), np.max(self.fake_SEG_loss.cpu().detach().numpy())))
            print('--------------------------------------------------------------------------------')

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_T1SEG = self.fake_T1SEG_pool.query(torch.cat((self.real_T1[:,1,:,:].unsqueeze(1),
                                                           self.fake_SEG_visualize.detach()), 1))
        fake_T2SEG = self.fake_T2SEG_pool.query(torch.cat((self.real_T2[:,1,:,:].unsqueeze(1),
                                                           self.fake_SEG_visualize.detach()), 1))
        pred_fake_T1SEG = self.netD_T1(fake_T1SEG)
        pred_fake_T2SEG = self.netD_T2(fake_T2SEG)
        self.loss_D_fake = self.criterionGAN(pred_fake_T1SEG, False) \
                           + self.criterionGAN(pred_fake_T2SEG, False) 

        # Real
        real_T1SEG = torch.cat((self.real_T1[:,1,:,:].unsqueeze(1),
                                self.real_SEG), 1)
        real_T2SEG = torch.cat((self.real_T2[:,1,:,:].unsqueeze(1),
                                self.real_SEG), 1)
        pred_real_T1SEG = self.netD_T1(real_T1SEG)
        pred_real_T2SEG = self.netD_T2(real_T2SEG)
        self.loss_D_real = self.criterionGAN(pred_real_T1SEG, True) \
                           + self.criterionGAN(pred_real_T2SEG, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_T1SEG = torch.cat((self.real_T1[:,1,:,:].unsqueeze(1),
                                self.fake_SEG_visualize), 1)
        fake_T2SEG = torch.cat((self.real_T2[:,1,:,:].unsqueeze(1),
                                self.fake_SEG_visualize), 1)
        pred_fakeT1SEG = self.netD_T1(fake_T1SEG)
        pred_fakeT2SEG = self.netD_T2(fake_T2SEG)
        self.loss_G_GAN = self.criterionGAN(pred_fakeT1SEG, True) \
                          + self.criterionGAN(pred_fakeT2SEG, True) 

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_SEG_visualize, self.real_SEG) * self.opt.lambda_L1
        self.loss_G_CE = self.criterionCE2D.forward(self.fake_SEG_loss, self.real_SEG) * self.opt.lambda_L1
        self.loss_dice_LOSS = self.criterionDice(self.fake_SEG_visualize, self.real_SEG)  * self.opt.lambda_L1

        self.loss_G = self.loss_G_GAN + self.loss_G_CE + self.loss_dice_LOSS + self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad([self.netD_T1, self.netD_T2], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad([self.netD_T1, self.netD_T2], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
