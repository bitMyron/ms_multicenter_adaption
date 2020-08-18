
import torch.nn as nn
import torchvision.models as models

from chaoyi.utils import *

ResnetsBackbone = {'resnet18' :{'layers':[2, 2, 2, 2],'filters':[64, 128, 256, 512], 'block':residualBlock,'expansion':1},
           'resnet34' :{'layers':[3, 4, 6, 3],'filters':[64, 128, 256, 512], 'block':residualBlock,'expansion':1},
           'resnet50' :{'layers':[3, 4, 6, 3],'filters':[64, 128, 256, 512], 'block':residualBottleneck,'expansion':4},
           'resnet101' :{'layers':[3, 4, 23, 3],'filters':[64, 128, 256, 512], 'block':residualBottleneck,'expansion':4},
           'resnet152':{'layers':[3, 8, 36, 3],'filters':[64, 128, 256, 512], 'block':residualBottleneck,'expansion':4}
            }

pretrained_models_t1 = {'resnet18':  models.resnet18(pretrained=True),
          'resnet34': models.resnet34(pretrained=True),
          'resnet50': models.resnet50(pretrained=True),
          'resnet101': models.resnet101(pretrained=True),
          'resnet152': models.resnet152(pretrained=True)
         }
pretrained_models_t2 = {'resnet18':  models.resnet18(pretrained=True),
          'resnet34': models.resnet34(pretrained=True),
          'resnet50': models.resnet50(pretrained=True),
          'resnet101': models.resnet101(pretrained=True),
          'resnet152': models.resnet152(pretrained=True)
         }

class MSBaseNet(nn.Module):

    def __init__(self, resnet='resnet18', feature_scale=4, pretrained=True, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(MSBaseNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.pretrained = pretrained

        assert resnet in ResnetsBackbone.keys(), 'Not a valid resnet, currently supported resnets are 18, 34, 50, 101 and 152'
        layers = ResnetsBackbone[resnet]['layers']
        filters = ResnetsBackbone[resnet]['filters']
        weights_t1 = pretrained_models_t1[resnet]
        weights_t2 = pretrained_models_t2[resnet]  # xx_t2 is acutually the flair modality. (For convenience)

        # filters = [x / self.feature_scale for x in filters]
        expansion =ResnetsBackbone[resnet]['expansion']

        self.inplanes = filters[0]


        # Encoder
        self.convbnrelu1_t1 = conv2DBatchNormRelu(in_channels=3, k_size=7, n_filters=64,
                                               padding=3, stride=2, bias=False)
        self.convbnrelu1_t2 = conv2DBatchNormRelu(in_channels=3, k_size=7, n_filters=64,
                                               padding=3, stride=2, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = ResnetsBackbone[resnet]['block']
        self.pretrained = True
        if self.pretrained:
            self.encoder1_t1 = weights_t1.layer1
            self.encoder2_t1 = weights_t1.layer2
            self.encoder3_t1 = weights_t1.layer3
            self.encoder4_t1 = weights_t1.layer4
            self.encoder1_t2 = weights_t2.layer1
            self.encoder2_t2 = weights_t2.layer2
            self.encoder3_t2 = weights_t2.layer3
            self.encoder4_t2 = weights_t2.layer4
        else:
            self.encoder1_t1 = self._make_layer(block, filters[0], layers[0])
            self.encoder2_t1 = self._make_layer(block, filters[1], layers[1], stride=2)
            self.encoder3_t1 = self._make_layer(block, filters[2], layers[2], stride=2)
            self.encoder4_t1 = self._make_layer(block, filters[3], layers[3], stride=2)
            self.encoder1_t2 = self._make_layer(block, filters[0], layers[0])
            self.encoder2_t2 = self._make_layer(block, filters[1], layers[1], stride=2)
            self.encoder3_t2 = self._make_layer(block, filters[2], layers[2], stride=2)
            self.encoder4_t2 = self._make_layer(block, filters[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Decoder
        self.decoder4 = linknetUp(filters[3] * expansion, filters[2] * expansion)
        self.decoder3 = linknetUp(filters[2] * expansion, filters[1] * expansion)
        self.decoder2 = linknetUp(filters[1] * expansion, filters[0] * expansion)
        self.decoder1 = linknetUp(filters[0] * expansion, filters[0])

        # Final Classifier
        self.finaldeconvbnrelu1 = deconv2DBatchNormRelu(filters[0], 32/feature_scale, 2, 2, 0)
        self.finalconvbnrelu2 = conv2DBatchNormRelu(in_channels=32/feature_scale, k_size=3, n_filters=32/feature_scale, padding=1, stride=1)
        self.finalconv3 = nn.Conv2d(int(32/feature_scale), 2, 3, 1, 1)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv2DBatchNorm(self.inplanes, planes*block.expansion, k_size=1, stride=stride, padding=0, bias=False)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, input):
        # Encoder
        input_t1 = input[:, 0:3, :, :]
        input_t1 = self.convbnrelu1_t1(input_t1)
        input_t1 = self.maxpool(input_t1)
        e1_t1 = self.encoder1_t1(input_t1)
        e2_t1 = self.encoder2_t1(e1_t1)
        e3_t1 = self.encoder3_t1(e2_t1)
        e4_t1 = self.encoder4_t1(e3_t1)

        input_t2 = input[:, 3:6, :, :]
        input_t2 = self.convbnrelu1_t1(input_t2)
        input_t2 = self.maxpool(input_t2)
        e1_t2 = self.encoder1_t2(input_t2)
        e2_t2 = self.encoder2_t2(e1_t2)
        e3_t2 = self.encoder3_t2(e2_t2)
        e4_t2 = self.encoder4_t2(e3_t2)  

        e4 = e4_t2 + e4_t1
        e3 = e3_t2 + e3_t1
        e2 = e2_t2 + e2_t1
        e1 = e1_t2 + e1_t1

        # 1-branch output
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        f1 = self.finaldeconvbnrelu1(d1)
        f2 = self.finalconvbnrelu2(f1)
        f3 = self.finalconv3(f2)
        return f3
