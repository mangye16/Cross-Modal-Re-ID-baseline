import torch
import torch.nn as nn
import torchvision
from torch.nn import init
import torch.nn.functional as F
from .pooling import GeneralizedMeanPoolingP


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ModelV2(nn.Module):

    def __init__(self, class_num):
        super(ModelV2, self).__init__()

        self.class_num = class_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)

        # cnn backbone
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool, # no relu
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, self.class_num, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):

        features = self.gap(self.resnet_conv(x)).squeeze()
        bn = self.bottleneck(features)
        cls_score = self.classifier(bn)

        if self.training:
            return features, cls_score
        else:
            return bn


class MemoryBankModel(nn.Module):
    def __init__(self, out_dim, dropout=0.5, num_classes=0, use_bnneck=True, pool_type='avgpool'):
        super(MemoryBankModel,self).__init__()
        
        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)

        # cnn backbone
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool, # no relu
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

        if pool_type == 'avgpool':
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        if pool_type == 'gempool':
            self.global_pool = GeneralizedMeanPoolingP()  # default initial norm=3

        self.bottleneck = nn.BatchNorm1d(out_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

        self.drop = nn.Dropout(dropout)     

        self.class_num = num_classes
        if self.class_num > 0:
            self.classifier = nn.Linear(2048, self.class_num, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.use_bnneck = use_bnneck

    def forward(self, x, output_feature=None):
        xx = self.resnet_conv(x)
        x = self.global_pool(xx).squeeze()
        #x = self.gap(self.resnet_conv(x)).squeeze()

        if self.use_bnneck:
            bn = self.bottleneck(x)
        else:
            bn = x

        if self.training == True:
            if output_feature=='src_feat' and self.class_num > 0:
                cls_score = self.classifier(bn)
                return x, cls_score  # cls_score is for source dataset train
   
            bn = F.normalize(bn, p=2, dim=1)
            return x, bn
        else:
            bn = F.normalize(bn, p=2, dim=1)          
            return bn


