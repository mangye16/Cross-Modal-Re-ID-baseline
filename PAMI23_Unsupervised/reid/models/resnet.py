from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from reid.lib.normalize import Normalize


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'ResNetV2', 'ResNetV3']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

        # Fix layers [conv1 ~ layer2]
        fixed_names = []
        for name, module in self.base._modules.items():
            if name == "layer3":
                # assert fixed_names == ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2"]
                break
            fixed_names.append(name)
            for param in module.parameters():
                param.requires_grad = False

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes
            #self.num_triplet_features = num_triplet_features
            #self.l2norm = Normalize(2)

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
                init.constant_(self.feat_bn.weight, 1)
                init.constant_(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout >= 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal_(self.classifier.weight, std=0.001)
                init.constant_(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x, output_feature=None):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            else:
                x = module(x)

        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if output_feature == 'pool5':
            x = F.normalize(x)
            return x

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
            tgt_feat = F.normalize(x)
            tgt_feat = self.drop(tgt_feat)
            if output_feature == 'tgt_feat':
                return tgt_feat
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)


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


class ResNetV2(nn.Module):

    def __init__(self, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNetV2, self).__init__()

        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        self.base = torchvision.models.resnet50(pretrained=True)

        # change the stride of the last residual block (new 1)
        self.base.layer4[0].conv2.stride = (1, 1)
        self.base.layer4[0].downsample[0].stride = (1, 1)

        # Fix layers [conv1 ~ layer2]
        fixed_names = []
        for name, module in self.base._modules.items():
            if name == "layer3":
                # assert fixed_names == ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2"]
                break
            fixed_names.append(name)
            for param in module.parameters():
                param.requires_grad = False

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes
            #self.num_triplet_features = num_triplet_features

            out_planes = self.base.fc.in_features

            # Append new layers
            # add BNNeck after GAP (new 2)
            #self.bottleneck = nn.BatchNorm1d(out_planes)
            #self.bottleneck.bias.requires_grad_(False)  # no shift
            #self.bottleneck.apply(weights_init_kaiming)

            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
                init.constant_(self.feat_bn.weight, 1)
                init.constant_(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout >= 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                #self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)  # (new 3)
                init.normal_(self.classifier.weight, std=0.001)
                init.constant_(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x, output_feature=None):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            else:
                x = module(x)

        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        trip_feat = x
        #x = self.bottleneck(x)  # add BNNeck (new)  

        if output_feature == 'pool5':  # for evaluation
            x = F.normalize(x)  # BNNeck feature
            return x

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
            tgt_feat = F.normalize(x)
            tgt_feat = self.drop(tgt_feat)
            if output_feature == 'tgt_feat':  # for memory bank
                return tgt_feat, trip_feat
        if self.norm:  # False
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x, trip_feat  # x for FC classification, trip_feat for triplet loss

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


class ResNetV3(nn.Module):
    def __init__(self, pretrained=True, num_features=0, dropout=0, num_classes=0):
        super(ResNetV3, self).__init__()

        self.pretrained = pretrained

        # Construct base (pretrained) resnet
        self.base = torchvision.models.resnet50(pretrained=True)

        # change the stride of the last residual block (new 1)
        self.base.layer4[0].conv2.stride = (1, 1)
        self.base.layer4[0].downsample[0].stride = (1, 1)

        # Fix layers [conv1 ~ layer2]
        fixed_names = []
        for name, module in self.base._modules.items():
            if name == "layer3":
                # assert fixed_names == ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2"]
                break
            fixed_names.append(name)
            for param in module.parameters():
                param.requires_grad = False

        self.num_features = num_features
        self.dropout = dropout
        self.has_embedding = num_features > 0
        self.num_classes = num_classes
        # self.num_triplet_features = num_triplet_features

        out_planes = self.base.fc.in_features

        # Append new layers
        # add BNNeck after GAP (new 2)
        self.bottleneck = nn.BatchNorm1d(out_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

        if self.has_embedding:
            self.feat = nn.Linear(out_planes, self.num_features)
            self.feat_bn = nn.BatchNorm1d(self.num_features)
            init.kaiming_normal_(self.feat.weight, mode='fan_out')
            init.constant_(self.feat.bias, 0)
            init.constant_(self.feat_bn.weight, 1)
            init.constant_(self.feat_bn.bias, 0)
        else:
            # Change the num_features to CNN output channels
            self.num_features = out_planes
        if self.dropout >= 0:
            self.drop = nn.Dropout(self.dropout)
        if self.num_classes > 0:
            self.classifier = nn.Linear(2048, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x, output_feature=None):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            else:
                x = module(x)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)  # GAP feature
        bn = self.bottleneck(x)

        if output_feature == 'pool5':  # for evaluation
            return bn

        if self.has_embedding:
            embed_feat = self.feat(bn)
            embed_feat = self.feat_bn(embed_feat)
            embed_feat = F.normalize(embed_feat)
            embed_feat = self.drop(embed_feat)
            if output_feature == 'tgt_feat':  # for memory bank
                return embed_feat, x         

        if self.num_classes > 0:  # src dataset classification
            cls_score = self.classifier(bn)
        return cls_score

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)
                    
                    

