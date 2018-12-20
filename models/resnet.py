import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, weights=None, get_feat=None):
        if weights==None:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            feat = x.view(x.size(0), -1)
            x = self.fc(feat)
            if get_feat:
                return x,feat
            else:
                return x
        else:
                        
            x = F.conv2d(x, weights['conv1.weight'], stride=2, padding=3)
            x = F.batch_norm(x, self.bn1.running_mean, self.bn1.running_var, weights['bn1.weight'], weights['bn1.bias'],training=True)            
            x = F.threshold(x, 0, 0, inplace=True)
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            #layer 1
            for i in range(3):
                residual = x
                out = F.conv2d(x, weights['layer1.%d.conv1.weight'%i], stride=1)
                out = F.batch_norm(out, self.layer1[i].bn1.running_mean, self.layer1[i].bn1.running_var, 
                                 weights['layer1.%d.bn1.weight'%i], weights['layer1.%d.bn1.bias'%i],training=True)      
                out = F.threshold(out, 0, 0, inplace=True)
                out = F.conv2d(out, weights['layer1.%d.conv2.weight'%i], stride=1, padding=1)
                out = F.batch_norm(out, self.layer1[i].bn2.running_mean, self.layer1[i].bn2.running_var, 
                                 weights['layer1.%d.bn2.weight'%i], weights['layer1.%d.bn2.bias'%i],training=True)     
                out = F.threshold(out, 0, 0, inplace=True)
                out = F.conv2d(out, weights['layer1.%d.conv3.weight'%i], stride=1)
                out = F.batch_norm(out, self.layer1[i].bn3.running_mean, self.layer1[i].bn3.running_var, 
                                 weights['layer1.%d.bn3.weight'%i], weights['layer1.%d.bn3.bias'%i],training=True)                               
                if i==0:
                    residual = F.conv2d(x, weights['layer1.%d.downsample.0.weight'%i], stride=1)  
                    residual = F.batch_norm(residual, self.layer1[i].downsample[1].running_mean, self.layer1[i].downsample[1].running_var, 
                                 weights['layer1.%d.downsample.1.weight'%i], weights['layer1.%d.downsample.1.bias'%i],training=True)  
                x = out + residual     
                x = F.threshold(x, 0, 0, inplace=True)
            #layer 2
            for i in range(4):
                residual = x
                out = F.conv2d(x, weights['layer2.%d.conv1.weight'%i], stride=1)
                out = F.batch_norm(out, self.layer2[i].bn1.running_mean, self.layer2[i].bn1.running_var, 
                                 weights['layer2.%d.bn1.weight'%i], weights['layer2.%d.bn1.bias'%i],training=True)     
                out = F.threshold(out, 0, 0, inplace=True)
                if i==0:
                    out = F.conv2d(out, weights['layer2.%d.conv2.weight'%i], stride=2, padding=1)
                else:
                    out = F.conv2d(out, weights['layer2.%d.conv2.weight'%i], stride=1, padding=1)
                out = F.batch_norm(out, self.layer2[i].bn2.running_mean, self.layer2[i].bn2.running_var, 
                                 weights['layer2.%d.bn2.weight'%i], weights['layer2.%d.bn2.bias'%i],training=True)    
                out = F.threshold(out, 0, 0, inplace=True)
                out = F.conv2d(out, weights['layer2.%d.conv3.weight'%i], stride=1)
                out = F.batch_norm(out, self.layer2[i].bn3.running_mean, self.layer2[i].bn3.running_var, 
                                 weights['layer2.%d.bn3.weight'%i], weights['layer2.%d.bn3.bias'%i],training=True)                    
                if i==0:
                    residual = F.conv2d(x, weights['layer2.%d.downsample.0.weight'%i], stride=2)  
                    residual = F.batch_norm(residual, self.layer2[i].downsample[1].running_mean, self.layer2[i].downsample[1].running_var, 
                                 weights['layer2.%d.downsample.1.weight'%i], weights['layer2.%d.downsample.1.bias'%i],training=True)  
                x = out + residual  
                x = F.threshold(x, 0, 0, inplace=True)
            #layer 3
            for i in range(6):
                residual = x
                out = F.conv2d(x, weights['layer3.%d.conv1.weight'%i], stride=1)
                out = F.batch_norm(out, self.layer3[i].bn1.running_mean, self.layer3[i].bn1.running_var, 
                                 weights['layer3.%d.bn1.weight'%i], weights['layer3.%d.bn1.bias'%i],training=True)   
                out = F.threshold(out, 0, 0, inplace=True)
                if i==0:
                    out = F.conv2d(out, weights['layer3.%d.conv2.weight'%i], stride=2, padding=1)
                else:
                    out = F.conv2d(out, weights['layer3.%d.conv2.weight'%i], stride=1, padding=1)
                out = F.batch_norm(out, self.layer3[i].bn2.running_mean, self.layer3[i].bn2.running_var, 
                                 weights['layer3.%d.bn2.weight'%i], weights['layer3.%d.bn2.bias'%i],training=True)     
                out = F.threshold(out, 0, 0, inplace=True)
                out = F.conv2d(out, weights['layer3.%d.conv3.weight'%i], stride=1)
                out = F.batch_norm(out, self.layer3[i].bn3.running_mean, self.layer3[i].bn3.running_var, 
                                 weights['layer3.%d.bn3.weight'%i], weights['layer3.%d.bn3.bias'%i],training=True)                    
                if i==0:
                    residual = F.conv2d(x, weights['layer3.%d.downsample.0.weight'%i], stride=2)  
                    residual = F.batch_norm(residual, self.layer3[i].downsample[1].running_mean, self.layer3[i].downsample[1].running_var, 
                                 weights['layer3.%d.downsample.1.weight'%i], weights['layer3.%d.downsample.1.bias'%i],training=True)  
                x = out + residual    
                x = F.threshold(x, 0, 0, inplace=True)
                
            #layer 4
            for i in range(3):
                residual = x
                out = F.conv2d(x, weights['layer4.%d.conv1.weight'%i], stride=1)
                out = F.batch_norm(out, self.layer4[i].bn1.running_mean, self.layer4[i].bn1.running_var, 
                                 weights['layer4.%d.bn1.weight'%i], weights['layer4.%d.bn1.bias'%i],training=True)   
                out = F.threshold(out, 0, 0, inplace=True)
                if i==0:
                    out = F.conv2d(out, weights['layer4.%d.conv2.weight'%i], stride=2, padding=1)
                else:
                    out = F.conv2d(out, weights['layer4.%d.conv2.weight'%i], stride=1, padding=1)
                out = F.batch_norm(out, self.layer4[i].bn2.running_mean, self.layer4[i].bn2.running_var, 
                                 weights['layer4.%d.bn2.weight'%i], weights['layer4.%d.bn2.bias'%i],training=True)   
                out = F.threshold(out, 0, 0, inplace=True)
                out = F.conv2d(out, weights['layer4.%d.conv3.weight'%i], stride=1)
                out = F.batch_norm(out, self.layer4[i].bn3.running_mean, self.layer4[i].bn3.running_var, 
                                 weights['layer4.%d.bn3.weight'%i], weights['layer4.%d.bn3.bias'%i],training=True)                    
                if i==0:
                    residual = F.conv2d(x, weights['layer4.%d.downsample.0.weight'%i], stride=2)  
                    residual = F.batch_norm(residual, self.layer4[i].downsample[1].running_mean, self.layer4[i].downsample[1].running_var, 
                                 weights['layer4.%d.downsample.1.weight'%i], weights['layer4.%d.downsample.1.bias'%i],training=True)  
                x = out + residual    
                x = F.threshold(x, 0, 0, inplace=True)
                
            x = F.avg_pool2d(x, kernel_size=7, stride=1, padding=0)
            x = x.view(x.size(0), -1)
            x = F.linear(x, weights['fc.weight'], weights['fc.bias'])                
            return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model