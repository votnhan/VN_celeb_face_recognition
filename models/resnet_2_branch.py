import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from .resnet_2_branch_utils import Bottleneck, load_state_dict, model_urls


__all__ = ['resnet_2branch_50']


class ResNet2Branch(nn.Module):

    def __init__(self, block, layers, num_classes=1000, num_projections=300):
        self.inplanes = 64
        super(ResNet2Branch, self).__init__()
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
        self.proj = nn.Linear(512 * block.expansion, num_projections)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x_cls = self.fc(x)
        x_proj = self.proj(x)
        return x_cls, x_proj


def resnet_2branch_50(pretrained=False, checkpoint_path=None,**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet2Branch(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print("Loading Pretrained data!")
        load_state_dict(model, model_zoo.load_url(model_urls['resnet50']))

    if checkpoint_path is not None:
        print('Loaded emotion model from checkpoint path {}'.format(checkpoint_path))
        state_dict = torch.load(checkpoint_path)['state_dict']
        model = nn.DataParallel(model)
        model.load_state_dict(state_dict)

    return model
