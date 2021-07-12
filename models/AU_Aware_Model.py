from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Attention_Block import Attention


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, block_inplanes, n_input_channels=3, conv1_t_size=7, conv1_t_stride=1,
                 no_max_pool=False, shortcut_type='B', widen_factor=1.0):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.conv1 = nn.Conv3d(n_input_channels, self.in_planes, kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2), padding=(conv1_t_size // 2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_1 = nn.Linear(2560, 1024)
        self.fc_2 = nn.Linear(1024, 7)

        self.local_nn_1 = nn.Sequential(conv3x3x3(3, 64, stride=2),
                                        nn.BatchNorm3d(64),
                                        nn.ReLU(),
                                        BasicBlock(in_planes=64, planes=128, stride=2,
                                                   downsample=nn.Sequential(conv1x1x1(64, 128, stride=2), nn.BatchNorm3d(128))),
                                        Attention(channel=128, spatial=10),
                                        BasicBlock(in_planes=128, planes=256, stride=2,
                                                   downsample=nn.Sequential(conv1x1x1(128, 256, stride=2), nn.BatchNorm3d(256))),
                                        BasicBlock(in_planes=256, planes=512, stride=2,
                                                   downsample=nn.Sequential(conv1x1x1(256, 512, stride=2), nn.BatchNorm3d(512))))
        self.local_nn_2 = nn.Sequential(conv3x3x3(3, 64, stride=2),
                                        nn.BatchNorm3d(64),
                                        nn.ReLU(),
                                        BasicBlock(in_planes=64, planes=128, stride=2,
                                                   downsample=nn.Sequential(conv1x1x1(64, 128, stride=2), nn.BatchNorm3d(128))),
                                        Attention(channel=128, spatial=10),
                                        BasicBlock(in_planes=128, planes=256, stride=2,
                                                   downsample=nn.Sequential(conv1x1x1(128, 256, stride=2), nn.BatchNorm3d(256))),
                                        BasicBlock(in_planes=256, planes=512, stride=2,
                                                   downsample=nn.Sequential(conv1x1x1(256, 512, stride=2), nn.BatchNorm3d(512))))
        self.local_nn_3 = nn.Sequential(conv3x3x3(3, 64, stride=2),
                                        nn.BatchNorm3d(64),
                                        nn.ReLU(),
                                        BasicBlock(in_planes=64, planes=128, stride=2,
                                                   downsample=nn.Sequential(conv1x1x1(64, 128, stride=2), nn.BatchNorm3d(128))),
                                        Attention(channel=128, spatial=10),
                                        BasicBlock(in_planes=128, planes=256, stride=2,
                                                   downsample=nn.Sequential(conv1x1x1(128, 256, stride=2), nn.BatchNorm3d(256))),
                                        BasicBlock(in_planes=256, planes=512, stride=2,
                                                   downsample=nn.Sequential(conv1x1x1(256, 512, stride=2), nn.BatchNorm3d(512))))
        self.local_nn_4 = nn.Sequential(conv3x3x3(3, 64, stride=2),
                                        nn.BatchNorm3d(64),
                                        nn.ReLU(),
                                        BasicBlock(in_planes=64, planes=128, stride=2,
                                                   downsample=nn.Sequential(conv1x1x1(64, 128, stride=2), nn.BatchNorm3d(128))),
                                        Attention(channel=128, spatial=10),
                                        BasicBlock(in_planes=128, planes=256, stride=2,
                                                   downsample=nn.Sequential(conv1x1x1(128, 256, stride=2), nn.BatchNorm3d(256))),
                                        BasicBlock(in_planes=256, planes=512, stride=2,
                                                   downsample=nn.Sequential(conv1x1x1(256, 512, stride=2), nn.BatchNorm3d(512))))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()
        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(conv1x1x1(self.in_planes, planes * block.expansion, stride), nn.BatchNorm3d(planes * block.expansion))
        layers = list()
        layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride, downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, local_bbox):

        # ----------Local----------
        local_size = 40
        batch_size, _, number_frames, _, _ = x.shape
        local_bbox = torch.reshape(local_bbox, (-1, number_frames, 16))
        local_bbox = local_bbox.int().cpu().numpy()
        x_local_1 = torch.zeros(batch_size, 3, number_frames, local_size, local_size)
        x_local_2 = torch.zeros(batch_size, 3, number_frames, local_size, local_size)
        x_local_3 = torch.zeros(batch_size, 3, number_frames, local_size, local_size)
        x_local_4 = torch.zeros(batch_size, 3, number_frames, local_size, local_size)
        for i in range(batch_size):
            for j in range(number_frames):
                x_local_1[i, :, j, :] = x[i, :, j, local_bbox[i, j, 1]:local_bbox[i, j, 3], local_bbox[i, j, 0]:local_bbox[i, j, 2]]      # left eye
                x_local_2[i, :, j, :] = x[i, :, j, local_bbox[i, j, 5]:local_bbox[i, j, 7], local_bbox[i, j, 4]:local_bbox[i, j, 6]]      # right eye
                x_local_3[i, :, j, :] = x[i, :, j, local_bbox[i, j, 9]:local_bbox[i, j, 11], local_bbox[i, j, 8]:local_bbox[i, j, 10]]    # nose
                x_local_4[i, :, j, :] = x[i, :, j, local_bbox[i, j, 13]:local_bbox[i, j, 15], local_bbox[i, j, 12]:local_bbox[i, j, 14]]  # mouth

        x_local_1 = x_local_1.cuda()
        x_local_1 = self.local_nn_1(x_local_1)
        x_local_1 = self.avgpool(x_local_1)
        x_local_1 = x_local_1.view(batch_size, -1)

        x_local_2 = x_local_2.cuda()
        x_local_2 = self.local_nn_2(x_local_2)
        x_local_2 = self.avgpool(x_local_2)
        x_local_2 = x_local_2.view(batch_size, -1)

        x_local_3 = x_local_3.cuda()
        x_local_3 = self.local_nn_3(x_local_3)
        x_local_3 = self.avgpool(x_local_3)
        x_local_3 = x_local_3.view(batch_size, -1)

        x_local_4 = x_local_4.cuda()
        x_local_4 = self.local_nn_4(x_local_4)
        x_local_4 = self.avgpool(x_local_4)
        x_local_4 = x_local_4.view(batch_size, -1)

        # ----------Global----------
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x_global = x.view(x.size(0), -1)

        # ----------Fusion----------
        result = torch.cat((x_local_1, x_local_2, x_local_3, x_local_4, x_global), 1)
        result = self.fc_1(result)
        result = self.fc_2(result)

        return result


def generate_model(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
