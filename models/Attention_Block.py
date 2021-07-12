import torch
import torch.nn as nn


class Channel(nn.Module):
    def __init__(self, in_channel):
        super(Channel, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.max_pooling = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.channel_attention = nn.Sequential(nn.Linear(in_channel, in_channel // 8),
                                               nn.BatchNorm1d(in_channel // 8),
                                               nn.ReLU(),
                                               nn.Linear(in_channel // 8, in_channel))

    def forward(self, in_tensor):

        x_avg = self.avg_pooling(in_tensor)
        x_avg = x_avg.view(in_tensor.size(0), -1)
        att_x_avg = self.channel_attention(x_avg)

        x_max = self.max_pooling(in_tensor)
        x_max = x_max.view(in_tensor.size(0), -1)
        att_x_max = self.channel_attention(x_max)

        att_c = torch.sigmoid(att_x_avg + att_x_max)
        att_c = att_c.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(in_tensor)

        return att_c


class Spatial(nn.Module):
    def __init__(self, in_channel):
        super(Spatial, self).__init__()
        input_channel = in_channel * 2
        self.channel_attention = nn.Sequential(nn.Conv2d(input_channel, input_channel//2, kernel_size=1, stride=1),
                                               nn.BatchNorm2d(input_channel//2),
                                               nn.ReLU(),
                                               nn.Conv2d(input_channel//2, 1, kernel_size=1, stride=1))

    def forward(self, in_tensor):
        x_avg = torch.mean(in_tensor, dim=2)
        x_max = torch.max(in_tensor, dim=2)[0]

        x_fusion = torch.cat((x_avg, x_max), 1)                # (batch, 64*2, 28, 28)
        att_s = torch.sigmoid(self.channel_attention(x_fusion))
        att_s = att_s.unsqueeze(2).expand_as(in_tensor)        # (batch, 64, 8, 28, 28)
        return att_s


class Temporal(nn.Module):
    def __init__(self, spatial):
        super(Temporal, self).__init__()
        size = spatial * spatial * 2
        self.temporal_attention = nn.Sequential(nn.Conv1d(size, size//2, 1, bias=False),
                                                nn.BatchNorm1d(size//2),
                                                nn.ReLU(),
                                                nn.Conv1d(size//2, 1, 1, bias=False))

    def forward(self, in_tensor):
        in_tensor_t = in_tensor.transpose(1, 2)
        x_avg = torch.mean(in_tensor_t, dim=2)      # (B, T, H, W)
        x_avg = x_avg.view(in_tensor.size(0), in_tensor.size(2), -1)

        x_max = torch.max(in_tensor_t, dim=2)[0]    # (B, T, H, W)
        x_max = x_max.view(in_tensor.size(0), in_tensor.size(2), -1)

        x = torch.cat((x_avg, x_max), dim=2)        # (B, T, H*W*2)
        x_squeeze_channel = x.permute(0, 2, 1)      # (B, H*W*2, T)
        att_t = torch.sigmoid(self.temporal_attention(x_squeeze_channel))
        att_t = att_t.view(x_squeeze_channel.size(0), -1)
        att_t = att_t.unsqueeze(1).unsqueeze(3).unsqueeze(4).expand_as(in_tensor)   # (B, C, T, H, W)

        return att_t


class Attention(nn.Module):
    def __init__(self, channel, spatial):
        super(Attention, self).__init__()
        self.channel_att = Channel(channel)
        self.temporal_att = Temporal(spatial)
        self.spatial_att = Spatial(channel)

    def forward(self, in_tensor):

        output = in_tensor * (1 + self.channel_att(in_tensor))
        output = output * (1 + self.temporal_att(in_tensor))
        output = output * (1 + self.spatial_att(in_tensor))

        return output


if __name__ == '__main__':
    tensor = torch.rand((128, 64, 8, 28, 28))
    attention = Attention(channel=64, spatial=28)
    out = attention(tensor)
