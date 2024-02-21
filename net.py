import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, width=512):
        super().__init__()

        self.extend = nn.Sequential(
            nn.Linear(9, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            # (input+2*padding-kernal)/stride+1
            nn.Conv1d(9, 64, 5, 1, 2, padding_mode="circular"),
            # 64
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 512, 5, 1, 2, padding_mode="circular"),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # 16
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(512, 512, 3, 1, 1, padding_mode="circular"),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 3, 1, 1, padding_mode="circular"),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.lin1 = nn.Sequential(
            nn.Linear(768, width),
            nn.ReLU(inplace=True),
        )
        self.lin2 = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
        )
        self.lin3 = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, 2)
        )

    @staticmethod
    def positional_encoding(input):
        shape = input.shape
        input = input.view([-1, 2])
        x, y = input[:, 0:1], input[:, 1:2]
        output = torch.cat([x, y, x ** 2, y ** 2, x * y, x ** 3, y ** 3, x * y * y, x * x * y], dim=1)
        output = output.view([*(shape[:-1]), -1])
        return output

    def get_feature(self, bound):
        if len(bound.shape) == 2:
            bound = bound.unsqueeze(0)
        N_bound = bound.shape[0]
        feature_1 = self.conv1(self.positional_encoding(bound).permute([0, 2, 1]))
        feature_2 = self.conv2(feature_1) + feature_1
        feature_3 = torch.mean(feature_2, dim=2).view([N_bound, 512])
        return feature_3

    def forward(self, x_, bound=None, feature=None):
        if len(x_.shape) == 2:
            x = x_.view([1, *x_.shape])
        else:
            x = x_
        N_bound, N_p = x.shape[0], x.shape[1]
        x_encode = self.positional_encoding(x)
        x_input = self.extend(x_encode.view([-1, 9]))
        # [n,Np,9]
        if feature is None:
            if bound is None:
                bound = torch.zeros([N_bound, 256, 2], dtype=x.dtype, device=x.device)
            if len(bound.shape) == 2:
                bound = bound.unsqueeze(0)
            feature_1 = self.conv1(self.positional_encoding(bound).permute([0, 2, 1]))
            feature_2 = self.conv2(feature_1) + feature_1
            feature_3 = torch.mean(feature_2, dim=2).view([N_bound, 512])
            # bound = bound.permute([0, 2, 1])
            # feature_2 = self.conv(bound).view([N_bound, -1])
            # feature = torch.cat([feature_1, feature_2], dim=1).unsqueeze(1)
            feature = feature_3.unsqueeze(1)
        else:
            feature = feature.unsqueeze(1)
        feature = feature.expand([N_bound, N_p, feature.shape[2]]).reshape([-1, feature.shape[2]])
        x_1 = self.lin1(torch.cat([x_input, feature], dim=1))
        x_2 = self.lin2(x_1)
        lin3_input = x_1 + x_2
        # x_3 = self.lin3(torch.cat([x_1, x_2], dim=1))
        x_3 = self.lin3(lin3_input)
        if len(x_.shape) == 3:
            x_3 = x_3.view([N_bound, N_p, 2])
        return x_3
