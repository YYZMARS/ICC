import torch
import torch.nn as nn


class InlierEncoder(nn.Module):
    def __init__(self, dim_in=4,dim=8):
        super(InlierEncoder, self).__init__()

        self.model1 = nn.Sequential(
            nn.Conv2d(dim_in, dim, kernel_size=(3, 1), bias=True, padding=(1, 0)),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim * 2, kernel_size=(3, 1), bias=True, padding=(1, 0)),
            nn.BatchNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True))

        self.model2 = nn.Sequential(
            nn.Conv2d(dim * 2, 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True))

        self.model3 = nn.Sequential(
            nn.Conv2d(dim * 2, 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True))

        self.model4 = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=1, bias=True),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(8, 1, kernel_size=1, bias=True),
            # nn.Tanh(),
        )

        self.tah = nn.Tanh()

    def forward(self, x, y):
        """
            Inlier Evaluation.
            Args:
                x: Source neighborhoods. Size [B, N, K, 3]
                y: Pesudo target neighborhoods. Size [B, N, K, 3]
            Returns:
                x: Inlier confidence. Size [B, 1, N]
        """
        b, n, k, _ = x.size()

        x_1x3 = self.model1(x.permute(0, 3, 2, 1)).permute(0, 1, 3, 2)

        y_1x3 = self.model1(y.permute(0, 3, 2, 1)).permute(0, 1, 3, 2)  # [b, n, k, 3]-[b, c, k, n]-->[b, c, n, k]

        x2 = x_1x3 - y_1x3  # Eq. (5)

        x = self.model2(x2)  # [b, c, n, k]
        weight = self.model3(x2)  # [b, c, n, k]
        weight = torch.softmax(weight, dim=-1)  # Eq. (6)
        x = (x * weight).sum(-1)  # [b, c, n]
        x = 1 - self.tah(torch.abs(self.model4(x)))
        return x
