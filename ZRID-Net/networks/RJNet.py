import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class PDU(nn.Module):  # physical block
    def __init__(self, channel):
        super(PDU, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ka = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.td = nn.Sequential(
            default_conv(channel, channel, 3),
            default_conv(channel, channel // 8, 3),
            nn.ReLU(inplace=True),
            default_conv(channel // 8, channel, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        a = self.avg_pool(x)
        a = self.ka(a)
        t = self.td(x)
        j = torch.mul((1 - t), a) + torch.mul(t, x)
        return j


class Block(nn.Module):  # origin
    def __init__(self, conv, dim, kernel_size, ):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.pdu = PDU(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.pdu(res)
        res += x
        return res


class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res


class JNet(nn.Module):
    def __init__(self, gps=3, blocks=6, conv=default_conv):
        super(JNet, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.pdu = PDU(self.dim)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size),
            nn.Sigmoid()]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)


    def forward(self, x1):

        x = self.pre(x1)

        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]

        res_g1w = w[:, 0, ::] * res1
        res_g2w = w[:, 1, ::] * res2
        res_g3w = w[:, 2, ::] * res3

        res_g = res_g1w + res_g2w + res_g3w

        res_pdu = self.pdu(res_g)

        J = self.post(res_pdu)

        return J, res_g3w


class RKTLoss(nn.Module):
    def __init__(self, t_path=None, requires_grad=False):
        super(RKTLoss, self).__init__()

        self.T = JNet().cuda()
        self.T.load_state_dict(torch.load(t_path))
        print("Load teacher model weights successfully")

        self.l1 = nn.L1Loss()

        if not requires_grad:
            print("Teacher model requires_grad = False")

            for param in self.T.parameters():
                param.requires_grad = requires_grad

    def forward(self, x, y):

        _, feats_y = self.T(y)

        loss = 0

        for i in range(len(feats_y)):

            d_an = self.l1(x[i], feats_y[i].detach())

            contrastive = 1.0 / (d_an + 1e-7)

            loss += contrastive

        return loss


if __name__ == "__main__":
    x=torch.randn(1,3,256,256)
    net = JNet(gps=3, blocks=19)
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(total_params))
    J = net(x)
    print(J.shape)

