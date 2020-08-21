import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch_scatter import scatter_mean


def im2col(x, kh=8, kw=8):
    # x: [b, c, h, w]
    transforms_list = transforms.Compose([transforms.Lambda(lambda x: torch.transpose(x, 2, 3)),
                                          transforms.Lambda(lambda x: x.permute(0, 2, 3, 1).unfold(1, kh, 1).unfold(2, kw, 1).permute(0, 3, 4, 5, 1, 2).reshape(x.shape[0], x.shape[1], kh * kw, -1))])
    x = transforms_list(x)

    return x


def col2im(x, h, w):
    fold = nn.Unfold(output_size=(h, w), kernel_size=(8, 8))
    transforms_list = transforms.Compose([
        transforms.Lambda(lambda x: torch.unsqueeze(x, dim=0)),
        transforms.Lambda(lambda x: fold(x)),
        transforms.Lambda(lambda x: torch.transpose(x, 1, 2))])
    x = transforms_list(x)

    return x


def avg_col2im(im, h, w):
    # im: [kh*kw, -1]

    def im2col_single(x, kh=8, kw=8):
        # x: [h, w]
        unfold = nn.Unfold(kernel_size=[kh, kw], padding=[0, 0], stride=[1, 1])
        transforms_list = transforms.Compose([transforms.Lambda(lambda x: torch.transpose(x, 1, 2)),
                                              transforms.Lambda(lambda x: torch.unsqueeze(x, dim=0)),
                                              transforms.Lambda(lambda x: unfold(x))])
        x = transforms_list(x)

        return x

    # index
    device = im.device
    index = torch.unsqueeze(torch.reshape(torch.arange(1, h * w + 1).double(), (h, w)), dim=0).to(device)
    index = im2col_single(index).long().flatten()
    im = im.reshape([im.shape[0], -1])

    # inefficient version
    # out = torch.zeros(h * w)
    # for i in range(1, 1 + h * w):
    #     out[i - 1] = torch.mean(im[(t == i).nonzero().flatten()])
    # out = torch.reshape(out, [h, w])

    # efficient version
    out = scatter_mean(src=im, index=index)
    out = torch.reshape(out[:, 1:], [im.shape[0], h, w])

    return out
