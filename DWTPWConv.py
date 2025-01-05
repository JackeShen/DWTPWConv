import torch.nn as nn
import pywt
from functools import partial
import pywt.data
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from .DWconv import DepthwiseSeparableConvWithWTConv2d

def create_wavelet_filter(wave, in_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    return dec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x






class DWTConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_=3, stride=1, wt_type='db1', wt_levels=1, bias=True,p = None,d = 1):
        super(DWTConv, self).__init__()

        # assert in_channels == out_channels

        self.in_channels = in_channels
        self.kernel_size = kernel_
        self.stride = stride
        #self.wt_levels = wt_levels

        self.wt_filters = nn.Parameter(create_wavelet_filter(wt_type, in_channels, type=torch.float),
                                       requires_grad=False)

        self.wt_trans = partial(wavelet_transform, filters=self.wt_filters)

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=self.kernel_size, stride=self.stride,padding=autopad(3,p,d))

    def forward(self, x):
        # print("x_shape_first:",x.shape)
        x_WT = self.wt_trans(x)
        x_WT_ll = x_WT[:, :, 0, :, :]
        x_WT_lh = x_WT[:, :, 1, :, :]
        x_WT_hl = x_WT[:, :, 2, :, :]
        x_WT_hh = x_WT[:, :, 3, :, :]
        x_WT_Conv = self.conv(x_WT_ll)


        #print("s :" ,stride)

        # print("x_WT_Conv_shape:",x_WT_Conv.shape)
        # print("x_WT_ll_shape:",x_WT_ll.shape)
        # print("x_WT_lh_shape:",x_WT_lh.shape)
        # print("x_WT_hl_shape:",x_WT_hl.shape)
        # print("x_WT_hh_shape:",x_WT_hh.shape)

        x = torch.cat([x_WT_Conv, x_WT_lh, x_WT_hl, x_WT_hh], dim=1)
        # print('x_shape:',x.shape)
        return x


class DWTPWConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()

        self.DWTconv = DWTConv(c1, 4*c1,3,1,wt_type='db1')
        self.PWconv = nn.Conv2d(4*c1, c2, 1, 1)
        #1*1卷积 降低通道数
        #self.conv_D_Channle = nn.Conv2d(c_,c2, kernel_size=1)

        #基于小波卷积的深度可分离卷积->降低通道数
        # self.conv_D_Channle = DepthwiseSeparableConvWithWTConv2d(c_, c2)

        self.DWTbn = nn.BatchNorm2d(c1*4)
        self.PWbn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()


    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        # print('x_shape1:',x.shape)
        x_ = self.act(self.DWTbn((self.DWTconv(x))))
        # print('x_shape2:',x_.shape)
        x = self.act(self.PWbn((self.PWconv(x_))))
        # print('x_shape3:',x.shape)
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p




if __name__ == "__main__":
    image_size = (1, 3, 640, 640)
    image = torch.rand(*image_size)

    mobilenet_v1 = DWTPWConv(3, 3)

    out = mobilenet_v1(image)
    print(out.size())

#    image = Image.open('D:/Vscode_YOLO/datasets/coco128/images/train2017/000000000025.jpg')
#    transform = transforms.Compose([transforms.ToTensor()])
#    image_tensor = transform(image)
#    print(image_tensor.shape)
#    filters = create_wavelet_filter('db1', 3, type=torch.float)
#    x = image_tensor.unsqueeze(0)
#    print(x.shape)
#    x_WT = wavelet_transform(x, filters)
#    print(x_WT.shape)

#    x_WT_ll = x_WT[:, :, 0, :, :]
#    x_WT_lh = x_WT[:, :, 1, :, :]

#    x_WT_ll = (x_WT_ll - x_WT_ll.min()) / (x_WT_ll.max() - x_WT_ll.min())
#    x_WT_lh = (x_WT_lh - x_WT_lh.min()) / (x_WT_lh.max() - x_WT_lh.min())
#    image = transforms.ToPILImage()(x_WT_ll[0])
#    image_lh = transforms.ToPILImage()(x_WT_lh[0])
#    image_lh.show()