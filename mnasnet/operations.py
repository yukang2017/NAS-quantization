import torch
import torch.nn as nn

'''
• ConvOP [0,1,2,3] conv/0, sep-conv/1, mobile-ib-conv-3/2, mobile-ib-conv-6/3.
• KernelSize [0,1] 3x3/0, 5x5/1.
• SkipOp [0,1,2,3] max/0, avg/1, id/2, no/3.
• Filters  Fi ...  omit currently.
• Layers [0,1,2,3] 1/0, 2/1, 3/2, 4/3.
• Quantz [0,1,2]   4/0, 8/1, 16/2.

[ConvOP, KernelSize, SkipOp, Layers, Quantz]
'''

'''
OPS = {
    'conv': lambda kernel_size, skip_op, num_layers: Conv(kernel_size=kernel_size, skip_op=skip_op, num_layers=num_layers),
    'sep_conv': lambda kernel_size, skip_op, num_layers: SepConv(kernel_size=kernel_size, skip_op=skip_op, num_layers=num_layers),
    'mib_conv': lambda kernel_size, skip_op, num_layers: MobInvBConv(kernel_size=kernel_size, skip_op=skip_op, num_layers=num_layers),
}
'''

MBConv3 = lambda C_in, C_out, kernel_size, stride, padding, affine: MBConv(C_in, C_out, kernel_size, stride, padding, t=3, affine=affine)
MBConv6 = lambda C_in, C_out, kernel_size, stride, padding, affine: MBConv(C_in, C_out, kernel_size, stride, padding, t=6, affine=affine)


class Skip(nn.Module):
    def __init__(self, C_in, C_out, skip_op, kernel_size, stride, padding):
        super(Skip, self).__init__()
        if stride>1 or not C_in==C_out:
            name = ''#'Skip-'
            skip_conv = nn.Sequential()
            skip_conv.add_module(name+'conv1',nn.Conv2d(C_in, C_out, kernel_size=1, stride=stride, padding=0, groups=1, bias=False))
            skip_conv.add_module(name+'bn1',nn.BatchNorm2d(C_out, affine=True))
            stride = 1
            padding=int((kernel_size-1)/2)

        if skip_op==0:
            self.op=nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        elif skip_op==1:
            self.op=nn.AvgPool2d(kernel_size, stride=stride, padding=padding, count_include_pad=False)
        elif skip_op==2:
            self.op=Identity()
        elif skip_op==3:
            self.op=Zero(stride)
        else:
            raise ValueError('Wrong skip_op {}'.format(skip_op))

        if stride>1 or not C_in==C_out:
            self.op=nn.Sequential(skip_conv,self.op)

    def forward(self,x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        name = ''#'com'
        self.op = nn.Sequential()
        self.op.add_module(name+'relu',nn.ReLU(inplace=False))
        self.op.add_module(name+'conv',nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False))
        self.op.add_module(name+'bn',nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        name = ''#'Sep-'
        self.op = nn.Sequential()
        self.op.add_module(name+'relu1',nn.ReLU(inplace=False))
        self.op.add_module(name+'conv1',nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False))
        self.op.add_module(name+'conv2',nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False))
        self.op.add_module(name+'bn1',nn.BatchNorm2d(C_in, affine=affine))
        self.op.add_module(name+'relu2',nn.ReLU(inplace=False))
        self.op.add_module(name+'conv3',nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False))
        self.op.add_module(name+'conv4',nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False))
        self.op.add_module(name+'bn2',nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class MBConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, t=3, affine=True):
        super(MBConv, self).__init__()
        name = ''#'MB-'
        self.op = nn.Sequential()
        self.op.add_module(name+'conv1',nn.Conv2d(C_in, C_in*t, kernel_size=1, stride=1, padding=0, groups=1, bias=False))
        self.op.add_module(name+'bn1',nn.BatchNorm2d(C_in*t, affine=affine))
        self.op.add_module(name+'relu1',nn.ReLU(inplace=False))

        self.op.add_module(name+'conv2',nn.Conv2d(C_in*t, C_in*t, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in*t, bias=False))
        self.op.add_module(name+'bn2',nn.BatchNorm2d(C_in*t, affine=affine))
        self.op.add_module(name+'relu1',nn.ReLU(inplace=False))
        self.op.add_module(name+'conv3',nn.Conv2d(C_in*t, C_out, kernel_size=1, padding=0, groups=1, bias=False))
        self.op.add_module(name+'bn3',nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)
