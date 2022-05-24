import torch
from torch import nn
from layer import SNN_C, SNN_F, Convolution0, Convolution1, Fully_Connect0, Fully_Connect1


class LeNet5(nn.Module):
    def __init__(self, c_in=1, c_dims0=20, c_dims1=50, f_in=800, f_dims0=500, num_classes=10):
        super(LeNet5, self).__init__()
        self.c_in = c_in
        self.c_dims0 = c_dims0
        self.c_dims1 = c_dims1
        self.f_in = f_in
        self.f_dims0 = f_dims0
        self.c_layer0 = Convolution0(self.c_in, self.c_dims0)
        self.c_layer1 = Convolution1(self.c_dims0, self.c_dims1)
        self.f_layer0 = Fully_Connect0(self.f_in, self.f_dims0)
        self.f_layer1 = Fully_Connect1(self.f_dims0, num_classes)
        self.snn_c = SNN_C()
        self.snn_f = SNN_F()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, epoch):
        if epoch < 4:
            # 1
            out = self.c_layer0(x, epoch, 0, 1)
            out = self.maxpool(out)
            c_gate0 = torch.zeros(self.c_dims0, 1).cuda()
            # 2
            out = self.c_layer1(out, epoch, 0, 1)
            out = self.maxpool(out)
            c_gate1 = torch.zeros(self.c_dims1, 1).cuda()
            o = out.view(out.size(0), -1)
            # 3
            out = self.f_layer0(o, epoch, 0, 1)
            f_gate0 = torch.zeros(self.f_in, 1).cuda()
            # 4
            out = self.f_layer1(out, epoch, 0, 1)
            f_gate1 = torch.zeros(self.f_dims0, 1).cuda()
        elif epoch >= 4:
            data0 = self.c_layer0(x, epoch, 0, 0)
            data1 = self.c_layer1(x, epoch, 0, 0)
            c_weights = torch.cat((data0, data1), dim=0).view(1, 1, -1, 7)
            c_gate = self.snn_c(c_weights, epoch)
            c_gate0 = c_gate[0:20]
            c_gate1 = c_gate[20:70]
            data2 = self.f_layer0(x, epoch, 0, 0)
            data3 = self.f_layer1(x, epoch, 0, 0)
            f_weights = torch.cat((data2, data3), dim=0).view(1, 1, -1, 25)
            f_gate = self.snn_f(f_weights, epoch)
            f_gate0 = f_gate[0:800]
            f_gate1 = f_gate[800:1300]
            # 1
            out = self.c_layer0(x, epoch, c_gate0, 1)
            out = self.maxpool(out)
            # 2
            out = self.c_layer1(out, epoch, c_gate1, 1)
            out = self.maxpool(out)
            o = out.view(out.size(0), -1)
            # 3
            out = self.f_layer0(o, epoch, f_gate0, 1)
            # 4
            out = self.f_layer1(out, epoch, f_gate1, 1)
        return out, c_gate0, c_gate1, f_gate0, f_gate1