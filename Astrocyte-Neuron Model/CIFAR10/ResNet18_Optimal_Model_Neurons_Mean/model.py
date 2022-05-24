import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import SNN_C, SNN_F, Convolution1, Convolution2, Convolution3, Convolution4, Convolution5, Convolution6, Convolution7, Convolution8, Convolution9, Convolution10\
    , Convolution11, Convolution12, Convolution13, Convolution14, Convolution15, Convolution16, Convolution17, Fully_Connect


class ResNet18(nn.Module):
    def __init__(self, in_channel=3, c_in0=64, c_in1=128, c_in2=256, c_in3=512, f_in=512, num_classes=10):
        super(ResNet18, self).__init__()
        self.c_in0 = c_in0
        self.c_in1 = c_in1
        self.c_in2 = c_in2
        self.c_in3 = c_in3
        self.f_in = f_in
        self.c_layer1 = Convolution1(in_channel, c_in0)
        self.c_layer2 = Convolution2(c_in0, c_in0)
        self.c_layer3 = Convolution3(c_in0, c_in0)
        self.c_layer4 = Convolution4(c_in0, c_in0)
        self.c_layer5 = Convolution5(c_in0, c_in0)
        self.c_layer6 = Convolution6(c_in0, c_in1)
        self.c_layer7 = Convolution7(c_in1, c_in1)
        self.c_layer8 = Convolution8(c_in1, c_in1)
        self.c_layer9 = Convolution9(c_in1, c_in1)
        self.c_layer10 = Convolution10(c_in1, c_in2)
        self.c_layer11 = Convolution11(c_in2, c_in2)
        self.c_layer12 = Convolution12(c_in2, c_in2)
        self.c_layer13 = Convolution13(c_in2, c_in2)
        self.c_layer14 = Convolution14(c_in2, c_in3)
        self.c_layer15 = Convolution15(c_in3, c_in3)
        self.c_layer16 = Convolution16(c_in3, c_in3)
        self.c_layer17 = Convolution17(c_in3, c_in3)
        self.f_layer18 = Fully_Connect(f_in, num_classes)
        self.snn_c = SNN_C()
        self.snn_f = SNN_F()

    def forward(self, x, epoch):
        if epoch < 4:
            # 1~17
            out = self.c_layer1(x, epoch, 0, 1)
            out, out0 = self.c_layer2(out, epoch, 0, 1)
            out = self.c_layer3(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer4(out, epoch, 0, 1)
            out = self.c_layer5(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer6(out, epoch, 0, 0, 1)
            out = self.c_layer7(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer8(out, epoch, 0, 1)
            out = self.c_layer9(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer10(out, epoch, 0, 0, 1)
            out = self.c_layer11(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer12(out, epoch, 0, 1)
            out = self.c_layer13(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer14(out, epoch, 0, 0, 1)
            out = self.c_layer15(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer16(out, epoch, 0, 1)
            out = self.c_layer17(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out = F.avg_pool2d(out, 4)
            o = out.view(out.size(0), -1)
            # 18
            out = self.f_layer18(o, epoch, 0, 1)
            c_gate1 = torch.zeros(self.c_in0, 1).cuda()
            c_gate2 = torch.zeros(self.c_in0, 1).cuda()
            c_gate3 = torch.zeros(self.c_in0, 1).cuda()
            c_gate4 = torch.zeros(self.c_in0, 1).cuda()
            c_gate5 = torch.zeros(self.c_in0, 1).cuda()
            c_gate6 = torch.zeros(self.c_in1, 1).cuda()
            c_gate7 = torch.zeros(self.c_in1, 1).cuda()
            c_gate8 = torch.zeros(self.c_in1, 1).cuda()
            c_gate9 = torch.zeros(self.c_in1, 1).cuda()
            c_gate10 = torch.zeros(self.c_in2, 1).cuda()
            c_gate11 = torch.zeros(self.c_in2, 1).cuda()
            c_gate12 = torch.zeros(self.c_in2, 1).cuda()
            c_gate13 = torch.zeros(self.c_in2, 1).cuda()
            c_gate14 = torch.zeros(self.c_in3, 1).cuda()
            c_gate15 = torch.zeros(self.c_in3, 1).cuda()
            c_gate16 = torch.zeros(self.c_in3, 1).cuda()
            c_gate17 = torch.zeros(self.c_in3, 1).cuda()
            f_gate18 = torch.zeros(self.f_in, 1).cuda()
        elif epoch >= 4:
            data1 = self.c_layer1(x, epoch, 0, 0)
            data2 = self.c_layer2(x, epoch, 0, 0)
            data3 = self.c_layer3(x, epoch, 0, 0)
            data4 = self.c_layer4(x, epoch, 0, 0)
            data5 = self.c_layer5(x, epoch, 0, 0)
            data6, shortcut_data6 = self.c_layer6(x, epoch, 0, 0, 0)
            data7 = self.c_layer7(x, epoch, 0, 0)
            data8 = self.c_layer8(x, epoch, 0, 0)
            data9 = self.c_layer9(x, epoch, 0, 0)
            data10, shortcut_data10 = self.c_layer10(x, epoch, 0, 0, 0)
            data11 = self.c_layer11(x, epoch, 0, 0)
            data12 = self.c_layer12(x, epoch, 0, 0)
            data13 = self.c_layer13(x, epoch, 0, 0)
            data14, shortcut_data14 = self.c_layer14(x, epoch, 0, 0, 0)
            data15 = self.c_layer15(x, epoch, 0, 0)
            data16 = self.c_layer16(x, epoch, 0, 0)
            data17 = self.c_layer17(x, epoch, 0, 0)
            c_weights = torch.cat((data1, data2, data3, data4, data5, data6, shortcut_data6, data7, data8, data9, data10, shortcut_data10,
                                   data11, data12, data13, data14, shortcut_data14, data15, data16, data17), dim=0).view(1, 1, -1, 64)  # [1, 1, 75, 64]
            c_gate = self.snn_c(c_weights, epoch)
            c_gate1 = c_gate[0:64]
            c_gate2 = c_gate[64:128]
            c_gate3 = c_gate[128:192]
            c_gate4 = c_gate[192:256]
            c_gate5 = c_gate[256:320]
            c_gate6 = c_gate[320:448]
            c_shortcut_gate6 = c_gate[448:576]
            c_gate7 = c_gate[576:704]
            c_gate8 = c_gate[704:832]
            c_gate9 = c_gate[832:960]
            c_gate10 = c_gate[960:1216]
            c_shortcut_gate10 = c_gate[1216:1472]
            c_gate11 = c_gate[1472:1728]
            c_gate12 = c_gate[1728:1984]
            c_gate13 = c_gate[1984:2240]
            c_gate14 = c_gate[2240:2752]
            c_shortcut_gate14 = c_gate[2752:3264]
            c_gate15 = c_gate[3264:3776]
            c_gate16 = c_gate[3776:4288]
            c_gate17 = c_gate[4288:4800]
            data18 = self.f_layer18(x, epoch, 0, 0)
            f_weights = data18.view(1, 1, -1, 16)  # [1, 1, 32, 16]
            f_gate18 = self.snn_f(f_weights, epoch)
            # 1~17
            out = self.c_layer1(x, epoch, c_gate1, 1)
            out, out0 = self.c_layer2(out, epoch, c_gate2, 1)
            out = self.c_layer3(out, epoch, c_gate3, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer4(out, epoch, c_gate4, 1)
            out = self.c_layer5(out, epoch, c_gate5, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer6(out, epoch, c_gate6, c_shortcut_gate6, 1)
            out = self.c_layer7(out, epoch, c_gate7, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer8(out, epoch, c_gate8, 1)
            out = self.c_layer9(out, epoch, c_gate9, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer10(out, epoch, c_gate10, c_shortcut_gate10, 1)
            out = self.c_layer11(out, epoch, c_gate11, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer12(out, epoch, c_gate12, 1)
            out = self.c_layer13(out, epoch, c_gate13, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer14(out, epoch, c_gate14, c_shortcut_gate14, 1)
            out = self.c_layer15(out, epoch, c_gate15, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer16(out, epoch, c_gate16, 1)
            out = self.c_layer17(out, epoch, c_gate17, 1)
            out += out0
            out = F.relu(out)
            out = F.avg_pool2d(out, 4)
            o = out.view(out.size(0), -1)
            # 18
            out = self.f_layer18(o, epoch, f_gate18, 1)
        return out, c_gate1, c_gate2, c_gate3, c_gate4, c_gate5, c_gate6, c_gate7, c_gate8, c_gate9, c_gate10, c_gate11, c_gate12, c_gate13, c_gate14, c_gate15, c_gate16, c_gate17, f_gate18