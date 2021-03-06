import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import SNN_C, SNN_F, Convolution0, Convolution1, Convolution2, Convolution3, Convolution4, Convolution5, Convolution6, Convolution7, Convolution8, Convolution9, \
    Convolution10, Convolution11, Convolution12, Convolution13, Convolution14, Convolution15, Convolution16, Convolution17, Convolution18, Convolution19, Convolution20, \
    Convolution21, Convolution22, Convolution23, Convolution24, Convolution25, Convolution26, Convolution27, Convolution28, Convolution29, Convolution30, Convolution31, \
    Convolution32, Fully_Connect


class ResNet34(nn.Module):
    def __init__(self, in_channel=3, c_in0=64, c_in1=128, c_in2=256, c_in3=512, f_in=512, num_classes=100):
        super(ResNet34, self).__init__()
        self.c_in0 = c_in0
        self.c_in1 = c_in1
        self.c_in2 = c_in2
        self.c_in3 = c_in3
        self.f_in = f_in
        self.c_layer0 = Convolution0(in_channel, c_in0)
        self.c_layer1 = Convolution1(c_in0, c_in0)
        self.c_layer2 = Convolution2(c_in0, c_in0)
        self.c_layer3 = Convolution3(c_in0, c_in0)
        self.c_layer4 = Convolution4(c_in0, c_in0)
        self.c_layer5 = Convolution5(c_in0, c_in0)
        self.c_layer6 = Convolution6(c_in0, c_in0)
        self.c_layer7 = Convolution7(c_in0, c_in1)
        self.c_layer8 = Convolution8(c_in1, c_in1)
        self.c_layer9 = Convolution9(c_in1, c_in1)
        self.c_layer10 = Convolution10(c_in1, c_in1)
        self.c_layer11 = Convolution11(c_in1, c_in1)
        self.c_layer12 = Convolution12(c_in1, c_in1)
        self.c_layer13 = Convolution13(c_in1, c_in1)
        self.c_layer14 = Convolution14(c_in1, c_in1)
        self.c_layer15 = Convolution15(c_in1, c_in2)
        self.c_layer16 = Convolution16(c_in2, c_in2)
        self.c_layer17 = Convolution17(c_in2, c_in2)
        self.c_layer18 = Convolution18(c_in2, c_in2)
        self.c_layer19 = Convolution19(c_in2, c_in2)
        self.c_layer20 = Convolution20(c_in2, c_in2)
        self.c_layer21 = Convolution21(c_in2, c_in2)
        self.c_layer22 = Convolution22(c_in2, c_in2)
        self.c_layer23 = Convolution23(c_in2, c_in2)
        self.c_layer24 = Convolution24(c_in2, c_in2)
        self.c_layer25 = Convolution25(c_in2, c_in2)
        self.c_layer26 = Convolution26(c_in2, c_in2)
        self.c_layer27 = Convolution27(c_in2, c_in3)
        self.c_layer28 = Convolution28(c_in3, c_in3)
        self.c_layer29 = Convolution29(c_in3, c_in3)
        self.c_layer30 = Convolution30(c_in3, c_in3)
        self.c_layer31 = Convolution31(c_in3, c_in3)
        self.c_layer32 = Convolution32(c_in3, c_in3)
        self.f_layer33 = Fully_Connect(f_in, num_classes)
        self.snn_c = SNN_C()
        self.snn_f = SNN_F()

    def forward(self, x, epoch):
        if epoch < 4:
            # 1~33
            out = self.c_layer0(x, epoch, 0, 1)
            out, out0 = self.c_layer1(out, epoch, 0, 1)
            out = self.c_layer2(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer3(out, epoch, 0, 1)
            out = self.c_layer4(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer5(out, epoch, 0, 1)
            out = self.c_layer6(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer7(out, epoch, 0, 0, 1)
            out = self.c_layer8(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer9(out, epoch, 0, 1)
            out = self.c_layer10(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer11(out, epoch, 0, 1)
            out = self.c_layer12(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer13(out, epoch, 0, 1)
            out = self.c_layer14(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer15(out, epoch, 0, 0, 1)
            out = self.c_layer16(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer17(out, epoch, 0, 1)
            out = self.c_layer18(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer19(out, epoch, 0, 1)
            out = self.c_layer20(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer21(out, epoch, 0, 1)
            out = self.c_layer22(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer23(out, epoch, 0, 1)
            out = self.c_layer24(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer25(out, epoch, 0, 1)
            out = self.c_layer26(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer27(out, epoch, 0, 0, 1)
            out = self.c_layer28(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer29(out, epoch, 0, 1)
            out = self.c_layer30(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer31(out, epoch, 0, 1)
            out = self.c_layer32(out, epoch, 0, 1)
            out += out0
            out = F.relu(out)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            o = out.view(out.size(0), -1)
            # 34
            out = self.f_layer33(o, epoch, 0, 1)
            c_gate0 = torch.zeros(self.c_in0, 1).cuda()
            c_gate1 = torch.zeros(self.c_in0, 1).cuda()
            c_gate2 = torch.zeros(self.c_in0, 1).cuda()
            c_gate3 = torch.zeros(self.c_in0, 1).cuda()
            c_gate4 = torch.zeros(self.c_in0, 1).cuda()
            c_gate5 = torch.zeros(self.c_in0, 1).cuda()
            c_gate6 = torch.zeros(self.c_in0, 1).cuda()
            c_gate7 = torch.zeros(self.c_in1, 1).cuda()
            c_gate8 = torch.zeros(self.c_in1, 1).cuda()
            c_gate9 = torch.zeros(self.c_in1, 1).cuda()
            c_gate10 = torch.zeros(self.c_in1, 1).cuda()
            c_gate11 = torch.zeros(self.c_in1, 1).cuda()
            c_gate12 = torch.zeros(self.c_in1, 1).cuda()
            c_gate13 = torch.zeros(self.c_in1, 1).cuda()
            c_gate14 = torch.zeros(self.c_in1, 1).cuda()
            c_gate15 = torch.zeros(self.c_in2, 1).cuda()
            c_gate16 = torch.zeros(self.c_in2, 1).cuda()
            c_gate17 = torch.zeros(self.c_in2, 1).cuda()
            c_gate18 = torch.zeros(self.c_in2, 1).cuda()
            c_gate19 = torch.zeros(self.c_in2, 1).cuda()
            c_gate20 = torch.zeros(self.c_in2, 1).cuda()
            c_gate21 = torch.zeros(self.c_in2, 1).cuda()
            c_gate22 = torch.zeros(self.c_in2, 1).cuda()
            c_gate23 = torch.zeros(self.c_in2, 1).cuda()
            c_gate24 = torch.zeros(self.c_in2, 1).cuda()
            c_gate25 = torch.zeros(self.c_in2, 1).cuda()
            c_gate26 = torch.zeros(self.c_in2, 1).cuda()
            c_gate27 = torch.zeros(self.c_in3, 1).cuda()
            c_gate28 = torch.zeros(self.c_in3, 1).cuda()
            c_gate29 = torch.zeros(self.c_in3, 1).cuda()
            c_gate30 = torch.zeros(self.c_in3, 1).cuda()
            c_gate31 = torch.zeros(self.c_in3, 1).cuda()
            c_gate32 = torch.zeros(self.c_in3, 1).cuda()
            f_gate33 = torch.zeros(self.f_in, 1).cuda()
        elif epoch >= 4:
            data0 = self.c_layer0(x, epoch, 0, 0)
            data1 = self.c_layer1(x, epoch, 0, 0)
            data2 = self.c_layer2(x, epoch, 0, 0)
            data3 = self.c_layer3(x, epoch, 0, 0)
            data4 = self.c_layer4(x, epoch, 0, 0)
            data5 = self.c_layer5(x, epoch, 0, 0)
            data6 = self.c_layer6(x, epoch, 0, 0)
            data7, shortcut_data7 = self.c_layer7(x, epoch, 0, 0, 0)
            data8 = self.c_layer8(x, epoch, 0, 0)
            data9 = self.c_layer9(x, epoch, 0, 0)
            data10 = self.c_layer10(x, epoch, 0, 0)
            data11 = self.c_layer11(x, epoch, 0, 0)
            data12 = self.c_layer12(x, epoch, 0, 0)
            data13 = self.c_layer13(x, epoch, 0, 0)
            data14 = self.c_layer14(x, epoch, 0, 0)
            data15, shortcut_data15 = self.c_layer15(x, epoch, 0, 0, 0)
            data16 = self.c_layer16(x, epoch, 0, 0)
            data17 = self.c_layer17(x, epoch, 0, 0)
            data18 = self.c_layer18(x, epoch, 0, 0)
            data19 = self.c_layer19(x, epoch, 0, 0)
            data20 = self.c_layer20(x, epoch, 0, 0)
            data21 = self.c_layer21(x, epoch, 0, 0)
            data22 = self.c_layer22(x, epoch, 0, 0)
            data23 = self.c_layer23(x, epoch, 0, 0)
            data24 = self.c_layer24(x, epoch, 0, 0)
            data25 = self.c_layer25(x, epoch, 0, 0)
            data26 = self.c_layer26(x, epoch, 0, 0)
            data27, shortcut_data27 = self.c_layer27(x, epoch, 0, 0, 0)
            data28 = self.c_layer28(x, epoch, 0, 0)
            data29 = self.c_layer29(x, epoch, 0, 0)
            data30 = self.c_layer30(x, epoch, 0, 0)
            data31 = self.c_layer31(x, epoch, 0, 0)
            data32 = self.c_layer32(x, epoch, 0, 0)
            c_weights = torch.cat((data0, data1, data2, data3, data4, data5, data6, data7, shortcut_data7, data8, data9, data10, data11, data12, data13, data14,
                                   data15, shortcut_data15, data16, data17, data18, data19, data20, data21, data22, data23, data24, data25, data26, data27, shortcut_data27,
                                   data28, data29, data30, data31, data32), dim=0).view(1, 1, -1, 64)  # [1, 1, 133, 64]
            c_gate = self.snn_c(c_weights, epoch)
            c_gate0 = c_gate[0:self.c_in0]
            c_gate1 = c_gate[self.c_in0:2*self.c_in0]
            c_gate2 = c_gate[2*self.c_in0:3*self.c_in0]
            c_gate3 = c_gate[3*self.c_in0:4*self.c_in0]
            c_gate4 = c_gate[4*self.c_in0:5*self.c_in0]
            c_gate5 = c_gate[5*self.c_in0:6*self.c_in0]
            c_gate6 = c_gate[6*self.c_in0:7*self.c_in0]
            c_gate7 = c_gate[7*self.c_in0:7*self.c_in0+self.c_in1]
            c_shortcut_gate7 = c_gate[7*self.c_in0+self.c_in1:7*self.c_in0+2*self.c_in1]
            c_gate8 = c_gate[7*self.c_in0+2*self.c_in1:7*self.c_in0+3*self.c_in1]
            c_gate9 = c_gate[7*self.c_in0+3*self.c_in1:7*self.c_in0+4*self.c_in1]
            c_gate10 = c_gate[7*self.c_in0+4*self.c_in1:7*self.c_in0+5*self.c_in1]
            c_gate11 = c_gate[7*self.c_in0+5*self.c_in1:7*self.c_in0+6*self.c_in1]
            c_gate12 = c_gate[7*self.c_in0+6*self.c_in1:7*self.c_in0+7*self.c_in1]
            c_gate13 = c_gate[7*self.c_in0+7*self.c_in1:7*self.c_in0+8*self.c_in1]
            c_gate14 = c_gate[7*self.c_in0+8*self.c_in1:7*self.c_in0+9*self.c_in1]
            c_gate15 = c_gate[7*self.c_in0+9*self.c_in1:7*self.c_in0+9*self.c_in1+self.c_in2]
            c_shortcut_gate15 = c_gate[7*self.c_in0+9*self.c_in1+self.c_in2:7*self.c_in0+9*self.c_in1+2*self.c_in2]
            c_gate16 = c_gate[7*self.c_in0+9*self.c_in1+2*self.c_in2:7*self.c_in0+9*self.c_in1+3*self.c_in2]
            c_gate17 = c_gate[7*self.c_in0+9*self.c_in1+3*self.c_in2:7*self.c_in0+9*self.c_in1+4*self.c_in2]
            c_gate18 = c_gate[7*self.c_in0+9*self.c_in1+4*self.c_in2:7*self.c_in0+9*self.c_in1+5*self.c_in2]
            c_gate19 = c_gate[7*self.c_in0+9*self.c_in1+5*self.c_in2:7*self.c_in0+9*self.c_in1+6*self.c_in2]
            c_gate20 = c_gate[7*self.c_in0+9*self.c_in1+6*self.c_in2:7*self.c_in0+9*self.c_in1+7*self.c_in2]
            c_gate21 = c_gate[7*self.c_in0+9*self.c_in1+7*self.c_in2:7*self.c_in0+9*self.c_in1+8*self.c_in2]
            c_gate22 = c_gate[7*self.c_in0+9*self.c_in1+8*self.c_in2:7*self.c_in0+9*self.c_in1+9*self.c_in2]
            c_gate23 = c_gate[7*self.c_in0+9*self.c_in1+9*self.c_in2:7*self.c_in0+9*self.c_in1+10*self.c_in2]
            c_gate24 = c_gate[7*self.c_in0+9*self.c_in1+10*self.c_in2:7*self.c_in0+9*self.c_in1+11*self.c_in2]
            c_gate25 = c_gate[7*self.c_in0+9*self.c_in1+11*self.c_in2:7*self.c_in0+9*self.c_in1+12*self.c_in2]
            c_gate26 = c_gate[7*self.c_in0+9*self.c_in1+12*self.c_in2:7*self.c_in0+9*self.c_in1+13*self.c_in2]
            c_gate27 = c_gate[7*self.c_in0+9*self.c_in1+13*self.c_in2:7*self.c_in0+9*self.c_in1+13*self.c_in2+self.c_in3]
            c_shortcut_gate27 = c_gate[7*self.c_in0+9*self.c_in1+13*self.c_in2+self.c_in3:7*self.c_in0+9*self.c_in1+13*self.c_in2+2*self.c_in3]
            c_gate28 = c_gate[7*self.c_in0+9*self.c_in1+13*self.c_in2+2*self.c_in3:7*self.c_in0+9*self.c_in1+13*self.c_in2+3*self.c_in3]
            c_gate29 = c_gate[7*self.c_in0+9*self.c_in1+13*self.c_in2+3*self.c_in3:7*self.c_in0+9*self.c_in1+13*self.c_in2+4*self.c_in3]
            c_gate30 = c_gate[7*self.c_in0+9*self.c_in1+13*self.c_in2+4*self.c_in3:7*self.c_in0+9*self.c_in1+13*self.c_in2+5*self.c_in3]
            c_gate31 = c_gate[7*self.c_in0+9*self.c_in1+13*self.c_in2+5*self.c_in3:7*self.c_in0+9*self.c_in1+13*self.c_in2+6*self.c_in3]
            c_gate32 = c_gate[7*self.c_in0+9*self.c_in1+13*self.c_in2+6*self.c_in3:7*self.c_in0+9*self.c_in1+13*self.c_in2+7*self.c_in3]
            data33 = self.f_layer33(x, epoch, 0, 0)
            f_weights = data33.view(1, 1, -1, 16)  # [1, 1, 32, 16]
            f_gate33 = self.snn_f(f_weights, epoch)
            # 1~33
            out = self.c_layer0(x, epoch, c_gate0, 1)
            out, out0 = self.c_layer1(out, epoch, c_gate1, 1)
            out = self.c_layer2(out, epoch, c_gate2, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer3(out, epoch, c_gate3, 1)
            out = self.c_layer4(out, epoch, c_gate4, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer5(out, epoch, c_gate5, 1)
            out = self.c_layer6(out, epoch, c_gate6, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer7(out, epoch, c_gate7, c_shortcut_gate7, 1)
            out = self.c_layer8(out, epoch, c_gate8, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer9(out, epoch, c_gate9, 1)
            out = self.c_layer10(out, epoch, c_gate10, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer11(out, epoch, c_gate11, 1)
            out = self.c_layer12(out, epoch, c_gate12, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer13(out, epoch, c_gate13, 1)
            out = self.c_layer14(out, epoch, c_gate14, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer15(out, epoch, c_gate15, c_shortcut_gate15, 1)
            out = self.c_layer16(out, epoch, c_gate16, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer17(out, epoch, c_gate17, 1)
            out = self.c_layer18(out, epoch, c_gate18, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer19(out, epoch, c_gate19, 1)
            out = self.c_layer20(out, epoch, c_gate20, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer21(out, epoch, c_gate21, 1)
            out = self.c_layer22(out, epoch, c_gate22, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer23(out, epoch, c_gate23, 1)
            out = self.c_layer24(out, epoch, c_gate24, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer25(out, epoch, c_gate25, 1)
            out = self.c_layer26(out, epoch, c_gate26, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer27(out, epoch, c_gate27, c_shortcut_gate27, 1)
            out = self.c_layer28(out, epoch, c_gate28, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer29(out, epoch, c_gate29, 1)
            out = self.c_layer30(out, epoch, c_gate30, 1)
            out += out0
            out = F.relu(out)
            out, out0 = self.c_layer31(out, epoch, c_gate31, 1)
            out = self.c_layer32(out, epoch, c_gate32, 1)
            out += out0
            out = F.relu(out)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            o = out.view(out.size(0), -1)
            # 34
            out = self.f_layer33(o, epoch, f_gate33, 1)
        return out, c_gate0, c_gate1, c_gate2, c_gate3, c_gate4, c_gate5, c_gate6, c_gate7, c_gate8, c_gate9, c_gate10, c_gate11, c_gate12, c_gate13, c_gate14, c_gate15, c_gate16, \
               c_gate17, c_gate18, c_gate19, c_gate20, c_gate21, c_gate22, c_gate23, c_gate24, c_gate25, c_gate26, c_gate27, c_gate28, c_gate29, c_gate30, c_gate31, c_gate32, f_gate33