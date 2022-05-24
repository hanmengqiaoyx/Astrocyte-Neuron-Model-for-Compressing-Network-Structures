import torch
from torch import nn
from layer import SNN, Fully_Connect0, Fully_Connect1, Fully_Connect2


class MLP(nn.Module):
    def __init__(self, fc_in=784, fc_dims1=300, fc_dims2=100, class_num=10):
        super(MLP, self).__init__()
        self.fc_in = fc_in
        self.fc_dims1 = fc_dims1
        self.fc_dims2 = fc_dims2
        self.layer0 = Fully_Connect0(self.fc_in, self.fc_dims1)
        self.layer1 = Fully_Connect1(self.fc_dims1, self.fc_dims2)
        self.layer2 = Fully_Connect2(self.fc_dims2, class_num)
        self.snn = SNN()
        self.relu = nn.ReLU()

    def forward(self, x, epoch):
        o = x.view(x.size(0), -1)
        if epoch < 4:
            # 1
            out = self.layer0(o, epoch, 0, 1)
            f_gate0 = torch.zeros(self.fc_in, 1).cuda()
            # 2
            out = self.layer1(out, epoch, 0, 1)
            f_gate1 = torch.zeros(self.fc_dims1, 1).cuda()
            # 3
            out = self.layer2(out, epoch, 0, 1)
            f_gate2 = torch.zeros(self.fc_dims2, 1).cuda()
        elif epoch >= 4:
            data0 = self.layer0(o, epoch, 0, 0)
            data1 = self.layer1(o, epoch, 0, 0)
            data2 = self.layer2(o, epoch, 0, 0)
            weights = torch.cat((data0, data1, data2), dim=0).view(1, 1, -1, 32)  # [1, 1, 37, 32]
            gate = self.snn(weights, epoch)
            f_gate0 = gate[0:self.fc_in]
            f_gate1 = gate[self.fc_in:self.fc_in+self.fc_dims1]
            f_gate2 = gate[self.fc_in+self.fc_dims1:self.fc_in+self.fc_dims1+self.fc_dims2]
            # 1
            out = self.layer0(o, epoch, f_gate0, 1)
            # 2
            out = self.layer1(out, epoch, f_gate1, 1)
            # 3
            out = self.layer2(out, epoch, f_gate2, 1)
        return out, f_gate0, f_gate1, f_gate2