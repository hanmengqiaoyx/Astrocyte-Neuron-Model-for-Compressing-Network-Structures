import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class SNN_C(nn.Module):
    def __init__(self,):
        super(SNN_C, self).__init__()
        self.dw_weights0 = Parameter(torch.Tensor(8, 1, 3, 3))
        self.dw_bn0 = nn.BatchNorm2d(8)
        self.dw_weights1 = Parameter(torch.Tensor(16, 8, 3, 3))
        self.dw_bn1 = nn.BatchNorm2d(16)
        self.dw_weights2 = Parameter(torch.Tensor(32, 16, 3, 3))
        self.dw_bn2 = nn.BatchNorm2d(32)
        self.up_sample0 = Parameter(torch.Tensor(32, 16, 3, 3))
        self.up_bn00 = nn.BatchNorm2d(16)
        self.up_weights0 = Parameter(torch.Tensor(16, 32, 3, 3))
        self.up_bn01 = nn.BatchNorm2d(16)
        self.up_sample1 = Parameter(torch.Tensor(16, 8, 2, 3))
        self.up_bn10 = nn.BatchNorm2d(8)
        self.up_weights1 = Parameter(torch.Tensor(8, 16, 3, 3))
        self.up_bn11 = nn.BatchNorm2d(8)
        self.gate_weights0 = Parameter(torch.Tensor(1, 8, 3, 3))
        self.gate_bn0 = nn.BatchNorm2d(1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.dw_weights0, a=math.sqrt(5))
        init.kaiming_uniform_(self.dw_weights1, a=math.sqrt(5))
        init.kaiming_uniform_(self.dw_weights2, a=math.sqrt(5))
        init.kaiming_uniform_(self.up_sample0, a=math.sqrt(5))
        init.kaiming_uniform_(self.up_sample1, a=math.sqrt(5))
        init.kaiming_uniform_(self.up_weights0, a=math.sqrt(5))
        init.kaiming_uniform_(self.up_weights1, a=math.sqrt(5))
        init.kaiming_uniform_(self.gate_weights0, a=math.sqrt(5))

    def forward(self, input, epoch):
        if self.training:
            if epoch % 2 == 0:
                self.dw_weights0.requires_grad = True
                self.dw_weights1.requires_grad = True
                self.dw_weights2.requires_grad = True
                self.up_sample0.requires_grad = True
                self.up_sample1.requires_grad = True
                self.up_weights0.requires_grad = True
                self.up_weights1.requires_grad = True
                self.gate_weights0.requires_grad = True
                layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(input, self.dw_weights0, stride=1, padding=1, bias=None)))
                layer01 = self.maxpool(layer00)
                layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_weights1, stride=1, padding=1, bias=None)))
                layer11 = self.maxpool(layer10)
                layer2 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_weights2, stride=1, padding=1, bias=None)))  # [1, 32, 2, 1]
                layer30 = F.relu(self.up_bn00(nn.functional.conv_transpose2d(layer2, self.up_sample0, stride=2, bias=None)))
                layer31 = torch.cat((layer10, layer30), dim=1)
                layer32 = F.relu(self.up_bn01(nn.functional.conv2d(layer31, self.up_weights0, stride=1, padding=1, bias=None)))
                layer40 = F.relu(self.up_bn10(nn.functional.conv_transpose2d(layer32, self.up_sample1, stride=2, bias=None)))
                layer41 = torch.cat((layer00, layer40), dim=1)
                layer42 = F.relu(self.up_bn11(nn.functional.conv2d(layer41, self.up_weights1, stride=1, padding=1, bias=None)))
                layer_out = F.relu(self.gate_bn0(nn.functional.conv2d(layer42, self.gate_weights0, stride=1, padding=1, bias=None)))  # [1, 1, 10, 1]
                gate = layer_out.view(-1)
            elif epoch % 2 != 0:
                self.dw_weights0.requires_grad = False
                self.dw_weights1.requires_grad = False
                self.dw_weights2.requires_grad = False
                self.up_sample0.requires_grad = False
                self.up_sample1.requires_grad = False
                self.up_weights0.requires_grad = False
                self.up_weights1.requires_grad = False
                self.gate_weights0.requires_grad = False
                layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(input, self.dw_weights0, stride=1, padding=1, bias=None)))
                layer01 = self.maxpool(layer00)
                layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_weights1, stride=1, padding=1, bias=None)))
                layer11 = self.maxpool(layer10)
                layer2 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_weights2, stride=1, padding=1, bias=None)))  # [1, 32, 2, 1]
                layer30 = F.relu(self.up_bn00(nn.functional.conv_transpose2d(layer2, self.up_sample0, stride=2, bias=None)))
                layer31 = torch.cat((layer10, layer30), dim=1)
                layer32 = F.relu(self.up_bn01(nn.functional.conv2d(layer31, self.up_weights0, stride=1, padding=1, bias=None)))
                layer40 = F.relu(self.up_bn10(nn.functional.conv_transpose2d(layer32, self.up_sample1, stride=2, bias=None)))
                layer41 = torch.cat((layer00, layer40), dim=1)
                layer42 = F.relu(self.up_bn11(nn.functional.conv2d(layer41, self.up_weights1, stride=1, padding=1, bias=None)))
                layer_out = F.relu(self.gate_bn0(nn.functional.conv2d(layer42, self.gate_weights0, stride=1, padding=1, bias=None)))  # [1, 1, 10, 1]
                gate = layer_out.view(-1)
        elif not self.training:
                layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(input, self.dw_weights0, stride=1, padding=1, bias=None)))
                layer01 = self.maxpool(layer00)
                layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_weights1, stride=1, padding=1, bias=None)))
                layer11 = self.maxpool(layer10)
                layer2 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_weights2, stride=1, padding=1, bias=None)))  # [1, 32, 2, 1]
                layer30 = F.relu(self.up_bn00(nn.functional.conv_transpose2d(layer2, self.up_sample0, stride=2, bias=None)))
                layer31 = torch.cat((layer10, layer30), dim=1)
                layer32 = F.relu(self.up_bn01(nn.functional.conv2d(layer31, self.up_weights0, stride=1, padding=1, bias=None)))
                layer40 = F.relu(self.up_bn10(nn.functional.conv_transpose2d(layer32, self.up_sample1, stride=2, bias=None)))
                layer41 = torch.cat((layer00, layer40), dim=1)
                layer42 = F.relu(self.up_bn11(nn.functional.conv2d(layer41, self.up_weights1, stride=1, padding=1, bias=None)))
                layer_out = F.relu(self.gate_bn0(nn.functional.conv2d(layer42, self.gate_weights0, stride=1, padding=1, bias=None)))  # [1, 1, 10, 1]
                gate = layer_out.view(-1)
        return gate


class SNN_F(nn.Module):
    def __init__(self,):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        """
        super(SNN_F, self).__init__()
        self.dw_weights0 = Parameter(torch.Tensor(8, 1, 3, 3))
        self.dw_bn0 = nn.BatchNorm2d(8)
        self.dw_weights1 = Parameter(torch.Tensor(16, 8, 3, 3))
        self.dw_bn1 = nn.BatchNorm2d(16)
        self.dw_weights2 = Parameter(torch.Tensor(32, 16, 3, 3))
        self.dw_bn2 = nn.BatchNorm2d(32)
        self.dw_weights3 = Parameter(torch.Tensor(64, 32, 3, 3))
        self.dw_bn3 = nn.BatchNorm2d(64)
        self.dw_weights4 = Parameter(torch.Tensor(128, 64, 3, 3))
        self.dw_bn4 = nn.BatchNorm2d(128)
        self.up_sample0 = Parameter(torch.Tensor(128, 64, 2, 3))
        self.up_bn00 = nn.BatchNorm2d(64)
        self.up_weights0 = Parameter(torch.Tensor(64, 128, 3, 3))
        self.up_bn01 = nn.BatchNorm2d(64)
        self.up_sample1 = Parameter(torch.Tensor(64, 32, 3, 2))
        self.up_bn10 = nn.BatchNorm2d(32)
        self.up_weights1 = Parameter(torch.Tensor(32, 64, 3, 3))
        self.up_bn11 = nn.BatchNorm2d(32)
        self.up_sample2 = Parameter(torch.Tensor(32, 16, 2, 2))
        self.up_bn20 = nn.BatchNorm2d(16)
        self.up_weights2 = Parameter(torch.Tensor(16, 32, 3, 3))
        self.up_bn21 = nn.BatchNorm2d(16)
        self.up_sample3 = Parameter(torch.Tensor(16, 8, 2, 3))
        self.up_bn30 = nn.BatchNorm2d(8)
        self.up_weights3 = Parameter(torch.Tensor(8, 16, 3, 3))
        self.up_bn31 = nn.BatchNorm2d(8)
        self.gate_weights0 = Parameter(torch.Tensor(1, 8, 3, 3))
        self.gate_bn0 = nn.BatchNorm2d(1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.dw_weights0, a=math.sqrt(5))
        init.kaiming_uniform_(self.dw_weights1, a=math.sqrt(5))
        init.kaiming_uniform_(self.dw_weights2, a=math.sqrt(5))
        init.kaiming_uniform_(self.dw_weights3, a=math.sqrt(5))
        init.kaiming_uniform_(self.dw_weights4, a=math.sqrt(5))
        init.kaiming_uniform_(self.up_sample0, a=math.sqrt(5))
        init.kaiming_uniform_(self.up_sample1, a=math.sqrt(5))
        init.kaiming_uniform_(self.up_sample2, a=math.sqrt(5))
        init.kaiming_uniform_(self.up_sample3, a=math.sqrt(5))
        init.kaiming_uniform_(self.up_weights0, a=math.sqrt(5))
        init.kaiming_uniform_(self.up_weights1, a=math.sqrt(5))
        init.kaiming_uniform_(self.up_weights2, a=math.sqrt(5))
        init.kaiming_uniform_(self.up_weights3, a=math.sqrt(5))
        init.kaiming_uniform_(self.gate_weights0, a=math.sqrt(5))

    def forward(self, input, epoch):
        if self.training:
            if epoch % 2 == 0:
                self.dw_weights0.requires_grad = True
                self.dw_weights1.requires_grad = True
                self.dw_weights2.requires_grad = True
                self.dw_weights3.requires_grad = True
                self.dw_weights4.requires_grad = True
                self.up_sample0.requires_grad = True
                self.up_sample1.requires_grad = True
                self.up_sample2.requires_grad = True
                self.up_sample3.requires_grad = True
                self.up_weights0.requires_grad = True
                self.up_weights1.requires_grad = True
                self.up_weights2.requires_grad = True
                self.up_weights3.requires_grad = True
                self.gate_weights0.requires_grad = True
                layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(input, self.dw_weights0, stride=1, padding=1, bias=None)))
                layer01 = self.maxpool(layer00)
                layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_weights1, stride=1, padding=1, bias=None)))
                layer11 = self.maxpool(layer10)
                layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_weights2, stride=1, padding=1, bias=None)))
                layer21 = self.maxpool(layer20)
                layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_weights3, stride=1, padding=1, bias=None)))
                layer31 = self.maxpool(layer30)
                layer40 = F.relu(self.dw_bn4(nn.functional.conv2d(layer31, self.dw_weights4, stride=1, padding=1, bias=None)))  # [1, 128, 3, 1]
                layer60 = F.relu(self.up_bn00(nn.functional.conv_transpose2d(layer40, self.up_sample0, stride=2, bias=None)))
                layer61 = torch.cat((layer30, layer60), dim=1)
                layer62 = F.relu(self.up_bn01(nn.functional.conv2d(layer61, self.up_weights0, stride=1, padding=1, bias=None)))
                layer70 = F.relu(self.up_bn10(nn.functional.conv_transpose2d(layer62, self.up_sample1, stride=2, bias=None)))
                layer71 = torch.cat((layer20, layer70), dim=1)
                layer72 = F.relu(self.up_bn11(nn.functional.conv2d(layer71, self.up_weights1, stride=1, padding=1, bias=None)))
                layer80 = F.relu(self.up_bn20(nn.functional.conv_transpose2d(layer72, self.up_sample2, stride=2, bias=None)))
                layer81 = torch.cat((layer10, layer80), dim=1)
                layer82 = F.relu(self.up_bn21(nn.functional.conv2d(layer81, self.up_weights2, stride=1, padding=1, bias=None)))
                layer90 = F.relu(self.up_bn30(nn.functional.conv_transpose2d(layer82, self.up_sample3, stride=2, bias=None)))
                layer91 = torch.cat((layer00, layer90), dim=1)
                layer92 = F.relu(self.up_bn31(nn.functional.conv2d(layer91, self.up_weights3, stride=1, padding=1, bias=None)))
                layer_out = F.relu(self.gate_bn0(nn.functional.conv2d(layer92, self.gate_weights0, stride=1, padding=1, bias=None)))  # [1, 1, 52, 25]
                gate = layer_out.view(-1)
            elif epoch % 2 != 0:
                self.dw_weights0.requires_grad = False
                self.dw_weights1.requires_grad = False
                self.dw_weights2.requires_grad = False
                self.dw_weights3.requires_grad = False
                self.dw_weights4.requires_grad = False
                self.up_sample0.requires_grad = False
                self.up_sample1.requires_grad = False
                self.up_sample2.requires_grad = False
                self.up_sample3.requires_grad = False
                self.up_weights0.requires_grad = False
                self.up_weights1.requires_grad = False
                self.up_weights2.requires_grad = False
                self.up_weights3.requires_grad = False
                self.gate_weights0.requires_grad = False
                layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(input, self.dw_weights0, stride=1, padding=1, bias=None)))
                layer01 = self.maxpool(layer00)
                layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_weights1, stride=1, padding=1, bias=None)))
                layer11 = self.maxpool(layer10)
                layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_weights2, stride=1, padding=1, bias=None)))
                layer21 = self.maxpool(layer20)
                layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_weights3, stride=1, padding=1, bias=None)))
                layer31 = self.maxpool(layer30)
                layer40 = F.relu(self.dw_bn4(nn.functional.conv2d(layer31, self.dw_weights4, stride=1, padding=1, bias=None)))  # [1, 128, 3, 1]
                layer60 = F.relu(self.up_bn00(nn.functional.conv_transpose2d(layer40, self.up_sample0, stride=2, bias=None)))
                layer61 = torch.cat((layer30, layer60), dim=1)
                layer62 = F.relu(self.up_bn01(nn.functional.conv2d(layer61, self.up_weights0, stride=1, padding=1, bias=None)))
                layer70 = F.relu(self.up_bn10(nn.functional.conv_transpose2d(layer62, self.up_sample1, stride=2, bias=None)))
                layer71 = torch.cat((layer20, layer70), dim=1)
                layer72 = F.relu(self.up_bn11(nn.functional.conv2d(layer71, self.up_weights1, stride=1, padding=1, bias=None)))
                layer80 = F.relu(self.up_bn20(nn.functional.conv_transpose2d(layer72, self.up_sample2, stride=2, bias=None)))
                layer81 = torch.cat((layer10, layer80), dim=1)
                layer82 = F.relu(self.up_bn21(nn.functional.conv2d(layer81, self.up_weights2, stride=1, padding=1, bias=None)))
                layer90 = F.relu(self.up_bn30(nn.functional.conv_transpose2d(layer82, self.up_sample3, stride=2, bias=None)))
                layer91 = torch.cat((layer00, layer90), dim=1)
                layer92 = F.relu(self.up_bn31(nn.functional.conv2d(layer91, self.up_weights3, stride=1, padding=1, bias=None)))
                layer_out = F.relu(self.gate_bn0(nn.functional.conv2d(layer92, self.gate_weights0, stride=1, padding=1, bias=None)))  # [1, 1, 52, 25]
                gate = layer_out.view(-1)
        elif not self.training:
            layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(input, self.dw_weights0, stride=1, padding=1, bias=None)))
            layer01 = self.maxpool(layer00)
            layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_weights1, stride=1, padding=1, bias=None)))
            layer11 = self.maxpool(layer10)
            layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_weights2, stride=1, padding=1, bias=None)))
            layer21 = self.maxpool(layer20)
            layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_weights3, stride=1, padding=1, bias=None)))
            layer31 = self.maxpool(layer30)
            layer40 = F.relu(self.dw_bn4(nn.functional.conv2d(layer31, self.dw_weights4, stride=1, padding=1, bias=None)))  # [1, 128, 3, 1]
            layer60 = F.relu(self.up_bn00(nn.functional.conv_transpose2d(layer40, self.up_sample0, stride=2, bias=None)))
            layer61 = torch.cat((layer30, layer60), dim=1)
            layer62 = F.relu(self.up_bn01(nn.functional.conv2d(layer61, self.up_weights0, stride=1, padding=1, bias=None)))
            layer70 = F.relu(self.up_bn10(nn.functional.conv_transpose2d(layer62, self.up_sample1, stride=2, bias=None)))
            layer71 = torch.cat((layer20, layer70), dim=1)
            layer72 = F.relu(self.up_bn11(nn.functional.conv2d(layer71, self.up_weights1, stride=1, padding=1, bias=None)))
            layer80 = F.relu(self.up_bn20(nn.functional.conv_transpose2d(layer72, self.up_sample2, stride=2, bias=None)))
            layer81 = torch.cat((layer10, layer80), dim=1)
            layer82 = F.relu(self.up_bn21(nn.functional.conv2d(layer81, self.up_weights2, stride=1, padding=1, bias=None)))
            layer90 = F.relu(self.up_bn30(nn.functional.conv_transpose2d(layer82, self.up_sample3, stride=2, bias=None)))
            layer91 = torch.cat((layer00, layer90), dim=1)
            layer92 = F.relu(self.up_bn31(nn.functional.conv2d(layer91, self.up_weights3, stride=1, padding=1, bias=None)))
            layer_out = F.relu(self.gate_bn0(nn.functional.conv2d(layer92, self.gate_weights0, stride=1, padding=1, bias=None)))  # [1, 1, 52, 25]
            gate = layer_out.view(-1)
        return gate


class Convolution0(nn.Module):
    def __init__(self, in_features, out_features):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        """
        super(Convolution0, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 5, 5))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def forward(self, input, epoch, gate, i):
        if i == 0:
            data = torch.mean(self.weights.view(self.out_features, self.in_features * 5 * 5), dim=1)
            return data
        else:
            if self.training:
                if epoch < 4:
                    self.weights.requires_grad = True
                    output = F.relu(nn.functional.conv2d(input, self.weights, stride=1, bias=None))
                else:
                    if epoch >= 4 and epoch % 2 == 0:
                        self.weights.requires_grad = False
                        gate = gate.view(self.out_features, 1, 1, 1)
                        new_weights = self.weights.mul(gate)
                        output = F.relu(nn.functional.conv2d(input, new_weights, stride=1, bias=None))
                    elif epoch >= 4 and epoch % 2 != 0:
                        self.weights.requires_grad = True
                        gate = gate.view(self.out_features, 1, 1, 1)
                        new_weights = self.weights.mul(gate)
                        output = F.relu(nn.functional.conv2d(input, new_weights, stride=1, bias=None))
            elif not self.training:
                if epoch < 4:
                    output = F.relu(nn.functional.conv2d(input, self.weights, stride=1, bias=None))
                else:
                    gate = gate.view(self.out_features, 1, 1, 1)
                    new_weights = self.weights.mul(gate)
                    output = F.relu(nn.functional.conv2d(input, new_weights, stride=1, bias=None))
            return output


class Convolution1(nn.Module):
    def __init__(self, in_features, out_features):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        """
        super(Convolution1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 5, 5))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def forward(self, input, epoch, gate, i):
        if i == 0:
            data = torch.mean(self.weights.view(self.out_features, self.in_features * 5 * 5), dim=1)
            return data
        else:
            if self.training:
                if epoch < 4:
                    self.weights.requires_grad = True
                    output = F.relu(nn.functional.conv2d(input, self.weights, stride=1, bias=None))
                else:
                    if epoch >= 4 and epoch % 2 == 0:
                        self.weights.requires_grad = False
                        gate = gate.view(self.out_features, 1, 1, 1)
                        new_weights = self.weights.mul(gate)
                        output = F.relu(nn.functional.conv2d(input, new_weights, stride=1, bias=None))
                    elif epoch >= 4 and epoch % 2 != 0:
                        self.weights.requires_grad = True
                        gate = gate.view(self.out_features, 1, 1, 1)
                        new_weights = self.weights.mul(gate)
                        output = F.relu(nn.functional.conv2d(input, new_weights, stride=1, bias=None))
            elif not self.training:
                if epoch < 4:
                    output = F.relu(nn.functional.conv2d(input, self.weights, stride=1, bias=None))
                else:
                    gate = gate.view(self.out_features, 1, 1, 1)
                    new_weights = self.weights.mul(gate)
                    output = F.relu(nn.functional.conv2d(input, new_weights, stride=1, bias=None))
            return output


class Fully_Connect0(nn.Module):
    def __init__(self, in_features, out_features):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        """
        super(Fully_Connect0, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def forward(self, input, epoch, gate, i):
        if i == 0:
            data = torch.mean(self.weights.view(self.in_features, self.out_features), dim=1)
            return data
        else:
            if self.training:
                if epoch < 4:
                    self.weights.requires_grad = True
                    output = F.relu(input.mm(self.weights))
                else:
                    if epoch >= 4 and epoch % 2 == 0:
                        self.weights.requires_grad = False
                        gate = gate.view(1, self.in_features).expand(input.size(0), self.in_features)
                        new_input = input.mul(gate)
                        output = F.relu(new_input.mm(self.weights))
                    elif epoch >= 4 and epoch % 2 != 0:
                        self.weights.requires_grad = True
                        gate = gate.view(1, self.in_features).expand(input.size(0), self.in_features)
                        new_input = input.mul(gate)
                        output = F.relu(new_input.mm(self.weights))
            elif not self.training:
                if epoch < 4:
                    output = F.relu(input.mm(self.weights))
                else:
                    gate = gate.view(1, self.in_features).expand(input.size(0), self.in_features)
                    new_input = input.mul(gate)
                    output = F.relu(new_input.mm(self.weights))
            return output


class Fully_Connect1(nn.Module):
    def __init__(self, in_features, out_features):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        """
        super(Fully_Connect1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def forward(self, input, epoch, gate, i):
        if i == 0:
            data = torch.mean(self.weights.view(self.in_features, self.out_features), dim=1)
            return data
        else:
            if self.training:
                if epoch < 4:
                    self.weights.requires_grad = True
                    output = input.mm(self.weights)
                else:
                    if epoch >= 4 and epoch % 2 == 0:
                        self.weights.requires_grad = False
                        gate = gate.view(1, self.in_features).expand(input.size(0), self.in_features)
                        new_input = input.mul(gate)
                        output = new_input.mm(self.weights)
                    elif epoch >= 4 and epoch % 2 != 0:
                        self.weights.requires_grad = True
                        gate = gate.view(1, self.in_features).expand(input.size(0), self.in_features)
                        new_input = input.mul(gate)
                        output = new_input.mm(self.weights)
            elif not self.training:
                if epoch < 4:
                    output = input.mm(self.weights)
                else:
                    gate = gate.view(1, self.in_features).expand(input.size(0), self.in_features)
                    new_input = input.mul(gate)
                    output = new_input.mm(self.weights)
            return output