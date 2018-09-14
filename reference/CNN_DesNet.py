import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DenseNet3(nn.Module):
    def __init__(self, device, depth, lr, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0, beta=0.5, tao=10.0, b=2.0):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.learning_rate = lr
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        # self.fc1 = nn.Linear(in_planes, num_classes)
        self.fc1 = nn.Linear(in_planes, 500)
        self.fc2 = nn.Linear(500, 300)
        self.fc3 = nn.Linear(300, 100)
        self.in_planes = in_planes

        # add parameters
        self.ddml_layers = [self.fc1, self.fc2, self.fc3]
        self._s = F.tanh
        self.device = device

        self.beta = beta
        self.tao = tao
        self.b = b

        self.to(device)
        # end adding

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def des_forword(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)

        return out

    def fc_forward(self, x):
        x = self.des_forword(x)
        x = self._s(self.fc1(x))
        x = self._s(self.fc2(x))
        return x
        # return self.fc(out)

    def forward(self, x):
        x = self.fc_forward(x)
        x = self.fc3(x)
        return x

    def compute_distance(self, x1, x2):

        return (self.fc_forward(x1) - self.fc_forward(x2)).data.norm() ** 2

    # pairwise loss
    def pairwise_optimize(self, pairs):
        loss = 0.0
        layer_count = len(self.ddml_layers)

        params_W = []
        params_b = []

        for layer in self.ddml_layers:
            params = list(layer.parameters())

            params_W.append(params[0])
            params_b.append(params[1])

        # calculate z(m) and h(m)
        # z(m) is the output of m-th layer without function tanh(x)
        # h(m) start from 0, which is m-1
        z_i_m = [[0 for m in range(layer_count)] for index in range(len(pairs))]
        h_i_m = [[0 for m in range(layer_count + 1)] for index in range(len(pairs))]
        z_j_m = [[0 for m in range(layer_count)] for index in range(len(pairs))]
        h_j_m = [[0 for m in range(layer_count + 1)] for index in range(len(pairs))]

        for index, (si, sj) in enumerate(pairs):
            xi = self.des_forword(si[0].unsqueeze(0))
            xj = self.des_forword(sj[0].unsqueeze(0))
            h_i_m[index][-1] = xi
            h_j_m[index][-1] = xj
            for m, layer in enumerate(self.ddml_layers):
                xi = layer(xi)
                xj = layer(xj)
                z_i_m[index][m] = xi
                z_j_m[index][m] = xj
                xi = self._s(xi)
                xj = self._s(xj)
                h_i_m[index][m] = xi
                h_j_m[index][m] = xj

        # calculate delta_ij(m)
        # calculate delta_ji(m)
        delta_ij_m = [[0 for m in range(layer_count)] for index in range(len(pairs))]
        delta_ji_m = [[0 for m in range(layer_count)] for index in range(len(pairs))]

        # M = layer_count, then we also need to project 1,2,3 to 0,1,2
        M = layer_count - 1

        # calculate delta(M)
        for index, (si, sj) in enumerate(pairs):
            xi = si[0].unsqueeze(0)
            xj = sj[0].unsqueeze(0)
            yi = si[1]
            yj = sj[1]

            # calculate c and loss
            if int(yi) == int(yj):
                l = 1
            else:
                l = -1

            dist = self.compute_distance(xi, xj)
            c = self.b - l * (self.tao - dist)
            loss += self._g(c)

            # h(m) have M + 1 values and m start from 0, in fact, delta_ij_m have M value and m start from 1
            delta_ij_m[index][M] = (self._g_derivative(c) * l * (
            h_i_m[index][M] - h_j_m[index][M])) * self._s_derivative(z_i_m[index][M])
            delta_ji_m[index][M] = (self._g_derivative(c) * l * (
            h_j_m[index][M] - h_i_m[index][M])) * self._s_derivative(z_j_m[index][M])

        loss /= len(pairs)

        # calculate delta(m)
        for index in range(len(pairs)):
            for m in reversed(range(M)):
                delta_ij_m[index][m] = torch.mm(delta_ij_m[index][m + 1], params_W[m + 1]) * self._s_derivative(
                    z_i_m[index][m])
                delta_ji_m[index][m] = torch.mm(delta_ji_m[index][m + 1], params_W[m + 1]) * self._s_derivative(
                    z_j_m[index][m])

        # calculate partial derivative of W
        partial_derivative_W_m = [0 for m in range(layer_count)]

        for m in range(layer_count):
            for index in range(len(pairs)):
                partial_derivative_W_m[m] += torch.mm(delta_ij_m[index][m].t(), h_i_m[index][m - 1])
                partial_derivative_W_m[m] += torch.mm(delta_ji_m[index][m].t(), h_j_m[index][m - 1])

        # calculate partial derivative of b
        partial_derivative_b_m = [0 for m in range(layer_count)]

        for m in range(layer_count):
            for index in range(len(pairs)):
                partial_derivative_b_m[m] += (delta_ij_m[index][m] + delta_ji_m[index][m]).squeeze()

        for m, layer in enumerate(self.ddml_layers):
            params = list(layer.parameters())
            params[0].data.sub_(self.learning_rate * partial_derivative_W_m[m])
            params[1].data.sub_(self.learning_rate * partial_derivative_b_m[m])

        return loss

    def _g(self, z):
        z = torch.tensor(z)
        if z > 10:
            value = z
        else:
            value = torch.log(1 + torch.exp(self.beta * z)) / self.beta
        return value

    def _g_derivative(self, z):
        z = torch.tensor(z)
        return 1 / (torch.exp(-1 * self.beta * z) + 1)

    def _s_derivative(self, z):
        z = torch.tensor(z)
        return 1 - self._s(z) ** 2