import torch
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(self, input_c, output_c, kernel, stride, padding):
        super().__init__()

        self.downblock = nn.Sequential(nn.Conv2d(input_c, output_c, kernel_size=kernel, stride=stride, padding=padding, padding_mode='reflect'),
                          nn.ReLU(True),
                          nn.BatchNorm2d(output_c),
                          nn.Conv2d(output_c, output_c, kernel_size=kernel, stride=stride, padding=padding, padding_mode='reflect'),
                          nn.ReLU(True),
                          nn.BatchNorm2d(output_c))
    def forward(self, x):
        return self.downblock(x)

class batchn_relu(nn.Module):
    def __init__(self, input_c):
        super().__init__()

        self.bn = nn.BatchNorm2d(input_c)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, input_c, output_c):
        super().__init__()

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.residual = Residual_block(input_c+output_c, output_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], axis=1)
        x = self.residual(x)
        return x

class Residual_block(nn.Module):
    def __init__(self, input, output, stride=1):
        super().__init__()

        self.build_block = nn.Sequential(batchn_relu(input),
                                          nn.Conv2d(input, output, kernel_size=3, stride=stride, padding=1, padding_mode='reflect'),
                                          batchn_relu(output),
                                          nn.Conv2d(output, output, kernel_size=3, stride=1, padding=1, padding_mode='reflect'))
        self.skip = nn.Conv2d(input, output, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        return self.skip(x) + self.build_block(x)

class Generator(nn.Module):
    def __init__(self, input_c, nf=64):

        super().__init__()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()

        self.c11 = nn.Conv2d(input_c, nf, kernel_size=3, padding=1)
        self.br1 = batchn_relu(nf)
        self.c12 = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.c13 = nn.Conv2d(input_c, nf, kernel_size=1, padding=0)

        self.r2 = Residual_block(nf, 2*nf, stride=2)
        self.r3 = Residual_block(2*nf, 4*nf, stride=2)

        self.r4 = Residual_block(4*nf, 8*nf, stride=2)

        self.d1 = UpBlock(8*nf, 4*nf)
        self.d2 = UpBlock(4*nf, 2*nf)
        self.d3 = UpBlock(2*nf, nf)

        self.output = nn.Conv2d(nf, 3, kernel_size=1, padding=0)

    def forward(self, inputs):
        x = self.c11(inputs)
        x = self.br1(x)
        x = self.c12(x)
        s = self.c13(inputs)
        skip1 = x + s

        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)

        b = self.r4(skip3)

        d1 = self.d1(b, skip3)
        d2 = self.d2(d1, skip2)
        d3 = self.d3(d2, skip1)

        output = self.output(d3)
        return output

class Conv_Block(nn.Module):
    def __init__(self, input, output, kernel, stride, padding, *args):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(input, output, kernel, stride=stride, padding=padding),
                        nn.BatchNorm2d(output),
                        nn.LeakyReLU(0.2, True)
                    )
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, inputs, outputs=64, num_layers=3):
        super().__init__()
        model = [nn.Conv2d(inputs, outputs, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)]
        
        for n in range(1, num_layers):
            model += [Conv_Block(input=outputs*(2**(n-1)), output=outputs*(2**n), kernel=4, stride=2, padding=1)]
        
        model += [Conv_Block(outputs*(2**(num_layers-1)), outputs*(2**num_layers), kernel=4, stride=1, padding=1),
                nn.Conv2d(outputs*(2**num_layers), 1, kernel_size=4, padding=1)]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return torch.sigmoid(self.model(x))