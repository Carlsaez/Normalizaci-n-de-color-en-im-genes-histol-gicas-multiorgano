import torch
import torch.nn as nn
import torchvision.transforms.functional as fn


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

class UpBlock(nn.Module):
    def __init__(self, input_c, output_c, kernel, stride, padding):
        super().__init__()

        self.upblock = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                        nn.Conv2d(input_c, output_c, kernel_size=kernel, stride=stride, padding=padding),
                        nn.ReLU(True),
                        nn.BatchNorm2d(output_c))
    def forward(self, x):
        return self.upblock(x)


class Generator(nn.Module):
    def __init__(self, input_c, nf=64):
        super().__init__()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down.append(DownBlock(input_c, nf, kernel=3, stride=1, padding=1))

        for i in range(3):
            in_f = 2**i
            out_f = 2**(i+1)
            self.down.append(DownBlock(nf*in_f, nf*out_f, kernel=3, stride=1, padding=1))

            if i == 2:
                self.bottleneck = DownBlock(nf*out_f, 2*nf*out_f, kernel=3, stride=1, padding=1)


        for i in reversed(range(1, 5)):
            in_f = 2**i
            out_f = 2**(i-1)
            self.up.append(UpBlock(nf*in_f, nf*out_f, kernel=2, stride=2, padding=0))
            self.up.append(DownBlock(nf*in_f, nf*out_f, kernel=3, stride=1, padding=1))

        self.last_up = nn.Conv2d(nf, input_c, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        skip_conn = []

        # DOWN Part
        for layer_down in self.down:
            x = layer_down(x)
            skip_conn.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_conn = skip_conn[::-1]

        # UP Part
        for idx in range((len(self.up)//2)):
            x = self.up[idx*2](x)
            
            # En la última capa debido al up-sampling, la imagen se queda en 160x160 en lugar de 161x161, ya que el up-sampling "suele ser par"
            if x.shape != skip_conn[idx].shape:
                x = fn.center_crop(x, (skip_conn[idx].shape[-2], skip_conn[idx].shape[-1]))

            concat = torch.cat((skip_conn[idx], x), dim=1)
            x = self.up[idx*2+1](concat)
        
        return self.last_up(x)

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