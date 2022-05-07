import torch
import torch.nn as nn

class Conv_Block(nn.Module):
    def __init__(self, input, output, kernel, stride, activation=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(input, output, kernel, stride=stride, padding_mode='reflect', **kwargs),
                        nn.BatchNorm2d(output),
                        nn.ReLU(True) if activation else nn.Identity()
                    )
    def forward(self, x):
        return self.conv(x)

class Conv_UP_Block(nn.Module):
    def __init__(self, input, output, kernel, stride, activation=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(input, output, kernel, stride=stride, **kwargs),
            nn.BatchNorm2d(output),
            nn.ReLU(True) if activation else nn.Identity()
            )
    def forward(self, x):
        return self.conv(x)

class Residual_block(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.build_block = nn.Sequential(
            Conv_Block(input, output, kernel=3, stride=1, activation=True, padding=1),
            Conv_Block(input, output, kernel=3, stride=1, activation=False, padding=1)
        )
    def forward(self, x):
        return x + self.build_block(x)


class Generator(nn.Module):
    def __init__(self, input_c = 3, features=64, number_res=6):
        super().__init__()
        self.initial = nn.Sequential(*[nn.Conv2d(input_c, features, (7,7), stride=1, padding=3),
            nn.BatchNorm2d(features), 
            nn.ReLU(True)]
        )
        self.down_blocks = []
        self.up_blocks = []

        for i in range(3):
            in_f = 2**i
            out_f = 2**(i+1)
            self.down_blocks.append(Conv_Block(input=in_f*features, output=out_f*features, kernel=3, stride=2, padding=1))
            self.up_blocks.append(Conv_UP_Block(input=out_f*features, output=in_f*features, kernel=3, stride=2, padding=1))
        
        for _ in range(3):
            in_f = 8
            out_f = 8
            self.down_blocks.append(Conv_Block(input=in_f*features, output=out_f*features, kernel=3, stride=1, padding=1))
            self.up_blocks.append(Conv_UP_Block(input=in_f*features, output=out_f*features, kernel=3, stride=1, padding=1))
        
        self.down_blocks = nn.Sequential(*self.down_blocks)

        self.residual_block = nn.Sequential(*[Residual_block(8*features, 8*features) for _ in range(number_res)])

        self.up_blocks = nn.Sequential(*self.up_blocks[::-1])


        self.last_block = nn.Conv2d(features, input_c, (7,7), stride=1, padding=3)

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        for layer in self.residual_block:
            x = layer(x)
        for layer in self.up_blocks:
            x = layer(x)
        x = self.last_block(x)
        return torch.tanh(x)

### Discriminator

class Conv_Block_D(nn.Module):
    def __init__(self, input, output, kernel, stride, padding, *args):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(input, output, kernel, stride=stride, padding=padding),
                        nn.BatchNorm2d(output),
                        nn.LeakyReLU(0.2, True)
                    )
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, inputs=3, outputs=64, num_layers=3):
        super().__init__()
        model = [nn.Conv2d(inputs, outputs, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)]
        
        for n in range(1, num_layers):
            model += [Conv_Block_D(input=outputs*(2**(n-1)), output=outputs*(2**n), kernel=4, stride=2, padding=1)]
        
        model += [Conv_Block_D(outputs*(2**(num_layers-1)), outputs*(2**num_layers), kernel=4, stride=1, padding=1),
                nn.Conv2d(outputs*(2**num_layers), 1, kernel_size=4, padding=1)]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return torch.sigmoid(self.model(x))