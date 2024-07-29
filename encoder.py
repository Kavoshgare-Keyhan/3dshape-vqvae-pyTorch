import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        activation_map = {
            'relu': nn.ReLU(inplace=True),
            'leaky': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }

        self.config = config

        ##### Validate the configuration for the model is correctly setup #######
        assert config['conv_activation_fn'] is None or config['conv_activation_fn'] in activation_map
        self.latent_dim = config['latent_dim']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Encoder is just Conv bn activation blocks
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(config['convbn_channels'][i], config['convbn_channels'][i + 1],
                          kernel_size=config['conv_kernel_size'][i], stride=config['conv_kernel_strides'][i],padding=1),
                nn.BatchNorm2d(config['convbn_channels'][i + 1]),
                activation_map[config['conv_activation_fn']],
            )
            for i in range(config['convbn_blocks'])
        ])
    
    def forward(self, x):
        out = x
        for layer in self.encoder_layers:
            out = layer(out)
        return out


def get_encoder(config):
    encoder = Encoder(
        config=config['model_params']
    )
    return encoder
        