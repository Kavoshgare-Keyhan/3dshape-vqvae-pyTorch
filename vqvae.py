import torch
import torch.nn as nn
from encoder import get_encoder
from decoder import get_decoder
from quantizer import get_quantizer


class VQVAE(nn.Module):
    def __init__(self,
                 config
                 ):
        super(VQVAE, self).__init__()
        self.encoder = get_encoder(config)
        self.pre_quant_conv = nn.Conv2d(config['model_params']['convbn_channels'][-1],
                                        config['model_params']['latent_dim'],
                                        kernel_size=1)
        self.quantizer = get_quantizer(config)
        self.post_quant_conv = nn.Conv2d(config['model_params']['latent_dim'],
                                        config['model_params']['transposebn_channels'][0],
                                        kernel_size=1)
        self.decoder = get_decoder(config)
        
    def forward(self, x):
        enc = self.encoder(x) # Call forward method by automatically passing x to the forward method withing Encoder class and returning its output which is called out and is a sequential of layers
        quant_input = self.pre_quant_conv(enc)
        quant_output, quant_loss, quant_idxs = self.quantizer(quant_input) # Call forward method and return corresponding outputs where quant_output is z_q
        dec_input = self.post_quant_conv(quant_output)
        out = self.decoder(dec_input)
        return {
            'generated_image' : out,
            'quantized_output' : quant_output,
            'quantized_losses' : quant_loss,
            'quantized_indices' : quant_idxs
        }
    
    def decode_from_codebook_indices(self, indices):
        quantized_output = self.quantizer.quantize_indices(indices) # returns indices corresponds to q(z|x)
        dec_input = self.post_quant_conv(quantized_output)
        return self.decoder(dec_input)
        


def get_model(config):
    print(config)
    model = VQVAE(
        config=config
    )
    return model


