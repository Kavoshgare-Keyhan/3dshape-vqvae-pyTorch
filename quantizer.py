import torch
import torch.nn as nn
import numpy as np
from einops import einsum


class Quantizer(nn.Module):
    def __init__(self,
                 config
                 ):
        super(Quantizer, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config['codebook_size'], config['latent_dim'])
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), -1, x.size(-1))
        
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat((x.size(0), 1, 1)))
        min_encoding_indices = torch.argmin(dist, dim=-1)
        
        #
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1)) # returns embedding weights tensor in the range of min_encoding_indices along x-axis
        x = x.reshape((-1, x.size(-1)))
        commmitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {
            'codebook_loss' : codebook_loss,
            'commitment_loss' : commmitment_loss
        }
        quant_out = x + (quant_out - x).detach()
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        return quant_out, quantize_losses, min_encoding_indices
    
    def quantize_indices(self, indices):
        return einsum(indices, self.embedding.weight, 'b n h w, n d -> b d h w') 
        # Assuming 'indices' and 'self.embedding.weight' are NumPy arrays
        # result = np.einsum('bnhw,nd->bdhw', indices, self.embedding.weight)
        '''
        # indices: A tensor representing indices, likely used to select specific embeddings. Its shape is denoted as (b, n, h, w) - batch, number of indices, height, width.
        # self.embedding.weight: A tensor containing embedding vectors. Its shape is denoted as (n, d) - number of embeddings, embedding dimension.
        # 'b n h w, n d -> b d h w': The einsum equation, defining how the tensors are multiplied and summed. It specifies that the indices tensor (b, n, h, w) is multiplied by the embedding tensor (n, d) along the 'n' dimension, resulting in an output tensor of shape (b, d, h, w).
        '''


def get_quantizer(config):
    quantizer = Quantizer(
        config=config['model_params']
    )
    return quantizer
