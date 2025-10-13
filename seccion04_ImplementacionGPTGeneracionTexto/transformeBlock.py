from .multiHeadAttention import MultiHeadAttention
from .feedForward import FeedForward
from .layerNorm import LayerNorm
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            dropout=cfg["drop_rate"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):   
        ## Bloque 1 atencion
        shortcut = x    #conexion de acceso directo para el bloque de atencion
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut 

        #Bloque 2: Red neuronal feed_forward
        shortcut = x    #conexion de acceso directo para el bloque de avence
        x = self.norm2(x)   
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut    #agregaci´on de entrada principal
        return x