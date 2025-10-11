import torch
import torch.nn as nn
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads                        #Reduzca  la  atenuación  de  la  proyección  para  que  coincida  con  la  atenuación  de  salida  deseada
        #si d_out=512 y num_heads=8, cada cabeza tendrá dimensión 64
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)                   #Capa lineal para combinar las salidas de la cabeza
        self.dropout = nn.Dropout(dropout)
        self.register_buffer( #evitar mirar a tokens futuros
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    def forward(self, x):
        b, num_tokens, d_in = x.shape #(2, 6, 3)
        keys = self.W_key(x)                                      #(b, num_tokens, d_out)
        queries = self.W_query(x)                                 
        values = self.W_value(x)                                  
        #Dividir  implícitamente  la  matriz  añadiendo  una  dimensión  `num_heads`.  Luego,  desenrollamos  la  última  dimensión:  (b,
        #núm_tokens,  d_out)  >  (b,  núm_tokens,  núm_cabezas,  cabeza_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim) 
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2)  #Transponer  de  la  forma  (b,  num_tokens,  num_heads,  head_dim)  a  (b,  num_heads,  num_tokens,  head_dim)                             
        queries = queries.transpose(1, 2)                         
        values = values.transpose(1, 2)  

        attn_scores = queries @ keys.transpose(2, 3)            
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]   #Mascara de truncamiento 
        attn_scores.masked_fill_(mask_bool, -torch.inf)           
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)       
        #Combina  cabezas,  donde  self.d_out  =  self.num_heads  *  self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)         #Agregar  una  proyección  lineal  opcional         
        return context_vec