import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class ResBlockEmbedding(torch.nn.Module):
    def __init__(self, cin, cout, stride=1, down_sample=False):
        super().__init__()

        self.conv1 = torch.nn.Conv1d(cin, cout, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn1 = torch.nn.BatchNorm1d(cout)

        self.conv2 = torch.nn.Conv1d(cout, cout, kernel_size=3, bias=False, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(cout)

        self.downsample = torch.nn.Sequential(
                torch.nn.Conv1d(cin, cout, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm1d(cout)) if down_sample else nn.Identity()

        self.act = torch.nn.GELU()
        self.permute = Permute()

    def forward(self, x: torch.Tensor):
        residual = self.permute(x)
        out = self.conv1(residual)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out + self.downsample(residual))
        out = self.permute(out)
        return out


class PositionalEncoding(torch.nn.Module):
    def __init__(self, projection_size, max_seq_len= 1000):
        super().__init__()
        self.dropout                = torch.nn.Dropout(0.1)

        pe              = torch.zeros(max_seq_len, projection_size)
        position        = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term        = torch.exp(torch.arange(0, projection_size, 2).float() * (-math.log(10000.0) / projection_size))
        pe[:, 0::2]     = torch.sin(position * div_term)
        pe[:, 1::2]     = torch.cos(position * div_term)
        pe              = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class FeedForward(torch.nn.Module):
    ''' Projection Layer (Fully Connected Layers) '''
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        self.linear_1   = torch.nn.Linear(d_model, d_ff)
        self.dropout    = torch.nn.Dropout(dropout)
        self.linear_2   = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):

        # Apply the first linear layer, GeLU activation, and then dropout
        x = self.dropout(torch.nn.functional.gelu(self.linear_1(x)))

         # Apply the second linear layer to project the dimension back to d_model
        x = self.linear_2(x)

        return x
    

class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature    = temperature                       # Scaling factor for the dot product
        self.dropout        = torch.nn.Dropout(attn_dropout)    # Dropout layer for attention weights
        self.softmax        = torch.nn.Softmax(dim=-1)           # Softmax layer along the attention dimension

    def forward(self, q, k, v, mask=None):

        # Calculate the dot product between queries and keys.
        # attn = torch.bmm(q, k.transpose(1, 2))
        attn = (q @ k.transpose(-2, -1))

        # Scale the dot product by the temperature.
        attn = attn / self.temperature

        if mask is not None:
            # Apply the mask by setting masked positions to a large negative value.
            # This ensures they have a softmax score close to zero.
            # mask_value = -1e+30 if attn.dtype == torch.float32 else -1e+4
            attn = attn.masked_fill(mask, float('-inf'))

        # Apply softmax to obtain attention weights.
        attn    = self.softmax(attn)

        # Apply dropout to the attention weights.
        # Compute the weighted sum of values based on the attention weights.
        # output  = torch.bmm(self.dropout(attn), v)
        attn = self.dropout(attn)
        output = attn @ v

        return output, attn # Return the attention output and the attention weights.


class MultiHeadAttention(torch.nn.Module):
    ''' Multi-Head Attention Module '''
    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()

        self.n_head = n_head # Number of attention heads
        self.d_k    = d_model // n_head
        self.d_v    = d_model // n_head


        # Linear layers for projecting the input query, key, and value to multiple heads
        self.w_qs   = torch.nn.Linear(d_model, n_head * self.d_k)
        self.w_ks   = torch.nn.Linear(d_model, n_head * self.d_k)
        self.w_vs   = torch.nn.Linear(d_model, n_head * self.d_v)

        torch.nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        torch.nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        torch.nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_v)))

        # Initialize the weights of the linear layers
        self.attention = ScaledDotProductAttention(
            temperature=np.power(self.d_k, 0.5), attn_dropout=dropout)

        # Final linear layer to project the concatenated outputs of the attention heads back to the model dimension
        self.fc = torch.nn.Linear(n_head * self.d_v, d_model)
        torch.nn.init.normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):
        # following key, value, query standard computation
        d_k, d_v, n_head    = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _      = q.size()
        sz_b, len_k, _      = k.size()
        sz_b, len_v, _      = v.size()

        # Project the input query, key, and value to multiple heads
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Rearrange the dimensions to group the heads together for parallel processing
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        # k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        # v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv


        # Repeat the mask for each attention head if a mask is provided
        if mask is not None:
              # print(mask.shape)
              # mask = mask.repeat(n_head, 1, 1)
              mask = mask.unsqueeze(1).repeat(1, n_head, 1, 1)

        # Apply scaled dot-product attention to the projected query, key, and value
        output, attn    = self.attention(q, k, v, mask=mask)

        # Rearrange the output back to the original order and concatenate the heads
        # output          = output.view(n_head, sz_b, len_q, d_v)
        # output          = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        output          = self.dropout(self.fc(output))

        attn_weights = attn.mean(dim=(0, 1))

        return output, attn_weights


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # @TODO: fill in the blanks appropriately (given the modules above)
        self.mha        = MultiHeadAttention(num_heads, d_model)
        self.ffn        = FeedForward(d_model, d_ff, dropout)

        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.layernorm2 = torch.nn.LayerNorm(d_model)
        self.layernorm3 = torch.nn.LayerNorm(d_model)
        self.dropout1   = torch.nn.Dropout(dropout)
        self.dropout2   = torch.nn.Dropout(dropout)


    def forward(self, inp):
        # Multi-Head Attention
        #   (1) perform Multi-Head Attention on x
        inp = self.layernorm1(inp)
        attn_output, self_attention_weights = self.mha(inp, inp, inp)

        # Skip (Residual) Connection
        #   (1) perform dropout
        #   (2) add the input as a skip connection
        res = inp + self.dropout1(attn_output)

        # Layer Normalization
        #   (1) call layernorm on this resulting value
        res = self.layernorm2(res)

        # Feed Forward Network
        #   (1) apply feed forward layer
        #   (2) apply the padding mask  to the output
        ffn_output = self.ffn(res)
        ffn_output = self.dropout2(ffn_output)

        # Skip (Residual) Connection
        #   (1) add the result before LayerNorm and FFN as skip connection
        x = res + ffn_output

        # Layer Normalization
        #   (1) call layernorm on this resulting value
        block_output = self.layernorm3(x)
        return block_output, self_attention_weights
    
    
class Permute(torch.nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


