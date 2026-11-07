import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
class LSTMNET(nn.Module):
    def __init__(self, input_size: float, hidden: float, attention: Optional[bool] = True, sparse: Optional[bool] = True, num_heads: Optional[float] = 4, 
                 attention_dropout: Optional[float] = 0.2, dropout: float = 0.1, 
                 num_layer: int = 2, bidirectional: bool = False, output: int = 1, output_size: int = None, act:str=None):
        """Constructor of the LSTMENCODER class: generates the encoder of the autoencoder. It is up to the user whether to use attention and/or bi-directional LSTM layers. The attention mechanism can be customized by whether using the sparse implementation and more randomness

        Parameters
        ----------
        input_size : float
            size of the input layer
        hidden : float
            size of the hidden layer
        dropout : float, optional
            Value of how percent should be masked, the value needs to be between (0.0,1.0], by default 0.1
        attention : bool, optional
            To use attention in the encoder, set this value to true, by default True
        bidirectional : bool, optional
            To use the bidirectional LSTM layers set to true, by default False
        num_heads : Optional[float], optional
            number of heads for the multi-head attention, should be passed when attention=true. In addition num_heads =1 equals Single Attention, WARNING: Model Dimension / Number of Heads %0, a higher number of heads leads to a poor performance, by default 4
        sparse : Optional[bool], optional
            To use the sparse attention implementation for saving space, time and money, by default True
        """
        super().__init__()
        self.hidden = hidden
        self.input_size = input_size
        self.output_size =  output_size or int(hidden * 0.75)
        self.attention = attention
        self.dropout = dropout
        self.num_classes = output
        self.act = act or 'GELU'

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.output_size, num_layers=num_layer, batch_first=True, dropout=dropout, bidirectional=bidirectional)

        self.gelu = getattr(torch.nn, self.act)()
        
        if bidirectional: 
            self.output_size = self.output_size*2

        if attention:
            self.attn = nn.MultiheadAttention(embed_dim=self.output_size, num_heads=num_heads, dropout=attention_dropout, batch_first=True)
            #self.attn = MultiHeadAttention(dim_model=self.output_size, num_heads=num_heads, dropout=attention_dropout, sparse=sparse)
        self.fc = nn.Linear(self.output_size, self.num_classes)
        self.sigmoid  = nn.Sigmoid()
    def forward(self, in_data: torch.Tensor) -> torch.Tensor:
        in_data = in_data.permute(0,2,1)
        out, _ = self.lstm(in_data)
        out = self.gelu(out)
        
        if self.attention:
            #out = out.permute(0, 2, 1)
            out, attn_output_weights = self.attn(out, out, out)
        print(np.shape(out))
        
        out = self.fc(out[:,-1])
        out = self.sigmoid(out)
    
       
        print(np.shape(out))
        return out






class MultiHeadAttention(nn.Module):

    def __init__(self, dim_model:float, num_heads:float, dropout:float, sparse:bool):
        """Multi-Head Attention: Implementation of the single dot attention for the multi-head attention. With a sparse implementation to save ressources and the space, with Dropout to add some randomness to the attention weights

        Parameters
        ----------
        dim_model : float
            Dimension of the model
        num_heads : float
            Number of heads, the dimension of the models should be divideable by the number of heads. A higher number of heads might lead to a poor performance!
        dropout : float
            Amount of values which should be zeroed out
        sparse : bool
            To use the sparse implementation set it to True.
        """
        super().__init__()
        assert dim_model % num_heads == 0, "The dimension of the models should be dividable by the numbers of heads."
        
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_k = dim_model // num_heads

        if sparse: 
            self.W_q = nn.Linear(dim_model, dim_model)    
            self.W_k = nn.Linear(dim_model, dim_model)
            self.W_v = nn.Linear(dim_model, dim_model)
            # import sparselinear as sl
            # self.W_q = sl.SparseLinear(dim_model, dim_model)
            # self.W_k = sl.SparseLinear(dim_model, dim_model)
            # self.W_v = sl.SparseLinear(dim_model, dim_model)
            
        else:     
            self.W_q = nn.Linear(dim_model, dim_model)    
            self.W_k = nn.Linear(dim_model, dim_model)
            self.W_v = nn.Linear(dim_model, dim_model)

        self.dropout = dropout
        self.Dropout = nn.Dropout(dropout)
        self.W_o = nn.Linear(dim_model, dim_model)

        for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode='fan_in')
                        if m.bias is not None:
                            nn.init.constant_(m.bias,0)
        
        
    def scaled_dot_product_attention(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:Optional[torch.BoolTensor]=None)->torch.Tensor:
        """Scaled dot attention: Is the matrix muliplication between the query and the key, where the result is scaled by the square root of the key dimension. The scaled result is multiplied by the value.

        Parameters
        ----------
        query : torch.Tensor
            Query vector
        key : torch.Tensor
            Key vector
        value : torch.Tensor
            Value vector
        mask : Optional[torch.BoolTensor], optional
            boolean mask, by default None

        Returns
        -------
        torch.Tensor
            attention probabilities
        """
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, value)
        return output
        
    def split_heads(self, x:torch.Tensor)->torch.Tensor:
        """Splits the data in order to apply the attention per head

        Parameters
        ----------
        x : torch.Tensor
            encoded lstm output

        Returns
        -------
        torch.Tensor
            tensor made ready for the heads
            
        """
        batch_size, seq_length, dim_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.dim_k).transpose(1, 2)
        
    def combine_heads(self, x:torch.Tensor)->torch.Tensor:
        """Combining the output of the heads

        Parameters
        ----------
        x : torch.Tensor
            output of the attention mechanism

        Returns
        -------
        torch.Tensor
            combinded output of the attention mechanism
        """
        batch_size, num_heads, seq_length, dim_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.dim_model)
        
    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:Optional[None]=None)->torch.Tensor:
        """Foward call of the multi head attention. If query=key=value: Self.Attention

        Parameters
        ----------
        query : torch.Tensor
            Query tensor
        key : torch.Tensor
            Key
        value : torch.Tensor
            Value
        mask : Optional[None], optional
            mask for the scaled dot attention, by default None

        Returns
        -------
        torch.Tensor
            output of the attention mechanism
        """
        query = self.split_heads(self.Dropout(self.W_q(query)))
        key = self.split_heads(self.Dropout(self.W_k(key)))
        value = self.split_heads(self.Dropout(self.W_v(value)))
        
        attn_output = self.scaled_dot_product_attention(query, key, value, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    

