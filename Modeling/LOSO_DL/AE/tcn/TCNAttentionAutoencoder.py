import gc
from typing import Optional

import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from . import  conv


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





class TCNENCODER(nn.Module):
    def __init__(self,input_size:float,hidden:float,kernel_size:int,attention:Optional[bool]=True, sparse:Optional[bool]=True,num_heads:Optional[float]=4, 
                 attention_dropout:Optional[float]=0.2,stride:Optional[int]=1, groups:Optional[int]=1,padding_mode:Optional[str]='zeros',  dropout:float=0.1, dilation:Optional[int]=1):
        """Constructor of the LSTMENCODER class: genrates the encoder of the autoencoder. It is up to the user wether to use attention and/or bi-directional LSTM layers. The attention mechanism can be customized by wether using the sparse implementation and more randomness

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
        bilstm : bool, optional
            To use the bidirectional lstm layers set to true, by default True
        num_heads : Optional[float], optional
            number of heads for the multi-head attention, should be passed when attention=true. In addition num_heads =1 equals Single Attention, WARNING: Model Dimension / Number of Heads %0, a higher number of heads leads to a poor performance, by default 4
        sparse : Optional[bool], optional
            To use the sparse attention implementation for saving space, time and money, by default True
        """
        super().__init__()
        self.hidden = hidden
        self.input_size=input_size
        self.attention=attention
        self.dropout = dropout

        self.tcn=conv.TemporalConv1d(in_channels=self.input_size, out_channels=self.hidden,kernel_size = kernel_size,dilation = dilation, groups = groups, padding_mode = padding_mode, stride=stride, device=DEVICE)
        self.gelu = nn.GELU()
        
        if attention:
            self.attn = nn.MultiheadAttention(embed_dim=self.hidden, num_heads=num_heads, dropout=attention_dropout)
            
    def forward(self, in_data: torch.Tensor)->torch.Tensor:

      
      
     
        out = self.tcn(in_data)
        del in_data
        gc.collect()
        
        out = self.gelu(out)
        
        if self.attention:
            out = out.permute(0,2,1)
            # (batch_size, sequence_length, input_features)
            out, attn_output_weights = self.attn(out, out, out)
            out = out.permute(0,2,1)


        return out



class TCNDECODER(nn.Module):

    def __init__(self,hidden_size_encoder:float, hidden_size:float,output_size,kernel_size:int,
                stride:Optional[int]=1, groups:Optional[int]=1,
                padding_mode:Optional[str]='zeros',dilation:Optional[int]=1):
        """Constructor of the LSTMDECODER class.

        Parameters
        ----------
        hidden_size_encoder : float
            Size of the input = size of the output of the LSTMENCODER
        hidden_size : float
            Size of the hidden layer
        num_classes : int
            number of classes, if it is a regression task the number of classes should be 1
        bilstm : bool
            If bi-directional layers were used in the encoder
        dropout : float, optional
            value of how much should be masked (choose a value between (0.0, 1.0]), by default 0.1
        """
        super().__init__()
        
        
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size = hidden_size



        
        self.tcn=conv.TemporalConvTranspose1d(in_channels=self.hidden_size_encoder, out_channels=self.hidden_size,kernel_size = kernel_size,dilation = dilation, groups = groups, padding_mode = padding_mode, stride=stride, device=DEVICE)
           
        self.gelu = nn.GELU()
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=output_size, device=DEVICE)
        
        for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode='fan_in')
                        if m.bias is not None:
                            nn.init.constant_(m.bias,0)


    def forward(self, encoded_input:torch.Tensor)-> torch.Tensor:
        """Forward call of the LSTMDECODER, sends the data trough the LSTM Decoder.

        Parameters
        ----------
        encoded_input : torch.Tensor
            Output of the LSTMENCODER

        Returns
        -------
        torch.Tensor
            output of the autoencoder
        """

        out = self.gelu(self.tcn(encoded_input))
      #  print('shape before layer', out.shape)
        out = self.fc(out.permute(0,2,1))
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
            self.W_q = nn.Linear(dim_model, dim_model, device=DEVICE)    
            self.W_k = nn.Linear(dim_model, dim_model, device=DEVICE)
            self.W_v = nn.Linear(dim_model, dim_model, device=DEVICE)
            # import sparselinear as sl
            # self.W_q = sl.SparseLinear(dim_model, dim_model)
            # self.W_k = sl.SparseLinear(dim_model, dim_model)
            # self.W_v = sl.SparseLinear(dim_model, dim_model)
            
        else:     
            self.W_q = nn.Linear(dim_model, dim_model, device=DEVICE)    
            self.W_k = nn.Linear(dim_model, dim_model, device=DEVICE)
            self.W_v = nn.Linear(dim_model, dim_model, device=DEVICE)

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



class TCNAAE(nn.Module):

    def __init__(self,input_size:float, hidden_size_encoder: float,hidden_size_decoder: float,output_size:float, kernel_size:int, kernel_size_de:int,
                 dropout_encoder:float=0.1, dropout_decoder:float=0.1, 
                 attention:Optional[bool]=True, sparse:Optional[bool]=True,num_heads:Optional[float]=4, attention_dropout:Optional[float]=0.2,
                 stride:Optional[int]=1,stride_de:Optional[int]=1, groups:Optional[int]=1,
                 groups_de:Optional[int]=1,padding_mode:Optional[str]='zeros',padding_mode_de:Optional[str]='zeros', dilation:Optional[int]=1, dilation_de:Optional[int]=1):
        """Constructor of the autoencoder, this an implementation which can be customized:

            - use a simple autoencoder with LSTM layers: set bilstm = False, attention = False
            - use an autoencoder with bidirectional LSTM layers: set attention = False
            - use an autoencoder with sparse attention and uni-directional LSTM layers: set bilstm = False, sparse =True, attention = True
            - use an autoencoder with attention and uni-dirrectional LSTM layers: set bilstm = False, sparse = True, attention = True
            - use and autoencoder with sparse attention and bi-directional LSTM layes: set bilstm = True, sparse = True, attention = True

        Parameters
        ----------
        input_size : float
            size of the input data
        hidden_size_encoder : float
            size of the hidden layer in the encoder
        hidden_size_decoder : float
            size of the hidden layer in decoder
        output_size : float
            size of the output of the autoencoder, can be the same as the input, classical autoencoding, for classificaton: number of classes, for regression: 1 
        dropout_encoder : float, optional
            amount of values which are randomly set to zero, by default 0.1
        dropout_decoder : float, optional
            amount of values which are randomly set to zero, by default 0.1
        bilstm : Optional[bool], optional
            to use bidirectional LSTM layers, by default True
        attention : Optional[bool], optional
            to use attention, by default True
        sparse : Optional[bool], optional
            to use sparse attention, by default True
        num_heads : Optional[float], optional
            number of heads for the multihead attention, by default 4

        Notes
        -----
            Input shape LSTM and Attention: [batch_size, sequence_length, number of features]
            Input shape linear layer: [batch_size, number of features, sequence length]  

        """

       




        super().__init__()
        self.input_size = input_size
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size_decoder = hidden_size_decoder
        self.output_size = output_size
        self.dropout_decoder = dropout_decoder
        self.dropout_encoder = dropout_encoder
        self.stride = stride
        self.kernel_size = kernel_size
        self.groups =groups
        self.causual = True
        self.attention = attention
        self.sparse = sparse
        self.stride_de = stride_de
        self.kernel_size_de = kernel_size_de
        self.groups_de =groups_de
        self.causual = True
        


        self.encoder = TCNENCODER(input_size=self.input_size,hidden=hidden_size_encoder,kernel_size=kernel_size, groups=groups, padding_mode=padding_mode, dilation=dilation,dropout=self.dropout_encoder,stride = self.stride, attention=self.attention,attention_dropout=attention_dropout,num_heads=num_heads)
        self.decoder = TCNDECODER(hidden_size_encoder=self.hidden_size_encoder, hidden_size=self.hidden_size_decoder, output_size=self.output_size,kernel_size=kernel_size_de,stride=stride_de, groups=groups_de, padding_mode=padding_mode_de, dilation=dilation_de)

    def forward(self, time_series:torch.Tensor)-> torch.Tensor:
        """Forward call: Runs the autoencoder.

        Parameters
        ----------
        time_series : torch.Tensor
            Input data, a multi-modale time series

        Returns
        -------
        torch.Tensor
            Predicted values
        """
        #time_series = time_series.permute(0,2,1)

        encoder_output = self.encoder(time_series)
       # print(encoder_output.shape)
        decoder_output = self.decoder(encoder_output)

        decoder_output = decoder_output.permute(0,2,1)
        #print(decoder_output.shape)

        return decoder_output




