import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import numpy as np

from torch_geometric.nn import GATConv, global_mean_pool,global_max_pool

from typing import Union, Tuple, Optional
import torch_geometric as tg
from transformers import activations
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import TransformerConv

from torch_geometric.nn import global_mean_pool, TopKPooling, GlobalAttention

from torch.nn import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    # torch_sparse,
    Tensor
)
from torch_geometric.utils import (
    add_self_loops,
    # is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)

  
  

class SWiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim):
        super(SWiGLU, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        a = self.linear1(x)  
        b = F.selu(self.linear2(x))
        return self.output_layer(a * b) 
    

class ImprovedGNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=1, num_layers=2, heads=4, dropout=0.1,concat = False):
        super(ImprovedGNNModel, self).__init__()
        print("in_channels: ", in_channels)
        print("hidden_channels: ", hidden_channels)
        print("out_channels: ", out_channels)
        self.conv1 = TransformerConv(in_channels, out_channels, heads=heads, dropout=dropout, edge_dim=edge_dim,concat=concat)
        
        self.conv2 = TransformerConv(out_channels , out_channels, heads=heads, dropout=dropout, edge_dim=edge_dim,concat=concat)
        
        self.lin = torch.nn.Linear(out_channels , out_channels)
        self.global_to_line_mapper = nn.Linear(out_channels, out_channels)
        self.gate = nn.Linear(2 * out_channels, out_channels)
        self.alpha = nn.Parameter(torch.tensor(0.1))  
        self.belta = nn.Parameter(torch.tensor(0.1)) 

        self.swi_glu = SWiGLU(out_channels , out_channels * 4, out_channels)
        self.swi_glu2 = SWiGLU(out_channels , out_channels * 4, out_channels)

    def forward(self, text_x, edge_index, edge_attr, batch, training=True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        
        if edge_attr.shape[0] == 0:
            edge_index, edge_attr = add_self_loops(edge_index, num_nodes=x.size(0))
            edge_attr = torch.ones(edge_index.shape[1],1).to(device)
        edge_attr = edge_attr.view(-1, 1) 

        x = self.conv1(text_x, edge_index, edge_attr)

        concat_features =text_x + x
        x = self.swi_glu(concat_features)

        x = self.conv2(x, edge_index, edge_attr)
        concat_features =text_x + x
        x = self.swi_glu2(concat_features)

        global_feature = global_mean_pool(x, batch) 
        mapped_global_feature = self.global_to_line_mapper(global_feature)

        mapped_global_feature1 = mapped_global_feature[batch]  
        
        concat_features = torch.cat([text_x, self.alpha * mapped_global_feature1], dim=-1)
        gate = torch.sigmoid(self.gate(concat_features))  
        gated_feature = gate * text_x + (1 - gate) * mapped_global_feature1
        combined_feature = text_x +  self.belta * gated_feature  

        return mapped_global_feature, combined_feature 


class FusionBlock(MessagePassing):
    def __init__(self, gnn_size, llm_size, hidden_size,is_pretraining, **kwargs):
        super().__init__(aggr='mean',**kwargs)
        self.hidden_size = hidden_size
        self.llm_size = llm_size
        self.gnn_size = gnn_size
        self.prompt_lin = torch.nn.Linear(llm_size,hidden_size,bias=False)
        self.g_lin = torch.nn.Linear(hidden_size,hidden_size,bias=False)        
        self.fuse1 = torch.nn.Linear(hidden_size*2,hidden_size*10,bias=False)  
        self.fuse2 = torch.nn.Linear(hidden_size*10,hidden_size,bias=False)  
        self.extend = torch.nn.Linear(hidden_size,llm_size,bias=False)
        self.ACT2FN = activations.ACT2FN['silu']
        self.is_pretraining = is_pretraining
    
    def forward(self, x,node_ids, prompt):
        node_ids = torch.arange(len(x))
        node_ids = node_ids.view(-1) 
        token = self.prompt_lin(prompt) 
        
        out = x[node_ids] 
        out = self.g_lin(out) 
        out = torch.cat((out,token),dim=1) 
        out = self.ACT2FN(self.fuse1(out))
        out = self.fuse2(out) 
        if(self.is_pretraining): 
            out = self.extend(out)
        return out 
    
    def message(self, x_j, k: OptTensor,v,
                index: Tensor, ptr: OptTensor,q,
                size_i: Optional[int]) -> Tensor:
        v = v
        return v
    
import torch_scatter
class FunctionAggregator(nn.Module):
    def __init__(self, input_features, output_features):
        super(FunctionAggregator, self).__init__()
        self.linear = nn.Linear(input_features, output_features)

    def forward(self, x, batch_index):
        x = self.linear(x)  
        x = torch.relu(x) 
        x = torch_scatter.scatter(x, batch_index, dim=0, reduce='mean')  
        return x


class CrossAttentionFusion(nn.Module):
    def __init__(self, text_hidden_size, graph_hidden_size,llm_shape):
        super(CrossAttentionFusion, self).__init__()
        print("text_hidden_size: ", text_hidden_size)
        self.cross_attention = nn.MultiheadAttention(embed_dim=text_hidden_size, num_heads=4, batch_first=True)

        self.layer_norm = nn.LayerNorm(text_hidden_size)
        
        self.fc = nn.Linear(text_hidden_size + graph_hidden_size, llm_shape)

        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

    def forward(self, text_representation, graph_representation):

        fused_representation, _ = self.cross_attention(graph_representation, text_representation, text_representation)

        fused_representation = fused_representation.squeeze(1)

        fused_representation = self.layer_norm(fused_representation)

        fused_representation = fused_representation + text_representation 

        combined_representation = self.relu(fused_representation)

        return combined_representation
    
    
class GraphAdapter(torch.nn.Module):
    def __init__(self,llm_shape, hiddensize_gnn=64, hiddensize_fusion = 64,normal_hidden_size=64, num_layers=2, GNN_type='GAT', is_pretraining=True, only_coarse=True):
        super(GraphAdapter,self).__init__()
        self.fc = nn.Linear(llm_shape, hiddensize_fusion)
        
        if(GNN_type == 'ImprovedGNNModel'):
            self.graph_encode = ImprovedGNNModel(in_channels=hiddensize_gnn, hidden_channels=hiddensize_gnn*2, out_channels=hiddensize_fusion, edge_dim=1, num_layers=2)
        else:
            raise "GNN_type should be ImprovedGNNModel"
        
        self.fuse_model = FusionBlock(hiddensize_gnn, llm_shape, hiddensize_fusion,is_pretraining)

        self.fc_fun_out1 = nn.Linear(hiddensize_fusion, hiddensize_fusion*2)
        self.fc_fun_out2 = nn.Linear(hiddensize_fusion*2 , hiddensize_fusion)
        self.fc_fun_out3 = nn.Linear(hiddensize_fusion , 2)

        self.fc_line_out1 = nn.Linear(hiddensize_fusion, hiddensize_fusion*2)
        self.fc_line_out2 = nn.Linear(hiddensize_fusion*2 , hiddensize_fusion)
        self.fc_line_out3 = nn.Linear(hiddensize_fusion , 2)

        self.apply(self._init_weights) 

        self.coarse_aggregation = FunctionAggregator(input_features=hiddensize_fusion, output_features=hiddensize_fusion)
    
        self.fc_fun_llm = nn.Linear(llm_shape, hiddensize_fusion)
        self.cross_attention_fusion = CrossAttentionFusion(text_hidden_size=hiddensize_fusion, graph_hidden_size=hiddensize_fusion,llm_shape=llm_shape)
        
        self.relu = nn.ReLU()  

        self.gate = nn.Linear(2 * hiddensize_fusion, hiddensize_fusion)
        self.alpha = nn.Parameter(torch.tensor(0.1))  
        self.belta = nn.Parameter(torch.tensor(0.1)) 

        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight) 
            if module.bias is not None:
                module.bias.data.zero_() 
        elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu') 
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, data, training, only_coarse=True):
        
        x = self.fc(data.x)

        global_feature, out = self.graph_encode(x, data.edge_index, data.edge_attr, data.batch,training) 

        line_logits =self.fc_line_out3(torch.tanh( self.fc_line_out2(self.relu(self.fc_line_out1(out)))))
        
        return line_logits 


