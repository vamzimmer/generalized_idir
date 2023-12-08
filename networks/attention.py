import torch
import torch.nn.functional as F
import numpy as np
from networks.network_utils import LinearLayer

class AttentionLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='none'):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Define layers for queries, keys, and values
        self.query = torch.nn.Linear(input_dim, hidden_dim)
        self.key = torch.nn.Linear(input_dim, hidden_dim)
        self.value = torch.nn.Linear(input_dim, hidden_dim)
              
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

        # Define output layer
        self.out = LinearLayer(hidden_dim, output_dim, activation)

    def forward(self, x1, x2=None):
        # Self-attention
        if x2 == None:
            x2 = x1

        ndim = x1.ndim
        # Change view to have correct size for multiplication
        if x1.ndim == 2:
            x1 = x1.view(x1.shape[0], 1, -1)
        if x2.ndim == 2:
            x2 = x2.view(x2.shape[0], 1, -1)

        # Compute queries, keys, and values
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        
        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2))
        # Scale score
        scores = scores / np.sqrt(self.hidden_dim)
        
        # Compute attention weights and apply to values
        attn_weights = F.softmax(scores, dim=2) 
        attn_output = torch.bmm(attn_weights, v)
        
        # Apply output layer
        output = self.out(attn_output)

        if ndim == 2:
            output = output.view(output.shape[0], output.shape[-1])
        return output