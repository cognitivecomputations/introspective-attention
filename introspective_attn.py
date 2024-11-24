import math
import torch
import torch.nn.functional as F
from torch import nn

from .kernel.rotary import apply_rotary_emb
try:
    from apex.normalization import FusedRMSNorm as RMSNorm 
except ModuleNotFoundError:
    print("No fused RMSNorm")
    from .rms_norm import RMSNorm

def init_method(tensor):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))

class IntrospectiveAttention(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        num_heads,
        num_introspective_layers=3  # Number of P_i layers
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads // args.model_parallel_size
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        # Create P_i layers
        self.P_layers = nn.ModuleList([
            nn.ModuleDict({
                'q_proj': nn.Linear(embed_dim, embed_dim, bias=False),
                'k_proj': nn.Linear(embed_dim, embed_dim, bias=False),
                'v_proj': nn.Linear(embed_dim, embed_dim, bias=False),
                'layer_norm': RMSNorm(embed_dim, eps=1e-5)
            }) for _ in range(num_introspective_layers)
        ])
        
        # Learnable lambda weights for each layer
        self.lambda_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(1, dtype=torch.float32))
            for _ in range(num_introspective_layers)
        ])
        
        # Output projection and normalization
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.final_layer_norm = RMSNorm(embed_dim, eps=1e-5)
        
        # Residual connection weight
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32))
        
        # Initialize weights
        for p_layer in self.P_layers:
            init_method(p_layer['q_proj'].weight)
            init_method(p_layer['k_proj'].weight)
            init_method(p_layer['v_proj'].weight)
        init_method(self.out_proj.weight)

    def forward(self, x, rel_pos, attn_mask=None):
        bsz, tgt_len, embed_dim = x.size()
        
        # Store intermediate K, V for concatenation
        accumulated_kv = []
        layer_outputs = []
        
        # Pre-compute normalized lambda weights
        lambda_weights = F.layer_norm(
            torch.stack([F.sigmoid(l) for l in self.lambda_weights]), 
            normalized_shape=[len(self.lambda_weights)]
        )
        
        for i, (p_layer, lambda_weight) in enumerate(zip(self.P_layers, lambda_weights)):
            # Current layer projections
            q = p_layer['q_proj'](x).view(bsz, tgt_len, self.num_heads, self.head_dim)
            k = p_layer['k_proj'](x).view(bsz, tgt_len, self.num_heads, self.head_dim)
            v = p_layer['v_proj'](x).view(bsz, tgt_len, self.num_heads, self.head_dim)
            
            # Hierarchical access to previous layers' K,V
            if i > 0:
                # Concatenate previous layers' keys and values
                accumulated_k = torch.cat([prev_k.view(bsz, -1, self.head_dim) 
                                        for prev_k, _ in accumulated_kv], dim=1)
                accumulated_v = torch.cat([prev_v.view(bsz, -1, self.head_dim) 
                                        for _, prev_v in accumulated_kv], dim=1)
                k = torch.cat([k, accumulated_k], dim=1)
                v = torch.cat([v, accumulated_v], dim=1)
            
            # Apply rotary embeddings
            q = apply_rotary_emb(q, *rel_pos)
            k = apply_rotary_emb(k, *rel_pos)
            
            # Reshape for attention
            q = q.transpose(1, 2)  # [bsz, num_heads, tgt_len, head_dim]
            k = k.transpose(1, 2)  # [bsz, num_heads, k_len, head_dim]
            v = v.transpose(1, 2)  # [bsz, num_heads, k_len, head_dim]
            
            # Scaled dot-product attention
            q = q * self.scaling
            attn_weights = torch.matmul(q, k.transpose(-1, -2))
            
            # Adjust attention mask for concatenated K,V
            if attn_mask is not None:
                if i > 0:
                    # Expand mask to cover concatenated sequence length
                    expanded_mask = torch.zeros(
                        (bsz, self.num_heads, tgt_len, k.size(-2)), 
                        device=attn_mask.device
                    ).fill_(-float('inf'))
                    expanded_mask[:, :, :, :tgt_len] = attn_mask
                    attn_weights += expanded_mask
                else:
                    attn_weights += attn_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
            
            # Compute attention output
            attn = torch.matmul(attn_weights, v)
            attn = attn.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
            
            # Apply layer norm
            attn = p_layer['layer_norm'](attn)
            
            # Store normalized output with normalized lambda weights
            layer_outputs.append(attn * lambda_weight[i])
            
            # Store current K,V for next layer
            accumulated_kv.append((k, v))
        
        # Combine all layer outputs with their respective lambda weights
        combined_output = sum(layer_outputs)
        
        # Add residual connection
        output = self.final_layer_norm(combined_output + self.alpha * x)
        
        return self.out_proj(output) 