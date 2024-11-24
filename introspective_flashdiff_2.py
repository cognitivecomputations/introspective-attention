import math
import torch
import torch.nn.functional as F
from torch import nn

from .kernel.rotary import apply_rotary_emb
from flash_attn import flash_attn_func
try:
    from apex.normalization import FusedRMSNorm as RMSNorm 
except ModuleNotFoundError:
    print("No fused RMSNorm")
    from .rms_norm import RMSNorm

def init_method(tensor):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))

class IntrospectiveFlashDiff2(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        num_heads,
        num_introspective_layers=3,
        max_sequence_length=None  # Optional parameter to limit memory usage
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads // args.model_parallel_size
        self.num_kv_heads = (args.decoder_kv_attention_heads // args.model_parallel_size 
                           if args.decoder_kv_attention_heads is not None 
                           else num_heads // args.model_parallel_size)
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_sequence_length = max_sequence_length

        # Create P_i layers with flash attention components
        self.P_layers = nn.ModuleList([
            nn.ModuleDict({
                'q_proj': nn.Linear(embed_dim, embed_dim, bias=False),
                'k_proj': nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False),
                'v_proj': nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False),
                'layer_norm': RMSNorm(embed_dim, eps=1e-5)
            }) for _ in range(num_introspective_layers)
        ])

        # Learnable lambda weights
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

    def forward(
        self,
        x,
        rel_pos,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        
        # Check sequence length against maximum if set
        if self.max_sequence_length is not None and tgt_len > self.max_sequence_length:
            raise ValueError(f"Input sequence length {tgt_len} exceeds maximum allowed length {self.max_sequence_length}")
        
        # Pre-allocate tensors for accumulated K,V
        max_kv_len = tgt_len * len(self.P_layers)  # Maximum possible length after all concatenations
        accumulated_k = torch.empty(
            bsz, max_kv_len, self.num_kv_heads, self.head_dim,
            dtype=x.dtype, device=x.device
        )
        accumulated_v = torch.empty(
            bsz, max_kv_len, self.num_kv_heads, self.head_dim,
            dtype=x.dtype, device=x.device
        )
        
        # Track current position in accumulated tensors
        current_position = 0
        layer_outputs = []

        # Pre-compute normalized lambda weights
        lambda_weights = F.layer_norm(
            torch.stack([F.sigmoid(l) for l in self.lambda_weights]), 
            normalized_shape=[len(self.lambda_weights)]
        )

        for i, (p_layer, lambda_weight) in enumerate(zip(self.P_layers, lambda_weights)):
            # Current layer projections
            q = p_layer['q_proj'](x)
            k = p_layer['k_proj'](x)
            v = p_layer['v_proj'](x)

            # Reshape for flash attention
            q = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
            k = k.view(bsz, tgt_len, self.num_kv_heads, self.head_dim)
            v = v.view(bsz, tgt_len, self.num_kv_heads, self.head_dim)

            # Store current K,V in pre-allocated tensors
            next_position = current_position + tgt_len
            accumulated_k[:, current_position:next_position] = k
            accumulated_v[:, current_position:next_position] = v
            
            # Use accumulated K,V for attention
            k_to_use = accumulated_k[:, :next_position]
            v_to_use = accumulated_v[:, :next_position]

            # Apply rotary embeddings
            q = apply_rotary_emb(q, *rel_pos)
            k_to_use = apply_rotary_emb(k_to_use, *rel_pos)

            # Flash attention computation
            attn = flash_attn_func(
                q,  # [bsz, tgt_len, num_heads, head_dim]
                k_to_use,  # [bsz, src_len, num_kv_heads, head_dim]
                v_to_use,  # [bsz, src_len, num_kv_heads, head_dim]
                causal=True
            )

            # Reshape output and apply layer norm
            attn = attn.reshape(bsz, tgt_len, embed_dim)
            attn = p_layer['layer_norm'](attn)
            
            # Store normalized output with normalized lambda weights
            layer_outputs.append(attn * lambda_weight[i])
            
            # Update position for next iteration
            current_position = next_position

        # Combine all layer outputs with their respective lambda weights
        combined_output = sum(layer_outputs)
        
        # Add residual connection
        output = self.final_layer_norm(combined_output + self.alpha * x)
        
        return self.out_proj(output) 