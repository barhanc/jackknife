import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

STEER_CLASSES = [-35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
STEER_CLASSES_RAD = [math.radians(deg) for deg in STEER_CLASSES]
NUM_STEER_CLASSES = len(STEER_CLASSES)  # 15

def steer_to_class_idx(phi_rad: float) -> int:
    """Finds closest discrete steering class index."""
    phi_deg = math.degrees(phi_rad)
    diffs = [abs(phi_deg - c) for c in STEER_CLASSES]
    return int(diffs.index(min(diffs)))

def class_idx_to_steer(idx: int) -> float:
    """Converts class index to steering angle in radians."""
    return STEER_CLASSES_RAD[idx]


class ControlTransformer(nn.Module):
    """
    ControlTransformer policy network for B-Double multi-trailer reversing.
    
    Inputs:
        tokens: (B, 64, 7) - Egocentric lookahead path stations in Trailer 2 local frame
                [x_rel, y_rel, cos(theta_rel), sin(theta_rel), kappa, w_L, w_R]
        hitch_state: (B, 4) - Trailer hitch angles [sin(beta1), cos(beta1), sin(beta2), cos(beta2)]
        
    Outputs:
        logits: (B, 15) - Discrete steering angle classification logits
        attn_weights: (B, 64) - Cross-attention weights over the path tokens
    """
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        d_ff: int = 1024,
        num_encoder_layers: int = 6,
        num_cross_layers: int = 3,
        num_tokens: int = 64,
        token_dim: int = 7,
        state_dim: int = 4,
        num_classes: int = 15,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_tokens = num_tokens

        # Context Encoder projections
        self.token_proj = nn.Linear(token_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)

        # Transformer Encoder for path context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Hitch state query projection
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Cross-Attention Layers (Query attends to Path Memory)
        self.cross_attn1 = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_ff1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm_ff1 = nn.LayerNorm(d_model)

        self.cross_attn2 = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_ff2 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm_ff2 = nn.LayerNorm(d_model)

        # Policy MLP Head
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        hitch_state: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        tokens: (B, 32, 7)
        hitch_state: (B, 4)
        """
        B = tokens.size(0)

        # Encode path tokens
        x = self.token_proj(tokens) + self.pos_embed  # (B, 32, d_model)
        memory = self.encoder(x)                      # (B, 32, d_model)

        # Encode hitch state as query
        q = self.state_proj(hitch_state).unsqueeze(1) # (B, 1, d_model)

        # Cross-attention layer 1
        attn_out1, _ = self.cross_attn1(query=q, key=memory, value=memory)
        q = self.norm1(q + attn_out1)
        q = self.norm_ff1(q + self.cross_ff1(q))

        # Cross-attention layer 2 (capture attention weights)
        attn_out2, attn_weights = self.cross_attn2(
            query=q, key=memory, value=memory,
            need_weights=return_attention,
            average_attn_weights=True
        )
        q = self.norm2(q + attn_out2)
        q = self.norm_ff2(q + self.cross_ff2(q)) # (B, 1, d_model)

        # Predict steering logits
        q_squeezed = q.squeeze(1) # (B, d_model)
        logits = self.head(q_squeezed) # (B, 15)

        if return_attention and attn_weights is not None:
            # attn_weights shape: (B, 1, 32) -> squeeze to (B, 32)
            attn_weights = attn_weights.squeeze(1)
            return logits, attn_weights
        return logits, None

    def predict_action(
        self,
        tokens_np,
        hitch_state_np,
        device: torch.device,
        use_expectation: bool = True
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Convenience inference helper for single-sample runtime execution.
        Returns:
            steer_rad: continuous steering angle corresponding to argmax / expectation
            attention_weights: (num_tokens,) numpy array
            probs: (15,) action probabilities distribution numpy array
        """
        self.eval()
        with torch.no_grad():
            t_tok = torch.from_numpy(tokens_np).float().unsqueeze(0).to(device)
            t_state = torch.from_numpy(hitch_state_np).float().unsqueeze(0).to(device)

            logits, attn = self(t_tok, t_state, return_attention=True)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            
            if use_expectation:
                # Continuous softmax expected value: eliminates quantization dither
                steer_rad = float(np.sum(probs * np.array(STEER_CLASSES_RAD)))
            else:
                best_idx = int(np.argmax(probs))
                steer_rad = class_idx_to_steer(best_idx)
            
            attn_weights = attn.cpu().numpy()[0] if attn is not None else np.ones(t_tok.shape[1]) / float(t_tok.shape[1])
            return steer_rad, attn_weights, probs
