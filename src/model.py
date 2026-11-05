import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, Union

STEER_CLASSES = [-35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
STEER_CLASSES_RAD = [math.radians(deg) for deg in STEER_CLASSES]
NUM_STEER_CLASSES = len(STEER_CLASSES)  # 15

def class_idx_to_steer(idx: int) -> float:
    """Converts class index to steering angle in radians."""
    return STEER_CLASSES_RAD[idx]

def steer_to_soft_target(phi_rad: float, sigma_deg: float = 4.0) -> np.ndarray:
    """
    Converts a continuous steering angle into a smooth Gaussian target distribution
    over the 15 discrete steering classes.
    Preserves exact sub-degree steering precision without threshold quantization chatter.
    """
    phi_deg = math.degrees(phi_rad)
    classes = np.array(STEER_CLASSES, dtype=np.float32)
    sq_diffs = (classes - phi_deg) ** 2
    probs = np.exp(-0.5 * sq_diffs / (sigma_deg ** 2))
    probs /= np.sum(probs)
    return probs.astype(np.float32)



class MultiTrailerPFN(nn.Module):
    """
    Prior-Data Fitted Network (PFN) for B-Double multi-trailer reversing policy,
    built with PyTorch's native nn.Transformer and query diagonal masking.

    Inputs:
        tokens: (B, num_tokens, token_dim) - Egocentric lookahead path stations in Trailer 2 frame
                [x_rel, y_rel, cos(theta_rel), sin(theta_rel), kappa, w_L, w_R]
        hitch_state: (B, state_dim) or (B, query_size, state_dim) - Rig articulation state
                [sin(beta1), cos(beta1), sin(beta2), cos(beta2)]

    Outputs:
        logits: (B, num_classes) or (B, query_size, num_classes) - Steering action logits
    """
    def __init__(
        self,
        token_dim: int = 7,
        state_dim: int = 4,
        num_classes: int = NUM_STEER_CLASSES,
        dim_model: int = 256,
        dim_feedforward: int = 1024,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 2,
        num_tokens: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.state_dim = state_dim
        self.num_classes = num_classes
        self.dim_model = dim_model
        self.dim_feedforward = dim_feedforward
        self.num_tokens = num_tokens

        # Feature projections for memory (path tokens) and query (hitch state)
        self.map_src = nn.Linear(token_dim, dim_model)
        self.map_tgt = nn.Linear(state_dim, dim_model)

        # PyTorch Built-in Transformer
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        # Policy MLP Head
        self.map_out = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, num_classes),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        hitch_state: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Args:
            tokens (Tensor): Shape `(batch_size, num_tokens, token_dim)`
            hitch_state (Tensor): Shape `(batch_size, state_dim)` or `(batch_size, query_size, state_dim)`
            return_attention (bool): Optional flag for compatibility with tuple unpacking callers.

        Returns:
            Tensor or Tuple[Tensor, None]: Action logits `(batch_size, num_classes)`
        """
        assert tokens.ndim == 3, f"Expected tokens ndim == 3, got {tokens.ndim}"

        is_2d_query = (hitch_state.ndim == 2)
        X_query = hitch_state.unsqueeze(1) if is_2d_query else hitch_state

        assert tokens.size(0) == X_query.size(0), "Batch sizes must match"
        assert tokens.size(2) == self.token_dim, f"Token dim mismatch: expected {self.token_dim}, got {tokens.size(2)}"
        assert X_query.size(2) == self.state_dim, f"State dim mismatch: expected {self.state_dim}, got {X_query.size(2)}"

        query_size = X_query.size(1)

        # Diagonal mask: queries evaluate independently, attending only to self and encoder memory
        tgt_mask = -torch.eye(query_size, device=X_query.device) + 1.0
        tgt_mask[tgt_mask == 1.0] = float('-inf')

        # Encode trajectory context (relying purely on continuous spatial coordinates)
        src = self.map_src(tokens)

        # Query vehicle articulation state
        tgt = self.map_tgt(X_query)

        # PyTorch Built-in Transformer execution
        if not return_attention:
            out = self.transformer(src, tgt, tgt_mask=tgt_mask.to(tgt.dtype))
            attn_weights = None
        else:
            # Step through decoder to extract exact cross-attention weights
            memory = self.transformer.encoder(src)
            x = tgt
            for layer in self.transformer.decoder.layers[:-1]:
                x = layer(x, memory, tgt_mask=tgt_mask.to(tgt.dtype))

            last_layer = self.transformer.decoder.layers[-1]
            x_sa = last_layer.norm1(x + last_layer._sa_block(x, tgt_mask.to(tgt.dtype), None))
            attn_out, attn_weights = last_layer.multihead_attn(
                x_sa, memory, memory,
                need_weights=True,
                average_attn_weights=True
            )
            x = last_layer.norm2(x_sa + last_layer.dropout2(attn_out))
            out = last_layer.norm3(x + last_layer._ff_block(x))

        out = self.map_out(out)

        if is_2d_query:
            out = out.squeeze(1)
            if attn_weights is not None:
                attn_weights = attn_weights.squeeze(1)

        if return_attention:
            return out, attn_weights
        return out

    @torch.no_grad()
    def predict_action(
        self,
        tokens_np: np.ndarray,
        hitch_state_np: np.ndarray,
        device: torch.device,
        use_expectation: bool = True,
        temperature: float = 1.5
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Single-sample runtime inference helper for game simulation execution.
        Returns:
            steer_rad: Continuous steering angle (via expectation or argmax)
            attn_weights: (num_tokens,) array
            probs: (num_classes,) probability distribution
        """
        self.eval()
        t_tok = torch.from_numpy(tokens_np).float().unsqueeze(0).to(device)
        t_state = torch.from_numpy(hitch_state_np).float().unsqueeze(0).to(device)

        logits, attn = self(t_tok, t_state, return_attention=True)
        temp = max(float(temperature), 1e-4)
        probs = F.softmax(logits / temp, dim=-1).cpu().numpy()[0]

        if use_expectation:
            # Softmax expected value: eliminates quantization chatter
            steer_rad = float(np.sum(probs * np.array(STEER_CLASSES_RAD)))
        else:
            best_idx = int(np.argmax(probs))
            steer_rad = class_idx_to_steer(best_idx)

        if attn is not None:
            attn_weights = attn.cpu().numpy()[0]
        else:
            attn_weights = np.ones(tokens_np.shape[0], dtype=np.float32) / float(tokens_np.shape[0])
        return steer_rad, attn_weights, probs
