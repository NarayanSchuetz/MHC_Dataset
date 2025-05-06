import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Union
import math

from torch.nn.modules.normalization import RMSNorm
from torch.nn.functional import scaled_dot_product_attention

import numpy as np

from models.revin import RevIN

from einops import rearrange, einsum
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
    ):
        """
        Args:
            d_k (int): Embedding dimension size for the query or key tensor.
            theta (float): RoPE parameter.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.register_buffer("sin", torch.empty(max_seq_len, d_k // 2), persistent=False)
        self.register_buffer("cos", torch.empty(max_seq_len, d_k // 2), persistent=False)
        i: Float[Tensor, " max_seq_len"] = torch.arange(max_seq_len)
        k: Float[Tensor, " n_pairs"] = torch.arange(d_k // 2)

        base = theta ** (-2 * k / d_k)
        angles = einsum(i, base, "max_seq_len, n_pairs -> max_seq_len n_pairs")
        self.sin, self.cos = torch.sin(angles), torch.cos(angles)

    def forward(
        self, x: Float[Tensor, " ... sequence_length d_k"], token_positions: Int[Tensor, " ... sequence_length"]
    ) -> Float[Tensor, " ... sequence_length d_k"]:
        """
        Run RoPE for a given input tensor.

        Args:
            x (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
            token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
        Returns:
            Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
        """
        x_2d = rearrange(x, " ... seq_len (n_pairs two) -> ... seq_len n_pairs two", two=2)
        sin_pos, cos_pos = (
            rearrange(self.sin[token_positions], " ... seq_len n_pairs -> ... 1 seq_len n_pairs 1"),
            rearrange(self.cos[token_positions], " ... seq_len n_pairs -> ... 1 seq_len n_pairs 1"),
        )
        x_even, x_odd = x_2d[..., 0], x_2d[..., 1]
        x_even_out = x_even * cos_pos[..., 0] - x_odd * sin_pos[..., 0]
        x_odd_out = x_even * sin_pos[..., 0] + x_odd * cos_pos[..., 0]
        x_out = rearrange([x_even_out, x_odd_out], "two ... seq_len n_pairs -> ... seq_len (n_pairs two)")

        return x_out


class SwiGLU(nn.Module):
    """
    SwiGLU activation:
      Split dim in half: x[:,:hidden] and x[:,hidden:].
      Then do x1 * sigmoid(x2).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)


class TransformerBlock(nn.Module):
    """
    A single Transformer block (pre-norm) with:
      - RMSNorm
      - Multi-head self-attention
      - Rotary Positional Embeddings
      - Feed-forward network (SwiGLU)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        ffn_ratio: float = 4.0,
        max_seq_len: int = 4096,
        rope_base: float = 10000.0,
    ):
        """
        Args:
            d_model: Hidden dimension of the model
            n_heads: Number of attention heads
            dropout: Dropout probability
            ffn_ratio: Multiplicative factor for feed-forward hidden dimension
            max_seq_len: Max sequence length for precomputing rotary embeddings
            rope_base: The base "theta" for RoPE
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        # RMSNorm layers for pre-norm
        self.norm1 = RMSNorm(d_model, elementwise_affine=True)
        self.norm2 = RMSNorm(d_model, elementwise_affine=True)

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # RoPE
        self.rope = RotaryPositionalEmbedding(theta=rope_base, d_k=self.head_dim, max_seq_len=max_seq_len)

        self.attn_dropout = dropout

        # FFN: SwiGLU
        ffn_hidden = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * ffn_hidden),  # 2 for SwiGLU
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
        )
        self.resid_dropout = nn.Dropout(dropout)

        # Projection after multi-head attention
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. We expect token_positions to have shape (batch, seq_len).

        Args:
            x: (batch, seq_len, d_model)
            token_positions: (batch, seq_len)
        """
        b, s, _ = x.shape

        # === 1) Pre-norm & MHA ===
        normed_x = self.norm1(x)

        q = self.q_proj(normed_x)  # (b, s, d_model)
        k = self.k_proj(normed_x)  # (b, s, d_model)
        v = self.v_proj(normed_x)  # (b, s, d_model)

        # Reshape for multi-head attention => (b, s, n_heads, head_dim)
        q = q.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)  # (b, n_heads, s, head_dim)
        k = k.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to q, k
        # rope expects shape (b, n_heads, s, head_dim) plus token_positions => (b, s)
        q = self.rope(q, token_positions)  # (b, n_heads, s, head_dim)
        k = self.rope(k, token_positions)

        # scaled_dot_product_attention in PyTorch 2.0
        attn_out = scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )  # (b, n_heads, s, head_dim)

        # Merge heads back
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        attn_out = self.out_proj(attn_out)
        attn_out = self.resid_dropout(attn_out)

        # Residual
        x = x + attn_out

        # === 2) FFN ===
        normed_x2 = self.norm2(x)
        ffn_out = self.ffn(normed_x2)
        ffn_out = self.resid_dropout(ffn_out)

        # Residual
        x = x + ffn_out
        return x


################################################################
# Main ForecastingTransformer & RevInForecastingTransformer
################################################################

class ForecastingTransformer(nn.Module):
    """
    Baseline Transformer model for MHC dataset time-series data.
    Similar high-level interface to the LSTM version:
      - Preprocess (segment) each day into 30-min segments
      - Pass them through a Transformer to predict future segments
      - (Optional) label prediction from the final hidden state
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        ffn_ratio: float = 4.0,
        target_labels: Optional[List[str]] = None,
        prediction_horizon: int = 1,
        use_masked_loss: bool = False,
        num_features: int = 24,
        max_seq_len: int = 4096,
        rope_base: float = 10000.0,
    ):
        """
        Args:
            d_model: Transformer embedding dimension
            n_heads: Number of attention heads
            num_layers: Number of transformer blocks
            dropout: Dropout probability
            ffn_ratio: Expansion ratio for feed-forward
            target_labels: optional list of target labels
            prediction_horizon: how many future 30-min segments to predict
            use_masked_loss: whether to apply mask in loss
            num_features: number of features per minute
            max_seq_len: maximum sequence length for RoPE
            rope_base: base (theta) for RoPE
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.ffn_ratio = ffn_ratio
        self.target_labels = target_labels if target_labels else ["default"]
        self.prediction_horizon = prediction_horizon
        self.use_masked_loss = use_masked_loss
        self.num_features = num_features
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base

        # Data shape constants
        self.minutes_per_segment = 30
        self.segments_per_day = (24 * 60) // self.minutes_per_segment  # 48
        self.features_per_segment = self.num_features * self.minutes_per_segment  # e.g. 720 if 24 features

        # Input embedding from segment dimension -> d_model
        self.input_embedding = nn.Linear(self.features_per_segment, d_model)

        # Stacked Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                ffn_ratio=ffn_ratio,
                max_seq_len=max_seq_len,
                rope_base=rope_base
            )
            for _ in range(num_layers)
        ])

        # Final decoder: d_model -> features_per_segment
        self.decoder = nn.Linear(d_model, self.features_per_segment)

        # Optional label predictions
        self.output_layers = nn.ModuleDict({
            label: nn.Linear(d_model, 1) for label in self.target_labels
        })

    def preprocess_batch(
        self,
        batch_data: torch.Tensor,
        batch_mask: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Preprocess batch data from shape (batch_size, num_days, 24, 1440)
        into shape (batch_size, num_segments, features_per_segment).
        """
        batch_size, num_days, features, minutes_per_day = batch_data.shape
        if minutes_per_day != 24 * 60:
            raise ValueError(f"Expected 1440 minutes per day, got {minutes_per_day}")

        segments_per_day = minutes_per_day // self.minutes_per_segment  # 48
        x = batch_data.reshape(
            batch_size, num_days, features, segments_per_day, self.minutes_per_segment
        )  # => (B, D, F, 48, 30)
        x = x.permute(0, 1, 3, 2, 4).reshape(batch_size, num_days * segments_per_day, features * self.minutes_per_segment)
        x = torch.nan_to_num(x, nan=0.0)

        if batch_mask is not None:
            mask = batch_mask.reshape(
                batch_size, num_days, features, segments_per_day, self.minutes_per_segment
            )
            mask = mask.permute(0, 1, 3, 2, 4).reshape(batch_size, num_days * segments_per_day, features * self.minutes_per_segment)
            return x, mask

        return x

    def forward(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict]],
        return_predictions: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            batch: Dictionary with:
              - 'data': (batch_size, num_days, 24, 1440)
              - 'mask': optional, same shape if use_masked_loss
              - 'labels': dict of label values
            return_predictions: whether to return label predictions
        """
        x = batch['data']
        mask = batch.get('mask') if self.use_masked_loss else None

        if mask is not None:
            x, mask = self.preprocess_batch(x, mask)
        else:
            x = self.preprocess_batch(x)

        # We need token positions for RoPE: shape (batch_size, seq_len)
        # For a simple baseline, let positions = [0, 1, 2, ..., seq_len-1] for each sample.
        batch_size, seq_len, _ = x.shape
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)

        # Split input/target
        input_segments = x[:, :-self.prediction_horizon, :]   # (B, seq_len - horizon, feats)
        target_segments = x[:, self.prediction_horizon:, :]   # (B, seq_len - horizon, feats)

        # Build embeddings
        emb = self.input_embedding(input_segments)  # (B, seq_len-horizon, d_model)

        # For the Transformer blocks, we'll do shape => (B, seq_len, d_model)
        hidden_states = emb
        # Keep track of token positions for the "input" part only
        input_positions = token_positions[:, :-self.prediction_horizon]

        for block in self.blocks:
            hidden_states = block(hidden_states, input_positions)

        # Decode
        decoded_segments = self.decoder(hidden_states)  # (B, seq_len-horizon, features_per_segment)

        result = {
            'sequence_output': decoded_segments,
            'target_segments': target_segments
        }

        if self.use_masked_loss and (mask is not None):
            # Align mask with target
            target_mask = mask[:, self.prediction_horizon:, :]
            result['target_mask'] = target_mask

        if return_predictions:
            # We'll take the last time step's hidden state for label predictions
            final_hidden = hidden_states[:, -1, :]  # (B, d_model)
            label_preds = {}
            for label in self.target_labels:
                label_preds[label] = self.output_layers[label](final_hidden).squeeze(-1)
            result['label_predictions'] = label_preds

        return result

    def compute_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        batch: Dict[str, Union[torch.Tensor, Dict]]
    ) -> torch.Tensor:
        """
        Compute combined MSE loss for sequence + optional label prediction loss.
        """
        sequence_output = model_output['sequence_output']  # (B, seq_len-horizon, feats)
        target_segments = model_output['target_segments']  # (B, seq_len-horizon, feats)

        if self.use_masked_loss and ('target_mask' in model_output):
            target_mask = model_output['target_mask']
            squared_error = (sequence_output - target_segments) ** 2
            masked_error = squared_error * target_mask
            mask_sum = target_mask.sum() + 1e-10
            seq_loss = masked_error.sum() / mask_sum
        else:
            seq_loss = F.mse_loss(sequence_output, target_segments)

        total_loss = seq_loss

        # Label predictions
        if 'label_predictions' in model_output and 'labels' in batch:
            label_preds = model_output['label_predictions']
            labels_dict = batch['labels']
            for label in self.target_labels:
                if label in labels_dict:
                    label_value = labels_dict[label]
                    if not isinstance(label_value, torch.Tensor):
                        label_value = torch.tensor([label_value], device=label_preds[label].device)
                    elif label_value.dim() == 0:
                        label_value = label_value.unsqueeze(0)

                    valid_mask = ~torch.isnan(label_value)
                    if valid_mask.any():
                        pred = label_preds[label]
                        label_value_clean = torch.where(valid_mask, label_value, torch.zeros_like(label_value))
                        label_loss = F.mse_loss(pred[valid_mask], label_value_clean[valid_mask])
                        total_loss += label_loss

        return total_loss

    def predict_future(self, input_sequence: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Predict future steps (auto-regressive).
        input_sequence: (B, seq_len, features_per_segment)
        returns (B, steps, features_per_segment)
        """
        self.eval()
        with torch.no_grad():
            # We'll keep a running 'current_seq' that we feed each time
            current_seq = input_sequence
            b, seq_len, _ = current_seq.shape
            future_preds = []

            for step in range(steps):
                # The positions array extends as we add new timesteps
                positions = torch.arange(current_seq.shape[1], device=current_seq.device).unsqueeze(0).repeat(b, 1)

                emb = self.input_embedding(current_seq)  # (B, seq_len, d_model)
                hidden_states = emb

                for block in self.blocks:
                    hidden_states = block(hidden_states, positions)

                # Decode last position
                last_hidden = hidden_states[:, -1:, :]  # (B, 1, d_model)
                decoded = self.decoder(last_hidden)     # (B, 1, features_per_segment)
                future_preds.append(decoded)

                # Append to current sequence
                current_seq = torch.cat([current_seq, decoded], dim=1)

            return torch.cat(future_preds, dim=1)


class RevInForecastingTransformer(ForecastingTransformer):
    """
    Transformer-based model with RevIN normalization.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        ffn_ratio: float = 4.0,
        target_labels: Optional[List[str]] = None,
        prediction_horizon: int = 1,
        use_masked_loss: bool = False,
        num_features: int = 24,
        max_seq_len: int = 4096,
        rope_base: float = 10000.0,
        rev_in_affine: bool = False,
        rev_in_subtract_last: bool = False
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
            ffn_ratio=ffn_ratio,
            target_labels=target_labels,
            prediction_horizon=prediction_horizon,
            use_masked_loss=use_masked_loss,
            num_features=num_features,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
        )
        # RevIN
        self.rev_in = RevIN(
            num_features=self.features_per_segment,
            affine=rev_in_affine,
            subtract_last=rev_in_subtract_last
        )

    def forward(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict]],
        return_predictions: bool = True
    ) -> Dict[str, torch.Tensor]:
        x = batch['data']
        mask = batch.get('mask') if self.use_masked_loss else None

        if mask is not None:
            x, mask = self.preprocess_batch(x, mask)
        else:
            x = self.preprocess_batch(x)

        # RevIN normalize entire sequence
        x_norm = self.rev_in(x, mode='norm')

        # Split into input/target in normalized space
        input_segments_norm = x_norm[:, :-self.prediction_horizon, :]
        target_segments_norm = x_norm[:, self.prediction_horizon:, :]

        # Build positions for the input part
        b, seq_len, _ = x_norm.shape
        token_positions = torch.arange(seq_len, device=x_norm.device).unsqueeze(0).repeat(b, 1)
        input_positions = token_positions[:, :-self.prediction_horizon]

        # Transformer
        emb = self.input_embedding(input_segments_norm)
        hidden_states = emb
        for block in self.blocks:
            hidden_states = block(hidden_states, input_positions)

        # Decode
        decoded_norm = self.decoder(hidden_states)

        # De-normalize output and target
        sequence_output = self.rev_in(decoded_norm, mode='denorm')
        target_segments = self.rev_in(target_segments_norm, mode='denorm')

        result = {
            'sequence_output': sequence_output,
            'target_segments': target_segments
        }

        if self.use_masked_loss and mask is not None:
            target_mask = mask[:, self.prediction_horizon:, :]
            result['target_mask'] = target_mask

        if return_predictions:
            final_hidden = hidden_states[:, -1, :]
            label_preds = {}
            for label in self.target_labels:
                label_preds[label] = self.output_layers[label](final_hidden).squeeze(-1)
            result['label_predictions'] = label_preds

        return result

    def predict_future(self, input_sequence: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Autoregressive prediction with RevIN normalization.
        """
        self.eval()
        with torch.no_grad():
            # Normalize the entire known sequence
            current_seq_norm = self.rev_in(input_sequence, mode='norm')
            b, seq_len, _ = current_seq_norm.shape

            future_preds_norm = []
            for _ in range(steps):
                positions = torch.arange(current_seq_norm.shape[1], device=current_seq_norm.device).unsqueeze(0).repeat(b, 1)
                emb = self.input_embedding(current_seq_norm)
                hidden_states = emb
                for block in self.blocks:
                    hidden_states = block(hidden_states, positions)

                decoded_norm = self.decoder(hidden_states[:, -1:, :])
                future_preds_norm.append(decoded_norm)
                current_seq_norm = torch.cat([current_seq_norm, decoded_norm], dim=1)

            final_pred_norm = torch.cat(future_preds_norm, dim=1)
            final_pred = self.rev_in(final_pred_norm, mode='denorm')
            return final_pred


class TransformerTrainer:
    """
    Trainer for the ForecastingTransformer or RevInForecastingTransformer.
    """

    def __init__(
        self,
        model: ForecastingTransformer,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.model.to(device)

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            batch_data = batch['data'].to(self.device)
            batch['data'] = batch_data
            if self.model.use_masked_loss and 'mask' in batch:
                batch_mask = batch['mask'].to(self.device)
                batch['mask'] = batch_mask

            self.optimizer.zero_grad()
            output = self.model(batch)
            loss = self.model.compute_loss(output, batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / len(dataloader)

    def validate(self, dataloader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                batch_data = batch['data'].to(self.device)
                batch['data'] = batch_data
                if self.model.use_masked_loss and 'mask' in batch:
                    batch_mask = batch['mask'].to(self.device)
                    batch['mask'] = batch_mask

                output = self.model(batch)
                loss = self.model.compute_loss(output, batch)
                total_loss += loss.item()

        return total_loss / len(dataloader)
