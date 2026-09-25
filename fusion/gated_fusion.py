"""
PyTorch Multi-Source Fusion Architectures.

Implements 4 distinct architectures for comparative research evaluation:
1. SensorOnlyNet: Unimodal numerical baseline.
2. SimpleFusionNet: Early/joint concatenation baseline.
3. GatedFusionNet: Dynamic sigmoid gating mechanism.
4. ReliabilityGatedFusionNet: Adaptive reliability-conditioned gated fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class SensorOnlyNet(nn.Module):
    """
    Unimodal Numerical Sensor Classifier.
    """
    def __init__(self, sensor_dim: int = 7, latent_dim: int = 64, num_classes: int = 22, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(sensor_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x_sensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        e_s = self.encoder(x_sensor)
        logits = self.classifier(e_s)
        return logits


class SimpleFusionNet(nn.Module):
    """
    Simple Multi-Source Fusion via Unweighted Concatenation.
    """
    def __init__(
        self,
        sensor_dim: int = 7,
        text_dim: int = 384,
        latent_dim: int = 64,
        num_classes: int = 22,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
            nn.Dropout(dropout),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
            nn.Dropout(dropout),
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x_sensor: torch.Tensor, z_text: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        e_s = self.sensor_encoder(x_sensor)
        e_t = self.text_encoder(z_text)
        concat = torch.cat([e_s, e_t], dim=-1)
        z_fused = self.fusion_layer(concat)
        logits = self.classifier(z_fused)
        return logits


class GatedFusionNet(nn.Module):
    """
    Dynamic Sigmoid Gated Multi-Source Fusion Network.
    Formula:
        e_s = LayerNorm(ReLU(W_s x_s + b_s))
        e_t = LayerNorm(ReLU(W_t z_text + b_t))
        g = sigma(W_g [e_s ; e_t] + b_g)
        z_fused = g * e_s + (1 - g) * e_t
        logits = W_o z_fused + b_o
    """
    def __init__(
        self,
        sensor_dim: int = 7,
        text_dim: int = 384,
        latent_dim: int = 64,
        num_classes: int = 22,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Modality Encoders
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
            nn.Dropout(dropout),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
            nn.Dropout(dropout),
        )

        # Dynamic Gating Unit
        self.gate_layer = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid(),
        )

        # Output Classification Head
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(
        self, x_sensor: torch.Tensor, z_text: torch.Tensor, return_gate: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        e_s = self.sensor_encoder(x_sensor)
        e_t = self.text_encoder(z_text)

        # Compute dynamic gate g in (0, 1)^d
        g = self.gate_layer(torch.cat([e_s, e_t], dim=-1))

        # Convex combination element-wise fusion
        z_fused = g * e_s + (1.0 - g) * e_t

        logits = self.classifier(z_fused)
        if return_gate:
            return logits, g
        return logits


class ReliabilityGatedFusionNet(nn.Module):
    """
    Reliability-Aware Dynamic Gated Multi-Source Fusion Network.
    Formula:
        e_s = LayerNorm(ReLU(W_s x_s + b_s))
        e_t = LayerNorm(ReLU(W_t z_text + b_t))
        g = sigma(W_g [e_s ; e_t ; r] + b_g)
        effective_gate = r * g
        z_fused = effective_gate * e_s + (1 - effective_gate) * e_t
        logits = W_o z_fused + b_o
    """
    def __init__(
        self,
        sensor_dim: int = 7,
        text_dim: int = 384,
        latent_dim: int = 64,
        num_classes: int = 22,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Modality Encoders
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
            nn.Dropout(dropout),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
            nn.Dropout(dropout),
        )

        # Reliability-Conditioned Dynamic Gate: Input dim = 2*latent_dim + 1
        self.gate_layer = nn.Sequential(
            nn.Linear(latent_dim * 2 + 1, latent_dim),
            nn.Sigmoid(),
        )

        # Output Classification Head
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(
        self,
        x_sensor: torch.Tensor,
        z_text: torch.Tensor,
        reliability: Optional[torch.Tensor] = None,
        return_gate: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = x_sensor.shape[0]

        if reliability is None:
            r = torch.ones((batch_size, 1), dtype=torch.float32, device=x_sensor.device)
        elif reliability.ndim == 1:
            r = reliability.unsqueeze(1).to(x_sensor.device)
        else:
            r = reliability.to(x_sensor.device)

        e_s = self.sensor_encoder(x_sensor)
        e_t = self.text_encoder(z_text)

        # Dynamic Gating conditioned on e_s, e_t, and r
        gate_input = torch.cat([e_s, e_t, r], dim=-1)
        g = self.gate_layer(gate_input)

        # Reliability-weighted adaptive convex combination
        effective_gate = r * g
        z_fused = effective_gate * e_s + (1.0 - effective_gate) * e_t

        logits = self.classifier(z_fused)
        if return_gate:
            return logits, effective_gate
        return logits
