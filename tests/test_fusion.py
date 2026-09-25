"""
Unit and Integration Tests for PyTorch Multi-Source Fusion Architectures.
"""

import os
import pytest
import numpy as np
import torch

from fusion.gated_fusion import SensorOnlyNet, SimpleFusionNet, GatedFusionNet, ReliabilityGatedFusionNet
from fusion.dataset import SoilMultiModalDataset, get_multimodal_dataloaders


@pytest.fixture
def batch_tensors():
    B = 16
    x_s = torch.randn(B, 7)
    z_t = torch.randn(B, 384)
    # L2 normalize text
    z_t = torch.nn.functional.normalize(z_t, p=2, dim=1)
    r = torch.rand(B, 1)
    y = torch.randint(0, 22, (B,))
    return x_s, z_t, r, y


def test_tensor_shapes_all_models(batch_tensors):
    x_s, z_t, r, y = batch_tensors
    B = x_s.shape[0]

    m1 = SensorOnlyNet(latent_dim=64, num_classes=22)
    m2 = SimpleFusionNet(latent_dim=64, num_classes=22)
    m3 = GatedFusionNet(latent_dim=64, num_classes=22)
    m4 = ReliabilityGatedFusionNet(latent_dim=64, num_classes=22)

    out1 = m1(x_s)
    out2 = m2(x_s, z_t)
    out3, g3 = m3(x_s, z_t, return_gate=True)
    out4, g4 = m4(x_s, z_t, reliability=r, return_gate=True)

    assert out1.shape == (B, 22)
    assert out2.shape == (B, 22)
    assert out3.shape == (B, 22)
    assert out4.shape == (B, 22)

    assert g3.shape == (B, 64)
    assert g4.shape == (B, 64)


def test_sigmoid_gate_bounds(batch_tensors):
    x_s, z_t, _, _ = batch_tensors
    m = GatedFusionNet(latent_dim=64, num_classes=22)
    _, g = m(x_s, z_t, return_gate=True)

    assert (g >= 0.0).all() and (g <= 1.0).all()


def test_reliability_zero_suppression(batch_tensors):
    x_s, z_t, _, _ = batch_tensors
    B = x_s.shape[0]
    r_zero = torch.zeros(B, 1)

    m = ReliabilityGatedFusionNet(latent_dim=64, num_classes=22)
    _, effective_gate = m(x_s, z_t, reliability=r_zero, return_gate=True)

    # When reliability is zero, effective gate must be identically zero
    np.testing.assert_allclose(effective_gate.detach().numpy(), np.zeros((B, 64)), atol=1e-6)


def test_dataset_and_dataloaders():
    train_loader, test_loader = get_multimodal_dataloaders(batch_size=16)

    for x_s, z_t, r, y in train_loader:
        assert x_s.shape[1] == 7
        assert z_t.shape[1] == 384
        assert r.shape[1] == 1
        assert len(y) == len(x_s)
        break


def test_saved_fusion_checkpoints_loading():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt_dir = os.path.join(base_dir, "models", "checkpoints")

    gated_ckpt = os.path.join(ckpt_dir, "gatedfusionnet.pt")
    rel_ckpt = os.path.join(ckpt_dir, "reliabilitygatedfusionnet.pt")

    assert os.path.exists(gated_ckpt)
    assert os.path.exists(rel_ckpt)

    m_gated = GatedFusionNet(latent_dim=64, num_classes=22)
    m_gated.load_state_dict(torch.load(gated_ckpt, weights_only=True))
    m_gated.eval()

    test_x = torch.randn(2, 7)
    test_z = torch.randn(2, 384)
    with torch.no_grad():
        out = m_gated(test_x, test_z)
    assert out.shape == (2, 22)
