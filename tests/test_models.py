"""
tests/test_models.py - Unit tests for neural network architectures
Compatible with pytest and standard python -m unittest.
"""

import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from models import SingleResCNN, MultiResAttentionNet, MultiFeatureCoordNet, TimeFreqCoordAttention, TimeFreqAttention


class TestModelArchitectures(unittest.TestCase):
    def test_time_freq_attention(self):
        attn = TimeFreqAttention(channels=64, reduction=8)
        x = torch.randn(2, 64, 16, 27)
        out = attn(x)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_time_freq_coord_attention(self):
        coord_attn = TimeFreqCoordAttention(in_channels=64, reduction=8)
        x = torch.randn(2, 64, 16, 27)
        out = coord_attn(x)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_single_res_cnn_forward(self):
        model = SingleResCNN(n_classes=50, pretrained=False, freeze_early=True)
        model.eval()
        x = torch.randn(2, 3, 64, 216)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, 50))
        self.assertFalse(torch.isnan(out).any())

    def test_multi_res_attention_net_forward(self):
        model = MultiResAttentionNet(n_classes=50, pretrained=False, freeze_early=True)
        model.eval()
        x = torch.randn(2, 3, 64, 216)
        with torch.no_grad():
            out = model(x)
            weights = model.get_fusion_weights(x)
        self.assertEqual(out.shape, (2, 50))
        self.assertEqual(weights.shape, (2, 3))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5))

    def test_multifeature_coord_net_forward(self):
        model = MultiFeatureCoordNet(n_classes=50, pretrained=False, freeze_early=True)
        model.eval()
        x = torch.randn(2, 3, 64, 216)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, 50))
        self.assertFalse(torch.isnan(out).any())

    def test_unfreeze_and_param_groups(self):
        frozen_model = MultiFeatureCoordNet(n_classes=50, pretrained=False, freeze_early=True)
        unfrozen_model = MultiFeatureCoordNet(n_classes=50, pretrained=False, freeze_early=False)

        frozen_trainable = sum(p.numel() for p in frozen_model.parameters() if p.requires_grad)
        unfrozen_trainable = sum(p.numel() for p in unfrozen_model.parameters() if p.requires_grad)

        self.assertGreater(unfrozen_trainable, frozen_trainable)

        groups = unfrozen_model.get_param_groups(head_lr=1e-3, backbone_lr=1e-4)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["lr"], 1e-4)
        self.assertEqual(groups[1]["lr"], 1e-3)
        self.assertGreater(len(groups[0]["params"]), 0)
        self.assertGreater(len(groups[1]["params"]), 0)

    def test_gradient_backward_pass(self):
        model = MultiFeatureCoordNet(n_classes=50, pretrained=False, freeze_early=False)
        model.train()
        x = torch.randn(2, 3, 64, 216)
        target = torch.tensor([5, 42])
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, target)
        loss.backward()
        self.assertIsNotNone(model.classifier[1].weight.grad)


if __name__ == "__main__":
    unittest.main()
