"""
tests/test_data.py - Unit tests for audio loading, feature extraction, and augmentations
Compatible with pytest and standard python -m unittest.
"""

import os
import sys
import unittest
import torch
import torchaudio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from data import MultiResMelExtractor, SAMPLE_RATE, N_MELS, FIXED_FRAMES, load_audio, extract_static_delta_features


class TestDataProcessing(unittest.TestCase):
    def test_multires_mel_extractor_shape(self):
        extractor = MultiResMelExtractor(sample_rate=SAMPLE_RATE, n_mels=N_MELS, fixed_frames=FIXED_FRAMES, augment=False)
        wav = torch.sin(2 * 3.14159 * 440 * torch.linspace(0, 5, SAMPLE_RATE * 5)).unsqueeze(0)
        feat = extractor(wav)
        self.assertEqual(feat.shape, (3, N_MELS, FIXED_FRAMES))
        self.assertFalse(torch.isnan(feat).any())

    def test_extractor_with_augmentations(self):
        extractor = MultiResMelExtractor(sample_rate=SAMPLE_RATE, n_mels=N_MELS, fixed_frames=FIXED_FRAMES, augment=True)
        wav = torch.randn(1, SAMPLE_RATE * 5)
        feat = extractor(wav)
        self.assertEqual(feat.shape, (3, N_MELS, FIXED_FRAMES))
        self.assertFalse(torch.isnan(feat).any())

    def test_static_delta_extraction(self):
        wav = torch.randn(1, SAMPLE_RATE * 5)
        feat = extract_static_delta_features(wav, sample_rate=SAMPLE_RATE, n_mels=N_MELS)
        self.assertEqual(feat.shape[0], 3)
        self.assertEqual(feat.shape[1], N_MELS)
        self.assertFalse(torch.isnan(feat).any())

    def test_load_audio_sample(self):
        sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_sample.wav"))
        if os.path.exists(sample_path):
            wav, sr = load_audio(sample_path)
            self.assertIsInstance(wav, torch.Tensor)
            self.assertEqual(wav.ndim, 2)
            self.assertGreater(sr, 0)


if __name__ == "__main__":
    unittest.main()
