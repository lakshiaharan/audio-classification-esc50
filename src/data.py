import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 44100
N_MELS = 64
FIXED_FRAMES = 216

# fine/mid/coarse resolutions - vary hop length (not n_fft) so all 3 keep enough
# freq bins for 64 mel filters (n_fft=512 was giving empty filterbanks)
RESOLUTIONS = {
    "fine":   dict(n_fft=1024, hop_length=128),
    "mid":    dict(n_fft=1024, hop_length=512),
    "coarse": dict(n_fft=2048, hop_length=1024),
}


class MultiResMelExtractor:
    def __init__(self, sample_rate=SAMPLE_RATE, n_mels=N_MELS, fixed_frames=FIXED_FRAMES, augment=False):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.fixed_frames = fixed_frames
        self.augment = augment
        self.transforms = {
            name: torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=cfg["n_fft"], hop_length=cfg["hop_length"], n_mels=n_mels
            )
            for name, cfg in RESOLUTIONS.items()
        }
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=12)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=24)

    def _process_one(self, wav, name):
        mel = self.transforms[name](wav)
        mel = self.db(mel)
        mel = torch.nn.functional.interpolate(
            mel.unsqueeze(0), size=(self.n_mels, self.fixed_frames), mode="bilinear", align_corners=False
        ).squeeze(0)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        if self.augment:
            mel = self.freq_mask(mel)
            mel = self.time_mask(mel)
        return mel

    def __call__(self, wav):
        feats = [self._process_one(wav, name) for name in ["fine", "mid", "coarse"]]
        return torch.cat(feats, dim=0)


def load_audio(path):
    """Robust audio loader supporting torchaudio, soundfile, and scipy backends."""
    try:
        wav, sr = torchaudio.load(path, backend="soundfile")
        return wav, sr
    except Exception:
        pass
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32")
        wav = torch.from_numpy(data)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.transpose(0, 1)
        return wav, sr
    except Exception:
        import numpy as np
        import scipy.io.wavfile as wavfile
        sr, data = wavfile.read(path)
        wav = torch.from_numpy(data).float()
        if data.dtype == np.int16:
            wav = wav / 32768.0
        elif data.dtype == np.int32:
            wav = wav / 2147483648.0
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.transpose(0, 1)
        return wav, sr


MULTIRES_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "esc50_features.pt")
STATIC_DELTA_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "esc50_static_delta.pt")
_FEATURE_CACHES = {}


def extract_static_delta_features(wav, sample_rate=SAMPLE_RATE, n_mels=N_MELS):
    """Extract 3-channel [Static Mel, Delta 1 (Velocity), Delta 2 (Acceleration)]."""
    mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=n_mels)
    db_tf = torchaudio.transforms.AmplitudeToDB(top_db=80)
    mel = db_tf(mel_tf(wav))
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    delta1 = torchaudio.functional.compute_deltas(mel)
    delta2 = torchaudio.functional.compute_deltas(delta1)
    return torch.cat([mel, delta1, delta2], dim=0)


def precompute_and_cache_features(root_dir, feature_type="multires", cache_path=None):
    """Precompute and cache features (multires or static+delta)."""
    global _FEATURE_CACHES
    if cache_path is None:
        cache_path = STATIC_DELTA_CACHE_PATH if feature_type == "multifeature" else MULTIRES_CACHE_PATH
    
    if cache_path in _FEATURE_CACHES:
        return _FEATURE_CACHES[cache_path]
    
    if os.path.exists(cache_path):
        print(f"Loading precomputed features from {cache_path}...")
        _FEATURE_CACHES[cache_path] = torch.load(cache_path)
        return _FEATURE_CACHES[cache_path]

    print(f"Precomputing {feature_type} features for ESC-50 (one-time setup)...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    meta = pd.read_csv(os.path.join(root_dir, "meta", "esc50.csv"))
    audio_dir = os.path.join(root_dir, "audio")
    
    clean_extractor = MultiResMelExtractor(sample_rate=SAMPLE_RATE, augment=False)
    cache = {}
    
    for idx in range(len(meta)):
        row = meta.iloc[idx]
        path = os.path.join(audio_dir, row.filename)
        wav, sr = load_audio(path)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            
        if feature_type == "multifeature":
            feat = extract_static_delta_features(wav) # (3, 64, 431)
        else:
            feat = clean_extractor(wav)  # (3, 64, 216)
            
        cache[row.filename] = feat.cpu()
        if (idx + 1) % 500 == 0 or (idx + 1) == len(meta):
            print(f"  Processed {idx + 1}/{len(meta)} clips...")

    torch.save(cache, cache_path)
    print(f"Saved feature cache to {cache_path} ({os.path.getsize(cache_path)/1e6:.1f} MB)")
    _FEATURE_CACHES[cache_path] = cache
    return _FEATURE_CACHES[cache_path]


class ESC50Dataset(Dataset):
    def __init__(self, root_dir, folds, extractor=None, sample_rate=SAMPLE_RATE, feature_type="multires"):
        self.root_dir = root_dir
        meta = pd.read_csv(os.path.join(root_dir, "meta", "esc50.csv"))
        self.meta = meta[meta.fold.isin(folds)].reset_index(drop=True)
        self.sample_rate = sample_rate
        self.feature_type = feature_type
        self.extractor = extractor or MultiResMelExtractor(sample_rate=sample_rate)
        self.classes = sorted(meta.category.unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.cache = precompute_and_cache_features(root_dir, feature_type=feature_type)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        feat = self.cache[row.filename]
        
        if self.extractor.augment:
            feat = feat.clone()
            feat = self.extractor.freq_mask(feat)
            feat = self.extractor.time_mask(feat)
            if torch.rand(1).item() > 0.5:
                shift = torch.randint(-18, 18, (1,)).item()
                feat = torch.roll(feat, shifts=shift, dims=-1)
                
        label = self.class_to_idx[row.category]
        return feat, label


def get_dataloaders(root_dir, batch_size=32, test_fold=5, val_fold=4, num_workers=0, feature_type="multires"):
    from torch.utils.data import DataLoader
    train_folds = [f for f in [1, 2, 3, 4, 5] if f not in (test_fold, val_fold)]
    train_extractor = MultiResMelExtractor(augment=True)
    eval_extractor = MultiResMelExtractor(augment=False)
    train_ds = ESC50Dataset(root_dir, train_folds, train_extractor, feature_type=feature_type)
    val_ds = ESC50Dataset(root_dir, [val_fold], eval_extractor, feature_type=feature_type)
    test_ds = ESC50Dataset(root_dir, [test_fold], eval_extractor, feature_type=feature_type)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        train_ds.classes,
    )
