"""
predict.py
Load a trained checkpoint and predict the ESC-50 sound class of a single
uploaded .wav file, printing the top prediction + top-5 with confidence.
"""
import sys
import torch
import torch.nn.functional as F
import torchaudio
import pandas as pd

sys.path.insert(0, ".")
from data import MultiResMelExtractor, SAMPLE_RATE, load_audio
from models import MultiResAttentionNet, SingleResCNN, MultiFeatureCoordNet


def load_classes(esc50_root="ESC-50"):
    meta = pd.read_csv(f"{esc50_root}/meta/esc50.csv")
    return sorted(meta.category.unique())


def extract_features(wav, model_type="multifeature"):
    if model_type == "multifeature":
        mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_fft=1024, hop_length=512, n_mels=64)
        db_tf = torchaudio.transforms.AmplitudeToDB(top_db=80)
        mel = db_tf(mel_tf(wav))
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        delta = torchaudio.functional.compute_deltas(mel)
        delta2 = torchaudio.functional.compute_deltas(delta)
        return torch.cat([mel, delta, delta2], dim=0).unsqueeze(0)
    else:
        extractor = MultiResMelExtractor()
        return extractor(wav).unsqueeze(0)


def predict(audio_path, model_type="multifeature", checkpoint=None,
            esc50_root="ESC-50", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes(esc50_root)
    n_classes = len(classes)

    ckpt = checkpoint or f"results/{model_type}_best.pt"

    if model_type == "multires":
        model = MultiResAttentionNet(n_classes, pretrained=False).to(device)
    elif model_type == "multifeature":
        model = MultiFeatureCoordNet(n_classes, pretrained=False).to(device)
    else:
        model = SingleResCNN(n_classes, pretrained=False).to(device)

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    wav, sr = load_audio(audio_path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # ESC-50 clips are 5s; pad or trim uploaded audio to match
    target_len = SAMPLE_RATE * 5
    if wav.shape[1] < target_len:
        wav = F.pad(wav, (0, target_len - wav.shape[1]))
    else:
        wav = wav[:, :target_len]

    feat = extract_features(wav, model_type=model_type).to(device)

    with torch.no_grad():
        logits = model(feat)
        probs = torch.softmax(logits, dim=1)[0]

    top5 = torch.topk(probs, 5)
    print(f"\nModel: {model_type} | Checkpoint: {ckpt}")
    print(f"Predicted class: {classes[top5.indices[0]]}  "
          f"({top5.values[0]*100:.1f}% confidence)\n")
    print("Top 5 Predictions:")
    for rank, (idx, p) in enumerate(zip(top5.indices, top5.values), 1):
        print(f"  {rank}. {classes[idx]:<22} {p*100:.1f}%")
    return classes[top5.indices[0]], top5.values[0].item()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument("--model", choices=["multires", "baseline", "multifeature"], default="multifeature")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--esc50_root", default="ESC-50")
    args = parser.parse_args()
    predict(args.audio_path, args.model, args.checkpoint, args.esc50_root)
