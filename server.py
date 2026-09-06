"""
server.py - Production-Grade Lightweight Web Server for Audio Classification
Serves the interactive frontend and provides inference endpoints for ESC-50 models.
"""

import os
import sys
import json
import time
import io
import cgi
import urllib.parse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import torch
import torch.nn.functional as F
import torchaudio
import pandas as pd

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data import MultiResMelExtractor, SAMPLE_RATE, load_audio
from models import MultiResAttentionNet, SingleResCNN, MultiFeatureCoordNet

PORT = 8000
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
ESC50_ROOT = os.path.join(os.path.dirname(__file__), "ESC-50")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Server] Initializing on device: {DEVICE}")

# Load category metadata
meta_csv_path = os.path.join(ESC50_ROOT, "meta", "esc50.csv")
meta_df = None
if os.path.exists(meta_csv_path):
    try:
        meta_df = pd.read_csv(meta_csv_path)
        CLASSES = sorted(meta_df["category"].unique().tolist())
    except Exception:
        meta_df = None
if meta_df is None:
    CLASSES = [
        'dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow',
        'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds', 'water_drops',
        'wind', 'pouring_water', 'toilet_flush', 'thunderstorm', 'crying_baby', 'sneezing',
        'clapping', 'breathing', 'coughing', 'footsteps', 'laughing', 'brushing_teeth',
        'snoring', 'drinking_sipping', 'door_wood_creaks', 'mouse_click', 'keyboard_typing',
        'door_wood_knock', 'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm',
        'clock_tick', 'glass_breaking', 'helicopter', 'chainsaw', 'siren', 'car_horn',
        'engine', 'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw'
    ]
N_CLASSES = len(CLASSES)

# Preload models
MODELS = {}

def get_model(model_type="baseline"):
    if model_type in MODELS:
        return MODELS[model_type]

    print(f"[Server] Loading model: {model_type}...")
    if model_type == "multifeature":
        ckpt_path = os.path.join(RESULTS_DIR, "multifeature_best.pt")
        baseline_path = os.path.join(RESULTS_DIR, "baseline_best.pt")
        if os.path.exists(ckpt_path):
            model = MultiFeatureCoordNet(n_classes=N_CLASSES, pretrained=False).to(DEVICE)
            state = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(state)
            print(f"[Server] Loaded checkpoint from {ckpt_path}")
        elif os.path.exists(baseline_path):
            # Fallback to baseline SingleResCNN checkpoint
            model = SingleResCNN(n_classes=N_CLASSES, pretrained=False).to(DEVICE)
            state = torch.load(baseline_path, map_location=DEVICE)
            model.load_state_dict(state)
            print(f"[Server] Using baseline checkpoint as active engine")
        else:
            model = MultiFeatureCoordNet(n_classes=N_CLASSES, pretrained=False).to(DEVICE)
            print(f"[Server] Checkpoints not found; initialized MultiFeatureCoordNet with default weights")
    elif model_type == "multires":
        model = MultiResAttentionNet(n_classes=N_CLASSES, pretrained=False).to(DEVICE)
        ckpt_path = os.path.join(RESULTS_DIR, "multires_best.pt")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(state)
            print(f"[Server] Loaded checkpoint from {ckpt_path}")
        else:
            print(f"[Server] Checkpoint not found; initialized MultiResAttentionNet with default weights")
    else:  # baseline
        model = SingleResCNN(n_classes=N_CLASSES, pretrained=False).to(DEVICE)
        ckpt_path = os.path.join(RESULTS_DIR, "baseline_best.pt")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(state)
            print(f"[Server] Loaded checkpoint from {ckpt_path}")
        else:
            print(f"[Server] Checkpoint not found; initialized SingleResCNN with default weights")

    model.eval()
    MODELS[model_type] = model
    return model

# Warm up primary model
get_model("multifeature")
get_model("baseline")
get_model("multires")

# Feature extractors
multi_res_extractor = MultiResMelExtractor()
mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_fft=1024, hop_length=512, n_mels=64)
db_tf = torchaudio.transforms.AmplitudeToDB(top_db=80)

def extract_features(wav, model_type="baseline"):
    if model_type == "multires":
        return multi_res_extractor(wav).unsqueeze(0)
    elif model_type == "multifeature" and os.path.exists(os.path.join(RESULTS_DIR, "multifeature_best.pt")):
        mel = db_tf(mel_tf(wav))
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        delta = torchaudio.functional.compute_deltas(mel)
        delta2 = torchaudio.functional.compute_deltas(delta)
        return torch.cat([mel, delta, delta2], dim=0).unsqueeze(0)
    else:  # baseline / fallback
        return multi_res_extractor(wav).unsqueeze(0)

# Build 50 curated presets dynamically from esc50.csv
GROUPS_METADATA = {
    'dog': ('Animals', 'Dog Barking', '🐕'),
    'rooster': ('Animals', 'Rooster Crowing', '🐓'),
    'pig': ('Animals', 'Pig Oinking', '🐖'),
    'cow': ('Animals', 'Cow Mooing', '🐄'),
    'frog': ('Animals', 'Frog Croaking', '🐸'),
    'cat': ('Animals', 'Cat Meowing', '🐱'),
    'hen': ('Animals', 'Hen Clucking', '🐔'),
    'insects': ('Animals', 'Insects / Bug', '🦗'),
    'sheep': ('Animals', 'Sheep Bleating', '🐑'),
    'crow': ('Animals', 'Crow Cawing', '🐦'),

    'rain': ('Nature', 'Rain Shower', '🌧️'),
    'sea_waves': ('Nature', 'Sea Waves', '🌊'),
    'crackling_fire': ('Nature', 'Crackling Fire', '🔥'),
    'crickets': ('Nature', 'Crickets Chirping', '🦗'),
    'chirping_birds': ('Nature', 'Chirping Birds', '🐦'),
    'water_drops': ('Nature', 'Water Drops', '💧'),
    'wind': ('Nature', 'Wind Gusts', '💨'),
    'pouring_water': ('Nature', 'Pouring Water', '🫗'),
    'toilet_flush': ('Nature', 'Toilet Flush', '🚽'),
    'thunderstorm': ('Nature', 'Thunderstorm', '⚡'),

    'crying_baby': ('Human', 'Crying Baby', '👶'),
    'sneezing': ('Human', 'Sneezing', '🤧'),
    'clapping': ('Human', 'Clapping Hands', '👏'),
    'breathing': ('Human', 'Breathing', '🫁'),
    'coughing': ('Human', 'Coughing', '😷'),
    'footsteps': ('Human', 'Footsteps Walking', '👣'),
    'laughing': ('Human', 'Laughing', '😄'),
    'brushing_teeth': ('Human', 'Brushing Teeth', '🪥'),
    'snoring': ('Human', 'Snoring', '💤'),
    'drinking_sipping': ('Human', 'Drinking / Sipping', '🥤'),

    'door_wood_creaks': ('Domestic', 'Door Wood Creak', '🚪'),
    'mouse_click': ('Domestic', 'Mouse Click', '🖱️'),
    'keyboard_typing': ('Domestic', 'Keyboard Typing', '⌨️'),
    'door_wood_knock': ('Domestic', 'Door Knocking', '🚪'),
    'can_opening': ('Domestic', 'Can Opening', '🥫'),
    'washing_machine': ('Domestic', 'Washing Machine', '🫧'),
    'vacuum_cleaner': ('Domestic', 'Vacuum Cleaner', '🧹'),
    'clock_alarm': ('Domestic', 'Clock Alarm Ringing', '⏰'),
    'clock_tick': ('Domestic', 'Clock Ticking', '⏱️'),
    'glass_breaking': ('Domestic', 'Glass Shattering', '🍸'),

    'helicopter': ('Urban', 'Helicopter Blades', '🚁'),
    'chainsaw': ('Urban', 'Chainsaw Motor', '🪚'),
    'siren': ('Urban', 'Siren Alarm', '🚨'),
    'car_horn': ('Urban', 'Car Horn Honk', '🚗'),
    'engine': ('Urban', 'Car Engine', '🏎️'),
    'train': ('Urban', 'Train Horn', '🚆'),
    'church_bells': ('Urban', 'Church Bells', '🔔'),
    'airplane': ('Urban', 'Airplane Jet', '✈️'),
    'fireworks': ('Urban', 'Fireworks Exploding', '🎆'),
    'hand_saw': ('Urban', 'Hand Saw', '🪚'),
}

PRESETS = []
for cat, (group, display_name, icon) in GROUPS_METADATA.items():
    fn = ""
    if meta_df is not None and not meta_df.empty:
        sub = meta_df[meta_df["category"] == cat]
        if not sub.empty:
            fn = sub.iloc[0]["filename"]
    PRESETS.append({
        "id": cat,
        "name": display_name,
        "icon": icon,
        "group": group,
        "category": cat,
        "filename": fn
    })


def handle_predict(raw_body, content_type="", model_type_header="multifeature"):
    audio_bytes = None
    model_type = "multifeature"

    if "multipart/form-data" in content_type:
        boundary = None
        for part in content_type.split(";"):
            part = part.strip()
            if part.startswith("boundary="):
                boundary = part.split("=", 1)[1].strip('"').strip("'").encode()
        
        if boundary:
            chunks = raw_body.split(b"--" + boundary)
            for chunk in chunks:
                if b'name="model_type"' in chunk:
                    lines = chunk.split(b"\r\n\r\n", 1)
                    if len(lines) > 1:
                        model_type = lines[1].split(b"\r\n")[0].decode().strip()
                elif b'name="audio"' in chunk:
                    lines = chunk.split(b"\r\n\r\n", 1)
                    if len(lines) > 1:
                        audio_bytes = lines[1].rsplit(b"\r\n", 1)[0]
    elif "application/json" in content_type:
        data = json.loads(raw_body.decode())
        model_type = data.get("model_type", "multifeature")
        if "filename" in data:
            preset_path = os.path.join(ESC50_ROOT, "audio", data["filename"])
            if os.path.exists(preset_path):
                with open(preset_path, "rb") as f:
                    audio_bytes = f.read()
    else:
        model_type = model_type_header or "multifeature"
        audio_bytes = raw_body

    if not audio_bytes:
        return {"error": "No audio payload received"}

    # Process audio
    bio = io.BytesIO(audio_bytes)
    wav, sr = load_audio(bio)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    target_len = SAMPLE_RATE * 5
    if wav.shape[1] < target_len:
        wav = F.pad(wav, (0, target_len - wav.shape[1]))
    else:
        wav = wav[:, :target_len]

    # Run inference
    model = get_model(model_type)
    feat = extract_features(wav, model_type=model_type).to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        logits = model(feat)
        probs = torch.softmax(logits, dim=1)[0]
    latency_ms = (time.time() - t0) * 1000

    top5 = torch.topk(probs, 5)
    top5_list = []
    for idx, p in zip(top5.indices, top5.values):
        top5_list.append({
            "class": CLASSES[idx.item()],
            "confidence": round(p.item() * 100, 2),
            "probability": round(p.item(), 4),
        })

    return {
        "success": True,
        "model": model_type,
        "latency_ms": round(latency_ms, 2),
        "top1": top5_list[0],
        "top5": top5_list,
        "duration_sec": 5.0,
        "sample_rate": SAMPLE_RATE,
    }


class AudioServerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/samples":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(PRESETS).encode())
            return

        elif path.startswith("/api/audio/"):
            filename = os.path.basename(path)
            audio_path = os.path.join(ESC50_ROOT, "audio", filename)
            if not os.path.exists(audio_path):
                audio_path = os.path.join(STATIC_DIR, "samples", filename)
            if os.path.exists(audio_path):
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(os.path.getsize(audio_path)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(audio_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Audio file not found")
            return

        elif path == "/api/metrics":
            metrics_data = {}
            for name, fname in [
                ("baseline", "baseline_metrics.json"),
                ("multires", "multires_metrics.json"),
                ("multifeature", "multifeature_metrics.json"),
            ]:
                fpath = os.path.join(RESULTS_DIR, fname)
                if os.path.exists(fpath):
                    with open(fpath) as f:
                        metrics_data[name] = json.load(f)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(metrics_data).encode())
            return
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        # Default static file serving
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/predict":
            try:
                content_type = self.headers.get("Content-Type", "")
                length = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length) if length > 0 else b""
                model_type_header = self.headers.get("X-Model-Type", "multifeature")
                res = handle_predict(raw_body, content_type, model_type_header)
                if "error" in res:
                    self.send_json_response(res, status=400)
                else:
                    self.send_json_response(res, status=200)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_json_response({"error": str(e)}, status=500)
            return

        self.send_error(404, "Endpoint not found")

    def send_json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_server():
    os.makedirs(STATIC_DIR, exist_ok=True)
    server_address = ("127.0.0.1", PORT)
    httpd = ThreadingHTTPServer(server_address, AudioServerHandler)
    print("\n" + "=" * 60)
    print(f"[Server] Audio Classification Web App Running:")
    print(f"         URL: http://127.0.0.1:{PORT}")
    print("=" * 60 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down cleanly.")
        httpd.server_close()


if __name__ == "__main__":
    run_server()


# ==============================================================================
# Modern Vercel WSGI Entrypoint
# ==============================================================================
def application(environ, start_response):
    path = environ.get("PATH_INFO", "")
    method = environ.get("REQUEST_METHOD", "GET")

    def json_response(data, status="200 OK"):
        body = json.dumps(data).encode("utf-8")
        headers = [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
            ("Access-Control-Allow-Origin", "*"),
            ("Access-Control-Allow-Methods", "POST, GET, OPTIONS"),
            ("Access-Control-Allow-Headers", "Content-Type, X-Model-Type"),
        ]
        start_response(status, headers)
        return [body]

    if method == "OPTIONS":
        headers = [
            ("Access-Control-Allow-Origin", "*"),
            ("Access-Control-Allow-Methods", "POST, GET, OPTIONS"),
            ("Access-Control-Allow-Headers", "Content-Type, X-Model-Type"),
        ]
        start_response("200 OK", headers)
        return [b""]

    if method == "GET":
        if path == "/api/samples":
            return json_response(PRESETS)
        elif path == "/api/metrics":
            metrics_data = {}
            for name, fname in [
                ("baseline", "baseline_metrics.json"),
                ("multires", "multires_metrics.json"),
                ("multifeature", "multifeature_metrics.json"),
            ]:
                fpath = os.path.join(RESULTS_DIR, fname)
                if os.path.exists(fpath):
                    with open(fpath) as f:
                        metrics_data[name] = json.load(f)
            return json_response(metrics_data)
        elif path.startswith("/api/audio/"):
            filename = os.path.basename(path)
            audio_path = os.path.join(ESC50_ROOT, "audio", filename)
            if not os.path.exists(audio_path):
                audio_path = os.path.join(STATIC_DIR, "samples", filename)
            if os.path.exists(audio_path):
                with open(audio_path, "rb") as f:
                    data = f.read()
                headers = [
                    ("Content-Type", "audio/wav"),
                    ("Content-Length", str(len(data))),
                    ("Access-Control-Allow-Origin", "*"),
                ]
                start_response("200 OK", headers)
                return [data]
            start_response("404 Not Found", [("Content-Type", "text/plain")])
            return [b"Audio file not found"]
        elif path == "/favicon.ico":
            start_response("204 No Content", [])
            return [b""]
        else:
            rel_path = path.lstrip("/")
            if not rel_path or rel_path == "index.html":
                file_path = os.path.join(STATIC_DIR, "index.html")
            else:
                file_path = os.path.join(STATIC_DIR, rel_path)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                content_type = "text/html"
                if file_path.endswith(".css"):
                    content_type = "text/css"
                elif file_path.endswith(".js"):
                    content_type = "application/javascript"
                elif file_path.endswith(".json"):
                    content_type = "application/json"
                with open(file_path, "rb") as f:
                    data = f.read()
                headers = [("Content-Type", content_type), ("Content-Length", str(len(data)))]
                start_response("200 OK", headers)
                return [data]
            start_response("404 Not Found", [("Content-Type", "text/plain")])
            return [b"Not Found"]

    elif method == "POST" and path == "/api/predict":
        try:
            content_length = int(environ.get("CONTENT_LENGTH", 0))
            raw_body = environ["wsgi.input"].read(content_length) if content_length > 0 else b""
            content_type = environ.get("CONTENT_TYPE", "")
            model_type_header = environ.get("HTTP_X_MODEL_TYPE", "multifeature")
            res = handle_predict(raw_body, content_type, model_type_header)
            status = "400 Bad Request" if "error" in res else "200 OK"
            return json_response(res, status=status)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return json_response({"error": str(e)}, status="500 Internal Server Error")

    start_response("404 Not Found", [("Content-Type", "text/plain")])
    return [b"Not Found"]


# Vercel top-level WSGI / ASGI entrypoints
app = application
handler = application

