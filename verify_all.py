"""
verify_all.py - Complete End-to-End Verification Suite for SoundPulse Vercel Deployment
Tests every single preset (50/50), every model architecture (3/3), raw WAV inference,
endpoint routing, CORS, and static asset integrity.
"""

import os
import io
import json
import wave
import struct
import sys
from api.index import handler, PRESETS, MODEL_METRICS, CLASSES

class MockHandler(handler):
    def __init__(self, path, method="GET", body=b"", headers=None):
        self.path = path
        self.command = method
        self.requestline = f"{method} {path} HTTP/1.1"
        self.request_version = "HTTP/1.1"
        self.headers = headers or {}
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.response_status = None
        self.response_headers = {}

    def send_response(self, code, message=None):
        self.response_status = code

    def send_header(self, keyword, value):
        self.response_headers[keyword] = value

    def end_headers(self):
        pass

def run_tests():
    print("=" * 70)
    print("STARTING COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 70)

    # 1. Static Asset Checks
    print("\n[Test 1] Checking Static Assets in public/ and static/...")
    required_files = [
        "public/index.html",
        "public/style.css",
        "public/app.js",
        "public/data/samples.json",
        "public/data/metrics.json",
        "static/index.html",
        "static/style.css",
        "static/app.js",
        "static/data/samples.json",
        "static/data/metrics.json"
    ]
    for rf in required_files:
        assert os.path.exists(rf), f"Missing required file: {rf}"
        assert os.path.getsize(rf) > 0, f"Empty file: {rf}"
    print("  PASS: All 10 core static and data files exist and are populated.")

    # 2. GET /api/samples
    print("\n[Test 2] Testing GET /api/samples...")
    h = MockHandler("/api/samples")
    h.do_GET()
    assert h.response_status == 200, f"Expected 200, got {h.response_status}"
    samples = json.loads(h.wfile.getvalue().decode())
    assert len(samples) == 50, f"Expected 50 presets, got {len(samples)}"
    print(f"  PASS: /api/samples returned 50 presets successfully.")

    # 3. GET /api/metrics
    print("\n[Test 3] Testing GET /api/metrics...")
    h = MockHandler("/api/metrics")
    h.do_GET()
    assert h.response_status == 200, f"Expected 200, got {h.response_status}"
    metrics = json.loads(h.wfile.getvalue().decode())
    assert "multifeature" in metrics and "baseline" in metrics and "multires" in metrics
    print(f"  PASS: /api/metrics returned metrics for all 3 architectures.")

    # 4. Audio Sample Files & GET /api/audio/<filename>
    print("\n[Test 4] Testing GET /api/audio/<filename> for all 50 presets...")
    for idx, preset in enumerate(PRESETS):
        fn = preset["filename"]
        h = MockHandler(f"/api/audio/{fn}")
        h.do_GET()
        assert h.response_status == 200, f"Failed audio fetch for {fn}"
        audio_data = h.wfile.getvalue()
        assert len(audio_data) > 1000, f"Audio file {fn} is suspiciously small: {len(audio_data)} bytes"
        assert audio_data[:4] == b"RIFF", f"Audio file {fn} has invalid WAV RIFF header"
    print("  PASS: All 50 preset WAV audio files successfully fetched and verified with valid RIFF headers.")

    # 5. POST /api/predict for all 50 presets & architectures
    print("\n[Test 5] Testing POST /api/predict for all 50 categories across 3 models...")
    models = ["multifeature", "baseline", "multires"]
    total_predictions = 0

    for model_type in models:
        for preset in PRESETS:
            cat = preset["category"]
            fn = preset["filename"]
            req_body = json.dumps({"filename": fn, "model_type": model_type}).encode("utf-8")
            h = MockHandler(
                "/api/predict",
                method="POST",
                body=req_body,
                headers={"Content-Length": str(len(req_body)), "Content-Type": "application/json"}
            )
            h.do_POST()
            assert h.response_status == 200, f"Failed POST /api/predict for {fn} ({model_type})"
            res = json.loads(h.wfile.getvalue().decode())
            
            assert "top1" in res, f"Missing top1 in response: {res}"
            assert "top5" in res, f"Missing top5 in response: {res}"
            assert res["top1"]["class"] == cat, f"Mismatch: expected {cat}, got {res['top1']['class']}"
            assert res["top1"]["confidence"] > 50.0, f"Confidence too low: {res['top1']['confidence']}"
            assert len(res["top5"]) == 5, f"Expected 5 ranking items, got {len(res['top5'])}"
            assert res["latency_ms"] > 0, f"Invalid latency: {res['latency_ms']}"
            total_predictions += 1

    print(f"  PASS: Tested {total_predictions} predictions (50 classes x 3 models). 100% matched target categories with valid confidence and latency.")

    # 6. POST /api/predict for Raw WAV Audio Stream (Mic / Upload simulation)
    print("\n[Test 6] Testing Raw WAV Audio payload inference (Microphone/Upload)...")
    # Fetch a sample WAV
    h_audio = MockHandler("/api/audio/1-100032-A-0.wav")
    h_audio.do_GET()
    wav_bytes = h_audio.wfile.getvalue()

    h_raw = MockHandler(
        "/api/predict",
        method="POST",
        body=wav_bytes,
        headers={"Content-Length": str(len(wav_bytes)), "Content-Type": "audio/wav", "X-Model-Type": "multifeature"}
    )
    h_raw.do_POST()
    assert h_raw.response_status == 200, f"Raw WAV inference failed: {h_raw.response_status}"
    res_raw = json.loads(h_raw.wfile.getvalue().decode())
    assert "top1" in res_raw and "top5" in res_raw
    assert len(res_raw["top5"]) == 5
    print(f"  PASS: Raw WAV audio classified successfully as '{res_raw['top1']['class']}' ({res_raw['top1']['confidence']}% conf, latency: {res_raw['latency_ms']} ms).")

    # 7. OPTIONS / CORS Preflight Test
    print("\n[Test 7] Testing CORS Preflight (OPTIONS)...")
    h_opt = MockHandler("/api/predict", method="OPTIONS")
    h_opt.do_OPTIONS()
    assert h_opt.response_status == 200
    assert h_opt.response_headers.get("Access-Control-Allow-Origin") == "*"
    print("  PASS: CORS preflight headers verified.")

    print("\n" + "=" * 70)
    print("ALL 7 VERIFICATION TEST SUITES PASSED WITH 100% SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
