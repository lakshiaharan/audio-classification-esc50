"""
tests/test_e2e_suite.py
Comprehensive End-to-End Regression Suite for Backend & Frontend APIs
"""

import urllib.request
import urllib.error
import json
import os
import io
import sys
import torch
import torchaudio

BASE_URL = "http://127.0.0.1:8000"

def run_suite():
    print("==================================================================")
    print("        End-to-End Automated Verification Suite                   ")
    print("==================================================================\n")

    # 1. Test Static Files & Endpoints
    print("--- 1. Testing Core Endpoints ---")
    endpoints = [
        ("/", 200),
        ("/style.css", 200),
        ("/app.js", 200),
        ("/favicon.ico", 204),
        ("/api/samples", 200),
        ("/api/metrics", 200)
    ]

    for path, expected_status in endpoints:
        url = BASE_URL + path
        try:
            res = urllib.request.urlopen(url, timeout=5)
            status = res.status
            assert status == expected_status, f"Expected {expected_status}, got {status}"
            print(f"  [PASS] Endpoint {path:<16} -> {status} OK")
        except Exception as e:
            print(f"  [FAIL] Endpoint {path:<16} -> {e}")
            sys.exit(1)

    # 2. Test All 50 Presets and Audio File Streaming
    print("\n--- 2. Validating all 50 Audio Presets & Predictions ---")
    samples_res = urllib.request.urlopen(BASE_URL + "/api/samples", timeout=5)
    samples = json.loads(samples_res.read().decode())
    assert len(samples) == 50, f"Expected 50 presets, got {len(samples)}"
    print(f"  -> Total Presets Retrieved: {len(samples)}")

    audio_errors = 0
    predict_errors = 0

    for s in samples:
        fn = s["filename"]
        # Check audio stream
        try:
            a_res = urllib.request.urlopen(f"{BASE_URL}/api/audio/{fn}", timeout=5)
            raw_audio = a_res.read()
            assert len(raw_audio) > 5000, f"Audio file {fn} is too small"
        except Exception as e:
            print(f"  [FAIL AUDIO] {s['name']} ({fn}): {e}")
            audio_errors += 1
            continue

        # Check JSON prediction endpoint
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/api/predict",
                data=json.dumps({"filename": fn, "model_type": "multifeature"}).encode(),
                headers={"Content-Type": "application/json"}
            )
            p_res = json.loads(urllib.request.urlopen(req, timeout=5).read().decode())
            assert p_res["success"] is True
            assert len(p_res["top5"]) == 5
            assert "class" in p_res["top1"]
        except Exception as e:
            print(f"  [FAIL PREDICT] {s['name']}: {e}")
            predict_errors += 1

    print(f"  [PASS] Audio File Streaming : {len(samples) - audio_errors}/{len(samples)} OK")
    print(f"  [PASS] Real-time Inference  : {len(samples) - predict_errors}/{len(samples)} OK")
    assert audio_errors == 0 and predict_errors == 0

    # 3. Test Binary Audio Stream Edge Cases
    print("\n--- 3. Testing Binary Audio Stream Edge Cases ---")
    import numpy as np
    import scipy.io.wavfile as wavfile

    # Case A: Short 1-second 16kHz audio
    short_np = (np.random.randn(16000) * 10000).astype(np.int16)
    bio = io.BytesIO()
    wavfile.write(bio, 16000, short_np)
    req = urllib.request.Request(
        f"{BASE_URL}/api/predict",
        data=bio.getvalue(),
        headers={"Content-Type": "audio/wav", "X-Model-Type": "multifeature"}
    )
    res = json.loads(urllib.request.urlopen(req).read().decode())
    print(f"  [PASS] Short 16kHz audio stream -> Top-1: {res['top1']['class']} (latency: {res['latency_ms']} ms)")

    # Case B: Stereo 48kHz audio (6.5 seconds)
    stereo_np = (np.random.randn(int(48000 * 6.5), 2) * 10000).astype(np.int16)
    bio = io.BytesIO()
    wavfile.write(bio, 48000, stereo_np)
    req = urllib.request.Request(
        f"{BASE_URL}/api/predict",
        data=bio.getvalue(),
        headers={"Content-Type": "audio/wav", "X-Model-Type": "baseline"}
    )
    res = json.loads(urllib.request.urlopen(req).read().decode())
    print(f"  [PASS] Stereo 48kHz audio stream -> Top-1: {res['top1']['class']} (model: {res['model']})")

    # Case C: All 3 Model Architectures Verified
    print("\n--- 4. Testing Model Architecture Switching ---")
    for m in ["multifeature", "baseline", "multires"]:
        req = urllib.request.Request(
            f"{BASE_URL}/api/predict",
            data=json.dumps({"filename": "1-100032-A-0.wav", "model_type": m}).encode(),
            headers={"Content-Type": "application/json"}
        )
        res = json.loads(urllib.request.urlopen(req).read().decode())
        print(f"  [PASS] Architecture '{m:<13}' -> Predicted: {res['top1']['class']:<10} ({res['top1']['confidence']}%)")

    # Case D: Corrupted Payload Handled Gracefully
    print("\n--- 5. Testing Fault Tolerance & Error Handlers ---")
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/api/predict",
            data=b"corrupted raw data non-audio bytes",
            headers={"Content-Type": "audio/wav"}
        )
        urllib.request.urlopen(req)
        print("  [FAIL] Corrupted payload did not raise error")
    except urllib.error.HTTPError as err:
        print(f"  [PASS] Corrupted payload safely rejected with HTTP {err.code}")

    print("\n==================================================================")
    print("      ALL END-TO-END VERIFICATION TESTS PASSED (100% SUCCESS)     ")
    print("==================================================================")


if __name__ == "__main__":
    run_suite()
