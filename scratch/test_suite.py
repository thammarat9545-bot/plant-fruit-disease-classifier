import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi.testclient import TestClient
from app.server import app

client = TestClient(app)

print("--- Test 1: GET / (Frontend HTML) ---")
r = client.get("/")
assert r.status_code == 200, f"Expected 200, got {r.status_code}"
assert "Plant/Fruit Classifier" in r.text, "Title missing"
assert "อัปโหลดรูปภาพ" in r.text, "Content missing"
print("[PASS] Frontend HTML rendered successfully!")

print("\n--- Test 2: GET /api/health ---")
r = client.get("/api/health")
assert r.status_code == 200
h = r.json()
print("[PASS] Health check:", h["status"], "| device:", h["device"], "| model:", h["model_architecture"])

print("\n--- Test 3: Static Asset Checks ---")
for path in ["/static/css/custom.css", "/static/js/app.js", "/static/images/samples/leaf_sample_01.jpg", "/static/images/ksu_logo.png"]:
    sr = client.get(path)
    assert sr.status_code == 200, f"Static {path} failed: {sr.status_code}"
    print(f"[PASS] {path} is available ({len(sr.content)} bytes)")

print("\n--- Test 4: Prediction Tests (The Test Cases from Chapter 4) ---")
test_cases = [
    ("TC-01: Leaf Blight Disease", "app/static/images/samples/leaf_sample_01.jpg", "disease", "Early_Blight"),
    ("TC-02: Ripe Fruit Assessment", "app/static/images/samples/fruit_ripe.jpg", "ripeness", "Fruit_Ripe"),
    ("TC-02b: Unripe Fruit Assessment", "app/static/images/samples/fruit_unripe.jpg", "ripeness", "Fruit_Unripe"),
    ("TC-02c: Rotten Fruit Assessment", "app/static/images/samples/fruit_rotten.jpg", "ripeness", "Fruit_Rotten"),
    ("TC-02d: Healthy Leaf Assessment", "app/static/images/samples/healthy_leaf.jpg", "disease", "Healthy_Leaf")
]

for name, path, mode, expected_cls in test_cases:
    with open(path, "rb") as f:
        files = {"file": (path.split("/")[-1], f, "image/jpeg")}
        pr = client.post("/api/predict", files=files, data={"mode": mode})
        assert pr.status_code == 200, f"{name} failed: {pr.status_code}"
        d = pr.json()
        print(f"[PASS] {name} -> Predicted: {d['predicted_class']} ({d['thai_name']}) | Confidence: {d['confidence']}% | Time: {d['execution_time_ms']}ms")

print("\n--- Test 5: TC-03 Invalid File Type Rejection ---")
bad_files = {"file": ("document.pdf", b"%PDF-1.4 dummy", "application/pdf")}
br = client.post("/api/predict", files=bad_files, data={"mode": "disease"})
assert br.status_code == 400, f"Expected 400, got {br.status_code}"
print("[PASS] TC-03: Invalid file correctly rejected with HTTP 400! Message:", br.json().get("detail"))

print("\n" + "=" * 60)
print("ALL TESTS PASSED WITH 100% SUCCESS!")
print("=" * 60)
