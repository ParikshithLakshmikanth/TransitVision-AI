"""TransitVision AI - Final Integrity Verification Script."""
import hashlib
import json
from pathlib import Path

files_to_check = {
    "kandy_eta_training.parquet": ("data/processed/kandy_eta_training.parquet", "1d5af060533bd0b6821d69138b4eb99a428028c051d04b15308d0cf272e4983c"),
    "kandy_eta_validation.parquet": ("data/processed/kandy_eta_validation.parquet", "e01635b507790e9717d1f88db538a454dd4c378ae45262b287ee820106e1f1e4"),
    "kandy_eta_stream.parquet": ("data/processed/kandy_eta_stream.parquet", "332479b56571e88ecfa596abd2cd412f7f51e4c1ce96f1ac016517a58606ab85"),
    "v1.0.0 model.joblib": ("models/eta_model_v1.0.0/model.joblib", "460cf798484cce96873812ac771b7bdcfbf08d5d6297144f81bcdc911a8f9a56"),
    "v1.0.0 preprocessor.joblib": ("models/eta_model_v1.0.0/preprocessor.joblib", "42aa29b0f8a61db6b635dd744c47e3cda83a9853b2a4223bc96323eb7a6d941c"),
    "v1.0.0 feature_config.json": ("models/eta_model_v1.0.0/feature_config.json", "fded7b1e8290b7867ccc6dd7b91f637eff6f5ff3faf37c646cb8bb6c0b757e84"),
    "v1.0.0 metadata.json": ("models/eta_model_v1.0.0/metadata.json", "8aba89f780061421c23873d07683559080fcce6b3c5fca5078bd7ace00214ec5"),
    "drift_reference_profile.json": ("data/metadata/drift_reference_profile.json", "00711473c3a51d31bcea4f75c5bafb82623e1a7c653b60f0be5f6721c507311b"),
}

print("=== FINAL INTEGRITY AUDIT ===")
all_pass = True
for name, (path, expected) in files_to_check.items():
    p = Path(path)
    if not p.exists():
        print(f"[FAIL - NOT FOUND] {name}: {path}")
        all_pass = False
        continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    match = (h == expected)
    if not match:
        all_pass = False
    print(f"[{'PASS' if match else 'FAIL'}] {name:30s} -> {h}")

print(f"\nALL 8 CRITICAL ARTIFACTS VERIFIED: {all_pass}")

# Inspect model registry
reg_path = Path("data/metadata/model_registry.json")
with open(reg_path, "r", encoding="utf-8") as f:
    reg = json.load(f)

print(f"Production Model ID: {reg.get('production_model_id')}")
print(f"Models in registry ({len(reg.get('models', []))}):")
for m in reg.get("models", []):
    print(f"  - {m['model_id']}: {m['status']} ({m['algorithm']})")
