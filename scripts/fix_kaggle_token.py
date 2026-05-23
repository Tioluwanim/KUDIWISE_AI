"""
scripts/fix_kaggle_token.py
Fixes the UTF-16 encoding issue with kaggle.json on Windows.
Run this once, then run download_datasets.py again.

The problem: Windows Notepad sometimes saves JSON files as UTF-16
which produces the \xff\xfe BOM (byte order mark) that breaks kagglehub.
"""
import json
import shutil
from pathlib import Path

kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
backup_path = Path.home() / ".kaggle" / "kaggle.json.bak"

print(f"Checking: {kaggle_path}")

if not kaggle_path.exists():
    print("ERROR: kaggle.json not found at", kaggle_path)
    print("Download it from https://kaggle.com/settings → API → Create New Token")
    exit(1)

# Read the raw bytes to detect encoding
raw = kaggle_path.read_bytes()
print(f"File size: {len(raw)} bytes")
print(f"First 4 bytes (hex): {raw[:4].hex()}")

# Detect and fix encoding
if raw[:2] == b'\xff\xfe':
    print("DETECTED: UTF-16 LE encoding (Windows BOM) — this is the bug")
    content = raw.decode('utf-16')
elif raw[:2] == b'\xfe\xff':
    print("DETECTED: UTF-16 BE encoding — this is the bug")
    content = raw.decode('utf-16')
elif raw[:3] == b'\xef\xbb\xbf':
    print("DETECTED: UTF-8 BOM — removing it")
    content = raw[3:].decode('utf-8')
else:
    print("Encoding looks fine (UTF-8). Testing if JSON is valid...")
    content = raw.decode('utf-8')

# Validate it's proper JSON with the right keys
try:
    data = json.loads(content)
    if 'username' not in data or 'key' not in data:
        print("ERROR: kaggle.json is missing 'username' or 'key' fields")
        print("Contents:", list(data.keys()))
        exit(1)
    print(f"✓ Valid kaggle.json for user: {data['username']}")
except json.JSONDecodeError as e:
    print(f"ERROR: File is not valid JSON: {e}")
    exit(1)

# Back up original
shutil.copy(kaggle_path, backup_path)
print(f"✓ Backed up original to: {backup_path}")

# Write back as clean UTF-8
kaggle_path.write_text(json.dumps(data), encoding='utf-8')
print(f"✓ Rewrote kaggle.json as clean UTF-8")

# Verify
verify = json.loads(kaggle_path.read_bytes().decode('utf-8'))
print(f"✓ Verified: username={verify['username']}, key starts with {verify['key'][:8]}...")
print()
print("=== FIXED ===")
print("Now run: python scripts/download_datasets.py")