import os
import sys
import time
import subprocess
import urllib.request
import json
from pathlib import Path

DATA_DIR = Path(r"d:\Education_Material\Addition\SMLab\DO_AN\fallvision_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_RAR_DIR = DATA_DIR / "rar_files"
RAW_RAR_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR = DATA_DIR / "csv_files"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

UNRAR_PATH = Path(r"C:\Program Files\WinRAR\UnRAR.exe")

url = 'https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/75QPKK'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=15) as resp:
    data = json.loads(resp.read().decode())

files = data['data']['latestVersion']['files']
csv_rars = [f['dataFile'] for f in files if 'csv' in f['dataFile']['filename'].lower()]

print(f"Bắt đầu tải {len(csv_rars)} file keypoints CSV (tổng ~336 MB)...")

for i, f in enumerate(csv_rars, 1):
    fid = f['id']
    fname = f['filename']
    fsize_mb = f.get('filesize', 0) / (1024 * 1024)
    out_rar = RAW_RAR_DIR / fname

    if not out_rar.exists() or out_rar.stat().st_size < f.get('filesize', 0) * 0.95:
        print(f"[{i}/{len(csv_rars)}] Đang tải {fname} ({fsize_mb:.1f} MB)...", flush=True)
        down_url = f"https://dataverse.harvard.edu/api/access/datafile/{fid}"
        req_dl = urllib.request.Request(down_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_dl, timeout=60) as response, open(out_rar, 'wb') as out_file:
            chunk_size = 1024 * 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
    else:
        print(f"[{i}/{len(csv_rars)}] Đã có sẵn {fname}", flush=True)

    # Giải nén bằng UnRAR.exe
    cmd = [str(UNRAR_PATH), "x", "-y", "-inul", str(out_rar), str(EXTRACT_DIR)]
    subprocess.run(cmd, check=True)

print("\nHoàn tất tải và giải nén toàn bộ keypoints CSV của FallVision!")
csv_files = list(EXTRACT_DIR.rglob("*.csv"))
print(f"Tổng số file CSV đã giải nén: {len(csv_files)}")
