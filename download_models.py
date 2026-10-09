#!/usr/bin/env python3
"""
download_models.py - InsightFace / ArcFace Offline Model Downloader.
Downloads and caches official InsightFace ONNX models (buffalo_l or buffalo_sc)
into local storage using standard Python libraries (no pip dependencies required).
Prepares models for 100% offline air-gapped theme park deployments.
"""

import os
import sys
import time
import zipfile
import argparse
import urllib.request
import urllib.error

OFFICIAL_URLS = {
    "buffalo_l": [
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    ],
    "buffalo_sc": [
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
    ]
}


def download_with_progress(url: str, dest_path: str):
    """Download a file over HTTP/HTTPS with real-time progress display."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) InsightFace-Downloader/1.0"}
    req = urllib.request.Request(url, headers=headers)

    print(f"Mengunduh dari: {url}")
    with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.info().get("Content-Length", -1))
        downloaded = 0
        chunk_size = 1024 * 512 # 512 KB
        start_time = time.time()

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)

            if total_size > 0:
                pct = downloaded * 100.0 / total_size
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                elapsed = max(0.1, time.time() - start_time)
                speed = mb_down / elapsed
                sys.stdout.write(f"\r  Progress: [{pct:5.1f}%] {mb_down:.1f} MB / {mb_total:.1f} MB ({speed:.1f} MB/s)")
            else:
                mb_down = downloaded / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {mb_down:.1f} MB diunduh...")
            sys.stdout.flush()

    print("\n  Unduhan selesai!")


def ensure_model_downloaded(model_name: str, root_dir: str) -> bool:
    """
    Ensure the requested model pack is downloaded and unzipped into root_dir/models/{model_name}.
    Uses pure standard library. No pip / insightface / onnxruntime installation required on host!
    """
    models_dir = os.path.join(root_dir, "models")
    target_pack_dir = os.path.join(models_dir, model_name)
    os.makedirs(target_pack_dir, exist_ok=True)

    print(f"\n[+] Memeriksa model '{model_name}' di: {target_pack_dir}")

    # Check if already downloaded
    existing_onnx = [f for f in os.listdir(target_pack_dir) if f.endswith(".onnx")]
    if len(existing_onnx) >= 2:
        print(f"  ✔ Model '{model_name}' sudah terpasang lengkap ({len(existing_onnx)} file ONNX ditemukan):")
        for f in sorted(existing_onnx):
            sz_mb = os.path.getsize(os.path.join(target_pack_dir, f)) / (1024 * 1024)
            print(f"     - {f} ({sz_mb:.1f} MB)")
        return True

    urls = OFFICIAL_URLS.get(model_name, [])
    if not urls:
        print(f"  [ERROR] Model '{model_name}' tidak dikenal. Pilihan: buffalo_l, buffalo_sc")
        return False

    temp_zip = os.path.join(models_dir, f"{model_name}.zip")
    download_ok = False

    for url in urls:
        try:
            download_with_progress(url, temp_zip)
            download_ok = True
            break
        except Exception as e:
            print(f"  [GAGAL] Unduhan dari {url} error: {e}")
            if os.path.exists(temp_zip):
                try:
                    os.remove(temp_zip)
                except Exception:
                    pass

    if not download_ok or not os.path.exists(temp_zip):
        print(f"  [ERROR] Gagal mengunduh archive zip untuk model '{model_name}'.")
        return False

    print(f"  Mengekstrak {model_name}.zip ke {target_pack_dir}...")
    try:
        with zipfile.ZipFile(temp_zip, "r") as z:
            # Check if files inside zip are already prefixed with model_name/
            namelist = z.namelist()
            has_parent = any(name.startswith(f"{model_name}/") for name in namelist)
            
            if has_parent:
                # Extract into models_dir directly
                z.extractall(models_dir)
            else:
                # Extract directly into target_pack_dir
                z.extractall(target_pack_dir)

        # Cleanup zip file
        try:
            os.remove(temp_zip)
        except Exception:
            pass

        # Verify extracted .onnx files
        extracted_onnx = [f for f in os.listdir(target_pack_dir) if f.endswith(".onnx")]
        if not extracted_onnx:
            # Look in subdirectories if zip had an unexpected top-level folder
            for root, _, files in os.walk(target_pack_dir):
                for f in files:
                    if f.endswith(".onnx"):
                        src = os.path.join(root, f)
                        dst = os.path.join(target_pack_dir, f)
                        if src != dst:
                            os.replace(src, dst)
            extracted_onnx = [f for f in os.listdir(target_pack_dir) if f.endswith(".onnx")]

        print(f"  ✔ Ekstraksi sukses! Ditemukan {len(extracted_onnx)} file ONNX model weights:")
        for f in sorted(extracted_onnx):
            sz_mb = os.path.getsize(os.path.join(target_pack_dir, f)) / (1024 * 1024)
            print(f"     ✔ {f} ({sz_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"  [ERROR] Gagal mengekstrak archive zip: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Unduh dan simpan model InsightFace / ArcFace untuk sistem foto offline."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="buffalo_l",
        choices=["buffalo_l", "buffalo_sc", "all"],
        help="Model: 'buffalo_l' (ResNet-50 ArcFace 512-D, akurasi tertinggi), 'buffalo_sc' (MobileFaceNet, cepat), atau 'all'."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="",
        help="Direktori penyimpanan (default: DB_DIR/../models/insightface atau ./data/models/insightface)."
    )

    args = parser.parse_args()

    if args.output_dir:
        root_dir = os.path.abspath(args.output_dir)
    else:
        db_dir = os.getenv("DB_DIR", "./data/db")
        parent_dir = os.path.dirname(os.path.abspath(db_dir))
        root_dir = os.getenv("INSIGHTFACE_ROOT", os.path.join(parent_dir, "models", "insightface"))

    models = ["buffalo_l", "buffalo_sc"] if args.model == "all" else [args.model]

    print("=" * 68)
    print("  The Lodge Maribaya - InsightFace ArcFace 512-D Model Downloader")
    print("=" * 68)
    print(f"Lokasi Target : {root_dir}")
    print(f"Daftar Model  : {', '.join(models)}")

    all_ok = True
    for m in models:
        ok = ensure_model_downloaded(m, root_dir)
        if not ok:
            all_ok = False

    print("\n" + "=" * 68)
    if all_ok:
        print("  STATUS: BERHASIL!")
        print("  Semua bobot model ArcFace 512-D telah tersimpan di disk lokal.")
        print("  Sistem siap dijalankan 100% offline (air-gapped) tanpa internet.")
    else:
        print("  STATUS: ADA MODEL YANG GAGAL DIUNDUH.")
        print("  Periksa koneksi internet Anda atau coba jalankan kembali perintah ini.")
    print("=" * 68)


if __name__ == "__main__":
    main()
