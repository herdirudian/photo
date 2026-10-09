#!/usr/bin/env python3
"""
download_models.py - InsightFace / ArcFace Offline Model Downloader.
Downloads and caches official InsightFace ONNX models (buffalo_l or buffalo_sc)
into local storage so the Theme Park Photo Retrieval System operates 100% offline
in air-gapped environments without external internet connectivity.
"""

import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DownloadModels")


def ensure_model_downloaded(model_name: str, root_dir: str):
    """
    Download and initialize the InsightFace model pack using official InsightFace APIs.
    """
    os.makedirs(root_dir, exist_ok=True)
    target_pack_dir = os.path.join(root_dir, "models", model_name)

    logger.info(f"Checking InsightFace model '{model_name}' in: {target_pack_dir}")

    # Check if already downloaded
    if os.path.exists(target_pack_dir):
        onnx_files = [f for f in os.listdir(target_pack_dir) if f.endswith(".onnx")]
        if len(onnx_files) >= 2:
            logger.info(f"Model '{model_name}' is already downloaded ({len(onnx_files)} ONNX files found).")
            for f in onnx_files:
                logger.info(f"  - {f}")
            return True

    logger.info(f"Downloading model '{model_name}' to offline storage {root_dir}...")
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name=model_name, root=root_dir, providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info(f"Model '{model_name}' successfully prepared and verified!")
        
        onnx_files = [f for f in os.listdir(target_pack_dir) if f.endswith(".onnx")]
        logger.info(f"Verified {len(onnx_files)} ONNX model weights in {target_pack_dir}:")
        for f in onnx_files:
            logger.info(f"  ✔ {f}")
        return True
    except ImportError:
        logger.error("insightface or onnxruntime is not installed. Please run: pip install insightface onnxruntime")
        return False
    except Exception as e:
        logger.error(f"Failed to download/prepare model '{model_name}': {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download and cache InsightFace models for offline theme park kiosk.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="buffalo_l",
        choices=["buffalo_l", "buffalo_sc", "all"],
        help="Model to download: 'buffalo_l' (ResNet-50 ArcFace, best accuracy) or 'buffalo_sc' (MobileFaceNet, ultra-fast) or 'all'."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="",
        help="Target root folder (defaults to DB_DIR/../models/insightface or ./data/models/insightface)."
    )

    args = parser.parse_args()

    if args.output_dir:
        root_dir = os.path.abspath(args.output_dir)
    else:
        db_dir = os.getenv("DB_DIR", "./data/db")
        parent_dir = os.path.dirname(os.path.abspath(db_dir))
        root_dir = os.getenv("INSIGHTFACE_ROOT", os.path.join(parent_dir, "models", "insightface"))

    models_to_download = ["buffalo_l", "buffalo_sc"] if args.model == "all" else [args.model]

    print("=" * 65)
    print("  The Lodge Maribaya - InsightFace ArcFace 512-D Model Downloader")
    print("=" * 65)
    print(f"Target Root: {root_dir}\n")

    success_all = True
    for m in models_to_download:
        ok = ensure_model_downloaded(m, root_dir)
        if not ok:
            success_all = False

    print("\n" + "=" * 65)
    if success_all:
        print("  Semua model InsightFace / ArcFace berhasil diunduh!")
        print("  Sistem siap dijalankan 100% offline tanpa koneksi internet.")
    else:
        print("  Ada model yang gagal diunduh. Periksa koneksi internet atau hak akses folder.")
    print("=" * 65)


if __name__ == "__main__":
    main()
