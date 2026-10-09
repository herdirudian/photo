"""
config_manager.py - Persistent Configuration and Network Share Management.
Stores configuration in data/db/settings.json and handles SMB/CIFS connection testing,
subfolder inspection, and storage synchronization.
"""

import os
import json
import time
import socket
import shutil
import logging
from typing import Dict, Any, List

logger = logging.getLogger("PhotoRetrieval.Config")

CONFIG_FILE_PATH = os.path.join(os.getenv("DB_DIR", "/app/data/db"), "settings.json")

DEFAULT_SETTINGS: Dict[str, Any] = {
    "storage_type": "smb",
    "smb_host": "192.168.100.93",
    "smb_share": "photo",
    "smb_subfolder": "",
    "smb_username": "USER_SHARE",
    "smb_password": "",
    "smb_domain": "",
    "local_path": "/app/data/raw",
    "polling_interval": 15,
    "ai_engine": "insightface",  # "insightface" (ArcFace 512-D) or "dlib" (ResNet-34 128-D)
    "insightface_model": "buffalo_l",  # "buffalo_l" (ResNet-50 highest accuracy) or "buffalo_sc" (fast CPU)
    "insightface_det_thresh": 0.50,
    "default_threshold": 0.50,  # 0.50 for InsightFace 512-D; 0.82 for dlib 128-D
    "default_threshold_insightface": 0.50,
    "default_threshold_dlib": 0.82,
    "admin_pin": "BI5mill4h@@@",
    "operator_pin": "BI5mill4h@@@",
    "require_pin_to_access": True,
    "samba_network_prefix": r"\\192.168.100.90\park-photos\results"
}


def _write_settings_file(settings: Dict[str, Any]):
    """Write settings dictionary directly to JSON file."""
    os.makedirs(os.path.dirname(CONFIG_FILE_PATH), exist_ok=True)
    temp_path = CONFIG_FILE_PATH + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    os.replace(temp_path, CONFIG_FILE_PATH)


def load_settings() -> Dict[str, Any]:
    """Load settings from JSON file or create with defaults."""
    settings = dict(DEFAULT_SETTINGS)
    if os.path.exists(CONFIG_FILE_PATH):
        try:
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                settings.update(saved)
                
                # Intelligent threshold migration: if upgraded to insightface but old dlib threshold (0.82 or 0.86) is saved
                if settings.get("ai_engine") == "insightface" and settings.get("default_threshold", 0.50) >= 0.80:
                    logger.info("Auto-adjusting similarity threshold from dlib (>=0.80) to InsightFace ArcFace 512-D standard (0.50)")
                    settings["default_threshold"] = float(settings.get("default_threshold_insightface", 0.50))
        except Exception as e:
            logger.error(f"Error loading {CONFIG_FILE_PATH}: {e}")
    else:
        try:
            _write_settings_file(settings)
        except Exception as e:
            logger.error(f"Error saving initial {CONFIG_FILE_PATH}: {e}")
    return settings


def save_settings(new_settings: Dict[str, Any]) -> Dict[str, Any]:
    """Atomically save settings to persistent storage without recursion."""
    current = dict(DEFAULT_SETTINGS)
    if os.path.exists(CONFIG_FILE_PATH):
        try:
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                current.update(json.load(f))
        except Exception:
            pass
    current.update(new_settings)
    _write_settings_file(current)
    logger.info(f"Settings successfully saved to {CONFIG_FILE_PATH}")
    return current


def test_network_connection(host: str, port: int = 445, timeout: float = 4.0) -> Dict[str, Any]:
    """
    Test direct TCP connectivity to SMB server host on port 445 (or NetBIOS 139).
    Returns connection status and latency.
    """
    if not host or not host.strip():
        return {
            "success": False,
            "error": "Alamat IP atau hostname server tidak boleh kosong."
        }

    host = host.strip().lstrip(r"\/")
    start_time = time.time()

    # Try specified port first
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.close()
        latency_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "success": True,
            "host": host,
            "port": port,
            "latency_ms": latency_ms,
            "message": f"Koneksi berhasil! Server {host} pada port {port} merespons normal ({latency_ms} ms)."
        }
    except Exception as primary_err:
        # Fallback test with port 139 (NetBIOS SMB) if port 445 timed out or refused
        if port == 445:
            try:
                start_time_alt = time.time()
                sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock2.settimeout(2.5)
                sock2.connect((host, 139))
                sock2.close()
                latency_ms = round((time.time() - start_time_alt) * 1000, 2)
                return {
                    "success": True,
                    "host": host,
                    "port": 139,
                    "latency_ms": latency_ms,
                    "message": f"Koneksi berhasil melalui NetBIOS Port 139 ({latency_ms} ms)."
                }
            except Exception:
                pass

        return {
            "success": False,
            "host": host,
            "port": port,
            "error": (
                f"Koneksi ke {host}:{port} tidak merespons ({primary_err}). "
                "Pastikan PC sumber menyala, satu segmen jaringan, dan 'File and Printer Sharing' diizinkan di Windows Defender Firewall."
            )
        }


def inspect_storage_status(raw_dir: str) -> Dict[str, Any]:
    """
    Inspect the active photo directory:
    - Checks disk read accessibility
    - Recursively counts image files across all nested subdirectories
    - Lists ride subfolders at any depth with exact photo counts
    - Detects if currently mounted
    """
    if not os.path.exists(raw_dir):
        return {
            "exists": False,
            "readable": False,
            "total_images": 0,
            "root_images": 0,
            "subfolder_count": 0,
            "subfolders": [],
            "status": "Direktori raw belum dibuat."
        }

    is_readable = os.access(raw_dir, os.R_OK)
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

    subfolders_map: Dict[str, Dict[str, Any]] = {}
    total_images = 0
    root_images = 0

    try:
        # 1. Count root images directly in raw_dir
        for item in os.listdir(raw_dir):
            full_p = os.path.join(raw_dir, item)
            if os.path.isfile(full_p) and os.path.splitext(item)[1].lower() in valid_exts:
                root_images += 1
                total_images += 1

        # 2. Recursively discover all subdirectories and count images in each
        for root, dirs, files in os.walk(raw_dir, followlinks=True):
            if os.path.abspath(root) == os.path.abspath(raw_dir):
                continue

            rel_path = os.path.relpath(root, raw_dir).replace("\\", "/")
            direct_imgs = sum(1 for f in files if os.path.splitext(f)[1].lower() in valid_exts)
            total_images += direct_imgs

            subfolders_map[rel_path] = {
                "name": rel_path,
                "relative_path": rel_path,
                "direct_photo_count": direct_imgs,
                "photo_count": 0,  # will accumulate subtree total
                "depth": rel_path.count("/")
            }

        # 3. Accumulate subtree totals so parent folders (e.g. "2026-10-10") reflect nested photos ("2026-10-10/Ayunan")
        for path_a, info_a in subfolders_map.items():
            tot = info_a["direct_photo_count"]
            prefix = path_a + "/"
            for path_b, info_b in subfolders_map.items():
                if path_b.startswith(prefix):
                    tot += info_b["direct_photo_count"]
            info_a["photo_count"] = tot

        # 4. Sort alphabetically
        subfolders = sorted(list(subfolders_map.values()), key=lambda x: x["name"].lower())

        # Check mount status in Linux via /proc/mounts
        is_mounted = False
        mount_type = "local"
        if os.path.exists("/proc/mounts"):
            try:
                with open("/proc/mounts", "r") as f:
                    for line in f:
                        if raw_dir in line:
                            is_mounted = True
                            if "cifs" in line:
                                mount_type = "cifs"
                            elif "nfs" in line:
                                mount_type = "nfs"
                            break
            except Exception:
                pass

        return {
            "exists": True,
            "readable": is_readable,
            "total_images": total_images,
            "root_images": root_images,
            "subfolder_count": len(subfolders),
            "subfolders": subfolders,
            "is_mounted": is_mounted,
            "mount_type": mount_type,
            "raw_dir": raw_dir
        }

    except Exception as e:
        logger.error(f"Error inspecting storage {raw_dir}: {e}")
        return {
            "exists": True,
            "readable": False,
            "error": str(e),
            "total_images": 0,
            "root_images": 0,
            "subfolder_count": 0,
            "subfolders": []
        }


def create_subfolder(raw_dir: str, subfolder_path: str) -> Dict[str, Any]:
    """
    Safely create a new subfolder inside raw_dir (supports nested paths like '2026-10-10/Ayunan').
    Ensures path does not escape raw_dir.
    """
    if not subfolder_path or not subfolder_path.strip():
        raise ValueError("Nama subfolder tidak boleh kosong.")

    clean_sub = subfolder_path.strip().strip(r"\/").replace("\\", "/")
    parts = [p.strip() for p in clean_sub.split("/") if p.strip()]
    if not parts:
        raise ValueError("Nama subfolder tidak valid.")
    if ".." in parts:
        raise ValueError("Karakter '..' tidak diizinkan.")

    clean_rel = "/".join(parts)
    raw_dir_abs = os.path.abspath(raw_dir)
    target_abs = os.path.abspath(os.path.join(raw_dir_abs, clean_rel))

    if not target_abs.startswith(raw_dir_abs):
        raise ValueError("Target folder berada di luar direktori raw.")

    os.makedirs(target_abs, mode=0o777, exist_ok=True)
    try:
        os.chmod(target_abs, 0o777)
    except Exception:
        pass

    logger.info(f"Subfolder created successfully: {target_abs}")
    return {
        "success": True,
        "name": clean_rel,
        "path": target_abs,
        "message": f"Subfolder '{clean_rel}' berhasil dibuat."
    }


def delete_subfolder(raw_dir: str, subfolder_path: str) -> Dict[str, Any]:
    """
    Safely delete a subfolder and all its contents inside raw_dir.
    Ensures path does not escape raw_dir and prevents deleting raw_dir itself.
    """
    if not subfolder_path or not subfolder_path.strip():
        raise ValueError("Nama subfolder tidak boleh kosong.")

    clean_sub = subfolder_path.strip().strip(r"\/").replace("\\", "/")
    parts = [p.strip() for p in clean_sub.split("/") if p.strip()]
    if not parts:
        raise ValueError("Nama subfolder tidak valid.")
    if ".." in parts:
        raise ValueError("Karakter '..' tidak diizinkan.")

    clean_rel = "/".join(parts)
    raw_dir_abs = os.path.abspath(raw_dir)
    target_abs = os.path.abspath(os.path.join(raw_dir_abs, clean_rel))

    if not target_abs.startswith(raw_dir_abs) or target_abs == raw_dir_abs:
        raise ValueError("Tidak dapat menghapus direktori utama raw.")

    if not os.path.exists(target_abs):
        return {
            "success": True,
            "name": clean_rel,
            "path": target_abs,
            "message": f"Subfolder '{clean_rel}' sudah tidak ada di disk."
        }

    shutil.rmtree(target_abs)
    logger.info(f"Subfolder deleted successfully from disk: {target_abs}")
    return {
        "success": True,
        "name": clean_rel,
        "path": target_abs,
        "message": f"Subfolder '{clean_rel}' berhasil dihapus dari disk."
    }


def generate_mount_instructions(settings: Dict[str, Any], host_raw_path: str = "/opt/SistemPhoto/data/raw") -> Dict[str, str]:
    """
    Generate exact Linux mount commands for the Ubuntu host server.
    """
    host = settings.get("smb_host", "192.168.100.93").strip().lstrip(r"\/")
    share = settings.get("smb_share", "photo").strip().lstrip(r"\/")
    user = settings.get("smb_username", "USER_SHARE").strip()
    pwd = settings.get("smb_password", "").strip()

    auth_str = f"username={user}"
    if pwd:
        auth_str += f",password={pwd}"
    else:
        auth_str += ",guest"

    unc_path = f"//{host}/{share}"
    sub = settings.get("smb_subfolder", "").strip().strip(r"\/")
    if sub:
        unc_path += f"/{sub}"

    mount_cmd = (
        f"sudo mount -t cifs {unc_path} {host_raw_path} "
        f"-o {auth_str},file_mode=0777,dir_mode=0777,iocharset=utf8,vers=3.0"
    )

    umount_cmd = f"sudo umount {host_raw_path}"

    fstab_line = (
        f"{unc_path}  {host_raw_path}  cifs  "
        f"{auth_str},file_mode=0777,dir_mode=0777,iocharset=utf8,_netdev  0  0"
    )

    return {
        "unc_path": unc_path,
        "mount_command": mount_cmd,
        "umount_command": umount_cmd,
        "fstab_line": fstab_line
    }
