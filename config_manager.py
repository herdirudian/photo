"""
config_manager.py - Persistent Configuration and Network Share Management.
Stores configuration in data/db/settings.json and handles SMB/CIFS connection testing,
subfolder inspection, and storage synchronization.
"""

import os
import json
import time
import socket
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
    "default_threshold": 0.82,
    "admin_pin": "1234",
    "operator_pin": "1234",
    "samba_network_prefix": r"\\192.168.100.95\park-photos\results"
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
    - Counts image files
    - Lists ride subfolders
    - Detects if currently mounted
    """
    if not os.path.exists(raw_dir):
        return {
            "exists": False,
            "readable": False,
            "total_files": 0,
            "subfolders": [],
            "status": "Direktori raw belum dibuat."
        }

    is_readable = os.access(raw_dir, os.R_OK)
    subfolders: List[Dict[str, Any]] = []
    total_images = 0
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    try:
        entries = os.listdir(raw_dir)
        for item in entries:
            full_item_path = os.path.join(raw_dir, item)
            if os.path.isdir(full_item_path):
                # Count files inside this ride subfolder
                count = 0
                for root, _, files in os.walk(full_item_path):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in valid_exts:
                            count += 1
                total_images += count
                subfolders.append({
                    "name": item,
                    "photo_count": count
                })
            else:
                if os.path.splitext(item)[1].lower() in valid_exts:
                    total_images += 1

        subfolders.sort(key=lambda x: x["name"])

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
            "subfolders": []
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
