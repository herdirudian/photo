"""
main.py - FastAPI application, photo retrieval endpoints, and symlink generator.
Theme Park Offline Photo Retrieval System.
"""

import os
import io
import time
import uuid
import json
import zipfile
import shutil
import base64
import logging
import traceback
import threading
import re
import datetime
import asyncio
import secrets
import hmac
import hashlib
import tempfile
from contextlib import asynccontextmanager
from typing import Optional, List
import numpy as np

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from PIL import Image, ImageDraw, ImageOps

from database import DatabaseManager, FaceEngine, is_ignored_directory, is_ignored_path
from indexer import PhotoIndexer
import config_manager

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PhotoRetrieval.API")

# Environment paths
RAW_DIR = os.getenv("RAW_DIR", os.path.abspath("./data/raw"))
RESULTS_DIR = os.getenv("RESULTS_DIR", os.path.abspath("./data/results"))
DB_DIR = os.getenv("DB_DIR", os.path.abspath("./data/db"))
SAMBA_PREFIX = os.getenv("SAMBA_NETWORK_PREFIX", r"\\samba-server\park-photos\results")
THUMBNAILS_DIR = os.getenv("THUMBNAILS_DIR", os.path.join(DB_DIR, "thumbnails"))

# Singletons configuration
app_settings = config_manager.load_settings()
AI_ENGINE = os.getenv("AI_ENGINE", app_settings.get("ai_engine", "insightface"))
INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", app_settings.get("insightface_model", "buffalo_l"))
FACE_MODEL = os.getenv("FACE_DETECTION_MODEL", "hog")
DEFAULT_THRESHOLD = float(app_settings.get("default_threshold", 0.50 if AI_ENGINE == "insightface" else 0.82))
INSIGHTFACE_DET_SIZE = max(1280, int(os.getenv("INSIGHTFACE_DET_SIZE", app_settings.get("insightface_det_size", 1280))))
INSIGHTFACE_DET_THRESH = float(os.getenv("INSIGHTFACE_DET_THRESH", app_settings.get("insightface_det_thresh", 0.50)))

# Ensure required directories exist
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(DB_DIR)), "models", "insightface"))
for directory in [RAW_DIR, RESULTS_DIR, DB_DIR, MODELS_DIR, THUMBNAILS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Singletons initialization
face_engine = FaceEngine(
    engine=AI_ENGINE,
    model=INSIGHTFACE_MODEL if AI_ENGINE == "insightface" else FACE_MODEL,
    model_root=MODELS_DIR,
    det_size=(INSIGHTFACE_DET_SIZE, INSIGHTFACE_DET_SIZE),
    det_thresh=INSIGHTFACE_DET_THRESH
)
logger.info(
    f"Theme Park AI Face Engine active: model={INSIGHTFACE_MODEL}, "
    f"det_size={INSIGHTFACE_DET_SIZE}x{INSIGHTFACE_DET_SIZE}, "
    f"threshold={DEFAULT_THRESHOLD}, det_thresh={INSIGHTFACE_DET_THRESH}"
)
db_manager = DatabaseManager(db_dir=DB_DIR, vector_dim=face_engine.dimension)
indexer = PhotoIndexer(
    raw_dir=RAW_DIR, 
    db_manager=db_manager, 
    face_engine=face_engine,
    polling_interval=float(app_settings.get("polling_interval", 15.0))
)


def heal_customer_symlinks():
    """
    Self-heal customer result directories:
    Convert absolute container symlinks (/app/data/raw/...) into portable relative symlinks
    (../../raw/...) so they resolve identically on host Samba shares and Windows File Explorer.
    """
    if not os.path.exists(RESULTS_DIR):
        return

    healed_count = 0
    try:
        for entry in os.listdir(RESULTS_DIR):
            customer_path = os.path.join(RESULTS_DIR, entry)
            if not os.path.isdir(customer_path):
                continue

            for f in os.listdir(customer_path):
                f_path = os.path.join(customer_path, f)
                if os.path.islink(f_path):
                    try:
                        target = os.readlink(f_path)
                        # If symlink is absolute, heal to relative
                        if os.path.isabs(target):
                            candidate = target
                            if not os.path.exists(candidate) and "/app/data/raw" in target:
                                raw_sub = target.split("/app/data/raw")[-1].lstrip("/\\")
                                candidate = os.path.join(RAW_DIR, raw_sub)

                            if os.path.exists(candidate):
                                rel_target = os.path.relpath(candidate, start=customer_path)
                                os.remove(f_path)
                                os.symlink(rel_target, f_path)
                                healed_count += 1
                    except Exception as e:
                        logger.debug(f"Could not heal symlink {f_path}: {e}")
        if healed_count > 0:
            logger.info(f"Self-healed {healed_count} customer result symlinks to portable relative format.")
    except Exception as e:
        logger.warning(f"Error checking customer symlinks: {e}")


def cleanup_results_folder(purge_all: bool = False) -> int:
    """
    Remove customer result folders from RESULTS_DIR.
    If purge_all is True, removes all customer folders.
    If purge_all is False, removes folders whose modification date is older than today.
    Customer records and dwell time analytics in SQLite are 100% preserved.
    """
    if not os.path.exists(RESULTS_DIR):
        return 0

    today = datetime.date.today()
    deleted_count = 0

    try:
        for entry in os.listdir(RESULTS_DIR):
            entry_path = os.path.join(RESULTS_DIR, entry)
            should_delete = False
            if purge_all:
                should_delete = True
            else:
                try:
                    st = os.stat(entry_path)
                    mtime_date = datetime.date.fromtimestamp(st.st_mtime)
                    if mtime_date < today:
                        should_delete = True
                except Exception:
                    should_delete = True

            if should_delete:
                try:
                    if os.path.islink(entry_path) or os.path.isfile(entry_path):
                        os.unlink(entry_path)
                        deleted_count += 1
                    elif os.path.isdir(entry_path):
                        shutil.rmtree(entry_path)
                        deleted_count += 1
                except Exception as e_del:
                    logger.warning(f"Could not remove old result folder {entry_path}: {e_del}")
    except Exception as e:
        logger.error(f"Error reading RESULTS_DIR during cleanup: {e}")

    return deleted_count


def try_parse_folder_date(folder_name: str) -> Optional[datetime.date]:
    """Attempt parsing date from folder names (e.g. 2026-10-25, 2026_10_25, 25-10-2026, Ayunan_2026-10-25)."""
    name = folder_name.strip()
    for fmt in ("%Y-%m-%d", "%Y_%m_%d", "%Y.%m.%d", "%Y%m%d", "%d-%m-%Y", "%d_%m_%Y", "%d.%m.%Y"):
        try:
            return datetime.datetime.strptime(name, fmt).date()
        except ValueError:
            pass
    m = re.search(r"(\d{4}[-_.]\d{2}[-_.]\d{2})", name)
    if m:
        d_str = m.group(1).replace("_", "-").replace(".", "-")
        try:
            return datetime.datetime.strptime(d_str, "%Y-%m-%d").date()
        except ValueError:
            pass
    m2 = re.search(r"(\d{2}[-_.]\d{2}[-_.]\d{4})", name)
    if m2:
        d_str = m2.group(1).replace("_", "-").replace(".", "-")
        try:
            return datetime.datetime.strptime(d_str, "%d-%m-%Y").date()
        except ValueError:
            pass
    return None


def cleanup_raw_folder(max_age_days: int = 2, force: bool = False) -> dict:
    """
    Purge photo folders and image files in RAW_DIR that are older than max_age_days.
    NOTE: Automated deletion is 100% disabled by default (manual operator control).
    Will only execute if force=True (manual admin trigger) or auto_delete_raw setting is explicitly set to True.
    """
    if not force and not config_manager.get_setting("auto_delete_raw", False):
        logger.info("Raw auto-delete is disabled (100% manual operator control). No raw photos deleted.")
        return {"deleted_folders": 0, "deleted_files": 0, "purged_photos": 0}

    if not os.path.exists(RAW_DIR):
        return {"deleted_folders": 0, "deleted_files": 0, "purged_photos": 0}

    today = datetime.date.today()
    deleted_folders = 0
    deleted_files = 0

    try:
        # Step 1: Scan top-level and second-level subdirectories in RAW_DIR for date folders
        folders_to_delete = []

        try:
            for item in os.listdir(RAW_DIR):
                if is_ignored_directory(item) or item.startswith("."):
                    continue
                item_path = os.path.join(RAW_DIR, item)
                if not os.path.isdir(item_path):
                    continue

                parsed_d = try_parse_folder_date(item)
                if parsed_d is not None:
                    age_days = (today - parsed_d).days
                    if age_days >= max_age_days:
                        folders_to_delete.append(item_path)
                else:
                    # Check nested subfolders (e.g. raw/Ayunan/2026-10-25)
                    try:
                        for sub in os.listdir(item_path):
                            if is_ignored_directory(sub) or sub.startswith("."):
                                continue
                            sub_path = os.path.join(item_path, sub)
                            if os.path.isdir(sub_path):
                                sub_d = try_parse_folder_date(sub)
                                if sub_d is not None:
                                    sub_age = (today - sub_d).days
                                    if sub_age >= max_age_days:
                                        folders_to_delete.append(sub_path)
                    except Exception as e_sub:
                        logger.debug(f"Error scanning subfolder {item_path}: {e_sub}")
        except Exception as e_ls:
            logger.error(f"Error listing RAW_DIR: {e_ls}")

        # Delete the identified date folders
        for folder_path in folders_to_delete:
            try:
                if os.path.exists(folder_path):
                    logger.info(f"Auto-cleanup: Removing raw photo folder older than {max_age_days} days: {folder_path}")
                    shutil.rmtree(folder_path, ignore_errors=True)
                    db_manager.delete_folder(folder_path)
                    deleted_folders += 1
            except Exception as e_fdel:
                logger.error(f"Error removing old raw folder {folder_path}: {e_fdel}")

        # Step 2: Delete individual files older than max_age_days in remaining non-date folders or root
        for root, dirs, files in os.walk(RAW_DIR, topdown=False, followlinks=False):
            dirs[:] = [d for d in dirs if not is_ignored_directory(d)]
            if is_ignored_path(root):
                continue

            for f in files:
                if f.startswith(".") or f == ".gitkeep":
                    continue
                f_path = os.path.join(root, f)
                try:
                    st = os.stat(f_path)
                    if (time.time() - st.st_mtime) >= (max_age_days * 86400):
                        os.unlink(f_path)
                        db_manager.delete_photo_by_path(f_path)
                        deleted_files += 1
                except Exception:
                    pass

            # ONLY remove empty date subfolders (depth >= 2), NEVER delete top-level wahana folders directly under RAW_DIR!
            parent_dir = os.path.abspath(os.path.dirname(root))
            if parent_dir != os.path.abspath(RAW_DIR) and os.path.abspath(root) != os.path.abspath(RAW_DIR):
                try:
                    remaining = [x for x in os.listdir(root) if not x.startswith(".")]
                    if not remaining:
                        os.rmdir(root)
                        logger.debug(f"Removed now-empty subfolder: {root}")
                except Exception:
                    pass

        # Step 3: Purge any remaining orphaned records from SQLite and FAISS
        purge_stats = db_manager.purge_missing_photos(RAW_DIR)
        purged_photos = purge_stats.get("purged_photos", 0)

        logger.info(
            f"Raw retention cleanup complete (older than {max_age_days} days): "
            f"{deleted_folders} folder(s) deleted, {deleted_files} file(s) deleted, {purged_photos} database record(s) purged."
        )

        return {
            "deleted_folders": deleted_folders,
            "deleted_files": deleted_files,
            "purged_photos": purged_photos
        }

    except Exception as e:
        logger.error(f"Error during raw retention cleanup: {e}", exc_info=True)
        return {"deleted_folders": deleted_folders, "deleted_files": deleted_files, "purged_photos": 0}


results_cleanup_stop_event = threading.Event()


def _daily_results_cleanup_worker():
    """Background worker monitoring day rollover at midnight (00:00:00) to auto-purge results and raw folders."""
    last_date = datetime.date.today()
    logger.info(f"Daily maintenance auto-cleaner active. Monitoring day rollover from: {last_date}")
    while not results_cleanup_stop_event.is_set():
        if results_cleanup_stop_event.wait(60.0):
            break
        current_date = datetime.date.today()
        if current_date != last_date:
            logger.info(f"Day rollover detected ({last_date} -> {current_date}). Starting automated daily maintenance...")
            try:
                purged = cleanup_results_folder(purge_all=False)
                logger.info(f"Daily rollover cleanup complete: {purged} result folder(s) removed. Customer history preserved.")
            except Exception as e:
                logger.error(f"Error during rollover results cleanup: {e}")

            # Raw photo auto-delete is disabled (100% manual operator control)
            if config_manager.get_setting("auto_delete_raw", False):
                try:
                    retention_days = int(config_manager.get_setting("raw_retention_days", 2))
                    raw_stats = cleanup_raw_folder(max_age_days=retention_days)
                    logger.info(f"Daily raw rollover cleanup complete (older than {retention_days} days): {raw_stats}")
                except Exception as e_raw:
                    logger.error(f"Error during rollover raw cleanup: {e_raw}")

            last_date = current_date


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start watchdog folder monitoring on application launch, and stop on shutdown."""
    logger.info("Initializing Theme Park Photo Retrieval Engine...")
    indexer.start()
    results_cleanup_stop_event.clear()

    # Run heavy startup maintenance (symlinks, purge invalid boxes, raw & result retentions)
    # asynchronously in background thread so port 8000 opens INSTANTLY with zero 502 Bad Gateway!
    def _startup_maintenance():
        try:
            heal_customer_symlinks()
        except Exception as e_h:
            logger.warning(f"Symlink healing error: {e_h}")
        try:
            db_manager.purge_invalid_face_boxes()
        except Exception as e_purge:
            logger.warning(f"Could not purge invalid face boxes on startup: {e_purge}")
        try:
            startup_purged = cleanup_results_folder(purge_all=False)
            if startup_purged > 0:
                logger.info(f"Startup maintenance: Purged {startup_purged} result folder(s) from previous days.")
        except Exception as e_clean:
            logger.warning(f"Startup results cleanup error: {e_clean}")
        # Raw photo auto-delete is disabled (100% manual operator control)
        if config_manager.get_setting("auto_delete_raw", False):
            try:
                retention_days = int(config_manager.get_setting("raw_retention_days", 2))
                startup_raw_stats = cleanup_raw_folder(max_age_days=retention_days)
                if startup_raw_stats.get("deleted_folders", 0) > 0 or startup_raw_stats.get("deleted_files", 0) > 0:
                    logger.info(f"Startup raw maintenance: Cleaned photos older than {retention_days} days: {startup_raw_stats}")
            except Exception as e_raw_clean:
                logger.warning(f"Startup raw cleanup error: {e_raw_clean}")

    threading.Thread(target=_startup_maintenance, daemon=True, name="StartupMaintenanceWorker").start()
    cleanup_thread = threading.Thread(target=_daily_results_cleanup_worker, daemon=True, name="DailyResultsCleaner")
    cleanup_thread.start()
    yield
    logger.info("Shutting down Photo Retrieval Engine...")
    indexer.stop()
    results_cleanup_stop_event.set()


app = FastAPI(
    title="Theme Park Offline Photo Retrieval System",
    description="Automated raw ride photo ingestion, FAISS similarity search, and Samba symlink generation.",
    version="1.0.0",
    lifespan=lifespan
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
    err_tb = traceback.format_exc()
    logger.error(f"Global exception on {request.url.path}: {exc}\n{err_tb}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Kesalahan Pemrosesan Server",
            "detail": f"{type(exc).__name__}: {str(exc)}",
            "suggestion": "Silakan coba kembali atau periksa log container server."
        }
    )

# Static and Templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
IMG_DIR = os.path.join(BASE_DIR, "img")

os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if os.path.exists(IMG_DIR):
    app.mount("/img", StaticFiles(directory=IMG_DIR), name="img")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def format_samba_unc(prefix: str, subpath: str = "", fallback_host: str = "10.12.2.150") -> str:
    """
    Produce a strictly valid Windows UNC network path (e.g. \\<host>\park-photos\results\Customer_XXXX).
    Guarantees exactly two leading backslashes and single backslashes in between.
    Dynamically aligns to the accessed server IP / hostname.
    """
    import re
    if not prefix or prefix.startswith("/mnt") or "samba-server" in prefix or "192.168.100." in prefix:
        prefix = rf"\\{fallback_host}\park-photos\results"
    elif fallback_host and fallback_host not in ["127.0.0.1", "localhost", "0.0.0.0"]:
        parts = [p for p in re.split(r"[\\/]+", prefix.strip()) if p]
        if parts:
            parts[0] = fallback_host
            joined_parts = "\\".join(parts)
            prefix = rf"\\{joined_parts}"

    clean_p = re.sub(r"^[\\/]+", "", prefix.strip())
    clean_p = re.sub(r"[\\/]+", r"\\", clean_p)

    clean_sub = re.sub(r"^[\\/]+", "", subpath.strip()) if subpath else ""
    clean_sub = re.sub(r"[\\/]+", r"\\", clean_sub)

    if clean_sub:
        return rf"\\{clean_p}\{clean_sub}"
    return rf"\\{clean_p}"



def get_logo_file_path() -> Optional[str]:
    candidates = [
        os.path.join(STATIC_DIR, "logo.png"),
        os.path.join(IMG_DIR, "logotlm.png"),
        os.path.join(BASE_DIR, "static", "logo.png"),
        os.path.join(BASE_DIR, "img", "logotlm.png"),
        "/app/static/logo.png",
        "/app/img/logotlm.png",
        "static/logo.png",
        "img/logotlm.png",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def get_logo_base64() -> str:
    path = get_logo_file_path()
    if path:
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.warning(f"Failed to read logo for base64: {e}")
    return ""


@app.get("/logo.png")
async def serve_logo():
    """Serve The Lodge Maribaya official logo."""
    path = get_logo_file_path()
    if path:
        return FileResponse(path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Logo not found")


@app.get("/manifest.json")
async def serve_manifest():
    """Serve PWA Web App Manifest."""
    manifest_path = os.path.join(STATIC_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        return FileResponse(manifest_path, media_type="application/manifest+json")
    raise HTTPException(status_code=404, detail="Manifest not found")


@app.get("/sw.js")
async def serve_service_worker():
    """Serve PWA Service Worker script with root scope permission."""
    sw_path = os.path.join(STATIC_DIR, "sw.js")
    if os.path.exists(sw_path):
        return FileResponse(
            sw_path, 
            media_type="application/javascript",
            headers={
                "Service-Worker-Allowed": "/",
                "Cache-Control": "no-cache, no-store, must-revalidate"
            }
        )
    raise HTTPException(status_code=404, detail="Service worker not found")



def get_comprehensive_stats() -> dict:
    """Return real-time database, disk storage, and indexing progress metrics."""
    stats = db_manager.get_stats()
    storage_info = config_manager.inspect_storage_status(RAW_DIR)
    total_disk_photos = storage_info.get("total_images", 0)

    prog = indexer.get_progress()
    prog_is_indexing = bool(prog.get("is_indexing", False))
    queue_pending = prog.get("pending_in_queue", 0)
    scan_total = prog.get("total_to_process", 0)
    scan_processed = prog.get("current_processed", 0)
    scan_pending = max(0, scan_total - scan_processed) if (prog_is_indexing and scan_total > 0) else 0

    total_indexed = stats.get("total_photos", 0)
    
    # Calculate genuinely pending unindexed photos on disk
    disk_pending = max(0, total_disk_photos - total_indexed)

    if prog_is_indexing or queue_pending > 0 or scan_pending > 0:
        is_indexing = True
        pending_photos = max(disk_pending, queue_pending, scan_pending)
    else:
        is_indexing = False
        pending_photos = 0

    effective_total_disk = max(total_disk_photos, total_indexed + pending_photos)

    if not is_indexing:
        pct = 100.0
    elif scan_total > 0:
        pct = round((scan_processed / scan_total * 100), 1)
        pct = min(99.9, pct)
    elif pending_photos == 0:
        pct = 100.0
    else:
        pct = round((total_indexed / effective_total_disk * 100), 1) if effective_total_disk > 0 else 100.0
        pct = min(99.9, pct)

    stats["total_disk_photos"] = effective_total_disk
    stats["pending_photos"] = pending_photos
    stats["is_indexing"] = is_indexing
    stats["indexing_progress"] = {
        "is_active": is_indexing,
        "indexed": total_indexed,
        "total_disk": effective_total_disk,
        "pending": pending_photos,
        "percent": pct,
        "current_processed": scan_processed,
        "total_to_process": scan_total,
        "current_file": prog.get("current_file", ""),
        "last_completed_at": prog.get("last_completed_at")
    }
    stats["raw_dir"] = RAW_DIR
    stats["results_dir"] = RESULTS_DIR
    stats["samba_prefix"] = SAMBA_PREFIX
    stats["ai_engine"] = face_engine.engine_name
    stats["vector_dimension"] = face_engine.dimension
    stats["model_name"] = face_engine.model_name
    det_size_val = face_engine.det_size[0] if isinstance(getattr(face_engine, "det_size", None), (list, tuple)) else 1280
    stats["det_size"] = det_size_val
    stats["det_thresh"] = getattr(face_engine, "det_thresh", 0.50)
    return stats


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    stats = get_comprehensive_stats()
    cfg = config_manager.load_settings()
    storage_info = config_manager.inspect_storage_status(RAW_DIR)
    subfolders = [s["name"] for s in storage_info.get("subfolders", [])]
    context = {
        "request": request,
        "stats": stats,
        "ai_engine": face_engine.engine_name,
        "vector_dim": face_engine.dimension,
        "model_name": face_engine.model_name,
        "default_threshold": cfg.get("default_threshold", DEFAULT_THRESHOLD),
        "samba_prefix": cfg.get("samba_network_prefix", SAMBA_PREFIX),
        "raw_dir": RAW_DIR,
        "results_dir": RESULTS_DIR,
        "logo_base64": get_logo_base64(),
        "require_pin_to_access": cfg.get("require_pin_to_access", True),
        "subfolders": subfolders,
    }
    try:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=context
        )
    except TypeError:
        return templates.TemplateResponse(
            "index.html",
            context
        )


@app.get("/admin", response_class=HTMLResponse)
async def serve_admin_dashboard(request: Request):
    """Serve the Administrator Dashboard for network storage and system management."""
    stats = get_comprehensive_stats()
    context = {
        "request": request,
        "stats": stats,
        "ai_engine": face_engine.engine_name,
        "vector_dim": face_engine.dimension,
        "model_name": face_engine.model_name,
        "logo_base64": get_logo_base64()
    }
    try:
        return templates.TemplateResponse(
            request=request,
            name="admin.html",
            context=context
        )
    except TypeError:
        return templates.TemplateResponse("admin.html", context)


# ==========================================
# SECURITY & AUTHENTICATION UTILITIES
# ==========================================

SECRET_KEY_PATH = os.path.join(DB_DIR, ".secret_key")

def _get_or_create_secret_key() -> str:
    if os.path.exists(SECRET_KEY_PATH):
        try:
            with open(SECRET_KEY_PATH, "r", encoding="utf-8") as f:
                k = f.read().strip()
                if len(k) >= 32:
                    return k
        except Exception:
            pass
    k = secrets.token_hex(32)
    try:
        with open(SECRET_KEY_PATH, "w", encoding="utf-8") as f:
            f.write(k)
        os.chmod(SECRET_KEY_PATH, 0o600)
    except Exception:
        pass
    return k

APP_SECRET_KEY = _get_or_create_secret_key()

def create_admin_token() -> str:
    now = int(time.time())
    data = f"admin:{now}"
    sig = hmac.new(APP_SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
    return f"{data}:{sig}"

def verify_admin_token(token: str) -> bool:
    if not token or ":" not in token:
        return False
    parts = token.split(":")
    if len(parts) != 3:
        return False
    role, timestamp_str, sig = parts
    if role != "admin":
        return False
    try:
        ts = int(timestamp_str)
        if time.time() - ts > 86400 * 7 or time.time() < ts - 60:
            return False
    except ValueError:
        return False
    data = f"{role}:{timestamp_str}"
    expected_sig = hmac.new(APP_SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected_sig)

def require_admin(request: Request):
    token = request.cookies.get("lodge_admin_session") or request.headers.get("x-admin-token")
    if not token or not verify_admin_token(token):
        raise HTTPException(
            status_code=401,
            detail="Akses ditolak: Sesi Admin tidak valid atau telah kedaluwarsa. Silakan masukkan PIN Admin."
        )
    return True


# ==========================================
# ADMIN CONFIGURATION & NETWORK STORAGE APIS
# ==========================================

@app.get("/api/admin/settings")
async def get_admin_settings(request: Request):
    """Retrieve current system and network storage settings."""
    token = request.cookies.get("lodge_admin_session") or request.headers.get("x-admin-token")
    is_authenticated = verify_admin_token(token) if token else False

    cfg = config_manager.load_settings()
    safe_cfg = dict(cfg)
    if safe_cfg.get("smb_password"):
        safe_cfg["smb_password"] = "••••••••"

    # Only authenticated admin sees the real PINs and CRM API keys
    if not is_authenticated:
        safe_cfg["admin_pin"] = "••••"
        safe_cfg["operator_pin"] = "••••"
        safe_cfg["crm_guest_api_key"] = "••••••••"
    return safe_cfg


@app.post("/api/admin/settings")
async def update_admin_settings(request: Request):
    """Save new system and network storage configuration."""
    require_admin(request)
    data = await request.json()
    existing = config_manager.load_settings()

    # Preserve existing secrets if masked string or empty is sent
    if data.get("smb_password") in ["••••••••", "", None]:
        data["smb_password"] = existing.get("smb_password", "")
    if data.get("admin_pin") in ["••••", "", None]:
        data["admin_pin"] = existing.get("admin_pin", "1234")
    if data.get("operator_pin") in ["••••", "", None]:
        data["operator_pin"] = existing.get("operator_pin", "1234")
    if data.get("crm_guest_api_key") in ["••••••••", "", None]:
        data["crm_guest_api_key"] = existing.get("crm_guest_api_key", "")

    updated = config_manager.save_settings(data)
    if "polling_interval" in updated:
        try:
            indexer.polling_interval = float(updated["polling_interval"])
        except Exception:
            pass
    if "insightface_det_size" in updated:
        try:
            new_det_size = int(updated["insightface_det_size"])
            if hasattr(face_engine, "update_det_size"):
                face_engine.update_det_size(new_det_size)
        except Exception as e:
            logger.warning(f"Could not apply new det_size: {e}")
    return {"success": True, "settings": updated}


@app.post("/api/admin/test-connection")
async def test_admin_connection(request: Request):
    """Test direct TCP connectivity to SMB server host on port 445."""
    require_admin(request)
    data = await request.json()
    host = data.get("host", "").strip()
    port = int(data.get("port", 445))
    return config_manager.test_network_connection(host=host, port=port)


@app.get("/api/admin/storage-status")
async def get_storage_status():
    """Inspect active storage folder, total photos, and detected ride subfolders."""
    return config_manager.inspect_storage_status(RAW_DIR)


@app.get("/api/subfolders")
async def get_guest_subfolders():
    """Return all detected ride subfolders and nested directories with photo counts."""
    storage_info = config_manager.inspect_storage_status(RAW_DIR)
    return {
        "success": True,
        "total_images": storage_info.get("total_images", 0),
        "root_images": storage_info.get("root_images", 0),
        "subfolder_count": storage_info.get("subfolder_count", 0),
        "subfolders": storage_info.get("subfolders", [])
    }


@app.get("/api/admin/subfolders")
async def get_admin_subfolders():
    """Admin endpoint for ride subfolders."""
    return await get_guest_subfolders()


@app.post("/api/admin/subfolders")
async def create_new_subfolder(request: Request):
    """Create a new ride subfolder in raw storage."""
    require_admin(request)
    data = await request.json()
    folder_name = data.get("folder_name") or data.get("path") or data.get("name") or ""
    try:
        res = config_manager.create_subfolder(RAW_DIR, folder_name)
        storage_info = config_manager.inspect_storage_status(RAW_DIR)
        return {
            "success": True,
            "message": res["message"],
            "folder": res,
            "storage_status": storage_info
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error creating subfolder {folder_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal membuat subfolder: {e}")


@app.delete("/api/admin/subfolders")
async def delete_existing_subfolder(request: Request):
    """Safely delete a ride subfolder from disk and immediately purge database/FAISS records."""
    require_admin(request)
    data = await request.json()
    folder_path = data.get("folder_path") or data.get("folder_name") or data.get("path") or ""
    try:
        clean_sub = folder_path.strip().strip(r"\/").replace("\\", "/")
        target_abs = os.path.abspath(os.path.join(RAW_DIR, clean_sub))

        # 1. Purge from SQLite and FAISS
        purge_res = db_manager.delete_folder(target_abs)

        # 2. Delete directory tree from disk
        del_disk_res = config_manager.delete_subfolder(RAW_DIR, clean_sub)

        storage_info = config_manager.inspect_storage_status(RAW_DIR)
        return {
            "success": True,
            "message": f"Subfolder '{clean_sub}' berhasil dihapus ({purge_res['deleted_photos']} foto dibersihkan dari database).",
            "purged": purge_res,
            "storage_status": storage_info
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error deleting subfolder {folder_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal menghapus subfolder: {e}")


@app.post("/api/admin/subfolders/scan")
async def scan_specific_subfolder(request: Request):
    """Trigger indexing scan on a specific subfolder."""
    require_admin(request)
    data = await request.json()
    folder_path = data.get("folder_path") or data.get("path") or ""
    try:
        summary = indexer.scan_existing_files(force=False, target_subpath=folder_path)
        storage_info = config_manager.inspect_storage_status(RAW_DIR)
        return {
            "success": True,
            "message": f"Pemindaian subfolder '{folder_path}' selesai.",
            "summary": summary,
            "storage_status": storage_info
        }
    except Exception as e:
        logger.error(f"Error scanning subfolder {folder_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/purge-stale")
async def purge_stale_records(request: Request):
    """Manually trigger purging of photos missing from disk."""
    require_admin(request)
    return db_manager.purge_missing_photos(RAW_DIR)


@app.post("/api/admin/results/cleanup")
async def manual_cleanup_results(request: Request, all_folders: bool = Query(False)):
    """Manually purge customer result folders (all or older than today)."""
    require_admin(request)
    deleted = cleanup_results_folder(purge_all=all_folders)
    return {
        "success": True,
        "deleted_folders": deleted,
        "message": f"Berhasil membersihkan {deleted} folder hasil tamu dari disk."
    }


@app.post("/api/admin/raw/cleanup")
async def manual_cleanup_raw(request: Request, max_age_days: int = Query(2)):
    """Manually trigger retention cleanup of raw photo folders older than max_age_days."""
    require_admin(request)
    stats = cleanup_raw_folder(max_age_days=max_age_days, force=True)
    return {
        "success": True,
        "max_age_days": max_age_days,
        "deleted_folders": stats.get("deleted_folders", 0),
        "deleted_files": stats.get("deleted_files", 0),
        "purged_photos": stats.get("purged_photos", 0),
        "message": f"Berhasil membersihkan folder dan foto raw yang lebih dari {max_age_days} hari ({stats.get('deleted_folders', 0)} folder, {stats.get('deleted_files', 0)} file dihapus permanen)."
    }


@app.post("/api/admin/reset-database")
async def reset_database(request: Request):
    """Reset and empty SQLite photos and FAISS vector index."""
    require_admin(request)
    data = await request.json()
    cfg = config_manager.load_settings()
    admin_pin = str(cfg.get("admin_pin", "1234")).strip()
    if str(data.get("pin", "")).strip() != admin_pin:
        raise HTTPException(status_code=403, detail="PIN Admin tidak valid.")
    db_manager.reset_all_data()
    # Clear thumbnail cache
    try:
        if os.path.exists(THUMBNAILS_DIR):
            for fn in os.listdir(THUMBNAILS_DIR):
                fp = os.path.join(THUMBNAILS_DIR, fn)
                if os.path.isfile(fp):
                    os.remove(fp)
    except Exception as e_th:
        logger.warning(f"Could not purge thumbnails directory on reset: {e_th}")
    return {"success": True, "message": "Database dan indeks FAISS berhasil direset."}


@app.post("/api/admin/verify-pin")
async def verify_admin_pin(request: Request, response: Response):
    """Verify administrator PIN and issue secure admin session token."""
    data = await request.json()
    cfg = config_manager.load_settings()
    admin_pin = str(cfg.get("admin_pin", "1234")).strip()
    entered_pin = str(data.get("pin", "")).strip()
    is_valid = bool(entered_pin and entered_pin == admin_pin)
    if is_valid:
        token = create_admin_token()
        response.set_cookie(
            key="lodge_admin_session",
            value=token,
            httponly=True,
            samesite="lax",
            max_age=86400 * 7
        )
        return {"valid": True, "token": token, "message": "Akses Admin berhasil diverifikasi."}
    return {"valid": False, "message": "PIN Admin salah."}


@app.get("/api/admin/reports/dwell-time")
async def get_dwell_time_report(
    request: Request,
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format")
):
    """Retrieve daily visitor dwell time metrics and breakdown."""
    require_admin(request)
    summary = db_manager.get_analytics_summary(date)
    return summary


@app.get("/api/admin/reports/export-csv")
async def export_dwell_time_csv(
    request: Request,
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format or 'all'")
):
    """Download CSV report of guest dwell time and visits."""
    require_admin(request)
    csv_str = db_manager.export_analytics_csv(date)
    target_date = date or datetime.date.today().strftime("%Y-%m-%d")
    filename = f"laporan-kunjungan-thelodge-{target_date}.csv"
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@app.post("/api/verify-access-pin")
async def verify_access_pin(request: Request, response: Response):
    """
    Verify operator or administrator PIN before opening the photo search page.
    """
    data = await request.json()
    entered_pin = str(data.get("pin", "")).strip()
    cfg = config_manager.load_settings()
    operator_pin = str(cfg.get("operator_pin", "1234")).strip()
    admin_pin = str(cfg.get("admin_pin", "1234")).strip()

    if not entered_pin:
        return {"valid": False, "message": "PIN tidak boleh kosong."}

    is_admin = (entered_pin == admin_pin)
    is_operator = (entered_pin == operator_pin)
    is_match = is_admin or is_operator
    logger.info(f"[PIN CHECK] Attempt processed - Success: {is_match}")

    if is_match:
        role = "admin" if is_admin else "operator"
        if is_admin:
            token = create_admin_token()
            response.set_cookie(
                key="lodge_admin_session",
                value=token,
                httponly=True,
                samesite="lax",
                max_age=86400 * 7
            )
        return {
            "valid": True,
            "role": role,
            "message": "Akses diberikan."
        }
    return {"valid": False, "message": "PIN yang Anda masukkan salah."}


@app.get("/api/health")
async def health_check():
    """System health check endpoint for Docker container status."""
    stats = db_manager.get_stats()
    return {
        "status": "online",
        "ai_engine": face_engine.engine_name,
        "vector_dimension": face_engine.dimension,
        "model_name": face_engine.model_name,
        "faiss_ready": stats["faiss_ready"],
        "face_rec_ready": stats["face_rec_ready"],
        "insightface_ready": stats.get("insightface_ready", False),
        "indexed_photos": stats["total_photos"],
        "indexed_faces": stats["total_faces"],
        "timestamp": time.time()
    }


@app.get("/api/stats")
async def get_system_stats():
    """Return real-time database, disk storage, and indexing progress metrics."""
    return get_comprehensive_stats()


@app.post("/api/reindex")
async def trigger_reindex(force: bool = Query(False)):
    """Trigger manual re-scan of raw photos directory in background thread."""
    try:
        threading.Thread(
            target=indexer.scan_existing_files,
            kwargs={"force": force},
            daemon=True,
            name="ManualScanWorker"
        ).start()
        return {
            "success": True,
            "message": "Pemindaian foto wahana dimulai di latar belakang."
        }
    except Exception as e:
        logger.error(f"Error starting re-index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/guests/search")
async def search_crm_guests(
    q: Optional[str] = Query(None, description="Search keyword for guest name or phone number"),
    limit: int = Query(15, ge=1, le=50)
):
    """
    Proxy endpoint to query guest registrations from Cloud CRM (submit.thelodgegroup.id).
    Sends the request with x-api-key header and browser User-Agent.
    Handles network timeouts and VPS status gracefully.
    """
    cfg = config_manager.load_settings()
    api_url = os.getenv("CRM_GUEST_API_URL", cfg.get("crm_guest_api_url", "https://submit.thelodgegroup.id/api/guests/search"))
    api_key = os.getenv("CRM_GUEST_API_KEY", cfg.get("crm_guest_api_key", "thelodge-photo-secret-2026"))
    
    clean_q = (q or "").strip()
    
    def _fetch_from_crm():
        import urllib.request
        import urllib.parse
        import urllib.error
        import ssl
        
        params = {"limit": str(limit)}
        if clean_q:
            params["q"] = clean_q
            
        full_url = f"{api_url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            full_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "x-api-key": api_key,
                "Accept": "application/json"
            }
        )
        
        ctx = ssl.create_default_context()
        
        try:
            with urllib.request.urlopen(req, timeout=4.0, context=ctx) as resp:
                data = resp.read().decode("utf-8")
                return json.loads(data)
        except urllib.error.HTTPError as he:
            err_body = ""
            try:
                err_body = he.read().decode("utf-8")
            except Exception:
                pass
            logger.warning(f"CRM API HTTP error {he.code}: {err_body[:200]}")
            if he.code == 404:
                return {
                    "success": False,
                    "error": "Endpoint CRM belum aktif di server cloud (404). Silakan deploy pembaruan di VPS, atau gunakan input manual.",
                    "guests": []
                }
            elif he.code in (401, 403):
                return {
                    "success": False,
                    "error": "Akses ke CRM cloud ditolak (Autentikasi API Key tidak valid).",
                    "guests": []
                }
            return {
                "success": False,
                "error": f"Server CRM mengembalikan status {he.code}.",
                "guests": []
            }
        except urllib.error.URLError as ue:
            logger.warning(f"CRM API connection failed: {ue}")
            return {
                "success": False,
                "error": "Tidak dapat terhubung ke server CRM cloud (Koneksi offline/timeout). Silakan gunakan input manual.",
                "guests": []
            }
        except Exception as ex:
            logger.error(f"CRM API unexpected error: {ex}")
            return {
                "success": False,
                "error": f"Kesalahan koneksi CRM: {str(ex)}",
                "guests": []
            }

    data = await asyncio.to_thread(_fetch_from_crm)
    return data


def _sync_visit_to_crm(visit_record: dict):
    """Send visit completion callback to Cloud CRM (submit.thelodgegroup.id)."""
    try:
        import urllib.request
        import urllib.error
        import ssl
        cfg = config_manager.load_settings()
        base_api = os.getenv("CRM_GUEST_API_URL", cfg.get("crm_guest_api_url", "https://submit.thelodgegroup.id/api/guests/search"))
        callback_url = base_api.rsplit("/", 1)[0] + "/visit-complete"
        api_key = os.getenv("CRM_GUEST_API_KEY", cfg.get("crm_guest_api_key", "thelodge-photo-secret-2026"))

        payload = json.dumps(visit_record).encode("utf-8")
        req = urllib.request.Request(
            callback_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "LodgePhotoRetrieval/1.0",
                "x-api-key": api_key,
                "Accept": "application/json"
            },
            method="POST"
        )
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=5.0, context=ctx) as resp:
            logger.info(f"Successfully synced visit to CRM: HTTP {resp.status}")
    except Exception as e:
        logger.warning(f"Could not sync visit to CRM ({e})")


@app.post("/api/search")
async def search_guest_photos(
    request: Request,
    files: List[UploadFile] = File(...),
    threshold: float = Form(DEFAULT_THRESHOLD),
    top_k: int = Form(0),
    ride_filter: Optional[str] = Form(None),
    guest_name: Optional[str] = Form(None),
    guest_phone: Optional[str] = Form(None),
    guest_crm_id: Optional[str] = Form(None),
    guest_registered_at: Optional[str] = Form(None)
):
    """
    Search endpoint:
    1. Validates mandatory guest identification (CRM form or manual entry).
    2. Reads reference photos of guest (Multi-pose).
    3. Extracts 128-d or 512-d face vector and HSV color histogram from each pose.
    4. Searches FAISS for matching faces above similarity threshold.
    5. Creates virtual folder /app/data/results/Customer_{UUID} with Linux symlinks.
    6. Returns matches and network folder path.
    """
    try:
        # Mandatory validation: Guest data must be provided either via CRM or manual walk-in
        clean_name = (guest_name or "").strip()
        clean_phone = (guest_phone or "").strip()
        clean_crm = (guest_crm_id or "").strip()

        if not clean_name and not clean_phone and not clean_crm:
            logger.warning("Pencarian foto ditolak: Data identitas tamu wajib diisi.")
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": "Data Tamu Wajib Diisi",
                    "suggestion": "Silakan pilih data tamu dari Form Online CRM atau masukkan Nama Tamu (Walk-in) secara manual sebelum melakukan pencarian foto."
                }
            )

        start_time = time.time()
        face_engine.begin_search_priority()
        try:
            raw_query_faces = []
            
            for file_idx, file in enumerate(files):
                content_type = getattr(file, "content_type", "") or ""
                filename = getattr(file, "filename", "") or ""
                if not content_type.startswith("image/") and not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                    continue
                    
                try:
                    image_bytes = await file.read()
                    extracted_faces = face_engine.extract_faces_from_bytes(image_bytes)
                    
                    if extracted_faces:
                        # If multiple faces detected in one photo, filter out tiny background passerby faces
                        # Keep faces that are at least 12% the size of the largest face in that image
                        max_face_area = max(
                            (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][1] - f["bbox"][3])
                            for f in extracted_faces
                        )
                        min_valid_area = max(400, int(max_face_area * 0.12))

                        for face in extracted_faces:
                            area = (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][1] - face["bbox"][3])
                            if area >= min_valid_area:
                                face["source_file_idx"] = file_idx
                                raw_query_faces.append(face)
                        
                except Exception as e:
                    logger.error(f"Failed to process reference image: {e}")
                    continue

            if not raw_query_faces:
                return JSONResponse(
                    status_code=422,
                    content={
                        "success": False,
                        "error": "No face detected in reference photos.",
                        "suggestion": "Please capture clear, well-lit photos of the guest's face without heavy obstruction."
                    }
                )

            # Cluster extracted faces into distinct family / group members
            # Cosine similarity >= 0.62 indicates different poses/photos of the SAME individual
            # Similarity < 0.62 indicates a DIFFERENT family member
            members = []
            for face in raw_query_faces:
                face_enc = face["encoding"]
                src_idx = face.get("source_file_idx", 0)
                
                best_sim = -1.0
                best_member = None
                
                for m in members:
                    # Faces from the same photo cannot belong to the same person
                    if any(mf.get("source_file_idx") == src_idx for mf in m["faces"]):
                        continue
                    
                    for mf in m["faces"]:
                        sim = float(np.dot(face_enc, mf["encoding"]))
                        if sim > best_sim:
                            best_sim = sim
                            if sim >= 0.62:
                                best_member = m
                
                if best_member is not None and best_sim >= 0.62:
                    face["member_id"] = best_member["member_id"]
                    best_member["faces"].append(face)
                else:
                    new_id = len(members) + 1
                    face["member_id"] = new_id
                    members.append({
                        "member_id": new_id,
                        "name": f"Anggota {new_id}",
                        "faces": [face]
                    })

            total_members = len(members)
            logger.info(
                f"Reference processing complete: {len(raw_query_faces)} face(s) clustered into "
                f"{total_members} distinct member(s) across {len(files)} uploaded file(s)."
            )

            # FAISS Similarity Search with optional ride filter
            raw_matches = db_manager.search_similar_faces(
                query_faces=raw_query_faces,
                threshold=threshold,
                top_k=top_k,
                ride_filter=ride_filter
            )

            # Filter out missing/stale files and self-heal relocated paths to guarantee 100% valid thumbnails
            matched_photos = []
            for item in raw_matches:
                p_res = db_manager.resolve_photo(item["photo_id"], RAW_DIR)
                if not p_res:
                    logger.warning(f"Omitting stale match #{item['photo_id']} ({item.get('file_name')} not found on disk)")
                    continue
                _, actual_path = p_res
                item["file_path"] = actual_path
                matched_photos.append(item)

            elapsed_ms = round((time.time() - start_time) * 1000, 2)

            # Determine customer folder name (Option 2: [Nama_Tamu]_[4_Digit_NoHP])
            clean_guest_name = ""
            if guest_name and guest_name.strip():
                clean_guest_name = re.sub(r'[^a-zA-Z0-9]', '_', guest_name.strip())
                clean_guest_name = re.sub(r'_+', '_', clean_guest_name).strip('_')

            phone_suffix = ""
            if guest_phone and guest_phone.strip():
                digits = re.sub(r'\D', '', guest_phone.strip())
                if len(digits) >= 4:
                    phone_suffix = digits[-4:]
                elif digits:
                    phone_suffix = digits

            if clean_guest_name:
                if not phone_suffix:
                    phone_suffix = uuid.uuid4().hex[:4].upper()
                base_folder_name = f"{clean_guest_name}_{phone_suffix}"
            else:
                customer_uuid = uuid.uuid4().hex[:10].upper()
                base_folder_name = f"Customer_{customer_uuid}"

            # Guarantee unique result directory name
            customer_folder_name = base_folder_name
            counter = 2
            while os.path.exists(os.path.join(RESULTS_DIR, customer_folder_name)):
                customer_folder_name = f"{base_folder_name}_{counter}"
                counter += 1

            customer_dir = os.path.join(RESULTS_DIR, customer_folder_name)

            # Create destination virtual folder for customer
            os.makedirs(customer_dir, exist_ok=True)

            created_symlinks = []
            used_filenames = set()
            together_count = 0
            single_count = 0

            for item in matched_photos:
                if item.get("matched_member_count", 1) >= 2:
                    together_count += 1
                else:
                    single_count += 1

                src_path = item["file_path"]
                if not os.path.exists(src_path):
                    continue

                base_name = os.path.basename(src_path)
                # Prevent collisions if identical filenames exist across different ride subfolders
                target_name = base_name
                counter = 1
                name_root, ext = os.path.splitext(base_name)
                while target_name in used_filenames:
                    target_name = f"{name_root}_{counter}{ext}"
                    counter += 1
                used_filenames.add(target_name)

                link_dest = os.path.join(customer_dir, target_name)

                # Calculate relative path from customer_dir to src_path
                # so symlinks resolve identically inside Docker (/app/data/...)
                # and on the host Samba network share (/opt/SistemPhoto/data/...)!
                try:
                    rel_src = os.path.relpath(src_path, start=customer_dir)
                except Exception:
                    rel_src = src_path

                symlink_ok = False
                try:
                    if os.path.lexists(link_dest):
                        os.remove(link_dest)
                    os.symlink(rel_src, link_dest)
                    if os.path.exists(link_dest):
                        created_symlinks.append(link_dest)
                        symlink_ok = True
                    else:
                        if os.path.lexists(link_dest):
                            os.remove(link_dest)
                except OSError as e_sym:
                    logger.warning(f"Could not create symlink {link_dest} -> {rel_src}: {e_sym}")

                # Fallback to physical copy if symlink failed
                if not symlink_ok:
                    try:
                        shutil.copy2(src_path, link_dest)
                        created_symlinks.append(link_dest)
                    except OSError as e_cp:
                        logger.error(f"Could not copy photo {src_path} -> {link_dest}: {e_cp}")

                # Attach preview URL pointing specifically to the matched customer faces
                matched_vids = [str(f["vector_id"]) for f in item.get("matched_faces", [])]
                matched_vids_param = ",".join(matched_vids)
                best_vid = item.get("best_vector_id")

                if matched_vids_param:
                    item["preview_url"] = f"/api/photos/{item['photo_id']}/preview?matched_vid={matched_vids_param}"
                else:
                    item["preview_url"] = f"/api/photos/{item['photo_id']}/preview"

                if best_vid:
                    item["crop_url"] = f"/api/photos/{item['photo_id']}/crop?matched_vid={best_vid}"
                else:
                    item["crop_url"] = f"/api/photos/{item['photo_id']}/crop"

            # Write audit summary into the customer result folder
            summary_data = {
                "customer_id": customer_folder_name,
                "guest_name": (guest_name or "").strip(),
                "guest_phone": (guest_phone or "").strip(),
                "search_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "similarity_threshold": float(threshold),
                "query_time_ms": float(elapsed_ms),
                "total_matches": len(matched_photos),
                "total_members": total_members,
                "together_matches": together_count,
                "single_matches": single_count,
                "files": [
                    {
                        "file_name": p["file_name"],
                        "original_path": p["file_path"],
                        "max_similarity_score": float(p["max_score"]),
                        "matched_member_count": int(p.get("matched_member_count", 1))
                    }
                    for p in matched_photos
                ]
            }
            try:
                with open(os.path.join(customer_dir, "search_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(summary_data, f, indent=2)
            except Exception as e_sum:
                logger.warning(f"Could not write search_summary.json: {e_sum}")

            # Format Samba path dynamically using the accessed server IP / hostname
            host = request.headers.get("host", "").split(":")[0] or "10.12.2.150"
            samba_path = format_samba_unc(SAMBA_PREFIX, customer_folder_name, fallback_host=host)

            for item in matched_photos:
                item["samba_file_path"] = format_samba_unc(
                    SAMBA_PREFIX, f"{customer_folder_name}\\{item['file_name']}", fallback_host=host
                )

            # Record guest visit and calculate stay duration
            visit_record = None
            if (guest_name and guest_name.strip()) or guest_crm_id:
                try:
                    # Auto-resolve check-in time from Cloud CRM if not provided by client
                    if not guest_registered_at and ((guest_phone and guest_phone.strip()) or (guest_name and guest_name.strip())):
                        try:
                            lookup_kw = (guest_phone or guest_name).strip()
                            crm_res = await search_crm_guests(q=lookup_kw, limit=10)
                            if crm_res and crm_res.get("success") and crm_res.get("guests"):
                                matched_crm = None
                                if guest_phone:
                                    clean_ph = re.sub(r'\D', '', guest_phone)
                                    for g in crm_res["guests"]:
                                        g_ph = re.sub(r'\D', '', g.get("phone") or "")
                                        if clean_ph and (clean_ph in g_ph or g_ph in clean_ph):
                                            matched_crm = g
                                            break
                                if not matched_crm and guest_name:
                                    t_name = guest_name.strip().lower()
                                    for g in crm_res["guests"]:
                                        if (g.get("name") or "").strip().lower() == t_name:
                                            matched_crm = g
                                            break
                                if matched_crm:
                                    guest_crm_id = matched_crm.get("id") or guest_crm_id
                                    guest_registered_at = matched_crm.get("submitted_at")
                                    logger.info(f"Auto-matched CRM guest registration: {matched_crm.get('name')} -> {guest_registered_at}")
                        except Exception as e_lookup:
                            logger.warning(f"Could not auto-lookup guest from CRM: {e_lookup}")

                    visit_record = db_manager.record_guest_visit(
                        crm_guest_id=guest_crm_id,
                        guest_name=(guest_name or "Tamu").strip(),
                        guest_phone=guest_phone,
                        registered_at=guest_registered_at,
                        photo_count=len(matched_photos),
                        customer_folder=customer_folder_name
                    )
                    if guest_crm_id:
                        threading.Thread(
                            target=_sync_visit_to_crm,
                            args=(visit_record,),
                            daemon=True
                        ).start()
                except Exception as e_visit:
                    logger.error(f"Failed to record guest visit: {e_visit}")

            # Pre-warm thumbnails in background for instant browser display
            def _prewarm_matched_thumbnails(photos_to_warm):
                for p_item in photos_to_warm[:16]:
                    pid = p_item["photo_id"]
                    b_path = os.path.join(THUMBNAILS_DIR, f"base_{pid}.jpg")
                    m_path = os.path.join(THUMBNAILS_DIR, f"base_{pid}.json")
                    if not os.path.exists(b_path) or not os.path.exists(m_path):
                        f_src = p_item.get("file_path")
                        if f_src and os.path.exists(f_src):
                            try:
                                with Image.open(f_src) as meta_raw:
                                    meta_trans = ImageOps.exif_transpose(meta_raw)
                                    orig_w, orig_h = meta_trans.size
                                with Image.open(f_src) as raw_img:
                                    raw_img.draft("RGB", (800, 800))
                                    img = ImageOps.exif_transpose(raw_img)
                                    if img.mode != "RGB":
                                        img = img.convert("RGB")
                                    img.thumbnail((800, 800), Image.Resampling.BILINEAR)
                                    img.save(b_path, format="JPEG", quality=85)
                                    with open(m_path, "w", encoding="utf-8") as mf:
                                        json.dump({"w": orig_w, "h": orig_h, "tw": img.width, "th": img.height}, mf)
                            except Exception:
                                pass

            if matched_photos:
                threading.Thread(target=_prewarm_matched_thumbnails, args=(matched_photos,), daemon=True).start()

            return {
                "success": True,
                "customer_id": customer_folder_name,
                "guest_name": (guest_name or "").strip(),
                "guest_phone": (guest_phone or "").strip(),
                "result_dir": customer_dir,
                "samba_path": samba_path,
                "match_count": len(matched_photos),
                "query_time_ms": elapsed_ms,
                "threshold": threshold,
                "total_members": total_members,
                "together_count": together_count,
                "single_count": single_count,
                "matches": matched_photos,
                "visit": visit_record
            }
        finally:
            face_engine.end_search_priority()

    except Exception as e:
        logger.error(f"Search request failed: {e}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Pemrosesan Pencarian Gagal",
                "detail": str(e),
                "suggestion": "Silakan coba kembali atau periksa koneksi data kamera."
            }
        )


@app.get("/api/photos/{photo_id}/preview")
def preview_photo(
    photo_id: int,
    highlight: bool = Query(True, description="Draw bounding boxes on detected faces"),
    matched_vid: Optional[str] = Query(None, description="Vector ID(s) of the specifically matched customer face(s)")
):
    """
    Serve a lightweight thumbnail preview of the photo with high-visibility bounding box
    specifically on the matched customer's face, and subtle indicator on other passengers.
    Accelerated with multi-tier disk caching, fast DCT downsampling, and threadpool concurrency.
    """
    # 1. Parse matched vector IDs
    matched_vid_set = set()
    clean_vid_str = "all"
    if matched_vid is not None:
        parts = []
        for part in str(matched_vid).split(","):
            part = part.strip()
            if part.isdigit():
                matched_vid_set.add(int(part))
                parts.append(part)
        if parts:
            clean_vid_str = "_".join(sorted(parts))

    # 2. Check if the specifically rendered & highlighted thumbnail is already cached
    hl_flag = "1" if highlight else "0"
    cache_filename = f"prev_{photo_id}_hl{hl_flag}_vids_{clean_vid_str}.jpg"
    cache_path = os.path.join(THUMBNAILS_DIR, cache_filename)

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 500:
        return FileResponse(
            cache_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400, stale-while-revalidate=604800"}
        )

    # 3. Resolve original photo file
    photo_res = db_manager.resolve_photo(photo_id, RAW_DIR)
    if not photo_res:
        raise HTTPException(status_code=404, detail="Photo file not found on disk.")
    p_row, file_path = photo_res

    # 4. Check or generate clean base 800px thumbnail
    base_thumb_path = os.path.join(THUMBNAILS_DIR, f"base_{photo_id}.jpg")
    meta_path = os.path.join(THUMBNAILS_DIR, f"base_{photo_id}.json")

    base_img = None
    orig_w, orig_h = 0, 0

    if os.path.exists(base_thumb_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
                orig_w = meta.get("w", 0)
                orig_h = meta.get("h", 0)
            if orig_w > 0 and orig_h > 0:
                base_img = Image.open(base_thumb_path)
                base_img.load()
        except Exception as e_meta:
            logger.debug(f"Failed loading thumbnail cache for photo #{photo_id}: {e_meta}")
            base_img = None

    if base_img is None:
        try:
            with Image.open(file_path) as meta_raw:
                meta_trans = ImageOps.exif_transpose(meta_raw)
                orig_w, orig_h = meta_trans.size

            with Image.open(file_path) as raw_img:
                raw_img.draft("RGB", (800, 800))
                img = ImageOps.exif_transpose(raw_img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.thumbnail((800, 800), Image.Resampling.BILINEAR)
                img.save(base_thumb_path, format="JPEG", quality=85)

                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump({"w": orig_w, "h": orig_h, "tw": img.width, "th": img.height}, mf)

                base_img = img
        except Exception as e_gen:
            logger.error(f"Error creating base thumbnail for photo #{photo_id}: {e_gen}")
            return FileResponse(file_path, media_type="image/jpeg")

    # 5. If no highlight requested, return base thumbnail directly
    if not highlight:
        return FileResponse(
            base_thumb_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400, stale-while-revalidate=604800"}
        )

    # 6. Retrieve detected faces from DB
    with db_manager._get_connection() as conn:
        faces_rows = conn.execute(
            "SELECT vector_id, bbox_top, bbox_right, bbox_bottom, bbox_left FROM faces WHERE photo_id = ?",
            (photo_id,)
        ).fetchall()

    if not faces_rows:
        return FileResponse(
            base_thumb_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400, stale-while-revalidate=604800"}
        )

    # 7. Draw boxes onto copy of base thumbnail
    render_img = base_img.copy()
    safe_orig_w = max(1, orig_w)
    safe_orig_h = max(1, orig_h)
    scale_x = render_img.width / float(safe_orig_w)
    scale_y = render_img.height / float(safe_orig_h)

    draw = ImageDraw.Draw(render_img)
    for f in faces_rows:
        top = int(f["bbox_top"] * scale_y)
        right = int(f["bbox_right"] * scale_x)
        bottom = int(f["bbox_bottom"] * scale_y)
        left = int(f["bbox_left"] * scale_x)
        vid = f["vector_id"]

        is_customer = (vid in matched_vid_set) if matched_vid_set else (matched_vid is None)

        if is_customer or matched_vid is None:
            # BOLD VIBRANT GREEN BOX FOR MATCHED GUEST
            for width_offset in range(3):
                draw.rectangle(
                    [left - width_offset, top - width_offset, right + width_offset, bottom + width_offset],
                    outline=(16, 185, 129)  # Emerald-500
                )
            # "TAMU" badge banner above box
            badge_h = 16
            badge_w = 46
            draw.rectangle([left, max(0, top - badge_h), left + badge_w, top], fill=(16, 185, 129))
            try:
                draw.text((left + 4, max(0, top - badge_h) + 1), "TAMU", fill=(255, 255, 255))
            except Exception:
                pass
        else:
            # SUBTLE THIN SLATE BOX FOR OTHER PEOPLE IN RIDE
            draw.rectangle(
                [left, top, right, bottom],
                outline=(148, 163, 184)  # Slate-400
            )

    # 8. Save rendered thumbnail to cache and return FileResponse
    try:
        render_img.save(cache_path, format="JPEG", quality=85)
        return FileResponse(
            cache_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400, stale-while-revalidate=604800"}
        )
    except Exception as e_save:
        logger.warning(f"Could not persist cache {cache_path}: {e_save}")
        output_buf = io.BytesIO()
        render_img.save(output_buf, format="JPEG", quality=85)
        output_buf.seek(0)
        return StreamingResponse(
            output_buf,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"}
        )


@app.get("/api/photos/{photo_id}/crop")
def crop_matched_face(
    photo_id: int,
    matched_vid: Optional[int] = Query(None, description="Vector ID of the specifically matched customer face")
):
    """
    Serve a crisp, high-resolution zoomed close-up crop of the customer's face from the original photo.
    Enables operators and guests to instantly verify distant or crowded background faces with 100% clarity.
    Uses disk caching to return instantly (< 1ms).
    """
    vid_str = str(matched_vid) if matched_vid is not None else "primary"
    crop_cache_filename = f"crop_{photo_id}_{vid_str}.jpg"
    crop_cache_path = os.path.join(THUMBNAILS_DIR, crop_cache_filename)

    if os.path.exists(crop_cache_path) and os.path.getsize(crop_cache_path) > 500:
        return FileResponse(
            crop_cache_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400, stale-while-revalidate=604800"}
        )

    photo_res = db_manager.resolve_photo(photo_id, RAW_DIR)
    if not photo_res:
        raise HTTPException(status_code=404, detail="Photo file not found on disk.")
    p_row, file_path = photo_res

    with db_manager._get_connection() as conn:
        faces_rows = conn.execute(
            "SELECT vector_id, bbox_top, bbox_right, bbox_bottom, bbox_left FROM faces WHERE photo_id = ?",
            (photo_id,)
        ).fetchall()

    if not faces_rows:
        return FileResponse(file_path, media_type="image/jpeg")

    # Find the target face
    target_face = None
    if matched_vid is not None:
        for f in faces_rows:
            if f["vector_id"] == matched_vid:
                target_face = f
                break
    if target_face is None:
        target_face = faces_rows[0]

    try:
        raw_img = Image.open(file_path)
        img = ImageOps.exif_transpose(raw_img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        orig_w, orig_h = img.size
        top = target_face["bbox_top"]
        right = target_face["bbox_right"]
        bottom = target_face["bbox_bottom"]
        left = target_face["bbox_left"]

        face_w = right - left
        face_h = bottom - top

        # Calculate generous context padding (60% margin around face to include hair, head, and upper chest)
        pad_x = int(face_w * 0.65)
        pad_y_top = int(face_h * 0.55)
        pad_y_bottom = int(face_h * 0.75)

        crop_left = max(0, left - pad_x)
        crop_top = max(0, top - pad_y_top)
        crop_right = min(orig_w, right + pad_x)
        crop_bottom = min(orig_h, bottom + pad_y_bottom)

        cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))

        # Make crop square by padding or centered expanding for neat UI avatar display
        cw, ch = cropped.size
        target_dim = max(cw, ch)
        square_crop = Image.new("RGB", (target_dim, target_dim), (24, 34, 30))
        offset_x = (target_dim - cw) // 2
        offset_y = (target_dim - ch) // 2
        square_crop.paste(cropped, (offset_x, offset_y))

        # Resize to sharp 400x400 with fast BILINEAR filter
        square_crop = square_crop.resize((400, 400), Image.Resampling.BILINEAR)

        # Draw a stylish subtle emerald border & badge
        draw = ImageDraw.Draw(square_crop)
        for b_w in range(3):
            draw.rectangle([b_w, b_w, 399 - b_w, 399 - b_w], outline=(16, 185, 129))

        # Badge "ZOOM TAMU"
        badge_w, badge_h = 92, 22
        draw.rectangle([8, 8, 8 + badge_w, 8 + badge_h], fill=(16, 185, 129))
        try:
            draw.text((14, 12), "ZOOM TAMU", fill=(255, 255, 255))
        except Exception:
            pass

        # Save to disk cache
        try:
            square_crop.save(crop_cache_path, format="JPEG", quality=88)
            return FileResponse(
                crop_cache_path,
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=86400, stale-while-revalidate=604800"}
            )
        except Exception:
            output_buf = io.BytesIO()
            square_crop.save(output_buf, format="JPEG", quality=88)
            output_buf.seek(0)
            return StreamingResponse(
                output_buf,
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=86400"}
            )

    except Exception as e:
        logger.error(f"Error generating face crop for photo {photo_id}: {e}")
        return FileResponse(file_path, media_type="image/jpeg")


@app.get("/api/results/{customer_id}/download-zip")
async def download_results_zip(customer_id: str):
    """
    Package all matched raw photos into a single zip file for direct browser download by front-desk staff.
    Streams via disk tempfile to avoid high RAM memory consumption and prevent OOM issues.
    """
    clean_id = os.path.basename(customer_id.strip())
    customer_dir = os.path.abspath(os.path.join(RESULTS_DIR, clean_id))
    results_dir_abs = os.path.abspath(RESULTS_DIR)

    try:
        if os.path.commonpath([results_dir_abs, customer_dir]) != results_dir_abs:
            raise HTTPException(status_code=400, detail="Invalid customer ID.")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid customer ID.")

    if not os.path.exists(customer_dir) or not os.path.isdir(customer_dir):
        raise HTTPException(status_code=404, detail="Customer result folder not found.")

    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    temp_zip_path = temp_zip.name
    temp_zip.close()

    try:
        with zipfile.ZipFile(temp_zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for item in os.listdir(customer_dir):
                item_path = os.path.join(customer_dir, item)
                real_path = os.path.realpath(item_path)
                if os.path.isfile(real_path):
                    zip_file.write(real_path, arcname=item)

        def iterfile():
            try:
                with open(temp_zip_path, "rb") as f:
                    while True:
                        chunk = f.read(64 * 1024)
                        if not chunk:
                            break
                        yield chunk
            finally:
                if os.path.exists(temp_zip_path):
                    try:
                        os.remove(temp_zip_path)
                    except Exception:
                        pass

        safe_filename = re.sub(r'[^a-zA-Z0-9_\-]', '_', clean_id)
        return StreamingResponse(
            iterfile(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{safe_filename}_Photos.zip"'}
        )
    except Exception as e:
        if os.path.exists(temp_zip_path):
            try:
                os.remove(temp_zip_path)
            except Exception:
                pass
        logger.error(f"Error packaging zip for {customer_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal mengemas file foto ZIP: {e}")


@app.get("/api/setup-explorer-protocol")
async def download_explorer_protocol_reg():
    """
    Serve a Windows Registry script (.reg) that registers the custom protocol 'lodge:'.
    When installed (by double-clicking once on the client PC without admin rights),
    clicking 'Buka Folder di File Explorer' in Chrome/Edge will directly launch Windows File Explorer
    to the customer's UNC network folder.
    Protected against command injection via argument parameterization and strict character whitelisting.
    """
    reg_path = os.path.join(STATIC_DIR, "Aktifkan_Buka_File_Explorer.reg")
    if os.path.exists(reg_path):
        return FileResponse(
            reg_path,
            media_type="application/octet-stream",
            filename="Aktifkan_Buka_File_Explorer.reg"
        )
    reg_content = (
        "Windows Registry Editor Version 5.00\r\n\r\n"
        "[HKEY_CURRENT_USER\\Software\\Classes\\lodge]\r\n"
        '@="URL:The Lodge Photo Protocol"\r\n'
        '"URL Protocol"=""\r\n\r\n'
        "[HKEY_CURRENT_USER\\Software\\Classes\\lodge\\shell]\r\n\r\n"
        "[HKEY_CURRENT_USER\\Software\\Classes\\lodge\\shell\\open]\r\n\r\n"
        "[HKEY_CURRENT_USER\\Software\\Classes\\lodge\\shell\\open\\command]\r\n"
        '@="powershell.exe -WindowStyle Hidden -NoProfile -ExecutionPolicy Bypass -Command \\"& { param($p); $u = [System.Uri]::UnescapeDataString($p) -replace \'^lodge:/?/?\',\'\' -replace \'/\',\'\\\\\\\\\'; if ($u -match \'^[a-zA-Z0-9_\\\\\\\\\\-\\.\\ :]+$\') { & explorer.exe (\'\\\\\\\\\' + $u.TrimStart(\'\\\\\\\\\')) } }\\" \\"%1\\""\r\n'
    )
    return StreamingResponse(
        io.BytesIO(reg_content.encode("utf-8")),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": 'attachment; filename="Aktifkan_Buka_File_Explorer.reg"'
        }
    )

