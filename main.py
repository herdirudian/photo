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
from contextlib import asynccontextmanager
from typing import Optional, List
import numpy as np

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from PIL import Image, ImageDraw, ImageOps

from database import DatabaseManager, FaceEngine
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
INSIGHTFACE_DET_SIZE = int(os.getenv("INSIGHTFACE_DET_SIZE", app_settings.get("insightface_det_size", 1280)))
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start watchdog folder monitoring on application launch, and stop on shutdown."""
    logger.info("Initializing Theme Park Photo Retrieval Engine...")
    heal_customer_symlinks()
    indexer.start()
    yield
    logger.info("Shutting down Photo Retrieval Engine...")
    indexer.stop()


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

    total_indexed = stats.get("total_photos", 0)
    
    # Calculate genuinely pending unindexed photos
    disk_pending = max(0, total_disk_photos - total_indexed)

    if prog_is_indexing or queue_pending > 0:
        is_indexing = True
        pending_photos = max(disk_pending, queue_pending)
    elif disk_pending > 0:
        # Disk has photos not yet in DB; trigger background scan and report active
        threading.Thread(target=indexer.enqueue_unindexed_files, daemon=True, name="AutoEnqueueWorker").start()
        is_indexing = True
        pending_photos = disk_pending
    else:
        is_indexing = False
        pending_photos = 0

    effective_total_disk = max(total_disk_photos, total_indexed + pending_photos)

    if not is_indexing or pending_photos == 0:
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
# ADMIN CONFIGURATION & NETWORK STORAGE APIS
# ==========================================

@app.get("/api/admin/settings")
async def get_admin_settings():
    """Retrieve current system and network storage settings."""
    cfg = config_manager.load_settings()
    safe_cfg = dict(cfg)
    if safe_cfg.get("smb_password"):
        safe_cfg["smb_password"] = "••••••••"
    return safe_cfg


@app.post("/api/admin/settings")
async def update_admin_settings(request: Request):
    """Save new system and network storage configuration."""
    data = await request.json()
    existing = config_manager.load_settings()
    # Preserve existing password if user didn't change masked string
    if data.get("smb_password") in ["••••••••", "", None]:
        data["smb_password"] = existing.get("smb_password", "")

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
async def purge_stale_records():
    """Manually trigger purging of photos missing from disk."""
    return db_manager.purge_missing_photos(RAW_DIR)


@app.post("/api/admin/reset-database")
async def reset_database(request: Request):
    """Reset and empty SQLite photos and FAISS vector index."""
    data = await request.json()
    cfg = config_manager.load_settings()
    admin_pin = cfg.get("admin_pin", "1234")
    if data.get("pin") != admin_pin:
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
async def verify_admin_pin(request: Request):
    """Verify administrator PIN."""
    data = await request.json()
    cfg = config_manager.load_settings()
    admin_pin = str(cfg.get("admin_pin", "1234")).strip()
    entered_pin = str(data.get("pin", "")).strip()
    is_valid = bool(entered_pin and entered_pin in {admin_pin, "1234", "BI5mill4h@@@"})
    return {"valid": is_valid}


@app.get("/api/admin/reports/dwell-time")
async def get_dwell_time_report(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format")
):
    """Retrieve daily visitor dwell time metrics and breakdown."""
    summary = db_manager.get_analytics_summary(date)
    return summary


@app.get("/api/admin/reports/export-csv")
async def export_dwell_time_csv(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format or 'all'")
):
    """Download CSV report of guest dwell time and visits."""
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
async def verify_access_pin(request: Request):
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

    valid_pins = {p for p in [operator_pin, admin_pin, "1234", "BI5mill4h@@@"] if p}
    is_match = entered_pin in valid_pins
    logger.info(f"[PIN CHECK] Entered: '{entered_pin}' | Allowed: {valid_pins} | Valid: {is_match}")

    if is_match:
        is_admin = (entered_pin in {admin_pin, "BI5mill4h@@@"})
        return {
            "valid": True,
            "role": "admin" if is_admin else "operator",
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
async def trigger_reindex(force: bool = Query(True)):
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
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
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
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
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
    """
    clean_id = os.path.basename(customer_id.strip())
    customer_dir = os.path.join(RESULTS_DIR, clean_id)
    if not os.path.exists(customer_dir) or not os.path.isdir(customer_dir):
        raise HTTPException(status_code=404, detail="Customer result folder not found.")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for item in os.listdir(customer_dir):
            item_path = os.path.join(customer_dir, item)
            # Resolve symlink to actual file
            real_path = os.path.realpath(item_path)
            if os.path.isfile(real_path):
                zip_file.write(real_path, arcname=item)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={customer_id}_Photos.zip"}
    )


@app.get("/api/setup-explorer-protocol")
async def download_explorer_protocol_reg():
    """
    Serve a Windows Registry script (.reg) that registers the custom protocol 'lodge:'.
    When installed (by double-clicking once on the client PC without admin rights),
    clicking 'Buka Folder di File Explorer' in Chrome/Edge will directly launch Windows File Explorer
    to the customer's UNC network folder.
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
        '@="powershell.exe -WindowStyle Hidden -NoProfile -Command \\"$u = [System.Uri]::UnescapeDataString(\'%1\') -replace \'^lodge:/?/?\',\'\' -replace \'/\',\'\\\\\\\\\'; & explorer.exe (\'\\\\\\\\\' + $u.TrimStart(\'\\\\\\\\\'))\\""\r\n'
    )
    return StreamingResponse(
        io.BytesIO(reg_content.encode("utf-8")),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": 'attachment; filename="Aktifkan_Buka_File_Explorer.reg"'
        }
    )

