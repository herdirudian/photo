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
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from PIL import Image, ImageDraw

from database import DatabaseManager, FaceEngine
from indexer import PhotoIndexer

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PhotoRetrieval.API")

# Environment paths
RAW_DIR = os.getenv("RAW_DIR", os.path.abspath("./data/raw"))
RESULTS_DIR = os.getenv("RESULTS_DIR", os.path.abspath("./data/results"))
DB_DIR = os.getenv("DB_DIR", os.path.abspath("./data/db"))
SAMBA_PREFIX = os.getenv("SAMBA_NETWORK_PREFIX", r"\\samba-server\park-photos\results")
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.82"))
FACE_MODEL = os.getenv("FACE_DETECTION_MODEL", "hog")

# Ensure required directories exist
for directory in [RAW_DIR, RESULTS_DIR, DB_DIR]:
    os.makedirs(directory, exist_ok=True)

# Singletons
db_manager = DatabaseManager(db_dir=DB_DIR)
face_engine = FaceEngine(model=FACE_MODEL)
indexer = PhotoIndexer(raw_dir=RAW_DIR, db_manager=db_manager, face_engine=face_engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start watchdog folder monitoring on application launch, and stop on shutdown."""
    logger.info("Initializing Theme Park Photo Retrieval Engine...")
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


def get_logo_file_path() -> Optional[str]:
    candidates = [
        os.path.join(IMG_DIR, "logotlm.png"),
        os.path.join(BASE_DIR, "img", "logotlm.png"),
        "/app/img/logotlm.png",
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


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    stats = db_manager.get_stats()
    context = {
        "request": request,
        "stats": stats,
        "default_threshold": DEFAULT_THRESHOLD,
        "samba_prefix": SAMBA_PREFIX,
        "raw_dir": RAW_DIR,
        "results_dir": RESULTS_DIR,
        "logo_base64": get_logo_base64()
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


@app.get("/api/health")
async def health_check():
    """System health check endpoint for Docker container status."""
    stats = db_manager.get_stats()
    return {
        "status": "online",
        "faiss_ready": stats["faiss_ready"],
        "face_rec_ready": stats["face_rec_ready"],
        "indexed_photos": stats["total_photos"],
        "indexed_faces": stats["total_faces"],
        "timestamp": time.time()
    }


@app.get("/api/stats")
async def get_system_stats():
    """Return real-time database and indexing metrics."""
    stats = db_manager.get_stats()
    stats["raw_dir"] = RAW_DIR
    stats["results_dir"] = RESULTS_DIR
    stats["samba_prefix"] = SAMBA_PREFIX
    return stats


@app.post("/api/reindex")
async def trigger_reindex():
    """Trigger manual re-scan of raw photos directory."""
    try:
        indexer.scan_existing_files()
        return {"success": True, "message": "Raw photo scan completed successfully."}
    except Exception as e:
        logger.error(f"Error during re-index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search")
async def search_guest_photos(
    request: Request,
    file: UploadFile = File(...),
    threshold: float = Form(DEFAULT_THRESHOLD),
    top_k: int = Form(50)
):
    """
    Search endpoint:
    1. Reads reference photo of guest.
    2. Extracts 128-d face vector.
    3. Searches FAISS for matching faces above similarity threshold.
    4. Creates virtual folder /app/data/results/Customer_{UUID} with Linux symlinks.
    5. Returns matches and network folder path.
    """
    start_time = time.time()

    # Validate image format
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    try:
        image_bytes = await file.read()
        extracted_faces = face_engine.extract_faces_from_bytes(image_bytes)
    except Exception as e:
        logger.error(f"Failed to process reference image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")

    if not extracted_faces:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": "No face detected in reference photo.",
                "suggestion": "Please capture a clear, well-lit photo of the guest's face without heavy obstruction."
            }
        )

    # If multiple faces detected in reference photo, select the largest one (most prominent)
    if len(extracted_faces) > 1:
        extracted_faces.sort(
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][1] - x["bbox"][3]),
            reverse=True
        )
        logger.info(f"Multiple faces detected in reference image ({len(extracted_faces)}). Selected largest face.")

    query_face = extracted_faces[0]
    query_encoding = query_face["encoding"]

    # FAISS Similarity Search
    matched_photos = db_manager.search_similar_faces(
        query_encoding=query_encoding,
        threshold=threshold,
        top_k=top_k
    )

    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    customer_uuid = uuid.uuid4().hex[:10].upper()
    customer_folder_name = f"Customer_{customer_uuid}"
    customer_dir = os.path.join(RESULTS_DIR, customer_folder_name)

    # Create destination virtual folder for customer
    os.makedirs(customer_dir, exist_ok=True)

    created_symlinks = []
    used_filenames = set()

    for item in matched_photos:
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

        # Generate native file entry (prefer hardlink: 0 bytes disk overhead, native file for Windows Samba Explorer)
        try:
            if os.path.exists(link_dest) or os.path.islink(link_dest):
                os.remove(link_dest)
            os.link(src_path, link_dest)
            created_symlinks.append(link_dest)
        except OSError:
            # Fallback to direct file copy so Windows Explorer can always copy and preview
            try:
                shutil.copy2(src_path, link_dest)
                created_symlinks.append(link_dest)
            except OSError:
                try:
                    rel_src = os.path.relpath(src_path, customer_dir)
                    os.symlink(rel_src, link_dest)
                    created_symlinks.append(link_dest)
                except OSError:
                    pass

        # Attach preview URL pointing specifically to the matched customer face
        best_vid = item.get("best_vector_id")
        if best_vid:
            item["preview_url"] = f"/api/photos/{item['photo_id']}/preview?matched_vid={best_vid}"
        else:
            item["preview_url"] = f"/api/photos/{item['photo_id']}/preview"

    # Write audit summary into the customer result folder
    summary_data = {
        "customer_id": customer_folder_name,
        "search_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "similarity_threshold": threshold,
        "query_time_ms": elapsed_ms,
        "total_matches": len(matched_photos),
        "files": [
            {
                "file_name": p["file_name"],
                "original_path": p["file_path"],
                "max_similarity_score": p["max_score"]
            }
            for p in matched_photos
        ]
    }
    with open(os.path.join(customer_dir, "search_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Format Samba path dynamically using the accessed server IP / hostname
    host = request.headers.get("host", "").split(":")[0] or "192.168.100.90"
    samba_prefix = SAMBA_PREFIX
    if not samba_prefix or samba_prefix.startswith("/mnt") or "samba-server" in samba_prefix or not samba_prefix.startswith(r"\\"):
        samba_path = rf"\\{host}\park-photos\results\{customer_folder_name}"
    else:
        samba_path = os.path.join(samba_prefix, customer_folder_name).replace("/", "\\")
        if not samba_path.startswith(r"\\"):
            samba_path = rf"\\{host}\park-photos\results\{customer_folder_name}"

    return {
        "success": True,
        "customer_id": customer_folder_name,
        "result_dir": customer_dir,
        "samba_path": samba_path,
        "match_count": len(matched_photos),
        "query_time_ms": elapsed_ms,
        "threshold": threshold,
        "matches": matched_photos
    }


@app.get("/api/photos/{photo_id}/preview")
async def preview_photo(
    photo_id: int,
    highlight: bool = Query(True, description="Draw bounding boxes on detected faces"),
    matched_vid: Optional[int] = Query(None, description="Vector ID of the specifically matched customer face")
):
    """
    Serve a lightweight thumbnail preview of the photo with high-visibility bounding box
    specifically on the matched customer's face, and subtle indicator on other passengers.
    """
    with db_manager._get_connection() as conn:
        p_row = conn.execute("SELECT file_path, file_name FROM photos WHERE id = ?", (photo_id,)).fetchone()
        if not p_row or not os.path.exists(p_row["file_path"]):
            raise HTTPException(status_code=404, detail="Photo file not found on disk.")

        faces_rows = conn.execute(
            "SELECT vector_id, bbox_top, bbox_right, bbox_bottom, bbox_left FROM faces WHERE photo_id = ?",
            (photo_id,)
        ).fetchall()

    try:
        img = Image.open(p_row["file_path"])
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize for fast thumbnail delivery (max dimension 800px)
        img.thumbnail((800, 800), Image.Resampling.LANCZOS)

        # Scale factor if image was resized
        orig_img = Image.open(p_row["file_path"])
        scale_x = img.width / orig_img.width
        scale_y = img.height / orig_img.height

        if highlight and faces_rows:
            draw = ImageDraw.Draw(img)
            for f in faces_rows:
                top = int(f["bbox_top"] * scale_y)
                right = int(f["bbox_right"] * scale_x)
                bottom = int(f["bbox_bottom"] * scale_y)
                left = int(f["bbox_left"] * scale_x)
                vid = f["vector_id"]

                is_customer = (matched_vid is not None and vid == matched_vid)

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
                    # SUBTLE THIN SLATE BOX FOR OTHER PEOPLE IN RIDE (Clearly shows they are recognized as other people)
                    draw.rectangle(
                        [left, top, right, bottom],
                        outline=(148, 163, 184)  # Slate-400
                    )

        output_buf = io.BytesIO()
        img.save(output_buf, format="JPEG", quality=85)
        output_buf.seek(0)
        return StreamingResponse(output_buf, media_type="image/jpeg")

    except Exception as e:
        logger.error(f"Error generating preview for photo {photo_id}: {e}")
        return FileResponse(p_row["file_path"], media_type="image/jpeg")


@app.get("/api/results/{customer_id}/download-zip")
async def download_results_zip(customer_id: str):
    """
    Package all matched raw photos into a single zip file for direct browser download by front-desk staff.
    """
    customer_dir = os.path.join(RESULTS_DIR, customer_id)
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
