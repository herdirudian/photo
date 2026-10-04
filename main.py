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


def format_samba_unc(prefix: str, subpath: str = "", fallback_host: str = "192.168.100.95") -> str:
    """
    Produce a strictly valid Windows UNC network path (e.g. \\192.168.100.95\park-photos\results\Customer_XXXX).
    Guarantees exactly two leading backslashes and single backslashes in between.
    """
    import re
    if not prefix or prefix.startswith("/mnt") or "samba-server" in prefix or "192.168.100.90" in prefix:
        prefix = rf"\\{fallback_host}\park-photos\results"

    clean_p = re.sub(r"^[\\/]+", "", prefix.strip())
    clean_p = re.sub(r"[\\/]+", r"\\", clean_p)

    clean_sub = re.sub(r"^[\\/]+", "", subpath.strip()) if subpath else ""
    clean_sub = re.sub(r"[\\/]+", r"\\", clean_sub)

    if clean_sub:
        return rf"\\{clean_p}\{clean_sub}"
    return rf"\\{clean_p}"



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
    """Trigger manual re-scan of raw photos directory and purge stale records."""
    try:
        stats = indexer.scan_existing_files()
        return {
            "success": True,
            "message": (
                f"Pemindaian selesai: {stats.get('scanned', 0)} foto dicek, "
                f"{stats.get('newly_indexed', 0)} foto baru diindeks, "
                f"{stats.get('purged_photos', 0)} data usang dibersihkan."
            ),
            "stats": stats
        }
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
    4. Validates disk presence (self-healing relocated subfolders, filtering stale records).
    5. Creates virtual folder /app/data/results/Customer_{UUID} with Linux symlinks / copies.
    6. Returns matches and network folder path.
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
    raw_matches = db_manager.search_similar_faces(
        query_encoding=query_encoding,
        threshold=threshold,
        top_k=top_k
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
            item["crop_url"] = f"/api/photos/{item['photo_id']}/crop?matched_vid={best_vid}"
        else:
            item["preview_url"] = f"/api/photos/{item['photo_id']}/preview"
            item["crop_url"] = f"/api/photos/{item['photo_id']}/crop"

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
    host = request.headers.get("host", "").split(":")[0] or "192.168.100.95"
    samba_path = format_samba_unc(SAMBA_PREFIX, customer_folder_name, fallback_host=host)

    for item in matched_photos:
        item["samba_file_path"] = format_samba_unc(
            SAMBA_PREFIX, f"{customer_folder_name}\\{item['file_name']}", fallback_host=host
        )

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
    Self-heals if the photo was relocated into a ride subfolder.
    """
    photo_res = db_manager.resolve_photo(photo_id, RAW_DIR)
    if not photo_res:
        raise HTTPException(status_code=404, detail="Photo file not found on disk.")
    p_row, file_path = photo_res

    with db_manager._get_connection() as conn:
        faces_rows = conn.execute(
            "SELECT vector_id, bbox_top, bbox_right, bbox_bottom, bbox_left FROM faces WHERE photo_id = ?",
            (photo_id,)
        ).fetchall()

    try:
        img = Image.open(file_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize for fast thumbnail delivery (max dimension 800px)
        img.thumbnail((800, 800), Image.Resampling.LANCZOS)

        # Scale factor if image was resized
        orig_img = Image.open(file_path)
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
        return FileResponse(file_path, media_type="image/jpeg")


@app.get("/api/photos/{photo_id}/crop")
async def crop_matched_face(
    photo_id: int,
    matched_vid: Optional[int] = Query(None, description="Vector ID of the specifically matched customer face")
):
    """
    Serve a crisp, high-resolution zoomed close-up crop of the customer's face from the original photo.
    Enables operators and guests to instantly verify distant or crowded background faces with 100% clarity.
    Self-heals if the photo was relocated into a ride subfolder.
    """
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
        img = Image.open(file_path)
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

        # Resize to sharp 400x400
        square_crop = square_crop.resize((400, 400), Image.Resampling.LANCZOS)

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

        output_buf = io.BytesIO()
        square_crop.save(output_buf, format="JPEG", quality=92)
        output_buf.seek(0)
        return StreamingResponse(output_buf, media_type="image/jpeg")

    except Exception as e:
        logger.error(f"Error generating face crop for photo {photo_id}: {e}")
        return FileResponse(file_path, media_type="image/jpeg")


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

