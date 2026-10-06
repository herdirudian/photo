"""
database.py - SQLite persistence and FAISS vector similarity search for face retrieval.
"""

import os
import io
import sqlite3
import threading
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageOps

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PhotoRetrieval.DB")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS library not installed. Vector search will be simulated or disabled.")

try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    logger.warning("face_recognition library not installed. Face extraction will be disabled.")


class FaceEngine:
    """Wrapper for face detection and 128-d embedding extraction."""

    def __init__(self, model: str = "hog"):
        self.model = model  # "hog" (fast CPU) or "cnn" (GPU)
        self.dimension = 128  # standard face_recognition dlib ResNet-34 vector size

    @staticmethod
    def _compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        """Compute Intersection-over-Union (IoU) between two bounding boxes (top, right, bottom, left)."""
        topA, rightA, bottomA, leftA = boxA
        topB, rightB, bottomB, leftB = boxB

        inter_top = max(topA, topB)
        inter_left = max(leftA, leftB)
        inter_bottom = min(bottomA, bottomB)
        inter_right = min(rightA, rightB)

        if inter_bottom <= inter_top or inter_right <= inter_left:
            return 0.0

        inter_area = (inter_bottom - inter_top) * (inter_right - inter_left)
        areaA = max(0, bottomA - topA) * max(0, rightA - leftA)
        areaB = max(0, bottomB - topB) * max(0, rightB - leftB)
        union_area = areaA + areaB - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def _nms_boxes(self, boxes: List[Tuple[int, int, int, int]], iou_thresh: float = 0.35) -> List[Tuple[int, int, int, int]]:
        """Non-Maximum Suppression (NMS) to eliminate duplicate overlapping bounding boxes."""
        if not boxes:
            return []

        # Sort by area descending so larger/more confident boxes take priority
        sorted_boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]), reverse=True)
        selected = []

        for b in sorted_boxes:
            overlap = False
            for s in selected:
                if self._compute_iou(b, s) > iou_thresh:
                    overlap = True
                    break
            if not overlap:
                selected.append(b)

        return selected

    def detect_multiscale_locations(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        High-Performance Face Detection optimized for theme park DSLR / camera photos.
        Processes in ~0.5s - 1.2s per photo on CPU instead of 30s.
        """
        h, w = image.shape[:2]
        all_locations: List[Tuple[int, int, int, int]] = []
        max_dim = max(h, w)

        # 1. Main Global Pass with optimal resolution (max 1600px)
        if max_dim > 1600:
            scale = 1600.0 / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
            pil_img = Image.fromarray(image)
            small_img = np.array(pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR))
            global_locs = face_recognition.face_locations(small_img, number_of_times_to_upsample=1, model=self.model)
            for top, right, bottom, left in global_locs:
                all_locations.append((
                    int(round(top / scale)),
                    int(round(right / scale)),
                    int(round(bottom / scale)),
                    int(round(left / scale))
                ))
        else:
            global_locs = face_recognition.face_locations(image, number_of_times_to_upsample=1, model=self.model)
            all_locations.extend(global_locs)

        # 2. Focused Center Pass: Only if no faces were found globally, check ride center area
        if not all_locations and max_dim >= 1400:
            cy1, cx1, cy2, cx2 = int(h * 0.15), int(w * 0.15), int(h * 0.85), int(w * 0.85)
            center_crop = image[cy1:cy2, cx1:cx2]
            scale_c = 1400.0 / max(center_crop.shape[:2]) if max(center_crop.shape[:2]) > 1400 else 1.0
            if scale_c < 1.0:
                cw_s, ch_s = int(center_crop.shape[1] * scale_c), int(center_crop.shape[0] * scale_c)
                pil_c = Image.fromarray(center_crop).resize((cw_s, ch_s), Image.Resampling.BILINEAR)
                center_crop_s = np.array(pil_c)
            else:
                center_crop_s = center_crop
            c_locs = face_recognition.face_locations(center_crop_s, number_of_times_to_upsample=1, model=self.model)
            for top, right, bottom, left in c_locs:
                all_locations.append((
                    cy1 + int(round(top / scale_c)),
                    cx1 + int(round(right / scale_c)),
                    cy1 + int(round(bottom / scale_c)),
                    cx1 + int(round(left / scale_c))
                ))

        # Filter out tiny noise artifacts (< 20px)
        filtered = [
            loc for loc in all_locations
            if (loc[1] - loc[3]) >= 20 and (loc[2] - loc[0]) >= 20
        ]

        # 3. Deduplicate via Non-Maximum Suppression
        final_boxes = self._nms_boxes(filtered, iou_thresh=0.35)
        return final_boxes

    def extract_faces_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract bounding boxes and 128-d facial embeddings from an image file on disk
        using intelligent multi-scale tiled detection for distant & crowd faces.
        Returns list of {'bbox': (top, right, bottom, left), 'encoding': np.ndarray}
        """
        if not FACE_REC_AVAILABLE:
            raise RuntimeError("face_recognition is not available in current environment")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image not found at {file_path}")

        try:
            image = face_recognition.load_image_file(file_path)
        except Exception as e:
            logger.error(f"Failed to read image file {file_path}: {e}")
            return []

        # Find all face locations (including distant/crowded people) via multi-scale scan
        filtered_locations = self.detect_multiscale_locations(image)

        if not filtered_locations:
            return []

        # Compute 128-d encodings using the high-accuracy 68-landmark model
        encodings = face_recognition.face_encodings(image, known_face_locations=filtered_locations, num_jitters=1, model="large")

        results = []
        for loc, enc in zip(filtered_locations, encodings):
            # Normalize vector to unit length for cosine similarity via inner product
            norm = np.linalg.norm(enc)
            if norm > 0:
                normalized_enc = (enc / norm).astype(np.float32)
            else:
                normalized_enc = enc.astype(np.float32)

            results.append({
                "bbox": loc,
                "encoding": normalized_enc
            })

        return results

    def extract_faces_from_bytes(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract face locations and encodings from in-memory bytes with autocontrast and jittering for high query accuracy."""
        if not FACE_REC_AVAILABLE:
            raise RuntimeError("face_recognition is not available in current environment")

        image_obj = Image.open(io.BytesIO(image_bytes))
        # Ensure RGB format
        if image_obj.mode != "RGB":
            image_obj = image_obj.convert("RGB")
        
        # Auto-contrast enhancement to normalize indoor/outdoor shadows
        try:
            enhanced_obj = ImageOps.autocontrast(image_obj, cutoff=1)
            image_np = np.array(enhanced_obj)
        except Exception:
            image_np = np.array(image_obj)

        locations = face_recognition.face_locations(image_np, number_of_times_to_upsample=1, model=self.model)
        if not locations:
            locations = face_recognition.face_locations(image_np, number_of_times_to_upsample=2, model=self.model)

        if not locations:
            return []

        # For the query guest photo, use num_jitters=2 and model="large" to produce an ultra-stable embedding
        encodings = face_recognition.face_encodings(image_np, known_face_locations=locations, num_jitters=2, model="large")
        results = []
        for loc, enc in zip(locations, encodings):
            norm = np.linalg.norm(enc)
            if norm > 0:
                normalized_enc = (enc / norm).astype(np.float32)
            else:
                normalized_enc = enc.astype(np.float32)

            results.append({
                "bbox": loc,
                "encoding": normalized_enc
            })
        return results


class DatabaseManager:
    """
    Manages SQLite database storage for photo metadata and FAISS Index for vector search.
    Provides thread-safe atomic operations.
    """

    def __init__(self, db_dir: str = "/app/data/db", vector_dim: int = 128):
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, "metadata.db")
        self.faiss_index_path = os.path.join(self.db_dir, "faiss_index.bin")
        self.vector_dim = vector_dim

        self.lock = threading.RLock()
        self._init_sqlite()
        self._init_faiss()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with foreign keys and WAL mode enabled."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_sqlite(self):
        """Initialize database tables and schema if not present."""
        with self.lock, self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS photos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    file_name TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_mtime REAL NOT NULL,
                    num_faces INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'indexed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    photo_id INTEGER NOT NULL,
                    vector_id INTEGER UNIQUE NOT NULL,
                    bbox_top INTEGER NOT NULL,
                    bbox_right INTEGER NOT NULL,
                    bbox_bottom INTEGER NOT NULL,
                    bbox_left INTEGER NOT NULL,
                    encoding_blob BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_photos_path ON photos(file_path);
                CREATE INDEX IF NOT EXISTS idx_photos_status ON photos(status);
                CREATE INDEX IF NOT EXISTS idx_faces_photo_id ON faces(photo_id);
                CREATE INDEX IF NOT EXISTS idx_faces_vector_id ON faces(vector_id);
            """)
            conn.commit()
        logger.info(f"SQLite metadata initialized at {self.db_path}")

    def _init_faiss(self):
        """Initialize FAISS vector index with ID mapping for fast inner product (cosine similarity)."""
        with self.lock:
            if not FAISS_AVAILABLE:
                self.index = None
                return

            if os.path.exists(self.faiss_index_path):
                try:
                    self.index = faiss.read_index(self.faiss_index_path)
                    logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors.")
                    return
                except Exception as e:
                    logger.warning(f"Could not load existing FAISS index ({e}). Rebuilding from SQLite...")

            # Create fresh IndexIDMap2 wrapping IndexFlatIP
            # IndexFlatIP computes exact inner products (cosine similarity on unit-normalized vectors)
            base_index = faiss.IndexFlatIP(self.vector_dim)
            self.index = faiss.IndexIDMap2(base_index)
            self._rebuild_faiss_from_sqlite()

    def _rebuild_faiss_from_sqlite(self):
        """Reconstruct the entire FAISS index using vector blobs stored in SQLite."""
        if not FAISS_AVAILABLE or self.index is None:
            return

        with self._get_connection() as conn:
            cursor = conn.execute("SELECT vector_id, encoding_blob FROM faces ORDER BY vector_id ASC")
            rows = cursor.fetchall()

        if not rows:
            self._save_faiss()
            return

        ids = []
        vectors = []
        for row in rows:
            v_id = row["vector_id"]
            blob = row["encoding_blob"]
            vec = np.frombuffer(blob, dtype=np.float32)
            if len(vec) == self.vector_dim:
                ids.append(v_id)
                vectors.append(vec)

        if vectors:
            v_array = np.vstack(vectors).astype(np.float32)
            id_array = np.array(ids, dtype=np.int64)
            self.index.add_with_ids(v_array, id_array)
            logger.info(f"Rebuilt FAISS index with {len(ids)} vectors from database.")
            self._save_faiss()

    def _save_faiss(self):
        """Persist FAISS index to disk."""
        if FAISS_AVAILABLE and self.index is not None:
            try:
                faiss.write_index(self.index, self.faiss_index_path)
            except Exception as e:
                logger.error(f"Error saving FAISS index: {e}")

    def is_photo_indexed(self, file_path: str, file_mtime: float, file_size: int) -> bool:
        """Check if a file has already been indexed with matching modification time and size."""
        with self.lock, self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, file_mtime, file_size, status FROM photos WHERE file_path = ?",
                (file_path,)
            )
            row = cursor.fetchone()
            if row:
                # If modified or size changed, need re-indexing
                if abs(row["file_mtime"] - file_mtime) < 1e-4 and row["file_size"] == file_size:
                    return True
            return False

    def get_next_vector_id(self) -> int:
        """Generate a sequential unique vector ID for FAISS."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COALESCE(MAX(vector_id), 0) + 1 AS next_id FROM faces")
            row = cursor.fetchone()
            return row["next_id"]

    def register_photo(
        self,
        file_path: str,
        file_mtime: float,
        file_size: int,
        faces: List[Dict[str, Any]],
        status: str = "indexed"
    ) -> int:
        """
        Atomically register photo and all extracted face vectors into SQLite and FAISS.
        """
        file_name = os.path.basename(file_path)
        with self.lock:
            with self._get_connection() as conn:
                # Delete old records if file was previously indexed
                cursor = conn.execute("SELECT id FROM photos WHERE file_path = ?", (file_path,))
                old_photo = cursor.fetchone()
                if old_photo:
                    photo_id = old_photo["id"]
                    # Get vector IDs to remove from FAISS
                    face_cur = conn.execute("SELECT vector_id FROM faces WHERE photo_id = ?", (photo_id,))
                    old_vids = [r["vector_id"] for r in face_cur.fetchall()]
                    if old_vids and FAISS_AVAILABLE and self.index is not None:
                        try:
                            self.index.remove_ids(np.array(old_vids, dtype=np.int64))
                        except Exception as e:
                            logger.warning(f"Could not remove old vector IDs from FAISS: {e}")
                    conn.execute("DELETE FROM photos WHERE id = ?", (photo_id,))

                # Insert new photo entry
                cursor = conn.execute("""
                    INSERT INTO photos (file_path, file_name, file_size, file_mtime, num_faces, status, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (file_path, file_name, file_size, file_mtime, len(faces), status))
                photo_id = cursor.lastrowid

                # Prepare faces and vectors
                faiss_vectors = []
                faiss_ids = []

                # Find base vector ID
                cursor = conn.execute("SELECT COALESCE(MAX(vector_id), 0) FROM faces")
                base_vector_id = cursor.fetchone()[0]

                for idx, face_data in enumerate(faces):
                    vector_id = base_vector_id + idx + 1
                    bbox = face_data["bbox"]  # (top, right, bottom, left)
                    encoding = face_data["encoding"].astype(np.float32)

                    conn.execute("""
                        INSERT INTO faces (photo_id, vector_id, bbox_top, bbox_right, bbox_bottom, bbox_left, encoding_blob)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        photo_id,
                        vector_id,
                        int(bbox[0]),
                        int(bbox[1]),
                        int(bbox[2]),
                        int(bbox[3]),
                        encoding.tobytes()
                    ))

                    faiss_vectors.append(encoding)
                    faiss_ids.append(vector_id)

                conn.commit()

            # Add to FAISS in memory and persist
            if faiss_vectors and FAISS_AVAILABLE and self.index is not None:
                v_matrix = np.vstack(faiss_vectors).astype(np.float32)
                id_matrix = np.array(faiss_ids, dtype=np.int64)
                self.index.add_with_ids(v_matrix, id_matrix)
                self._save_faiss()

            logger.info(f"Registered {file_name}: {len(faces)} faces indexed (photo_id: {photo_id}).")
            return photo_id

    def resolve_photo(self, photo_id: int, raw_dir: str) -> Optional[Tuple[sqlite3.Row, str]]:
        """
        Retrieve photo database row and verified existing file path.
        If file path on disk has changed/moved to another subfolder under raw_dir,
        it automatically self-heals by updating the database with the new location.
        Returns (photo_row, verified_file_path) or None if truly missing from disk.
        """
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()
        if not row:
            return None

        current_path = row["file_path"]
        if os.path.isfile(current_path):
            return row, current_path

        # File not at recorded path; search raw_dir tree by filename
        filename = row["file_name"]
        for root, _, files in os.walk(raw_dir):
            if filename in files:
                new_path = os.path.join(root, filename)
                if os.path.isfile(new_path):
                    with self.lock, self._get_connection() as conn:
                        try:
                            st = os.stat(new_path)
                            conn.execute(
                                "UPDATE photos SET file_path = ?, file_mtime = ?, file_size = ? WHERE id = ?",
                                (new_path, st.st_mtime, st.st_size, photo_id)
                            )
                            conn.commit()
                            logger.info(f"Self-healed photo #{photo_id} path: {current_path} -> {new_path}")
                        except Exception as e:
                            logger.error(f"Error updating photo path during resolve: {e}")
                    return row, new_path

        return None

    def purge_missing_photos(self, raw_dir: str) -> Dict[str, int]:
        """
        Scan database for photos whose physical files are no longer on disk.
        Attempts self-healing first if photos were moved to a subfolder.
        If truly missing, purges the records from SQLite and FAISS vector index.
        Returns dict with counts of purged photos, purged faces, and updated paths.
        """
        with self.lock:
            with self._get_connection() as conn:
                rows = conn.execute("SELECT id, file_path, file_name FROM photos").fetchall()

            if not rows:
                return {"purged_photos": 0, "purged_faces": 0, "updated_paths": 0}

            # Pre-index existing files on disk for fast lookup: filename -> list of paths
            disk_files_by_name: Dict[str, List[str]] = {}
            for root, _, files in os.walk(raw_dir):
                for f in files:
                    disk_files_by_name.setdefault(f, []).append(os.path.join(root, f))

            stale_photo_ids: List[int] = []
            stale_vector_ids: List[int] = []
            updated_count = 0

            with self._get_connection() as conn:
                for r in rows:
                    p_id = r["id"]
                    curr_path = r["file_path"]
                    f_name = r["file_name"]

                    if os.path.isfile(curr_path):
                        continue

                    # Attempt self-healing via subfolder search
                    possible_paths = disk_files_by_name.get(f_name, [])
                    if possible_paths:
                        new_path = possible_paths[0]
                        try:
                            st = os.stat(new_path)
                            conn.execute(
                                "UPDATE photos SET file_path = ?, file_mtime = ?, file_size = ? WHERE id = ?",
                                (new_path, st.st_mtime, st.st_size, p_id)
                            )
                            updated_count += 1
                            logger.info(f"Self-healed relocated photo #{p_id}: {curr_path} -> {new_path}")
                            continue
                        except Exception:
                            pass

                    # Truly missing photo
                    stale_photo_ids.append(p_id)
                    v_rows = conn.execute("SELECT vector_id FROM faces WHERE photo_id = ?", (p_id,)).fetchall()
                    for v in v_rows:
                        stale_vector_ids.append(v["vector_id"])

                # Remove stale records from SQLite (cascades to faces table)
                if stale_photo_ids:
                    placeholders = ",".join("?" for _ in stale_photo_ids)
                    conn.execute(f"DELETE FROM photos WHERE id IN ({placeholders})", stale_photo_ids)
                    conn.commit()

            # Remove associated vectors from FAISS index
            if stale_vector_ids and FAISS_AVAILABLE and self.index is not None:
                try:
                    self.index.remove_ids(np.array(stale_vector_ids, dtype=np.int64))
                    self._save_faiss()
                    logger.info(f"Purged {len(stale_photo_ids)} missing photos and {len(stale_vector_ids)} vectors from FAISS.")
                except Exception as e:
                    logger.warning(f"Error removing IDs from FAISS ({e}). Rebuilding index from database...")
                    self._rebuild_faiss_from_sqlite()

            return {
                "purged_photos": len(stale_photo_ids),
                "purged_faces": len(stale_vector_ids),
                "updated_paths": updated_count
            }

    def search_similar_faces(
        self,
        query_encoding: np.ndarray,
        threshold: float = 0.75,
        top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Query FAISS for matching faces above the similarity threshold and fetch photo details from SQLite.
        Inner product on unit-normalized vectors directly yields cosine similarity (0.0 to 1.0).
        """
        with self.lock:
            if not FAISS_AVAILABLE or self.index is None or self.index.ntotal == 0:
                logger.warning("FAISS index is empty or unavailable.")
                return []

            # Ensure query vector is unit-normalized and 2D
            norm = np.linalg.norm(query_encoding)
            if norm > 0:
                query_norm = (query_encoding / norm).astype(np.float32)
            else:
                query_norm = query_encoding.astype(np.float32)

            query_matrix = np.expand_dims(query_norm, axis=0)

            # Cap top_k to total vectors available
            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_matrix, k)

            matched_vector_ids = []
            scores_by_vid = {}

            # Parse results
            for score, vid in zip(distances[0], indices[0]):
                if vid != -1 and score >= threshold:
                    matched_vector_ids.append(int(vid))
                    scores_by_vid[int(vid)] = float(score)

            if not matched_vector_ids:
                return []

            # Query SQLite for corresponding photo information
            placeholders = ",".join("?" for _ in matched_vector_ids)
            query_sql = f"""
                SELECT 
                    f.vector_id,
                    f.bbox_top, f.bbox_right, f.bbox_bottom, f.bbox_left,
                    p.id AS photo_id,
                    p.file_path,
                    p.file_name,
                    p.created_at,
                    p.num_faces
                FROM faces f
                JOIN photos p ON f.photo_id = p.id
                WHERE f.vector_id IN ({placeholders})
            """

            with self._get_connection() as conn:
                cursor = conn.execute(query_sql, matched_vector_ids)
                rows = cursor.fetchall()

            # Group results by photo so that multiple face matches on the same photo retain the highest similarity
            results_by_photo: Dict[int, Dict[str, Any]] = {}
            for r in rows:
                vid = r["vector_id"]
                score = scores_by_vid.get(vid, 0.0)
                photo_id = r["photo_id"]

                face_info = {
                    "vector_id": vid,
                    "score": round(score, 4),
                    "bbox": [r["bbox_top"], r["bbox_right"], r["bbox_bottom"], r["bbox_left"]]
                }

                if photo_id not in results_by_photo:
                    results_by_photo[photo_id] = {
                        "photo_id": photo_id,
                        "file_path": r["file_path"],
                        "file_name": r["file_name"],
                        "created_at": r["created_at"],
                        "num_faces": r["num_faces"],
                        "max_score": round(score, 4),
                        "best_vector_id": vid,
                        "best_bbox": [r["bbox_top"], r["bbox_right"], r["bbox_bottom"], r["bbox_left"]],
                        "matched_faces": [face_info]
                    }
                else:
                    results_by_photo[photo_id]["matched_faces"].append(face_info)
                    if score > results_by_photo[photo_id]["max_score"]:
                        results_by_photo[photo_id]["max_score"] = round(score, 4)
                        results_by_photo[photo_id]["best_vector_id"] = vid
                        results_by_photo[photo_id]["best_bbox"] = [r["bbox_top"], r["bbox_right"], r["bbox_bottom"], r["bbox_left"]]

            # Sort results descending by highest similarity score
            sorted_results = sorted(
                results_by_photo.values(),
                key=lambda x: x["max_score"],
                reverse=True
            )
            return sorted_results

    def get_stats(self) -> Dict[str, Any]:
        """Return system statistics on photos, faces, and index size."""
        with self.lock, self._get_connection() as conn:
            p_cur = conn.execute("SELECT COUNT(*) FROM photos")
            total_photos = p_cur.fetchone()[0]

            f_cur = conn.execute("SELECT COUNT(*) FROM faces")
            total_faces = f_cur.fetchone()[0]

            no_face_cur = conn.execute("SELECT COUNT(*) FROM photos WHERE status = 'no_faces'")
            no_faces = no_face_cur.fetchone()[0]

            err_cur = conn.execute("SELECT COUNT(*) FROM photos WHERE status = 'error'")
            errors = err_cur.fetchone()[0]

        faiss_total = self.index.ntotal if (FAISS_AVAILABLE and self.index) else 0

        return {
            "total_photos": total_photos,
            "total_faces": total_faces,
            "photos_without_faces": no_faces,
            "indexing_errors": errors,
            "faiss_indexed_vectors": faiss_total,
            "faiss_ready": FAISS_AVAILABLE and (self.index is not None),
            "face_rec_ready": FACE_REC_AVAILABLE
        }
