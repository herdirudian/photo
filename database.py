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
import cv2

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
        High-Performance Multi-Scale Face Detection optimized for The Lodge Maribaya ride photos:
        - Sky Swing / Ayunan (prominent center subject, vertical ropes)
        - Hammock / Zip Line (suspended subjects, horizontal cables, deep forest background)
        - Hot Air Balloon (close & wide basket, groups of 1-6 people, glasses, and hijabs)
        """
        h, w = image.shape[:2]
        all_locations: List[Tuple[int, int, int, int]] = []
        max_dim = max(h, w)

        # 1. Main Global Pass
        if max_dim > 2000:
            scale = 2000.0 / max_dim
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
            if len(global_locs) == 0 and max_dim <= 1200:
                all_locations.extend(face_recognition.face_locations(image, number_of_times_to_upsample=2, model=self.model))

        # 2. Targeted Ride Zone Pass:
        # Across all rides (Swing, Hammock, Hot Air Balloon), guests are ALWAYS in y in [0.20, 0.98]
        # and x in [0.05, 0.95]. (Excludes pure sky and treetop canopy above rides).
        ry1, rx1, ry2, rx2 = int(h * 0.20), int(w * 0.05), int(h * 0.98), int(w * 0.95)
        ride_crop = image[ry1:ry2, rx1:rx2]
        if ride_crop.size > 0:
            c_dim = max(ride_crop.shape[:2])
            scale_r = 1800.0 / c_dim if c_dim > 1800 else 1.0
            if scale_r < 1.0:
                cw_s, ch_s = int(ride_crop.shape[1] * scale_r), int(ride_crop.shape[0] * scale_r)
                pil_c = Image.fromarray(ride_crop).resize((cw_s, ch_s), Image.Resampling.BILINEAR)
                ride_crop_s = np.array(pil_c)
            else:
                ride_crop_s = ride_crop

            upsample_num = 2 if max(ride_crop_s.shape[:2]) <= 1100 else 1
            r_locs = face_recognition.face_locations(ride_crop_s, number_of_times_to_upsample=upsample_num, model=self.model)
            for top, right, bottom, left in r_locs:
                all_locations.append((
                    ry1 + int(round(top / scale_r)),
                    rx1 + int(round(right / scale_r)),
                    ry1 + int(round(bottom / scale_r)),
                    rx1 + int(round(left / scale_r))
                ))

        # 3. Dedicated Balloon Basket Scan (Bottom-Center Crop: y in [0.65, 0.98], x in [0.20, 0.80])
        # In Hot Air Balloon photos, 6 guests stand closely packed in the basket.
        # This targeted crop ensures distant balloon basket faces are ALWAYS caught.
        by1, bx1, by2, bx2 = int(h * 0.65), int(w * 0.20), int(h * 0.98), int(w * 0.80)
        basket_crop = image[by1:by2, bx1:bx2]
        if basket_crop.size > 0:
            b_dim = max(basket_crop.shape[:2])
            scale_b = 1200.0 / b_dim if b_dim > 1200 else 1.0
            if scale_b < 1.0:
                bw_s, bh_s = int(basket_crop.shape[1] * scale_b), int(basket_crop.shape[0] * scale_b)
                basket_crop_s = np.array(Image.fromarray(basket_crop).resize((bw_s, bh_s), Image.Resampling.BILINEAR))
            else:
                basket_crop_s = basket_crop

            b_locs = face_recognition.face_locations(basket_crop_s, number_of_times_to_upsample=1, model=self.model)
            for top, right, bottom, left in b_locs:
                all_locations.append((
                    by1 + int(round(top / scale_b)),
                    bx1 + int(round(right / scale_b)),
                    by1 + int(round(bottom / scale_b)),
                    bx1 + int(round(left / scale_b))
                ))

        # Deduplicate overlapping bounding boxes from different scales via Non-Maximum Suppression
        dedup_boxes = self._nms_boxes(all_locations, iou_thresh=0.35)

        # Adaptive minimum face dimension based on image size:
        # In 1024x682: min_size = 14px (catches distant balloon faces)
        # In 6000x4000: min_size = 50px (filters distant leaves & bark noise)
        min_dim = max(14, int(min(h, w) * 0.015))

        filtered = []
        for loc in dedup_boxes:
            box_h = loc[2] - loc[0]
            box_w = loc[1] - loc[3]
            if box_w >= min_dim and box_h >= min_dim:
                ratio = float(box_h) / float(box_w)
                # Human face aspect ratio (height / width)
                # Rejects tall skinny ropes/poles (ratio > 1.55) and horizontal rails (ratio < 0.65)
                if 0.65 <= ratio <= 1.55:
                    filtered.append(loc)

        return filtered

    def is_valid_human_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Validate whether a candidate bounding box is genuinely a human face
        using anatomical 68-point facial landmark geometry and aspect ratio checks.
        Optimized for The Lodge Maribaya ride photos (Swing, Hammock, Hot Air Balloon,
        including guests wearing glasses and hijabs).
        Filters out false positives on trees, pine bark, wooden poles, ropes, and metal equipment.
        """
        if not FACE_REC_AVAILABLE:
            return True

        img_h, img_w = image.shape[:2]
        top, right, bottom, left = bbox
        w = right - left
        h = bottom - top

        # 1. Adaptive dimension check
        min_dim = max(14, int(min(img_h, img_w) * 0.015))
        if w < min_dim or h < min_dim:
            return False

        # 2. Aspect ratio check
        aspect_ratio = float(h) / float(w)
        if aspect_ratio < 0.65 or aspect_ratio > 1.55:
            return False

        # 3. Variance / Contrast Check
        crop_top = max(0, min(img_h, top))
        crop_bottom = max(0, min(img_h, bottom))
        crop_left = max(0, min(img_w, left))
        crop_right = max(0, min(img_w, right))
        
        if crop_bottom <= crop_top or crop_right <= crop_left:
            return False
            
        crop = image[crop_top:crop_bottom, crop_left:crop_right]
        if crop.size == 0 or np.std(crop) < 10.0:
            return False

        # 4. 68-Landmark Anatomical Validation:
        try:
            landmarks_list = face_recognition.face_landmarks(image, face_locations=[bbox], model="large")
        except Exception as e:
            logger.warning(f"Error computing face landmarks for bbox {bbox}: {e}")
            landmarks_list = []

        if not landmarks_list or len(landmarks_list) == 0:
            # If landmarks could not be extracted on a very small distant face (< 28px),
            # allow it only if aspect ratio is strictly oval/circular (0.75 - 1.35)
            if min(w, h) < 28:
                return (0.75 <= aspect_ratio <= 1.35)
            return False

        lm = landmarks_list[0]

        left_eye = lm.get("left_eye")
        right_eye = lm.get("right_eye")
        nose_tip = lm.get("nose_tip")
        bottom_lip = lm.get("bottom_lip")
        top_lip = lm.get("top_lip")
        chin = lm.get("chin")

        if not left_eye or not right_eye:
            return False

        # Compute eye centers
        lex = sum(p[0] for p in left_eye) / len(left_eye)
        ley = sum(p[1] for p in left_eye) / len(left_eye)
        rex = sum(p[0] for p in right_eye) / len(right_eye)
        rey = sum(p[1] for p in right_eye) / len(right_eye)

        eye_dist_x = rex - lex

        # In real human faces (even 3/4 view or wearing glasses), left eye is to the left of right eye
        if eye_dist_x <= 0:
            return False

        # Distance between eyes: typical 12% to 80% of face width
        # (On vertical poles/cables, points collapse horizontally so eye_dist_x is near zero)
        if eye_dist_x < (0.12 * w) or eye_dist_x > (0.80 * w):
            return False

        # Eye tilt / alignment: accommodates playful head tilts on swings/rides up to 48 degrees
        eye_diff_y = abs(rey - ley)
        if eye_diff_y > (1.10 * eye_dist_x):
            return False

        # Vertical anatomical ordering (Eyes above Nose, Nose above Mouth):
        eyes_y = (ley + rey) / 2.0
        if nose_tip:
            nose_y = sum(p[1] for p in nose_tip) / len(nose_tip)
            if (nose_y - eyes_y) < -0.05 * h:
                return False

        mouth_pts = bottom_lip if bottom_lip else top_lip
        if mouth_pts and nose_tip:
            mouth_y = sum(p[1] for p in mouth_pts) / len(mouth_pts)
            if (mouth_y - nose_y) < -0.05 * h:
                return False

        # Chin check: HIJAB-SAFE
        # Guests wearing hijab have fabric covering the chin, so chin points rest on the cloth.
        # Only verify that chin is not inverted above the mouth.
        if chin and mouth_pts:
            chin_y = max(p[1] for p in chin)
            if chin_y < (mouth_y - 0.05 * h):
                return False

        return True

    def compute_torso_color_hist(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract upper body crop and compute normalized HSV histogram.
        bbox is (top, right, bottom, left)
        """
        top, right, bottom, left = bbox
        h, w, _ = image.shape
        face_h = bottom - top
        face_w = right - left
        
        # Estimate torso below the face
        # Torso width is roughly 2.5x face width, height is roughly 2.0x face height
        torso_top = bottom + int(face_h * 0.1) # small gap below chin
        torso_bottom = min(h, torso_top + int(face_h * 2.0))
        
        torso_center_x = left + (face_w // 2)
        torso_width = int(face_w * 2.5)
        torso_left = max(0, torso_center_x - (torso_width // 2))
        torso_right = min(w, torso_center_x + (torso_width // 2))
        
        if torso_bottom <= torso_top or torso_right <= torso_left:
            # Fallback to zero histogram if bounding box is invalid/out of bounds
            return np.zeros(256, dtype=np.float32)
            
        crop = image[torso_top:torso_bottom, torso_left:torso_right]
        
        # Convert RGB to HSV
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        
        # Compute 2D histogram of Hue and Saturation
        # Hue range: 0-180, Saturation: 0-256
        hist = cv2.calcHist([hsv_crop], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        return hist.flatten().astype(np.float32)

    def extract_faces_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract bounding boxes and 128-d facial embeddings from an image file on disk
        using intelligent multi-scale tiled detection with anatomical landmark validation.
        Eliminates poles, trees, pine bark, and park equipment from face index.
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

        # Find all face candidate locations via multi-scale scan
        candidate_locations = self.detect_multiscale_locations(image)

        if not candidate_locations:
            return []

        # Validate each face candidate through strict anatomical landmark checks
        valid_locations = []
        for loc in candidate_locations:
            if self.is_valid_human_face(image, loc):
                valid_locations.append(loc)
            else:
                logger.info(f"Filtered out non-human false detection at {loc} in {os.path.basename(file_path)} (pole/tree/equipment)")

        if not valid_locations:
            return []

        # Compute 128-d encodings using the high-accuracy 68-landmark model ONLY for verified human faces
        encodings = face_recognition.face_encodings(image, known_face_locations=valid_locations, num_jitters=1, model="large")

        results = []
        for loc, enc in zip(valid_locations, encodings):
            # Normalize vector to unit length for cosine similarity via inner product
            norm = np.linalg.norm(enc)
            if norm > 0:
                normalized_enc = (enc / norm).astype(np.float32)
            else:
                normalized_enc = enc.astype(np.float32)

            color_hist = self.compute_torso_color_hist(image, loc)

            results.append({
                "bbox": loc,
                "encoding": normalized_enc,
                "color_hist": color_hist
            })

        return results

    def extract_faces_from_bytes(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract face locations and encodings from in-memory bytes with autocontrast and landmark validation."""
        if not FACE_REC_AVAILABLE:
            raise RuntimeError("face_recognition is not available in current environment")

        image_obj = Image.open(io.BytesIO(image_bytes))
        # Ensure RGB format
        if image_obj.mode != "RGB":
            image_obj = image_obj.convert("RGB")
        
        image_np = np.array(image_obj)

        locations = face_recognition.face_locations(image_np, number_of_times_to_upsample=1, model=self.model)
        if not locations:
            locations = face_recognition.face_locations(image_np, number_of_times_to_upsample=2, model=self.model)

        if not locations:
            return []

        # Validate human face geometry for webcam/uploaded photo
        valid_locations = [
            loc for loc in locations
            if self.is_valid_human_face(image_np, loc)
        ]

        # Sane aspect ratio fallback for webcam closeups if extreme lighting
        if not valid_locations:
            valid_locations = [
                loc for loc in locations
                if 0.65 <= ((loc[2] - loc[0]) / max(1, loc[1] - loc[3])) <= 1.55
            ]

        if not valid_locations:
            return []

        # For the query guest photo, use num_jitters=2 and model="large" to produce an ultra-stable embedding
        encodings = face_recognition.face_encodings(image_np, known_face_locations=valid_locations, num_jitters=2, model="large")
        results = []
        for loc, enc in zip(valid_locations, encodings):
            norm = np.linalg.norm(enc)
            if norm > 0:
                normalized_enc = (enc / norm).astype(np.float32)
            else:
                normalized_enc = enc.astype(np.float32)

            color_hist = self.compute_torso_color_hist(image_np, loc)

            results.append({
                "bbox": loc,
                "encoding": normalized_enc,
                "color_hist": color_hist
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
                    color_hist_blob BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_photos_path ON photos(file_path);
                CREATE INDEX IF NOT EXISTS idx_photos_status ON photos(status);
                CREATE INDEX IF NOT EXISTS idx_faces_photo_id ON faces(photo_id);
                CREATE INDEX IF NOT EXISTS idx_faces_vector_id ON faces(vector_id);
            """)
            
            # Migration
            try:
                conn.execute("ALTER TABLE faces ADD COLUMN color_hist_blob BLOB")
                logger.info("Migrated faces table: added color_hist_blob column.")
            except sqlite3.OperationalError:
                pass # Column already exists or table is fresh
                
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
                    color_hist = face_data.get("color_hist")
                    color_blob = color_hist.tobytes() if color_hist is not None else None

                    conn.execute("""
                        INSERT INTO faces (photo_id, vector_id, bbox_top, bbox_right, bbox_bottom, bbox_left, encoding_blob, color_hist_blob)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        photo_id,
                        vector_id,
                        int(bbox[0]),
                        int(bbox[1]),
                        int(bbox[2]),
                        int(bbox[3]),
                        encoding.tobytes(),
                        color_blob
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

            # Also purge any invalid face bounding boxes (poles, trees)
            invalid_face_stats = self.purge_invalid_face_boxes()

            return {
                "purged_photos": len(stale_photo_ids),
                "purged_faces": len(stale_vector_ids) + invalid_face_stats.get("purged_faces", 0),
                "updated_paths": updated_count,
                "purged_invalid_faces": invalid_face_stats.get("purged_faces", 0)
            }

    def purge_invalid_face_boxes(self) -> Dict[str, int]:
        """
        Scan existing face records in SQLite and purge false positives
        (e.g., vertical poles, tree bark, ropes, mechanical equipment)
        based on bounding box dimensions and aspect ratios.
        Removes invalid vector IDs from FAISS and updates photo counts.
        """
        with self.lock:
            stale_face_ids: List[int] = []
            stale_vector_ids: List[int] = []
            affected_photo_ids = set()

            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT id, photo_id, vector_id, bbox_top, bbox_right, bbox_bottom, bbox_left FROM faces"
                ).fetchall()

                for r in rows:
                    w = r["bbox_right"] - r["bbox_left"]
                    h = r["bbox_bottom"] - r["bbox_top"]

                    is_invalid = False
                    if w < 32 or h < 32:
                        is_invalid = True
                    else:
                        ratio = float(h) / float(w)
                        if ratio < 0.70 or ratio > 1.48:
                            is_invalid = True

                    if is_invalid:
                        stale_face_ids.append(r["id"])
                        stale_vector_ids.append(r["vector_id"])
                        affected_photo_ids.add(r["photo_id"])

                if stale_face_ids:
                    placeholders = ",".join("?" for _ in stale_face_ids)
                    conn.execute(f"DELETE FROM faces WHERE id IN ({placeholders})", stale_face_ids)

                    # Update num_faces and status on affected photos
                    for p_id in affected_photo_ids:
                        cnt = conn.execute("SELECT COUNT(*) FROM faces WHERE photo_id = ?", (p_id,)).fetchone()[0]
                        status = "indexed" if cnt > 0 else "no_faces"
                        conn.execute("UPDATE photos SET num_faces = ?, status = ? WHERE id = ?", (cnt, status, p_id))

                    conn.commit()
                    logger.info(f"Purged {len(stale_face_ids)} invalid pole/tree face records from database.")

            # Remove invalid vectors from FAISS index
            if stale_vector_ids and FAISS_AVAILABLE and self.index is not None:
                try:
                    self.index.remove_ids(np.array(stale_vector_ids, dtype=np.int64))
                    self._save_faiss()
                    logger.info(f"Removed {len(stale_vector_ids)} invalid vectors from FAISS index.")
                except Exception as e:
                    logger.warning(f"Error removing IDs from FAISS ({e}). Rebuilding index from database...")
                    self._rebuild_faiss_from_sqlite()

            return {
                "purged_faces": len(stale_face_ids),
                "affected_photos": len(affected_photo_ids)
            }

    def search_similar_faces(
        self,
        query_faces: List[Dict[str, Any]],
        threshold: float = 0.75,
        top_k: int = 50,
        ride_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query FAISS for matching faces above the similarity threshold and fetch photo details from SQLite.
        Inner product on unit-normalized vectors directly yields cosine similarity (0.0 to 1.0).
        Supports optional filtering by ride subfolder (e.g. 'BL' for Hot Air Balloon).
        Applies score boost for matches with similar upper-body clothing color.
        """
        with self.lock:
            if not FAISS_AVAILABLE or self.index is None or self.index.ntotal == 0:
                logger.warning("FAISS index is empty or unavailable.")
                return []

            matched_vector_ids = set()
            scores_by_vid = {}
            k = min(top_k, self.index.ntotal)
            
            # Helper to safely compute color boost
            def get_color_boost(db_blob):
                if not db_blob:
                    return 0.0
                try:
                    db_hist = np.frombuffer(db_blob, dtype=np.float32)
                    if db_hist.size != 256:
                        return 0.0
                    
                    best_boost = 0.0
                    for qf in query_faces:
                        q_hist = qf.get("color_hist")
                        if q_hist is not None and q_hist.size == 256:
                            dist = cv2.compareHist(q_hist, db_hist, cv2.HISTCMP_BHATTACHARYYA)
                            boost = max(0.0, 0.10 * (1.0 - dist))
                            if boost > best_boost:
                                best_boost = boost
                    return best_boost
                except Exception:
                    return 0.0

            for qf in query_faces:
                q_enc = qf["encoding"]
                norm = np.linalg.norm(q_enc)
                if norm > 0:
                    query_norm = (q_enc / norm).astype(np.float32)
                else:
                    query_norm = q_enc.astype(np.float32)

                query_matrix = np.expand_dims(query_norm, axis=0)
                distances, indices = self.index.search(query_matrix, k)
                
                for score, vid in zip(distances[0], indices[0]):
                    if vid != -1 and score >= threshold:
                        matched_vector_ids.add(int(vid))
                        # Keep highest score if vector matches multiple poses
                        if int(vid) not in scores_by_vid or score > scores_by_vid[int(vid)]:
                            scores_by_vid[int(vid)] = float(score)

            if not matched_vector_ids:
                return []

            # Query SQLite for corresponding photo information
            matched_vector_ids_list = list(matched_vector_ids)
            placeholders = ",".join("?" for _ in matched_vector_ids_list)
            query_sql = f"""
                SELECT 
                    f.vector_id,
                    f.bbox_top, f.bbox_right, f.bbox_bottom, f.bbox_left,
                    f.color_hist_blob,
                    p.id AS photo_id,
                    p.file_path,
                    p.file_name,
                    p.created_at,
                    p.num_faces
                FROM faces f
                JOIN photos p ON f.photo_id = p.id
                WHERE f.vector_id IN ({placeholders})
            """
            params = matched_vector_ids_list.copy()

            # Apply ride subfolder filter if specified (e.g. 'BL' for Hot Air Balloon)
            if ride_filter and ride_filter.strip() and ride_filter.lower() not in ["all", "semua"]:
                clean_rf = ride_filter.strip()
                query_sql += " AND (p.file_path LIKE ? OR p.file_path LIKE ?)"
                params.extend([f"%/{clean_rf}/%", f"%\\{clean_rf}\\%"])

            with self._get_connection() as conn:
                cursor = conn.execute(query_sql, params)
                rows = cursor.fetchall()

            # Group results by photo so that multiple face matches on the same photo retain the highest similarity
            results_by_photo: Dict[int, Dict[str, Any]] = {}
            for r in rows:
                vid = r["vector_id"]
                base_score = scores_by_vid.get(vid, 0.0)
                boost = get_color_boost(r["color_hist_blob"])
                final_score = base_score + boost
                
                photo_id = r["photo_id"]

                face_info = {
                    "vector_id": vid,
                    "score": round(final_score, 4),
                    "base_score": round(base_score, 4),
                    "color_boost": round(boost, 4),
                    "bbox": [r["bbox_top"], r["bbox_right"], r["bbox_bottom"], r["bbox_left"]]
                }

                if photo_id not in results_by_photo:
                    results_by_photo[photo_id] = {
                        "photo_id": photo_id,
                        "file_path": r["file_path"],
                        "file_name": r["file_name"],
                        "created_at": r["created_at"],
                        "num_faces": r["num_faces"],
                        "max_score": round(final_score, 4),
                        "best_vector_id": vid,
                        "best_bbox": [r["bbox_top"], r["bbox_right"], r["bbox_bottom"], r["bbox_left"]],
                        "matched_faces": [face_info]
                    }
                else:
                    results_by_photo[photo_id]["matched_faces"].append(face_info)
                    if final_score > results_by_photo[photo_id]["max_score"]:
                        results_by_photo[photo_id]["max_score"] = round(final_score, 4)
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
