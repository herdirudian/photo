"""
database.py - SQLite persistence and FAISS vector similarity search for face retrieval.
"""

import os
import sqlite3
import threading
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

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

    def extract_faces_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract bounding boxes and 128-d facial embeddings from an image file on disk.
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

        # Find all face locations (top, right, bottom, left)
        locations = face_recognition.face_locations(image, model=self.model)
        if not locations:
            return []

        # Compute 128-d encodings for each location
        encodings = face_recognition.face_encodings(image, known_face_locations=locations)

        results = []
        for loc, enc in zip(locations, encodings):
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
        """Extract face locations and encodings from in-memory bytes."""
        if not FACE_REC_AVAILABLE:
            raise RuntimeError("face_recognition is not available in current environment")

        import io
        from PIL import Image

        image_obj = Image.open(io.BytesIO(image_bytes))
        # Ensure RGB format
        if image_obj.mode != "RGB":
            image_obj = image_obj.convert("RGB")
        image_np = np.array(image_obj)

        locations = face_recognition.face_locations(image_np, model=self.model)
        if not locations:
            return []

        encodings = face_recognition.face_encodings(image_np, known_face_locations=locations)
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
                        "matched_faces": [face_info]
                    }
                else:
                    results_by_photo[photo_id]["matched_faces"].append(face_info)
                    if score > results_by_photo[photo_id]["max_score"]:
                        results_by_photo[photo_id]["max_score"] = round(score, 4)

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
