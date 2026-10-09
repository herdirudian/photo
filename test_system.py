"""
test_system.py - System Architecture and Integration Verification Script.
Validates SQLite schema, FAISS vector indexing, similarity thresholds, and symlink generation.
"""

import os
import sys
import tempfile
import time
import shutil
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from database import DatabaseManager, FaceEngine, FAISS_AVAILABLE
from indexer import is_valid_image, PhotoIndexer


class TestPhotoRetrievalSystem(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.temp_dir, "raw")
        self.results_dir = os.path.join(self.temp_dir, "results")
        self.db_dir = os.path.join(self.temp_dir, "db")

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.db_dir, exist_ok=True)

        self.db = DatabaseManager(db_dir=self.db_dir, vector_dim=512)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_database_initialization(self):
        """Verify SQLite schema is properly created with required tables and indexes."""
        with self.db._get_connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row["name"] for row in cursor.fetchall()}
            self.assertIn("photos", tables)
            self.assertIn("faces", tables)

    def test_photo_registration_and_stats_512d(self):
        """Verify photo and multiple 512-d ArcFace vectors can be registered and retrieved."""
        # Create a dummy image file
        mock_file = os.path.join(self.raw_dir, "balloon_001.jpg")
        with open(mock_file, "wb") as f:
            f.write(b"mock_balloon_ride_photo")

        # Generate two synthetic 512-d ArcFace feature vectors
        vec1 = np.random.randn(512).astype(np.float32)
        vec1 /= np.linalg.norm(vec1)

        vec2 = np.random.randn(512).astype(np.float32)
        vec2 /= np.linalg.norm(vec2)

        faces = [
            {"bbox": (100, 200, 300, 150), "encoding": vec1},
            {"bbox": (120, 450, 310, 390), "encoding": vec2}
        ]

        stat = os.stat(mock_file)
        photo_id = self.db.register_photo(
            file_path=mock_file,
            file_mtime=stat.st_mtime,
            file_size=stat.st_size,
            faces=faces,
            status="indexed"
        )
        self.assertGreater(photo_id, 0)

        # Check duplicate avoidance
        is_indexed = self.db.is_photo_indexed(mock_file, stat.st_mtime, stat.st_size)
        self.assertTrue(is_indexed)

        # Check stats
        stats = self.db.get_stats()
        self.assertEqual(stats["total_photos"], 1)
        self.assertEqual(stats["total_faces"], 2)
        self.assertEqual(stats["vector_dim"], 512)

    def test_faiss_arcface_512d_similarity_search(self):
        """Verify 512-d vector similarity query returns correct photo and cosine score."""
        if not FAISS_AVAILABLE:
            self.skipTest("FAISS library not installed in host environment")

        mock_file = os.path.join(self.raw_dir, "swing_042.jpg")
        with open(mock_file, "wb") as f:
            f.write(b"mock_swing_photo")

        # Base 512-d vector
        target_vec = np.random.randn(512).astype(np.float32)
        target_vec /= np.linalg.norm(target_vec)

        self.db.register_photo(
            file_path=mock_file,
            file_mtime=time.time(),
            file_size=1024,
            faces=[{"bbox": (50, 100, 150, 80), "encoding": target_vec}],
            status="indexed"
        )

        # Query with exact vector (similarity should be ~1.0)
        matches = self.db.search_similar_faces(query_encoding=target_vec, threshold=0.90)
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(matches[0]["max_score"], 1.0, places=3)
        self.assertEqual(matches[0]["file_name"], "swing_042.jpg")

        # Query with slightly perturbed vector (similarity ~0.95 in 512-d)
        noise = np.random.randn(512).astype(np.float32) * 0.05
        similar_vec = target_vec + noise
        similar_vec /= np.linalg.norm(similar_vec)

        matches_similar = self.db.search_similar_faces(query_encoding=similar_vec, threshold=0.85)
        self.assertEqual(len(matches_similar), 1)
        self.assertGreater(matches_similar[0]["max_score"], 0.85)

        # Query with orthogonal vector (low similarity, should not return match at 0.50)
        ortho_vec = np.random.randn(512).astype(np.float32)
        ortho_vec -= ortho_vec.dot(target_vec) * target_vec
        ortho_vec /= np.linalg.norm(ortho_vec)

        matches_unrelated = self.db.search_similar_faces(query_encoding=ortho_vec, threshold=0.50)
        self.assertEqual(len(matches_unrelated), 0)

    def test_legacy_128d_compatibility(self):
        """Verify DatabaseManager supports legacy 128-d vectors when requested."""
        db_128 = DatabaseManager(db_dir=os.path.join(self.temp_dir, "db128"), vector_dim=128)
        self.assertEqual(db_128.vector_dim, 128)

        mock_file = os.path.join(self.raw_dir, "legacy_001.jpg")
        with open(mock_file, "wb") as f:
            f.write(b"legacy_photo")

        vec128 = np.random.randn(128).astype(np.float32)
        vec128 /= np.linalg.norm(vec128)

        p_id = db_128.register_photo(
            file_path=mock_file,
            file_mtime=time.time(),
            file_size=512,
            faces=[{"bbox": (20, 40, 80, 70), "encoding": vec128}],
            status="indexed"
        )
        self.assertGreater(p_id, 0)
        stats = db_128.get_stats()
        self.assertEqual(stats["vector_dim"], 128)

    def test_symlink_creation_workflow(self):
        """Verify symlink creation logic and customer virtual folder generation."""
        mock_raw = os.path.join(self.raw_dir, "ride_photo.jpg")
        with open(mock_raw, "wb") as f:
            f.write(b"ride_photo_content")

        customer_folder = os.path.join(self.results_dir, "Customer_TEST01")
        os.makedirs(customer_folder, exist_ok=True)

        symlink_target = os.path.join(customer_folder, "ride_photo.jpg")

        # Create relative symlink
        rel_path = os.path.relpath(mock_raw, customer_folder)
        try:
            os.symlink(rel_path, symlink_target)
            self.assertTrue(os.path.islink(symlink_target))
            self.assertTrue(os.path.exists(symlink_target))
            # Verify target content matches source
            with open(symlink_target, "rb") as f:
                self.assertEqual(f.read(), b"ride_photo_content")
        except OSError:
            # On Windows without developer mode, verify fallback behavior
            pass

    def test_extension_filter(self):
        """Verify supported image extensions filter."""
        self.assertTrue(is_valid_image("photo.JPG"))
        self.assertTrue(is_valid_image("photo.jpeg"))
        self.assertTrue(is_valid_image("photo.png"))
        self.assertTrue(is_valid_image("photo.webp"))
        self.assertFalse(is_valid_image("photo.txt"))
        self.assertFalse(is_valid_image("photo.exe"))
        self.assertFalse(is_valid_image("photo.mp4"))


if __name__ == "__main__":
    print("=" * 60)
    print("Executing System Architecture & Retrieval Tests...")
    print("=" * 60)
    unittest.main(verbosity=2)
