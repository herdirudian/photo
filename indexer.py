"""
indexer.py - Real-time folder monitoring and automated face vector extraction using Watchdog.
Monitors the raw theme park photos directory and updates SQLite + FAISS.
"""

import os
import time
import logging
import threading
from typing import Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent

from database import DatabaseManager, FaceEngine

# Configure logging
logger = logging.getLogger("PhotoRetrieval.Indexer")

SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def is_valid_image(file_path: str) -> bool:
    """Check if the file has a supported image extension."""
    _, ext = os.path.splitext(file_path)
    return ext.lower() in SUPPORTED_EXTENSIONS


def wait_for_file_transfer_complete(file_path: str, max_wait_sec: float = 6.0, poll_interval: float = 0.3) -> bool:
    """
    Ensure the file is completely written to disk before attempting computer vision extraction.
    Crucial for theme park network shares (Samba) or camera offloads where files arrive in chunks.
    """
    if not os.path.exists(file_path):
        return False

    # Fast path for existing stable files (copied more than 2 seconds ago)
    try:
        stat = os.stat(file_path)
        if stat.st_size > 0 and (time.time() - stat.st_mtime) > 2.0:
            with open(file_path, "rb") as f:
                f.seek(0)
                head = f.read(10)
                f.seek(max(0, stat.st_size - 10))
                tail = f.read(10)
                if head and tail:
                    return True
    except OSError:
        pass

    start_time = time.time()
    last_size = -1

    while time.time() - start_time < max_wait_sec:
        if not os.path.exists(file_path):
            return False

        try:
            current_size = os.path.getsize(file_path)
        except OSError:
            # File may be exclusively locked by transfer process
            time.sleep(poll_interval)
            continue

        if current_size > 0 and current_size == last_size:
            # Size has stabilized over the interval, quick check if readable
            try:
                with open(file_path, "rb") as f:
                    # Read the first and last few bytes to confirm non-empty write
                    f.seek(0)
                    head = f.read(10)
                    f.seek(max(0, current_size - 10))
                    tail = f.read(10)
                    if head and tail:
                        return True
            except OSError:
                pass

        last_size = current_size
        time.sleep(poll_interval)

    # Final check
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0


class RawPhotoEventHandler(FileSystemEventHandler):
    """Watchdog event handler that indexes new photo files as they arrive."""

    def __init__(self, db_manager: DatabaseManager, face_engine: FaceEngine):
        super().__init__()
        self.db_manager = db_manager
        self.face_engine = face_engine
        self._processing_lock = threading.Lock()

    def process_file(self, file_path: str):
        """Process a single image file for face detection and indexing."""
        if not is_valid_image(file_path):
            return

        # Normalized path
        file_path = os.path.abspath(file_path)
        if not wait_for_file_transfer_complete(file_path):
            logger.warning(f"File transfer timeout or incomplete write for {file_path}")
            return

        try:
            stat = os.stat(file_path)
            file_mtime = stat.st_mtime
            file_size = stat.st_size

            # Check if file has already been indexed
            if self.db_manager.is_photo_indexed(file_path, file_mtime, file_size):
                logger.debug(f"Skipping already indexed file: {file_path}")
                return

            logger.info(f"Extracting faces from new photo: {os.path.basename(file_path)}")
            faces = self.face_engine.extract_faces_from_file(file_path)

            status = "indexed" if len(faces) > 0 else "no_faces"
            self.db_manager.register_photo(
                file_path=file_path,
                file_mtime=file_mtime,
                file_size=file_size,
                faces=faces,
                status=status
            )
            logger.info(f"Indexed {len(faces)} face(s) in {os.path.basename(file_path)}")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            try:
                stat = os.stat(file_path)
                self.db_manager.register_photo(
                    file_path=file_path,
                    file_mtime=stat.st_mtime,
                    file_size=stat.st_size,
                    faces=[],
                    status="error"
                )
            except Exception:
                pass

    def on_created(self, event: FileCreatedEvent):
        if not event.is_directory:
            self.process_file(event.src_path)

    def on_moved(self, event: FileMovedEvent):
        if not event.is_directory:
            self.process_file(event.dest_path)


class PhotoIndexer:
    """
    Manages directory scanning, Watchdog observer lifecycle, and periodic polling.
    Periodic polling is essential for CIFS/Samba/NFS network mounts where Linux kernel inotify
    events are not triggered by remote file additions or deletions.
    """

    def __init__(self, raw_dir: str, db_manager: DatabaseManager, face_engine: FaceEngine, polling_interval: float = 15.0):
        self.raw_dir = raw_dir
        self.db_manager = db_manager
        self.face_engine = face_engine
        self.polling_interval = polling_interval
        self.observer: Optional[Observer] = None
        self._polling_thread: Optional[threading.Thread] = None
        self._running = False
        self._scan_lock = threading.Lock()
        os.makedirs(self.raw_dir, exist_ok=True)

    def scan_existing_files(self) -> dict:
        """
        Scan the raw directory recursively:
        1. Purge missing/stale photos from SQLite & FAISS (with self-healing for relocated files).
        2. Index newly discovered photos across all ride subfolders.
        Returns a dictionary of scan metrics.
        """
        if not self._scan_lock.acquire(blocking=False):
            logger.info("Scan already in progress. Skipping concurrent request.")
            return {"status": "busy"}

        try:
            logger.info(f"Starting directory scan on: {self.raw_dir}")

            # Step 1: Self-heal moved photos and purge deleted/missing photos
            purge_stats = self.db_manager.purge_missing_photos(self.raw_dir)
            if purge_stats["purged_photos"] > 0:
                logger.info(f"Cleaned up {purge_stats['purged_photos']} missing photos ({purge_stats['purged_faces']} face vectors).")
            if purge_stats["updated_paths"] > 0:
                logger.info(f"Self-healed paths for {purge_stats['updated_paths']} relocated photos.")

            # Step 2: Index new or updated files
            handler = RawPhotoEventHandler(self.db_manager, self.face_engine)
            scanned_count = 0
            new_indexed_count = 0
            subfolders_detected = set()

            for root, dirs, files in os.walk(self.raw_dir):
                for d in dirs:
                    rel_d = os.path.relpath(os.path.join(root, d), self.raw_dir)
                    subfolders_detected.add(rel_d)
                for filename in files:
                    full_path = os.path.join(root, filename)
                    if is_valid_image(full_path):
                        scanned_count += 1
                        try:
                            stat = os.stat(full_path)
                            if not self.db_manager.is_photo_indexed(full_path, stat.st_mtime, stat.st_size):
                                handler.process_file(full_path)
                                new_indexed_count += 1
                        except Exception as e:
                            logger.error(f"Error checking {full_path}: {e}")

            summary = {
                "scanned": scanned_count,
                "newly_indexed": new_indexed_count,
                "subfolders_count": len(subfolders_detected),
                "subfolders": sorted(list(subfolders_detected)),
                "purged_photos": purge_stats.get("purged_photos", 0),
                "purged_faces": purge_stats.get("purged_faces", 0),
                "updated_paths": purge_stats.get("updated_paths", 0)
            }
            logger.info(f"Scan complete on {self.raw_dir}. Scanned: {scanned_count} files ({len(subfolders_detected)} subfolders), Newly processed: {new_indexed_count}, Purged: {summary['purged_photos']}")
            return summary

        finally:
            self._scan_lock.release()

    def _polling_worker(self):
        """
        Background worker that periodically polls the network share (CIFS/Samba)
        to detect newly added photos or cleaned up files without relying on kernel inotify.
        """
        logger.info(f"Network share periodic polling worker started (interval: {self.polling_interval}s).")
        while self._running:
            # Sleep in small steps to react quickly to shutdown
            for _ in range(max(1, int(self.polling_interval))):
                if not self._running:
                    return
                time.sleep(1)

            try:
                self.scan_existing_files()
            except Exception as e:
                logger.error(f"Error in background polling scanner: {e}")

    def start(self):
        """Start the watchdog observer and the periodic network share polling worker."""
        if self._running:
            return

        self._running = True

        # Perform initial scan synchronously to ensure clean state
        self.scan_existing_files()

        # Start Watchdog observer (works on local file systems)
        try:
            event_handler = RawPhotoEventHandler(self.db_manager, self.face_engine)
            self.observer = Observer()
            self.observer.schedule(event_handler, self.raw_dir, recursive=True)
            self.observer.daemon = True
            self.observer.start()
            logger.info(f"Watchdog folder monitor active on: {self.raw_dir}")
        except Exception as e:
            logger.warning(f"Could not initialize inotify observer ({e}). Relying on polling worker.")

        # Start background polling thread (essential for CIFS / Samba network mounts)
        self._polling_thread = threading.Thread(target=self._polling_worker, daemon=True, name="SharePollingWorker")
        self._polling_thread.start()

    def stop(self):
        """Stop both the watchdog observer and polling worker."""
        if not self._running:
            return

        self._running = False

        if self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=3.0)
            except Exception:
                pass

        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=3.0)

        logger.info("Watchdog and polling monitors stopped cleanly.")


if __name__ == "__main__":
    import sys
    raw_directory = os.getenv("RAW_DIR", "/app/data/raw")
    db_directory = os.getenv("DB_DIR", "/app/data/db")

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting standalone Photo Indexer daemon...")

    db = DatabaseManager(db_dir=db_directory)
    engine = FaceEngine(model=os.getenv("FACE_DETECTION_MODEL", "hog"))
    indexer = PhotoIndexer(raw_dir=raw_directory, db_manager=db, face_engine=engine)

    indexer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        indexer.stop()
        logger.info("Indexer exited cleanly.")
