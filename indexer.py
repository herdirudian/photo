"""
indexer.py - Real-time folder monitoring and automated face vector extraction using Watchdog.
Monitors the raw theme park photos directory and updates SQLite + FAISS.
"""

import os
import time
import logging
import threading
import queue
from typing import Set, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent, FileDeletedEvent

from database import DatabaseManager, FaceEngine

# Configure logging
logger = logging.getLogger("PhotoRetrieval.Indexer")

SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def is_valid_image(file_path: str) -> bool:
    """Check if the file has a supported image extension."""
    _, ext = os.path.splitext(file_path)
    return ext.lower() in SUPPORTED_EXTENSIONS


def wait_for_file_transfer_complete(file_path: str, max_wait_sec: float = 6.0, poll_interval: float = 0.3) -> bool:
    """
    Ensure the file is completely written to disk before attempting computer vision extraction.
    Crucial for theme park network shares (Samba) or camera offloads where files arrive in chunks.
    Fast path: If the file is already complete (> 1KB with valid image headers & EOF), returns True immediately!
    """
    if not os.path.exists(file_path):
        return False

    try:
        sz = os.path.getsize(file_path)
        if sz > 1024:
            with open(file_path, "rb") as f:
                head = f.read(4)
                f.seek(max(0, sz - 4))
                tail = f.read(4)
                # Valid JPEG: starts with \xff\xd8 and ends with \xff\xd9
                if head.startswith(b"\xff\xd8") and tail.endswith(b"\xff\xd9"):
                    return True
                # Valid PNG: starts with \x89PNG and ends with IEND...
                if head.startswith(b"\x89PNG") and b"IEND" in tail:
                    return True
    except (OSError, IOError):
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

    def __init__(self, db_manager: DatabaseManager, face_engine: FaceEngine, indexer: Optional['PhotoIndexer'] = None):
        super().__init__()
        self.db_manager = db_manager
        self.face_engine = face_engine
        self.indexer = indexer
        self._processing_lock = threading.Lock()

    def process_file(self, file_path: str, force: bool = False):
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

            # Check if file has already been indexed (unless forced)
            if not force and self.db_manager.is_photo_indexed(file_path, file_mtime, file_size):
                logger.debug(f"Skipping already indexed file: {file_path}")
                return

            logger.info(f"Extracting faces from photo: {os.path.basename(file_path)}")
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

    def process_directory(self, dir_path: str, force: bool = False):
        """
        Recursively scan and index all valid photos in a newly created or moved directory.
        """
        if not os.path.isdir(dir_path):
            return
        logger.info(f"Scanning directory recursively for photos: {dir_path}")
        time.sleep(0.5)
        for root, _, files in os.walk(dir_path, followlinks=True):
            for filename in sorted(files):
                full_path = os.path.join(root, filename)
                if is_valid_image(full_path):
                    self.process_file(full_path, force=force)

    def on_created(self, event):
        if self.indexer:
            self.indexer.enqueue_path(event.src_path)
        else:
            if event.is_directory:
                threading.Thread(
                    target=self.process_directory,
                    args=(event.src_path,),
                    daemon=True,
                    name=f"DirIndex-{os.path.basename(event.src_path)}"
                ).start()
            else:
                self.process_file(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            self.db_manager.delete_folder(event.src_path)
            if self.indexer:
                self.indexer.enqueue_path(event.dest_path)
            else:
                threading.Thread(
                    target=self.process_directory,
                    args=(event.dest_path,),
                    daemon=True,
                    name=f"DirMove-{os.path.basename(event.dest_path)}"
                ).start()
        else:
            self.db_manager.delete_photo_by_path(event.src_path)
            if self.indexer:
                self.indexer.enqueue_file(event.dest_path)
            else:
                self.process_file(event.dest_path)

    def on_deleted(self, event):
        if event.is_directory:
            logger.info(f"Directory deleted on disk: {event.src_path}. Purging database records...")
            self.db_manager.delete_folder(event.src_path)
        else:
            logger.info(f"File deleted on disk: {event.src_path}. Purging database record...")
            self.db_manager.delete_photo_by_path(event.src_path)


class PhotoIndexer:
    """
    Manages directory scanning, Watchdog observer lifecycle, real-time ingestion queue,
    and periodic network share polling.
    """

    def __init__(self, raw_dir: str, db_manager: DatabaseManager, face_engine: FaceEngine, polling_interval: float = 3.0):
        self.raw_dir = raw_dir
        self.db_manager = db_manager
        self.face_engine = face_engine
        self.polling_interval = polling_interval
        self.observer: Optional[Observer] = None
        self._polling_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._scan_lock = threading.Lock()
        
        # Real-time ingestion queue
        self._queue = queue.Queue()
        self._enqueued_paths: Set[str] = set()
        self._queue_lock = threading.Lock()
        self._is_worker_busy = False

        # Real-time progress metrics
        self.is_indexing = False
        self.current_processed = 0
        self.total_to_process = 0
        self.newly_indexed_count = 0
        self.current_file = ""
        self.last_completed_at = None

        self._handler = RawPhotoEventHandler(self.db_manager, self.face_engine, indexer=self)

        os.makedirs(self.raw_dir, exist_ok=True)

    def enqueue_path(self, path: str, force: bool = False):
        """Enqueue either a single file or a folder of files."""
        if os.path.isdir(path):
            self.enqueue_directory(path, force=force)
        else:
            self.enqueue_file(path, force=force)

    def enqueue_file(self, file_path: str, force: bool = False):
        """Enqueue an image file into the background processing queue."""
        if not is_valid_image(file_path):
            return

        abs_path = os.path.abspath(file_path)
        with self._queue_lock:
            if abs_path in self._enqueued_paths:
                return
            self._enqueued_paths.add(abs_path)
            self._queue.put((abs_path, force))
            self.is_indexing = True
            self.total_to_process = max(self.total_to_process, self._queue.qsize() + (1 if self._is_worker_busy else 0))

        logger.info(f"Enqueued photo: {os.path.basename(abs_path)} (Queue length: {self._queue.qsize()})")

    def enqueue_directory(self, dir_path: str, force: bool = False):
        """Recursively discover and enqueue photos from a directory."""
        if not os.path.isdir(dir_path):
            return
        logger.info(f"Scanning directory to enqueue photos: {dir_path}")
        time.sleep(0.3)
        for root, _, files in os.walk(dir_path, followlinks=True):
            for filename in sorted(files):
                full_path = os.path.join(root, filename)
                if is_valid_image(full_path):
                    self.enqueue_file(full_path, force=force)

    def enqueue_unindexed_files(self):
        """
        Fast scan that discovers any photos on disk that haven't been indexed in DB
        and enqueues them. Guarantees that files copied over Samba / CIFS are picked up
        even if kernel inotify events were missed.
        """
        if not os.path.exists(self.raw_dir):
            return

        for root, _, files in os.walk(self.raw_dir, followlinks=True):
            for filename in files:
                full_path = os.path.abspath(os.path.join(root, filename))
                if is_valid_image(full_path):
                    with self._queue_lock:
                        if full_path in self._enqueued_paths:
                            continue
                    try:
                        stat = os.stat(full_path)
                        if not self.db_manager.is_photo_indexed(full_path, stat.st_mtime, stat.st_size):
                            self.enqueue_file(full_path)
                    except OSError:
                        pass

    def _queue_worker(self):
        """Dedicated background thread that drains the photo ingestion queue."""
        logger.info("Photo ingestion queue worker started.")
        while self._running:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                with self._queue_lock:
                    if self._queue.empty() and not self._is_worker_busy:
                        if self.is_indexing and not self._scan_lock.locked():
                            self.is_indexing = False
                            self.current_file = ""
                            self.total_to_process = 0
                            self.current_processed = 0
                            self.last_completed_at = time.time()
                continue

            file_path, force = item
            self._is_worker_busy = True
            self.is_indexing = True
            self.current_file = os.path.basename(file_path)

            try:
                self._handler.process_file(file_path, force=force)
                self.current_processed += 1
                self.newly_indexed_count += 1
            except Exception as e:
                logger.error(f"Error processing {file_path} from queue: {e}")
            finally:
                with self._queue_lock:
                    self._enqueued_paths.discard(file_path)
                    self._is_worker_busy = False
                    if self._queue.empty() and not self._scan_lock.locked():
                        self.is_indexing = False
                        self.current_file = ""
                        self.total_to_process = 0
                        self.current_processed = 0
                        self.last_completed_at = time.time()
                self._queue.task_done()

    def get_progress(self) -> dict:
        """Return real-time progress of ongoing directory indexing and queue metrics."""
        with self._queue_lock:
            q_size = self._queue.qsize()
            is_busy = bool(self._is_worker_busy or self.is_indexing or (q_size > 0) or self._scan_lock.locked())
            pending = q_size + (1 if self._is_worker_busy else 0)
        return {
            "is_indexing": is_busy,
            "queue_size": q_size,
            "pending_in_queue": pending,
            "current_processed": self.current_processed,
            "total_to_process": self.total_to_process,
            "newly_indexed": self.newly_indexed_count,
            "current_file": self.current_file,
            "last_completed_at": self.last_completed_at
        }

    def scan_existing_files(self, force: bool = False, target_subpath: Optional[str] = None) -> dict:
        """
        Scan the raw directory recursively (or a targeted subfolder if specified):
        1. Purge missing/stale photos and invalid pole/tree face records from SQLite & FAISS.
        2. Index newly discovered photos across all ride subfolders at any nested depth.
        Returns a dictionary of scan metrics.
        """
        if not self._scan_lock.acquire(blocking=False):
            logger.info("Scan already in progress. Skipping concurrent request.")
            return {"status": "busy"}

        self.is_indexing = True
        self.current_processed = 0
        self.total_to_process = 0
        self.newly_indexed_count = 0

        try:
            if target_subpath and target_subpath.strip():
                clean_target = target_subpath.strip().strip(r"\/").replace("\\", "/")
                scan_dir = os.path.abspath(os.path.join(self.raw_dir, clean_target))
            else:
                clean_target = None
                scan_dir = self.raw_dir

            logger.info(f"Starting directory scan on: {scan_dir} (force={force})")

            # Step 1: Self-heal moved photos, purge deleted photos, and clean invalid face detections
            purge_stats = self.db_manager.purge_missing_photos(self.raw_dir)
            if purge_stats["purged_photos"] > 0:
                logger.info(f"Cleaned up {purge_stats['purged_photos']} missing photos.")
            if purge_stats.get("purged_invalid_faces", 0) > 0:
                logger.info(f"Purged {purge_stats['purged_invalid_faces']} invalid face boxes (poles/trees).")
            if purge_stats["updated_paths"] > 0:
                logger.info(f"Self-healed paths for {purge_stats['updated_paths']} relocated photos.")

            # Step 2: Index new or updated files
            handler = self._handler
            scanned_count = 0
            new_indexed_count = 0

            if os.path.exists(scan_dir):
                files_to_check = []
                for root, _, files in os.walk(scan_dir, followlinks=True):
                    for filename in files:
                        full_path = os.path.join(root, filename)
                        if is_valid_image(full_path):
                            files_to_check.append(full_path)

                self.total_to_process = len(files_to_check)

                for full_path in files_to_check:
                    scanned_count += 1
                    self.current_file = os.path.basename(full_path)
                    try:
                        stat = os.stat(full_path)
                        if force or not self.db_manager.is_photo_indexed(full_path, stat.st_mtime, stat.st_size):
                            handler.process_file(full_path, force=force)
                            new_indexed_count += 1
                            self.newly_indexed_count = new_indexed_count
                    except Exception as e:
                        logger.error(f"Error checking {full_path}: {e}")
                    self.current_processed += 1

            summary = {
                "scanned": scanned_count,
                "newly_indexed": new_indexed_count,
                "purged_photos": purge_stats.get("purged_photos", 0),
                "purged_faces": purge_stats.get("purged_faces", 0),
                "purged_invalid_faces": purge_stats.get("purged_invalid_faces", 0),
                "updated_paths": purge_stats.get("updated_paths", 0),
                "target": clean_target or "all"
            }
            logger.info(f"Scan complete. Scanned: {scanned_count}, Processed: {new_indexed_count}, Purged faces: {summary['purged_faces']}")
            return summary

        finally:
            self.is_indexing = False
            self.current_file = ""
            self.last_completed_at = time.time()
            self._scan_lock.release()

    def _polling_worker(self):
        """
        Background worker that periodically polls the network share (CIFS/Samba)
        to detect newly added photos or cleaned up files without relying on kernel inotify.
        """
        logger.info(f"Network share periodic polling worker started (interval: {self.polling_interval}s).")
        
        # Initial scan runs in this background thread so Uvicorn web server starts immediately!
        try:
            self.scan_existing_files()
        except Exception as e:
            logger.error(f"Error in initial background directory scan: {e}")

        while self._running:
            # Sleep in small steps to react quickly to shutdown
            for _ in range(max(1, int(self.polling_interval))):
                if not self._running:
                    return
                time.sleep(1)

            try:
                self.enqueue_unindexed_files()
            except Exception as e:
                logger.error(f"Error in background polling scanner: {e}")

    def start(self):
        """Start the watchdog observer, queue worker, and periodic network share polling worker."""
        if self._running:
            return

        self._running = True

        # Start ingestion queue worker
        self._worker_thread = threading.Thread(target=self._queue_worker, daemon=True, name="IndexerQueueWorker")
        self._worker_thread.start()

        # Start Watchdog observer (works on local file systems)
        try:
            self.observer = Observer()
            self.observer.schedule(self._handler, self.raw_dir, recursive=True)
            self.observer.daemon = True
            self.observer.start()
            logger.info(f"Watchdog folder monitor active on: {self.raw_dir}")
        except Exception as e:
            logger.warning(f"Could not initialize inotify observer ({e}). Relying on polling worker.")

        # Start background polling thread (essential for CIFS / Samba network mounts)
        self._polling_thread = threading.Thread(target=self._polling_worker, daemon=True, name="SharePollingWorker")
        self._polling_thread.start()

    def stop(self):
        """Stop observer, queue worker, and polling worker cleanly."""
        if not self._running:
            return

        self._running = False

        if self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=2.0)
            except Exception:
                pass

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=2.0)

        logger.info("Watchdog, queue worker, and polling monitors stopped cleanly.")


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
