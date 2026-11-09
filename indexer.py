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
    """Check if the file has a supported image extension and is not a hidden or system metadata file."""
    base = os.path.basename(file_path)
    if base.startswith(".") or base.startswith("~") or base.startswith("Thumbs"):
        return False
    _, ext = os.path.splitext(base)
    return ext.lower() in SUPPORTED_EXTENSIONS


def is_image_file_ready(file_path: str) -> bool:
    """
    Sub-millisecond verification if an image file has finished transferring and is ready to read.
    """
    try:
        stat = os.stat(file_path)
        sz = stat.st_size
        if sz <= 1024:
            return False

        # If file was modified more than 1.5 seconds ago, it is already completely written
        if (time.time() - stat.st_mtime) >= 1.5:
            return True

        with open(file_path, "rb") as f:
            head = f.read(16)
            if head.startswith(b"\xff\xd8"):
                # JPEG: read last 2048 bytes to find \xff\xd9 EOI marker (handles camera trailers/padding)
                f.seek(max(0, sz - 2048))
                tail = f.read(2048)
                if b"\xff\xd9" in tail:
                    return True
            elif head.startswith(b"\x89PNG"):
                # PNG: check IEND in last 64 bytes
                f.seek(max(0, sz - 64))
                tail = f.read(64)
                if b"IEND" in tail:
                    return True
            elif head.startswith(b"RIFF") and b"WEBP" in head:
                return True
    except (OSError, IOError):
        pass
    return False


def wait_for_file_transfer_complete(file_path: str, max_wait_sec: float = 3.0, poll_interval: float = 0.2) -> bool:
    """
    Ensure the file is completely written to disk before attempting computer vision extraction.
    Crucial for theme park network shares (Samba) or camera offloads where files arrive in chunks.
    Fast path: If the file is already complete (> 1KB with valid image headers & EOF), returns True immediately!
    """
    if not os.path.exists(file_path):
        return False

    if is_image_file_ready(file_path):
        return True

    start_time = time.time()
    last_size = -1

    while time.time() - start_time < max_wait_sec:
        if not os.path.exists(file_path):
            return False

        try:
            stat = os.stat(file_path)
            current_size = stat.st_size
        except OSError:
            time.sleep(poll_interval)
            continue

        if current_size == 0 and (time.time() - stat.st_mtime > 3.0):
            return False

        if is_image_file_ready(file_path):
            return True

        if current_size > 1024 and current_size == last_size:
            return True

        last_size = current_size
        time.sleep(poll_interval)

    return is_image_file_ready(file_path)


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
        is_ready = wait_for_file_transfer_complete(file_path)

        if not is_ready:
            # If the file exists on disk but is 0 KB or unreadable, register as empty_file
            # so the database accounts for it and does not enter an infinite indexing retry loop.
            try:
                if os.path.exists(file_path):
                    stat = os.stat(file_path)
                    if stat.st_size == 0:
                        logger.warning(f"File {os.path.basename(file_path)} is 0 KB on disk. Registered as empty_file.")
                        self.db_manager.register_photo(
                            file_path=file_path,
                            file_mtime=stat.st_mtime,
                            file_size=0,
                            faces=[],
                            status="empty_file"
                        )
                    else:
                        logger.info(f"File {os.path.basename(file_path)} still transferring over network/Samba. Re-queueing in 1s...")
                        if self.indexer:
                            threading.Timer(1.0, self.indexer.enqueue_file, args=[file_path]).start()
            except Exception as e:
                logger.debug(f"Could not register incomplete file status: {e}")
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

    def on_modified(self, event):
        if not event.is_directory and is_valid_image(event.src_path):
            if self.indexer:
                self.indexer.enqueue_file(event.src_path)
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
        self.num_workers = int(os.getenv("INDEXER_WORKERS", "4"))
        self._worker_threads = []
        self._running = False
        self._scan_lock = threading.Lock()
        self._enqueue_lock = threading.Lock()
        
        # Real-time ingestion queue
        self._queue = queue.Queue()
        self._enqueued_paths: Set[str] = set()
        self._queue_lock = threading.Lock()
        self._active_workers = 0

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
            self.total_to_process = max(self.total_to_process, self._queue.qsize() + self._active_workers)

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
        Uses in-memory indexed metadata mapping to avoid thousands of individual SQLite queries.
        """
        if not os.path.exists(self.raw_dir):
            return

        if not self._enqueue_lock.acquire(blocking=False):
            return  # Scan already in progress; avoid overlapping disk walks

        try:
            try:
                indexed_map = self.db_manager.get_all_indexed_metadata()
            except Exception as e:
                logger.debug(f"Failed to load indexed metadata cache: {e}")
                indexed_map = {}

            to_enqueue = []
            for root, _, files in os.walk(self.raw_dir, followlinks=True):
                for filename in files:
                    full_path = os.path.abspath(os.path.join(root, filename))
                    if is_valid_image(full_path):
                        with self._queue_lock:
                            if full_path in self._enqueued_paths:
                                continue
                        try:
                            stat = os.stat(full_path)
                            cached = indexed_map.get(full_path)
                            if cached is not None and abs(cached[0] - stat.st_mtime) < 1e-4 and cached[1] == stat.st_size:
                                continue
                            to_enqueue.append(full_path)
                        except OSError:
                            pass

            for p in to_enqueue:
                self.enqueue_file(p)
        finally:
            self._enqueue_lock.release()

    def _queue_worker(self):
        """Dedicated background thread worker that drains the photo ingestion queue."""
        thread_name = threading.current_thread().name
        logger.info(f"Photo ingestion worker [{thread_name}] started.")
        while self._running:
            # Yield CPU to real-time search when customer is searching at cashier counter
            while self.face_engine.is_search_active and self._running:
                time.sleep(0.05)

            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                with self._queue_lock:
                    if self._queue.empty() and self._active_workers == 0:
                        if self.is_indexing and not self._scan_lock.locked():
                            self.is_indexing = False
                            self.current_file = ""
                            self.total_to_process = 0
                            self.current_processed = 0
                            self.last_completed_at = time.time()
                            self.db_manager.flush_faiss()
                continue

            file_path, force = item
            with self._queue_lock:
                self._active_workers += 1
                self.is_indexing = True
                self.current_file = os.path.basename(file_path)

            try:
                self._handler.process_file(file_path, force=force)
                with self._queue_lock:
                    self.current_processed += 1
                    self.newly_indexed_count += 1
            except Exception as e:
                logger.error(f"Error processing {file_path} from queue in [{thread_name}]: {e}")
            finally:
                with self._queue_lock:
                    self._enqueued_paths.discard(file_path)
                    self._active_workers = max(0, self._active_workers - 1)
                    if self._queue.empty() and self._active_workers == 0 and not self._scan_lock.locked():
                        self.is_indexing = False
                        self.current_file = ""
                        self.total_to_process = 0
                        self.current_processed = 0
                        self.last_completed_at = time.time()
                        self.db_manager.flush_faiss()
                self._queue.task_done()

    def get_progress(self) -> dict:
        """Return real-time progress of ongoing directory indexing and queue metrics."""
        with self._queue_lock:
            q_size = self._queue.qsize()
            is_busy = bool((self._active_workers > 0) or self.is_indexing or (q_size > 0) or self._scan_lock.locked())
            pending = q_size + self._active_workers
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

                files_to_index = []
                for full_path in files_to_check:
                    scanned_count += 1
                    try:
                        stat = os.stat(full_path)
                        if force or not self.db_manager.is_photo_indexed(full_path, stat.st_mtime, stat.st_size):
                            files_to_index.append((full_path, stat))
                    except Exception as e:
                        logger.error(f"Error checking {full_path}: {e}")

                self.total_to_process = len(files_to_index)
                self.current_processed = 0

                if files_to_index:
                    from concurrent.futures import ThreadPoolExecutor, as_completed

                    def _scan_worker_task(item):
                        f_path, _ = item
                        self.current_file = os.path.basename(f_path)
                        handler.process_file(f_path, force=force)
                        return f_path

                    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                        futures = [executor.submit(_scan_worker_task, item) for item in files_to_index]
                        for future in as_completed(futures):
                            try:
                                future.result()
                                new_indexed_count += 1
                                self.newly_indexed_count = new_indexed_count
                            except Exception as e:
                                logger.error(f"Error processing file in parallel scan: {e}")
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
            self.db_manager.flush_faiss()
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

            if self.face_engine.is_search_active:
                continue

            try:
                self.enqueue_unindexed_files()
            except Exception as e:
                logger.error(f"Error in background polling scanner: {e}")

    def start(self):
        """Start the watchdog observer, queue workers pool, and periodic network share polling worker."""
        if self._running:
            return

        self._running = True

        # Start ingestion queue worker pool (utilizes multiple Xeon CPU threads)
        self._worker_threads = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self._queue_worker, daemon=True, name=f"IndexerWorker-{i+1}")
            t.start()
            self._worker_threads.append(t)
        logger.info(f"Started {self.num_workers} photo ingestion workers.")

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

        for t in self._worker_threads:
            if t and t.is_alive():
                t.join(timeout=1.5)

        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=1.5)

        self.db_manager.flush_faiss()
        logger.info("Watchdog, queue workers, and polling monitors stopped cleanly.")


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
