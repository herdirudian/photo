"""
database.py - SQLite persistence and FAISS vector similarity search for face retrieval.
"""

import os
import io
import re
import csv
import time
import datetime
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

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    FaceAnalysis = None
    logger.info("InsightFace library not yet installed. dlib will serve as fallback engine.")

try:
    import onnxruntime
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

try:
    import multiprocessing
    # Configure OpenCV thread pool to optimize throughput without oversubscribing Xeon cores
    _cpu_cnt = multiprocessing.cpu_count()
    cv2.setNumThreads(min(4, max(1, _cpu_cnt // 2)))
except Exception:
    pass


def read_image_exif_aware(img_source: Any) -> Optional[np.ndarray]:
    """
    Decodes an image file path or bytes into a BGR numpy array while honoring EXIF orientation.
    DSLR cameras frequently record portrait photos with EXIF Orientation tags (3, 6, or 8).
    Fast path: Uses libjpeg-turbo C++ cv2.imread / cv2.imdecode and rotates directly in C++,
    avoiding slow Pillow full-image RGB conversions and reducing memory pressure.
    """
    try:
        if isinstance(img_source, (str, os.PathLike)):
            orientation = 1
            try:
                with Image.open(img_source) as pil_img:
                    exif = pil_img.getexif()
                    if exif:
                        orientation = exif.get(0x0112, 1)
            except Exception:
                pass

            arr = cv2.imread(str(img_source), cv2.IMREAD_COLOR)
            if arr is not None:
                if orientation == 1:
                    return arr
                elif orientation == 3:
                    return cv2.rotate(arr, cv2.ROTATE_180)
                elif orientation == 6:
                    return cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
                elif orientation == 8:
                    return cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                return arr

        elif isinstance(img_source, (bytes, bytearray)):
            orientation = 1
            try:
                with Image.open(io.BytesIO(img_source)) as pil_img:
                    exif = pil_img.getexif()
                    if exif:
                        orientation = exif.get(0x0112, 1)
            except Exception:
                pass

            nparr = np.frombuffer(img_source, np.uint8)
            arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if arr is not None:
                if orientation == 1:
                    return arr
                elif orientation == 3:
                    return cv2.rotate(arr, cv2.ROTATE_180)
                elif orientation == 6:
                    return cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
                elif orientation == 8:
                    return cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                return arr
    except Exception:
        pass

    try:
        if isinstance(img_source, (str, os.PathLike)):
            return cv2.imread(str(img_source), cv2.IMREAD_COLOR)
        elif isinstance(img_source, (bytes, bytearray)):
            nparr = np.frombuffer(img_source, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None


WIB = datetime.timezone(datetime.timedelta(hours=7))


def get_now_wib() -> datetime.datetime:
    """Return current timestamp in Western Indonesia Time (WIB, UTC+7)."""
    return datetime.datetime.now(datetime.timezone.utc).astimezone(WIB)


def parse_datetime_flexible(val: Any) -> Optional[datetime.datetime]:
    """Parse various ISO and datetime formats into a Western Indonesia Time (WIB, UTC+7) datetime object."""
    if not val:
        return None
    if isinstance(val, datetime.datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=WIB)
        return val.astimezone(WIB)
    s = str(val).strip()
    if not s or s.lower() in ("none", "null"):
        return None
    try:
        if s.endswith("Z"):
            dt = datetime.datetime.fromisoformat(s[:-1] + "+00:00")
            return dt.astimezone(WIB)
        if "+" in s or ("-" in s[10:] and len(s) > 19):
            dt = datetime.datetime.fromisoformat(s)
            return dt.astimezone(WIB)
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d"):
            try:
                dt = datetime.datetime.strptime(s, fmt)
                return dt.replace(tzinfo=WIB)
            except ValueError:
                pass
        return datetime.datetime.fromisoformat(s).replace(tzinfo=WIB)
    except Exception:
        return None


def format_dwell_time(seconds: int) -> str:
    """Format dwell time duration into clear Indonesian time string."""
    if seconds <= 0:
        return "Baru Masuk (< 1 Menit)"
    total_minutes = seconds // 60
    if total_minutes < 1:
        return "< 1 Menit"
    hours = total_minutes // 60
    mins = total_minutes % 60
    if hours > 0:
        if mins > 0:
            return f"{hours} Jam {mins} Menit"
        return f"{hours} Jam"
    return f"{mins} Menit"


def mask_phone_number(phone: Optional[str]) -> str:
    """Mask guest phone number for privacy display."""
    if not phone:
        return ""
    digits = re.sub(r"\D", "", str(phone).strip())
    if len(digits) >= 8:
        return f"{digits[:4]}****{digits[-4:]}"
    elif len(digits) >= 4:
        return f"{digits[:2]}**{digits[-2:]}"
    return digits


class FaceEngine:
    """
    Unified Biometric Face Recognition Engine.

    Engines supported:
    1. 'insightface' (ArcFace / RetinaFace via ONNX Runtime - 512-Dimensional Vector):
       - Global biometric industry standard.
       - Outstanding resistance against sunglasses, masks, dappled tree shadows,
         and extreme head poses up to 60 degrees.
       - Model packs: 'buffalo_l' (ResNet-50 ArcFace) or 'buffalo_sc' (MobileFaceNet ArcFace).
    2. 'dlib' (dlib ResNet-34 - 128-Dimensional Vector):
       - Ultra-low CPU fallback for legacy servers.
    """

    def __init__(
        self,
        engine: str = "insightface",
        model: str = "buffalo_l",
        model_root: Optional[str] = None,
        det_thresh: float = 0.50,
        det_size: Tuple[int, int] = (1280, 1280)
    ):
        self.requested_engine = (engine or "insightface").lower()
        self.model_name = model or "buffalo_l"
        self.det_thresh = det_thresh
        self.det_size = det_size

        # Model root path for offline cache
        if not model_root:
            base_dir = os.getenv("DB_DIR", "./data/db")
            data_parent = os.path.dirname(os.path.abspath(base_dir))
            self.model_root = os.getenv("INSIGHTFACE_ROOT", os.path.join(data_parent, "models", "insightface"))
        else:
            self.model_root = model_root

        os.makedirs(self.model_root, exist_ok=True)

        self.engine_name = "insightface"
        self.dimension = 512
        self.insight_app = None
        self.model = "hog"
        self.inference_lock = threading.RLock()

        if self.requested_engine == "insightface":
            if INSIGHTFACE_AVAILABLE and FaceAnalysis is not None:
                try:
                    logger.info(f"Initializing InsightFace ArcFace Engine (model: {self.model_name}, root: {self.model_root})...")
                    providers = ['CPUExecutionProvider']
                    if ONNXRUNTIME_AVAILABLE:
                        try:
                            available_p = onnxruntime.get_available_providers()
                            if 'CUDAExecutionProvider' in available_p:
                                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                        except Exception:
                            pass

                    try:
                        self.insight_app = FaceAnalysis(
                            name=self.model_name,
                            root=self.model_root,
                            providers=providers,
                            allowed_modules=['detection', 'recognition']
                        )
                    except TypeError:
                        self.insight_app = FaceAnalysis(
                            name=self.model_name,
                            root=self.model_root,
                            providers=providers
                        )
                    ctx_id = 0 if ('CUDAExecutionProvider' in providers) else -1
                    self.insight_app.prepare(ctx_id=ctx_id, det_size=self.det_size)
                    self.engine_name = "insightface"
                    self.dimension = 512
                    logger.info(f"InsightFace ArcFace 512-D initialized successfully with {providers[0]}")
                except Exception as e:
                    logger.warning(f"Could not load InsightFace ({e}). Falling back to dlib ResNet-34 128-D.")
                    self._init_dlib_fallback(model)
            else:
                logger.warning("InsightFace/ONNXRuntime not installed. Falling back to dlib ResNet-34 128-D.")
                self._init_dlib_fallback(model)
        else:
            self._init_dlib_fallback(model)

    def _init_dlib_fallback(self, model: str):
        self.engine_name = "dlib"
        self.dimension = 128
        self.model = model if model in ["hog", "cnn"] else "hog"
        self.insight_app = None
        logger.info(f"Active Face Engine: dlib ResNet-34 (128-D, detector: {self.model})")

    def update_det_size(self, new_size: int):
        """Update detection resolution dynamically (e.g. 640 or 1280)."""
        new_size = int(new_size)
        self.det_size = (new_size, new_size)
        if self.insight_app is not None:
            try:
                with self.inference_lock:
                    self.insight_app.prepare(ctx_id=-1, det_size=self.det_size)
                logger.info(f"InsightFace det_size successfully updated to {self.det_size}")
            except Exception as e:
                logger.warning(f"Could not re-prepare InsightFace det_size: {e}")

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
        if crop.size == 0:
            return np.zeros(256, dtype=np.float32)
        
        # Convert BGR crop directly to HSV (avoiding massive full-image RGB conversion)
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Compute 2D histogram of Hue and Saturation
        # Hue range: 0-180, Saturation: 0-256
        hist = cv2.calcHist([hsv_crop], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        return hist.flatten().astype(np.float32)

    def extract_faces_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract bounding boxes and facial embeddings from an image file on disk.
        Routes to InsightFace (512-D ArcFace) or dlib (128-D ResNet) based on active engine.
        Guarantees thread-safe inference when background indexing and search execute concurrently.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image not found at {file_path}")

        # Decode JPEG/PNG image from disk OUTSIDE the inference lock
        # so multiple background workers decompress images in parallel!
        image_bgr = read_image_exif_aware(file_path)
        if image_bgr is None:
            logger.error(f"Failed to read image file {file_path}")
            return []

        if self.engine_name == "insightface" and self.insight_app is not None:
            return self._extract_insightface_file(file_path, image_bgr=image_bgr)
        with self.inference_lock:
            return self._extract_dlib_file(file_path)

    def _extract_insightface_file(self, file_path: str, image_bgr: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        High-Performance 512-Dimensional ArcFace embedding extraction using ONNX Runtime.
        Highly resilient against sunglasses, hats, masks, dappled tree shadows,
        and extreme face pose angles up to 60 degrees.
        """
        if image_bgr is None:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Image not found at {file_path}")
            image_bgr = read_image_exif_aware(file_path)
            if image_bgr is None:
                logger.error(f"Failed to read image file {file_path}")
                return []

        orig_h, orig_w = image_bgr.shape[:2]
        max_dim = max(orig_h, orig_w)
        # Fast bilinear interpolation (cv2.INTER_LINEAR) runs OUTSIDE inference lock!
        if max_dim > 3200:
            scale_factor = 3200.0 / float(max_dim)
            proc_w = int(round(orig_w * scale_factor))
            proc_h = int(round(orig_h * scale_factor))
            proc_bgr = cv2.resize(image_bgr, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        else:
            scale_factor = 1.0
            proc_bgr = image_bgr
            proc_h, proc_w = orig_h, orig_w

        h, w = proc_h, proc_w
        all_raw_faces = []

        # Critical AI Section: Lock is held ONLY during ONNX Runtime model inference!
        with self.inference_lock:
            # 1. Main Global Detection Pass
            try:
                global_faces = self.insight_app.get(proc_bgr)
                all_raw_faces.extend(global_faces)
            except Exception as e:
                logger.error(f"InsightFace inference error on {file_path}: {e}")
                return []

            # Multi-Orientation Fallback:
            # Only run if zero faces were detected at 0°!
            # If faces were already found upright, rotation is completely unnecessary.
            if not all_raw_faces:
                for rot_code, rot_name in [
                    (cv2.ROTATE_90_CLOCKWISE, "90° CW"),
                    (cv2.ROTATE_90_COUNTERCLOCKWISE, "270° CW")
                ]:
                    try:
                        rotated_bgr = cv2.rotate(proc_bgr, rot_code)
                        rot_faces = self.insight_app.get(rotated_bgr)
                        valid_rot_faces = [f for f in rot_faces if getattr(f, "det_score", 0.0) >= 0.52]
                        if valid_rot_faces:
                            logger.info(
                                f"Multi-Orientation Recovery: Found {len(valid_rot_faces)} face(s) in "
                                f"{os.path.basename(file_path)} after rotating {rot_name}."
                            )
                            for rf in valid_rot_faces:
                                bx1, by1, bx2, by2 = rf.bbox
                                if rot_code == cv2.ROTATE_90_CLOCKWISE:
                                    orig_x1 = by1
                                    orig_x2 = by2
                                    orig_y1 = h - 1 - bx2
                                    orig_y2 = h - 1 - bx1
                                elif rot_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
                                    orig_x1 = w - 1 - by2
                                    orig_x2 = w - 1 - by1
                                    orig_y1 = bx1
                                    orig_y2 = bx2

                                rf.bbox = np.array([
                                    min(orig_x1, orig_x2),
                                    min(orig_y1, orig_y2),
                                    max(orig_x1, orig_x2),
                                    max(orig_y1, orig_y2)
                                ])
                                all_raw_faces.append(rf)
                            break
                    except Exception as e_rot:
                        logger.debug(f"Rotation check error ({rot_name}): {e_rot}")

            # 2. Targeted Ride Zone Crop:
            # High-resolution DSLR balloon ride photos pack 6 guests in the basket (y in [0.60, 0.98], x in [0.15, 0.85]).
            # Optimize: Only execute crop inference if the photo belongs to a Balloon ride OR as fallback if 0 faces were found.
            p_lower = file_path.lower()
            is_balloon_ride = any(k in p_lower for k in ["balon", "balloon", "basket", "udara", "/bl/"])
            if is_balloon_ride and max(orig_h, orig_w) > 1800:
                by1, bx1, by2, bx2 = int(h * 0.60), int(w * 0.15), int(h * 0.98), int(w * 0.85)
                basket_crop = proc_bgr[by1:by2, bx1:bx2]
                if basket_crop.size > 0:
                    try:
                        basket_faces = self.insight_app.get(basket_crop)
                        for bf in basket_faces:
                            # Translate bbox to full frame coordinates
                            bx1_c, by1_c, bx2_c, by2_c = bf.bbox
                            bf.bbox = np.array([
                                bx1_c + bx1,
                                by1_c + by1,
                                bx2_c + bx1,
                                by2_c + by1
                            ])
                            all_raw_faces.append(bf)
                    except Exception:
                        pass

        # 3. Filter, validate, normalize, and extract upper-body color histograms OUTSIDE inference lock
        results = []
        for face in all_raw_faces:
            det_score = float(getattr(face, "det_score", 1.0))
            if det_score < self.det_thresh:
                continue

            bbox = face.bbox
            if scale_factor != 1.0:
                # Scale bbox coordinates back to full original image coordinate space
                orig_bbox = bbox / scale_factor
            else:
                orig_bbox = bbox

            x1, y1, x2, y2 = int(round(orig_bbox[0])), int(round(orig_bbox[1])), int(round(orig_bbox[2])), int(round(orig_bbox[3]))
            left = max(0, min(orig_w - 1, x1))
            top = max(0, min(orig_h - 1, y1))
            right = max(left + 1, min(orig_w, x2))
            bottom = max(top + 1, min(orig_h, y2))

            box_w = right - left
            box_h = bottom - top

            # Filter tiny noise (< 16px)
            if box_w < 16 or box_h < 16:
                continue

            # Aspect ratio check: accommodates natural head tilts on swings/rides
            ratio = float(box_h) / float(box_w)
            if ratio < 0.50 or ratio > 1.80:
                continue

            # Deduplicate overlapping detections via IoU check
            box_tuple = (top, right, bottom, left)
            overlap = False
            for sf in results:
                if self._compute_iou(box_tuple, sf["bbox"]) > 0.40:
                    overlap = True
                    break
            if overlap:
                continue

            # Extract 512-d normalized embedding
            if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
                embedding = face.normed_embedding.astype(np.float32)
            else:
                raw_emb = face.embedding
                norm = np.linalg.norm(raw_emb)
                embedding = (raw_emb / norm if norm > 0 else raw_emb).astype(np.float32)

            if embedding.shape[0] != 512:
                continue

            # Torso color histogram directly cropped from image_bgr (no full-image RGB conversion)
            color_hist = self.compute_torso_color_hist(image_bgr, box_tuple)

            results.append({
                "bbox": box_tuple,
                "encoding": embedding,
                "color_hist": color_hist,
                "det_score": det_score,
                "pose": getattr(face, "pose", None)
            })

        return results

    def _extract_dlib_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract bounding boxes and 128-d facial embeddings using dlib ResNet-34.
        """
        if not FACE_REC_AVAILABLE:
            raise RuntimeError("face_recognition / dlib is not available in current environment")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image not found at {file_path}")

        try:
            image = face_recognition.load_image_file(file_path)
        except Exception as e:
            logger.error(f"Failed to read image file {file_path}: {e}")
            return []

        candidate_locations = self.detect_multiscale_locations(image)
        if not candidate_locations:
            return []

        valid_locations = []
        for loc in candidate_locations:
            if self.is_valid_human_face(image, loc):
                valid_locations.append(loc)
            else:
                logger.info(f"Filtered out non-human false detection at {loc} in {os.path.basename(file_path)} (pole/tree/equipment)")

        if not valid_locations:
            return []

        encodings = face_recognition.face_encodings(image, known_face_locations=valid_locations, num_jitters=1, model="large")
        results = []
        for loc, enc in zip(valid_locations, encodings):
            norm = np.linalg.norm(enc)
            normalized_enc = (enc / norm).astype(np.float32) if norm > 0 else enc.astype(np.float32)
            color_hist = self.compute_torso_color_hist(image, loc)
            results.append({
                "bbox": loc,
                "encoding": normalized_enc,
                "color_hist": color_hist
            })
        return results

    def extract_faces_from_bytes(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Extract face locations and encodings from in-memory bytes for guest reference selfies.
        Guarantees thread-safe inference when background indexing and search execute concurrently.
        """
        with self.inference_lock:
            if self.engine_name == "insightface" and self.insight_app is not None:
                return self._extract_insightface_bytes(image_bytes)
            return self._extract_dlib_bytes(image_bytes)

    def _extract_insightface_bytes(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Extract 512-d ArcFace embedding from in-memory selfie upload using InsightFace.
        """
        image_bgr = read_image_exif_aware(image_bytes)
        if image_bgr is None:
            return []

        h, w = image_bgr.shape[:2]
        with self.inference_lock:
            raw_faces = self.insight_app.get(image_bgr)

        # Fallback multi-orientation check for selfies taken sideways on smartphones
        if not raw_faces:
            for rot_code in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
                try:
                    rotated_bgr = cv2.rotate(image_bgr, rot_code)
                    with self.inference_lock:
                        rot_faces = self.insight_app.get(rotated_bgr)
                    if rot_faces:
                        image_bgr = rotated_bgr
                        h, w = image_bgr.shape[:2]
                        raw_faces = rot_faces
                        break
                except Exception:
                    pass

        # Fallback with relaxed det_thresh if tricky webcam lighting
        if not raw_faces and self.det_thresh > 0.35:
            with self.inference_lock:
                raw_faces = [f for f in self.insight_app.get(image_bgr) if getattr(f, "det_score", 1.0) >= 0.30]

        results = []
        for face in raw_faces:
            det_score = float(getattr(face, "det_score", 1.0))
            bbox = face.bbox
            x1, y1, x2, y2 = int(round(bbox[0])), int(round(bbox[1])), int(round(bbox[2])), int(round(bbox[3]))
            left = max(0, min(w - 1, x1))
            top = max(0, min(h - 1, y1))
            right = max(left + 1, min(w, x2))
            bottom = max(top + 1, min(h, y2))

            box_tuple = (top, right, bottom, left)

            if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
                embedding = face.normed_embedding.astype(np.float32)
            else:
                raw_emb = face.embedding
                norm = np.linalg.norm(raw_emb)
                embedding = (raw_emb / norm if norm > 0 else raw_emb).astype(np.float32)

            if embedding.shape[0] != 512:
                continue

            color_hist = self.compute_torso_color_hist(image_bgr, box_tuple)

            results.append({
                "bbox": box_tuple,
                "encoding": embedding,
                "color_hist": color_hist,
                "det_score": det_score,
                "pose": getattr(face, "pose", None)
            })

        return results

    def _extract_dlib_bytes(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract face locations and encodings from in-memory bytes using dlib ResNet-34."""
        if not FACE_REC_AVAILABLE:
            raise RuntimeError("face_recognition is not available in current environment")

        image_obj = Image.open(io.BytesIO(image_bytes))
        if image_obj.mode != "RGB":
            image_obj = image_obj.convert("RGB")
        
        image_np = np.array(image_obj)
        locations = face_recognition.face_locations(image_np, number_of_times_to_upsample=1, model=self.model)
        if not locations:
            locations = face_recognition.face_locations(image_np, number_of_times_to_upsample=2, model=self.model)

        if not locations:
            return []

        valid_locations = [loc for loc in locations if self.is_valid_human_face(image_np, loc)]
        if not valid_locations:
            valid_locations = [loc for loc in locations if 0.65 <= ((loc[2] - loc[0]) / max(1, loc[1] - loc[3])) <= 1.55]

        if not valid_locations:
            return []

        encodings = face_recognition.face_encodings(image_np, known_face_locations=valid_locations, num_jitters=2, model="large")
        results = []
        for loc, enc in zip(valid_locations, encodings):
            norm = np.linalg.norm(enc)
            normalized_enc = (enc / norm).astype(np.float32) if norm > 0 else enc.astype(np.float32)
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

    def __init__(self, db_dir: str = "/app/data/db", vector_dim: int = 512):
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, "metadata.db")
        self.faiss_index_path = os.path.join(self.db_dir, "faiss_index.bin")
        self.vector_dim = vector_dim

        self.lock = threading.RLock()
        self._faiss_dirty = False
        self._last_faiss_save_time = time.time()
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

                CREATE TABLE IF NOT EXISTS guest_visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    crm_guest_id TEXT,
                    guest_name TEXT NOT NULL,
                    guest_phone TEXT,
                    masked_phone TEXT,
                    registered_at TIMESTAMP,
                    claimed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    dwell_time_seconds INTEGER DEFAULT 0,
                    dwell_time_formatted TEXT,
                    photo_count INTEGER DEFAULT 0,
                    customer_folder TEXT,
                    date_key TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_guest_visits_date ON guest_visits(date_key);
                CREATE INDEX IF NOT EXISTS idx_guest_visits_crm_id ON guest_visits(crm_guest_id);
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
                    if self.index.d != self.vector_dim:
                        logger.warning(
                            f"FAISS index on disk has dimension {self.index.d}, but current engine uses {self.vector_dim}-D. "
                            f"Rebuilding fresh {self.vector_dim}-D index from SQLite..."
                        )
                        base_index = faiss.IndexFlatIP(self.vector_dim)
                        self.index = faiss.IndexIDMap2(base_index)
                        self._rebuild_faiss_from_sqlite()
                        return
                    else:
                        logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors ({self.index.d}-D).")
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
        mismatched_dim = 0
        for row in rows:
            v_id = row["vector_id"]
            blob = row["encoding_blob"]
            vec = np.frombuffer(blob, dtype=np.float32)
            if len(vec) == self.vector_dim:
                ids.append(v_id)
                vectors.append(vec)
            else:
                mismatched_dim += 1

        if mismatched_dim > 0:
            logger.info(
                f"Skipped {mismatched_dim} face vector(s) in SQLite with dimension != {self.vector_dim}-D. "
                "Trigger 'Pindai Ulang Sekarang' to re-extract with active AI engine."
            )

        if vectors:
            v_array = np.vstack(vectors).astype(np.float32)
            id_array = np.array(ids, dtype=np.int64)
            self.index.add_with_ids(v_array, id_array)
            logger.info(f"Rebuilt FAISS index with {len(ids)} vectors from database ({self.vector_dim}-D).")
            self._save_faiss()

    def _save_faiss(self):
        """Persist FAISS index to disk."""
        if FAISS_AVAILABLE and self.index is not None:
            try:
                faiss.write_index(self.index, self.faiss_index_path)
                self._faiss_dirty = False
            except Exception as e:
                logger.error(f"Error saving FAISS index: {e}")

    def flush_faiss(self):
        """Ensure any pending in-memory FAISS modifications are written to disk."""
        with self.lock:
            if getattr(self, "_faiss_dirty", False) and FAISS_AVAILABLE and self.index is not None:
                self._save_faiss()
                self._last_faiss_save_time = time.time()
                self._faiss_dirty = False
                logger.info("Flushed FAISS index to disk.")

    def is_photo_indexed(self, file_path: str, file_mtime: float, file_size: int) -> bool:
        """Check if a file has already been indexed with matching modification time, size, and vector dimension."""
        with self.lock, self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, file_mtime, file_size, num_faces, status FROM photos WHERE file_path = ?",
                (file_path,)
            )
            row = cursor.fetchone()
            if row:
                # If file previously had error or was 0 KB, only retry if file on disk changed (e.g. replaced/overwritten)
                if row["status"] in ("error", "empty_file"):
                    if abs(row["file_mtime"] - file_mtime) < 1e-4 and row["file_size"] == file_size:
                        return True
                    return False

                # If modified or size changed, need re-indexing
                if abs(row["file_mtime"] - file_mtime) < 1e-4 and row["file_size"] == file_size:
                    # If photo has registered faces, ensure stored vector dimension matches current engine dimension
                    if row["num_faces"] > 0:
                        face_cur = conn.execute(
                            "SELECT LENGTH(encoding_blob) AS blob_size FROM faces WHERE photo_id = ? LIMIT 1",
                            (row["id"],)
                        )
                        f_row = face_cur.fetchone()
                        if f_row and f_row["blob_size"] != (self.vector_dim * 4):
                            # Stored vector dimension changed (e.g. 128-d vs 512-d); needs re-indexing with upgraded engine
                            return False
                    return True
            return False

    def get_all_indexed_metadata(self) -> Dict[str, Tuple[float, int]]:
        """
        High-performance batch check: returns {file_path: (file_mtime, file_size)}
        in a single query across the entire database, eliminating thousands of individual SQLite queries.
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT file_path, file_mtime, file_size, status FROM photos")
            rows = cursor.fetchall()
            return {
                r["file_path"]: (float(r["file_mtime"]), int(r["file_size"]))
                for r in rows
                if r["status"] not in ("error", "empty_file")
            }

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

            # Add to FAISS in memory and persist with debouncing
            if faiss_vectors and FAISS_AVAILABLE and self.index is not None:
                v_matrix = np.vstack(faiss_vectors).astype(np.float32)
                id_matrix = np.array(faiss_ids, dtype=np.int64)
                self.index.add_with_ids(v_matrix, id_matrix)
                self._faiss_dirty = True
                # Debounced save: persist to disk at most every 15 seconds during bulk ingestion
                now = time.time()
                if now - getattr(self, "_last_faiss_save_time", 0) >= 15.0:
                    self._save_faiss()
                    self._last_faiss_save_time = now

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

    def delete_photo_by_path(self, file_path: str) -> bool:
        """
        Immediately delete a photo from SQLite and remove all its face vectors from FAISS.
        Called on Watchdog file deletion or manual purge.
        """
        file_path = os.path.abspath(file_path)
        with self.lock:
            vector_ids = []
            photo_id = None
            with self._get_connection() as conn:
                row = conn.execute("SELECT id FROM photos WHERE file_path = ?", (file_path,)).fetchone()
                if not row:
                    return False
                photo_id = row["id"]
                face_rows = conn.execute("SELECT vector_id FROM faces WHERE photo_id = ?", (photo_id,)).fetchall()
                vector_ids = [r["vector_id"] for r in face_rows]
                conn.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
                conn.commit()

            if vector_ids and FAISS_AVAILABLE and self.index is not None:
                try:
                    self.index.remove_ids(np.array(vector_ids, dtype=np.int64))
                    self._save_faiss()
                except Exception as e:
                    logger.warning(f"Error removing vector IDs from FAISS ({e}). Rebuilding...")
                    self._rebuild_faiss_from_sqlite()

            logger.info(f"Deleted photo #{photo_id} ({os.path.basename(file_path)}) and {len(vector_ids)} vectors.")
            return True

    def delete_folder(self, folder_path: str) -> Dict[str, int]:
        """
        Immediately delete all photos located in or under folder_path from SQLite
        and remove all associated face vectors from FAISS.
        """
        folder_path = os.path.abspath(folder_path).rstrip(r"\/")
        prefix_slash = folder_path + "/"
        prefix_backslash = folder_path + "\\"

        with self.lock:
            photo_ids = []
            vector_ids = []
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT id FROM photos WHERE file_path = ? OR file_path LIKE ? OR file_path LIKE ?",
                    (folder_path, prefix_slash + "%", prefix_backslash + "%")
                ).fetchall()
                photo_ids = [r["id"] for r in rows]

                if not photo_ids:
                    return {"deleted_photos": 0, "deleted_faces": 0}

                placeholders = ",".join("?" for _ in photo_ids)
                face_rows = conn.execute(
                    f"SELECT vector_id FROM faces WHERE photo_id IN ({placeholders})",
                    photo_ids
                ).fetchall()
                vector_ids = [r["vector_id"] for r in face_rows]

                conn.execute(f"DELETE FROM photos WHERE id IN ({placeholders})", photo_ids)
                conn.commit()

            if vector_ids and FAISS_AVAILABLE and self.index is not None:
                try:
                    self.index.remove_ids(np.array(vector_ids, dtype=np.int64))
                    self._save_faiss()
                except Exception as e:
                    logger.warning(f"Error removing vector IDs from FAISS ({e}). Rebuilding...")
                    self._rebuild_faiss_from_sqlite()

            logger.info(f"Deleted folder {folder_path}: {len(photo_ids)} photos, {len(vector_ids)} face vectors removed.")
            return {"deleted_photos": len(photo_ids), "deleted_faces": len(vector_ids)}

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
                    if w < 16 or h < 16:
                        is_invalid = True
                    else:
                        ratio = float(h) / float(w)
                        if ratio < 0.50 or ratio > 1.80:
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

    def reset_all_data(self):
        """Reset photos and faces index while safely preserving guest visit analytics."""
        with self.lock:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM faces")
                conn.execute("DELETE FROM photos")
                conn.commit()
            if FAISS_AVAILABLE:
                self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.vector_dim))
                self._save_faiss()
            logger.info("Database photos and FAISS index reset. Guest visits history preserved.")

    def search_similar_faces(
        self,
        query_faces: Optional[List[Dict[str, Any]]] = None,
        threshold: float = 0.50,
        top_k: int = 0,
        ride_filter: Optional[str] = None,
        query_encoding: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Query FAISS for matching faces above the similarity threshold and fetch photo details from SQLite.
        Inner product on unit-normalized vectors directly yields cosine similarity (0.0 to 1.0).
        Supports optional filtering by ride subfolder (e.g. 'BL' for Hot Air Balloon).
        Applies score boost for matches with similar upper-body clothing color.
        """
        if query_faces is None:
            if query_encoding is not None:
                query_faces = [{"encoding": query_encoding, "color_hist": None}]
            else:
                return []
        elif isinstance(query_faces, np.ndarray):
            query_faces = [{"encoding": query_faces, "color_hist": None}]
        with self.lock:
            if not FAISS_AVAILABLE or self.index is None or self.index.ntotal == 0:
                logger.warning("FAISS index is empty or unavailable.")
                return []

            matched_vector_ids = set()
            scores_by_vid = {}
            member_scores_by_vid: Dict[int, Dict[int, float]] = {}
            k = self.index.ntotal if top_k <= 0 else min(top_k, self.index.ntotal)
            
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
                q_member_id = qf.get("member_id", 1)
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
                        vid_int = int(vid)
                        matched_vector_ids.add(vid_int)
                        
                        # Track member-specific scores for each vector
                        if vid_int not in member_scores_by_vid:
                            member_scores_by_vid[vid_int] = {}
                        if q_member_id not in member_scores_by_vid[vid_int] or score > member_scores_by_vid[vid_int][q_member_id]:
                            member_scores_by_vid[vid_int][q_member_id] = float(score)

                        # Keep highest score if vector matches multiple poses
                        if vid_int not in scores_by_vid or score > scores_by_vid[vid_int]:
                            scores_by_vid[vid_int] = float(score)

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

            # Apply ride subfolder filter if specified (e.g. 'Ayunan', '2026-10-10/Ayunan')
            if ride_filter and ride_filter.strip() and ride_filter.lower() not in ["all", "semua"]:
                clean_rf = ride_filter.strip().strip("/\\")
                rf_slash = clean_rf.replace("\\", "/")
                rf_bslash = clean_rf.replace("/", "\\")
                query_sql += " AND (p.file_path LIKE ? OR p.file_path LIKE ?)"
                params.extend([f"%/{rf_slash}/%", f"%\\{rf_bslash}\\%"])

            with self._get_connection() as conn:
                cursor = conn.execute(query_sql, params)
                rows = cursor.fetchall()

            # Group results by photo so that multiple face matches on the same photo retain the highest similarity
            results_by_photo: Dict[int, Dict[str, Any]] = {}
            for r in rows:
                vid = r["vector_id"]
                base_score = scores_by_vid.get(vid, 0.0)
                boost = get_color_boost(r["color_hist_blob"])
                final_score = min(1.0, float(base_score + boost))
                
                photo_id = r["photo_id"]

                # Determine which family/group member matched this specific face best
                member_map = member_scores_by_vid.get(vid, {})
                best_member_id = max(member_map.keys(), key=lambda m: member_map[m]) if member_map else 1

                face_info = {
                    "vector_id": vid,
                    "score": round(final_score, 4),
                    "base_score": round(base_score, 4),
                    "color_boost": round(boost, 4),
                    "member_id": best_member_id,
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
                        "matched_faces": [face_info],
                        "matched_member_ids": {best_member_id}
                    }
                else:
                    results_by_photo[photo_id]["matched_faces"].append(face_info)
                    results_by_photo[photo_id]["matched_member_ids"].add(best_member_id)
                    if final_score > results_by_photo[photo_id]["max_score"]:
                        results_by_photo[photo_id]["max_score"] = round(final_score, 4)
                        results_by_photo[photo_id]["best_vector_id"] = vid
                        results_by_photo[photo_id]["best_bbox"] = [r["bbox_top"], r["bbox_right"], r["bbox_bottom"], r["bbox_left"]]

            # Convert matched_member_ids to sorted list and calculate count
            for p in results_by_photo.values():
                m_ids = p.get("matched_member_ids", set())
                p["matched_member_count"] = len(m_ids)
                p["matched_member_ids"] = sorted(list(m_ids))

            # Sort results descending: Photos with more family members together rank FIRST,
            # followed by highest similarity score.
            sorted_results = sorted(
                results_by_photo.values(),
                key=lambda x: (
                    x.get("matched_member_count", 1),
                    x["max_score"]
                ),
                reverse=True
            )
            if top_k > 0 and len(sorted_results) > top_k:
                return sorted_results[:top_k]
            return sorted_results

    def get_stats(self) -> Dict[str, Any]:
        """Return system statistics on photos, faces, and index size."""
        with self.lock, self._get_connection() as conn:
            p_cur = conn.execute("SELECT COUNT(*) FROM photos WHERE status != 'empty_file'")
            total_photos = p_cur.fetchone()[0]

            all_cur = conn.execute("SELECT COUNT(*) FROM photos")
            all_recorded = all_cur.fetchone()[0]

            empty_cur = conn.execute("SELECT COUNT(*) FROM photos WHERE status = 'empty_file'")
            empty_photos = empty_cur.fetchone()[0]

            f_cur = conn.execute("SELECT COUNT(*) FROM faces")
            total_faces = f_cur.fetchone()[0]

            no_face_cur = conn.execute("SELECT COUNT(*) FROM photos WHERE status = 'no_faces'")
            no_faces = no_face_cur.fetchone()[0]

            err_cur = conn.execute("SELECT COUNT(*) FROM photos WHERE status = 'error'")
            errors = err_cur.fetchone()[0]

        faiss_total = self.index.ntotal if (FAISS_AVAILABLE and self.index) else 0

        return {
            "total_photos": total_photos,
            "all_recorded_photos": all_recorded,
            "empty_photos": empty_photos,
            "total_faces": total_faces,
            "photos_without_faces": no_faces,
            "indexing_errors": errors,
            "faiss_indexed_vectors": faiss_total,
            "vector_dim": self.vector_dim,
            "vector_dimension": self.vector_dim,
            "faiss_ready": FAISS_AVAILABLE and (self.index is not None),
            "face_rec_ready": FACE_REC_AVAILABLE,
            "insightface_ready": INSIGHTFACE_AVAILABLE and ONNXRUNTIME_AVAILABLE
        }

    def record_guest_visit(
        self,
        crm_guest_id: Optional[str],
        guest_name: str,
        guest_phone: Optional[str] = None,
        registered_at: Optional[str] = None,
        claimed_at: Optional[datetime.datetime] = None,
        photo_count: int = 0,
        customer_folder: Optional[str] = None
    ) -> Dict[str, Any]:
        """Record guest visit and calculate stay duration."""
        claimed_dt = claimed_at if (isinstance(claimed_at, datetime.datetime) and claimed_at.tzinfo is not None) else get_now_wib()
        reg_dt = parse_datetime_flexible(registered_at)

        if reg_dt is not None:
            dwell_time_seconds = max(0, int((claimed_dt - reg_dt).total_seconds()))
            dwell_time_formatted = format_dwell_time(dwell_time_seconds)
            reg_str = reg_dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            dwell_time_seconds = 0
            dwell_time_formatted = "Input Manual (Tanpa Form)"
            reg_str = None

        claimed_str = claimed_dt.strftime("%Y-%m-%d %H:%M:%S")
        date_key = claimed_dt.strftime("%Y-%m-%d")
        clean_name = (guest_name or "Tamu").strip()
        masked_phone = mask_phone_number(guest_phone)

        with self.lock, self._get_connection() as conn:
            existing_id = None
            if crm_guest_id:
                cur = conn.execute(
                    "SELECT id FROM guest_visits WHERE crm_guest_id = ? AND date_key = ?",
                    (str(crm_guest_id).strip(), date_key)
                )
                row = cur.fetchone()
                if row:
                    existing_id = row[0]

            if not existing_id and clean_name != "Tamu" and masked_phone:
                cur = conn.execute(
                    "SELECT id FROM guest_visits WHERE guest_name = ? AND masked_phone = ? AND date_key = ?",
                    (clean_name, masked_phone, date_key)
                )
                row = cur.fetchone()
                if row:
                    existing_id = row[0]

            if existing_id:
                conn.execute("""
                    UPDATE guest_visits
                    SET claimed_at = ?,
                        dwell_time_seconds = CASE WHEN ? > 0 THEN ? ELSE dwell_time_seconds END,
                        dwell_time_formatted = CASE WHEN ? IS NOT NULL THEN ? ELSE dwell_time_formatted END,
                        photo_count = MAX(photo_count, ?),
                        customer_folder = COALESCE(?, customer_folder),
                        crm_guest_id = COALESCE(?, crm_guest_id),
                        registered_at = COALESCE(?, registered_at)
                    WHERE id = ?
                """, (
                    claimed_str,
                    dwell_time_seconds, dwell_time_seconds,
                    reg_str, dwell_time_formatted,
                    photo_count,
                    customer_folder,
                    crm_guest_id,
                    reg_str,
                    existing_id
                ))
                visit_id = existing_id

                cur = conn.execute("SELECT crm_guest_id, registered_at, dwell_time_formatted, dwell_time_seconds FROM guest_visits WHERE id = ?", (existing_id,))
                u_row = cur.fetchone()
                if u_row:
                    crm_guest_id = u_row[0] or crm_guest_id
                    reg_str = u_row[1] or reg_str
                    dwell_time_formatted = u_row[2] or dwell_time_formatted
                    dwell_time_seconds = u_row[3] or dwell_time_seconds
            else:
                cur = conn.execute("""
                    INSERT INTO guest_visits (
                        crm_guest_id, guest_name, guest_phone, masked_phone,
                        registered_at, claimed_at, dwell_time_seconds, dwell_time_formatted,
                        photo_count, customer_folder, date_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    crm_guest_id, clean_name, guest_phone, masked_phone,
                    reg_str, claimed_str, dwell_time_seconds, dwell_time_formatted,
                    photo_count, customer_folder, date_key
                ))
                visit_id = cur.lastrowid
            conn.commit()

        return {
            "id": visit_id,
            "crm_guest_id": crm_guest_id,
            "guest_name": clean_name,
            "guest_phone": guest_phone,
            "masked_phone": masked_phone,
            "registered_at": reg_str,
            "claimed_at": claimed_str,
            "dwell_time_seconds": dwell_time_seconds,
            "dwell_time_formatted": dwell_time_formatted,
            "photo_count": photo_count,
            "customer_folder": customer_folder,
            "date_key": date_key
        }

    def get_analytics_summary(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """Fetch daily dwell time analytics, visitor counts, and hourly trends."""
        target_date = date_str or get_now_wib().strftime("%Y-%m-%d")
        with self.lock, self._get_connection() as conn:
            cur = conn.execute("""
                SELECT 
                    COUNT(*),
                    COALESCE(AVG(CASE WHEN dwell_time_seconds > 0 THEN dwell_time_seconds END), 0),
                    COALESCE(MIN(CASE WHEN dwell_time_seconds > 0 THEN dwell_time_seconds END), 0),
                    COALESCE(MAX(CASE WHEN dwell_time_seconds > 0 THEN dwell_time_seconds END), 0),
                    COALESCE(SUM(photo_count), 0),
                    COUNT(CASE WHEN dwell_time_seconds > 0 THEN 1 END)
                FROM guest_visits
                WHERE date_key = ?
            """, (target_date,))
            row = cur.fetchone()
            total_visits, avg_sec, min_sec, max_sec, total_photos, valid_dwell_count = row

            cur = conn.execute("""
                SELECT id, crm_guest_id, guest_name, guest_phone, masked_phone,
                       registered_at, claimed_at, dwell_time_seconds, dwell_time_formatted,
                       photo_count, customer_folder, created_at
                FROM guest_visits
                WHERE date_key = ?
                ORDER BY claimed_at DESC
            """, (target_date,))
            visits = []
            for r in cur.fetchall():
                visits.append({
                    "id": r[0],
                    "crm_guest_id": r[1] or "",
                    "guest_name": r[2],
                    "guest_phone": r[3] or "",
                    "masked_phone": r[4] or "",
                    "registered_at": r[5] or "",
                    "claimed_at": r[6] or "",
                    "dwell_time_seconds": r[7] or 0,
                    "dwell_time_formatted": r[8] or "-",
                    "photo_count": r[9] or 0,
                    "customer_folder": r[10] or "",
                    "created_at": r[11]
                })

            hourly_entry = {f"{h:02d}:00": 0 for h in range(7, 19)}
            hourly_exit = {f"{h:02d}:00": 0 for h in range(7, 19)}

            for v in visits:
                if v["registered_at"]:
                    try:
                        hr = v["registered_at"][11:13] + ":00"
                        if hr in hourly_entry:
                            hourly_entry[hr] += 1
                    except Exception:
                        pass
                if v["claimed_at"]:
                    try:
                        hr = v["claimed_at"][11:13] + ":00"
                        if hr in hourly_exit:
                            hourly_exit[hr] += 1
                    except Exception:
                        pass

            cur = conn.execute("""
                SELECT 
                    date_key,
                    COUNT(*),
                    COALESCE(AVG(CASE WHEN dwell_time_seconds > 0 THEN dwell_time_seconds END), 0),
                    COALESCE(SUM(photo_count), 0)
                FROM guest_visits
                WHERE date_key <= ?
                GROUP BY date_key
                ORDER BY date_key DESC
                LIMIT 7
            """, (target_date,))
            trend = []
            for r in cur.fetchall():
                trend.append({
                    "date": r[0],
                    "total_guests": r[1],
                    "avg_dwell_seconds": int(r[2]),
                    "avg_dwell_formatted": format_dwell_time(int(r[2])),
                    "total_photos": r[3]
                })
            trend.reverse()

        return {
            "date": target_date,
            "total_visits": total_visits,
            "valid_dwell_count": valid_dwell_count,
            "avg_dwell_seconds": int(avg_sec),
            "avg_dwell_formatted": format_dwell_time(int(avg_sec)),
            "min_dwell_seconds": int(min_sec),
            "min_dwell_formatted": format_dwell_time(int(min_sec)) if min_sec > 0 else "-",
            "max_dwell_seconds": int(max_sec),
            "max_dwell_formatted": format_dwell_time(int(max_sec)) if max_sec > 0 else "-",
            "total_photos_claimed": total_photos,
            "visits": visits,
            "hourly_entry": hourly_entry,
            "hourly_exit": hourly_exit,
            "trend_7days": trend
        }

    def export_analytics_csv(self, date_str: Optional[str] = None) -> str:
        """Export guest visits data to CSV format."""
        target_date = date_str or get_now_wib().strftime("%Y-%m-%d")
        with self.lock, self._get_connection() as conn:
            if target_date.lower() == "all":
                cur = conn.execute("""
                    SELECT id, date_key, crm_guest_id, guest_name, guest_phone,
                           registered_at, claimed_at, dwell_time_formatted, dwell_time_seconds,
                           photo_count, customer_folder
                    FROM guest_visits
                    ORDER BY claimed_at DESC
                """)
            else:
                cur = conn.execute("""
                    SELECT id, date_key, crm_guest_id, guest_name, guest_phone,
                           registered_at, claimed_at, dwell_time_formatted, dwell_time_seconds,
                           photo_count, customer_folder
                    FROM guest_visits
                    WHERE date_key = ?
                    ORDER BY claimed_at DESC
                """, (target_date,))
            rows = cur.fetchall()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "No",
            "Tanggal",
            "ID CRM",
            "Nama Tamu",
            "No Telepon",
            "Waktu Registrasi Masuk",
            "Waktu Klaim Kios Foto",
            "Durasi di Area",
            "Durasi (Detik)",
            "Jumlah Foto Cocok",
            "Folder Hasil"
        ])
        for idx, r in enumerate(rows, start=1):
            writer.writerow([
                idx,
                r[1] or "",
                r[2] or "",
                r[3] or "",
                r[4] or "",
                r[5] or "",
                r[6] or "",
                r[7] or "",
                r[8] or 0,
                r[9] or 0,
                r[10] or ""
            ])
        return output.getvalue()
