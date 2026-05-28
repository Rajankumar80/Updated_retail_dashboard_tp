from __future__ import annotations
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import threading
import queue
import time
import sqlite3
import urllib.request
import json
import multiprocessing
import os
import math
import sys
from collections import deque, defaultdict
from contextlib import nullcontext
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn.functional as F
import inspect
import transformers
from transformers import AutoModelForImageClassification, AutoConfig, AutoImageProcessor

transformers.logging.set_verbosity_error()

CONFIG_API_URL = "http://localhost:5000/api/all_stores_config"
YOLO_INPUT_SIZE = 640
YOLO_CONF = 0.35
YOLO_IOU = 0.50
PERSON_CLASS = 0
FACE_DET_SIZE = 640
FACE_DET_THRESH = 0.55
COLOR_MALE = (255, 140, 0)
COLOR_FEMALE = (219, 39, 119)
COLOR_UNKNOWN = (160, 160, 160)
COLOR_HUD = (220, 220, 220)
COLOR_OUTER_LINE = (0, 200, 255)
COLOR_INNER_LINE = (0, 255, 100)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LABEL = 0.52
FONT_SCALE_HUD = 0.65
FONT_THICK = 1
BOX_THICK = 2
FRAME_QUEUE_SIZE = 2
RECONNECT_DELAY = 3.0
RECONNECT_MAX = 30
FULL_RES_W = 1280
FULL_RES_H = 720
PROC_EVERY = 1
VOTE_WINDOW = 120
MIN_GENDER_VOTES_TO_DECIDE = 5
DB_COMMIT_MIN_VOTES = 5
DB_COMMIT_MIN_STABLE_STREAK = 3
MAX_ALLOWED_FLIPS = 2
GENDER_MALE_THRESHOLD = 0.55
GENDER_FEMALE_THRESHOLD = 0.45
FACE_LOCK_STRONG_HIGH = 0.75
FACE_LOCK_STRONG_LOW = 0.25
BODY_CONFIDENT_HIGH = 0.62
BODY_CONFIDENT_LOW = 0.38
FACE_VOTE_WEIGHT = 3
BODY_VOTE_WEIGHT = 2
MIN_FACE_VOTES_TO_DECIDE = 4
FACE_LOCK_THRESHOLD = 6
BACK_VIEW_MIN_FACE_AREA_RATIO = 0.003
BACK_VIEW_FACE_PRESENCE_WINDOW = 30
BACK_VIEW_FACE_MIN_RATIO = 0.25
ENTRY_LINE_COOLDOWN_FRAMES = 45
ENTRY_COMMIT_FRAME_BUDGET = 900
SHARPNESS_THRESHOLD = 3.5
FACE_SHARPNESS_THRESHOLD = 5.0
MIN_FACE_SIZE = 40
MIN_BODY_SIZE = 16
YOLOV8_FACE_MODEL_PATH = "/home/in_rajshek/AIML/new_gender_model/demo/yolov8n-face.pt"
PRESENCE_DEBOUNCE_SEC = float(os.environ.get("PRESENCE_DEBOUNCE_SEC", "1.5"))
ALERTS_UPDATE_INTERVAL_MIN = float(os.environ.get("ALERTS_UPDATE_INTERVAL_MIN", "1.0"))
MIN_PRESENCE_TO_RESOLVE_SEC = float(os.environ.get("MIN_PRESENCE_TO_RESOLVE_SEC", "20.0"))
FOOTFALL_DETECT_INTERVAL = 15
MAX_ABSENCE_SEC_DEFAULT = 300.0
ENTRY_COUNTER_SAW_OUTER_TIMEOUT_FRAMES = 300
ZONE_MIN_FRAMES_INSIDE = 3
ZONE_MIN_FRAMES_OUTSIDE = 2
ZONE_VISIT_MIN_DWELL_SEC = 5.0
_ZONE_PALETTE = [(0,200,255),(0,220,80),(200,60,255),(255,140,0),(255,60,100),(60,180,255),(120,255,80)]
_ZONE_FONT = cv2.FONT_HERSHEY_SIMPLEX
_ZONE_FONT_SCALE = 0.58
_ZONE_FONT_THICK = 1
_PRESENCE_COLOR_PRESENT = (0, 255, 80)
_PRESENCE_COLOR_ABSENT = (0, 0, 255)
_PRESENCE_COLOR_ALERT = (0, 0, 200)
_db_lock = threading.Lock()
BOTSORT_MAX_AGE = 100
BOTSORT_MIN_HITS = 1
BOTSORT_IOU_THRESHOLD = 0.30
BOTSORT_COAST_AGE = 60
BOTSORT_CASCADE_IOU_RATIO = 0.40
SUPPRESS_IOU_THRESHOLD = 0.50
APPEARANCE_HIST_BINS = 32
APPEARANCE_BUFFER_LEN = 8
APPEARANCE_WEIGHT = 0.35
FACE_CROP_MARGIN_RATIO = 0.15
MIN_FPS_TARGET = 20
DB_BATCH_FLUSH_INTERVAL = 0.5
CLASSIFIER_SKIP_FRAMES = 2


def _is_true_back_view(person_bbox, face_bbox_or_none, frame_shape):
    px1, py1, px2, py2 = person_bbox[:4]
    person_height = py2 - py1
    person_width = px2 - px1
    if face_bbox_or_none is not None:
        fx1, fy1, fx2, fy2 = face_bbox_or_none[:4]
        face_area = (fx2 - fx1) * (fy2 - fy1)
        person_area = person_height * person_width
        if person_area > 0 and face_area / person_area < BACK_VIEW_MIN_FACE_AREA_RATIO:
            return True
        face_cy = (fy1 + fy2) / 2.0
        expected_face_cy = py1 + person_height * 0.25
        if face_cy > expected_face_cy + person_height * 0.25:
            return True
        return False
    upper_body_cutoff = frame_shape[0] * 0.15
    if py1 < upper_body_cutoff:
        return False
    if person_height > frame_shape[0] * 0.15 and py2 > frame_shape[0] * 0.4:
        aspect_ratio = person_width / max(person_height, 1)
        if aspect_ratio < 0.65:
            return True
    return False


def _validate_face_quality(face_crop_bgr):
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return False
    fh, fw = face_crop_bgr.shape[:2]
    if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
        return False
    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < FACE_SHARPNESS_THRESHOLD:
        return False
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    entropy = -np.sum(hist * np.log2(hist + 1e-6))
    if entropy < 4.5:
        return False
    return True


@dataclass
class _PresenceState:
    last_seen: float = field(default_factory=time.time)
    currently_occupied: bool = False
    committed_occupied: bool = False
    pending_state: Optional[bool] = None
    pending_since: float = 0.0
    is_alerting: bool = False
    absent_since_mono: float = 0.0
    absent_since_wall: float = 0.0
    present_since_mono: float = 0.0
    present_since_wall: float = 0.0
    alert_row_id: Optional[int] = None
    alert_triggered_at: Optional[str] = None
    alert_triggered_at_mono: float = 0.0
    alert_last_update_mono: float = 0.0


@dataclass
class _ZoneState:
    inside_streak: int = 0
    outside_streak: int = 0
    committed_inside: bool = False
    born_inside: bool = True
    lockdown_until: float = 0.0
    dwell_start_mono: float = 0.0
    visit_counted: bool = False


class ZoneEvent:
    __slots__ = ("zone", "track_id", "direction", "timestamp", "foot")

    def __init__(self, zone, track_id, direction, timestamp, foot):
        self.zone = zone
        self.track_id = track_id
        self.direction = direction
        self.timestamp = timestamp
        self.foot = foot


def _compute_color_histogram(crop_bgr, bins=APPEARANCE_HIST_BINS):
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros(bins * 3, dtype=np.float32)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    s = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
    v = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten()
    feat = np.concatenate([h, s, v]).astype(np.float32)
    n = feat.sum()
    return feat / (n + 1e-6)


def _histogram_similarity(a, b):
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


class _AppearanceBuffer:
    def __init__(self, maxlen=APPEARANCE_BUFFER_LEN):
        self._buf = deque(maxlen=maxlen)

    def update(self, hist):
        self._buf.append(hist)

    def mean_hist(self):
        if not self._buf:
            return None
        return np.mean(np.stack(list(self._buf)), axis=0)

    def similarity_to(self, hist):
        mh = self.mean_hist()
        if mh is None:
            return 0.0
        dist = _histogram_similarity(mh, hist)
        return 1.0 - dist


def _suppress_duplicate_tracks(tracks):
    if len(tracks) < 2:
        return tracks
    is_coasted = tracks[:, 5].astype(bool)
    active_idx = np.where(~is_coasted)[0]
    coasted_idx = np.where(is_coasted)[0]
    if len(active_idx) == 0 or len(coasted_idx) == 0:
        return tracks
    boxes_c = tracks[coasted_idx, :4].astype(np.float32)
    boxes_a = tracks[active_idx, :4].astype(np.float32)
    xx1 = np.maximum(boxes_c[:, None, 0], boxes_a[None, :, 0])
    yy1 = np.maximum(boxes_c[:, None, 1], boxes_a[None, :, 1])
    xx2 = np.minimum(boxes_c[:, None, 2], boxes_a[None, :, 2])
    yy2 = np.minimum(boxes_c[:, None, 3], boxes_a[None, :, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_c = np.maximum(1.0, (boxes_c[:, 2] - boxes_c[:, 0]) * (boxes_c[:, 3] - boxes_c[:, 1]))
    area_a = np.maximum(1.0, (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))
    iou = inter / (area_c[:, None] + area_a[None, :] - inter + 1e-6)
    suppress = np.any(iou > SUPPRESS_IOU_THRESHOLD, axis=1)
    keep = np.ones(len(tracks), dtype=bool)
    keep[coasted_idx[suppress]] = False
    return tracks[keep]


def _check_camera_alive(url, timeout=5):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    deadline = time.time() + timeout
    alive = False
    while time.time() < deadline:
        ret, frame = cap.read()
        if ret and frame is not None:
            alive = True
            break
        time.sleep(0.2)
    cap.release()
    return alive


_GUI_AVAILABLE = None


def gui_available():
    global _GUI_AVAILABLE
    if _GUI_AVAILABLE is None:
        try:
            cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("__probe__")
            _GUI_AVAILABLE = True
        except cv2.error:
            _GUI_AVAILABLE = False
    return _GUI_AVAILABLE


def _load_yolo_internal():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO("yolov8s.pt")
    yolo.to(device)
    yolo.fuse()
    half = device == "cuda"
    _ = yolo(
        np.zeros((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3), dtype=np.uint8),
        imgsz=YOLO_INPUT_SIZE,
        verbose=False,
        half=half,
    )
    return yolo


def _load_face_yolo_internal():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(YOLOV8_FACE_MODEL_PATH)
    model.to(device)
    half = device == "cuda"
    _ = model(
        np.zeros((FACE_DET_SIZE, FACE_DET_SIZE, 3), dtype=np.uint8),
        imgsz=FACE_DET_SIZE,
        verbose=False,
        half=half,
    )
    return model


class AsyncDBWriter:
    def __init__(self, db_path):
        self._db_path = db_path
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="AsyncDBWriter")
        self._thread.start()

    def _worker(self):
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        pending = []
        last_flush = time.monotonic()
        while not self._stop.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.1)
                pending.append(item)
            except queue.Empty:
                pass
            now = time.monotonic()
            if pending and (now - last_flush >= DB_BATCH_FLUSH_INTERVAL or len(pending) >= 50):
                try:
                    with conn:
                        for sql, params in pending:
                            conn.execute(sql, params)
                except Exception as e:
                    pass
                pending.clear()
                last_flush = now
        if pending:
            try:
                with conn:
                    for sql, params in pending:
                        conn.execute(sql, params)
            except Exception:
                pass
        conn.close()

    def execute(self, sql, params=()):
        self._queue.put((sql, params))

    def execute_sync(self, sql, params=()):
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            with _db_lock:
                cur = conn.execute(sql, params)
                conn.commit()
                return cur.lastrowid
        finally:
            conn.close()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5.0)


class ZoneCounter:
    def __init__(self, zones, event_lockdown=3.0):
        self.event_lockdown = event_lockdown
        self._polygons = {}
        self.counts = {}
        self._states = {}
        for name, cfg in zones.items():
            arr = np.array(cfg["points"], dtype=np.int32)
            if len(arr) > 1 and np.array_equal(arr[0], arr[-1]):
                arr = arr[:-1]
            if len(arr) < 3:
                continue
            self._polygons[name] = arr
            self.counts[name] = {"entries": 0, "effective_gen_age": cfg.get("effective_gen_age", False)}

    def _is_inside(self, pt, poly):
        return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), measureDist=False) >= 0

    def update(self, tracks):
        now_mono = time.monotonic()
        now_wall = datetime.now().isoformat(timespec="milliseconds")
        active_ids = {t["id"] for t in tracks}
        fired = []
        for track in tracks:
            tid = track["id"]
            bbox = track["bbox"]
            foot = ((bbox[0] + bbox[2]) >> 1, bbox[3])
            for zone_name, poly in self._polygons.items():
                key = (tid, zone_name)
                currently_inside = self._is_inside(foot, poly)
                st = self._states.get(key)
                if st is None:
                    st = _ZoneState(
                        inside_streak=1 if currently_inside else 0,
                        outside_streak=0 if currently_inside else 1,
                        committed_inside=currently_inside,
                        born_inside=currently_inside,
                        lockdown_until=0.0,
                        dwell_start_mono=now_mono if currently_inside else 0.0,
                        visit_counted=False,
                    )
                    self._states[key] = st
                    continue
                if currently_inside:
                    st.inside_streak += 1
                    st.outside_streak = 0
                    if st.dwell_start_mono == 0.0:
                        st.dwell_start_mono = now_mono
                else:
                    st.outside_streak += 1
                    st.inside_streak = 0
                if now_mono < st.lockdown_until:
                    continue
                if not st.committed_inside and st.inside_streak >= ZONE_MIN_FRAMES_INSIDE:
                    st.committed_inside = True
                    if st.dwell_start_mono == 0.0:
                        st.dwell_start_mono = now_mono
                elif st.committed_inside and st.outside_streak >= ZONE_MIN_FRAMES_OUTSIDE:
                    if (
                        st.dwell_start_mono > 0.0
                        and (now_mono - st.dwell_start_mono) >= ZONE_VISIT_MIN_DWELL_SEC
                        and not st.visit_counted
                    ):
                        self.counts[zone_name]["entries"] += 1
                        st.visit_counted = True
                        st.lockdown_until = now_mono + self.event_lockdown
                        fired.append(ZoneEvent(zone_name, tid, "entry", now_wall, foot))
                    st.committed_inside = False
                    st.dwell_start_mono = 0.0
        stale_keys = [k for k in self._states if k[0] not in active_ids]
        for k in stale_keys:
            st = self._states[k]
            zone_name = k[1]
            if (
                st.committed_inside
                and not st.visit_counted
                and st.dwell_start_mono > 0.0
                and (now_mono - st.dwell_start_mono) >= ZONE_VISIT_MIN_DWELL_SEC
            ):
                self.counts[zone_name]["entries"] += 1
                fired.append(ZoneEvent(zone_name, k[0], "entry", now_wall, (0, 0)))
            del self._states[k]
        return fired

    def draw(self, frame, alpha=0.18):
        for idx, (zone_name, poly) in enumerate(self._polygons.items()):
            color = _ZONE_PALETTE[idx % len(_ZONE_PALETTE)]
            cnts = self.counts[zone_name]
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly], color)
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
            cv2.polylines(frame, [poly], True, color, 2, cv2.LINE_AA)
            for pt in poly:
                cv2.circle(frame, tuple(pt.tolist()), 4, color, -1, cv2.LINE_AA)
            cx, cy = int(poly[:, 0].mean()), int(poly[:, 1].mean())
            label = f"{zone_name} IN:{cnts['entries']}"
            (tw, th), bl = cv2.getTextSize(label, _ZONE_FONT, _ZONE_FONT_SCALE, _ZONE_FONT_THICK)
            pad = 6
            cv2.rectangle(frame, (cx - tw // 2 - pad, cy - th - bl - pad), (cx + tw // 2 + pad, cy + pad), (0, 0, 0), -1)
            cv2.rectangle(frame, (cx - tw // 2 - pad, cy - th - bl - pad), (cx + tw // 2 + pad, cy + pad), color, 1)
            cv2.putText(frame, label, (cx - tw // 2, cy - bl), _ZONE_FONT, _ZONE_FONT_SCALE, color, _ZONE_FONT_THICK, cv2.LINE_AA)


class StaffPresenceMonitor:
    def __init__(self, zones, camera_id, store_name, db_writer):
        self.camera_id = camera_id
        self.store_name = store_name
        self._db = db_writer
        self._polygons = {}
        self._meta = {}
        self._states = {}
        now_wall = time.time()
        now_mono = time.monotonic()
        for name, cfg in zones.items():
            pts = np.array(cfg["points"], dtype=np.int32)
            if len(pts) > 1 and np.array_equal(pts[0], pts[-1]):
                pts = pts[:-1]
            if len(pts) < 3:
                continue
            self._polygons[name] = pts
            self._meta[name] = {
                "max_absence_sec": float(cfg.get("max_absence_sec", MAX_ABSENCE_SEC_DEFAULT)),
                "min_presence_to_resolve": float(cfg.get("min_presence_to_resolve_sec", MIN_PRESENCE_TO_RESOLVE_SEC)),
                "role_type": str(cfg.get("role_type", "cashier")).lower().strip(),
            }
            self._states[name] = _PresenceState(last_seen=now_wall, absent_since_mono=now_mono, absent_since_wall=now_wall)

    @staticmethod
    def _centroid(bbox):
        return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

    @staticmethod
    def _inside(pt, poly):
        return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), measureDist=False) >= 0

    def _open_alert(self, zone_name):
        st = self._states[zone_name]
        triggered_at_iso = datetime.fromtimestamp(st.absent_since_wall).isoformat(timespec="seconds")
        now_mono = time.monotonic()
        initial_duration = max(0.0, now_mono - st.absent_since_mono)
        role_type = self._meta[zone_name]["role_type"]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(
            f"[ALERT OPEN] timestamp={ts} | store={self.store_name} | camera_id={self.camera_id} | "
            f"zone={zone_name} | role={role_type} | triggered_at={triggered_at_iso} | "
            f"duration_min={initial_duration / 60.0:.3f}"
        )
        row_id = self._db.execute_sync(
            "INSERT INTO alerts (store_name, camera_id, zone_name, alert_label, triggered_at, duration_min, role_type) VALUES (?,?,?,?,?,?,?)",
            (self.store_name, self.camera_id, zone_name, "Absent", triggered_at_iso, initial_duration / 60.0, role_type),
        )
        st.alert_row_id = row_id
        st.alert_triggered_at = triggered_at_iso
        st.alert_triggered_at_mono = st.absent_since_mono
        st.alert_last_update_mono = now_mono
        st.is_alerting = True

    def _update_alert_duration(self, zone_name):
        st = self._states[zone_name]
        if st.alert_row_id is None:
            return
        now_mono = time.monotonic()
        duration_sec = max(0.0, now_mono - st.alert_triggered_at_mono)
        self._db.execute(
            "UPDATE alerts SET duration_min=? WHERE id=?",
            (duration_sec / 60.0, st.alert_row_id),
        )
        st.alert_last_update_mono = now_mono

    def _close_alert(self, zone_name):
        st = self._states[zone_name]
        if st.alert_row_id is None:
            return
        resolved_at = datetime.fromtimestamp(st.present_since_wall).isoformat(timespec="seconds")
        duration_sec = max(0.0, st.present_since_mono - st.alert_triggered_at_mono)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(
            f"[ALERT CLOSE] timestamp={ts} | store={self.store_name} | camera_id={self.camera_id} | "
            f"zone={zone_name} | resolved_at={resolved_at} | duration_min={duration_sec / 60.0:.3f}"
        )
        self._db.execute(
            "UPDATE alerts SET resolved_at=?, duration_min=? WHERE id=?",
            (resolved_at, duration_sec / 60.0, st.alert_row_id),
        )
        st.is_alerting = False
        st.alert_row_id = None
        st.alert_triggered_at = None
        st.alert_triggered_at_mono = 0.0
        st.alert_last_update_mono = 0.0
        st.absent_since_mono = 0.0
        st.absent_since_wall = 0.0
        st.present_since_mono = 0.0
        st.present_since_wall = 0.0

    def _close_alert_on_shutdown(self, zone_name):
        st = self._states[zone_name]
        if st.alert_row_id is None:
            return
        now_wall = time.time()
        now_mono = time.monotonic()
        resolved_at = datetime.fromtimestamp(now_wall).isoformat(timespec="seconds")
        duration_sec = max(0.0, now_mono - st.alert_triggered_at_mono)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(
            f"[ALERT SHUTDOWN-CLOSE] timestamp={ts} | store={self.store_name} | camera_id={self.camera_id} | "
            f"zone={zone_name} | resolved_at={resolved_at} | duration_min={duration_sec / 60.0:.3f}"
        )
        self._db.execute(
            "UPDATE alerts SET resolved_at=?, duration_min=? WHERE id=?",
            (resolved_at, duration_sec / 60.0, st.alert_row_id),
        )
        st.alert_row_id = None

    def update(self, tracks):
        now_wall = time.time()
        now_mono = time.monotonic()
        zone_occupied = {z: False for z in self._polygons}
        for t in tracks:
            bbox = t.get("bbox")
            if bbox is None:
                continue
            c = self._centroid(bbox)
            for zone_name, poly in self._polygons.items():
                if self._inside(c, poly):
                    zone_occupied[zone_name] = True
                    self._states[zone_name].last_seen = now_wall
        for zone_name in self._polygons:
            st = self._states[zone_name]
            meta = self._meta[zone_name]
            raw = zone_occupied[zone_name]
            st.currently_occupied = raw
            if raw == st.committed_occupied:
                st.pending_state = None
            else:
                if st.pending_state != raw:
                    st.pending_state = raw
                    st.pending_since = now_wall
                if (now_wall - st.pending_since) >= PRESENCE_DEBOUNCE_SEC:
                    if not raw:
                        st.absent_since_mono = now_mono
                        st.absent_since_wall = now_wall
                        st.present_since_mono = 0.0
                        st.present_since_wall = 0.0
                    else:
                        if st.absent_since_mono > 0 and st.alert_row_id is None:
                            st.absent_since_mono = 0.0
                            st.absent_since_wall = 0.0
                        if st.alert_row_id is not None:
                            if st.present_since_mono == 0.0:
                                st.present_since_mono = now_mono
                                st.present_since_wall = now_wall
                    st.committed_occupied = raw
                    st.pending_state = None
            if (
                not st.committed_occupied
                and st.alert_row_id is None
                and st.absent_since_mono > 0
                and (now_mono - st.absent_since_mono) >= meta["max_absence_sec"]
            ):
                self._open_alert(zone_name)
            if st.alert_row_id is not None:
                if (now_mono - st.alert_last_update_mono) >= max(1.0, ALERTS_UPDATE_INTERVAL_MIN * 60.0):
                    self._update_alert_duration(zone_name)
            if (
                st.alert_row_id is not None
                and st.committed_occupied
                and st.present_since_mono > 0
                and (now_mono - st.present_since_mono) >= meta["min_presence_to_resolve"]
            ):
                self._close_alert(zone_name)
            st.is_alerting = st.alert_row_id is not None

    def draw(self, frame, alpha=0.20):
        now_mono = time.monotonic()
        for zone_name, poly in self._polygons.items():
            st = self._states[zone_name]
            if st.is_alerting:
                color = _PRESENCE_COLOR_ALERT
            elif st.currently_occupied:
                color = _PRESENCE_COLOR_PRESENT
            else:
                color = _PRESENCE_COLOR_ABSENT
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly], color)
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
            cv2.polylines(frame, [poly], True, color, 2, cv2.LINE_AA)
            for pt in poly:
                cv2.circle(frame, tuple(pt.tolist()), 4, color, -1, cv2.LINE_AA)
            cx, cy = int(poly[:, 0].mean()), int(poly[:, 1].mean())
            if st.currently_occupied:
                status_str = f"{zone_name}: PRESENT"
            else:
                absent_min = (now_mono - st.absent_since_mono) / 60.0 if st.absent_since_mono > 0 else 0.0
                status_str = f"{zone_name}: ABSENT {absent_min:.1f}m"
            if st.is_alerting:
                alert_min = (now_mono - st.alert_triggered_at_mono) / 60.0 if st.alert_triggered_at_mono > 0 else 0.0
                status_str += f" [ALERT {alert_min:.1f}m]" if st.alert_triggered_at_mono > 0 else " [ALERT]"
            (tw, th), bl = cv2.getTextSize(status_str, _ZONE_FONT, _ZONE_FONT_SCALE, _ZONE_FONT_THICK)
            pad = 6
            cv2.rectangle(frame, (cx - tw // 2 - pad, cy - th - bl - pad), (cx + tw // 2 + pad, cy + pad), (0, 0, 0), -1)
            cv2.rectangle(frame, (cx - tw // 2 - pad, cy - th - bl - pad), (cx + tw // 2 + pad, cy + pad), color, 1)
            cv2.putText(
                frame, status_str, (cx - tw // 2, cy - bl), _ZONE_FONT, _ZONE_FONT_SCALE, color, _ZONE_FONT_THICK, cv2.LINE_AA
            )

    @property
    def states(self):
        return self._states

    def close_open_alerts(self):
        for zone_name in list(self._states):
            if self._states[zone_name].alert_row_id is not None:
                self._close_alert_on_shutdown(zone_name)


class GenderAgeClassifier:
    def __init__(self, device):
        self.device = device
        self._infer_dtype = torch.float16 if device.type == "cuda" else torch.float32
        self._stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

        self.config = AutoConfig.from_pretrained("iitolstykh/mivolo_v2", trust_remote_code=True)
        self.model = AutoModelForImageClassification.from_pretrained(
            "iitolstykh/mivolo_v2",
            config=self.config,
            trust_remote_code=True,
            dtype=self._infer_dtype,
        ).to(device)
        self.model.eval()

        if device.type == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                pass

        self.processor = AutoImageProcessor.from_pretrained("iitolstykh/mivolo_v2", trust_remote_code=True)
        probe = self.processor(images=[np.zeros((64, 64, 3), dtype=np.uint8)])["pixel_values"]
        self.CROP_SIZE = (probe.shape[3], probe.shape[2])
        self._blank_face = np.zeros((self.CROP_SIZE[1], self.CROP_SIZE[0], 3), dtype=np.uint8)
        self._male_class_id = None
        self._female_class_id = None
        for k, v in self.config.gender_id2label.items():
            if v.lower() == "male":
                self._male_class_id = k
            elif v.lower() == "female":
                self._female_class_id = k
        if self._male_class_id is None:
            self._male_class_id = 0
        if self._female_class_id is None:
            self._female_class_id = 1
        self._face_kwarg = self._detect_face_kwarg()
        self._norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=self._infer_dtype).view(1, 3, 1, 1)
        self._norm_std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=self._infer_dtype).view(1, 3, 1, 1)
        self._crop_h = self.CROP_SIZE[1]
        self._crop_w = self.CROP_SIZE[0]

    def _detect_face_kwarg(self):
        try:
            sig = inspect.signature(self.model.forward)
            params = list(sig.parameters.keys())
            if "faces_input" in params:
                return "faces_input"
            if "face_input" in params:
                return "face_input"
        except Exception:
            pass
        return "faces_input"

    def _crops_to_tensor_gpu(self, crops_bgr_list):
        n = len(crops_bgr_list)
        ch, cw = self._crop_h, self._crop_w
        arr = np.empty((n, ch, cw, 3), dtype=np.uint8)
        for i, crop in enumerate(crops_bgr_list):
            resized = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_LINEAR)
            arr[i] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).to(dtype=self._infer_dtype, device=self.device, non_blocking=True)
        t = t / 255.0
        t = (t - self._norm_mean) / self._norm_std
        return t

    def predict_batch(self, frame, items):
        if not items:
            return []
        h, w = frame.shape[:2]
        body_crops_bgr, face_crops_bgr, valid_ids, has_face_flags = [], [], [], []
        for track_id, p_bbox, f_bbox in items:
            px1, py1, px2, py2 = max(0, int(p_bbox[0])), max(0, int(p_bbox[1])), min(w, int(p_bbox[2])), min(h, int(p_bbox[3]))
            person_h = py2 - py1
            person_w = px2 - px1
            if person_h < MIN_BODY_SIZE or person_w < MIN_BODY_SIZE:
                continue
            body_crop_bgr = frame[py1:py2, px1:px2]
            if body_crop_bgr.size == 0:
                continue
            body_crops_bgr.append(body_crop_bgr)
            has_valid_face = False
            if f_bbox is not None:
                fx1, fy1, fx2, fy2 = max(0, int(f_bbox[0])), max(0, int(f_bbox[1])), min(w, int(f_bbox[2])), min(h, int(f_bbox[3]))
                face_raw = frame[fy1:fy2, fx1:fx2]
                if _validate_face_quality(face_raw):
                    margin_x = max(1, int((fx2 - fx1) * FACE_CROP_MARGIN_RATIO))
                    margin_y = max(1, int((fy2 - fy1) * FACE_CROP_MARGIN_RATIO))
                    efx1 = max(0, fx1 - margin_x)
                    efy1 = max(0, fy1 - margin_y)
                    efx2 = min(w, fx2 + margin_x)
                    efy2 = min(h, fy2 + margin_y)
                    face_extended = frame[efy1:efy2, efx1:efx2]
                    if face_extended.size > 0:
                        face_crops_bgr.append(face_extended)
                        has_valid_face = True
            if not has_valid_face:
                face_crops_bgr.append(self._blank_face)
            has_face_flags.append(has_valid_face)
            valid_ids.append(track_id)
        if not body_crops_bgr:
            return []

        stream_ctx = torch.cuda.stream(self._stream) if self._stream else nullcontext()
        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )

        with stream_ctx:
            b_pv = self._crops_to_tensor_gpu(body_crops_bgr)
            f_pv = self._crops_to_tensor_gpu(face_crops_bgr)
            with torch.inference_mode(), autocast_ctx:
                out = self.model(body_input=b_pv, **{self._face_kwarg: f_pv})
            if self._stream is not None:
                self._stream.synchronize()

        ages = out.age_output.float().cpu().numpy().flatten()
        gender_logits = out.raw_gender_output.float()
        gender_probs = torch.softmax(gender_logits, dim=-1).cpu().numpy()
        genders_idx = np.argmax(gender_probs, axis=1)
        male_prob_arr = gender_probs[:, self._male_class_id]
        results = []
        for i in range(len(valid_ids)):
            gender_label = self.config.gender_id2label[int(genders_idx[i])]
            results.append((valid_ids[i], gender_label, int(ages[i]), float(male_prob_arr[i]), has_face_flags[i]))
        return results


def match_face_to_bbox_batch(person_bboxes, face_boxes):
    if not face_boxes or not person_bboxes:
        return {}
    result = {}
    face_arr = np.array(face_boxes, dtype=np.float32)
    face_w = face_arr[:, 2] - face_arr[:, 0]
    face_h = face_arr[:, 3] - face_arr[:, 1]
    face_cx = (face_arr[:, 0] + face_arr[:, 2]) / 2.0
    face_cy = (face_arr[:, 1] + face_arr[:, 3]) / 2.0
    valid_face_mask = (face_w >= MIN_FACE_SIZE) & (face_h >= MIN_FACE_SIZE)
    if not np.any(valid_face_mask):
        return {}
    valid_face_idx = np.where(valid_face_mask)[0]
    face_arr_v = face_arr[valid_face_idx]
    face_cx_v = face_cx[valid_face_idx]
    face_cy_v = face_cy[valid_face_idx]
    face_area_v = face_w[valid_face_idx] * face_h[valid_face_idx]

    for pid, (track_id, p_bbox) in enumerate(person_bboxes):
        px1, py1, px2, py2 = p_bbox[:4]
        pcx = (px1 + px2) / 2.0
        ph = py2 - py1
        pw = px2 - px1
        in_box_x = (face_cx_v >= px1 - pw * 0.1) & (face_cx_v <= px2 + pw * 0.1)
        in_box_y = (face_cy_v >= py1 - ph * 0.1) & (face_cy_v <= py2 + ph * 0.1)
        candidates = np.where(in_box_x & in_box_y)[0]
        if len(candidates) == 0:
            continue
        best_score = -1.0
        best_fidx = -1
        expected_face_cy = py1 + ph * 0.28
        max_vert_dev = ph * 0.45
        max_horiz_dev = pw * 0.7
        expected_face_area = ph * pw * 0.06
        for ci in candidates:
            fcy_c = face_cy_v[ci]
            fcx_c = face_cx_v[ci]
            fa = face_area_v[ci]
            vert_dev = abs(fcy_c - expected_face_cy)
            vp = 1.0 - (vert_dev / max_vert_dev) if vert_dev <= max_vert_dev else 0.0
            horiz_dev = abs(fcx_c - pcx)
            hp_s = 1.0 - (horiz_dev / max_horiz_dev) if horiz_dev <= max_horiz_dev else 0.0
            ss = min(1.0, fa / max(expected_face_area, 100.0))
            sp = min(1.0, max(0.0, (fa - expected_face_area * 4) / (expected_face_area * 10 + 1e-6)))
            ss = ss * (1.0 - 0.3 * sp)
            f = face_arr_v[ci]
            cont = 1.0 if (px1 <= f[0] and f[2] <= px2 and py1 <= f[1] and f[3] <= py2) else (0.6 if (px1 <= fcx_c <= px2 and py1 <= fcy_c <= py2) else 0.0)
            score = vp * 0.35 + hp_s * 0.25 + ss * 0.20 + cont * 0.20
            if score > best_score:
                best_score = score
                best_fidx = ci
        if best_score >= 0.35:
            f = face_arr_v[best_fidx]
            result[track_id] = (int(f[0]), int(f[1]), int(f[2]), int(f[3]))
    return result


def get_age_bucket(age):
    if age < 0:
        return "?"
    if age <= 10:
        return "0-10"
    if age <= 20:
        return "10-20"
    if age <= 30:
        return "20-30"
    if age <= 40:
        return "30-40"
    if age <= 50:
        return "40-50"
    if age <= 60:
        return "50-60"
    if age <= 70:
        return "60-70"
    if age <= 80:
        return "70-80"
    return "80+"


def _is_sharp_enough(crop_bgr, threshold=SHARPNESS_THRESHOLD):
    if crop_bgr is None or crop_bgr.size == 0:
        return False
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < 8 or gray.shape[1] < 8:
        return False
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= threshold


def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    a1 = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    a2 = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    return inter / (a1 + a2 - inter + 1e-6)


def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return np.array([bbox[0] + w / 2.0, bbox[1] + h / 2.0, w * h, w / float(h + 1e-6)]).reshape((4, 1))


def convert_x_to_bbox(x):
    w = np.sqrt(abs(x[2] * x[3]))
    h = x[2] / (w + 1e-6)
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=float,
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.appearance = _AppearanceBuffer()

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)


class BoTSORT:
    def __init__(
        self,
        max_age=BOTSORT_MAX_AGE,
        min_hits=BOTSORT_MIN_HITS,
        iou_threshold=BOTSORT_IOU_THRESHOLD,
        coast_age=BOTSORT_COAST_AGE,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.coast_age = coast_age
        self.trackers = []
        self.frame_count = 0
        self._last_frame = None

    def set_frame(self, frame):
        self._last_frame = frame

    def _extract_appearance(self, bbox, frame):
        if frame is None:
            return None
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = frame[y1:y2, x1:x2]
        return _compute_color_histogram(crop)

    def _combined_cost(self, dets, trks, det_hists, trk_hists):
        iou_mat = iou_batch(dets[:, :4], trks[:, :4])
        if det_hists is None or trk_hists is None:
            return iou_mat
        n_d, n_t = len(dets), len(trks)
        app_mat = np.zeros((n_d, n_t), dtype=np.float32)
        for di in range(n_d):
            for ti in range(n_t):
                if det_hists[di] is not None and trk_hists[ti] is not None:
                    dist = _histogram_similarity(det_hists[di], trk_hists[ti])
                    app_mat[di, ti] = 1.0 - dist
        return (1.0 - APPEARANCE_WEIGHT) * iou_mat + APPEARANCE_WEIGHT * app_mat

    def update(self, dets):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t in range(len(self.trackers)):
            pos = self.trackers[t].predict()[0]
            trks[t, :4] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        det_hists = None
        trk_hists = None
        if self._last_frame is not None and len(dets) > 0 and len(trks) > 0:
            det_hists = [self._extract_appearance(dets[i, :4], self._last_frame) for i in range(len(dets))]
            trk_hists = [self.trackers[t].appearance.mean_hist() for t in range(len(self.trackers))]
        matched, unmatched_dets, unmatched_trks = self._associate(dets, trks, self.iou_threshold, det_hists, trk_hists)
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])
            if self._last_frame is not None:
                hist = self._extract_appearance(dets[m[0], :4], self._last_frame)
                if hist is not None:
                    self.trackers[m[1]].appearance.update(hist)
        if len(unmatched_dets) > 0 and len(unmatched_trks) > 0:
            coasted_mask = np.array([self.trackers[t].time_since_update > 0 for t in unmatched_trks], dtype=bool)
            coasted_trk_idx = [unmatched_trks[i] for i in range(len(unmatched_trks)) if coasted_mask[i]]
            if coasted_trk_idx:
                sub_trks = trks[coasted_trk_idx] if len(trks) > 0 else np.empty((0, 5))
                sub_dets = dets[unmatched_dets]
                cascade_thresh = self.iou_threshold * BOTSORT_CASCADE_IOU_RATIO
                sub_det_hists = [det_hists[i] for i in unmatched_dets] if det_hists is not None else None
                sub_trk_hists = (
                    [self.trackers[t].appearance.mean_hist() for t in coasted_trk_idx] if trk_hists is not None else None
                )
                matched2, still_unmatched, _ = self._associate(sub_dets, sub_trks, cascade_thresh, sub_det_hists, sub_trk_hists)
                for m in matched2:
                    trk_real = coasted_trk_idx[m[1]]
                    self.trackers[trk_real].update(sub_dets[m[0], :4])
                    if self._last_frame is not None:
                        hist = self._extract_appearance(sub_dets[m[0], :4], self._last_frame)
                        if hist is not None:
                            self.trackers[trk_real].appearance.update(hist)
                unmatched_dets = [unmatched_dets[i] for i in still_unmatched]
        for i in unmatched_dets:
            new_trk = KalmanBoxTracker(dets[i, :4])
            if self._last_frame is not None:
                hist = self._extract_appearance(dets[i, :4], self._last_frame)
                if hist is not None:
                    new_trk.appearance.update(hist)
            self.trackers.append(new_trk)
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            d = trk.get_state()[0]
            tsu = trk.time_since_update
            if tsu == 0 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits or trk.hits >= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1, 0])).reshape(1, -1))
            elif 0 < tsu <= self.coast_age and trk.hits >= self.min_hits:
                ret.append(np.concatenate((d, [trk.id + 1, 1])).reshape(1, -1))
            if tsu > self.max_age:
                self.trackers.pop(i)
        return np.concatenate(ret, axis=0) if ret else np.empty((0, 6))

    def _associate(self, dets, trks, iou_threshold, det_hists=None, trk_hists=None):
        if len(trks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(dets)), np.empty((0,), dtype=int)
        if len(dets) == 0:
            return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trks))
        cost_matrix = self._combined_cost(dets, trks, det_hists, trk_hists)
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        matched_indices = np.stack([row_ind, col_ind], axis=1)
        unmatched_dets = [d for d in range(len(dets)) if d not in matched_indices[:, 0]]
        unmatched_trks = [t for t in range(len(trks)) if t not in matched_indices[:, 1]]
        matches = []
        for m in matched_indices:
            iou_val = iou_batch(dets[m[0] : m[0] + 1, :4], trks[m[1] : m[1] + 1, :4])[0, 0]
            if iou_val < iou_threshold:
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        return (
            np.concatenate(matches, axis=0) if matches else np.empty((0, 2), dtype=int),
            np.array(unmatched_dets),
            np.array(unmatched_trks),
        )


class VotingClassifier:
    def __init__(self, window=VOTE_WINDOW):
        self._window = window
        self.male_probs_face = deque(maxlen=window)
        self.male_probs_body = deque(maxlen=window)
        self.age_votes = deque(maxlen=window)
        self._gender = "Unknown"
        self._age = "?"
        self._raw_age = -1.0
        self._confirmed = False
        self._confirm_streak = 0
        self._stable_streak = 0
        self._flip_count = 0
        self._last_decided_gender = None
        self._face_locked_gender = None
        self._face_lock_count = 0
        self._face_count = 0
        self._body_count = 0
        self._consecutive_same_gender = 0
        self._last_raw_gender = None

    def _raw_gender_from_prob(self, male_prob):
        if male_prob is None:
            return None
        if male_prob >= GENDER_MALE_THRESHOLD:
            return "Male"
        if male_prob <= GENDER_FEMALE_THRESHOLD:
            return "Female"
        return None

    def _gender_from_weighted_prob(self, prob):
        if prob >= GENDER_MALE_THRESHOLD:
            return "Male"
        if prob <= GENDER_FEMALE_THRESHOLD:
            return "Female"
        return None

    def _update_age_vote(self):
        if self.age_votes:
            counts = defaultdict(int)
            for a in self.age_votes:
                counts[a] += 1
            self._age = max(counts, key=counts.get)

    def update(self, male_prob, age, raw_age=-1.0, has_face=False, gender_label=None):
        if age and age != "?":
            self.age_votes.append(age)
        if raw_age is not None and raw_age >= 0:
            self._raw_age = float(raw_age)
        current_raw_gender = self._raw_gender_from_prob(male_prob)
        if current_raw_gender is not None and current_raw_gender == self._last_raw_gender:
            self._consecutive_same_gender += 1
        elif current_raw_gender is not None:
            self._consecutive_same_gender = 1
        self._last_raw_gender = current_raw_gender
        if not has_face:
            if male_prob is not None:
                self.male_probs_body.append(float(male_prob))
                self._body_count += 1
            self._update_from_body()
            self._update_age_vote()
            return
        if male_prob is None:
            self._update_age_vote()
            return
        self.male_probs_face.append(float(male_prob))
        self._face_count += 1
        self._face_lock_count += 1
        weighted_prob = self._get_weighted_prob()
        if self._face_lock_count >= FACE_LOCK_THRESHOLD and self._face_locked_gender is None:
            avg_face_prob = float(np.mean(self.male_probs_face))
            if avg_face_prob >= FACE_LOCK_STRONG_HIGH:
                self._face_locked_gender = "Male"
            elif avg_face_prob <= FACE_LOCK_STRONG_LOW:
                self._face_locked_gender = "Female"
        n_face = len(self.male_probs_face)
        if n_face < MIN_FACE_VOTES_TO_DECIDE:
            candidate = self._gender_from_weighted_prob(weighted_prob)
            if candidate is not None:
                self._gender = candidate
                self._last_decided_gender = self._gender
            self._update_age_vote()
            return
        if self._face_locked_gender is not None:
            new_gender = self._face_locked_gender
        else:
            new_gender = self._gender_from_weighted_prob(weighted_prob)
            if new_gender is None:
                self._update_age_vote()
                return
        if self._last_decided_gender is not None and new_gender != self._last_decided_gender:
            if self._consecutive_same_gender < 3:
                self._flip_count += 1
                self._stable_streak = 0
            else:
                self._stable_streak += 1
        else:
            self._stable_streak += 1
        self._last_decided_gender = new_gender
        if not self._confirmed:
            self._gender = new_gender
            self._confirm_streak += 1
            if self._confirm_streak >= MIN_GENDER_VOTES_TO_DECIDE:
                self._confirmed = True
        else:
            if new_gender == self._gender:
                self._confirm_streak = min(self._confirm_streak + 1, 60)
            else:
                if self._face_locked_gender is None:
                    if self._consecutive_same_gender < 3:
                        self._flip_count += 1
                        self._confirm_streak -= 3
                        self._stable_streak = 0
                        if self._confirm_streak <= 0:
                            self._gender = new_gender
                            self._confirmed = False
                            self._confirm_streak = MIN_GENDER_VOTES_TO_DECIDE // 2
                    else:
                        self._gender = new_gender
                        self._confirm_streak = MIN_GENDER_VOTES_TO_DECIDE
                else:
                    self._gender = self._face_locked_gender
                    self._confirm_streak = MIN_GENDER_VOTES_TO_DECIDE
        self._update_age_vote()

    def _update_from_body(self):
        n_body = len(self.male_probs_body)
        if n_body < MIN_GENDER_VOTES_TO_DECIDE:
            return
        avg_body_prob = float(np.mean(self.male_probs_body))
        if avg_body_prob >= BODY_CONFIDENT_HIGH:
            body_gender = "Male"
        elif avg_body_prob <= BODY_CONFIDENT_LOW:
            body_gender = "Female"
        else:
            return
        if self._gender == "Unknown" or self._face_count < MIN_FACE_VOTES_TO_DECIDE:
            self._gender = body_gender
            self._last_decided_gender = body_gender

    def _get_weighted_prob(self):
        n_face = len(self.male_probs_face)
        n_body = len(self.male_probs_body)
        if n_face == 0 and n_body == 0:
            return 0.5
        if n_face == 0:
            return float(np.mean(self.male_probs_body))
        if n_body == 0:
            return float(np.mean(self.male_probs_face))
        face_std = np.std(self.male_probs_face) if n_face > 1 else 0.3
        body_std = np.std(self.male_probs_body) if n_body > 1 else 0.3
        face_confidence_factor = 1.0 / (1.0 + face_std * 3)
        body_confidence_factor = 1.0 / (1.0 + body_std * 3)
        face_weight = FACE_VOTE_WEIGHT * n_face * face_confidence_factor
        body_weight = BODY_VOTE_WEIGHT * n_body * body_confidence_factor
        total_weight = face_weight + body_weight
        face_contrib = float(np.mean(self.male_probs_face)) * face_weight
        body_contrib = float(np.mean(self.male_probs_body)) * body_weight
        return (face_contrib + body_contrib) / total_weight

    @property
    def gender(self):
        return self._gender

    @property
    def age(self):
        return self._age

    @property
    def raw_age(self):
        return self._raw_age

    @property
    def total_votes(self):
        return len(self.male_probs_face) + len(self.male_probs_body)

    @property
    def stable_streak(self):
        return self._stable_streak

    @property
    def flip_count(self):
        return self._flip_count

    @property
    def has_confirmed_gender(self):
        return self._gender != "Unknown" and (
            len(self.male_probs_face) >= 1 or len(self.male_probs_body) >= MIN_GENDER_VOTES_TO_DECIDE
        )

    @property
    def is_stable_for_commit(self):
        return (
            self._gender != "Unknown"
            and len(self.male_probs_face) >= DB_COMMIT_MIN_VOTES
            and self._stable_streak >= DB_COMMIT_MIN_STABLE_STREAK
            and self._flip_count <= MAX_ALLOWED_FLIPS
        )

    @property
    def has_confirmed_age(self):
        return self._age != "?"

    @property
    def mean_male_prob(self):
        if self.male_probs_face:
            return float(np.mean(self.male_probs_face))
        if self.male_probs_body:
            return float(np.mean(self.male_probs_body))
        return -1.0


def _sign(lx1, ly1, lx2, ly2, px, py):
    return (lx2 - lx1) * (py - ly1) - (ly2 - ly1) * (px - lx1)


def _segment_line_intersection_t(ax, ay, bx, by, lx1, ly1, lx2, ly2):
    d1x = bx - ax
    d1y = by - ay
    d2x = lx2 - lx1
    d2y = ly2 - ly1
    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-9:
        return None
    t = ((lx1 - ax) * d2y - (ly1 - ay) * d2x) / denom
    u = ((lx1 - ax) * d1y - (ly1 - ay) * d1x) / denom
    if -1e-9 <= t <= 1.0 + 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
        return max(0.0, min(1.0, t))
    return None


def _line_midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)


class EntryCounter:
    _IDLE = 0
    _SAW_OUTER = 1
    _COUNTED = 2

    def __init__(self, outer_line, inner_line, reentry_allowed=True):
        (self.ox1, self.oy1), (self.ox2, self.oy2) = outer_line
        (self.ix1, self.iy1), (self.ix2, self.iy2) = inner_line
        self.entry_count = 0
        self.reentry_allowed = reentry_allowed
        self._state = {}
        self._prev_pos = {}
        self._outer_side_before = {}
        self._outer_frame = {}
        self._cooldown_until_frame = 0

    def _outer_intersect(self, px, py, cx, cy):
        return _segment_line_intersection_t(px, py, cx, cy, self.ox1, self.oy1, self.ox2, self.oy2)

    def _inner_intersect(self, px, py, cx, cy):
        return _segment_line_intersection_t(px, py, cx, cy, self.ix1, self.iy1, self.ix2, self.iy2)

    def _side_of_outer(self, px, py):
        return _sign(self.ox1, self.oy1, self.ox2, self.oy2, px, py)

    def _side_of_inner(self, px, py):
        return _sign(self.ix1, self.iy1, self.ix2, self.iy2, px, py)

    def _reset(self, track_id):
        self._state[track_id] = self._IDLE
        self._outer_side_before.pop(track_id, None)
        self._outer_frame.pop(track_id, None)

    def update(self, track_id, cx, cy, frame_no=0, coasted=False):
        if coasted:
            return False
        cx, cy = float(cx), float(cy)
        prev = self._prev_pos.get(track_id)
        self._prev_pos[track_id] = (cx, cy)
        if prev is None:
            self._state.setdefault(track_id, self._IDLE)
            return False
        px, py = prev
        if abs(cx - px) < 1.0 and abs(cy - py) < 1.0:
            return False
        state = self._state.get(track_id, self._IDLE)
        t_outer = self._outer_intersect(px, py, cx, cy)
        t_inner = self._inner_intersect(px, py, cx, cy)
        if state == self._COUNTED:
            if self.reentry_allowed and t_outer is not None:
                side_before = self._side_of_outer(px, py)
                if side_before != 0:
                    self._outer_side_before[track_id] = side_before
                    self._outer_frame[track_id] = frame_no
                    self._state[track_id] = self._SAW_OUTER
            return False
        if state == self._IDLE:
            if t_outer is not None:
                side_before = self._side_of_outer(px, py)
                if side_before == 0:
                    return False
                self._outer_side_before[track_id] = side_before
                self._outer_frame[track_id] = frame_no
                self._state[track_id] = self._SAW_OUTER
            return False
        if state == self._SAW_OUTER:
            elapsed = frame_no - self._outer_frame.get(track_id, frame_no)
            if elapsed > ENTRY_COUNTER_SAW_OUTER_TIMEOUT_FRAMES:
                self._reset(track_id)
                return False
            if t_outer is not None:
                side_now = self._side_of_outer(px, py)
                orig_side = self._outer_side_before.get(track_id, 0)
                if orig_side != 0 and side_now != 0 and side_now * orig_side < 0:
                    self._reset(track_id)
                else:
                    self._outer_frame[track_id] = frame_no
                return False
            if t_inner is not None:
                side_before_inner = self._side_of_inner(px, py)
                orig_outer_side = self._outer_side_before.get(track_id, 0)
                if orig_outer_side != 0 and side_before_inner != 0 and orig_outer_side * side_before_inner > 0:
                    if frame_no >= self._cooldown_until_frame:
                        self.entry_count += 1
                        self._cooldown_until_frame = frame_no + ENTRY_LINE_COOLDOWN_FRAMES
                        self._state[track_id] = self._COUNTED
                        self._outer_side_before.pop(track_id, None)
                        self._outer_frame.pop(track_id, None)
                        return True
                    self._state[track_id] = self._COUNTED
                    self._outer_side_before.pop(track_id, None)
                    self._outer_frame.pop(track_id, None)
                    return False
                self._reset(track_id)
        return False

    def remove_track(self, track_id):
        self._prev_pos.pop(track_id, None)
        self._state.pop(track_id, None)
        self._outer_side_before.pop(track_id, None)
        self._outer_frame.pop(track_id, None)


def init_db(db_path):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS visits (id INTEGER PRIMARY KEY AUTOINCREMENT,
        store_name TEXT, timestamp TEXT, cam_id TEXT, track_id INTEGER, gender TEXT, age_group TEXT, raw_age REAL, event TEXT)"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS footfall (id INTEGER PRIMARY KEY AUTOINCREMENT,store_name TEXT,
        timestamp TEXT, cam_id TEXT, track_id INTEGER, type TEXT)"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS zone_events (id INTEGER PRIMARY KEY AUTOINCREMENT,
        store_name TEXT, timestamp TEXT, cam_id TEXT, zone TEXT, track_id INTEGER, direction TEXT,
        gender TEXT, age_group TEXT)"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT,
        store_name TEXT, camera_id TEXT NOT NULL, zone_name TEXT NOT NULL, alert_label TEXT NOT NULL,
        triggered_at TEXT NOT NULL, resolved_at TEXT, duration_min REAL DEFAULT 0.0,
        role_type TEXT DEFAULT NULL)"""
    )
    conn.commit()
    conn.close()


def fetch_all_camera_configs(api_url):
    try:
        with urllib.request.urlopen(api_url, timeout=8) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return []
    cameras = []
    for store in data:
        store_name = store.get("store_name", "default")
        for cam in store.get("cameras", []):
            if not cam.get("active", False):
                continue
            camera_id = cam.get("camera_id", "unknown")
            rtsp_url = cam.get("url", "")
            cam_gender_age = cam.get("gender_age", False)
            line_configs = []
            for line in cam.get("lines", []):
                pts = line.get("points", [])
                if len(pts) >= 2:
                    line_configs.append(
                        {
                            "name": line.get("name", f"line_{len(line_configs)}"),
                            "outer": [tuple(p) for p in pts[0]],
                            "inner": [tuple(p) for p in pts[1]],
                            "effective_gen_age": cam_gender_age and line.get("gender_age", False),
                        }
                    )
            visit_zones = {}
            presence_zones = {}
            for zone in cam.get("zones", []):
                pts = zone.get("points", [])
                if len(pts) < 3:
                    continue
                name = zone.get("name", f"zone_{len(visit_zones) + len(presence_zones)}")
                detection_mode = zone.get("detection_mode", "visit")
                eff_gen = cam_gender_age and zone.get("gender_age", False)
                if detection_mode == "presence":
                    presence_zones[name] = {
                        "points": [tuple(p) for p in pts],
                        "max_absence_sec": float(zone.get("max_absence_sec", MAX_ABSENCE_SEC_DEFAULT)),
                        "min_presence_to_resolve_sec": float(zone.get("min_presence_to_resolve_sec", MIN_PRESENCE_TO_RESOLVE_SEC)),
                        "role_type": str(zone.get("role_type", "cashier")).lower().strip(),
                    }
                else:
                    visit_zones[name] = {"points": [tuple(p) for p in pts], "effective_gen_age": eff_gen}
            cameras.append(
                {
                    "camera_id": camera_id,
                    "store_name": store_name,
                    "rtsp_url": rtsp_url,
                    "line_configs": line_configs,
                    "visit_zones": visit_zones,
                    "presence_zones": presence_zones,
                    "gen_age_enabled": cam_gender_age,
                }
            )
    return cameras


class PersonDetector:
    def __init__(self, yolo):
        self.yolo = yolo
        self._half = torch.cuda.is_available()

    def detect(self, frame):
        boxes = []
        for r in self.yolo(
            frame,
            imgsz=YOLO_INPUT_SIZE,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            classes=[PERSON_CLASS],
            verbose=False,
            half=self._half,
        ):
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                if (x2 - x1) < 12 or (y2 - y1) < 12:
                    continue
                boxes.append((x1, y1, x2, y2, float(b.conf[0])))
        return boxes


class FaceDetectorYOLO:
    def __init__(self, model):
        self.model = model
        self._half = torch.cuda.is_available()

    def detect_faces(self, frame_bgr):
        faces = []
        for r in self.model(
            frame_bgr,
            imgsz=FACE_DET_SIZE,
            conf=FACE_DET_THRESH,
            verbose=False,
            half=self._half,
        ):
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                face_w = x2 - x1
                face_h = y2 - y1
                if face_w < MIN_FACE_SIZE // 2 or face_h < MIN_FACE_SIZE // 2:
                    continue
                faces.append({"bbox": (x1, y1, x2, y2)})
        return faces


class RTSPReader(threading.Thread):
    def __init__(self, url, frame_queue, stop_event):
        super().__init__(daemon=True, name=f"RTSPReader-{url[-20:]}")
        self.url = url
        self.queue = frame_queue
        self.stop_event = stop_event
        self.connected = threading.Event()
        self.reconnect_count = 0

    def _open_cap(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FULL_RES_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FULL_RES_H)
        return cap

    def run(self):
        while not self.stop_event.is_set():
            if self.reconnect_count > RECONNECT_MAX:
                break
            cap = self._open_cap()
            if not cap.isOpened():
                self.connected.clear()
                cap.release()
                self.reconnect_count += 1
                time.sleep(RECONNECT_DELAY)
                continue
            self.connected.set()
            self.reconnect_count = 0
            fail_streak = 0
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    fail_streak += 1
                    if fail_streak >= 15:
                        self.connected.clear()
                        break
                    time.sleep(0.01)
                    continue
                fail_streak = 0
                while not self.queue.empty():
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        break
                try:
                    self.queue.put_nowait(frame)
                except queue.Full:
                    pass
            cap.release()
            if not self.stop_event.is_set():
                self.reconnect_count += 1
                time.sleep(RECONNECT_DELAY)


class FPSMeter:
    def __init__(self, window=60):
        self._ts = deque(maxlen=window)

    def tick(self):
        self._ts.append(time.perf_counter())

    def get(self):
        if len(self._ts) < 2:
            return 0.0
        span = self._ts[-1] - self._ts[0]
        return (len(self._ts) - 1) / span if span > 0 else 0.0


def draw_person(frame, x1, y1, x2, y2, track_id, voter):
    gender = voter.gender
    is_stable = voter.is_stable_for_commit
    color = COLOR_MALE if gender == "Male" else COLOR_FEMALE if gender == "Female" else COLOR_UNKNOWN
    thickness = BOX_THICK + 1 if is_stable else BOX_THICK
    if voter.has_confirmed_gender:
        marker = "*" if is_stable else "~"
        mp = voter.mean_male_prob
        label = f"ID:{track_id} {gender}{marker} {voter.age} v={voter.total_votes}{f' p={mp:.2f}' if mp >= 0 else ''}"
    else:
        label = f"ID:{track_id} Person"
    (tw, th), bl = cv2.getTextSize(label, FONT, FONT_SCALE_LABEL, FONT_THICK)
    tag_y1 = max(0, y1 - th - bl - 6)
    cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - bl - 1), FONT, FONT_SCALE_LABEL, (255, 255, 255), FONT_THICK, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.circle(frame, ((x1 + x2) // 2, y2), 4, color, -1)


def draw_counting_lines(frame, line_configs):
    for lc in line_configs:
        ox1, oy1 = lc["outer"][0]
        ox2, oy2 = lc["outer"][1]
        cv2.line(frame, (ox1, oy1), (ox2, oy2), COLOR_OUTER_LINE, 2, cv2.LINE_AA)
        omx, omy = _line_midpoint(lc["outer"][0], lc["outer"][1])
        (otw, oth), obl = cv2.getTextSize("Outer", FONT, 0.55, 2)
        cv2.rectangle(frame, (omx - otw // 2 - 4, omy - oth - obl - 4), (omx + otw // 2 + 4, omy + 4), (0, 0, 0), -1)
        cv2.putText(frame, "Outer", (omx - otw // 2, omy - obl), FONT, 0.55, COLOR_OUTER_LINE, 2, cv2.LINE_AA)
        ix1, iy1 = lc["inner"][0]
        ix2, iy2 = lc["inner"][1]
        cv2.line(frame, (ix1, iy1), (ix2, iy2), COLOR_INNER_LINE, 3, cv2.LINE_AA)
        imx, imy = _line_midpoint(lc["inner"][0], lc["inner"][1])
        (itw, ith), ibl = cv2.getTextSize("Inner", FONT, 0.55, 2)
        cv2.rectangle(frame, (imx - itw // 2 - 4, imy - ith - ibl - 4), (imx + itw // 2 + 4, imy + 4), (0, 0, 0), -1)
        cv2.putText(frame, "Inner", (imx - itw // 2, imy - ibl), FONT, 0.55, COLOR_INNER_LINE, 2, cv2.LINE_AA)
        cv2.putText(frame, lc["name"], (ox1 + 6, oy1 - 8), FONT, 0.48, COLOR_OUTER_LINE, 1, cv2.LINE_AA)


def draw_hud(frame, camera_id, n_persons, fps, latency_ms, is_live, line_counters, zone_counter, presence_monitor, store_name=""):
    n_lines = len(line_counters)
    n_visit = len(zone_counter._polygons) if zone_counter else 0
    n_presence = len(presence_monitor._polygons) if presence_monitor else 0
    hud_h = 155 + n_lines * 24 + n_visit * 22 + n_presence * 22
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (460, hud_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    dot_color = (0, 210, 70) if is_live else (30, 30, 220)
    cv2.circle(frame, (16, 16), 7, dot_color, -1)
    cv2.putText(frame, "LIVE" if is_live else "RECONNECTING...", (30, 21), FONT, 0.52, dot_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Store  : {store_name}", (10, 42), FONT, FONT_SCALE_HUD, (0, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Camera : {camera_id}", (10, 62), FONT, FONT_SCALE_HUD, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Persons: {n_persons}", (10, 82), FONT, FONT_SCALE_HUD, COLOR_HUD, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS    : {fps:.1f}", (10, 102), FONT, FONT_SCALE_HUD, COLOR_HUD, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Latency: {latency_ms:.0f} ms", (10, 122), FONT, FONT_SCALE_HUD, COLOR_HUD, 1, cv2.LINE_AA)
    y = 146
    for line_name, ec in line_counters.items():
        cv2.putText(frame, f"Entries [{line_name}] : {ec.entry_count}", (10, y), FONT, FONT_SCALE_HUD, (0, 230, 120), 2, cv2.LINE_AA)
        y += 24
    if zone_counter:
        for zone_name, cnts in zone_counter.counts.items():
            cv2.putText(frame, f"{zone_name} IN:{cnts['entries']}", (10, y), FONT, 0.48, (0, 200, 255), 1, cv2.LINE_AA)
            y += 22
    if presence_monitor:
        now_mono = time.monotonic()
        for zone_name, st in presence_monitor.states.items():
            if st.currently_occupied:
                status = "PRESENT"
                color = (0, 255, 80)
            else:
                absent_min = (now_mono - st.absent_since_mono) / 60.0 if st.absent_since_mono > 0 else 0.0
                status = f"ABSENT {absent_min:.1f}m"
                color = (0, 0, 255) if not st.is_alerting else (0, 0, 200)
            label = f"{zone_name}: {status}"
            if st.is_alerting:
                alert_min = (now_mono - st.alert_triggered_at_mono) / 60.0 if st.alert_triggered_at_mono > 0 else 0.0
                label += f" [ALERT {alert_min:.1f}m]" if st.alert_triggered_at_mono > 0 else " [ALERT]"
            cv2.putText(frame, label, (10, y), FONT, 0.48, color, 1, cv2.LINE_AA)
            y += 22
    cv2.putText(frame, "Q/ESC:quit", (10, frame.shape[0] - 10), FONT, 0.42, (120, 120, 120), 1, cv2.LINE_AA)


def print_entry_event(tracking_id, entry_count, gender, age, camera_id, line_name, votes, stable, store_name=""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(
        f"[ENTRY] store={store_name} | camera_id={camera_id} | line={line_name} | "
        f"tracking_id={tracking_id} | entry={entry_count} | timestamp={ts} | "
        f"gender={gender} | age={age} | votes={votes} | stable={stable}"
    )


def resolve_gender_for_commit(voter):
    if voter is None:
        return "Unknown", False
    if voter.is_stable_for_commit:
        return voter.gender, True
    if voter.gender != "Unknown":
        return voter.gender, False
    return "Unknown", False


class CameraPipeline:
    def __init__(
        self,
        cam_cfg,
        db_path,
        no_display=False,
        shared_yolo=None,
        shared_face_yolo=None,
        shared_classifier=None,
    ):
        self.camera_id = cam_cfg["camera_id"]
        self.store_name = cam_cfg.get("store_name", "")
        self.rtsp_url = cam_cfg["rtsp_url"]
        self.line_configs = cam_cfg["line_configs"]
        self.visit_zones_cfg = cam_cfg["visit_zones"]
        self.presence_zones_cfg = cam_cfg["presence_zones"]
        self.gen_age_enabled = cam_cfg["gen_age_enabled"]
        self.db_path = db_path
        self.no_display = no_display
        self._shared_yolo = shared_yolo
        self._shared_face_yolo = shared_face_yolo
        self._shared_classifier = shared_classifier
        self.line_counters = {lc["name"]: EntryCounter(lc["outer"], lc["inner"]) for lc in self.line_configs}
        self.zone_counter = ZoneCounter(self.visit_zones_cfg) if self.visit_zones_cfg else None
        self.presence_monitor = None
        self.tracker = BoTSORT()
        self.voters = {}
        self.db_written_face = set()
        self.pending_entry_commits = {}
        self.entries_emitted = set()
        self.active_ids = set()
        self._footfall_tick = {}
        self._face_absence_counter = {}
        self._classifier_skip = {}
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.stop_event = threading.Event()
        self.reader = RTSPReader(self.rtsp_url, self.frame_queue, self.stop_event)
        self.fps_meter = FPSMeter()
        self.frame_count = 0
        self.last_display = None
        self.last_n = 0
        self.window_name = f"[{self.store_name}] Cam: {self.camera_id}"
        self.db_writer = None
        self.yolo = self.face_yolo = None
        self.face_detector = self.person_detector = self.classifier = self.device = None

    def start(self):
        self.db_writer = AsyncDBWriter(self.db_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.yolo = self._shared_yolo if self._shared_yolo is not None else _load_yolo_internal()
        self.person_detector = PersonDetector(self.yolo)
        if self.gen_age_enabled:
            self.face_yolo = self._shared_face_yolo if self._shared_face_yolo is not None else _load_face_yolo_internal()
            self.face_detector = FaceDetectorYOLO(self.face_yolo)
            self.classifier = self._shared_classifier if self._shared_classifier is not None else GenderAgeClassifier(self.device)
        if self.presence_zones_cfg:
            self.presence_monitor = StaffPresenceMonitor(self.presence_zones_cfg, self.camera_id, self.store_name, self.db_writer)
        self.reader.start()
        if not self.reader.connected.wait(timeout=20.0):
            self.stop_event.set()
            return False
        if not self.no_display and gui_available():
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, FULL_RES_W, FULL_RES_H)
        return True

    def stop(self):
        if self.presence_monitor is not None:
            self.presence_monitor.close_open_alerts()
        self.stop_event.set()
        self.reader.join(timeout=4.0)
        if self.db_writer is not None:
            self.db_writer.stop()
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _build_zone_tracks(self, tracks):
        return [
            {
                "id": int(tr[4]),
                "bbox": (int(tr[0]), int(tr[1]), int(tr[2]), int(tr[3])),
                "coasted": bool(tr[5]),
                "gen_age_enabled": self.gen_age_enabled,
                "voter": self.voters.get(int(tr[4])),
            }
            for tr in tracks
        ]

    def _flush_entry_commit(self, tid, line_name, voter, write_gender):
        key = (tid, line_name)
        if key in self.entries_emitted:
            self.pending_entry_commits.pop(key, None)
            return
        pec = self.pending_entry_commits.pop(key, None)
        if pec is None:
            return
        cg, stable = resolve_gender_for_commit(voter)
        final_gender = cg if write_gender else "Unknown"
        conf_age = voter.age if voter and voter.has_confirmed_age and write_gender else "?"
        conf_raw = voter.raw_age if voter and voter.has_confirmed_age and write_gender else -1.0
        print_entry_event(
            tid,
            pec["entry_count"],
            final_gender,
            conf_age,
            self.camera_id,
            line_name,
            voter.total_votes if voter else 0,
            stable,
            self.store_name,
        )
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.db_writer.execute(
            "INSERT INTO visits (store_name, timestamp, cam_id, track_id, gender, age_group, raw_age, event) VALUES (?,?,?,?,?,?,?,?)",
            (self.store_name, ts, self.camera_id, tid, final_gender, conf_age, conf_raw, "entry"),
        )
        self.db_writer.execute(
            "INSERT INTO footfall (timestamp,cam_id,track_id,type) VALUES (?,?,?,?)",
            (ts, self.camera_id, tid, "entry"),
        )
        self.entries_emitted.add(key)

    def process_one(self):
        try:
            raw = self.frame_queue.get(timeout=6.0)
        except queue.Empty:
            if not self.reader.is_alive():
                return False, None
            if self.last_display is not None and not self.no_display:
                disp = self.last_display.copy()
                draw_hud(
                    disp,
                    self.camera_id,
                    self.last_n,
                    self.fps_meter.get(),
                    0,
                    False,
                    self.line_counters,
                    self.zone_counter,
                    self.presence_monitor,
                    self.store_name,
                )
                return True, disp
            return True, None

        self.frame_count += 1
        t0 = time.perf_counter()
        fh, fw = raw.shape[:2]
        full_res_frame = (
            cv2.resize(raw, (FULL_RES_W, FULL_RES_H), interpolation=cv2.INTER_LINEAR)
            if (fw != FULL_RES_W or fh != FULL_RES_H)
            else raw.copy()
        )

        if self.frame_count % max(1, PROC_EVERY) == 0 or self.last_display is None:
            yolo_frame = cv2.resize(full_res_frame, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            scale_x = FULL_RES_W / YOLO_INPUT_SIZE
            scale_y = FULL_RES_H / YOLO_INPUT_SIZE

            pboxes = [
                (int(bx1 * scale_x), int(by1 * scale_y), int(bx2 * scale_x), int(by2 * scale_y), c)
                for bx1, by1, bx2, by2, c in self.person_detector.detect(yolo_frame)
            ]
            dets = (
                np.array([[x1, y1, x2, y2, c] for x1, y1, x2, y2, c in pboxes], dtype=float)
                if pboxes
                else np.empty((0, 5))
            )

            self.tracker.set_frame(full_res_frame)
            raw_tracks = self.tracker.update(dets)
            tracks = _suppress_duplicate_tracks(raw_tracks)

            face_boxes = []
            if self.gen_age_enabled and self.face_detector is not None:
                face_boxes = [f["bbox"] for f in self.face_detector.detect_faces(full_res_frame)]

            track_face_map = {}
            if face_boxes and len(tracks) > 0:
                active_tracks_for_face = [
                    (int(t[4]), (int(t[0]), int(t[1]), int(t[2]), int(t[3])))
                    for t in tracks if not bool(t[5])
                ]
                track_face_map = match_face_to_bbox_batch(active_tracks_for_face, face_boxes)

            if self.gen_age_enabled and self.classifier is not None:
                batch_items = []
                for track in tracks:
                    if bool(track[5]):
                        continue
                    tid = int(track[4])
                    skip_count = self._classifier_skip.get(tid, 0)
                    if skip_count > 0:
                        self._classifier_skip[tid] = skip_count - 1
                        continue
                    tbbox = (int(track[0]), int(track[1]), int(track[2]), int(track[3]))
                    face_bbox = track_face_map.get(tid)
                    is_back = _is_true_back_view(tbbox, face_bbox, full_res_frame.shape)
                    if is_back and face_bbox is None:
                        self._face_absence_counter[tid] = self._face_absence_counter.get(tid, 0) + 1
                        if self._face_absence_counter[tid] > BACK_VIEW_FACE_PRESENCE_WINDOW:
                            self._classifier_skip[tid] = CLASSIFIER_SKIP_FRAMES
                            continue
                        batch_items.append((tid, tbbox, None))
                        continue
                    if face_bbox is None and not is_back:
                        crop = full_res_frame[tbbox[1]:tbbox[3], tbbox[0]:tbbox[2]]
                        if not _is_sharp_enough(crop):
                            self._classifier_skip[tid] = CLASSIFIER_SKIP_FRAMES
                            continue
                        batch_items.append((tid, tbbox, None))
                        continue
                    self._face_absence_counter[tid] = 0
                    crop = full_res_frame[tbbox[1]:tbbox[3], tbbox[0]:tbbox[2]]
                    if not _is_sharp_enough(crop):
                        continue
                    batch_items.append((tid, tbbox, face_bbox))
                if batch_items:
                    for result in self.classifier.predict_batch(full_res_frame, batch_items):
                        tid, gender_label, raw_age, male_prob, has_face = result
                        if tid not in self.voters:
                            self.voters[tid] = VotingClassifier()
                        self.voters[tid].update(
                            male_prob,
                            get_age_bucket(raw_age),
                            float(raw_age),
                            has_face=has_face,
                            gender_label=gender_label,
                        )
                        voter = self.voters[tid]
                        if voter.is_stable_for_commit:
                            self._classifier_skip[tid] = CLASSIFIER_SKIP_FRAMES

            current_ids = set()
            for track in tracks:
                tid = int(track[4])
                x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
                coasted = bool(track[5])
                current_ids.add(tid)
                if tid not in self.voters:
                    self.voters[tid] = VotingClassifier()
                voter = self.voters[tid]
                foot_cx = float((x1 + x2) / 2)
                foot_cy = float(y2)
                for line_name, counter in self.line_counters.items():
                    if counter.update(tid, foot_cx, foot_cy, self.frame_count, coasted=coasted):
                        key = (tid, line_name)
                        if key not in self.entries_emitted and key not in self.pending_entry_commits:
                            self.pending_entry_commits[key] = {
                                "entry_count": counter.entry_count,
                                "frame_budget": ENTRY_COMMIT_FRAME_BUDGET,
                            }
                pending_keys_for_tid = [(tid, ln) for ln in self.line_counters if (tid, ln) in self.pending_entry_commits]
                for key in pending_keys_for_tid:
                    pec = self.pending_entry_commits[key]
                    pec["frame_budget"] -= 1
                    cg, is_fully_stable = resolve_gender_for_commit(voter)
                    if is_fully_stable or pec["frame_budget"] <= 0:
                        self._flush_entry_commit(tid, key[1], voter, self.gen_age_enabled)
                if not coasted:
                    last_ft = self._footfall_tick.get(tid, -FOOTFALL_DETECT_INTERVAL)
                    if self.frame_count - last_ft >= FOOTFALL_DETECT_INTERVAL:
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        self.db_writer.execute(
                            "INSERT INTO footfall (timestamp,cam_id,track_id,type) VALUES (?,?,?,?)",
                            (ts, self.camera_id, tid, "detected"),
                        )
                        self._footfall_tick[tid] = self.frame_count
                    if voter.is_stable_for_commit and tid not in self.db_written_face:
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        self.db_writer.execute(
                            "INSERT INTO visits (store_name, timestamp, cam_id, track_id, gender, age_group, raw_age, event) VALUES (?,?,?,?,?,?,?,?)",
                            (self.store_name, ts, self.camera_id, tid, voter.gender, voter.age, voter.raw_age, "face_confirmed"),
                        )
                        self.db_written_face.add(tid)
                    draw_person(full_res_frame, x1, y1, x2, y2, tid, voter)

            for gid in list(self.active_ids - current_ids):
                voter_gone = self.voters.get(gid)
                for line_name in list(self.line_counters):
                    if (gid, line_name) in self.pending_entry_commits:
                        self._flush_entry_commit(gid, line_name, voter_gone, self.gen_age_enabled)
                if voter_gone and gid not in self.db_written_face:
                    if voter_gone.has_confirmed_gender or voter_gone.has_confirmed_age:
                        cg, _ = resolve_gender_for_commit(voter_gone)
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        self.db_writer.execute(
                            "INSERT INTO visits (store_name, timestamp, cam_id, track_id, gender, age_group, raw_age, event) VALUES (?,?,?,?,?,?,?,?)",
                            (
                                self.store_name, ts, self.camera_id, gid,
                                cg if self.gen_age_enabled else "Unknown",
                                voter_gone.age if voter_gone.has_confirmed_age and self.gen_age_enabled else "?",
                                voter_gone.raw_age if voter_gone.has_confirmed_age and self.gen_age_enabled else -1.0,
                                "lost_with_partial_face",
                            ),
                        )
                ts_lost = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.db_writer.execute(
                    "INSERT INTO footfall (timestamp,cam_id,track_id,type) VALUES (?,?,?,?)",
                    (ts_lost, self.camera_id, gid, "lost"),
                )
                for ec in self.line_counters.values():
                    ec.remove_track(gid)
                self.voters.pop(gid, None)
                self.db_written_face.discard(gid)
                self._footfall_tick.pop(gid, None)
                self._face_absence_counter.pop(gid, None)
                self._classifier_skip.pop(gid, None)
                for line_name in list(self.line_counters):
                    self.entries_emitted.discard((gid, line_name))

            self.active_ids = current_ids
            zone_tracks = self._build_zone_tracks(tracks)

            if self.zone_counter is not None:
                for ev in self.zone_counter.update(zone_tracks):
                    cnts = self.zone_counter.counts.get(ev.zone, {})
                    eff_gen = cnts.get("effective_gen_age", False)
                    voter = self.voters.get(ev.track_id)
                    self.db_writer.execute(
                        "INSERT INTO zone_events (store_name, timestamp, cam_id, zone, track_id, direction, gender, age_group) VALUES (?,?,?,?,?,?,?,?)",
                        (
                            self.store_name, ev.timestamp, self.camera_id, ev.zone, ev.track_id, ev.direction,
                            voter.gender if voter and eff_gen else None,
                            voter.age if voter and voter.has_confirmed_age and eff_gen else None,
                        ),
                    )
                self.zone_counter.draw(full_res_frame)

            if self.presence_monitor is not None:
                self.presence_monitor.update(zone_tracks)
                self.presence_monitor.draw(full_res_frame)

            draw_counting_lines(full_res_frame, self.line_configs)
            self.last_display = full_res_frame
            self.last_n = len([t for t in tracks if not bool(t[5])])
            self.fps_meter.tick()

        latency_ms = (time.perf_counter() - t0) * 1000
        hud_frame = self.last_display.copy() if self.last_display is not None else full_res_frame.copy()
        draw_hud(
            hud_frame,
            self.camera_id,
            self.last_n,
            self.fps_meter.get(),
            latency_ms,
            self.reader.connected.is_set(),
            self.line_counters,
            self.zone_counter,
            self.presence_monitor,
            self.store_name,
        )
        return True, hud_frame


def camera_process_worker(cam_cfg: dict, db_path: str, no_display: bool):
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()

    rtsp_url = cam_cfg.get("rtsp_url", "")
    if not rtsp_url:
        return

    if not _check_camera_alive(rtsp_url, timeout=10):
        print(f"[WARN] Camera {cam_cfg.get('camera_id')} not reachable, skipping.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_yolo = _load_yolo_internal()
    shared_face_yolo = None
    shared_classifier = None

    if cam_cfg.get("gen_age_enabled", False):
        shared_face_yolo = _load_face_yolo_internal()
        shared_classifier = GenderAgeClassifier(device)

    pipeline = CameraPipeline(
        cam_cfg,
        db_path,
        no_display,
        shared_yolo=shared_yolo,
        shared_face_yolo=shared_face_yolo,
        shared_classifier=shared_classifier,
    )

    if not pipeline.start():
        return

    can_show = not no_display and gui_available()
    stop_flag = threading.Event()

    try:
        while not stop_flag.is_set() and not pipeline.stop_event.is_set():
            ok, frame = pipeline.process_one()
            if not ok:
                break
            if frame is not None and can_show:
                cv2.imshow(pipeline.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    stop_flag.set()
                    break
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if can_show:
            try:
                cv2.destroyWindow(pipeline.window_name)
            except cv2.error:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_all_cameras(api_url=CONFIG_API_URL, no_display=False):
    all_cam_cfgs = fetch_all_camera_configs(api_url)
    if not all_cam_cfgs:
        print("[ERROR] No active camera configs found from API.")
        return

    db_path = r"/home/rajan/store_pulse/database/analytics.db"
    init_db(db_path)

    effective_no_display = no_display or not gui_available()

    processes = []
    for cam_cfg in all_cam_cfgs:
        p = multiprocessing.Process(
            target=camera_process_worker,
            args=(cam_cfg, db_path, effective_no_display),
            name=f"CameraWorker-{cam_cfg['camera_id']}",
        )
        p.start()
        processes.append((cam_cfg["camera_id"], p))

    if not processes:
        return

    try:
        while True:
            all_dead = all(not p.is_alive() for _, p in processes)
            if all_dead:
                break
            if not effective_no_display:
                try:
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                except cv2.error:
                    effective_no_display = True
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        for cid, p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=8.0)
                if p.is_alive():
                    p.kill()
        if not effective_no_display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


def main():
    global YOLO_CONF, YOLO_IOU
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=YOLO_CONF)
    parser.add_argument("--iou", type=float, default=YOLO_IOU)
    parser.add_argument("--api-url", type=str, default=CONFIG_API_URL)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    YOLO_CONF = args.conf
    YOLO_IOU = args.iou
    multiprocessing.set_start_method("spawn", force=True)
    run_all_cameras(api_url=args.api_url, no_display=args.no_display)


if __name__ == "__main__":
    main()
