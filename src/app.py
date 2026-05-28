from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, get_jwt
import sqlite3
from datetime import datetime
import cv2
import json
import pandas as pd
import gc
import hashlib

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'your-super-secret-key-change-this-in-production'
jwt = JWTManager(app)

CORS(app, resources={
    r"/api/*": {
        "origins": [
            
            "http://217.18.55.78:5173",
            "http://217.18.55.78:5000"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

DB_PATH = "/home/rajan/store_pulse/database/analytics.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS stores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_name TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_name TEXT NOT NULL,
            name TEXT NOT NULL,
            url TEXT NOT NULL,
            webrtc_url TEXT,
            location TEXT NOT NULL,
            reid INTEGER DEFAULT 0,
            gender_age INTEGER DEFAULT 0,
            activate INTEGER DEFAULT 1,
            tested INTEGER DEFAULT 0,
            connection_status TEXT DEFAULT 'unknown',
            last_tested_at TEXT DEFAULT NULL,
            FOREIGN KEY (store_name) REFERENCES stores(store_name) ON DELETE CASCADE ON UPDATE CASCADE
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS counting_lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_name TEXT NOT NULL,
            cam_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            points TEXT NOT NULL,
            reid INTEGER DEFAULT 0,
            gender_age INTEGER DEFAULT 0,
            FOREIGN KEY (cam_id) REFERENCES cameras(id) ON DELETE CASCADE,
            FOREIGN KEY (store_name) REFERENCES stores(store_name) ON DELETE CASCADE ON UPDATE CASCADE
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS detection_zones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_name TEXT NOT NULL,
            cam_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            points TEXT NOT NULL,
            reid INTEGER DEFAULT 0,
            gender_age INTEGER DEFAULT 0,
            closed INTEGER DEFAULT 1,
            detection_mode TEXT DEFAULT 'visit',
            role_type TEXT DEFAULT NULL,
            FOREIGN KEY (cam_id) REFERENCES cameras(id) ON DELETE CASCADE,
            FOREIGN KEY (store_name) REFERENCES stores(store_name) ON DELETE CASCADE ON UPDATE CASCADE
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_name TEXT,
            timestamp TEXT NOT NULL,
            gender TEXT,
            age_group TEXT,
            event TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS zone_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_name TEXT,
            timestamp TEXT NOT NULL,
            cam_id INTEGER,
            zone_name TEXT,
            track_id INTEGER,
            event TEXT,
            direction TEXT,
            zone TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_name TEXT NOT NULL,
            camera_id INTEGER,
            zone_name TEXT,
            alert_label TEXT,
            triggered_at TEXT,
            resolved_at TEXT,
            duration_min REAL,
            FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
        )
    """)

    conn.commit()

    column_migrations = [
        ("visits", "store_name", "TEXT"),
        ("visits", "gender", "TEXT"),
        ("visits", "age_group", "TEXT"),
        ("visits", "event", "TEXT"),
        ("zone_events", "store_name", "TEXT"),
        ("zone_events", "cam_id", "INTEGER"),
        ("zone_events", "zone_name", "TEXT"),
        ("zone_events", "track_id", "INTEGER"),
        ("zone_events", "event", "TEXT"),
        ("zone_events", "direction", "TEXT"),
        ("zone_events", "zone", "TEXT"),
        ("cameras", "connection_status", "TEXT DEFAULT 'unknown'"),
        ("cameras", "last_tested_at", "TEXT DEFAULT NULL"),
        ("cameras", "tested", "INTEGER DEFAULT 0"),
        ("cameras", "activate", "INTEGER DEFAULT 1"),
        ("cameras", "reid", "INTEGER DEFAULT 0"),
        ("cameras", "gender_age", "INTEGER DEFAULT 0"),
        ("cameras", "webrtc_url", "TEXT"),
        ("detection_zones", "detection_mode", "TEXT DEFAULT 'visit'"),
        ("detection_zones", "role_type", "TEXT DEFAULT NULL"),
        ("detection_zones", "closed", "INTEGER DEFAULT 1"),
        ("alerts", "duration_min", "REAL"),
    ]

    for table, column, col_type in column_migrations:
        try:
            existing_cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if column not in existing_cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                conn.commit()
        except Exception:
            pass

    admin_password_hash = hashlib.sha256('admin'.encode()).hexdigest()
    conn.execute(
        "INSERT OR IGNORE INTO stores (store_name, email, password, created_at) VALUES (?, ?, ?, ?)",
        ('Admin', 'admin@123', admin_password_hash, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


init_db()
ADMIN_STORE_NAME = 'Admin'


def resolve_store_name(cam_id=None):
    claims = get_jwt()
    is_admin = claims.get("is_admin", False)
    identity = get_jwt_identity()

    if not is_admin:
        return identity

    store = request.args.get('store', '').strip()

    if not store:
        try:
            body = request.get_json(silent=True) or {}
            store = body.get('store_name', '').strip()
        except Exception:
            pass

    if not store and cam_id is not None:
        conn = get_connection()
        row = conn.execute("SELECT store_name FROM cameras WHERE id=?", (cam_id,)).fetchone()
        conn.close()
        if row:
            store = row['store_name']

    return store if store else identity


@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    store_name = data.get('store_name')
    email = data.get('email')
    password = data.get('password')
    if not all([store_name, email, password]):
        return jsonify({"error": "All fields required"}), 400
    hashed = hashlib.sha256(password.encode()).hexdigest()
    try:
        conn = get_connection()
        conn.execute(
            "INSERT INTO stores (store_name, email, password, created_at) VALUES (?, ?, ?, ?)",
            (store_name, email, hashed, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        return jsonify({"message": "Store registered successfully"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Store name or email already exists"}), 409


@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    hashed = hashlib.sha256(password.encode()).hexdigest()
    conn = get_connection()
    user = conn.execute(
        "SELECT * FROM stores WHERE email = ? AND password = ?", (email, hashed)
    ).fetchone()
    conn.close()
    if user:
        is_admin = user['store_name'] == ADMIN_STORE_NAME
        access_token = create_access_token(
            identity=user['store_name'],
            additional_claims={"is_admin": is_admin}
        )
        return jsonify({
            "token": access_token,
            "store_name": user['store_name'],
            "is_admin": is_admin
        })
    return jsonify({"error": "Invalid credentials"}), 401


@app.route('/api/stores', methods=['GET'])
@jwt_required()
def get_stores():
    claims = get_jwt()
    is_admin = claims.get("is_admin", False)
    conn = get_connection()
    if is_admin:
        stores = conn.execute(
            "SELECT id, store_name FROM stores WHERE store_name != ? ORDER BY store_name",
            (ADMIN_STORE_NAME,)
        ).fetchall()
    else:
        store_name = get_jwt_identity()
        stores = conn.execute(
            "SELECT id, store_name FROM stores WHERE store_name = ?", (store_name,)
        ).fetchall()
    conn.close()
    return jsonify([dict(s) for s in stores])


@app.route('/api/cameras', methods=['GET'])
@jwt_required()
def get_cameras():
    store_name = resolve_store_name()
    conn = get_connection()
    cams = conn.execute(
        "SELECT * FROM cameras WHERE store_name=? ORDER BY id", (store_name,)
    ).fetchall()
    conn.close()
    gc.collect()
    return jsonify([dict(c) for c in cams])


@app.route('/api/cameras', methods=['POST'])
@jwt_required()
def add_camera():
    claims = get_jwt()
    is_admin = claims.get("is_admin", False)
    identity = get_jwt_identity()
    data = request.get_json()

    if not data or not all(k in data for k in ['name', 'url', 'location']):
        return jsonify({"success": False, "message": "Missing required fields"}), 400

    if is_admin:
        store_name = data.get('store_name', '').strip() or request.args.get('store', '').strip() or identity
    else:
        store_name = identity

    conn = get_connection()
    existing = conn.execute(
        "SELECT id FROM cameras WHERE store_name=? AND (LOWER(TRIM(name))=LOWER(TRIM(?)) OR LOWER(TRIM(url))=LOWER(TRIM(?)))",
        (store_name, data['name'], data['url'])
    ).fetchone()
    if existing:
        conn.close()
        gc.collect()
        return jsonify({"success": False, "message": "A camera with the same name or URL already exists"}), 409

    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO cameras (store_name, name, url, location, webrtc_url, reid, gender_age, activate, tested, connection_status, last_tested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            store_name, data['name'], data['url'], data['location'],
            data.get('webrtc_url', ''),
            data.get('reid', False), data.get('gender_age', False),
            data.get('activate', True),
            1 if data.get('tested') else 0,
            data.get('connection_status', 'unknown'),
            data.get('last_tested_at', None)
        )
    )
    conn.commit()
    new_id = cursor.lastrowid
    new_cam = conn.execute("SELECT * FROM cameras WHERE id=?", (new_id,)).fetchone()
    conn.close()
    gc.collect()
    return jsonify({"success": True, "camera": dict(new_cam)})


@app.route('/api/cameras/<int:cam_id>', methods=['PUT'])
@jwt_required()
def edit_camera(cam_id):
    store_name = resolve_store_name(cam_id=cam_id)
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data"}), 400

    conn = get_connection()
    existing = conn.execute(
        "SELECT id FROM cameras WHERE store_name=? AND (LOWER(TRIM(name))=LOWER(TRIM(?)) OR LOWER(TRIM(url))=LOWER(TRIM(?))) AND id!=?",
        (store_name, data.get('name', ''), data.get('url', ''), cam_id)
    ).fetchone()
    if existing:
        conn.close()
        gc.collect()
        return jsonify({"success": False, "message": "A camera with the same name or URL already exists"}), 409

    conn.execute(
        "UPDATE cameras SET name=?, url=?, location=?, webrtc_url=?, reid=?, gender_age=?, activate=?, tested=?, connection_status=?, last_tested_at=? WHERE id=? AND store_name=?",
        (
            data.get('name'), data.get('url'), data.get('location'),
            data.get('webrtc_url', ''),
            data.get('reid', False), data.get('gender_age', False),
            data.get('activate', True),
            1 if data.get('tested') else 0,
            data.get('connection_status', 'unknown'),
            data.get('last_tested_at', None),
            cam_id, store_name
        )
    )
    conn.commit()
    updated_cam = conn.execute(
        "SELECT * FROM cameras WHERE id=? AND store_name=?", (cam_id, store_name)
    ).fetchone()
    conn.close()
    gc.collect()
    return jsonify({"success": True, "camera": dict(updated_cam)})


@app.route('/api/cameras/<int:cam_id>/status', methods=['PUT'])
@jwt_required()
def update_camera_status(cam_id):
    store_name = resolve_store_name(cam_id=cam_id)
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data"}), 400
    conn = get_connection()
    conn.execute(
        "UPDATE cameras SET tested=?, connection_status=?, last_tested_at=? WHERE id=? AND store_name=?",
        (
            1 if data.get('tested') else 0,
            data.get('connection_status', 'unknown'),
            data.get('last_tested_at'),
            cam_id, store_name
        )
    )
    conn.commit()
    conn.close()
    gc.collect()
    return jsonify({"success": True})

@app.route('/api/cameras/<int:cam_id>/toggle', methods=['PUT'])
@jwt_required()
def toggle_camera_field(cam_id):
    data = request.get_json() or {}
    field = data.get('field')
    if field not in ('reid', 'gender_age', 'activate'):
        return jsonify({"success": False, "message": "Invalid field"}), 400
    value = 1 if data.get('value') else 0
    conn = get_connection()
    row = conn.execute("SELECT store_name FROM cameras WHERE id=?", (cam_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({"success": False, "message": "Camera not found"}), 404
    claims = get_jwt()
    if not claims.get("is_admin", False) and row['store_name'] != get_jwt_identity():
        conn.close()
        return jsonify({"success": False, "message": "Forbidden"}), 403
    conn.execute(f"UPDATE cameras SET {field}=? WHERE id=?", (value, cam_id))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "field": field, "value": bool(value)})
@app.route('/api/cameras/<int:cam_id>', methods=['DELETE'])
@jwt_required()
def delete_camera(cam_id):
    store_name = resolve_store_name(cam_id=cam_id)
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM cameras WHERE id=? AND store_name=?", (cam_id, store_name)
        )
        conn.commit()
        conn.close()
        gc.collect()
        return jsonify({"success": True, "message": "Camera and all related data deleted successfully"})
    except Exception as e:
        conn.close()
        gc.collect()
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/camera_snapshot/<int:cam_id>')
@jwt_required()
def camera_snapshot(cam_id):
    store_name = resolve_store_name(cam_id=cam_id)
    conn = get_connection()
    row = conn.execute(
        "SELECT url FROM cameras WHERE id=? AND store_name=?", (cam_id, store_name)
    ).fetchone()
    conn.close()
    if not row:
        gc.collect()
        return jsonify({"error": "Camera not found"}), 404
    url = row['url']
    try:
        source = int(url) if str(url).isdigit() else url
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            gc.collect()
            return jsonify({"error": "Cannot open stream"}), 400
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            gc.collect()
            return jsonify({"error": "Failed to read frame"}), 400
        frame = cv2.resize(frame, (640, 640))
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        del frame
        gc.collect()
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    except Exception as e:
        gc.collect()
        return jsonify({"error": str(e)}), 500


@app.route('/api/cameras/<int:cam_id>/lines', methods=['GET'])
@jwt_required()
def get_lines(cam_id):
    store_name = resolve_store_name(cam_id=cam_id)
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM counting_lines WHERE cam_id=? AND store_name=? ORDER BY id",
        (cam_id, store_name)
    ).fetchall()
    conn.close()
    data = []
    for r in rows:
        d = dict(r)
        d['points'] = json.loads(d['points'])
        data.append(d)
    gc.collect()
    return jsonify(data)


@app.route('/api/cameras/<int:cam_id>/lines', methods=['POST'])
@jwt_required()
def add_line(cam_id):
    store_name = resolve_store_name(cam_id=cam_id)
    data = request.get_json()
    if not data or not data.get('name') or not data.get('points'):
        return jsonify({"success": False, "message": "Invalid data"}), 400
    points_json = json.dumps(data['points'])
    conn = get_connection()
    conn.execute(
        "INSERT INTO counting_lines (store_name, cam_id, name, points, reid, gender_age) VALUES (?,?,?,?,?,?)",
        (store_name, cam_id, data['name'], points_json, data.get('reid', False), data.get('gender_age', False))
    )
    conn.commit()
    conn.close()
    gc.collect()
    return jsonify({"success": True})


@app.route('/api/cameras/<int:cam_id>/lines/<int:line_id>', methods=['DELETE'])
@jwt_required()
def delete_line(cam_id, line_id):
    store_name = resolve_store_name(cam_id=cam_id)
    conn = get_connection()
    conn.execute(
        "DELETE FROM counting_lines WHERE id=? AND cam_id=? AND store_name=?",
        (line_id, cam_id, store_name)
    )
    conn.commit()
    conn.close()
    gc.collect()
    return jsonify({"success": True})


@app.route('/api/cameras/<int:cam_id>/zones', methods=['GET'])
@jwt_required()
def get_zones(cam_id):
    store_name = resolve_store_name(cam_id=cam_id)
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM detection_zones WHERE cam_id=? AND store_name=? ORDER BY id",
        (cam_id, store_name)
    ).fetchall()
    conn.close()
    data = []
    for r in rows:
        d = dict(r)
        d['points'] = json.loads(d['points'])
        data.append(d)
    gc.collect()
    return jsonify(data)


@app.route('/api/cameras/<int:cam_id>/zones', methods=['POST'])
@jwt_required()
def add_zone(cam_id):
    store_name = resolve_store_name(cam_id=cam_id)
    data = request.get_json()
    if not data or not data.get('name') or not data.get('points'):
        return jsonify({"success": False, "message": "Invalid data"}), 400
    points_json = json.dumps(data['points'])
    conn = get_connection()
    conn.execute(
        "INSERT INTO detection_zones (store_name, cam_id, name, points, reid, gender_age, closed, detection_mode, role_type) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            store_name, cam_id, data['name'], points_json,
            data.get('reid', False), data.get('gender_age', False),
            data.get('closed', True),
            data.get('detection_mode', 'visit'),
            data.get('role_type', None)
        )
    )
    conn.commit()
    conn.close()
    gc.collect()
    return jsonify({"success": True})


@app.route('/api/cameras/<int:cam_id>/zones/<int:zone_id>', methods=['DELETE'])
@jwt_required()
def delete_zone(cam_id, zone_id):
    store_name = resolve_store_name(cam_id=cam_id)
    conn = get_connection()
    conn.execute(
        "DELETE FROM detection_zones WHERE id=? AND cam_id=? AND store_name=?",
        (zone_id, cam_id, store_name)
    )
    conn.commit()
    conn.close()
    gc.collect()
    return jsonify({"success": True})


@app.route('/api/analytics', methods=['GET'])
@jwt_required()
def analytics():
    store_name = resolve_store_name()
    selected_date = request.args.get("date")
    conn = get_connection()
    ze_cols_info = conn.execute("PRAGMA table_info(zone_events)").fetchall()
    ze_cols = [r[1] for r in ze_cols_info]
    zone_rows = conn.execute(
        "SELECT * FROM zone_events WHERE (store_name=? OR store_name IS NULL OR TRIM(store_name)='')",
        (store_name,)
    ).fetchall()
    alert_rows = conn.execute(
        """SELECT a.id, a.camera_id, a.zone_name, a.alert_label, a.triggered_at,
                  a.resolved_at, a.duration_min, a.store_name,
                  COALESCE(dz.role_type, 'cashier') AS role_type
           FROM alerts a
           LEFT JOIN detection_zones dz
             ON LOWER(TRIM(a.zone_name)) = LOWER(TRIM(dz.name))
            AND a.store_name = dz.store_name
           WHERE (a.store_name=? OR a.store_name IS NULL OR TRIM(a.store_name)='')
           ORDER BY a.triggered_at DESC""",
        (store_name,)
    ).fetchall()
    conn.close()

    empty_response = {
        "zoneFootfall": {}, "hourlyByZone": {}, "lastDayHourlyByZone": {},
        "weekdayVsWeekendByZone": {}, "peakHoursByZone": {}, "totalVisitorsToday": 0,
        "mostVisitedZone": {"zone": "N/A", "count": 0, "date": ""},
        "busiestDayThisWeek": {"day": "N/A", "count": 0, "date": ""},
        "counterSummary": [], "securitySummary": []
    }

    if not zone_rows or not ze_cols:
        gc.collect()
        return jsonify(empty_response)

    df = pd.DataFrame(zone_rows, columns=ze_cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        gc.collect()
        return jsonify(empty_response)

    _EVENT_MAP = {
        "entry": "entry", "enter": "entry", "entered": "entry", "in": "entry",
        "zone_in": "entry", "entry_event": "entry", "exit": "exit", "leave": "exit",
        "left": "exit", "out": "exit", "zone_out": "exit"
    }
    direction_col = next((c for c in ["direction", "event"] if c in df.columns), None)
    if not direction_col:
        gc.collect()
        return jsonify(empty_response)
    df["direction"] = df[direction_col].astype(str).str.strip().str.lower().map(lambda v: _EVENT_MAP.get(v, v))
    zone_col = next((c for c in ["zone", "zone_name"] if c in df.columns), None)
    if not zone_col:
        gc.collect()
        return jsonify(empty_response)
    df["zone"] = df[zone_col].astype(str).str.strip()
    df["_ts_date"] = df["timestamp"].dt.date
    df["is_weekend"] = df["timestamp"].dt.weekday >= 5
    df["hour"] = df["timestamp"].dt.strftime("%I %p").str.lstrip("0")

    today = datetime.now().date()
    target_date = pd.to_datetime(selected_date).date() if selected_date else today

    df_valid = df[df["direction"] == "entry"].copy()
    df_range = df_valid[df_valid["_ts_date"] == target_date].copy()
    df_last = df_valid[df_valid["_ts_date"] == (target_date - pd.Timedelta(days=1))].copy()
    week_start = target_date - pd.Timedelta(days=target_date.weekday())
    week_end = week_start + pd.Timedelta(days=6)
    df_week = df_valid[(df_valid["_ts_date"] >= week_start) & (df_valid["_ts_date"] <= week_end)].copy()

    zone_footfall = df_range.groupby("zone").size().astype(int).to_dict()

    def _hourly(grp):
        e = grp.groupby("hour").size()
        return {h: {"entry": int(c)} for h, c in e.items()}

    hourly_by_zone = {z: _hourly(g) for z, g in df_range.groupby("zone")}
    last_day_hourly_by_zone = {z: _hourly(g) for z, g in df_last.groupby("zone")}
    weekday_weekend_by_zone = {
        z: {"weekday": int((~g["is_weekend"]).sum()), "weekend": int(g["is_weekend"].sum())}
        for z, g in df_week.groupby("zone")
    } if not df_week.empty else {}

    peak_hours_by_zone = {}
    for z, g in df_range.groupby("zone"):
        hourly_entries = g.groupby("hour").size()
        peak_hours_by_zone[z] = {
            "peakHour": hourly_entries.idxmax() if not hourly_entries.empty else "N/A",
            "entryCount": int(hourly_entries.max()) if not hourly_entries.empty else 0,
            "date": str(target_date)
        }

    total_visitors_today = int(len(df_range))
    most_visited = df_range.groupby("zone").size().idxmax() if not df_range.empty else "N/A"
    mv_count = int(df_range.groupby("zone").size().max()) if not df_range.empty else 0

    daily_counts = df_week.groupby("_ts_date").size()
    busiest_day_name = daily_counts.idxmax().strftime("%A") if not daily_counts.empty else "N/A"
    busiest_day_count = int(daily_counts.max()) if not daily_counts.empty else 0
    busiest_day_date = str(daily_counts.idxmax()) if not daily_counts.empty else ""

    counter_summary = []
    security_summary = []

    if alert_rows:
        al_cols = ["id", "camera_id", "zone_name", "alert_label", "triggered_at", "resolved_at", "duration_min", "store_name", "role_type"]
        df_al = pd.DataFrame([dict(zip(al_cols, row)) for row in alert_rows])
        df_al["triggered_at"] = pd.to_datetime(df_al["triggered_at"], errors="coerce")
        df_al["resolved_at"] = pd.to_datetime(df_al["resolved_at"], errors="coerce")
        df_al["duration_min"] = pd.to_numeric(df_al["duration_min"], errors="coerce")
        df_al["_day"] = df_al["triggered_at"].dt.date
        df_al["role_type"] = df_al["role_type"].astype(str).str.lower().str.strip()
        df_al["camera_id"] = df_al["camera_id"].astype(str).str.strip()
        df_al["zone_name"] = df_al["zone_name"].astype(str).str.strip()

        df_al_day = df_al[df_al["_day"] == target_date].copy()

        if not df_al_day.empty:
            df_al_day = df_al_day.sort_values("triggered_at", ascending=False)
            for (camera_id, zone_name, role_type), group in df_al_day.groupby(["camera_id", "zone_name", "role_type"], sort=False):
                records = []
                for _, row in group.head(5).iterrows():
                    start_ts = row["triggered_at"]
                    end_ts = row["resolved_at"]
                    dur = row["duration_min"]
                    if pd.notna(end_ts):
                        resolved_str = end_ts.strftime("%d %b %Y, %I:%M %p")
                        is_ongoing = False
                    else:
                        resolved_str = "Ongoing"
                        is_ongoing = True
                    if pd.isna(dur) and pd.notna(start_ts) and pd.notna(end_ts):
                        dur = (end_ts - start_ts).total_seconds() / 60.0
                    records.append({
                        "day": str(row["_day"]),
                        "cameraId": str(camera_id),
                        "zone": str(zone_name),
                        "roleType": str(role_type),
                        "alertType": str(row["alert_label"]),
                        "triggeredTime": start_ts.strftime("%d %b %Y, %I:%M %p") if pd.notna(start_ts) else "N/A",
                        "resolvedTime": resolved_str,
                        "isOngoing": is_ongoing,
                        "durationMinutes": f"{round(float(dur), 1)}" if pd.notna(dur) else "N/A"
                    })
                if role_type == "security":
                    security_summary.extend(records)
                else:
                    counter_summary.extend(records)

    gc.collect()
    res = {
        "zoneFootfall": zone_footfall, "hourlyByZone": hourly_by_zone,
        "lastDayHourlyByZone": last_day_hourly_by_zone, "weekdayVsWeekendByZone": weekday_weekend_by_zone,
        "peakHoursByZone": peak_hours_by_zone, "totalVisitorsToday": total_visitors_today,
        "mostVisitedZone": {"zone": most_visited, "count": mv_count, "date": str(target_date)},
        "busiestDayThisWeek": {"day": busiest_day_name, "count": busiest_day_count, "date": busiest_day_date},
        "counterSummary": counter_summary, "securitySummary": security_summary
    }
    return jsonify(res)


@app.route('/api/dashboard_stats')
@jwt_required()
def dashboard():
    jwt_store = get_jwt_identity()
    claims = get_jwt()
    is_admin = claims.get("is_admin", False)
    requested_date = request.args.get('date')
    selected_store = request.args.get('store', '').strip()
    current_date = datetime.now().date()
    target_date = pd.Timestamp(requested_date).date() if requested_date else current_date
    is_today = (target_date == current_date)
    conn = get_connection()
    valid_store_names = [row[0] for row in conn.execute("SELECT store_name FROM stores").fetchall()]
    if is_admin:
        if selected_store and selected_store in valid_store_names:
            filter_store = selected_store
        else:
            filter_store = None
    else:
        filter_store = jwt_store
    visits_cols = [r[1] for r in conn.execute("PRAGMA table_info(visits)").fetchall()]
    has_store_name = "store_name" in visits_cols
    try:
        if not has_store_name or (is_admin and not filter_store):
            all_rows = conn.execute(
                "SELECT timestamp, gender, age_group, event FROM visits WHERE timestamp >= datetime(?, '-7 days') AND timestamp < datetime(?, '+1 day')",
                (str(target_date), str(target_date))
            ).fetchall()
        else:
            all_rows = conn.execute(
                "SELECT timestamp, gender, age_group, event FROM visits WHERE store_name=? AND timestamp >= datetime(?, '-7 days') AND timestamp < datetime(?, '+1 day')",
                (filter_store, str(target_date), str(target_date))
            ).fetchall()
    except Exception as e:
        conn.close()
        gc.collect()
        return jsonify({"error": str(e)}), 500
    conn.close()

    empty_response = {
        "footfall": 0, "openingTime": "N/A", "closingTime": "N/A", "busiestDay": "--",
        "busiestHour": "N/A", "malePercent": 0, "femalePercent": 0, "malePeak": "N/A",
        "femalePeak": "N/A", "ageGroups": {}, "genderAge": {}, "entryCount": 0,
        "hourlyTraffic": {}, "yesterdayHourlyTraffic": {}, "mostVisitedAgeGroup": "N/A",
        "weekdayVsWeekend": {"weekday": 0, "weekend": 0}, "last7Days": []
    }

    if not all_rows:
        gc.collect()
        return jsonify(empty_response)

    df_all = pd.DataFrame(all_rows, columns=['timestamp', 'gender', 'age_group', 'event'])
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], format='mixed', errors='coerce')
    df_all = df_all.dropna(subset=['timestamp'])
    df_all['gender'] = df_all['gender'].astype(str).str.lower().str.strip()
    df_all['age_group'] = df_all['age_group'].fillna('unknown').astype(str).str.strip()
    df_all['event'] = df_all['event'].astype(str).str.upper().str.strip()
    df_all['hour'] = df_all['timestamp'].dt.strftime('%I %p').str.lstrip('0')
    df_all['day'] = df_all['timestamp'].dt.strftime('%a, %d %b %Y')
    df_all['date'] = df_all['timestamp'].dt.date
    df_all['weekday'] = df_all['timestamp'].dt.weekday
    df_all['is_weekend'] = df_all['weekday'] >= 5
    df_all_valid = df_all[df_all['event'].isin(['ENTRY', 'EXIT']) & df_all['gender'].isin(['male', 'female'])].reset_index(drop=True)
    busiest_day = df_all_valid.groupby('day').size().idxmax() if not df_all_valid.empty else '--'
    df = df_all[df_all['date'] == target_date].reset_index(drop=True)
    df_valid = df[df['event'].isin(['ENTRY', 'EXIT']) & df['gender'].isin(['male', 'female'])].reset_index(drop=True)
    if df_valid.empty:
        gc.collect()
        empty_response['busiestDay'] = busiest_day
        return jsonify(empty_response)

    open_time = df_valid['timestamp'].min()
    close_time = df_valid['timestamp'].max()
    entry_df = df_valid[df_valid['event'] == 'ENTRY']
    total_entries = len(entry_df)
    total_footfall = len(df_valid)
    male_count = len(df_valid[df_valid['gender'] == 'male'])
    female_count = len(df_valid[df_valid['gender'] == 'female'])
    male_percent = round(male_count / total_footfall * 100) if total_footfall else 0
    female_percent = round(female_count / total_footfall * 100) if total_footfall else 0
    male_peak = entry_df[entry_df['gender'] == 'male']['hour'].value_counts().idxmax() if not entry_df[entry_df['gender'] == 'male'].empty else 'N/A'
    female_peak = entry_df[entry_df['gender'] == 'female']['hour'].value_counts().idxmax() if not entry_df[entry_df['gender'] == 'female'].empty else 'N/A'
    busiest_hour = df_valid['hour'].value_counts().idxmax() if not df_valid.empty else 'N/A'
    age_groups = df_valid['age_group'].value_counts().to_dict()
    most_visited_age_group = max(age_groups, key=age_groups.get) if age_groups else "N/A"
    gender_age = df_valid.groupby(['age_group', 'gender']).size().unstack(fill_value=0).to_dict(orient='index')
    gender_age = {age: {"male": int(v.get("male", 0)), "female": int(v.get("female", 0))} for age, v in gender_age.items()}
    hourly_entry = entry_df.groupby('hour').size().rename('entry')
    hourly_exit = df_valid[df_valid['event'] == 'EXIT'].groupby('hour').size().rename('exit')
    hourly_traffic = pd.concat([hourly_entry, hourly_exit], axis=1).fillna(0).astype(int).apply(
        lambda r: {"entry": int(r['entry']), "exit": int(r['exit'])}, axis=1
    ).to_dict()
    yesterday = target_date - pd.Timedelta(days=1)
    yesterday_valid = df_all[
        (df_all['date'] == yesterday) &
        df_all['event'].isin(['ENTRY', 'EXIT']) &
        df_all['gender'].isin(['male', 'female'])
    ]
    y_entry = yesterday_valid[yesterday_valid['event'] == 'ENTRY'].groupby('hour').size().rename('entry')
    y_exit = yesterday_valid[yesterday_valid['event'] == 'EXIT'].groupby('hour').size().rename('exit')
    yesterday_hourly = pd.concat([y_entry, y_exit], axis=1).fillna(0).astype(int).apply(
        lambda r: {"entry": int(r['entry']), "exit": int(r['exit'])}, axis=1
    ).to_dict()
    week_start = target_date - pd.Timedelta(days=target_date.weekday())
    week_df = df_all_valid[(df_all_valid['date'] >= week_start) & (df_all_valid['date'] <= target_date)]
    weekday_count = len(week_df[~week_df['is_weekend']])
    weekend_count = len(week_df[week_df['is_weekend']])
    last7 = []
    for i in range(1, 8):
        day_date = target_date - pd.Timedelta(days=i)
        day_df = df_all_valid[df_all_valid['date'] == day_date]
        day_total = len(day_df)
        day_male = len(day_df[day_df['gender'] == 'male'])
        day_female = len(day_df[day_df['gender'] == 'female'])
        day_peak_hour = day_df['hour'].value_counts().idxmax() if not day_df.empty else "--"
        day_peak_count = int(day_df['hour'].value_counts().max()) if not day_df.empty else ""
        opening_time = day_df['timestamp'].min().strftime("%I:%M %p").lstrip("0") if not day_df.empty else "--"
        closing_time = day_df['timestamp'].max().strftime("%I:%M %p").lstrip("0") if not day_df.empty else "--"
        last7.append({
            "date": day_date.strftime('%d %b %Y'), "total": int(day_total),
            "male": int(day_male), "female": int(day_female),
            "peakHour": day_peak_hour, "peakCount": day_peak_count,
            "openingTime": opening_time, "closingTime": closing_time
        })
    gc.collect()
    return jsonify({
        "footfall": int(total_footfall),
        "openingTime": open_time.strftime("%I:%M %p").lstrip("0") if pd.notna(open_time) else "N/A",
        "closingTime": "N/A" if is_today else (close_time.strftime("%I:%M %p").lstrip("0") if pd.notna(close_time) else "N/A"),
        "busiestDay": busiest_day, "busiestHour": busiest_hour,
        "malePercent": int(male_percent), "femalePercent": int(female_percent),
        "malePeak": male_peak, "femalePeak": female_peak,
        "ageGroups": {k: int(v) for k, v in age_groups.items()},
        "genderAge": gender_age, "entryCount": int(total_entries),
        "hourlyTraffic": hourly_traffic, "yesterdayHourlyTraffic": yesterday_hourly,
        "mostVisitedAgeGroup": most_visited_age_group,
        "weekdayVsWeekend": {"weekday": int(weekday_count), "weekend": int(weekend_count)},
        "last7Days": last7
    })


@app.route('/api/test_camera', methods=['POST'])
@jwt_required()
def test_camera():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"success": False, "message": "Missing camera URL"}), 400
    url = data.get('url')
    try:
        source = int(url) if str(url).isdigit() else url
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            gc.collect()
            return jsonify({"success": False, "message": "Could not open video stream"}), 200
        ret, _ = cap.read()
        cap.release()
        gc.collect()
        return jsonify({"success": True}) if ret else jsonify({"success": False, "message": "Failed to retrieve frame"}), 200
    except Exception:
        gc.collect()
        return jsonify({"success": False, "message": "Error"}), 200


@app.route('/api/all_stores_config', methods=['GET'])
def get_all_stores_config():
    conn = get_connection()
    stores = conn.execute(
        "SELECT store_name FROM stores WHERE store_name != ? ORDER BY store_name",
        (ADMIN_STORE_NAME,)
    ).fetchall()
    result = []
    for store_row in stores:
        store_name = store_row['store_name']
        cameras = conn.execute(
            "SELECT * FROM cameras WHERE store_name=? ORDER BY id", (store_name,)
        ).fetchall()
        store_data = {"store_name": store_name, "cameras": []}
        for cam_row in cameras:
            cam = dict(cam_row)
            lines = conn.execute(
                "SELECT * FROM counting_lines WHERE cam_id=? AND store_name=? ORDER BY id",
                (cam['id'], store_name)
            ).fetchall()
            zones = conn.execute(
                "SELECT * FROM detection_zones WHERE cam_id=? AND store_name=? ORDER BY id",
                (cam['id'], store_name)
            ).fetchall()
            cam_data = {
                "active": bool(cam['activate']),
                "camera_id": cam['name'],
                "url": cam['url'],
                "webrtc_url": cam.get('webrtc_url', ''),
                "location": cam.get('location', ''),
                "reid": bool(cam['reid']),
                "gender_age": bool(cam['gender_age']),
                "lines": [],
                "zones": []
            }
            for line_row in lines:
                line = dict(line_row)
                cam_data["lines"].append({
                    "name": line['name'],
                    "points": json.loads(line['points']),
                    "reid": bool(line['reid']),
                    "gender_age": bool(line['gender_age'])
                })
            for zone_row in zones:
                zone = dict(zone_row)
                cam_data["zones"].append({
                    "name": zone['name'],
                    "points": json.loads(zone['points']),
                    "reid": bool(zone['reid']),
                    "gender_age": bool(zone['gender_age']),
                    "detection_mode": zone.get('detection_mode', 'visit'),
                    "role_type": zone.get('role_type', None),
                    "closed": bool(zone.get('closed', True))
                })
            store_data["cameras"].append(cam_data)
        result.append(store_data)
    conn.close()
    gc.collect()
    return jsonify(result)


@app.route('/api/cameras/coordinates', methods=['GET'])
def get_all_camera_coordinates():
    store_name = request.args.get('store_name')
    if not store_name:
        return jsonify({"success": False, "message": "store_name is required"}), 400
    conn = get_connection()
    cameras = conn.execute(
        "SELECT * FROM cameras WHERE store_name=? ORDER BY id", (store_name,)
    ).fetchall()
    result = []
    for cam_row in cameras:
        cam = dict(cam_row)
        lines = conn.execute(
            "SELECT * FROM counting_lines WHERE cam_id=? AND store_name=? ORDER BY id",
            (cam['id'], store_name)
        ).fetchall()
        zones = conn.execute(
            "SELECT * FROM detection_zones WHERE cam_id=? AND store_name=? ORDER BY id",
            (cam['id'], store_name)
        ).fetchall()
        cam_data = {
            "camera_id": cam['name'],
            "url": cam['url'],
            "reid": bool(cam['reid']),
            "gender_age": bool(cam['gender_age']),
            "lines": [],
            "zones": [],
            "active": bool(cam['activate'])
        }
        for line_row in lines:
            line = dict(line_row)
            cam_data["lines"].append({
                "name": line['name'],
                "points": json.loads(line['points']),
                "reid": bool(line['reid']),
                "gender_age": bool(line['gender_age'])
            })
        for zone_row in zones:
            zone = dict(zone_row)
            cam_data["zones"].append({
                "name": zone['name'],
                "points": json.loads(zone['points']),
                "reid": bool(zone['reid']),
                "gender_age": bool(zone['gender_age']),
                "detection_mode": zone['detection_mode'],
                "role_type": zone['role_type']
            })
        result.append(cam_data)
    conn.close()
    gc.collect()
    return jsonify(result)


@app.route('/api/live/cameras', methods=['GET'])
@jwt_required()
def get_live_cameras():
    claims = get_jwt()
    is_admin = claims.get("is_admin", False)
    store_name = get_jwt_identity()
    conn = get_connection()
    if is_admin:
        requested_store = request.args.get('store_name', '').strip()
        store_filter = requested_store if requested_store else None
    else:
        store_filter = store_name
    if store_filter:
        cams = conn.execute(
            "SELECT id, name, url, webrtc_url, location, connection_status FROM cameras WHERE store_name=? AND activate=1 ORDER BY name",
            (store_filter,)
        ).fetchall()
    else:
        cams = conn.execute(
            "SELECT id, name, url, webrtc_url, location, connection_status FROM cameras WHERE activate=1 ORDER BY store_name, name"
        ).fetchall()
    conn.close()
    gc.collect()
    return jsonify([dict(c) for c in cams])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
