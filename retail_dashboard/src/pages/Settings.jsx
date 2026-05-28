
import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Camera, Plus, Trash2, Edit3, CheckCircle2, XCircle,
  Wifi, WifiOff, Layers, Minus, Save, Monitor, MapPin,
  X, Loader2, Undo2, Pencil, Play, ChevronDown, AlertTriangle, Store,
} from 'lucide-react';
import './Settings.css';
const API = `${import.meta.env.VITE_API_URL}/api`;
const CW = 1280;
const CH = 720;
const truncateUrl = (url, max = 42) => {
  if (!url) return '';
  if (url.length <= max) return url;
  return url.slice(0, max - 3) + '...';
};
const getToken = () => localStorage.getItem('token');
const getIsAdmin = () => {
  const stored = localStorage.getItem('is_admin');
  if (stored !== null) return stored === 'true';
  const token = localStorage.getItem('token');
  if (!token) return false;
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    return payload.is_admin === true;
  } catch {
    return false;
  }
};
const authHeaders = () => ({
  'Content-Type': 'application/json',
  Authorization: `Bearer ${getToken()}`,
});
const authFetchHeaders = () => ({
  Authorization: `Bearer ${getToken()}`,
});
function useToast() {
  const [toasts, setToasts] = useState([]);
  const add = useCallback((message, type = 'success') => {
    const id = Date.now() + Math.random();
    setToasts(p => [...p, { id, message, type }]);
    setTimeout(() => setToasts(p => p.filter(t => t.id !== id)), 4000);
  }, []);
  const remove = useCallback(id => setToasts(p => p.filter(t => t.id !== id)), []);
  return { toasts, addToast: add, removeToast: remove };
}
function Toast({ toasts, removeToast }) {
  return (
    <div className="toast-container">
      {toasts.map(t => (
        <div key={t.id} className={`toast toast--${t.type}`}>
          <span className="toast-icon">
            {t.type === 'success' ? <CheckCircle2 size={14} /> : <XCircle size={14} />}
          </span>
          <span>{t.message}</span>
          <button className="toast__close" onClick={() => removeToast(t.id)} type="button">
            <X size={12} />
          </button>
        </div>
      ))}
    </div>
  );
}
function ConfirmModal({ title, message, itemName, onConfirm, onCancel }) {
  return (
    <div className="modal-overlay confirm-overlay" onClick={e => e.target === e.currentTarget && onCancel()}>
      <div className="modal confirm-modal" role="dialog" aria-modal="true">
        <div className="confirm-modal__header">
          <div className="confirm-modal__icon">
            <AlertTriangle size={22} />
          </div>
          <h3>{title}</h3>
        </div>
        <div className="confirm-modal__body">
          <p>{message}</p>
          {itemName && <span className="confirm-item-name">{itemName}</span>}
        </div>
        <div className="confirm-modal__footer">
          <button type="button" className="btn-secondary" onClick={onCancel}>Cancel</button>
          <button type="button" className="btn-danger" onClick={onConfirm}>
            <Trash2 size={13} />Delete
          </button>
        </div>
      </div>
    </div>
  );
}
function ToggleChip({ label, checked, onChange, colored }) {
  return (
    <label
      className={`toggle-chip ${colored ? (checked ? 'toggle-chip--green' : 'toggle-chip--red') : ''}`}
      onClick={() => onChange(!checked)}
    >
      <span className={`toggle-switch ${checked ? 'toggle-switch--on' : ''}`}>
        <span className="toggle-switch__thumb" />
      </span>
      <span className="toggle-chip__label">{label}</span>
    </label>
  );
}
function StatusDot({ active }) {
  return (
    <span
      className={`cam-card__active-dot ${active ? 'cam-card__active-dot--on' : 'cam-card__active-dot--off'}`}
      title={active ? 'Active' : 'Inactive'}
    />
  );
}
function SelectField({ value, onChange, options, className }) {
  return (
    <div className={`custom-select-wrap ${className || ''}`}>
      <select
        className="custom-select"
        value={value}
        onChange={e => onChange(e.target.value)}
      >
        {options.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
      <ChevronDown size={11} className="custom-select-arrow" />
    </div>
  );
}
function CameraCard({ cam, onEdit, onDelete, onToggle }) {
  const status = cam.connection_status || 'unknown';
  const statusConfig = {
    unknown: { icon: <Wifi size={11} />, label: 'Not Tested', cls: 'unknown' },
    success: { icon: <CheckCircle2 size={11} />, label: 'Connected', cls: 'success' },
    fail: { icon: <WifiOff size={11} />, label: 'Failed', cls: 'fail' },
  };
  const s = statusConfig[status] ?? statusConfig.unknown;
  return (
    <div className={`cam-card cam-card--${status}`}>
      <div className="cam-card__header">
        <div className="cam-card__icon">
          <Camera size={16} />
        </div>
        <div className="cam-card__info">
          <div className="cam-card__name-row">
            <h3 className="cam-card__name" title={cam.name}>{cam.name}</h3>
            <StatusDot active={!!cam.activate} />
          </div>
          <p className="cam-card__url" title={cam.url}>{truncateUrl(cam.url)}</p>
        </div>
        <div className="cam-card__actions">
          <button type="button" className="icon-btn" onClick={() => onEdit(cam)} title="Edit">
            <Edit3 size={12} />
          </button>
          <button type="button" className="icon-btn icon-btn--danger" onClick={() => onDelete(cam)} title="Delete">
            <Trash2 size={12} />
          </button>
        </div>
      </div>
      <div className="cam-card__meta">
        <span className="meta-chip"><MapPin size={10} />{cam.location}</span>
        <span className="meta-chip"><Monitor size={10} />{truncateUrl(cam.webrtc_url, 24)}</span>
        <span className={`cam-status cam-status--${s.cls}`}>
          <span className="cam-status__dot" />
          {s.label}
        </span>
      </div>
      <div className="cam-card__toggles">
        <ToggleChip label="Re-ID" checked={!!cam.reid} onChange={v => onToggle(cam, 'reid', v)} colored />
        <ToggleChip label="Gender&Age" checked={!!cam.gender_age} onChange={v => onToggle(cam, 'gender_age', v)} colored />
        <ToggleChip label="Activate" checked={!!cam.activate} onChange={v => onToggle(cam, 'activate', v)} colored />
      </div>
      {cam.tested ? (
        <p className="cam-card__tested">
          Last tested: {cam.last_tested_at ? new Date(cam.last_tested_at).toLocaleString() : 'Yes'}
        </p>
      ) : (
        <p className="cam-card__tested">Connection not tested yet</p>
      )}
    </div>
  );
}
function CameraModal({ cam, onSave, onClose, saving }) {
  const [form, setForm] = useState({
    name: cam?.name ?? '',
    url: cam?.url ?? '',
    location: cam?.location ?? '',
    webrtc_url: cam?.webrtc_url ?? '',
    reid: cam?.reid ?? false,
    gender_age: cam?.gender_age ?? false,
    activate: cam?.activate ?? false,
  });
  const [testing, setTesting] = useState(false);
  const [testStatus, setTestStatus] = useState(null);
  const set = (k, v) => {
    if (k === 'url') setTestStatus(null);
    setForm(p => ({ ...p, [k]: v }));
  };
  const valid = form.name.trim() && form.url.trim() && form.location.trim();
  const handleTest = useCallback(async () => {
    if (!form.url.trim()) return;
    setTesting(true);
    setTestStatus(null);
    try {
      const res = await fetch(`${API}/test_camera`, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({ url: form.url }),
        signal: AbortSignal.timeout(35000),
      });
      const result = await res.json();
      setTestStatus(result.success ? 'success' : 'fail');
    } catch {
      setTestStatus('fail');
    }
    setTesting(false);
  }, [form.url]);
  return (
    <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-modal="true">
        <div className="modal__header">
          <h3>{cam ? 'Edit Camera' : 'Add Camera'}</h3>
          <button type="button" className="modal__close" onClick={onClose} aria-label="Close">
            <X size={16} />
          </button>
        </div>
        <div className="modal__body">
          {[
            { label: 'Camera Name', key: 'name', ph: 'Entrance Camera 1' },
            { label: 'Stream URL', key: 'url', ph: 'rtsp://192.168.1.x:554/stream' },
            { label: 'Location / Zone', key: 'location', ph: 'Main Entrance' },
          ].map(({ label, key, ph }) => (
            <div className="form-group" key={key}>
              <label>{label} <span className="req">*</span></label>
              <input
                className="form-input"
                placeholder={ph}
                value={form[key]}
                onChange={e => set(key, e.target.value)}
              />
            </div>
          ))}
          <div className="form-group">
            <label>WebRTC URL</label>
            <input
              className="form-input"
              placeholder="https://webrtc.example.com/stream"
              value={form.webrtc_url}
              onChange={e => set('webrtc_url', e.target.value)}
            />
          </div>
          <div className="form-group">
            <label>Analytics &amp; Status</label>
            <div className="form-toggles">
              <ToggleChip label="Re-ID" checked={form.reid} onChange={v => set('reid', v)} colored />
              <ToggleChip label="Gender &amp; Age" checked={form.gender_age} onChange={v => set('gender_age', v)} colored />
              <ToggleChip label="Activate" checked={form.activate} onChange={v => set('activate', v)} colored />
            </div>
          </div>
          {testStatus === 'success' && (
            <div className="test-result-banner test-result-banner--success">
              <CheckCircle2 size={18} />
              <span>Connection successful</span>
            </div>
          )}
        </div>
        <div className="modal__footer">
          <button type="button" className="btn-secondary" onClick={onClose}>Cancel</button>
          {testStatus !== 'success' ? (
            <button
              type="button"
              className={`btn-test btn-test--compact${testStatus === 'fail' ? ' btn-test--fail' : ''}`}
              onClick={handleTest}
              disabled={testing || !form.url.trim()}
            >
              {testing
                ? <><Loader2 size={12} className="spin" />Testing...</>
                : testStatus === 'fail'
                ? <><WifiOff size={12} />Failed - Retry</>
                : <><Play size={12} />Test Connection</>}
            </button>
          ) : (
            <button
              type="button"
              className="btn-primary"
              onClick={() => valid && onSave({
                ...form,
                connection_status: 'success',
                tested: true,
                last_tested_at: new Date().toISOString(),
              })}
              disabled={!valid || saving}
            >
              {saving ? <Loader2 size={13} className="spin" /> : <Save size={13} />}
              {cam ? 'Update Camera' : 'Add Camera'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
function CanvasEditor({ cameras, addToast, activeStore }) {
  const canvasRef = useRef(null);
  const [selCamId, setSelCamId] = useState(cameras[0]?.id ?? null);
  const [toolMode, setToolMode] = useState('LINE');
  const [drawActive, setDrawActive] = useState(false);
  const [currentLine, setCurrentLine] = useState([]);
  const [doorName, setDoorName] = useState('Entrance');
  const [zoneName, setZoneName] = useState('Clothes');
  const [zoneDetectionMode, setZoneDetectionMode] = useState('visit');
  const [zoneRoleType, setZoneRoleType] = useState('cashier');
  const [pendingLines, setPendingLines] = useState([]);
  const [pendingZonePts, setPendingZonePts] = useState([]);
  const [zoneClosed, setZoneClosed] = useState(false);
  const [lines, setLines] = useState([]);
  const [zones, setZones] = useState([]);
  const [bgImage, setBgImage] = useState(null);
  const [frameLoading, setFrameLoading] = useState(false);
  const [hoveredPt, setHoveredPt] = useState(null);
  const [saving, setSaving] = useState(false);
  const [editingItem, setEditingItem] = useState(null);
  const [lineReid, setLineReid] = useState(false);
  const [lineGenderAge, setLineGenderAge] = useState(false);
  const [zoneReid, setZoneReid] = useState(false);
  const [zoneGenderAge, setZoneGenderAge] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(null);
  const camera = cameras.find(c => c.id === selCamId);
  const clearPending = useCallback(() => {
    setPendingLines([]); setPendingZonePts([]); setZoneClosed(false);
    setDoorName(''); setZoneName('');
    setZoneDetectionMode('visit'); setZoneRoleType('cashier');
    setCurrentLine([]); setEditingItem(null);
    setLineReid(false); setLineGenderAge(false);
    setZoneReid(false); setZoneGenderAge(false);
  }, []);
  const clearAll = useCallback(() => { clearPending(); setDrawActive(false); }, [clearPending]);
  useEffect(() => {
    if (cameras.length > 0 && !cameras.find(c => c.id === selCamId)) {
      setSelCamId(cameras[0].id);
    }
  }, [cameras]);
  useEffect(() => {
    if (!selCamId || !activeStore) return;
    setFrameLoading(true); setBgImage(null); clearAll();
    const storeParam = `store=${encodeURIComponent(activeStore)}`;
    fetch(`${API}/camera_snapshot/${selCamId}?${storeParam}`, { headers: authFetchHeaders() })
      .then(r => { if (!r.ok) throw new Error(); return r.blob(); })
      .then(blob => new Promise((res, rej) => {
        const reader = new FileReader();
        reader.onload = () => res(reader.result);
        reader.onerror = () => rej();
        reader.readAsDataURL(blob);
      }))
      .then(dataUrl => {
        const img = new window.Image();
        img.onload = () => { setBgImage(img); setFrameLoading(false); };
        img.onerror = () => setFrameLoading(false);
        img.src = dataUrl;
      })
      .catch(() => setFrameLoading(false));
    fetch(`${API}/cameras/${selCamId}/lines?${storeParam}`, { headers: authFetchHeaders() })
      .then(r => r.json()).then(setLines).catch(() => {});
    fetch(`${API}/cameras/${selCamId}/zones?${storeParam}`, { headers: authFetchHeaders() })
      .then(r => r.json()).then(setZones).catch(() => {});
  }, [selCamId, activeStore, clearAll]);
  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, CW, CH);
    if (bgImage) {
      ctx.drawImage(bgImage, 0, 0, CW, CH);
      ctx.fillStyle = 'rgba(0,0,0,0.32)'; ctx.fillRect(0, 0, CW, CH);
    } else {
      ctx.fillStyle = '#080d1a'; ctx.fillRect(0, 0, CW, CH);
      ctx.strokeStyle = 'rgba(255,255,255,0.035)'; ctx.lineWidth = 1;
      for (let x = 0; x <= CW; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, CH); ctx.stroke(); }
      for (let y = 0; y <= CH; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(CW, y); ctx.stroke(); }
      ctx.fillStyle = 'rgba(79,124,247,0.4)';
      ctx.font = '500 14px "IBM Plex Mono", monospace'; ctx.textAlign = 'center';
      ctx.fillText(camera ? `${camera.name} ${camera.location}` : 'Select a camera', CW / 2, CH / 2 - 14);
      ctx.fillStyle = 'rgba(255,255,255,0.15)';
      ctx.font = '13px "IBM Plex Mono", monospace';
      ctx.fillText('Click "Draw" then click canvas to place points', CW / 2, CH / 2 + 14);
    }
    const doorGroups = {};
    lines.forEach(l => { if (!doorGroups[l.name]) doorGroups[l.name] = []; doorGroups[l.name].push(l); });
    Object.entries(doorGroups).forEach(([door, dLines]) => {
      const row = dLines[0];
      const segments = row.points || [];
      segments.forEach((segment, idx) => {
        const color = idx === 0 ? '#00e5ff' : '#34d399';
        const label = idx === 0 ? 'OUTER' : 'INNER';
        const [p1, p2] = segment;
        ctx.save();
        ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(...p1); ctx.lineTo(...p2); ctx.stroke();
        [p1, p2].forEach(p => {
          ctx.fillStyle = color; ctx.beginPath(); ctx.arc(p[0], p[1], 5, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = '#0a0f1e'; ctx.beginPath(); ctx.arc(p[0], p[1], 2, 0, Math.PI * 2); ctx.fill();
        });
        const mx = (p1[0] + p2[0]) / 2, my = (p1[1] + p2[1]) / 2;
        ctx.fillStyle = color; ctx.font = 'bold 11px "IBM Plex Mono",monospace';
        ctx.textAlign = 'center'; ctx.shadowColor = 'rgba(0,0,0,0.9)'; ctx.shadowBlur = 4;
        ctx.fillText(`${door} - ${label}`, mx, my - 10); ctx.shadowBlur = 0;
        if (idx === 0) {
          const dx = p2[0] - p1[0], dy = p2[1] - p1[1];
          let nx = -dy, ny = dx;
          const len = Math.hypot(nx, ny);
          if (len > 0) {
            nx /= len; ny /= len;
            const arrowLen = 50;
            const cx2 = (p1[0] + p2[0]) / 2, cy2 = (p1[1] + p2[1]) / 2;
            const entryTip = [cx2 + nx * arrowLen, cy2 + ny * arrowLen];
            const exitTip = [cx2 - nx * arrowLen, cy2 - ny * arrowLen];
            ctx.strokeStyle = '#34d399'; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(cx2, cy2); ctx.lineTo(...entryTip); ctx.stroke();
            ctx.fillStyle = '#34d399'; ctx.font = 'bold 10px "IBM Plex Mono",monospace';
            ctx.fillText('ENTRY', entryTip[0] + 12, entryTip[1]);
            ctx.strokeStyle = '#f87171';
            ctx.beginPath(); ctx.moveTo(cx2, cy2); ctx.lineTo(...exitTip); ctx.stroke();
            ctx.fillStyle = '#f87171'; ctx.fillText('EXIT', exitTip[0] + 12, exitTip[1]);
          }
        }
        ctx.restore();
      });
    });
    zones.forEach(z => {
      if (!z.points || z.points.length < 2) return;
      const np = z.points;
      ctx.save();
      ctx.fillStyle = 'rgba(99,102,241,0.18)'; ctx.strokeStyle = '#6366f1';
      ctx.lineWidth = 2; ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(np[0][0], np[0][1]);
      np.slice(1).forEach(p => ctx.lineTo(p[0], p[1]));
      if (z.closed) ctx.closePath();
      ctx.fill(); ctx.stroke();
      const cx2 = np.reduce((s, p) => s + p[0], 0) / np.length;
      const cy2 = np.reduce((s, p) => s + p[1], 0) / np.length;
      ctx.fillStyle = '#a5b4fc'; ctx.font = 'bold 12px "IBM Plex Mono",monospace';
      ctx.textAlign = 'center'; ctx.shadowColor = 'rgba(0,0,0,0.9)'; ctx.shadowBlur = 4;
      ctx.fillText(z.name, cx2, cy2 + 4); ctx.shadowBlur = 0;
      np.forEach(p => { ctx.fillStyle = '#6366f1'; ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, Math.PI * 2); ctx.fill(); });
      ctx.restore();
    });
    pendingLines.forEach((pts, idx) => {
      const color = idx === 0 ? '#00e5ff' : '#34d399';
      ctx.save();
      ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); ctx.lineTo(pts[1][0], pts[1][1]); ctx.stroke();
      pts.forEach(p => {
        ctx.fillStyle = color; ctx.beginPath(); ctx.arc(p[0], p[1], 5, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#0a0f1e'; ctx.beginPath(); ctx.arc(p[0], p[1], 2, 0, Math.PI * 2); ctx.fill();
      });
      ctx.fillStyle = color; ctx.font = 'bold 11px "IBM Plex Mono",monospace';
      ctx.textAlign = 'center'; ctx.shadowColor = 'rgba(0,0,0,0.9)'; ctx.shadowBlur = 4;
      ctx.fillText(idx === 0 ? 'OUTER' : 'INNER', (pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2 - 10);
      ctx.shadowBlur = 0; ctx.restore();
    });
    if (currentLine.length === 1 && hoveredPt && toolMode === 'LINE' && drawActive) {
      ctx.save();
      const color = pendingLines.length === 0 ? '#00e5ff' : '#34d399';
      ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(currentLine[0][0], currentLine[0][1]); ctx.lineTo(hoveredPt.x, hoveredPt.y); ctx.stroke();
      ctx.fillStyle = color; ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(currentLine[0][0], currentLine[0][1], 6, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    }
    if (pendingZonePts.length > 0 && toolMode === 'ZONE') {
      const color = '#f472b6';
      ctx.save();
      if (zoneClosed) {
        ctx.fillStyle = 'rgba(244,114,182,0.18)'; ctx.strokeStyle = color;
        ctx.lineWidth = 2; ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(pendingZonePts[0][0], pendingZonePts[0][1]);
        pendingZonePts.slice(1).forEach(p => ctx.lineTo(p[0], p[1]));
        ctx.closePath(); ctx.fill(); ctx.stroke();
      } else {
        ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
        if (pendingZonePts.length >= 2) {
          ctx.beginPath(); ctx.moveTo(pendingZonePts[0][0], pendingZonePts[0][1]);
          pendingZonePts.slice(1).forEach(p => ctx.lineTo(p[0], p[1]));
          if (hoveredPt && drawActive) ctx.lineTo(hoveredPt.x, hoveredPt.y);
          ctx.stroke();
        } else if (pendingZonePts.length === 1 && hoveredPt && drawActive) {
          ctx.beginPath(); ctx.moveTo(pendingZonePts[0][0], pendingZonePts[0][1]); ctx.lineTo(hoveredPt.x, hoveredPt.y); ctx.stroke();
        }
      }
      pendingZonePts.forEach((p, i) => {
        ctx.setLineDash([]); ctx.fillStyle = color;
        ctx.beginPath(); ctx.arc(p[0], p[1], 6, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#0a0f1e'; ctx.font = 'bold 10px sans-serif'; ctx.textAlign = 'center';
        ctx.fillText(i + 1, p[0], p[1] + 3.5);
      });
      ctx.restore();
    }
    if (drawActive) {
      ctx.save();
      ctx.strokeStyle = toolMode === 'LINE' ? 'rgba(0,229,255,0.45)' : 'rgba(244,114,182,0.45)';
      ctx.lineWidth = 2; ctx.setLineDash([10, 6]);
      ctx.strokeRect(2, 2, CW - 4, CH - 4);
      ctx.restore();
    }
  }, [bgImage, lines, zones, pendingLines, pendingZonePts, currentLine, hoveredPt, toolMode, zoneClosed, camera, drawActive]);
  useEffect(() => { redraw(); }, [redraw]);
  const getCoords = useCallback(e => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    return {
      x: Math.round((e.clientX - rect.left) * (CW / rect.width)),
      y: Math.round((e.clientY - rect.top) * (CH / rect.height)),
    };
  }, []);
  const handleMouseMove = useCallback(e => setHoveredPt(getCoords(e)), [getCoords]);
  const handleMouseLeave = useCallback(() => setHoveredPt(null), []);
  const handleClick = useCallback(e => {
    if (!camera || !drawActive) return;
    const pt = getCoords(e);
    if (toolMode === 'LINE') {
      if (currentLine.length === 0) {
        if (pendingLines.length >= 2) return;
        setCurrentLine([[pt.x, pt.y]]);
      } else {
        const newSeg = [currentLine[0], [pt.x, pt.y]];
        setPendingLines(prev => prev.length >= 2 ? prev : [...prev, newSeg]);
        setCurrentLine([]);
        addToast(pendingLines.length === 0 ? 'Outer line placed draw the inner line' : 'Both lines placed click Save', 'success');
      }
    } else {
      if (zoneClosed) return;
      setPendingZonePts(prev => {
        const updated = [...prev, [pt.x, pt.y]];
        if (updated.length === 4) setTimeout(() => setZoneClosed(true), 60);
        return updated;
      });
    }
  }, [camera, drawActive, toolMode, currentLine, pendingLines, zoneClosed, getCoords, addToast]);
  const handleUndo = useCallback(() => {
    if (toolMode === 'LINE') {
      if (currentLine.length > 0) setCurrentLine([]);
      else if (pendingLines.length > 0) setPendingLines(prev => prev.slice(0, -1));
    } else {
      if (zoneClosed) { setZoneClosed(false); setPendingZonePts(prev => prev.slice(0, -1)); }
      else if (pendingZonePts.length > 0) setPendingZonePts(prev => prev.slice(0, -1));
    }
  }, [toolMode, currentLine, pendingLines, zoneClosed, pendingZonePts]);
  const storeParam = activeStore ? `store=${encodeURIComponent(activeStore)}` : '';
  const saveLines = useCallback(async () => {
    if (pendingLines.length < 2) { addToast('Draw both outer and inner lines first', 'error'); return; }
    setSaving(true);
    try {
      if (editingItem?.type === 'line') {
        const toDelete = lines.filter(l => l.name === editingItem.name);
        for (const l of toDelete) {
          await fetch(`${API}/cameras/${selCamId}/lines/${l.id}?${storeParam}`, {
            method: 'DELETE',
            headers: authFetchHeaders(),
          });
        }
      }
      await fetch(`${API}/cameras/${selCamId}/lines?${storeParam}`, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({
          name: doorName,
          flip_normal: false,
          reid: lineReid,
          gender_age: lineGenderAge,
          points: pendingLines,
          store_name: activeStore,
        }),
      });
      const refreshed = await fetch(`${API}/cameras/${selCamId}/lines?${storeParam}`, { headers: authFetchHeaders() }).then(r => r.json());
      setLines(refreshed);
      clearAll();
      addToast(`"${doorName}" saved`, 'success');
    } catch { addToast('Failed to save lines', 'error'); }
    setSaving(false);
  }, [pendingLines, doorName, lineReid, lineGenderAge, selCamId, editingItem, lines, clearAll, addToast, storeParam, activeStore]);
  const saveZone = useCallback(async () => {
    const name = zoneName.trim() || `Zone-${zones.length + 1}`;
    setSaving(true);
    try {
      if (editingItem?.type === 'zone') {
        await fetch(`${API}/cameras/${selCamId}/zones/${editingItem.id}?${storeParam}`, {
          method: 'DELETE',
          headers: authFetchHeaders(),
        });
      }
      await fetch(`${API}/cameras/${selCamId}/zones?${storeParam}`, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({
          name,
          points: pendingZonePts,
          closed: true,
          reid: zoneReid,
          gender_age: zoneGenderAge,
          detection_mode: zoneDetectionMode,
          role_type: zoneDetectionMode === 'presence' ? zoneRoleType : null,
          store_name: activeStore,
        }),
      });
      const refreshed = await fetch(`${API}/cameras/${selCamId}/zones?${storeParam}`, { headers: authFetchHeaders() }).then(r => r.json());
      setZones(refreshed);
      clearAll();
      addToast(`Zone "${name}" saved`, 'success');
    } catch { addToast('Failed to save zone', 'error'); }
    setSaving(false);
  }, [zoneName, zones.length, pendingZonePts, zoneReid, zoneGenderAge, zoneDetectionMode, zoneRoleType, selCamId, editingItem, clearAll, addToast, storeParam, activeStore]);
  const requestDeleteLine = useCallback(dName => {
    setConfirmDelete({ type: 'line', name: dName });
  }, []);
  const requestDeleteZone = useCallback(zone => {
    setConfirmDelete({ type: 'zone', id: zone.id, name: zone.name });
  }, []);
  const handleConfirmDelete = useCallback(async () => {
    if (!confirmDelete) return;
    if (confirmDelete.type === 'line') {
      const toDelete = lines.filter(l => l.name === confirmDelete.name);
      for (const l of toDelete) {
        await fetch(`${API}/cameras/${selCamId}/lines/${l.id}?${storeParam}`, {
          method: 'DELETE',
          headers: authFetchHeaders(),
        });
      }
      setLines(prev => prev.filter(l => l.name !== confirmDelete.name));
      addToast(`"${confirmDelete.name}" deleted`, 'success');
    } else {
      await fetch(`${API}/cameras/${selCamId}/zones/${confirmDelete.id}?${storeParam}`, {
        method: 'DELETE',
        headers: authFetchHeaders(),
      });
      setZones(prev => prev.filter(z => z.id !== confirmDelete.id));
      addToast('Zone deleted', 'success');
    }
    setConfirmDelete(null);
  }, [confirmDelete, lines, selCamId, addToast, storeParam]);
  const startEdit = useCallback((type, item) => {
    clearPending();
    if (type === 'line') {
      const row = lines.find(l => l.name === item.name);
      if (row) {
        setPendingLines(row.points || []);
        setDoorName(item.name);
        setLineReid(!!row.reid);
        setLineGenderAge(!!row.gender_age);
        setEditingItem({ type: 'line', name: item.name });
      }
    } else {
      setPendingZonePts(item.points || []);
      setZoneName(item.name);
      setZoneReid(!!item.reid);
      setZoneGenderAge(!!item.gender_age);
      setZoneDetectionMode(item.detection_mode || 'visit');
      setZoneRoleType(item.role_type || 'cashier');
      setEditingItem({ type: 'zone', id: item.id });
      if ((item.points || []).length >= 4) setZoneClosed(true);
    }
    setToolMode(type === 'line' ? 'LINE' : 'ZONE');
    setDrawActive(true);
    addToast(`Editing "${item.name}"`, 'success');
  }, [lines, clearPending, addToast]);
  const canSave = toolMode === 'LINE' ? pendingLines.length === 2 : zoneClosed;
  const canUndo = toolMode === 'LINE' ? (currentLine.length > 0 || pendingLines.length > 0) : pendingZonePts.length > 0;
  const doorGroups = {};
  lines.forEach(l => { if (!doorGroups[l.name]) doorGroups[l.name] = []; doorGroups[l.name].push(l); });
  const detectionOptions = [
    { value: 'visit', label: 'Visit Count' },
    { value: 'presence', label: 'Presence' },
  ];
  const roleOptions = [
    { value: 'cashier', label: 'Cashier' },
    { value: 'security', label: 'Security' },
  ];
  return (
    <div className="canvas-editor">
      {confirmDelete && (
        <ConfirmModal
          title={confirmDelete.type === 'line' ? 'Delete Counting Line' : 'Delete Zone'}
          message={confirmDelete.type === 'line'
            ? 'Are you sure you want to delete this counting line? This action cannot be undone.'
            : 'Are you sure you want to delete this zone? This action cannot be undone.'}
          itemName={confirmDelete.name}
          onConfirm={handleConfirmDelete}
          onCancel={() => setConfirmDelete(null)}
        />
      )}
      <div className="ce-toolbar ce-toolbar--main">
        <div className="ce-cam-select">
          <Camera size={13} />
          <select
            className="canvas-select"
            value={selCamId ?? ''}
            onChange={e => { setSelCamId(Number(e.target.value)); clearAll(); }}
          >
            <option value="" disabled>Select camera</option>
            {cameras.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
          </select>
        </div>
        <div className="tool-toggle-group" role="group">
          <button
            type="button"
            className={`tool-pill ${toolMode === 'LINE' ? 'tool-pill--active' : ''}`}
            onClick={() => { setToolMode('LINE'); clearAll(); }}
          >
            <Minus size={13} /><span>Lines</span>
          </button>
          <button
            type="button"
            className={`tool-pill ${toolMode === 'ZONE' ? 'tool-pill--active' : ''}`}
            onClick={() => { setToolMode('ZONE'); clearAll(); }}
          >
            <Layers size={13} /><span>Zones</span>
          </button>
        </div>
        {toolMode === 'LINE' && (
          <>
            <div className="name-input-group">
              <input className="form-input form-input--sm" value={doorName} onChange={e => setDoorName(e.target.value)} placeholder="Door-1" />
            </div>
            <ToggleChip label="Re-ID" checked={lineReid} onChange={setLineReid} colored />
            <ToggleChip label="Gender&Age" checked={lineGenderAge} onChange={setLineGenderAge} colored />
          </>
        )}
        {toolMode === 'ZONE' && (
          <>
            <div className="name-input-group">
              <input className="form-input form-input--sm" value={zoneName} onChange={e => setZoneName(e.target.value)} placeholder="Zone-1" />
            </div>
            <SelectField
              value={zoneDetectionMode}
              onChange={v => { setZoneDetectionMode(v); if (v !== 'presence') setZoneRoleType('cashier'); }}
              options={detectionOptions}
            />
            {zoneDetectionMode === 'presence' && (
              <SelectField
                value={zoneRoleType}
                onChange={setZoneRoleType}
                options={roleOptions}
                className="role-select-animate"
              />
            )}
            <ToggleChip label="Re-ID" checked={zoneReid} onChange={setZoneReid} colored />
            <ToggleChip label="Gender&Age" checked={zoneGenderAge} onChange={setZoneGenderAge} colored />
          </>
        )}
        <div className="draw-actions">
          {!drawActive ? (
            <button type="button" className="action-pill action-pill--draw" onClick={() => setDrawActive(true)}>
              <Pencil size={13} /><span>Draw</span>
            </button>
          ) : (
            <>
              <button type="button" className="action-pill action-pill--undo" onClick={handleUndo} disabled={!canUndo}>
                <Undo2 size={13} /><span>Undo</span>
              </button>
              {canSave && (
                <button type="button" className="action-pill action-pill--save" onClick={toolMode === 'LINE' ? saveLines : saveZone} disabled={saving}>
                  {saving ? <Loader2 size={13} className="spin" /> : <Save size={13} />}<span>Save</span>
                </button>
              )}
              <button type="button" className="action-pill action-pill--cancel" onClick={clearAll}>
                <X size={13} /><span>Cancel</span>
              </button>
            </>
          )}
        </div>
      </div>
      <div className="canvas-main-layout">
        <div className="canvas-area">
          <div className="canvas-wrap">
            {frameLoading && (
              <div className="frame-loading">
                <Loader2 size={22} className="spin" />
                <span>Fetching frame...</span>
              </div>
            )}
            <canvas
              ref={canvasRef}
              width={CW} height={CH}
              className={`editor-canvas ${drawActive ? (toolMode === 'LINE' ? 'cursor-line' : 'cursor-zone') : 'cursor-default'}`}
              onClick={handleClick}
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
            />
          </div>
        </div>
        <div className="saved-items-sidebar">
          <div className="saved-items-panel">
            <div className="saved-items-panel__head">
              <span><Minus size={12} />Counting Lines</span>
              <span className="badge">{Object.keys(doorGroups).length}</span>
            </div>
            {Object.keys(doorGroups).length === 0 ? (
              <p className="saved-empty">No lines defined</p>
            ) : (
              Object.entries(doorGroups).map(([door, dLines]) => (
                <div key={door} className="saved-item">
                  <div className="saved-item__info">
                    <span className="saved-item__name">{door}</span>
                    {(dLines[0].points || []).map((pts, i) => (
                      <div key={i} className="saved-item__coords">
                        {i === 0 ? 'outer' : 'inner'}: ({pts[0][0]},{pts[0][1]}) ({pts[1][0]},{pts[1][1]})
                      </div>
                    ))}
                    <div className="saved-item__toggles">
                      <ToggleChip
                        label="Re-ID"
                        checked={!!dLines[0].reid}
                        onChange={v => setLines(prev => prev.map(l => l.name === door ? { ...l, reid: v } : l))}
                        colored
                      />
                      <ToggleChip
                        label="Gender&Age"
                        checked={!!dLines[0].gender_age}
                        onChange={v => setLines(prev => prev.map(l => l.name === door ? { ...l, gender_age: v } : l))}
                        colored
                      />
                    </div>
                  </div>
                  <div className="saved-item__btns">
                    <button type="button" className="saved-item__edit" onClick={() => startEdit('line', { name: door })}><Edit3 size={11} /></button>
                    <button type="button" className="saved-item__del" onClick={() => requestDeleteLine(door)}><Trash2 size={11} /></button>
                  </div>
                </div>
              ))
            )}
          </div>
          <div className="saved-items-panel">
            <div className="saved-items-panel__head">
              <span><Layers size={12} />Zones</span>
              <span className="badge">{zones.length}</span>
            </div>
            {zones.length === 0 ? (
              <p className="saved-empty">No zones defined</p>
            ) : (
              zones.map(z => (
                <div key={z.id} className="saved-item">
                  <div className="saved-item__info">
                    <span className="saved-item__name">{z.name}</span>
                    <span className="saved-item__coords">
                      {z.points.length} vertices &middot;{' '}
                      {z.detection_mode === 'presence'
                        ? `Presence${z.role_type ? ` / ${z.role_type.charAt(0).toUpperCase() + z.role_type.slice(1)}` : ''}`
                        : 'Visit Count'}
                    </span>
                    <div className="saved-item__toggles">
                      <ToggleChip
                        label="Re-ID"
                        checked={!!z.reid}
                        onChange={v => setZones(prev => prev.map(zn => zn.id === z.id ? { ...zn, reid: v } : zn))}
                        colored
                      />
                      <ToggleChip
                        label="Gender&Age"
                        checked={!!z.gender_age}
                        onChange={v => setZones(prev => prev.map(zn => zn.id === z.id ? { ...zn, gender_age: v } : zn))}
                        colored
                      />
                    </div>
                  </div>
                  <div className="saved-item__btns">
                    <button type="button" className="saved-item__edit" onClick={() => startEdit('zone', z)}><Edit3 size={11} /></button>
                    <button type="button" className="saved-item__del" onClick={() => requestDeleteZone(z)}><Trash2 size={11} /></button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
export default function Settings({ mode = 'all' }) {
  const [activeTab, setActiveTab] = useState(
  mode === 'store' ? 'editor' : 'cameras'
);
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [editCam, setEditCam] = useState(null);
  const [modalSaving, setModalSaving] = useState(false);
  const [confirmDeleteCam, setConfirmDeleteCam] = useState(null);
  const [stores, setStores] = useState([]);
  const [activeStore, setActiveStore] = useState('');
  const [isAdmin] = useState(() => getIsAdmin());
  const { toasts, addToast, removeToast } = useToast();
  const loadCameras = useCallback(async (storeName) => {
    if (!storeName) return;
    setLoading(true);
    try {
      const url = `${API}/cameras?store=${encodeURIComponent(storeName)}`;
      const res = await fetch(url, { headers: authFetchHeaders() });
      if (res.status === 401) {
        localStorage.removeItem('token');
        window.location.href = '/login';
        return;
      }
      const data = await res.json();
      setCameras(data.map(c => ({ ...c, connection_status: c.connection_status || 'unknown' })));
    } catch {
      addToast('Failed to load cameras', 'error');
    }
    setLoading(false);
  }, [addToast]);
  useEffect(() => {
    const fetchStores = async () => {
      try {
        const res = await fetch(`${API}/stores`, { headers: authFetchHeaders() });
        if (res.status === 401) {
          localStorage.removeItem('token');
          window.location.href = '/login';
          return;
        }
        const data = await res.json();
        setStores(data);
        if (data.length > 0) {
          const first = data[0].store_name;
          setActiveStore(first);
          loadCameras(first);
        } else {
          setLoading(false);
        }
      } catch {
        addToast('Failed to load stores', 'error');
        setLoading(false);
      }
    };
    fetchStores();
  }, []);
  useEffect(() => {
    if (activeStore) {
      loadCameras(activeStore);
    }
  }, [activeStore]);
 const handleToggle = useCallback(async (cam, field, value) => {
  setCameras(prev => prev.map(c => c.id === cam.id ? { ...c, [field]: value } : c));
  try {
    const res = await fetch(`${API}/cameras/${cam.id}/toggle`, {
      method: 'PUT',
      headers: authHeaders(),
      body: JSON.stringify({ field, value }),
    });
    const result = await res.json().catch(() => ({}));
    if (!res.ok || !result.success) throw new Error(result.message || 'failed');
  } catch (e) {
    setCameras(prev => prev.map(c => c.id === cam.id ? { ...c, [field]: !value } : c));
    addToast('Failed to update camera', 'error');
  }
}, [addToast]);
  const saveCamera = useCallback(async formData => {
    setModalSaving(true);
    try {
      if (editCam) {
        const res = await fetch(`${API}/cameras/${editCam.id}?store=${encodeURIComponent(activeStore)}`, {
          method: 'PUT',
          headers: authHeaders(),
          body: JSON.stringify({ ...formData, store_name: activeStore }),
        });
        const result = await res.json();
        if (!result.success) {
          addToast(result.message || 'Failed to update camera', 'error');
          setModalSaving(false);
          return;
        }
        setCameras(prev => prev.map(c =>
          c.id === editCam.id
            ? {
                ...c,
                ...result.camera,
                connection_status: formData.connection_status ?? result.camera.connection_status ?? c.connection_status ?? 'unknown',
                tested: formData.tested ?? result.camera.tested,
                last_tested_at: formData.last_tested_at ?? result.camera.last_tested_at,
              }
            : c
        ));
        addToast('Camera updated', 'success');
      } else {
        const res = await fetch(`${API}/cameras?store=${encodeURIComponent(activeStore)}`, {
          method: 'POST',
          headers: authHeaders(),
          body: JSON.stringify({ ...formData, store_name: activeStore }),
        });
        const result = await res.json();
        if (!result.success) {
          addToast(result.message || 'Failed to add camera', 'error');
          setModalSaving(false);
          return;
        }
        setCameras(prev => [...prev, {
          ...result.camera,
          connection_status: formData.connection_status ?? 'unknown',
          tested: formData.tested ?? result.camera.tested ?? 0,
          last_tested_at: formData.last_tested_at ?? result.camera.last_tested_at ?? null,
        }]);
        addToast('Camera added', 'success');
      }
      setShowModal(false);
      setEditCam(null);
    } catch {
      addToast('Failed to save camera', 'error');
    }
    setModalSaving(false);
  }, [editCam, addToast, activeStore]);
  const requestDeleteCamera = useCallback(cam => {
    setConfirmDeleteCam(cam);
  }, []);
  const handleConfirmDeleteCamera = useCallback(async () => {
    if (!confirmDeleteCam) return;
    const id = confirmDeleteCam.id;
    setCameras(prev => prev.filter(c => c.id !== id));
    setConfirmDeleteCam(null);
    try {
      await fetch(`${API}/cameras/${id}?store=${encodeURIComponent(activeStore)}`, {
        method: 'DELETE',
        headers: authFetchHeaders(),
      });
      addToast('Camera removed', 'success');
    } catch {
      const restored = await fetch(
        `${API}/cameras?store=${encodeURIComponent(activeStore)}`,
        { headers: authFetchHeaders() }
      ).then(r => r.json()).catch(() => []);
      setCameras(restored.map(c => ({ ...c, connection_status: c.connection_status || 'unknown' })));
      addToast('Failed to delete camera', 'error');
    }
  }, [confirmDeleteCam, activeStore, addToast]);
  const openEdit = useCallback(cam => { setEditCam(cam); setShowModal(true); }, []);
  const openAdd = useCallback(() => { setEditCam(null); setShowModal(true); }, []);
  const TABS = [
    { id: 'cameras', label: 'Cameras', icon: Camera },
    { id: 'editor', label: 'Lines & Zones', icon: Layers },
  ];
  return (
    <div className="settings-page">
      <Toast toasts={toasts} removeToast={removeToast} />
      {confirmDeleteCam && (
        <ConfirmModal
          title="Delete Camera"
          message="Are you sure you want to delete this camera? All associated lines and zones will also be removed."
          itemName={confirmDeleteCam.name}
          onConfirm={handleConfirmDeleteCamera}
          onCancel={() => setConfirmDeleteCam(null)}
        />
      )}
      <div className="settings-header">
        <div className="settings-header__left">
          {(isAdmin || stores.length > 1) && (
            <div className="settings-store-wrap">
              <Store size={13} className="settings-store-wrap__icon" />
              <select
                className="settings-store-select"
                value={activeStore}
                onChange={e => setActiveStore(e.target.value)}
              >
                {stores.map(s => (
                  <option key={s.id} value={s.store_name}>{s.store_name}</option>
                ))}
              </select>
              <ChevronDown size={11} className="settings-store-arrow" />
            </div>
          )}
        </div>
      {mode === 'all' && (
  <div className="settings-header__right">
    <div className="settings-tabs">
      {TABS.map(({ id, label, icon: Icon }) => (
        <button
          key={id}
          type="button"
          className={`settings-tab ${
            activeTab === id ? 'settings-tab--active' : ''
          }`}
          onClick={() => setActiveTab(id)}
        >
          <Icon size={14} />
          {label}
        </button>
      ))}
    </div>
  </div>
)}
      </div>
     {(mode === 'camera' || activeTab === 'cameras') && (
        <div className="settings-section">
          <div className="settings-section-header">
            <div>
              <h2 className="settings-section-title">Camera Sources</h2>
              <p className="settings-section-sub">
                Configure and verify your camera connections
                {activeStore && <span className="settings-section-sub__store"> - {activeStore}</span>}
              </p>
            </div>
            <button type="button" className="btn-add-cam" onClick={openAdd}>
              <Plus size={13} />Add Camera
            </button>
          </div>
          {loading ? (
            <div className="empty-state">
              <Loader2 size={30} className="spin" />
              <p>Loading cameras...</p>
            </div>
          ) : cameras.length === 0 ? (
            <div className="empty-state">
              <Camera size={36} />
              <p>No cameras configured yet</p>
              <button type="button" className="btn-add-cam" onClick={openAdd}>
                <Plus size={13} />Add your first camera
              </button>
            </div>
          ) : (
            <div className="cameras-grid">
              {cameras.map(cam => (
                <CameraCard
                  key={cam.id}
                  cam={cam}
                  onEdit={openEdit}
                  onDelete={requestDeleteCamera}
                  onToggle={handleToggle}
                />
              ))}
            </div>
          )}
        </div>
      )}
      {(mode === 'store' || activeTab === 'editor') && (
        <div className="settings-section">
          {cameras.length === 0 ? (
            <div className="empty-state">
              <Layers size={36} />
              <p>Add a camera first to define lines and zones</p>
            </div>
          ) : (
            <CanvasEditor cameras={cameras} addToast={addToast} activeStore={activeStore} />
          )}
        </div>
      )}
      {showModal && (
        <CameraModal
          cam={editCam}
          onSave={saveCamera}
          onClose={() => { setShowModal(false); setEditCam(null); }}
          saving={modalSaving}
        />
      )}
    </div>
  );
}
