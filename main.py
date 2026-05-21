"""
Vocal Pitch Monitor  v2.7
Author: ZaVoZ

Implemented Upgrades:
  - STAGE 1 OPT: Asynchronous audio loading worker (QThread) to prevent GUI freezes on load.
  - STAGE 1 OPT: Vocal range pitch clamping ( Aubio YIN candidate noise filtering).
  - STAGE 2 NEW: Live Vocal Range Tracker (Tessitura diagnostic: Min/Max notes hit).
  - STAGE 2 NEW: Session snapshot exporter (Saves accuracy metrics & graph to PNG).
  - NEW UX: Auto-resuming lyrics scroll window (pauses on user scroll, snaps back after 2.5s).
"""
import sys
import time as time_module
import threading
import collections
import logging
import numpy as np
import soundfile as sf
import sounddevice as sd
from aubio import pitch as aubio_pitch
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QCheckBox,
    QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QSlider,
    QDialog, QDoubleSpinBox, QDialogButtonBox, QFormLayout,
    QFrame, QSizePolicy, QGraphicsDropShadowEffect, QMessageBox,
    QProgressDialog, QComboBox, QSpinBox, QTextEdit, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QGraphicsOpacityEffect, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QObject, QPropertyAnimation
from PyQt6.QtGui import QFont, QColor, QKeySequence, QShortcut
from pydub import AudioSegment
from io import BytesIO
import tempfile
import os
import json
import shutil
import uuid
import re
from datetime import datetime
try:
    from vpm_core import smooth_pitch_rust
except ImportError:
    # Фоллбек-заглушка на случай, если Rust модуль не скомпилирован
    def smooth_pitch_rust(new_pitch, buffer):
        if not np.isnan(new_pitch):
            if len(buffer) > 0:
                med = np.median(buffer)
                if not np.isnan(med) and abs(new_pitch - med) > 1.5:
                    buffer.clear()
            buffer.append(new_pitch)
            return float(np.median(buffer)), buffer
        else:
            buffer.clear()
            return np.nan, buffer

# ─── ROCm / Polaris workaround (BEFORE importing torch) ──────────────────────
_POLARIS_GFX = {"gfx803", "gfx801", "gfx802", "gfx810"}


def _rocm_workaround():
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        return
    gfx = _detect_gfx_via_rocminfo()
    if gfx is not None and gfx in _POLARIS_GFX:
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"


def _detect_gfx_via_rocminfo() -> str | None:
    try:
        import subprocess, re
        out = subprocess.check_output(
            ["rocminfo"], stderr=subprocess.DEVNULL, timeout=6
        ).decode(errors="ignore")
        for line in out.splitlines():
            m = re.search(r'\bgfx\d+\b', line, re.IGNORECASE)
            if m:
                return m.group(0).lower()
    except Exception:
        pass
    return None


_rocm_workaround()

DEMUCS_AVAILABLE = False
TORCH_AVAILABLE = False
ROCM_AVAILABLE = False
CUDA_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    ROCM_AVAILABLE = getattr(torch.version, 'hip', None) is not None
    try:
        from demucs import pretrained
        from demucs.apply import apply_model
        DEMUCS_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass

DEMUCS_MODELS = {
    "htdemucs_ft  — fine-tuned, best quality": "htdemucs_ft",
    "htdemucs     — fast, ~2.5 GB VRAM": "htdemucs",
    "mdx_extra_q  — great on 4 GB VRAM": "mdx_extra_q",
    "mdx_extra    — high-quality MDX": "mdx_extra",
}
DEMUCS_DEFAULT = "htdemucs_ft"


def get_torch_device() -> str:
    if TORCH_AVAILABLE and (CUDA_AVAILABLE or ROCM_AVAILABLE):
        return "cuda"
    return "cpu"


def gpu_info_str() -> str:
    if not TORCH_AVAILABLE:
        return "PyTorch not installed"
    if ROCM_AVAILABLE:
        try:
            name = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "ROCm"
        except Exception:
            name = "ROCm"
        ov = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "")
        return f"ROCm  {name}" + (f"  [HSA={ov}]" if ov else "")
    if CUDA_AVAILABLE:
        try:
            return f"CUDA  {torch.cuda.get_device_name(0)}"
        except Exception:
            return "CUDA GPU"
    return "CPU only (no GPU detected)"


# ─── PALETTE ──────────────────────────────────────────────────────────────────
C_SONG = "#4fc3f7"
C_USER = "#ffca28"
C_GREEN = "#66bb6a"
C_YELLOW = "#ffa726"
C_RED = "#ef5350"
C_ACCENT = "#b39ddb"
BG = "#0d0d14"
SURFACE = "#161620"
SURFACE2 = "#1e1e2c"
SURFACE3 = "#262636"
BORDER = "#2e2e46"
TEXT = "#dcdcf0"
TEXT_DIM = "#525278"
TEXT_MID = "#9090b8"

SS = f"""
QMainWindow, QDialog  {{ background-color: {BG}; }}
QWidget {{
    color: {TEXT};
    font-family: 'JetBrains Mono','Fira Code','Consolas',monospace;
    font-size: 12px;
}}
QLabel {{ font-weight: 500; }}
QToolTip {{
    background: {SURFACE2}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 4px; padding: 4px 8px; font-size: 11px;
}}
QPushButton {{
    background: {SURFACE2}; border: 1px solid {BORDER};
    border-radius: 7px; padding: 6px 14px;
    font-weight: 600; color: {TEXT}; min-height: 32px;
}}
QPushButton:hover   {{ background: {SURFACE3}; border-color: {C_SONG}; color:#fff; }}
QPushButton:pressed {{ background: #2a3a4a; }}
QPushButton:disabled {{ background: {SURFACE}; color: {TEXT_DIM}; border-color:{BORDER}; }}
QPushButton[active="true"] {{
    background: #1a2e1a; border-color: {C_GREEN}; color: {C_GREEN};
}}
QSlider::groove:horizontal {{
    height:4px; background:{SURFACE3}; border-radius:2px;
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {C_SONG}, stop:1 #7be0ff);
    border-radius:2px;
}}
QSlider::handle:horizontal {{
    background:{TEXT}; width:12px; height:12px; margin:-4px 0;
    border-radius:6px; border:2px solid {C_SONG};
}}
QSlider::handle:horizontal:hover {{ background:white; }}
QCheckBox {{ spacing:7px; color:{TEXT_MID}; }}
QCheckBox::indicator {{
    width:15px; height:15px;
    border:1.5px solid {BORDER}; border-radius:4px; background:{SURFACE2};
}}
QCheckBox::indicator:checked {{ background:{C_SONG}; border-color:{C_SONG}; }}
QProgressDialog {{
    background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px;
}}
QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {{
    background:{SURFACE2}; border:1px solid {BORDER}; border-radius:5px;
    padding:3px 7px; color:{TEXT};
}}
QComboBox QAbstractItemView {{
    background:{SURFACE2}; color:{TEXT};
    selection-background-color:{SURFACE3};
}}
QDialogButtonBox QPushButton {{ min-width:80px; }}
QTableWidget {{
    background: {SURFACE2}; color: {TEXT}; gridline-color: {BORDER};
    border: 1px solid {BORDER}; border-radius: 6px;
}}
QTableWidget::item:selected {{ background: {SURFACE3}; }}
QHeaderView::section {{
    background: {SURFACE}; color: {TEXT_MID}; border: none; padding: 4px; font-weight: bold;
}}
QListWidget {{
    background: {SURFACE2}; border: 1px solid {BORDER}; border-radius: 6px; padding: 5px;
}}
QListWidget::item {{
    padding: 6px; border-radius: 4px; color: {TEXT_MID};
}}
QListWidget::item:selected {{
    background: {SURFACE3}; color: {C_SONG}; font-weight: bold;
}}
"""

# ─── Cache Manager ────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".vpm_cache")
INDEX_FILE = os.path.join(CACHE_DIR, "index.json")


class CacheManager:
    @staticmethod
    def init():
        os.makedirs(CACHE_DIR, exist_ok=True)
        if not os.path.exists(INDEX_FILE):
            with open(INDEX_FILE, "w", encoding="utf-8") as f:
                json.dump({}, f)

    @staticmethod
    def get_index():
        try:
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def save_index(data):
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def add(orig_name, model, tmp_wav_path, duration):
        CacheManager.init()
        idx = CacheManager.get_index()
        cid = uuid.uuid4().hex[:12]
        dest_name = f"{cid}.wav"
        dest_path = os.path.join(CACHE_DIR, dest_name)

        try:
            shutil.copy(tmp_wav_path, dest_path)
            idx[cid] = {
                "orig_name": orig_name,
                "model": model,
                "duration": duration,
                "filename": dest_name,
                "timestamp": time_module.time()
            }
            CacheManager.save_index(idx)
        except Exception as e:
            logging.error(f"Cache save failed: {e}")

    @staticmethod
    def remove(cid):
        idx = CacheManager.get_index()
        if cid in idx:
            try:
                os.remove(os.path.join(CACHE_DIR, idx[cid]["filename"]))
            except OSError:
                pass
            del idx[cid]
            CacheManager.save_index(idx)

    @staticmethod
    def get_size_mb():
        if not os.path.exists(CACHE_DIR):
            return 0.0
        total = sum(os.path.getsize(os.path.join(CACHE_DIR, f))
                    for f in os.listdir(CACHE_DIR) if os.path.isfile(os.path.join(CACHE_DIR, f)))
        return total / (1024 * 1024)


# ─── Thread-safe pitch queue ──────────────────────────────────────────────────
class PitchQueue:
    MAX_LEN = 2048

    def __init__(self):
        self._d = collections.deque(maxlen=self.MAX_LEN)
        self._lock = threading.Lock()

    def put(self, val: float):
        with self._lock:
            self._d.append(val)

    def get_nowait(self):
        with self._lock:
            return self._d.popleft() if self._d else None

    def clear(self):
        with self._lock:
            self._d.clear()

    def __len__(self):
        return len(self._d)


# ─── STAGE 1 OPT: Async Audio Decoder Worker ──────────────────────────────────
class AudioLoadWorker(QObject):
    finished = pyqtSignal(np.ndarray, int, str)
    error = pyqtSignal(str)

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            if self.file_path.lower().endswith(".mp3"):
                buf = BytesIO()
                AudioSegment.from_mp3(self.file_path).export(buf, format="wav")
                buf.seek(0)
                full_data, sr = sf.read(buf, dtype="float32")
            else:
                full_data, sr = sf.read(self.file_path, dtype="float32")

            if full_data.ndim > 1:
                full_data = np.mean(full_data, axis=1)

            self.finished.emit(full_data, sr, self.file_path)
        except Exception as e:
            self.error.emit(str(e))


# ─── Pitch Shift Worker ───────────────────────────────────────────────────────
class PitchShiftWorker(QObject):
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, data: np.ndarray, sr: int, semitones: int):
        super().__init__()
        self.data = data
        self.sr = sr
        self.semitones = semitones

    def run(self):
        try:
            import librosa
            shifted = librosa.effects.pitch_shift(
                y=self.data,
                sr=self.sr,
                n_steps=self.semitones,
                bins_per_octave=12,
                res_type='kaiser_best'
            )
            self.finished.emit(shifted)
        except Exception as e:
            import traceback
            self.error.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")


# ─── Demucs worker thread ─────────────────────────────────────────────────────
class DemucsWorker(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    progress_pct = pyqtSignal(int)

    def __init__(self, audio_path: str, device: str,
                 model_name: str = DEMUCS_DEFAULT):
        super().__init__()
        self.audio_path = audio_path
        self.device = device
        self.model_name = model_name
        self._tmp_files: list[str] = []

    def run(self):
        tmp_out = None
        try:
            self.progress.emit(f"Loading  '{self.model_name}'…")
            self.progress_pct.emit(5)
            model = pretrained.get_model(self.model_name)
            model.eval()

            self.progress.emit("Reading audio…")
            self.progress_pct.emit(12)
            wav, sr = sf.read(self.audio_path)
            if wav.ndim == 1:
                wav = np.stack([wav, wav])
            else:
                wav = wav.T
            wav = wav.astype(np.float32)

            device = self.device
            segment_sec = None
            if device == "cuda" and TORCH_AVAILABLE:
                try:
                    total_vram = torch.cuda.get_device_properties(0).total_memory
                    if total_vram < 5 * 1024 ** 3:
                        segment_sec = 7
                        self.progress.emit(
                            f"VRAM {total_vram // 1024 ** 2} MB  → segment={segment_sec}s…"
                        )
                except Exception:
                    segment_sec = 7

            mode = f"segmented {segment_sec}s" if segment_sec else "full"
            self.progress.emit(f"Separating on {device.upper()}  ({mode})…")
            self.progress_pct.emit(22)

            wav_t = torch.from_numpy(wav).unsqueeze(0)
            apply_kw = dict(device=device, progress=False, num_workers=0)
            if segment_sec:
                apply_kw["segment"] = segment_sec

            with torch.no_grad():
                sources = apply_model(model, wav_t, **apply_kw)

            self.progress_pct.emit(85)
            self.progress.emit("Extracting vocal stem…")

            src_names = list(model.sources)
            v_idx = (src_names.index("vocals")
                     if "vocals" in src_names
                     else len(src_names) - 1)
            vocal_stereo = sources[0, v_idx].cpu().numpy()
            vocal_mono = vocal_stereo.mean(axis=0)

            self.progress.emit("Spectral cleanup…")
            self.progress_pct.emit(92)
            vocal_mono = self._spectral_cleanup(vocal_mono, wav.mean(axis=0), sr)

            tmp_out = tempfile.mktemp(suffix="_vocal.wav")
            self._tmp_files.append(tmp_out)
            sf.write(tmp_out, vocal_mono.astype(np.float32), sr)

            self.progress_pct.emit(100)
            self.finished.emit(tmp_out)

        except Exception as exc:
            if TORCH_AVAILABLE and isinstance(exc, torch.cuda.OutOfMemoryError):
                self.progress.emit("⚠ GPU OOM — retrying on CPU…")
                self.device = "cpu"
                self.run()
                return
            import traceback
            self.error.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")
            self._cleanup_tmp()

    def _cleanup_tmp(self):
        for p in self._tmp_files:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        self._tmp_files.clear()

    @staticmethod
    def _spectral_cleanup(
            vocal: np.ndarray, mix: np.ndarray, sr: int,
            n_fft: int = 2048, hop: int = 512, power: float = 2.0,
    ) -> np.ndarray:
        try:
            from scipy import signal as sig
            from scipy.ndimage import uniform_filter

            _, _, V = sig.stft(vocal, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
            _, _, M = sig.stft(mix, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)

            eps = 1e-8
            magV = np.abs(V) ** power
            magR = np.abs(M - V) ** power
            mask = uniform_filter(
                magV / (magV + magR + eps), size=(1, 5)
            )
            _, out = sig.istft(V * mask, fs=sr,
                               nperseg=n_fft, noverlap=n_fft - hop)
            n = len(vocal)
            if len(out) >= n:
                return out[:n].astype(np.float32)
            return np.pad(out, (0, n - len(out))).astype(np.float32)
        except Exception:
            return vocal


# ─── Lyrics Fetcher Worker ────────────────────────────────────────────────────
class LyricsFetchWorker(QObject):
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)

    def __init__(self, query: str):
        super().__init__()
        self.query = query

    def run(self):
        import urllib.request
        import urllib.parse
        import json
        try:
            encoded_query = urllib.parse.quote(self.query)
            url = f"https://lrclib.net/api/search?q={encoded_query}"
            req = urllib.request.Request(url, headers={'User-Agent': 'VPM-Lyrics/2.7'})
            with urllib.request.urlopen(req, timeout=7) as response:
                data = json.loads(response.read().decode('utf-8'))
                if data:
                    for item in data:
                        if item.get("syncedLyrics"):
                            info = f"{item.get('artist', 'Unknown')} - {item.get('title', 'Unknown')}"
                            self.finished.emit(item["syncedLyrics"], info)
                            return
                    self.error.emit("Track found, but synchronized lyrics (LRC) are unavailable.")
                else:
                    self.error.emit("No results found in the database.")
        except Exception as e:
            self.error.emit(f"Connection failed: {e}")


# ─── Clickable Lyrics Panel ───────────────────────────────────────────────────
class ClickableFrame(QFrame):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


# ─── Full Lyrics Viewer Dialog with Auto-Reset ────────────────────────────────
class FullLyricsDialog(QDialog):
    seek_requested = pyqtSignal(float)

    def __init__(self, lyrics: list[tuple[float, str]], parent=None):
        super().__init__(parent)
        self.lyrics = lyrics
        self.setWindowTitle("📖 Full Song Lyrics")
        self.setMinimumSize(450, 500)

        # Scrolling interaction trackers
        self.is_user_scrolling = False
        self.scroll_reset_timer = QTimer(self)
        self.scroll_reset_timer.setSingleShot(True)
        self.scroll_reset_timer.timeout.connect(self._reset_scroll_lock)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)

        info_lbl = QLabel("Scroll freely. Autoscroll snaps back after 2.5s of inactivity.")
        info_lbl.setStyleSheet(f"color:{TEXT_MID}; font-size:11px;")
        info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(info_lbl)

        self.list_widget = QListWidget()
        lay.addWidget(self.list_widget)

        for timestamp, text in self.lyrics:
            item = QListWidgetItem(f"[{_fmt(timestamp)}]  {text}")
            item.setData(Qt.ItemDataRole.UserRole, timestamp)
            self.list_widget.addItem(item)

        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)

        # Wire scrollbar signals to intercept manual scroll
        self.scrollbar = self.list_widget.verticalScrollBar()
        self.scrollbar.sliderPressed.connect(self._on_slider_pressed)
        self.scrollbar.sliderReleased.connect(self._on_slider_released)
        self.scrollbar.valueChanged.connect(self._on_scroll_val_changed)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _on_item_double_clicked(self, item: QListWidgetItem):
        timestamp = item.data(Qt.ItemDataRole.UserRole)
        if timestamp is not None:
            self.seek_requested.emit(timestamp)

    def _on_slider_pressed(self):
        self.is_user_scrolling = True
        self.scroll_reset_timer.stop()

    def _on_slider_released(self):
        self.scroll_reset_timer.start(2500)  # Reset follow state after 2.5s of idling

    def _on_scroll_val_changed(self):
        # Fallback to capture trackpad/mousewheel triggers
        if not self.is_user_scrolling and not self.scroll_reset_timer.isActive():
            if self.list_widget.hasFocus() or self.list_widget.underMouse():
                self.is_user_scrolling = True
                self.scroll_reset_timer.start(2500)

    def _reset_scroll_lock(self):
        self.is_user_scrolling = False

    def highlight_line(self, index: int):
        if 0 <= index < self.list_widget.count():
            # Only update selection state, but skip viewport scroll if user is manually scrolling
            self.list_widget.setCurrentRow(index)
            if not self.is_user_scrolling:
                self.list_widget.scrollToItem(self.list_widget.currentItem(), QAbstractItemView.ScrollHint.PositionAtCenter)


# ─── Debug stats ─────────────────────────────────────────────────────────────
class DebugStats:
    __slots__ = (
        "cb_time_song_ms", "cb_time_mic_ms",
        "cb_time_monitor_ms",
        "queue_song", "queue_mic",
        "dropped_song", "dropped_mic",
        "rms_mic", "conf_mic",
        "rms_song", "conf_song",
        "monitor_vol",
        "frames_rendered", "frames_skipped",
        "last_reset",
    )

    def __init__(self):
        self.cb_time_song_ms: float = 0.0
        self.cb_time_mic_ms: float = 0.0
        self.cb_time_monitor_ms: float = 0.0
        self.queue_song: int = 0
        self.queue_mic: int = 0
        self.dropped_song: int = 0
        self.dropped_mic: int = 0
        self.rms_mic: float = 0.0
        self.conf_mic: float = 0.0
        self.rms_song: float = 0.0
        self.conf_song: float = 0.0
        self.monitor_vol: float = 1.0
        self.frames_rendered: int = 0
        self.frames_skipped: int = 0
        self.last_reset = time_module.time()

    def reset_counters(self):
        self.dropped_song = 0
        self.dropped_mic = 0
        self.frames_rendered = 0
        self.frames_skipped = 0
        self.last_reset = time_module.time()


# ─── Debug window ─────────────────────────────────────────────────────────────
class DebugWindow(QDialog):
    def __init__(self, stats: DebugStats, parent=None):
        super().__init__(parent)
        self.stats = stats
        self.setWindowTitle("🐛  Debug — Vocal Pitch Monitor")
        self.setMinimumWidth(440)
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        hdr = QLabel("Live audio & performance diagnostics (Updates 2Hz)")
        hdr.setStyleSheet(
            f"color:{C_ACCENT}; font-weight:700; font-size:11px;")
        lay.addWidget(hdr)

        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setFont(QFont("JetBrains Mono,Fira Code,Consolas,monospace", 11))
        self.txt.setStyleSheet(
            f"background:{SURFACE2}; color:{TEXT}; border:1px solid {BORDER};"
            " border-radius:6px;")
        self.txt.setMinimumHeight(380)
        lay.addWidget(self.txt)

        btn_row = QHBoxLayout()
        btn_reset = QPushButton("↺  Reset counters")
        btn_reset.clicked.connect(self._reset)
        btn_row.addWidget(btn_reset)
        btn_row.addStretch()
        btn_close = QPushButton("✕  Close")
        btn_close.clicked.connect(self.hide)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(500)
        self._refresh()

    def _reset(self):
        self.stats.reset_counters()

    def _bar(self, val: float, max_val: float, width: int = 20) -> str:
        filled = int(min(1.0, val / max(max_val, 1e-9)) * width)
        return "█" * filled + "░" * (width - filled)

    def _refresh(self):
        s = self.stats
        elapsed = max(0.001, time_module.time() - s.last_reset)
        fps_rendered = s.frames_rendered / elapsed
        fps_skipped = s.frames_skipped / elapsed

        mon_budget_ms = 256 / 44100 * 1000
        mic_budget_ms = 512 / 44100 * 1000

        rms_bar = self._bar(s.rms_mic, 0.5)
        conf_bar = self._bar(s.conf_mic, 1.0)

        cb_mon_warn = " ⚠ OVER" if s.cb_time_monitor_ms > mon_budget_ms * 0.8 else ""
        cb_mic_warn = " ⚠ OVER" if s.cb_time_mic_ms > mic_budget_ms * 0.8 else ""
        cb_song_warn = " ⚠ OVER" if s.cb_time_song_ms > mic_budget_ms * 0.8 else ""

        lines = [
            "─── AUDIO CALLBACKS ───────────────────────",
            f"  Monitor callback  : {s.cb_time_monitor_ms:6.3f} ms / {mon_budget_ms:.1f} ms {cb_mon_warn}",
            f"  Mic callback      : {s.cb_time_mic_ms:6.3f} ms / {mic_budget_ms:.1f} ms {cb_mic_warn}",
            f"  Song callback     : {s.cb_time_song_ms:6.3f} ms / {mic_budget_ms:.1f} ms {cb_song_warn}",
            "",
            "─── PITCH QUEUES ───────────────────────────",
            f"  Song queue depth  : {s.queue_song:4d} items",
            f"  Mic  queue depth  : {s.queue_mic:4d} items",
            f"  Dropped (song)    : {s.dropped_song:4d}  ({s.dropped_song / elapsed:.1f}/s)",
            f"  Dropped (mic)     : {s.dropped_mic:4d}  ({s.dropped_mic / elapsed:.1f}/s)",
            "",
            "─── SIGNAL & CONFIDENCE ────────────────────",
            f"  Mic RMS           : {s.rms_mic:.4f}  {rms_bar}",
            f"  Mic Confidence    : {s.conf_mic:.3f}  {conf_bar}",
            f"  Song RMS          : {s.rms_song:.4f}",
            f"  Song Confidence   : {s.conf_song:.3f}",
            "",
            "─── MONITOR ────────────────────────────────",
            f"  Monitor volume    : {s.monitor_vol * 100:.0f}%",
            "",
            "─── RENDER LOOP (PERF) ─────────────────────",
            f"  Frames rendered   : {s.frames_rendered:6d}  ({fps_rendered:.1f} FPS)",
            f"  Frames skipped    : {s.frames_skipped:6d}  ({fps_skipped:.1f} FPS) (Guard)",
            f"  Counter window    : {elapsed:.1f} s",
        ]
        self.txt.setPlainText("\n".join(lines))

    def closeEvent(self, event):
        self._timer.stop()
        self.hide()
        event.ignore()


# ─── Cache Dialog ─────────────────────────────────────────────────────────────
class CacheDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🗃️ Cached Vocals Library")
        self.resize(720, 360)
        self.selected_path = None
        self.selected_name = None

        lay = QVBoxLayout(self)

        info_lay = QHBoxLayout()
        info_lay.addWidget(QLabel("Select a previously separated track to load it instantly."))
        info_lay.addStretch()
        self.size_lbl = QLabel()
        self.size_lbl.setStyleSheet(f"color:{TEXT_MID};")
        info_lay.addWidget(self.size_lbl)
        lay.addLayout(info_lay)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Date", "Original Track", "Model", "Duration"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        lay.addWidget(self.table)

        btn_lay = QHBoxLayout()
        btn_del = QPushButton("🗑 Delete Selected")
        btn_del.clicked.connect(self._delete_selected)
        btn_lay.addWidget(btn_del)
        btn_lay.addStretch()

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_load = QPushButton("📂 Load")
        btn_load.setProperty("active", "true")
        btn_load.clicked.connect(self._load_selected)

        btn_lay.addWidget(btn_cancel)
        btn_lay.addWidget(btn_load)
        lay.addLayout(btn_lay)

        self._refresh()

    def _refresh(self):
        CacheManager.init()
        idx = CacheManager.get_index()
        self.table.setRowCount(len(idx))

        self.size_lbl.setText(f"Cache size: {CacheManager.get_size_mb():.1f} MB")

        for row, (cid, data) in enumerate(sorted(idx.items(), key=lambda x: x[1].get('timestamp', 0), reverse=True)):
            dt = datetime.fromtimestamp(data.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M")
            dur = f"{data.get('duration', 0):.1f}s"

            i_date = QTableWidgetItem(dt)
            i_date.setData(Qt.ItemDataRole.UserRole, cid)

            self.table.setItem(row, 0, i_date)
            self.table.setItem(row, 1, QTableWidgetItem(data.get("orig_name", "Unknown")))
            self.table.setItem(row, 2, QTableWidgetItem(data.get("model", "")))
            self.table.setItem(row, 3, QTableWidgetItem(dur))

    def _delete_selected(self):
        r = self.table.currentRow()
        if r < 0: return
        cid = self.table.item(r, 0).data(Qt.ItemDataRole.UserRole)
        CacheManager.remove(cid)
        self._refresh()

    def _load_selected(self):
        r = self.table.currentRow()
        if r < 0: return
        cid = self.table.item(r, 0).data(Qt.ItemDataRole.UserRole)
        idx = CacheManager.get_index()
        if cid in idx:
            path = os.path.join(CACHE_DIR, idx[cid]["filename"])
            if not os.path.exists(path):
                QMessageBox.warning(self, "Error", "Cached file is missing from disk.")
                CacheManager.remove(cid)
                self._refresh()
                return
            self.selected_path = path
            self.selected_name = f"Cached: {idx[cid].get('orig_name')}"
            self.accept()


# ─── JumpSlider ───────────────────────────────────────────────────────────────
class JumpSlider(QSlider):
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            ratio = event.pos().x() / max(1, self.width())
            val = int(self.minimum() +
                      (self.maximum() - self.minimum()) * ratio)
            self.setValue(val)
        super().mousePressEvent(event)


# ─── TrimDialog ───────────────────────────────────────────────────────────────
class TrimDialog(QDialog):
    def __init__(self, parent, full_data: np.ndarray, sr: int, display_name: str = ""):
        super().__init__(parent)
        title_suffix = f" - {display_name}" if display_name else ""
        self.setWindowTitle(f"✂  Trim Audio{title_suffix}")
        self.setModal(True)
        self.total_sec = len(full_data) / sr

        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        wp = pg.PlotWidget()
        wp.setBackground(BG)
        wp.setMouseEnabled(x=True, y=False)
        wp.setLabel("bottom", "Time  (s)")
        wp.getAxis("bottom").setPen(pg.mkPen(TEXT_DIM))
        wp.getAxis("bottom").setTextPen(pg.mkPen(TEXT_MID))
        wp.getAxis("left").hide()

        step = max(1, len(full_data) // 8000)
        t = np.arange(0, len(full_data), step) / sr
        wp.plot(t, full_data[::step], pen=pg.mkPen(C_SONG, width=1.2))

        self.start_line = pg.InfiniteLine(
            pos=0, angle=90, movable=True,
            pen=pg.mkPen(C_USER, width=2.5),
            label="▶ start", labelOpts={"color": C_USER, "position": 0.93},
        )
        self.end_line = pg.InfiniteLine(
            pos=self.total_sec, angle=90, movable=True,
            pen=pg.mkPen(C_GREEN, width=2.5),
            label="◼ end", labelOpts={"color": C_GREEN, "position": 0.93},
        )
        wp.addItem(self.start_line)
        wp.addItem(self.end_line)
        self.start_line.sigDragged.connect(self._lines_to_spins)
        self.end_line.sigDragged.connect(self._lines_to_spins)
        lay.addWidget(wp)

        form = QFormLayout()
        form.setSpacing(6)

        def _spin(lo, hi, val, step=0.1):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setSuffix(" s")
            s.setDecimals(2)
            return s

        self.start_spin = _spin(0.0, self.total_sec - 0.05, 0.0)
        self.end_spin = _spin(0.05, self.total_sec, self.total_sec)
        self.dur_label = QLabel()

        self.start_spin.valueChanged.connect(self._spins_to_lines)
        self.end_spin.valueChanged.connect(self._spins_to_lines)
        self._refresh_dur()

        form.addRow("Start:", self.start_spin)
        form.addRow("End:", self.end_spin)
        form.addRow("Duration:", self.dur_label)
        lay.addLayout(form)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("✂  Trim & Load")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)
        self.resize(960, 440)

    def _refresh_dur(self):
        dur = max(0.0, self.end_spin.value() - self.start_spin.value())
        self.dur_label.setText(f"{dur:.2f} s  ({_fmt(dur)})")
        self.dur_label.setStyleSheet(f"color:{TEXT_MID};")

    def _lines_to_spins(self):
        s = max(0.0, self.start_line.value())
        e = min(self.total_sec, self.end_line.value())
        if s >= e - 0.05:
            e = s + 0.05
        for w in (self.start_spin, self.end_spin):
            w.blockSignals(True)
        self.start_spin.setValue(s)
        self.end_spin.setValue(e)
        for w in (self.start_spin, self.end_spin):
            w.blockSignals(False)
        self._refresh_dur()

    def _spins_to_lines(self):
        self.start_line.setValue(self.start_spin.value())
        self.end_line.setValue(self.end_spin.value())
        self._refresh_dur()

    def get_trim(self):
        return self.start_spin.value(), self.end_spin.value()


# ─── helpers ──────────────────────────────────────────────────────────────────
def _fmt(sec: float) -> str:
    m, s = int(sec) // 60, int(sec) % 60
    return f"{m}:{s:02d}"


# ─── Main Window ──────────────────────────────────────────────────────────────
class VocalPitchMonitor(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎤  Vocal Pitch Monitor  v2.7")
        self.setMinimumSize(980, 880)

        # ── audio state ──────────────────────────────────────────────────────
        self.stats = DebugStats()
        self._rendering = False

        self.audio_data: np.ndarray | None = None
        self.original_audio_data: np.ndarray | None = None
        self.sample_rate: int | None = None
        self.current_frame: int = 0
        self.is_playing: bool = False
        self.mic_active: bool = False
        self.overlay_mic: bool = False
        self.stream = None
        self.mic_stream = None
        self.monitor_stream = None
        self.monitor_active: bool = False
        self.monitor_volume: float = 1.0
        self._stream_lock = threading.Lock()

        self.volume: float = 0.8
        self.hop_size: int = 512
        self.conf_thresh: float = 0.4

        # ── pitch history ─────────────────────────────────────────────────────
        self.history_size = 600
        self.plot_song = np.full(self.history_size, np.nan)
        self.plot_user = np.full(self.history_size, np.nan)

        self.song_pitch_queue = PitchQueue()
        self.user_pitch_queue = PitchQueue()

        self.song_buffer = collections.deque(maxlen=3)
        self.user_buffer = collections.deque(maxlen=3)

        self.song_tracker = {"last": np.nan, "hold": 0}
        self.user_tracker = {"last": np.nan, "hold": 0}

        # ── STAGE 2: Live Tessitura metrics (Vocal Range) ─────────────────────
        self.tess_min_midi = 999.0
        self.tess_max_midi = 0.0

        # ── UI / camera state ─────────────────────────────────────────────────
        self.camera_y: float = 60.0
        self.auto_scroll_paused_until: float = 0.0
        self.accuracy_history = collections.deque(maxlen=300)
        self._seek_guard: bool = False
        self.slider_dragging: bool = False
        self.last_update_time: float = time_module.time()
        self.song_scroll_accum: float = 0.0
        self.user_scroll_accum: float = 0.0

        # ── Lyrics & Trim Offset state ────────────────────────────────────────
        self.lyrics: list[tuple[float, str]] = []
        self._last_lyrics_idx: int = -2
        self.trim_start_sec: float = 0.0
        self.lyrics_offset: float = 0.0
        self.full_lyrics_view: FullLyricsDialog | None = None

        # ── Thread & Workers ──────────────────────────────────────────────────
        self._demucs_thread = None
        self._demucs_worker = None
        self._demucs_dlg = None

        self._ps_thread = None
        self._ps_worker = None
        self._pitch_dlg = None

        self._lyrics_thread = None
        self._lyrics_worker = None
        self._lyrics_dlg = None

        self._load_thread = None
        self._load_worker = None
        self._load_dlg = None

        self._pending_cache_info = None

        self.setup_ui()
        self._setup_shortcuts()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(16)

        self.create_user_detector()

    def create_user_detector(self):
        self.user_detector = aubio_pitch("yin", 2048, self.hop_size, 44100)
        self.user_detector.set_unit("midi")
        self.user_detector.set_tolerance(self.conf_thresh)

    def create_song_detector(self, sr: int):
        self.song_detector = aubio_pitch("yin", 2048, self.hop_size, sr)
        self.song_detector.set_unit("midi")
        self.song_detector.set_tolerance(self.conf_thresh)

    def _setup_shortcuts(self):
        def _sc(key, fn):
            s = QShortcut(QKeySequence(key), self)
            s.activated.connect(fn)

        _sc("Space", self.toggle_play)
        _sc("R", self.do_restart)
        _sc("M", self.toggle_mic_only)
        _sc("H", self.toggle_monitor)
        _sc("C", self.clear_graph)
        _sc("Ctrl+O", self.load_track)

    # ─────────────────────────────────────────────────────────────────────────
    # UI Setup
    # ─────────────────────────────────────────────────────────────────────────
    def setup_ui(self):
        root = QWidget()
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(16, 12, 16, 10)
        vbox.setSpacing(8)

        # ── top cards + center ────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(12)

        self.song_card, self.song_note_label = self._make_note_card(C_SONG, "🎵 Original")
        self.user_card, self.user_note_label = self._make_note_card(C_USER, "🎤 Your voice")

        center_w = QWidget()
        center_w.setSizePolicy(QSizePolicy.Policy.Expanding,
                               QSizePolicy.Policy.Preferred)
        cv = QVBoxLayout(center_w)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(3)
        cv.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_label = QLabel("Load a track or enable microphone")
        self.status_label.setFont(QFont("Consolas", 11))
        self.status_label.setStyleSheet(f"color:{TEXT_DIM};")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.cents_label = QLabel("")
        self.cents_label.setFont(QFont("Consolas", 18, QFont.Weight.Bold))
        self.cents_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.accuracy_label = QLabel("")
        self.accuracy_label.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px;")
        self.accuracy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.accuracy_label.setWordWrap(True)

        # STAGE 2: Vocal Range tessitura panel label
        self.tess_label = QLabel("Vocal range: Min -- | Max --")
        self.tess_label.setStyleSheet(f"color:{C_ACCENT}; font-size:11px; font-weight:bold;")
        self.tess_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gpu_label = QLabel(f"⚙  {gpu_info_str()}")
        self.gpu_label.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:10px; font-style:italic;")
        self.gpu_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        cv.addWidget(self.status_label)
        cv.addWidget(self.cents_label)
        cv.addWidget(self.accuracy_label)
        cv.addWidget(self.tess_label)
        cv.addSpacing(1)
        cv.addWidget(self.gpu_label)

        top.addWidget(self.song_card)
        top.addWidget(center_w)
        top.addWidget(self.user_card)
        vbox.addLayout(top)

        # ── pitch plot ────────────────────────────────────────────────────────
        self.plot = pg.PlotWidget()
        self.plot.setBackground(BG)
        self.plot.setMouseEnabled(x=False, y=True)
        self.plot.getViewBox().setLimits(
            minYRange=12, maxYRange=80, yMin=20, yMax=100)
        self.plot.setYRange(47, 73, padding=0)
        self.plot.hideAxis("bottom")
        self.plot.showAxis("right")

        for m in range(20, 101):
            bc = "#111118" if (m % 12) in {1, 3, 6, 8, 10} else "#161622"
            self.plot.addItem(
                pg.InfiniteLine(pos=m, angle=0, pen=pg.mkPen(bc, width=18)))
            if m % 12 == 0:
                self.plot.addItem(
                    pg.InfiniteLine(pos=m - 0.5, angle=0,
                                    pen=pg.mkPen("#2d2d50", width=2)))

        NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        ticks = [(m, f"{NOTES[m % 12]}{m // 12 - 1}") for m in range(20, 101)]
        for ax in ("left", "right"):
            a = self.plot.getAxis(ax)
            a.setTicks([ticks, []])
            a.setStyle(tickFont=QFont("Consolas", 9))
            a.setPen(pg.mkPen("#2d2d50"))
            a.setTextPen(pg.mkPen(TEXT_DIM))

        def _glow(c, a=45):
            q = QColor(c)
            q.setAlpha(a)
            return q

        self.curve_song_glow = self.plot.plot(
            pen=pg.mkPen(_glow(C_SONG), width=9))
        self.curve_song = self.plot.plot(pen=pg.mkPen(C_SONG, width=2.5))
        self.curve_user_glow = self.plot.plot(
            pen=pg.mkPen(_glow(C_USER), width=9))
        self.curve_user = self.plot.plot(pen=pg.mkPen(C_USER, width=2.5))

        vbox.addWidget(self.plot, stretch=1)

        # ── SYNCHRONIZED LYRICS PANEL ─────────────────────────────────────────
        self.lyrics_panel = ClickableFrame()
        self.lyrics_panel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lyrics_panel.setToolTip("Click to open full song lyrics")
        self.lyrics_panel.setStyleSheet(
            f"QFrame {{ background: {SURFACE}; border: 1.5px solid {BORDER};"
            " border-radius: 10px; margin-bottom: 2px; }"
        )
        ly_main_layout = QHBoxLayout(self.lyrics_panel)
        ly_main_layout.setContentsMargins(16, 6, 16, 6)
        ly_main_layout.setSpacing(12)

        ly_text_container = QWidget()
        ly_text_container.setStyleSheet("background: transparent; border: none;")
        ly_layout = QVBoxLayout(ly_text_container)
        ly_layout.setContentsMargins(0, 0, 0, 0)
        ly_layout.setSpacing(2)

        self.lyrics_prev_lbl = QLabel("")
        self.lyrics_prev_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lyrics_prev_lbl.setFont(QFont("JetBrains Mono", 10))
        self.lyrics_prev_lbl.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; border: none;")

        self.lyrics_curr_lbl = QLabel("Lyrics teleprompter")
        self.lyrics_curr_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lyrics_curr_lbl.setFont(QFont("JetBrains Mono", 14, QFont.Weight.Bold))
        self.lyrics_curr_lbl.setStyleSheet(f"color: {C_SONG}; background: transparent; border: none;")

        self.ly_opacity_effect = QGraphicsOpacityEffect(self.lyrics_curr_lbl)
        self.lyrics_curr_lbl.setGraphicsEffect(self.ly_opacity_effect)
        self.ly_fade_anim = QPropertyAnimation(self.ly_opacity_effect, b"opacity")
        self.ly_fade_anim.setDuration(180)

        self.lyrics_next_lbl = QLabel("")
        self.lyrics_next_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lyrics_next_lbl.setFont(QFont("JetBrains Mono", 10))
        self.lyrics_next_lbl.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; border: none;")

        ly_layout.addWidget(self.lyrics_prev_lbl)
        ly_layout.addWidget(self.lyrics_curr_lbl)
        ly_layout.addWidget(self.lyrics_next_lbl)
        ly_main_layout.addWidget(ly_text_container, stretch=1)

        self.offset_widget = QWidget()
        self.offset_widget.setStyleSheet("background: transparent; border: none;")
        offset_layout = QVBoxLayout(self.offset_widget)
        offset_layout.setContentsMargins(0, 0, 0, 0)
        offset_layout.setSpacing(3)
        offset_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        offset_hdr = QLabel("Sync Offset")
        offset_hdr.setFont(QFont("Consolas", 9))
        offset_hdr.setStyleSheet(f"color: {TEXT_DIM};")
        offset_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.btn_offset_dec = QPushButton("⏪ -0.5s")
        self.btn_offset_dec.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self.btn_offset_dec.setStyleSheet("padding: 2px 6px; min-height:22px; max-width:64px;")

        self.btn_offset_inc = QPushButton("+0.5s ⏩")
        self.btn_offset_inc.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self.btn_offset_inc.setStyleSheet("padding: 2px 6px; min-height:22px; max-width:64px;")

        btn_row.addWidget(self.btn_offset_dec)
        btn_row.addWidget(self.btn_offset_inc)

        self.offset_val_label = QLabel("0.0s")
        self.offset_val_label.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self.offset_val_label.setStyleSheet(f"color: {TEXT_MID};")
        self.offset_val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        offset_layout.addWidget(offset_hdr)
        offset_layout.addLayout(btn_row)
        offset_layout.addWidget(self.offset_val_label)
        ly_main_layout.addWidget(self.offset_widget)

        self.btn_offset_dec.clicked.connect(lambda: self.adjust_lyrics_offset(-0.5))
        self.btn_offset_inc.clicked.connect(lambda: self.adjust_lyrics_offset(0.5))
        self.lyrics_panel.clicked.connect(self.show_full_lyrics_viewer)

        vbox.addWidget(self.lyrics_panel)

        # ── seek ──────────────────────────────────────────────────────────────
        seek_row = QHBoxLayout()
        seek_row.setSpacing(10)
        self.seek_slider = JumpSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 10000)
        self.seek_slider.setValue(0)
        self.seek_slider.setEnabled(False)
        self.seek_slider.sliderPressed.connect(self._on_seek_pressed)
        self.seek_slider.valueChanged.connect(self._on_seek_moved)
        self.seek_slider.sliderReleased.connect(self._on_seek_released)
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:11px; font-family:'Consolas',monospace;")
        self.time_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.time_label.setFixedWidth(100)
        seek_row.addWidget(self.seek_slider)
        seek_row.addWidget(self.time_label)
        vbox.addLayout(seek_row)

        # ── controls ──────────────────────────────────────────────────────────
        ctrl_wrap = QVBoxLayout()
        ctrl_wrap.setSpacing(5)

        row1 = QHBoxLayout()
        row1.setSpacing(6)

        self.btn_load_track = QPushButton("📂 Load Track")
        self.btn_load_track.setMinimumWidth(110)
        self.btn_load_separate = QPushButton("🔀 Separate Vocals")
        self.btn_load_separate.setMinimumWidth(130)
        if not DEMUCS_AVAILABLE:
            self.btn_load_separate.setEnabled(False)
            self.btn_load_separate.setToolTip("Demucs not installed. Run: pip install demucs")
        else:
            self.btn_load_separate.setToolTip(f"Vocal separation via Demucs\n{gpu_info_str()}")

        self.btn_cache = QPushButton("🗃️ Cache")
        self.btn_cache.setMinimumWidth(80)
        self.btn_cache.setToolTip("Load previously separated vocals instantly")

        self.btn_lyrics = QPushButton("🎵 Lyrics")
        self.btn_lyrics.setMinimumWidth(80)
        self.btn_lyrics.setToolTip("Import local LRC or fetch synced lyrics online")
        self.btn_lyrics.clicked.connect(self.show_lyrics_dialog)

        # Pitch Shift Controls
        lbl_pitch = QLabel("Key:")
        lbl_pitch.setStyleSheet(f"color:{TEXT_MID};")
        self.spin_pitch = QSpinBox()
        self.spin_pitch.setRange(-24, 24)
        self.spin_pitch.setValue(0)
        self.spin_pitch.setSuffix(" st")
        self.spin_pitch.setFixedWidth(64)

        self.btn_apply_pitch = QPushButton("🔄 Apply")
        self.btn_apply_pitch.setMinimumWidth(80)
        self.btn_apply_pitch.clicked.connect(self.apply_pitch_shift)

        self.btn_restart = QPushButton("⏮ Restart")
        self.btn_restart.setMinimumWidth(90)
        self.btn_restart.setEnabled(False)

        self.btn_play = QPushButton("▶  Play")
        self.btn_play.setEnabled(False)
        self.btn_play.setMinimumWidth(110)

        # Playback volume
        vol_icon = QLabel("🔊")
        vol_icon.setStyleSheet(f"color:{TEXT_DIM};")
        self.vol_slider = JumpSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(80)
        self.vol_slider.setFixedWidth(100)

        for w in (self.btn_load_track, self.btn_load_separate, self.btn_cache, self.btn_lyrics,
                  self._vline(),
                  lbl_pitch, self.spin_pitch, self.btn_apply_pitch,
                  self._vline(),
                  self.btn_restart, self.btn_play):
            row1.addWidget(w)
        row1.addStretch()
        row1.addWidget(vol_icon)
        row1.addWidget(self.vol_slider)

        row2 = QHBoxLayout()
        row2.setSpacing(6)

        self.btn_mic = QPushButton("🎤 Mic")
        self.btn_mic.setMinimumWidth(80)

        self.btn_monitor = QPushButton("🎧 Monitor")
        self.btn_monitor.setMinimumWidth(100)

        lbl_mon_vol = QLabel("Mon:")
        lbl_mon_vol.setStyleSheet(f"color:{TEXT_MID}; font-size:11px;")
        self.monitor_vol_slider = JumpSlider(Qt.Orientation.Horizontal)
        self.monitor_vol_slider.setRange(0, 150)
        self.monitor_vol_slider.setValue(100)
        self.monitor_vol_slider.setFixedWidth(80)

        self.chk_overlay = QCheckBox("Sing along")
        self.chk_overlay.setEnabled(False)

        self.btn_clear = QPushButton("🧹 Clear")
        self.btn_clear.setMinimumWidth(80)

        # STAGE 2: Session Export Button
        self.btn_export = QPushButton("📸 Export")
        self.btn_export.setMinimumWidth(80)
        self.btn_export.setToolTip("Export current session progress graph as PNG image")
        self.btn_export.clicked.connect(self.export_session_snapshot)

        # Zoom Widget
        zoom_w = QWidget()
        zl = QHBoxLayout(zoom_w)
        zl.setSpacing(2)
        zl.setContentsMargins(0, 0, 0, 0)
        self.btn_zoom_x_out = QPushButton("◀▶")
        self.btn_zoom_x_in = QPushButton("▶◀")
        self.btn_zoom_y_out = QPushButton("▼▲")
        self.btn_zoom_y_in = QPushButton("▲▼")
        for b in (self.btn_zoom_x_out, self.btn_zoom_x_in,
                  self.btn_zoom_y_out, self.btn_zoom_y_in):
            b.setFixedWidth(34)
            b.setStyleSheet("font-size:11px;font-weight:700;padding:3px 4px;min-height:28px;")
            zl.addWidget(b)

        # Confidence threshold slider
        lbl_conf = QLabel("Conf:")
        lbl_conf.setStyleSheet(f"color:{TEXT_MID}; font-size:11px;")
        self.conf_slider = JumpSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(int(self.conf_thresh * 100))
        self.conf_slider.setFixedWidth(75)
        self.conf_val_label = QLabel(f"{self.conf_thresh:.2f}")
        self.conf_val_label.setStyleSheet(f"color:{TEXT_MID}; font-size:11px;")
        self.conf_val_label.setFixedWidth(30)

        def _on_conf_changed(v: int):
            self.conf_thresh = v / 100.0
            self.conf_val_label.setText(f"{self.conf_thresh:.2f}")
            self.create_user_detector()
            if self.audio_data is not None and self.sample_rate is not None:
                self.create_song_detector(self.sample_rate)

        self.conf_slider.valueChanged.connect(_on_conf_changed)

        self.btn_debug = QPushButton("🐛 Debug")
        self.btn_debug.setMinimumWidth(80)

        self.btn_gpu_info = QPushButton("⚡ GPU")
        self.btn_gpu_info.setMinimumWidth(70)

        for w in (self.btn_mic, self.btn_monitor, lbl_mon_vol,
                  self.monitor_vol_slider, self.chk_overlay,
                  self._vline(),
                  self.btn_clear, self.btn_export, zoom_w,
                  self._vline(),
                  lbl_conf, self.conf_slider, self.conf_val_label,
                  self._vline(),
                  self.btn_debug, self.btn_gpu_info):
            row2.addWidget(w)
        row2.addStretch()

        self.btn_load_track.clicked.connect(self.load_track)
        self.btn_load_separate.clicked.connect(self.load_and_separate)
        self.btn_cache.clicked.connect(self.open_cache)
        self.btn_restart.clicked.connect(self.do_restart)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_mic.clicked.connect(self.toggle_mic_only)
        self.btn_monitor.clicked.connect(self.toggle_monitor)
        self.monitor_vol_slider.valueChanged.connect(self._on_monitor_vol_changed)
        self.btn_clear.clicked.connect(self.clear_graph)
        self.chk_overlay.stateChanged.connect(self._on_overlay_changed)
        self.vol_slider.valueChanged.connect(lambda v: setattr(self, "volume", v / 100))
        self.btn_zoom_x_in.clicked.connect(self.zoom_x_in)
        self.btn_zoom_x_out.clicked.connect(self.zoom_x_out)
        self.btn_zoom_y_in.clicked.connect(self.zoom_y_in)
        self.btn_zoom_y_out.clicked.connect(self.zoom_y_out)
        self.btn_debug.clicked.connect(self._show_debug_window)
        self.btn_gpu_info.clicked.connect(self._show_gpu_info)

        ctrl_wrap.addLayout(row1)
        ctrl_wrap.addLayout(row2)
        vbox.addLayout(ctrl_wrap)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 2, 18, 6)
        footer.addStretch()
        lbl = QLabel("by ZaVoZ  ·  v2.7  ·  Space=play  R=restart  M=mic  H=monitor  C=clear")
        lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px; font-style:italic; letter-spacing:1px;")
        sh = QGraphicsDropShadowEffect(self)
        sh.setBlurRadius(16)
        sh.setColor(QColor(79, 195, 247, 55))
        sh.setOffset(0, 0)
        lbl.setGraphicsEffect(sh)
        footer.addWidget(lbl)
        vbox.addLayout(footer)
        self.setCentralWidget(root)

    def _make_note_card(self, color: str, title: str):
        frame = QFrame()
        frame.setObjectName("noteCard")
        frame.setStyleSheet(
            f"QFrame#noteCard {{background:{SURFACE}; border:1.5px solid {BORDER};"
            f" border-radius:12px;}}")
        frame.setFixedWidth(170)
        inner = QVBoxLayout(frame)
        inner.setContentsMargins(8, 8, 8, 10)
        inner.setSpacing(2)

        lbl_t = QLabel(title)
        lbl_t.setStyleSheet(
            f"color:{color}; font-weight:700; font-size:11px;"
            " background:transparent; border:none;")
        lbl_t.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lbl_n = QLabel("--")
        lbl_n.setFont(QFont("Consolas", 38, QFont.Weight.Bold))
        lbl_n.setStyleSheet(
            f"color:{TEXT_DIM}; background:transparent; border:none;")
        lbl_n.setAlignment(Qt.AlignmentFlag.AlignCenter)

        inner.addWidget(lbl_t)
        inner.addWidget(lbl_n)
        return frame, lbl_n

    def _vline(self):
        f = QFrame()
        f.setFrameShape(QFrame.Shape.VLine)
        f.setStyleSheet(f"color:{BORDER};")
        return f

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2: Session Snapshot Exporter
    # ─────────────────────────────────────────────────────────────────────────
    def export_session_snapshot(self):
        try:
            exp_path = "vpm_session_export.png"
            exporter = pg.exporters.ImageExporter(self.plot.plotItem)
            exporter.parameters()['width'] = 1200
            exporter.export(exp_path)
            QMessageBox.information(
                self, "Export Successful",
                f"Session progress graph saved to workspace directory as:\n'{exp_path}'"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not export graph frame:\n{e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Synchronized Lyrics Engine (LRC)
    # ─────────────────────────────────────────────────────────────────────────
    def show_lyrics_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("🎵 Sync Lyrics (LRC) Setup")
        dlg.setMinimumWidth(440)
        lay = QVBoxLayout(dlg)

        desc = QLabel("Get synced karaoke-style lyrics for your song.")
        desc.setStyleSheet(f"color:{TEXT_MID};")
        lay.addWidget(desc)

        btn_load = QPushButton("📂 Load local .lrc file")
        lay.addWidget(btn_load)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"color:{BORDER};")
        lay.addWidget(divider)

        lay.addWidget(QLabel("Search and Fetch online (LRCLIB API):"))
        search_box = QHBoxLayout()
        search_input = QLineEdit()
        search_input.setPlaceholderText("Artist - Title")
        search_box.addWidget(search_input)

        btn_search = QPushButton("🔍 Search")
        search_box.addWidget(btn_search)
        lay.addLayout(search_box)

        if hasattr(self, "_last_loaded_name") and self._last_loaded_name:
            clean_name = os.path.splitext(self._last_loaded_name)[0]
            clean_name = re.sub(r'\(.*?\)|\[.*?\]', '', clean_name).strip()
            search_input.setText(clean_name)

        def on_local_load():
            path, _ = QFileDialog.getOpenFileName(dlg, "Open LRC File", "", "Lyrics Files (*.lrc)")
            if path:
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    self.parse_lrc(text)
                    self.lyrics_curr_lbl.setText("Local LRC lyrics loaded!")
                    self._last_lyrics_idx = -2
                    dlg.accept()
                except Exception as ex:
                    QMessageBox.critical(dlg, "Error", f"Failed to load local file:\n{ex}")

        def on_online_search():
            q = search_input.text().strip()
            if not q:
                return

            self._lyrics_dlg = QProgressDialog("Connecting to LRCLIB database...", None, 0, 0, dlg)
            self._lyrics_dlg.setWindowTitle("Searching Synced Lyrics")
            self._lyrics_dlg.setWindowModality(Qt.WindowModality.WindowModal)
            self._lyrics_dlg.setCancelButton(None)
            self._lyrics_dlg.show()

            self._lyrics_worker = LyricsFetchWorker(q)
            self._lyrics_thread = QThread(self)
            self._lyrics_worker.moveToThread(self._lyrics_thread)
            self._lyrics_thread.started.connect(self._lyrics_worker.run)

            def on_fetched(lrc, info):
                self._lyrics_dlg.close()
                self.parse_lrc(lrc)
                self.lyrics_curr_lbl.setText(f"Loaded: {info}")
                self._last_lyrics_idx = -2
                dlg.accept()

            def on_fetch_error(err):
                self._lyrics_dlg.close()
                QMessageBox.warning(dlg, "Not Found", err)

            self._lyrics_worker.finished.connect(on_fetched)
            self._lyrics_worker.error.connect(on_fetch_error)

            self._lyrics_worker.finished.connect(self._lyrics_thread.quit)
            self._lyrics_worker.error.connect(self._lyrics_thread.quit)
            self._lyrics_thread.finished.connect(self._lyrics_thread.deleteLater)
            self._lyrics_thread.start()

        btn_load.clicked.connect(on_local_load)
        btn_search.clicked.connect(on_online_search)
        search_input.returnPressed.connect(on_online_search)

        dlg.exec()

    def parse_lrc(self, lrc_text: str):
        self.lyrics.clear()
        pattern = re.compile(r'\[(\d+):(\d+(?:\.\d+)?)]')
        for line in lrc_text.splitlines():
            line = line.strip()
            matches = pattern.findall(line)
            if matches:
                clean_text = pattern.sub('', line).strip()
                if clean_text.startswith('[') and clean_text.endswith(']'):
                    continue
                for m in matches:
                    minutes = int(m[0])
                    seconds = float(m[1])
                    total_seconds = minutes * 60 + seconds
                    self.lyrics.append((total_seconds, clean_text))
        self.lyrics.sort(key=lambda x: x[0])
        self._last_lyrics_idx = -2
        if self.full_lyrics_view:
            self.full_lyrics_view.close()
            self.full_lyrics_view = None

    def adjust_lyrics_offset(self, amount: float):
        self.lyrics_offset += amount
        self.offset_val_label.setText(f"{self.lyrics_offset:+.1f}s")
        self._last_lyrics_idx = -2

    def show_full_lyrics_viewer(self):
        if not self.lyrics:
            QMessageBox.information(self, "Lyrics", "Please load a song and lyrics (.lrc) first.")
            return

        if self.full_lyrics_view is None:
            self.full_lyrics_view = FullLyricsDialog(self.lyrics, self)
            self.full_lyrics_view.seek_requested.connect(self._seek_to_timestamp)

        self.full_lyrics_view.show()
        self.full_lyrics_view.raise_()
        self.full_lyrics_view.activateWindow()

        if self.audio_data is not None and self.sample_rate is not None:
            cur_t = self.current_frame / self.sample_rate
            self._update_full_lyrics_highlight(cur_t)

    def _seek_to_timestamp(self, timestamp: float):
        if self.audio_data is None or self.sample_rate is None:
            return

        trimmed_t = timestamp - self.trim_start_sec
        total_dur = len(self.audio_data) / self.sample_rate
        trimmed_t = max(0.0, min(trimmed_t, total_dur))

        was_playing = self.is_playing
        if self.is_playing:
            self.is_playing = False
            if self.stream:
                try:
                    self.stream.stop()
                except Exception:
                    pass

        self.current_frame = int(trimmed_t * self.sample_rate)
        self.song_pitch_queue.clear()
        self.song_buffer.clear()
        self.song_tracker = {"last": np.nan, "hold": 0}
        self.song_scroll_accum = 0.0
        self._last_lyrics_idx = -2
        self._create_output_stream()

        if was_playing:
            self.is_playing = True
            self.stream.start()

        self._refresh_play_btn()
        self.update_status()

    def update_lyrics_teleprompter(self, current_time: float):
        if not self.lyrics:
            return

        lookup_time = current_time + self.trim_start_sec + self.lyrics_offset

        active_idx = -1
        for i, (timestamp, _) in enumerate(self.lyrics):
            if lookup_time >= timestamp:
                active_idx = i
            else:
                break

        self._update_full_lyrics_highlight(lookup_time)

        if self._last_lyrics_idx == active_idx:
            return
        self._last_lyrics_idx = active_idx

        prev_text = self.lyrics[active_idx - 1][1] if active_idx > 0 else ""
        curr_text = self.lyrics[active_idx][1] if active_idx >= 0 else "..."
        next_text = self.lyrics[active_idx + 1][1] if active_idx < len(self.lyrics) - 1 else ""

        self.lyrics_prev_lbl.setText(prev_text)

        self.ly_fade_anim.stop()
        self.ly_opacity_effect.setOpacity(0.3)
        self.lyrics_curr_lbl.setText(curr_text)
        self.ly_fade_anim.setStartValue(0.3)
        self.ly_fade_anim.setEndValue(1.0)
        self.ly_fade_anim.start()

        self.lyrics_next_lbl.setText(next_text)

    def _update_full_lyrics_highlight(self, lookup_time: float):
        if not self.full_lyrics_view or not self.full_lyrics_view.isVisible():
            return

        active_idx = -1
        for i, (timestamp, _) in enumerate(self.lyrics):
            if lookup_time >= timestamp:
                active_idx = i
            else:
                break

        if active_idx >= 0:
            self.full_lyrics_view.highlight_line(active_idx)

    def load_local_lrc_matching(self, audio_path: str):
        self.lyrics.clear()
        base, _ = os.path.splitext(audio_path)
        lrc_path = base + ".lrc"
        if os.path.exists(lrc_path):
            try:
                with open(lrc_path, "r", encoding="utf-8", errors="ignore") as f:
                    self.parse_lrc(f.read())
                self.lyrics_curr_lbl.setText("Matching local LRC loaded")
            except Exception:
                pass
        else:
            self.lyrics_curr_lbl.setText("No synced lyrics (LRC) loaded")
            self.lyrics_prev_lbl.setText("")
            self.lyrics_next_lbl.setText("")

    # ─────────────────────────────────────────────────────────────────────────
    # GPU info dialog
    # ─────────────────────────────────────────────────────────────────────────
    def _show_gpu_info(self):
        lines = [
            f"<b>Current device:</b>  {gpu_info_str()}",
            "",
            "<b>Install PyTorch with ROCm (AMD RX 470 on Arch Linux):</b>",
            "<code>pip install torch torchvision torchaudio \\\n"
            "  --index-url https://download.pytorch.org/whl/rocm5.7</code>",
            "",
            "<b>Install Demucs:</b>",
            "<code>pip install demucs scipy</code>",
            "",
            "<b>Install Librosa (for Pitch Shifting):</b>",
            "<code>pip install librosa</code>",
            "",
            "<b>RX 470 note (gfx803 / Polaris):</b>",
            "HSA_OVERRIDE_GFX_VERSION is set automatically by this app.",
            f"Current value: <code>{os.environ.get('HSA_OVERRIDE_GFX_VERSION', '(not set)')}</code>",
            "",
            "<b>Verify ROCm torch:</b>",
            "<code>python -c \"import torch; print(torch.cuda.is_available())\"</code>",
        ]
        dlg = QDialog(self)
        dlg.setWindowTitle("⚡ GPU / Installation Info")
        dlg.resize(520, 340)
        lay = QVBoxLayout(dlg)
        lbl = QLabel("<br>".join(lines))
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color:{TEXT}; font-size:11px; line-height:1.6;")
        lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(lbl)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        dlg.exec()

    def _show_debug_window(self):
        if not hasattr(self, "debug_win"):
            self.debug_win = DebugWindow(self.stats, self)
        self.debug_win.show()
        self.debug_win.raise_()

    # ─────────────────────────────────────────────────────────────────────────
    # Stream helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _create_output_stream(self):
        with self._stream_lock:
            if self.stream is not None:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.hop_size,
                dtype="float32",
                callback=self.audio_callback,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1 OPT: QThread Async Loading Orchestrator
    # ─────────────────────────────────────────────────────────────────────────
    def load_track(self):
        if self.mic_active and not self.overlay_mic:
            self.stop_mic()
        if self.is_playing:
            self.pause()
        path, _ = QFileDialog.getOpenFileName(
            self, "Select audio file", "",
            "Audio  (*.mp3 *.wav *.flac *.ogg *.m4a)")
        if path:
            self._load_async_dispatch(path)

    def _load_async_dispatch(self, file_path: str):
        self._load_dlg = QProgressDialog("Loading track asynchronously...", None, 0, 0, self)
        self._load_dlg.setWindowTitle("Loading Audio")
        self._load_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._load_dlg.setCancelButton(None)
        self._load_dlg.show()

        self._load_worker = AudioLoadWorker(file_path)
        self._load_thread = QThread(self)
        self._load_worker.moveToThread(self._load_thread)

        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.finished.connect(self._on_load_async_done)
        self._load_worker.error.connect(self._on_load_async_error)

        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.error.connect(self._load_thread.quit)
        self._load_thread.finished.connect(self._load_thread.deleteLater)

        self._load_thread.start()

    def _on_load_async_done(self, full_data: np.ndarray, sr: int, file_path: str):
        if self._load_dlg:
            self._load_dlg.close()
            self._load_dlg = None

        self._last_loaded_name = os.path.basename(file_path)
        self.load_local_lrc_matching(file_path)

        dlg = TrimDialog(self, full_data, sr, display_name=self._last_loaded_name)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        s, e = dlg.get_trim()
        trimmed = full_data[int(s * sr): int(e * sr)].copy()

        self.trim_start_sec = s
        self.lyrics_offset = 0.0
        self.offset_val_label.setText("0.0s")

        self._apply_audio(trimmed, sr, is_new_track=True)

    def _on_load_async_error(self, err_msg: str):
        if self._load_dlg:
            self._load_dlg.close()
            self._load_dlg = None
        QMessageBox.critical(self, "Load Error", f"Async file loader failed:\n{err_msg}")

    def open_cache(self):
        if self.is_playing:
            self.pause()
        dlg = CacheDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.selected_path:
            self._load_async_dispatch(dlg.selected_path)

    def load_and_separate(self):
        if not DEMUCS_AVAILABLE:
            self._show_gpu_info()
            return
        if self.is_playing:
            self.pause()

        dlg = QDialog(self)
        dlg.setWindowTitle("🔀 Vocal Separation")
        dlg.setModal(True)
        dlg.resize(440, 200)
        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)

        lay.addWidget(QLabel(f"<b>GPU:</b>  {gpu_info_str()}"))

        combo = QComboBox()
        for label in DEMUCS_MODELS:
            combo.addItem(label)
        for i, v in enumerate(DEMUCS_MODELS.values()):
            if v == DEMUCS_DEFAULT:
                combo.setCurrentIndex(i)
                break

        note = QLabel(
            "Segment mode is auto-enabled for 4 GB VRAM  (RX 470 = OK).\n"
            "htdemucs_ft = best quality.  mdx_extra_q = lowest VRAM.")
        note.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px;")
        note.setWordWrap(True)

        lay.addWidget(QLabel("Model:"))
        lay.addWidget(combo)
        lay.addWidget(note)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        model_name = list(DEMUCS_MODELS.values())[combo.currentIndex()]
        path, _ = QFileDialog.getOpenFileName(
            self, "Select audio to separate", "",
            "Audio  (*.mp3 *.wav *.flac *.ogg)")
        if path:
            # Separation still starts from raw file loader
            try:
                if path.lower().endswith(".mp3"):
                    buf = BytesIO()
                    AudioSegment.from_mp3(path).export(buf, format="wav")
                    buf.seek(0)
                    full_data, sr = sf.read(buf, dtype="float32")
                else:
                    full_data, sr = sf.read(path, dtype="float32")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Cannot open file for Demucs:\n{e}")
                return

            if full_data.ndim > 1:
                full_data = np.mean(full_data, axis=1)

            self._last_loaded_name = os.path.basename(path)
            self.load_local_lrc_matching(path)

            trim_dlg = TrimDialog(self, full_data, sr, display_name=self._last_loaded_name)
            if trim_dlg.exec() != QDialog.DialogCode.Accepted:
                return

            s, e = trim_dlg.get_trim()
            trimmed = full_data[int(s * sr): int(e * sr)].copy()

            self.trim_start_sec = s
            self.lyrics_offset = 0.0
            self.offset_val_label.setText("0.0s")

            self._pending_cache_info = {
                "orig_name": os.path.basename(path),
                "model": model_name,
                "duration": e - s
            }
            tmp = tempfile.mktemp(suffix="_input.wav")
            sf.write(tmp, trimmed, sr)
            self._run_demucs(tmp, sr, trimmed, model_name)

    def _apply_audio(self, data: np.ndarray, sr: int, is_new_track: bool = True):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        if is_new_track:
            self.original_audio_data = data.copy()
            self.spin_pitch.blockSignals(True)
            self.spin_pitch.setValue(0)
            self.spin_pitch.blockSignals(False)
            self.current_frame = 0

        self.audio_data = data
        self.sample_rate = sr

        self.song_buffer.clear()
        self.song_pitch_queue.clear()
        self.song_tracker = {"last": np.nan, "hold": 0}

        self.create_song_detector(sr)
        self._create_output_stream()
        self.clear_graph_song_only()

        self.chk_overlay.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_restart.setEnabled(True)
        self.seek_slider.setEnabled(True)
        self.song_scroll_accum = 0.0

        self.update_status()
        self._refresh_play_btn()
        self._update_seek_display()

    def apply_pitch_shift(self):
        if self.original_audio_data is None:
            return

        semitones = self.spin_pitch.value()
        if semitones == 0:
            was_playing = self.is_playing
            if self.is_playing:
                self.pause()
            self._apply_audio(self.original_audio_data, self.sample_rate, is_new_track=False)
            if was_playing:
                self.play()
            return

        try:
            import librosa
        except ImportError:
            QMessageBox.warning(self, "Missing Dependency", "Pitch shifting requires 'librosa'.")
            return

        self._pitch_dlg = QProgressDialog("Shifting pitch (High Quality)...", None, 0, 0, self)
        self._pitch_dlg.setWindowTitle("Pitch Shift")
        self._pitch_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._pitch_dlg.setCancelButton(None)
        self._pitch_dlg.show()

        was_playing = self.is_playing
        if self.is_playing:
            self.pause()

        self._ps_worker = PitchShiftWorker(self.original_audio_data, self.sample_rate, semitones)
        self._ps_thread = QThread(self)
        self._ps_worker.moveToThread(self._ps_thread)
        self._ps_thread.started.connect(self._ps_worker.run)
        self._ps_worker.finished.connect(lambda shifted: self._on_pitch_shift_done(shifted, was_playing))
        self._ps_worker.error.connect(self._on_pitch_shift_error)

        self._ps_worker.finished.connect(self._ps_thread.quit)
        self._ps_worker.error.connect(self._ps_thread.quit)
        self._ps_thread.finished.connect(self._ps_thread.deleteLater)

        self._ps_thread.start()

    def _on_pitch_shift_done(self, shifted_data: np.ndarray, was_playing: bool):
        if self._pitch_dlg:
            self._pitch_dlg.close()
            self._pitch_dlg = None
        self._apply_audio(shifted_data, self.sample_rate, is_new_track=False)
        if was_playing:
            self.play()

    def _on_pitch_shift_error(self, err_msg: str):
        if self._pitch_dlg:
            self._pitch_dlg.close()
            self._pitch_dlg = None
        QMessageBox.critical(self, "Error", f"Pitch shift failed:\n{err_msg}")

    # ─────────────────────────────────────────────────────────────────────────
    # Demucs async
    # ─────────────────────────────────────────────────────────────────────────
    def _run_demucs(self, audio_path: str, sr: int,
                    fallback: np.ndarray, model: str):
        device = get_torch_device()

        self._demucs_dlg = QProgressDialog(
            "Initialising…", None, 0, 100, self)
        self._demucs_dlg.setWindowTitle("🔀 Demucs  — Vocal Separation")
        self._demucs_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._demucs_dlg.setMinimumDuration(0)
        self._demucs_dlg.setAutoClose(False)
        self._demucs_dlg.setCancelButton(None)
        self._demucs_dlg.setValue(0)
        self._demucs_dlg.resize(480, 120)
        self._demucs_dlg.show()

        worker = DemucsWorker(audio_path, device, model)
        thread = QThread(self)
        worker.moveToThread(thread)

        worker.progress.connect(
            lambda m: self._demucs_dlg.setLabelText(f"⚙  {m}"))
        worker.progress_pct.connect(self._demucs_dlg.setValue)
        worker.finished.connect(lambda p: self._demucs_done(p, sr))
        worker.error.connect(
            lambda e: self._demucs_error(e, fallback, sr))
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)

        self._demucs_thread = thread
        self._demucs_worker = worker
        thread.start()

    def _demucs_done(self, vocal_path: str, sr: int):
        if self._demucs_dlg:
            self._demucs_dlg.close()
            self._demucs_dlg = None

        if self._pending_cache_info:
            info = self._pending_cache_info
            CacheManager.add(info["orig_name"], info["model"], vocal_path, info["duration"])
            self._pending_cache_info = None

        try:
            vocal, out_sr = sf.read(vocal_path, dtype="float32")
            if vocal.ndim > 1:
                vocal = np.mean(vocal, axis=1)
        except Exception as exc:
            QMessageBox.critical(self, "Read error", str(exc))
            return
        finally:
            try:
                os.remove(vocal_path)
            except OSError:
                pass
        self._apply_audio(vocal, out_sr, is_new_track=True)
        QMessageBox.information(
            self, "✅ Done",
            f"Vocals separated & saved to cache!\nDevice: {get_torch_device().upper()}"
            f"  ({gpu_info_str()})")

    def _demucs_error(self, msg: str, fallback: np.ndarray, sr: int):
        if self._demucs_dlg:
            self._demucs_dlg.close()
            self._demucs_dlg = None
        QMessageBox.critical(
            self, "Demucs error",
            f"Separation failed:\n{msg}\n\nLoading original audio.")
        self._apply_audio(fallback, sr, is_new_track=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Audio callbacks
    # ─────────────────────────────────────────────────────────────────────────
    def audio_callback(self, outdata, frames, cb_time, status):
        t0 = time_module.perf_counter()
        if not self.is_playing or self.audio_data is None:
            outdata.fill(0)
            return

        data = self.audio_data
        s, e = self.current_frame, self.current_frame + frames
        if e >= len(data):
            chunk = data[s:].copy()
            chunk = np.pad(chunk, (0, frames - len(chunk)))
            self.is_playing = False
            self.current_frame = len(data)
        else:
            chunk = data[s:e].copy()
            self.current_frame = e

        outdata[:] = chunk[:, np.newaxis] * self.volume

        raw, rms, conf = self._process_pitch(chunk, self.song_detector, self.song_tracker)
        self.stats.rms_song = rms
        self.stats.conf_song = conf

        # Интеграция Rust: конвертируем deque в list, вызываем Rust, сохраняем буфер обратно
        buf_list = list(self.song_buffer)
        med, new_buf = smooth_pitch_rust(raw, buf_list)
        self.song_buffer = collections.deque(new_buf, maxlen=3)

        # В очередь отправляем строго одно число float, а не кортеж!
        self.song_pitch_queue.put(med)

        self.stats.queue_song = len(self.song_pitch_queue)
        self.stats.cb_time_song_ms = (time_module.perf_counter() - t0) * 1000

    def mic_callback(self, indata, frames, cb_time, status):
        if not self.mic_active:
            return
        t0 = time_module.perf_counter()
        chunk = indata[:, 0] if indata.ndim > 1 else indata.flatten()

        raw, rms, conf = self._process_pitch(chunk, self.user_detector, self.user_tracker)
        self.stats.rms_mic = rms
        self.stats.conf_mic = conf

        # Интеграция Rust для микрофона
        buf_list = list(self.user_buffer)
        med, new_buf = smooth_pitch_rust(raw, buf_list)
        self.user_buffer = collections.deque(new_buf, maxlen=3)

        # В очередь отправляем строго одно число float
        self.user_pitch_queue.put(med)

        self.stats.queue_mic = len(self.user_pitch_queue)
        self.stats.cb_time_mic_ms = (time_module.perf_counter() - t0) * 1000

    # ─────────────────────────────────────────────────────────────────────────
    # Playback control
    # ─────────────────────────────────────────────────────────────────────────
    def toggle_play(self):
        if self.audio_data is None:
            return
        self.pause() if self.is_playing else self.play()

    def play(self):
        if self.audio_data is None or self.stream is None:
            return
        if self.current_frame >= len(self.audio_data):
            self.current_frame = 0
        self.is_playing = True
        try:
            self.stream.start()
        except Exception:
            self._create_output_stream()
            self.stream.start()
        if self.chk_overlay.isChecked() and not self.mic_active:
            self._start_overlay_mic_internal()
        self._refresh_play_btn()
        self.update_status()

    def pause(self):
        self.is_playing = False
        if self.stream:
            try:
                self.stream.stop()
            except Exception:
                pass
        self.song_pitch_queue.clear()
        self.song_scroll_accum = 0.0
        self._refresh_play_btn()
        self.update_status()

    def do_restart(self):
        was_playing = self.is_playing
        if self.is_playing:
            self.pause()
        self.current_frame = 0
        self.song_pitch_queue.clear()
        self.song_buffer.clear()
        self.song_tracker = {"last": np.nan, "hold": 0}
        self.song_scroll_accum = 0.0
        self._last_lyrics_idx = -2
        self.clear_graph_song_only()
        self._create_output_stream()
        if was_playing:
            self.play()

    # ─────────────────────────────────────────────────────────────────────────
    # Seek
    # ─────────────────────────────────────────────────────────────────────────
    def _on_seek_pressed(self):
        self.slider_dragging = True

    def _on_seek_moved(self, value: int):
        if self._seek_guard or self.audio_data is None:
            return
        sec = value / 10000.0 * (len(self.audio_data) / self.sample_rate)
        tot = len(self.audio_data) / self.sample_rate
        self.time_label.setText(f"{_fmt(sec)} / {_fmt(tot)}")

    def _on_seek_released(self):
        if self.audio_data is None:
            self.slider_dragging = False
            return
        was_playing = self.is_playing
        if self.is_playing:
            self.is_playing = False
            if self.stream:
                try:
                    self.stream.stop()
                except Exception:
                    pass

        value = self.seek_slider.value()
        nf = int(value / 10000.0 * len(self.audio_data))
        self.current_frame = max(0, min(nf, len(self.audio_data) - 1))

        self.song_pitch_queue.clear()
        self.song_buffer.clear()
        self.song_tracker = {"last": np.nan, "hold": 0}
        self.song_scroll_accum = 0.0
        self._last_lyrics_idx = -2
        self._create_output_stream()
        self.slider_dragging = False

        if was_playing:
            self.is_playing = True
            self.stream.start()

        self._refresh_play_btn()
        self.update_status()

    # ─────────────────────────────────────────────────────────────────────────
    # Microphone
    # ─────────────────────────────────────────────────────────────────────────
    def toggle_mic_only(self):
        if self.mic_active:
            self.stop_mic()
        else:
            self.start_mic_only()

    def start_mic_only(self):
        if self.is_playing:
            self.pause()
        self.chk_overlay.blockSignals(True)
        self.chk_overlay.setChecked(False)
        self.chk_overlay.blockSignals(False)

        self.mic_active = True
        self.overlay_mic = False
        self._style_mic_btn(True)

        self.plot_song.fill(np.nan)
        self.song_pitch_queue.clear()
        self.song_tracker = {"last": np.nan, "hold": 0}
        self.user_buffer.clear()
        self.user_pitch_queue.clear()
        self.user_tracker = {"last": np.nan, "hold": 0}
        self.accuracy_history.clear()
        self.song_scroll_accum = 0.0
        self.user_scroll_accum = 0.0

        # Reset session range tracking
        self.tess_min_midi = 999.0
        self.tess_max_midi = 0.0
        self.tess_label.setText("Vocal range: Min -- | Max --")

        self.create_user_detector()
        try:
            self.mic_stream = sd.InputStream(
                samplerate=44100, channels=1,
                blocksize=self.hop_size, dtype="float32",
                callback=self.mic_callback)
            self.mic_stream.start()
        except Exception as exc:
            QMessageBox.critical(self, "Microphone error", str(exc))
            self.mic_active = False
            self._style_mic_btn(False)
        self.update_status()

    def stop_mic(self):
        self.mic_active = False
        self.overlay_mic = False
        self._style_mic_btn(False)
        if self.mic_stream:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
            except Exception:
                pass
            self.mic_stream = None
        self.update_status()

    def _on_overlay_changed(self, state: int):
        checked = (state == Qt.CheckState.Checked.value)
        was_overlay = self.overlay_mic
        self.overlay_mic = checked
        if checked:
            if not self.mic_active:
                self._start_overlay_mic_internal()
        else:
            if self.mic_active and was_overlay:
                self.mic_active = False
                if self.mic_stream:
                    try:
                        self.mic_stream.stop()
                        self.mic_stream.close()
                    except Exception:
                        pass
                    self.mic_stream = None
                self.user_pitch_queue.clear()
                self.user_buffer.clear()
                self.user_tracker = {"last": np.nan, "hold": 0}
        self.update_status()

    def _start_overlay_mic_internal(self):
        if self.mic_stream is not None:
            return
        self.overlay_mic = True
        self.create_user_detector()
        self.user_pitch_queue.clear()
        self.user_tracker = {"last": np.nan, "hold": 0}
        self.user_buffer.clear()
        try:
            self.mic_stream = sd.InputStream(
                samplerate=44100, channels=1,
                blocksize=self.hop_size, dtype="float32",
                callback=self.mic_callback)
            self.mic_stream.start()
            self.mic_active = True
        except Exception as exc:
            QMessageBox.critical(self, "Microphone error", str(exc))
            self.overlay_mic = False

    def _style_mic_btn(self, active: bool):
        self.btn_mic.setProperty("active", "true" if active else "false")
        self.btn_mic.setText("⏹ Stop mic" if active else "🎤 Microphone")
        self.btn_mic.style().unpolish(self.btn_mic)
        self.btn_mic.style().polish(self.btn_mic)

    # ─────────────────────────────────────────────────────────────────────────
    # Monitor (passthrough)
    # ─────────────────────────────────────────────────────────────────────────
    def _on_monitor_vol_changed(self, val: int):
        self.monitor_volume = val / 100.0
        self.stats.monitor_vol = self.monitor_volume

    def toggle_monitor(self):
        if self.monitor_active:
            self.stop_monitor()
        else:
            self.start_monitor()

    def start_monitor(self):
        if self.monitor_stream is not None:
            return

        def _passthrough(indata, outdata, frames, cb_time, status):
            t0 = time_module.perf_counter()
            outdata[:] = indata * self.monitor_volume
            self.stats.cb_time_monitor_ms = (time_module.perf_counter() - t0) * 1000

        try:
            self.monitor_stream = sd.Stream(
                samplerate=44100,
                channels=1,
                blocksize=256,
                dtype="float32",
                callback=_passthrough,
            )
            self.monitor_stream.start()
            self.monitor_active = True
            self._style_monitor_btn(True)
        except Exception as exc:
            QMessageBox.critical(self, "Monitor error",
                                 f"Could not open audio device:\n{exc}")
            self.monitor_active = False
            if self.monitor_stream:
                try:
                    self.monitor_stream.close()
                except Exception:
                    pass
                self.monitor_stream = None

    def stop_monitor(self):
        self.monitor_active = False
        if self.monitor_stream:
            try:
                self.monitor_stream.stop()
                self.monitor_stream.close()
            except Exception:
                pass
            self.monitor_stream = None
        self._style_monitor_btn(False)

    def _style_monitor_btn(self, active: bool):
        self.btn_monitor.setProperty("active", "true" if active else "false")
        self.btn_monitor.setText(
            "⏹ Stop Mon" if active else "🎧 Monitor")
        self.btn_monitor.style().unpolish(self.btn_monitor)
        self.btn_monitor.style().polish(self.btn_monitor)

    # ─────────────────────────────────────────────────────────────────────────
    # Pitch processing with Stage 1 clamping
    # ─────────────────────────────────────────────────────────────────────────
    def _process_pitch(self, chunk: np.ndarray, detector, tracker) -> tuple[float, float, float]:
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        raw = np.nan
        conf = 0.0
        if rms >= 0.01:
            p = float(detector(chunk.astype(np.float32))[0])
            conf = float(detector.get_confidence())

            # STAGE 1 OPT: Clamp pitch bounds instantly to filter noise (human voice limits: ~50Hz-1000Hz / MIDI 30-88)
            if conf > self.conf_thresh and 30.0 <= p <= 88.0:
                raw = p
        if not np.isnan(raw):
            if not np.isnan(tracker["last"]):
                diff = raw - tracker["last"]
                if 10.5 < diff < 13.5:
                    raw -= 12
                elif -13.5 < diff < -10.5:
                    raw += 12
            tracker["last"] = raw
            tracker["hold"] = 4
            return raw, rms, conf
        else:
            if tracker["hold"] > 0:
                tracker["hold"] -= 1
                return tracker["last"], rms, conf
            tracker["last"] = np.nan
            return np.nan, rms, conf

    @staticmethod
    def _smooth_pitch(new_pitch: float, buf: collections.deque) -> float:
        if not np.isnan(new_pitch):
            if len(buf) > 0:
                med = np.median(list(buf))
                if not np.isnan(med) and abs(new_pitch - med) > 1.5:
                    buf.clear()
            buf.append(new_pitch)
            return float(np.median(list(buf)))
        else:
            buf.clear()
            return np.nan

    @staticmethod
    def midi_to_note(midi: float) -> str:
        if np.isnan(midi) or midi < 10:
            return "--"
        mi = int(round(midi))
        ns = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return f"{ns[mi % 12]}{mi // 12 - 1}"

    # ─────────────────────────────────────────────────────────────────────────
    # Main render loop
    # ─────────────────────────────────────────────────────────────────────────
    def update_plot(self):
        if self._rendering:
            self.stats.frames_skipped += 1
            return
        self._rendering = True

        try:
            if (self.audio_data is not None
                    and self.current_frame >= len(self.audio_data)
                    and self.is_playing):
                self.is_playing = False
                if self.stream:
                    try:
                        self.stream.stop()
                    except Exception:
                        pass
                self.song_pitch_queue.clear()
                self.song_scroll_accum = 0.0
                self._refresh_play_btn()
                self.update_status()

            now = time_module.perf_counter()
            dt = min(now - self.last_update_time, 0.2)
            self.last_update_time = now
            need = False

            # ── song pitch scroll ────────────────────
            if self.is_playing and self.audio_data is not None:
                song_rate = self.sample_rate / self.hop_size
                self.song_scroll_accum += dt * song_rate
                steps = int(self.song_scroll_accum)
                self.song_scroll_accum -= steps

                if steps > 0:
                    items = []
                    for _ in range(steps):
                        v = self.song_pitch_queue.get_nowait()
                        items.append(v if v is not None else np.nan)

                    k = len(items)
                    if k < self.history_size:
                        self.plot_song[:-k] = self.plot_song[k:]
                        self.plot_song[-k:] = items
                    else:
                        self.plot_song[:] = items[-self.history_size:]
                    need = True
            elif not self.is_playing:
                while self.song_pitch_queue.get_nowait() is not None:
                    pass
                self.song_scroll_accum = 0.0

            # ── user pitch scroll ────────────────────
            if self.mic_active:
                user_rate = 44100.0 / self.hop_size
                self.user_scroll_accum += dt * user_rate
                steps = int(self.user_scroll_accum)
                self.user_scroll_accum -= steps

                if steps > 0:
                    items = []
                    for _ in range(steps):
                        v = self.user_pitch_queue.get_nowait()
                        items.append(v if v is not None else np.nan)

                    k = len(items)
                    if k < self.history_size:
                        self.plot_user[:-k] = self.plot_user[k:]
                        self.plot_user[-k:] = items
                    else:
                        self.plot_user[:] = items[-self.history_size:]
                    need = True

            if need:
                self._update_curves()

            # ── camera ────────────────────────────────────────────────────────
            active = np.nan
            up = self.plot_user[-1]
            sp = self.plot_song[-1]
            if self.mic_active and not np.isnan(up):
                active = up
            elif self.is_playing and not np.isnan(sp):
                active = sp

            if QApplication.mouseButtons() == Qt.MouseButton.LeftButton:
                self.auto_scroll_paused_until = time_module.perf_counter() + 3.0

            if not np.isnan(active) and time_module.perf_counter() > self.auto_scroll_paused_until:
                dz, lr = 4.0, 0.035
                if active > self.camera_y + dz:
                    self.camera_y += ((active - dz) - self.camera_y) * lr
                elif active < self.camera_y - dz:
                    self.camera_y += ((active + dz) - self.camera_y) * lr
                self.camera_y = max(33.0, min(87.0, self.camera_y))
                self.plot.setYRange(self.camera_y - 13,
                                    self.camera_y + 13, padding=0)
            else:
                lo, hi = self.plot.viewRange()[1]
                self.camera_y = (lo + hi) / 2.0

            # ── STAGE 2: Real-Time Tessitura Diagnostic ───────────────────────
            if self.mic_active and not np.isnan(up) and self.stats.conf_mic > 0.75:
                if up < self.tess_min_midi:
                    self.tess_min_midi = up
                if up > self.tess_max_midi:
                    self.tess_max_midi = up
                self.tess_label.setText(
                    f"Vocal range: Min {self.midi_to_note(self.tess_min_midi)} | "
                    f"Max {self.midi_to_note(self.tess_max_midi)}"
                )

            # ── note cards ────────────────────────────────────────────────────
            self._set_note(self.song_note_label, sp, C_SONG,
                           enabled=self.is_playing)
            self._set_note(self.user_note_label, up, C_USER,
                           enabled=self.mic_active)

            self._update_seek_display()
            self._update_accuracy()

            # ── lyrics update ─────────────────────────────────────────────────
            if self.audio_data is not None and self.sample_rate is not None:
                cur_t = self.current_frame / self.sample_rate
                self.update_lyrics_teleprompter(cur_t)

            self.stats.frames_rendered += 1

        finally:
            self._rendering = False

    def _set_note(self, label: QLabel, midi_val: float,
                  color: str, enabled: bool):
        if not enabled:
            txt = "--"
            c = TEXT_DIM
            style = f"color:{c}; background:transparent; border:none;"
        else:
            txt = self.midi_to_note(midi_val)
            c = color if txt != "--" else TEXT_DIM
            style = f"color:{c}; background:transparent; border:none;"

        if getattr(label, "_last_txt", None) == txt and getattr(label, "_last_style", None) == style:
            return

        label.setText(txt)
        label.setStyleSheet(style)
        label._last_txt = txt
        label._last_style = style

    def _update_curves(self):
        cs = self._conn(self.plot_song)
        self.curve_song_glow.setData(self.plot_song, connect=cs)
        self.curve_song.setData(self.plot_song, connect=cs)
        cu = self._conn(self.plot_user)
        self.curve_user_glow.setData(self.plot_user, connect=cu)
        self.curve_user.setData(self.plot_user, connect=cu)

    @staticmethod
    def _conn(data: np.ndarray) -> np.ndarray:
        c = np.ones(len(data), dtype=int)
        with np.errstate(invalid="ignore"):
            c[:-1] = (np.abs(np.diff(data)) <= 5.0).astype(int)
        c[np.isnan(data)] = 0
        return c

    def _update_seek_display(self):
        if (self.audio_data is None or self.sample_rate is None
                or self.slider_dragging):
            return
        total = len(self.audio_data)
        if total == 0:
            return
        pos = int(self.current_frame / total * 10000)
        self._seek_guard = True
        self.seek_slider.setValue(pos)
        self._seek_guard = False
        self.time_label.setText(
            f"{_fmt(self.current_frame / self.sample_rate)} / "
            f"{_fmt(total / self.sample_rate)}")

    def _update_accuracy(self):
        sp = self.plot_song[-1]
        up = self.plot_user[-1]

        if not self.mic_active or not self.is_playing:
            if not self.mic_active:
                if getattr(self.cents_label, "_last_txt", None) != "":
                    self.cents_label.setText("")
                    self.accuracy_label.setText("")
                    self.cents_label._last_txt = ""
            return

        if np.isnan(sp) or np.isnan(up):
            return

        diff_c = (up - sp) * 100.0
        abs_c = abs(diff_c)
        arrow = "▲" if diff_c > 0 else "▼"

        if abs_c <= 20:
            color, icon = C_GREEN, "🎯"
        elif abs_c <= 50:
            color, icon = "#8bc34a", "✓"
        elif abs_c <= 100:
            color, icon = C_YELLOW, "⚠"
        else:
            color, icon = C_RED, "✗"

        text_cents = f"{icon} {arrow}{abs_c:.0f}¢" if abs_c > 2 else "🎯 ±0¢"
        style_cents = f"color:{color}; font-size:17px; font-weight:700;"

        if getattr(self.cents_label, "_last_txt", None) != text_cents:
            self.cents_label.setText(text_cents)
            self.cents_label.setStyleSheet(style_cents)
            self.cents_label._last_txt = text_cents

        self.accuracy_history.append(abs_c <= 50)
        if self.accuracy_history:
            pct = sum(self.accuracy_history) / len(self.accuracy_history) * 100
            acc_c = C_GREEN if pct >= 80 else (C_YELLOW if pct >= 50 else C_RED)

            text_acc = f"Session accuracy: {pct:.0f}%"
            style_acc = f"color:{acc_c}; font-size:11px; font-weight:600;"

            if getattr(self.accuracy_label, "_last_txt", None) != text_acc:
                self.accuracy_label.setText(text_acc)
                self.accuracy_label.setStyleSheet(style_acc)
                self.accuracy_label._last_txt = text_acc

    def update_status(self):
        if self.mic_active:
            if self.overlay_mic and self.audio_data is not None:
                txt = ("🟢 Track + Mic  ·  singing along"
                       if self.is_playing else
                       "🟡 Mic listening  ·  track paused")
            else:
                txt = "🟢 Microphone — listening…"
        elif self.is_playing:
            txt = "▶  Playing"
        elif self.audio_data is not None:
            txt = "⏸  Track loaded  ·  press Play  [Space]"
        else:
            txt = "Load a track or enable microphone"

        if getattr(self.status_label, "_last_txt", None) != txt:
            self.status_label.setText(txt)
            col = TEXT if (self.is_playing or self.mic_active) else TEXT_DIM
            self.status_label.setStyleSheet(f"color:{col};")
            self.status_label._last_txt = txt

    # ─────────────────────────────────────────────────────────────────────────
    # Graph helpers
    # ─────────────────────────────────────────────────────────────────────────
    def clear_graph(self):
        self.plot_song.fill(np.nan)
        self.plot_user.fill(np.nan)
        self.accuracy_history.clear()
        self._update_curves()
        self.cents_label.setText("")
        self.accuracy_label.setText("")
        self.cents_label._last_txt = ""

    def clear_graph_song_only(self):
        self.plot_song.fill(np.nan)
        self._update_curves()

    def _refresh_play_btn(self):
        self.btn_play.setText("⏸  Pause" if self.is_playing else "▶  Play")
        self.btn_play.style().unpolish(self.btn_play)
        self.btn_play.style().polish(self.btn_play)

    # ─────────────────────────────────────────────────────────────────────────
    # Zoom
    # ─────────────────────────────────────────────────────────────────────────
    def zoom_x_in(self):
        self._resize_history(max(200, int(self.history_size * 0.8)))

    def zoom_x_out(self):
        self._resize_history(min(2400, int(self.history_size * 1.25)))

    def zoom_y_in(self):
        lo, hi = self.plot.viewRange()[1]
        new_span = max(15.0, (hi - lo) * 0.8)
        self.plot.setYRange(self.camera_y - new_span / 2,
                            self.camera_y + new_span / 2, padding=0)

    def zoom_y_out(self):
        lo, hi = self.plot.viewRange()[1]
        new_span = min(70.0, (hi - lo) * 1.25)
        self.plot.setYRange(self.camera_y - new_span / 2,
                            self.camera_y + new_span / 2, padding=0)

    def _resize_history(self, new_size: int):
        if new_size == self.history_size:
            return
        old_s, old_u = self.plot_song.copy(), self.plot_user.copy()
        self.history_size = new_size
        self.plot_song = np.full(new_size, np.nan)
        self.plot_user = np.full(new_size, np.nan)
        n = min(len(old_s), new_size)
        self.plot_song[-n:] = old_s[-n:]
        self.plot_user[-n:] = old_u[-n:]
        self._update_curves()

    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        self.timer.stop()
        self.is_playing = False
        self.mic_active = False
        self.monitor_active = False
        for s in (self.stream, self.mic_stream, self.monitor_stream):
            if s is not None:
                try:
                    s.stop()
                    s.close()
                except Exception:
                    pass
        if self._demucs_thread and self._demucs_thread.isRunning():
            self._demucs_thread.quit()
            self._demucs_thread.wait(3000)

        if hasattr(self, "_ps_thread") and self._ps_thread and self._ps_thread.isRunning():
            self._ps_thread.quit()
            self._ps_thread.wait(3000)

        if hasattr(self, "_lyrics_thread") and self._lyrics_thread and self._lyrics_thread.isRunning():
            self._lyrics_thread.quit()
            self._lyrics_thread.wait(2000)

        if hasattr(self, "_load_thread") and self._load_thread and self._load_thread.isRunning():
            self._load_thread.quit()
            self._load_thread.wait(2000)

        if self.full_lyrics_view:
            self.full_lyrics_view.close()

        if hasattr(self, "debug_win"):
            self.debug_win.close()

        event.accept()


if __name__ == "__main__":
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    app = QApplication(sys.argv)
    app.setStyleSheet(SS)

    win = VocalPitchMonitor()
    win.show()
    sys.exit(app.exec())
