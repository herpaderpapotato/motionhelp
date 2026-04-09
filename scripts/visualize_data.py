"""Data visualizer for VideoToMotion processed scenes with curation workflow.

Shows video frames with skeleton overlay (using config keypoint bones), a label
timeline and optical flow plot, plus dataset curation controls: approve/reject,
force-val, stage-2 keypoint review.  Scene list is filterable by stage and
sortable by name or stage.

Usage:
    python tools/visualize_data.py
    python tools/visualize_data.py --data_dir path/to/data
    python tools/visualize_data.py --config configs/default.yaml
"""

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageTk

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.curation import (
    approve, approve_stage2, discover_scenes, get_scene_stage,
    read_review, reject, reject_stage2, resolve_flow_path,
    resolve_keypoints_path, set_force_val,
)

# ─── Stage colours + icons ────────────────────────────────────────────────────

STAGE_COLORS: dict[str, str] = {
    "legacy":    "#6a9955",
    "pending":   "#d79921",
    "approved":  "#458588",
    "keypoints": "#689d6a",
    "stage2_ok": "#b8bb26",
    "flow":      "#98971a",
    "rejected":  "#cc241d",
}
STAGE_ICONS: dict[str, str] = {
    "legacy":    "·",
    "pending":   "?",
    "approved":  "✓",
    "keypoints": "K",
    "stage2_ok": "★",
    "flow":      "K+",
    "rejected":  "✗",
}

# Two colours: person 0 = cyan, person 1 = orange
PERSON_COLORS = [
    (0, 200, 255),   # cyan (BGR)
    (0, 140, 255),   # orange-ish (BGR)
    (80, 255, 80),   # green (BGR) — for additional partners
    (255, 100, 255), # magenta (BGR)
    (100, 200, 200), # teal (BGR)
    (200, 150, 50),  # blue-ish (BGR) — beholder slot
]

CONF_THRESHOLD = 0.25

# Flow feature layout (matches _summarize_flow in src/data/flow.py)
FLOW_GRID_SIDE = 4
FLOW_N_ACTIVE_CELLS = 4
FLOW_GLOBAL_OFFSET = 16


# ─── Utilities ────────────────────────────────────────────────────────────────

def _decode_flow_features(features: np.ndarray) -> dict:
    """Decode a 64-dim flow feature vector into structured spatial and global stats.

    Returns a dict with:
      cells: list of 4 dicts, each {mean_mag, std_mag, mean_fx, mean_fy, row, col}
      global_mean_mag, global_std_mag, global_fx, global_fy, global_p90, global_angle
    """
    cells = []
    for i in range(FLOW_N_ACTIVE_CELLS):
        base = i * 4
        cells.append({
            "mean_mag": float(features[base]),
            "std_mag":  float(features[base + 1]),
            "mean_fx":  float(features[base + 2]),
            "mean_fy":  float(features[base + 3]),
            "row": 0,
            "col": i,
        })
    g = FLOW_GLOBAL_OFFSET
    return {
        "cells": cells,
        "global_mean_mag":   float(features[g]),
        "global_std_mag":    float(features[g + 1]),
        "global_fx":         float(features[g + 2]),
        "global_fy":         float(features[g + 3]),
        "global_p90":        float(features[g + 4]),
        "global_angle":      float(features[g + 5]),
    }


def _magnitude_to_bgr(mag: float, max_mag: float = 30.0) -> tuple[int, int, int]:
    """Map a scalar magnitude to a BGR colour (blue=low → red=high)."""
    t = min(1.0, abs(mag) / max(max_mag, 1e-6))
    # Blue → cyan → green → yellow → red
    r = int(255 * min(1.0, 2 * t))
    g = int(255 * min(1.0, 2 * t, 2 * (1 - t)))
    b = int(255 * max(0.0, 1.0 - 2 * t))
    return (b, g, r)


def draw_flow_overlay(
    frame: np.ndarray,
    features: np.ndarray,
    alpha: float = 0.38,
    arrow_scale: float = 4.0,
) -> np.ndarray:
    """Draw spatial flow cells and global arrow onto *frame* (BGR, copy).

    Each of the 4 active cells (top quarter of frame, 4 horizontal columns) is
    drawn as a semi-transparent coloured tile. An arrow shows the mean flow
    direction for that cell. A larger global arrow is shown in the top-left.
    """
    frame = frame.copy()
    h, w = frame.shape[:2]
    info = _decode_flow_features(features)

    # Reference magnitudes for colour scaling (use p90 so outliers don't wash out)
    ref_mag = max(info["global_p90"], 1.0)

    cell_h = h // FLOW_GRID_SIDE
    cell_w = w // FLOW_GRID_SIDE

    overlay = frame.copy()
    for cell in info["cells"]:
        r, c = cell["row"], cell["col"]
        y1, y2 = r * cell_h, min((r + 1) * cell_h, h)
        x1, x2 = c * cell_w, min((c + 1) * cell_w, w)

        color = _magnitude_to_bgr(cell["mean_mag"], ref_mag)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # Arrow from cell centre in flow direction
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        ex = int(cx + cell["mean_fx"] * arrow_scale)
        ey = int(cy + cell["mean_fy"] * arrow_scale)
        ex = max(0, min(w - 1, ex))
        ey = max(0, min(h - 1, ey))
        cv2.arrowedLine(overlay, (cx, cy), (ex, ey),
                        (255, 255, 255), 2, tipLength=0.35, line_type=cv2.LINE_AA)

    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Global direction arrow in top-left corner
    gfx, gfy = info["global_fx"], info["global_fy"]
    gmag = info["global_mean_mag"]
    arrow_len = min(h, w) // 8
    cx0, cy0 = arrow_len + 4, arrow_len + 4
    scale = arrow_len / max(ref_mag, 1e-6)
    ex0 = int(cx0 + gfx * scale)
    ey0 = int(cy0 + gfy * scale)
    gcolor = _magnitude_to_bgr(gmag, ref_mag)
    cv2.circle(frame, (cx0, cy0), arrow_len + 2, (30, 30, 30), -1)
    cv2.arrowedLine(frame, (cx0, cy0), (ex0, ey0),
                    gcolor, 3, tipLength=0.3, line_type=cv2.LINE_AA)
    cv2.circle(frame, (cx0, cy0), 3, (255, 255, 255), -1)

    # Label
    cv2.putText(frame, f"flow: mag={gmag:.1f} p90={info['global_p90']:.1f}",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
                cv2.LINE_AA)
    return frame


def draw_position_overlay(frame: np.ndarray, position: float) -> np.ndarray:
    """Draw a vertical position meter on the right edge of *frame* (BGR, copy).

    The meter fills from the bottom (0.0) to the top (1.0) matching the
    funscript convention, with a white tick marker and numeric readout.
    """
    frame = frame.copy()
    h, w = frame.shape[:2]
    position = float(np.clip(position, 0.0, 1.0))

    meter_w = max(14, w // 45)
    pad = max(6, w // 100)
    x1 = w - meter_w - pad
    x2 = w - pad
    y1 = pad
    y2 = h - pad
    track_h = max(1, y2 - y1)

    # Semi-transparent dark strip behind the meter + label area
    label_clearance = max(42, w // 14)
    strip_x1 = max(0, x1 - label_clearance)
    roi = frame[y1:y2, strip_x1:w].copy()
    frame[y1:y2, strip_x1:w] = cv2.addWeighted(
        roi, 0.38, np.zeros_like(roi), 0.62, 0
    )

    # Track background
    cv2.rectangle(frame, (x1, y1), (x2, y2), (55, 55, 55), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (110, 110, 110), 1)

    # Filled bar from bottom up to current position (blue→amber gradient)
    fill_h = int(position * track_h)
    if fill_h > 0:
        fy1 = y2 - fill_h
        r = int(80 + 170 * position)
        g = int(130 - 30 * position)
        b = int(240 - 160 * position)
        cv2.rectangle(frame, (x1, fy1), (x2, y2), (b, g, r), -1)

    # White tick marker at current position
    ind_y = max(y1, min(y2, y2 - int(position * track_h)))
    cv2.rectangle(frame, (x1 - 3, ind_y - 2), (x2 + 3, ind_y + 2),
                  (255, 255, 255), -1)

    # Numeric readout to the left of the tick
    label = f"{position:.2f}"
    fs = max(0.38, w / 1600.0)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    tx = x1 - tw - 6
    ty = max(y1 + th, min(y2 - 2, ind_y + th // 2))
    cv2.putText(frame, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)

    # Scale labels at top, middle, bottom of track
    lfs = max(0.28, w / 2200.0)
    for val in (1.0, 0.5, 0.0):
        vy = y2 - int(val * track_h)
        vy = max(y1 + 8, min(y2, vy + 4))
        cv2.putText(frame, f"{val:.1f}", (x1 + 2, vy),
                    cv2.FONT_HERSHEY_SIMPLEX, lfs, (130, 130, 130), 1, cv2.LINE_AA)

    return frame


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    bones: list[tuple[int, int]],
    conf_thresh: float = CONF_THRESHOLD,
    vis: np.ndarray | None = None,
) -> np.ndarray:
    """Draw skeleton for all persons onto *frame* (BGR, in-place).

    keypoints: [max_persons, n_kpts, 3]  (x_norm, y_norm, conf)
    bones: list of (a, b) index pairs from config
    vis: optional [max_persons, n_kpts] int array — 0=hidden, 1=visible, 2=occluded.
         When provided, overrides conf_thresh logic and drives per-keypoint colour.
    """
    OCCLUDED_COLOR = (255, 180, 100)  # light blue (BGR) for vis=2

    h, w = frame.shape[:2]
    for p_idx, person_kp in enumerate(keypoints):
        color = PERSON_COLORS[p_idx % len(PERSON_COLORS)]
        n_kpts = len(person_kp)

        def _kpt_visible(idx: int) -> bool:
            if vis is not None:
                return int(vis[p_idx, idx]) > 0
            return float(person_kp[idx][2]) >= conf_thresh

        def _kpt_color(idx: int) -> tuple[int, int, int]:
            if vis is not None and int(vis[p_idx, idx]) == 2:
                return OCCLUDED_COLOR
            return color

        # Draw limbs
        for a, b in bones:
            if a >= n_kpts or b >= n_kpts:
                continue
            if not _kpt_visible(a) or not _kpt_visible(b):
                continue
            xa, ya = float(person_kp[a][0]), float(person_kp[a][1])
            xb, yb = float(person_kp[b][0]), float(person_kp[b][1])
            # Use lighter of the two endpoint colors for the limb
            lcolor = _kpt_color(a) if _kpt_color(a) == _kpt_color(b) else color
            pt1 = (int(xa * w), int(ya * h))
            pt2 = (int(xb * w), int(yb * h))
            cv2.line(frame, pt1, pt2, lcolor, 2, cv2.LINE_AA)

        # Draw keypoints
        for ki in range(n_kpts):
            if not _kpt_visible(ki):
                continue
            x, y = float(person_kp[ki][0]), float(person_kp[ki][1])
            pt = (int(x * w), int(y * h))
            kc = _kpt_color(ki)
            cv2.circle(frame, pt, 4, kc, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 4, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def fit_image(frame_bgr: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Scale frame down to fit within max_w × max_h, keeping aspect ratio."""
    h, w = frame_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        return cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return frame_bgr


# ─── Main Application ─────────────────────────────────────────────────────────

class DataVisualizer:
    def __init__(self, root: tk.Tk, data_dir: Path, cfg: Config):
        self.root = root
        self.data_dir = data_dir
        self.processed_dir = data_dir / "processed"
        self.preprocessed_dir = data_dir / "preprocessed"
        self.cfg = cfg

        # Skeleton definition from config
        self.bones: list[tuple[int, int]] = [tuple(b) for b in cfg.pose.keypoint_bones]  # type: ignore[misc]

        # All discovered scenes: list of (scene_id, review_state)
        self._all_scenes: list[tuple[str, dict]] = []
        # Filtered/sorted view
        self.scenes: list[str] = []

        self.current_scene: str | None = None
        self.current_frame: int = 0
        self.total_frames: int = 0
        self.fps: float = 10.0
        self.current_review: dict = {}

        # Loaded data
        self.cap: cv2.VideoCapture | None = None
        self.pose_data: np.ndarray | None = None   # [N, max_persons, n_kpt, 3]
        self.label_data: np.ndarray | None = None  # [N]
        self.flow_data: np.ndarray | None = None   # [N, flow_dim] or None
        self.metadata: dict = {}

        # Playback
        self._playing = False
        self._play_after_id = None

        # Pose keypoint editing
        self._edit_mode: bool = False
        self._is_dirty: bool = False
        self._edit_pose: np.ndarray | None = None          # writable copy for current scene
        self._kpt_conf_threshold: float = CONF_THRESHOLD   # applied at load time
        self._conf_threshold_prompted: bool = False
        # keyframes[(person, kpt_idx)] = {frame_idx: (x, y, vis)}
        self._keyframes: dict[tuple[int, int], dict[int, tuple[float, float, int]]] = {}
        self._selected_person: int = 0
        self._selected_kpt: int | None = None
        self._dragging: bool = False
        self._display_frame_hw: tuple[int, int] = (640, 640)  # (h, w) of displayed image
        # Zoom state
        self._zoom: float = 1.0
        self._zoom_origin: tuple[float, float] = (0.5, 0.5)
        self._zoom_crop: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
        # Multi-select & rubber-band
        self._selected_kpts: set[tuple[int, int]] = set()
        self._rubber_band_start: tuple[int, int] | None = None
        self._rubber_band_end: tuple[int, int] | None = None
        self._rubber_band_active: bool = False
        self._last_drag_norm: tuple[float, float] | None = None
        self._undo_stack: list = []
        # vis_overrides[(person, kpt)] = {frame: vis_int} — step function, holds forward
        self._vis_overrides: dict[tuple[int, int], dict[int, int]] = {}

        self._build_ui()
        self._discover_scenes()
        self._schedule_auto_save()

    # ── UI Construction ────────────────────────────────────────────────────

    def _build_ui(self):
        screen_h = self.root.winfo_screenheight()
        usable_h = int(screen_h * 0.88)
        controls_h = 148
        plot_px = int(usable_h * 0.28)
        vid_px = min(usable_h - controls_h - plot_px, 600)
        vid_px = max(vid_px, 280)
        self.video_display_size = (vid_px, vid_px)
        _plot_figheight = max(1.8, plot_px / 100.0)

        self.root.title("VideoToMotion — Data Visualizer")
        self.root.configure(bg="#1e1e1e")

        # ── Left sidebar: scene list ──────────────────────────────────────
        sidebar = tk.Frame(self.root, bg="#252526", width=240)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 0), pady=4)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="Scenes", bg="#252526", fg="#cccccc",
                 font=("Segoe UI", 10, "bold"), anchor="w").pack(fill=tk.X, padx=6, pady=(6, 2))

        # Filter controls
        filter_frame = tk.LabelFrame(sidebar, text="Filter", bg="#252526", fg="#888888",
                                     font=("Segoe UI", 8), bd=1, relief=tk.GROOVE)
        filter_frame.pack(fill=tk.X, padx=4, pady=(0, 2))

        self._filter_vars: dict[str, tk.BooleanVar] = {}
        _filter_defs: list[tuple[str, bool]] = [
            ("pending",   True),
            ("approved",  True),
            ("keypoints", True),
            ("stage2_ok", True),
            ("flow",      True),
            ("legacy",    True),
            ("rejected",  False),
        ]
        _filter_labels = {
            "pending":   "? Pending",
            "approved":  "✓ Approved",
            "keypoints": "K Keypoints",
            "stage2_ok": "★ Stage 2 OK",
            "flow":      "K+ Flow",
            "legacy":    "· Legacy",
            "rejected":  "✗ Rejected",
        }
        for stage, default in _filter_defs:
            var = tk.BooleanVar(value=default)
            self._filter_vars[stage] = var
            color = STAGE_COLORS.get(stage, "#888888")
            tk.Checkbutton(
                filter_frame, text=_filter_labels[stage], variable=var,
                bg="#252526", fg=color, selectcolor="#1e1e1e",
                activebackground="#252526", activeforeground=color,
                font=("Segoe UI", 8), command=self._refresh_scene_list, anchor="w",
            ).pack(fill=tk.X, padx=4, pady=0)

        # Sort controls
        sort_frame = tk.Frame(sidebar, bg="#252526")
        sort_frame.pack(fill=tk.X, padx=4, pady=(2, 2))
        tk.Label(sort_frame, text="Sort:", bg="#252526", fg="#888888",
                 font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=2)
        self._sort_var = tk.StringVar(value="Name")
        ttk.OptionMenu(
            sort_frame, self._sort_var, "Name", "Name", "Stage", "MSE",
            command=lambda _: self._refresh_scene_list(),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Scene listbox
        scroll = tk.Scrollbar(sidebar, orient=tk.VERTICAL)
        self.scene_listbox = tk.Listbox(
            sidebar, yscrollcommand=scroll.set, bg="#1e1e1e", fg="#d4d4d4",
            selectbackground="#264f78", selectforeground="#ffffff",
            font=("Consolas", 8), activestyle="none", borderwidth=0,
            highlightthickness=0,
        )
        scroll.config(command=self.scene_listbox.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.scene_listbox.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.scene_listbox.bind("<<ListboxSelect>>", self._on_scene_select)

        # ── Right main area ───────────────────────────────────────────────
        main = tk.Frame(self.root, bg="#1e1e1e")
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Video + info row
        top_row = tk.Frame(main, bg="#1e1e1e")
        top_row.pack(fill=tk.BOTH, expand=True)

        # Video canvas
        self.video_canvas = tk.Canvas(
            top_row, bg="#000000", width=self.video_display_size[0],
            height=self.video_display_size[1], highlightthickness=0,
        )
        self.video_canvas.pack(side=tk.LEFT, padx=(0, 4))
        self.video_canvas.bind("<Button-1>",        self._on_canvas_click)
        self.video_canvas.bind("<B1-Motion>",       self._on_canvas_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.video_canvas.bind("<Button-3>",        self._on_canvas_right_click)

        # Keypoint panel (shown in edit mode, between video and info)
        self.kpt_panel = tk.Frame(top_row, bg="#252526", width=160)
        # Not packed initially — shown when edit mode is ON

        kpt_panel_title = tk.Label(self.kpt_panel, text="Keypoints", bg="#252526",
                                   fg="#cccccc", font=("Segoe UI", 9, "bold"), anchor="w")
        kpt_panel_title.pack(fill=tk.X, padx=6, pady=(6, 2))

        # Person selector
        person_row = tk.Frame(self.kpt_panel, bg="#252526")
        person_row.pack(fill=tk.X, padx=4)
        tk.Label(person_row, text="Person:", bg="#252526", fg="#888888",
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._person_var = tk.IntVar(value=0)
        for i in range(2):
            tk.Radiobutton(person_row, text=str(i), variable=self._person_var, value=i,
                           bg="#252526", fg="#cccccc", selectcolor="#1e1e1e",
                           activebackground="#252526", font=("Segoe UI", 8),
                           command=self._refresh_kpt_panel,
                           ).pack(side=tk.LEFT, padx=2)

        # Keypoint listbox
        kp_scroll = tk.Scrollbar(self.kpt_panel, orient=tk.VERTICAL)
        self.kpt_listbox = tk.Listbox(
            self.kpt_panel, yscrollcommand=kp_scroll.set, bg="#1e1e1e", fg="#d4d4d4",
            selectbackground="#264f78", font=("Consolas", 8), activestyle="none",
            borderwidth=0, highlightthickness=0,
        )
        kp_scroll.config(command=self.kpt_listbox.yview)
        kp_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.kpt_listbox.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.kpt_listbox.bind("<<ListboxSelect>>", self._on_kpt_listbox_select)
        self.kpt_listbox.bind("<Button-3>", self._on_kpt_list_right_click)

        # Keyframe button
        _kf_btn = {"bg": "#3c3c3c", "fg": "#cccccc", "activebackground": "#505050",
                   "activeforeground": "#ffffff", "relief": tk.FLAT,
                   "font": ("Segoe UI", 8), "padx": 6, "pady": 2, "cursor": "hand2"}
        self.btn_mark_kf = tk.Button(self.kpt_panel, text="⬥ Mark Keyframe",
                                     command=self._mark_keyframe, **_kf_btn)
        self.btn_mark_kf.pack(fill=tk.X, padx=4, pady=(2, 0))
        self.btn_clear_kf = tk.Button(self.kpt_panel, text="✕ Clear Keyframe",
                                      command=self._clear_keyframe, **_kf_btn)
        self.btn_clear_kf.pack(fill=tk.X, padx=4, pady=(2, 4))

        # Info + curation panel
        info_outer = tk.Frame(top_row, bg="#252526", width=270)
        info_outer.pack(side=tk.LEFT, fill=tk.Y)
        info_outer.pack_propagate(False)

        # Metadata text
        self.info_text = tk.Text(
            info_outer, bg="#252526", fg="#cccccc", font=("Consolas", 8),
            borderwidth=0, highlightthickness=0, state=tk.DISABLED,
            wrap=tk.WORD, height=11,
        )
        self.info_text.pack(fill=tk.X, padx=4, pady=(4, 0))

        ttk.Separator(info_outer, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=4, pady=4)

        # ── Curation controls ─────────────────────────────────────────────
        curation_frame = tk.Frame(info_outer, bg="#252526")
        curation_frame.pack(fill=tk.X, padx=4)

        # Status badge row
        status_row = tk.Frame(curation_frame, bg="#252526")
        status_row.pack(fill=tk.X, pady=(0, 4))
        tk.Label(status_row, text="Status:", bg="#252526", fg="#888888",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.status_badge = tk.Label(
            status_row, text="—", bg="#3c3c3c", fg="#cccccc",
            font=("Segoe UI", 9, "bold"), padx=6, pady=2,
        )
        self.status_badge.pack(side=tk.LEFT, padx=(4, 0))
        self.stage2_badge = tk.Label(
            status_row, text="", bg="#252526", fg="#888888",
            font=("Segoe UI", 8), padx=4, pady=2,
        )
        self.stage2_badge.pack(side=tk.LEFT, padx=(4, 0))
        self.force_val_badge = tk.Label(
            status_row, text="", bg="#252526", fg="#d79921",
            font=("Segoe UI", 8, "bold"),
        )
        self.force_val_badge.pack(side=tk.LEFT, padx=(4, 0))

        # Stage-1 action buttons
        _btn = {
            "bg": "#3c3c3c", "fg": "#cccccc", "activebackground": "#505050",
            "activeforeground": "#ffffff", "relief": tk.FLAT,
            "font": ("Segoe UI", 9), "padx": 8, "pady": 3, "cursor": "hand2", "bd": 0,
        }
        s1_row = tk.Frame(curation_frame, bg="#252526")
        s1_row.pack(fill=tk.X, pady=(0, 2))
        tk.Button(s1_row, text="✓ Approve", command=self._approve,
                  **{**_btn, "bg": "#1a6b3a", "activebackground": "#2a8a4a"}).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(s1_row, text="✗ Reject", command=self._reject,
                  **{**_btn, "bg": "#6b1a1a", "activebackground": "#8a2a2a"}).pack(side=tk.LEFT)

        fv_row = tk.Frame(curation_frame, bg="#252526")
        fv_row.pack(fill=tk.X, pady=(0, 4))
        self.btn_force_val = tk.Button(fv_row, text="★ Force Val: OFF",
                                       command=self._toggle_force_val, **_btn)
        self.btn_force_val.pack(side=tk.LEFT)

        # Stage-2 section (hidden until keypoints exist)
        self._stage2_sep = ttk.Separator(info_outer, orient=tk.HORIZONTAL)
        self._stage2_frame = tk.LabelFrame(
            info_outer, text="Stage 2: Keypoints", bg="#252526", fg="#689d6a",
            font=("Segoe UI", 8), bd=1, relief=tk.GROOVE,
        )
        self.kpt_info_label = tk.Label(
            self._stage2_frame, text="—", bg="#252526", fg="#888888",
            font=("Consolas", 8), wraplength=240, anchor="w",
        )
        self.kpt_info_label.pack(fill=tk.X, padx=4, pady=(2, 4))
        s2_row = tk.Frame(self._stage2_frame, bg="#252526")
        s2_row.pack(fill=tk.X, padx=4, pady=(0, 4))
        tk.Button(s2_row, text="✓✓ Approve KP", command=self._approve_stage2,
                  **{**_btn, "bg": "#1a6b3a", "activebackground": "#2a8a4a"}).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(s2_row, text="✗ Reject KP", command=self._reject_stage2,
                  **{**_btn, "bg": "#6b1a1a", "activebackground": "#8a2a2a"}).pack(side=tk.LEFT)

        # ── Options row ────────────────────────────────────────────────────
        opt_row = tk.Frame(main, bg="#1e1e1e")
        opt_row.pack(fill=tk.X, pady=(4, 0))

        _cb = {"bg": "#1e1e1e", "fg": "#cccccc", "selectcolor": "#1e1e1e",
               "activebackground": "#1e1e1e", "font": ("Segoe UI", 9)}

        self.show_skeleton_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_row, text="Skeleton", variable=self.show_skeleton_var,
                       command=self._refresh_frame, **_cb).pack(side=tk.LEFT, padx=4)

        self.show_flow_var = tk.BooleanVar(value=False)
        self.flow_check = tk.Checkbutton(opt_row, text="Flow overlay",
                                         variable=self.show_flow_var,
                                         command=self._refresh_frame, **_cb)
        self.flow_check.pack(side=tk.LEFT, padx=4)

        self.show_position_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_row, text="Position meter", variable=self.show_position_var,
                       command=self._refresh_frame, **_cb).pack(side=tk.LEFT, padx=4)

        self.speed_var = tk.DoubleVar(value=1.0)
        tk.Label(opt_row, text="Speed:", bg="#1e1e1e", fg="#888888",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(12, 2))
        for spd in (0.25, 0.5, 1.0, 2.0):
            tk.Radiobutton(opt_row, text=f"{spd}×", variable=self.speed_var, value=spd,
                           bg="#1e1e1e", fg="#cccccc", selectcolor="#1e1e1e",
                           activebackground="#1e1e1e", font=("Segoe UI", 9),
                           ).pack(side=tk.LEFT, padx=2)

        _edit_btn = {"relief": tk.FLAT, "font": ("Segoe UI", 9), "padx": 8, "pady": 3,
                     "cursor": "hand2", "bd": 0}
        self.btn_edit_mode = tk.Button(
            opt_row, text="✎ Edit Keypoints", command=self._toggle_edit_mode,
            bg="#3c3c3c", fg="#cccccc", activebackground="#505050", **_edit_btn,
        )
        self.btn_edit_mode.pack(side=tk.LEFT, padx=(16, 4))
        self.btn_export = tk.Button(
            opt_row, text="⬆ Export Pose Dataset", command=self._export_pose_dataset,
            bg="#3c3c3c", fg="#cccccc", activebackground="#505050", **_edit_btn,
        )
        self.btn_export.pack(side=tk.LEFT, padx=2)

        # ── Controls row ───────────────────────────────────────────────────
        ctrl = tk.Frame(main, bg="#1e1e1e")
        ctrl.pack(fill=tk.X, pady=(4, 0))

        _nav = {"bg": "#3c3c3c", "fg": "#cccccc", "activebackground": "#505050",
                "activeforeground": "#ffffff", "relief": tk.FLAT,
                "font": ("Segoe UI", 9), "padx": 8, "pady": 4, "cursor": "hand2"}
        tk.Button(ctrl, text="◀◀", command=self._step_back_10, **_nav).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="◀",  command=self._step_back,    **_nav).pack(side=tk.LEFT, padx=2)
        self.btn_play = tk.Button(ctrl, text="▶ Play", command=self._toggle_play, **_nav)
        self.btn_play.pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="▶",  command=self._step_forward,    **_nav).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="▶▶", command=self._step_forward_10, **_nav).pack(side=tk.LEFT, padx=2)

        self.frame_label = tk.Label(ctrl, text="Frame: 0 / 0  |  t=0.0s",
                                    bg="#1e1e1e", fg="#888888", font=("Consolas", 9))
        self.frame_label.pack(side=tk.LEFT, padx=12)

        self.pos_label = tk.Label(ctrl, text="pos=—", bg="#1e1e1e",
                                  fg="#4ec9b0", font=("Consolas", 9, "bold"))
        self.pos_label.pack(side=tk.LEFT, padx=4)

        # ── Timeline plot ──────────────────────────────────────────────────
        plot_frame = tk.Frame(main, bg="#1e1e1e")
        plot_frame.pack(fill=tk.X)

        self.fig = plt.figure(figsize=(9, _plot_figheight), facecolor="#1e1e1e")
        gs = self.fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
        self.ax      = self.fig.add_subplot(gs[0])
        self.ax_flow = self.fig.add_subplot(gs[1], sharex=self.ax)
        for ax in (self.ax, self.ax_flow):
            ax.set_facecolor("#1e1e1e")
            ax.tick_params(colors="#888888")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")
        plt.setp(self.ax.get_xticklabels(), visible=False)

        self.timeline_canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.timeline_canvas.get_tk_widget().pack(fill=tk.X)
        self.timeline_canvas.mpl_connect("button_press_event", self._on_plot_click)

        # Scrubber
        self.scrubber_var = tk.IntVar()
        self.scrubber = ttk.Scale(main, from_=0, to=1, orient=tk.HORIZONTAL,
                                  variable=self.scrubber_var, command=self._on_scrub)
        self.scrubber.pack(fill=tk.X, pady=(4, 2))

        # Keyboard shortcuts
        self.root.bind("<space>",       lambda e: self._toggle_play())
        self.root.bind("<Left>",        lambda e: self._step_back())
        self.root.bind("<Right>",       lambda e: self._step_forward())
        self.root.bind("<Shift-Left>",  lambda e: self._step_back_10())
        self.root.bind("<Shift-Right>", lambda e: self._step_forward_10())
        self.root.bind("<a>",           lambda e: self._hotkey_approve())
        self.root.bind("<d>",           lambda e: self._hotkey_reject())
        self.root.bind("<z>",           lambda e: self._reset_zoom())
        self.root.bind_all("<Control-z>", lambda e: self._undo())
        # Intercept a/d/space on the Listbox: prevent its letter-search navigation
        self.scene_listbox.bind("<a>",     lambda e: self._hotkey_approve() or "break")
        self.scene_listbox.bind("<d>",     lambda e: self._hotkey_reject() or "break")
        self.scene_listbox.bind("<space>", lambda e: self._toggle_play() or "break")
        self.video_canvas.bind("<MouseWheel>", self._on_canvas_zoom)
        self.video_canvas.bind("<Button-2>",   self._on_pan_start)
        self.video_canvas.bind("<B2-Motion>",  self._on_pan)

        self._draw_empty_plot()

    # ── Scene Discovery & Filtering ────────────────────────────────────────

    def _scene_stage(self, scene_dir: Path) -> str:
        return get_scene_stage(
            scene_dir,
            model_name=self.cfg.pose.model_name,
            flow_method=self.cfg.flow.method,
            flow_output_features=self.cfg.flow.output_features,
            flow_scale=self.cfg.flow.scale,
        )

    def _discover_scenes(self):
        if not self.processed_dir.exists():
            return
        self._all_scenes = list(discover_scenes(
            self.data_dir, include_rejected=True, require_labels=True,
        ))
        self._refresh_scene_list()

    def _refresh_scene_list(self):
        """Re-apply filter/sort to _all_scenes and repopulate the listbox."""
        active = {s for s, v in self._filter_vars.items() if v.get()}
        sort_mode = self._sort_var.get()
        _order = {"rejected": 0, "pending": 1, "legacy": 2, "approved": 3,
                  "keypoints": 4, "stage2_ok": 5, "flow": 6}

        filtered = [
            (sid, st) for sid, st in self._all_scenes
            if self._scene_stage(self.processed_dir / sid) in active
        ]

        if sort_mode == "Stage":
            filtered.sort(key=lambda x: (
                _order.get(self._scene_stage(self.processed_dir / x[0]), 99), x[0]
            ))
        elif sort_mode == "MSE":
            def _mse_key(item):
                review = read_review(self.processed_dir / item[0])
                mse_str = review.get("mse", "")
                try:
                    return float(mse_str)
                except (ValueError, TypeError):
                    return float("inf")
            filtered.sort(key=_mse_key)
        else:
            filtered.sort(key=lambda x: x[0])

        self.scenes = [sid for sid, _ in filtered]

        self.scene_listbox.delete(0, tk.END)
        for sid in self.scenes:
            stage = self._scene_stage(self.processed_dir / sid)
            review = read_review(self.processed_dir / sid)
            icon  = STAGE_ICONS.get(stage, "·")
            star  = " ★" if review.get("force_val", False) else ""
            vid   = " ▶" if (self.preprocessed_dir / f"{sid}.mp4").exists() else ""
            mse_str = review.get("mse", "")
            mse_tag = f"  mse={mse_str}" if mse_str else ""
            self.scene_listbox.insert(tk.END, f"{icon} {sid}{star}{vid}{mse_tag}")
            self.scene_listbox.itemconfig(tk.END, fg=STAGE_COLORS.get(stage, "#d4d4d4"))

        # Restore selection
        if self.current_scene and self.current_scene in self.scenes:
            idx = self.scenes.index(self.current_scene)
            self.scene_listbox.selection_set(idx)
            self.scene_listbox.see(idx)
        elif self.scenes and self.current_scene is None:
            self.scene_listbox.selection_set(0)
            self._load_scene(self.scenes[0])

    # ── Scene Loading ──────────────────────────────────────────────────────

    def _load_scene(self, scene_id: str):
        # Save any unsaved edits from the previous scene first
        if self._is_dirty:
            self._save_edits()
        # Reset edit state
        self._edit_mode = False
        self._is_dirty = False
        self._edit_pose = None
        self._keyframes = {}
        self._vis_overrides = {}
        self._selected_kpt = None
        self.btn_edit_mode.config(text="✎ Edit Keypoints", bg="#3c3c3c")

        was_playing = self._playing
        self._stop_playback()
        self.current_scene = scene_id
        self._zoom = 1.0
        self._zoom_origin = (0.5, 0.5)
        self._zoom_crop = (0.0, 0.0, 1.0, 1.0)
        self._selected_kpts = set()
        scene_dir = self.processed_dir / scene_id

        # Pose keypoints: try multiclass first, then configured model, then legacy
        multiclass_kpt_path = scene_dir / "keypoints" / "vrlens-finetunes-multiclass-v2-yolo11m-pose.npy"
        if multiclass_kpt_path.exists():
            kpt_path = multiclass_kpt_path
            self._is_multiclass = True
        else:
            kpt_path = resolve_keypoints_path(scene_dir, self.cfg.pose.model_name)
            self._is_multiclass = False
        self.pose_data = np.load(str(kpt_path), mmap_mode="r") if kpt_path else None

        # Ask for confidence threshold at load time and apply to vis once per session
        if self.pose_data is not None:
            if not self._conf_threshold_prompted:
                import tkinter.simpledialog as sd
                thresh = sd.askfloat(
                    "Confidence Threshold",
                    "Hide keypoints below confidence (0–1):\n"
                    "(below threshold → invisible/grey; vis=2 → light blue)",
                    initialvalue=self._kpt_conf_threshold,
                    minvalue=0.0, maxvalue=1.0,
                )
                if thresh is not None:
                    self._kpt_conf_threshold = thresh
                self._conf_threshold_prompted = True
            self.kpt_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
            self.kpt_panel.pack_propagate(False)
        else:
            self.kpt_panel.pack_forget()

        self.label_data = np.load(str(scene_dir / "labels.npy"), mmap_mode="r")

        # Optical flow: new path first, then legacy
        flow_p = resolve_flow_path(scene_dir, self.cfg.flow.method,
                                   self.cfg.flow.output_features, self.cfg.flow.scale)
        self.flow_data = np.load(str(flow_p), mmap_mode="r") if flow_p else None

        self.total_frames = len(self.label_data)
        self.current_frame = 0

        # Metadata
        meta_path = scene_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        self.fps = float(self.metadata.get("fps", 10.0))

        # Video capture
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        video_path = self.preprocessed_dir / f"{scene_id}.mp4"
        if video_path.exists():
            self.cap = cv2.VideoCapture(str(video_path))

        # Flow checkbox
        if self.flow_data is not None:
            self.flow_check.config(state=tk.NORMAL)
        else:
            self.show_flow_var.set(False)
            self.flow_check.config(state=tk.DISABLED)

        self.scrubber.config(to=max(1, self.total_frames - 1))
        self.scrubber_var.set(0)

        self._update_curation_ui()
        self._update_info_panel()
        self._draw_plot()
        self._refresh_frame()
        if was_playing:
            self._start_playback()

    def _on_scene_select(self, event=None):
        sel = self.scene_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx < len(self.scenes):
            scene_id = self.scenes[idx]
            if scene_id != self.current_scene:
                self._load_scene(scene_id)

    # ── Info Panel ─────────────────────────────────────────────────────────

    def _update_info_panel(self):
        m = self.metadata
        lines = [
            f"Scene: {self.current_scene}",
            "",
            f"Frames:   {self.total_frames}",
            f"FPS:      {self.fps}",
            f"Duration: {self.total_frames / max(self.fps, 1):.1f}s",
            f"Res:      {m.get('width', '?')}×{m.get('height', '?')}",
            "",
        ]
        if self.pose_data is not None:
            n_kpt = self.pose_data.shape[2] if self.pose_data.ndim == 4 else "?"
            try:
                visible = (self.pose_data[:, 0, :, 2] > CONF_THRESHOLD).any(axis=-1)
                det_rate = visible.mean() * 100
                lines.append(f"Kpts: {n_kpt}kpt  det {det_rate:.1f}%")
            except Exception:
                lines.append(f"Kpts: {n_kpt}kpt")
        else:
            lines.append("Kpts: none")
        lines.append(f"Flow: {'yes' if self.flow_data is not None else 'no'}")
        if self.label_data is not None:
            lines += [
                "",
                f"Labels: [{self.label_data.min():.3f}, {self.label_data.max():.3f}]",
                f"Mean: {self.label_data.mean():.3f}  Std: {self.label_data.std():.3f}",
            ]
        # MSE from review.json
        if self.current_review:
            mse_str = self.current_review.get("mse", "")
            if mse_str:
                lines.append(f"MSE: {mse_str}")
        if getattr(self, "_is_multiclass", False):
            lines.append("Mode: multiclass")
        lines += ["", "Space: play/pause", "←/→: 1fr  ⇧←/⇧→: 10fr"]

        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, "\n".join(lines))
        self.info_text.config(state=tk.DISABLED)

    # ── Curation UI ────────────────────────────────────────────────────────

    def _update_curation_ui(self):
        if self.current_scene is None:
            return
        scene_dir = self.processed_dir / self.current_scene
        self.current_review = read_review(scene_dir)
        stage = self._scene_stage(scene_dir)

        # Status badge
        color = STAGE_COLORS.get(stage, "#888888")
        status = self.current_review.get("status", "pending")
        self.status_badge.config(text=status.upper(), bg=color, fg="#ffffff")

        # Stage-2 badge
        s2 = self.current_review.get("stage2_status", "")
        if s2 == "approved":
            s2_color = STAGE_COLORS["stage2_ok"]
        elif s2 == "rejected":
            s2_color = STAGE_COLORS["rejected"]
        else:
            s2_color = "#888888"
        self.stage2_badge.config(text=s2.upper() if s2 and s2 != "pending" else "", fg=s2_color)

        # Force-val badge + button
        force_val = self.current_review.get("force_val", False)
        self.force_val_badge.config(text="★ FORCE-VAL" if force_val else "")
        self.btn_force_val.config(
            text="★ Force Val: ON" if force_val else "★ Force Val: OFF",
            bg="#5a4a00" if force_val else "#3c3c3c",
        )

        # Show/hide stage-2 section
        has_kpts = resolve_keypoints_path(scene_dir, self.cfg.pose.model_name) is not None
        if has_kpts:
            self._stage2_sep.pack(fill=tk.X, padx=4, pady=4)
            self._stage2_frame.pack(fill=tk.X, padx=4, pady=(0, 4))
            if self.pose_data is not None:
                try:
                    n_kpt = self.pose_data.shape[2] if self.pose_data.ndim == 4 else "?"
                    visible = (self.pose_data[:, 0, :, 2] > CONF_THRESHOLD).any(axis=-1)
                    det_rate = visible.mean() * 100
                    self.kpt_info_label.config(
                        text=f"{n_kpt}kpt  |  det {det_rate:.1f}%  |  shape {self.pose_data.shape}",
                    )
                except Exception:
                    self.kpt_info_label.config(text="Keypoints loaded")
        else:
            self._stage2_sep.pack_forget()
            self._stage2_frame.pack_forget()

    def _approve(self):
        if self.current_scene:
            approve(self.processed_dir / self.current_scene)
            self._update_curation_ui()
            self._refresh_scene_list()

    def _reject(self):
        if self.current_scene:
            reject(self.processed_dir / self.current_scene)
            self._update_curation_ui()
            self._refresh_scene_list()

    def _toggle_force_val(self):
        if self.current_scene:
            current = self.current_review.get("force_val", False)
            set_force_val(self.processed_dir / self.current_scene, not current)
            self._update_curation_ui()
            self._refresh_scene_list()

    def _approve_stage2(self):
        if self.current_scene:
            approve_stage2(self.processed_dir / self.current_scene)
            self._update_curation_ui()
            self._refresh_scene_list()

    def _reject_stage2(self):
        if self.current_scene:
            reject_stage2(self.processed_dir / self.current_scene)
            self._update_curation_ui()
            self._refresh_scene_list()

    def _advance_scene(self):
        """Move to the next scene in the filtered list."""
        if self.current_scene not in self.scenes:
            return
        idx = self.scenes.index(self.current_scene)
        if idx + 1 < len(self.scenes):
            next_id = self.scenes[idx + 1]
            self.scene_listbox.selection_clear(0, tk.END)
            self.scene_listbox.selection_set(idx + 1)
            self.scene_listbox.see(idx + 1)
            self._load_scene(next_id)

    def _advance_after(self, prev_idx: int):
        """Load the scene that comes after prev_idx in the possibly-mutated list."""
        if not self.scenes:
            return
        if self.current_scene in self.scenes:
            # Scene still present (e.g. filter shows approved): move past it
            target_idx = self.scenes.index(self.current_scene) + 1
        else:
            # Scene was filtered out; take the slot that now occupies prev_idx
            target_idx = min(prev_idx, len(self.scenes) - 1)
        if target_idx < 0 or target_idx >= len(self.scenes):
            return
        next_id = self.scenes[target_idx]
        if next_id == self.current_scene:
            return
        self.scene_listbox.selection_clear(0, tk.END)
        self.scene_listbox.selection_set(target_idx)
        self.scene_listbox.see(target_idx)
        self._load_scene(next_id)

    def _hotkey_approve(self):
        if isinstance(self.root.focus_get(), (tk.Entry, tk.Text)):
            return
        if not self.current_scene:
            return
        prev_idx = self.scenes.index(self.current_scene) if self.current_scene in self.scenes else -1
        self._approve()
        self._advance_after(prev_idx)

    def _hotkey_reject(self):
        if isinstance(self.root.focus_get(), (tk.Entry, tk.Text)):
            return
        if not self.current_scene:
            return
        prev_idx = self.scenes.index(self.current_scene) if self.current_scene in self.scenes else -1
        self._reject()
        self._advance_after(prev_idx)

    # ── Frame Display ──────────────────────────────────────────────────────

    def _get_video_frame(self, frame_idx: int) -> np.ndarray | None:
        if self.cap is None or not self.cap.isOpened():
            return None
        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_pos != frame_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def _refresh_frame(self, *_args):
        if self.current_scene is None:
            return
        frame_idx = max(0, min(self.current_frame, self.total_frames - 1))

        frame = self._get_video_frame(frame_idx)
        if frame is None:
            frame = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No video", (240, 320), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (80, 80, 80), 2)

        if self.show_skeleton_var.get() and self._get_active_pose() is not None:
            kp = self._get_active_pose()[frame_idx]   # [max_persons, n_kpt, 3]
            n_persons, n_kpt = kp.shape[0], kp.shape[1]
            vis_arr = np.zeros((n_persons, n_kpt), dtype=np.int32)
            for pi in range(n_persons):
                for ki in range(n_kpt):
                    _, _, v = self._get_effective_kpt(pi, ki, frame_idx)
                    vis_arr[pi, ki] = v
            frame = draw_skeleton(frame.copy(), kp, self.bones, vis=vis_arr)

        if self.show_flow_var.get() and self.flow_data is not None:
            frame = draw_flow_overlay(frame, self.flow_data[frame_idx])

        # Zoom: crop a sub-region of the display frame and upscale back to
        # the same size so the canvas widget never changes dimensions.
        frame = fit_image(frame, *self.video_display_size)
        self._display_frame_hw = (frame.shape[0], frame.shape[1])

        if self._zoom > 1.001:
            dh, dw = frame.shape[:2]
            half = 0.5 / self._zoom
            cx, cy = self._zoom_origin
            xi = max(0, int((cx - half) * dw))
            xa = min(dw, max(xi + 2, int((cx + half) * dw)))
            yi = max(0, int((cy - half) * dh))
            ya = min(dh, max(yi + 2, int((cy + half) * dh)))
            crop = frame[yi:ya, xi:xa]
            frame = cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_LINEAR)
            self._zoom_crop = (xi / dw, yi / dh, xa / dw, ya / dh)
        else:
            self._zoom_crop = (0.0, 0.0, 1.0, 1.0)

        if self.show_position_var.get() and self.label_data is not None:
            frame = draw_position_overlay(frame, float(self.label_data[frame_idx]))

        # Draw all selected keypoint highlights
        if self._get_active_pose() is not None:
            active_p = self._person_var.get() if hasattr(self, "_person_var") else 0
            draw_kpts = self._selected_kpts if self._selected_kpts else (
                {(active_p, self._selected_kpt)} if self._selected_kpt is not None else set()
            )
            for (p, k) in draw_kpts:
                x, y, _ = self._get_effective_kpt(p, k, frame_idx)
                px, py = self._norm_to_canvas(x, y)
                is_primary = (p == active_p and k == self._selected_kpt)
                cv2.circle(frame, (px, py), 10,
                           (0, 255, 255) if is_primary else (255, 255, 0), 2, cv2.LINE_AA)
                kkey = (p, k)
                if kkey in self._keyframes and frame_idx in self._keyframes[kkey]:
                    cv2.circle(frame, (px, py), 6, (0, 255, 255), -1, cv2.LINE_AA)

        # Rubber-band selection rectangle overlay
        if self._rubber_band_active and self._rubber_band_start and self._rubber_band_end:
            cv2.rectangle(frame,
                          (int(self._rubber_band_start[0]), int(self._rubber_band_start[1])),
                          (int(self._rubber_band_end[0]),   int(self._rubber_band_end[1])),
                          (255, 255, 0), 2)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        self.video_canvas.config(width=frame.shape[1], height=frame.shape[0])
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.video_canvas._photo = photo  # prevent GC

        t = frame_idx / max(self.fps, 1)
        pos = float(self.label_data[frame_idx]) if self.label_data is not None else 0
        zoom_str = f"  |  zoom {self._zoom:.1f}x" if self._zoom > 1.001 else ""
        self.frame_label.config(text=f"Frame: {frame_idx} / {self.total_frames - 1}  |  t={t:.2f}s{zoom_str}")

        pos_text = f"pos={pos:.3f}"
        if self.flow_data is not None:
            finfo = _decode_flow_features(self.flow_data[frame_idx])
            gmag = finfo["global_mean_mag"]
            gangle_deg = float(np.degrees(finfo["global_angle"]))
            pos_text += f"  |  flow mag={gmag:.1f}  dir={gangle_deg:.0f}°"
        self.pos_label.config(text=pos_text)

        self._update_cursor(frame_idx)
        if self.pose_data is not None:
            self._refresh_kpt_panel()

    # ── Timeline Plot ──────────────────────────────────────────────────────

    def _draw_empty_plot(self):
        for ax in (self.ax, self.ax_flow):
            ax.clear()
            ax.set_facecolor("#1e1e1e")
            ax.tick_params(colors="#555555", labelsize=7)
        self.ax.set_ylabel("Position", color="#666666", fontsize=8)
        self.ax_flow.set_xlabel("Frame", color="#666666", fontsize=8)
        self.ax_flow.set_ylabel("Flow mag", color="#666666", fontsize=8)
        self.fig.tight_layout(pad=0.5)
        self.timeline_canvas.draw()

    def _draw_plot(self):
        if self.label_data is None:
            return
        frames = np.arange(self.total_frames)

        self.ax.clear()
        self.ax.set_facecolor("#1e1e1e")
        self.ax.plot(frames, self.label_data, color="#4ec9b0", lw=1.0)
        self.ax.set_xlim(0, self.total_frames)
        self.ax.set_ylim(-0.05, 1.05)
        self.ax.set_ylabel("Position [0,1]", color="#666666", fontsize=8)
        self.ax.tick_params(colors="#555555", labelsize=7)
        plt.setp(self.ax.get_xticklabels(), visible=False)
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#333333")
        self._cursor_line = self.ax.axvline(x=0, color="#ffcc00", lw=1.2, alpha=0.85)

        self.ax_flow.clear()
        self.ax_flow.set_facecolor("#1e1e1e")
        if self.flow_data is not None:
            global_mag = self.flow_data[:, FLOW_GLOBAL_OFFSET].copy()
            global_p90 = self.flow_data[:, FLOW_GLOBAL_OFFSET + 4].copy()
            self.ax_flow.fill_between(frames, 0, global_mag, color="#ce9178", alpha=0.45, lw=0)
            self.ax_flow.plot(frames, global_mag, color="#ce9178", lw=0.9, label="mean mag")
            self.ax_flow.plot(frames, global_p90, color="#d7ba7d", lw=0.7, linestyle="--",
                              alpha=0.7, label="p90 mag")
            cell_colors = ["#569cd6", "#4ec9b0", "#b5cea8", "#9cdcfe"]
            for ci in range(FLOW_N_ACTIVE_CELLS):
                self.ax_flow.plot(frames, self.flow_data[:, ci * 4], color=cell_colors[ci],
                                  lw=0.5, alpha=0.5, label=f"cell {ci}")
            self.ax_flow.legend(fontsize=6, loc="upper right", facecolor="#1e1e1e",
                                labelcolor="#cccccc", edgecolor="#444444", ncol=3)
            ymax = max(global_p90.max() * 1.15, 1.0)
            self.ax_flow.set_ylim(0, ymax)
        else:
            self.ax_flow.text(0.5, 0.5, "No flow data", transform=self.ax_flow.transAxes,
                              ha="center", va="center", color="#555555", fontsize=9)

        self.ax_flow.set_xlim(0, self.total_frames)
        self.ax_flow.set_xlabel("Frame", color="#666666", fontsize=8)
        self.ax_flow.set_ylabel("Flow mag (px)", color="#666666", fontsize=8)
        self.ax_flow.tick_params(colors="#555555", labelsize=7)
        for spine in self.ax_flow.spines.values():
            spine.set_edgecolor("#333333")
        self._cursor_line_flow = self.ax_flow.axvline(x=0, color="#ffcc00", lw=1.2, alpha=0.85)

        self.fig.tight_layout(pad=0.5)
        self.timeline_canvas.draw()

    def _update_cursor(self, frame_idx: int):
        try:
            if hasattr(self, "_cursor_line"):
                self._cursor_line.set_xdata([frame_idx, frame_idx])
            if hasattr(self, "_cursor_line_flow"):
                self._cursor_line_flow.set_xdata([frame_idx, frame_idx])
            self.timeline_canvas.draw_idle()
        except Exception:
            pass

    def _on_plot_click(self, event):
        if event.inaxes not in (self.ax, self.ax_flow) or self.total_frames == 0:
            return
        self._seek(int(round(event.xdata)))

    # ── Navigation ────────────────────────────────────────────────────────

    def _seek(self, frame: int):
        self.current_frame = max(0, min(frame, self.total_frames - 1))
        self.scrubber_var.set(self.current_frame)
        self._refresh_frame()

    def _on_scrub(self, value):
        frame = int(float(value))
        if frame != self.current_frame:
            self.current_frame = frame
            self._refresh_frame()

    def _step_back(self):      self._seek(self.current_frame - 1)
    def _step_forward(self):   self._seek(self.current_frame + 1)
    def _step_back_10(self):   self._seek(self.current_frame - 10)
    def _step_forward_10(self): self._seek(self.current_frame + 10)

    # ── Playback ──────────────────────────────────────────────────────────

    def _toggle_play(self):
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        self._playing = True
        self.btn_play.config(text="⏸ Pause")
        self._schedule_next_frame()

    def _stop_playback(self):
        self._playing = False
        self.btn_play.config(text="▶ Play")
        if self._play_after_id is not None:
            self.root.after_cancel(self._play_after_id)
            self._play_after_id = None

    def _schedule_next_frame(self):
        if not self._playing:
            return
        delay_ms = max(1, int(1000 / (self.fps * self.speed_var.get())))
        self._play_after_id = self.root.after(delay_ms, self._advance_frame)

    def _advance_frame(self):
        if not self._playing:
            return
        if self.current_frame >= self.total_frames - 1:
            self._stop_playback()
            return
        self.current_frame += 1
        self.scrubber_var.set(self.current_frame)
        self._refresh_frame()
        self._schedule_next_frame()

    def on_close(self):
        if self._is_dirty:
            self._save_edits()
        self._stop_playback()
        if self.cap is not None:
            self.cap.release()
        plt.close(self.fig)
        self.root.destroy()

    # ── Pose Keypoint Editing ──────────────────────────────────────────────

    def _toggle_edit_mode(self):
        self._edit_mode = not self._edit_mode
        if self._edit_mode:
            # Enter edit mode: make a writable copy
            if self.pose_data is not None:
                self._edit_pose = np.array(self.pose_data, dtype=np.float32)
            else:
                self._edit_pose = None
            self._keyframes = {}
            self._vis_overrides = {}
            self._selected_kpt = None
            self._selected_kpts = set()
            self.btn_edit_mode.config(text="✎ Exit Edit Mode", bg="#5a3a00")
        else:
            if self._is_dirty:
                self._save_edits()
            self.btn_edit_mode.config(text="✎ Edit Keypoints", bg="#3c3c3c")
            self._edit_pose = None
            self._keyframes = {}
            self._vis_overrides = {}
            self._selected_kpt = None
            self._selected_kpts = set()
        self._refresh_kpt_panel()
        self._refresh_frame()

    def _get_active_pose(self) -> np.ndarray | None:
        """Return the pose array being displayed (edit copy or original)."""
        if self._edit_mode and self._edit_pose is not None:
            return self._edit_pose
        return self.pose_data

    def _get_effective_kpt(self, person: int, kpt: int, frame: int) -> tuple[float, float, int]:
        """Return (x, y, vis) for a keypoint at a given frame, applying keyframe interpolation."""
        key = (person, kpt)
        src = (self._edit_pose if self._edit_mode and self._edit_pose is not None
               else self.pose_data)
        if src is None:
            return (0.0, 0.0, 0)

        raw = src[frame, person, kpt]
        base_x, base_y, base_conf = float(raw[0]), float(raw[1]), float(raw[2])
        # Sentinel encoding: occluded = 3.0 (new) or ~0.5 (legacy backward compat)
        if base_conf >= 2.0 or (0.45 <= base_conf <= 0.55):
            base_vis = 2
        elif base_conf >= self._kpt_conf_threshold:
            base_vis = 1
        else:
            base_vis = 0

        # Visibility override: step function — find most recent override at or before frame
        eff_vis = base_vis
        if key in self._vis_overrides:
            frames_at_or_before = [f for f in self._vis_overrides[key] if f <= frame]
            if frames_at_or_before:
                eff_vis = self._vis_overrides[key][max(frames_at_or_before)]

        if key not in self._keyframes:
            return (base_x, base_y, eff_vis)

        kf_dict = self._keyframes[key]
        if not kf_dict:
            return (base_x, base_y, eff_vis)

        frames_sorted = sorted(kf_dict.keys())

        if frame in kf_dict:
            kx, ky, _ = kf_dict[frame]
            return (kx, ky, eff_vis)

        # Find surrounding keyframes
        before = [f for f in frames_sorted if f < frame]
        after  = [f for f in frames_sorted if f > frame]

        if before and after:
            f0, f1 = before[-1], after[0]
            x0, y0, _ = kf_dict[f0]
            x1, y1, _ = kf_dict[f1]
            t = (frame - f0) / (f1 - f0)
            return (x0 + t * (x1 - x0), y0 + t * (y1 - y0), eff_vis)
        elif before:
            return (base_x, base_y, eff_vis)  # after last keyframe: use base data
        else:
            return (base_x, base_y, eff_vis)  # before first keyframe: use base data

    def _canvas_to_norm(self, cx: int, cy: int) -> tuple[float, float]:
        """Convert canvas pixels to normalised [0,1] image coords, accounting for zoom."""
        h, w = self._display_frame_hw
        fx, fy = cx / max(w, 1), cy / max(h, 1)
        x1n, y1n, x2n, y2n = self._zoom_crop
        return (x1n + fx * (x2n - x1n), y1n + fy * (y2n - y1n))

    def _norm_to_canvas(self, x: float, y: float) -> tuple[int, int]:
        """Convert normalised [0,1] image coords to canvas pixels, accounting for zoom."""
        h, w = self._display_frame_hw
        x1n, y1n, x2n, y2n = self._zoom_crop
        fx = (x - x1n) / max(x2n - x1n, 1e-6)
        fy = (y - y1n) / max(y2n - y1n, 1e-6)
        return (int(fx * w), int(fy * h))

    def _on_canvas_zoom(self, event):
        """Mouse wheel: zoom in/out centred on cursor position."""
        zoom_factor = 1.2 if event.delta > 0 else 1.0 / 1.2
        new_zoom = max(1.0, min(10.0, self._zoom * zoom_factor))
        if abs(new_zoom - self._zoom) < 0.01:
            return
        h, w = self._display_frame_hw
        cursor_nx, cursor_ny = self._canvas_to_norm(event.x, event.y)  # OLD crop
        self._zoom = new_zoom
        fx, fy = event.x / max(w, 1), event.y / max(h, 1)
        # Keep cursor position stable: new_ox = cursor + (0.5 - fx) / new_zoom
        half = 0.5 / new_zoom
        self._zoom_origin = (
            max(half, min(1.0 - half, cursor_nx + (0.5 - fx) / new_zoom)),
            max(half, min(1.0 - half, cursor_ny + (0.5 - fy) / new_zoom)),
        )
        self._refresh_frame()

    def _on_pan_start(self, event):
        """Middle-click: start panning the zoomed view."""
        self._pan_start = (event.x, event.y)
        self._pan_origin_start = self._zoom_origin

    def _on_pan(self, event):
        """Middle-click drag: pan the zoomed view."""
        if not hasattr(self, "_pan_start") or self._zoom <= 1.001:
            return
        x1n, y1n, x2n, y2n = self._zoom_crop
        h, w = self._display_frame_hw
        dx = -(event.x - self._pan_start[0]) / max(w, 1) * (x2n - x1n)
        dy = -(event.y - self._pan_start[1]) / max(h, 1) * (y2n - y1n)
        half = 0.5 / self._zoom
        self._zoom_origin = (
            max(half, min(1.0 - half, self._pan_origin_start[0] + dx)),
            max(half, min(1.0 - half, self._pan_origin_start[1] + dy)),
        )
        self._refresh_frame()

    def _reset_zoom(self):
        """Reset zoom to 1x (press Z)."""
        self._zoom = 1.0
        self._zoom_origin = (0.5, 0.5)
        self._zoom_crop = (0.0, 0.0, 1.0, 1.0)
        self._refresh_frame()

    def _find_keypoint_near(self, cx: int, cy: int, radius: int = 12) -> tuple[int, int] | None:
        """Return (person, kpt_idx) of the keypoint nearest to canvas (cx,cy), or None."""
        src = self._get_active_pose()
        if src is None:
            return None
        best_dist = radius
        best = None
        frame = self.current_frame
        for p_idx in range(src.shape[1]):
            n_kpt = src.shape[2]
            for k in range(n_kpt):
                x, y, _ = self._get_effective_kpt(p_idx, k, frame)
                px, py = self._norm_to_canvas(x, y)
                d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best = (p_idx, k)
        return best

    def _on_canvas_click(self, event):
        hit = self._find_keypoint_near(event.x, event.y)
        if hit is not None:
            # Keep the multi-selection when clicking on a member of it;
            # only collapse to a single selection otherwise.
            if not (hit in self._selected_kpts and len(self._selected_kpts) > 1):
                self._selected_kpts = {hit}
            self._selected_person, self._selected_kpt = hit
            self._person_var.set(self._selected_person)
            if self._edit_mode:
                self._push_undo()
                self._dragging = True
                self._last_drag_norm = self._canvas_to_norm(event.x, event.y)
            self._refresh_kpt_panel()
            self._refresh_frame()
        elif self._edit_mode:
            # Start rubber-band selection on empty-space click
            self._rubber_band_start = (event.x, event.y)
            self._rubber_band_end = (event.x, event.y)
            self._rubber_band_active = True
            self._selected_kpts = set()

    def _on_canvas_drag(self, event):
        if self._rubber_band_active:
            self._rubber_band_end = (event.x, event.y)
            self._refresh_frame()
            return
        if not self._edit_mode or not self._dragging:
            return
        if self._edit_pose is None or self._last_drag_norm is None:
            return
        nx, ny = self._canvas_to_norm(event.x, event.y)
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        dx = nx - self._last_drag_norm[0]
        dy = ny - self._last_drag_norm[1]
        self._last_drag_norm = (nx, ny)
        frame = self.current_frame
        targets = self._selected_kpts if self._selected_kpts else (
            {(self._selected_person, self._selected_kpt)} if self._selected_kpt is not None else set()
        )
        for (p, k) in targets:
            new_x = max(0.0, min(1.0, float(self._edit_pose[frame, p, k, 0]) + dx))
            new_y = max(0.0, min(1.0, float(self._edit_pose[frame, p, k, 1]) + dy))
            self._edit_pose[frame, p, k, 0] = new_x
            self._edit_pose[frame, p, k, 1] = new_y
            self._edit_pose[frame, p, k, 2] = max(0.9, float(self._edit_pose[frame, p, k, 2]))
            kkey = (p, k)
            if kkey in self._keyframes and frame in self._keyframes[kkey]:
                _, _, vis = self._keyframes[kkey][frame]
                self._keyframes[kkey][frame] = (new_x, new_y, vis)
        self._is_dirty = True
        self._refresh_frame()

    def _on_canvas_release(self, event):
        self._dragging = False
        self._last_drag_norm = None
        if self._rubber_band_active and self._rubber_band_start and self._rubber_band_end:
            self._rubber_band_active = False
            rx1 = min(self._rubber_band_start[0], self._rubber_band_end[0])
            rx2 = max(self._rubber_band_start[0], self._rubber_band_end[0])
            ry1 = min(self._rubber_band_start[1], self._rubber_band_end[1])
            ry2 = max(self._rubber_band_start[1], self._rubber_band_end[1])
            self._rubber_band_start = None
            self._rubber_band_end = None
            src = self._get_active_pose()
            if src is not None and (rx2 - rx1) > 3 and (ry2 - ry1) > 3:
                new_sel: set[tuple[int, int]] = set()
                frame = self.current_frame
                for pi in range(src.shape[1]):
                    for ki in range(src.shape[2]):
                        x, y, _ = self._get_effective_kpt(pi, ki, frame)
                        cx, cy = self._norm_to_canvas(x, y)
                        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                            new_sel.add((pi, ki))
                if new_sel:
                    self._selected_kpts = new_sel
                    first = sorted(new_sel)[0]
                    self._selected_person, self._selected_kpt = first
                    self._person_var.set(self._selected_person)
                    self._refresh_kpt_panel()
            self._refresh_frame()
        else:
            self._rubber_band_active = False
            self._rubber_band_start = None
            self._rubber_band_end = None

    def _on_canvas_right_click(self, event):
        """Right-click on a keypoint to cycle its visibility: 0→1→2→0.
        Visibility holds forward from this frame until another vis override."""
        if not self._edit_mode:
            return
        hit = self._find_keypoint_near(event.x, event.y)
        if hit is None:
            return
        p, k = hit
        self._push_undo()
        self._selected_person, self._selected_kpt = p, k
        self._person_var.set(p)
        frame = self.current_frame
        _, _, curr_vis = self._get_effective_kpt(p, k, frame)
        new_vis = (curr_vis + 1) % 3
        key = (p, k)
        if key not in self._vis_overrides:
            self._vis_overrides[key] = {}
        self._vis_overrides[key][frame] = new_vis
        self._is_dirty = True
        self._refresh_kpt_panel()
        self._refresh_frame()

    def _on_kpt_listbox_select(self, event=None):
        sel = self.kpt_listbox.curselection()
        if not sel:
            return
        self._selected_kpt = sel[0]
        self._selected_person = self._person_var.get()
        self._refresh_frame()

    def _on_kpt_list_right_click(self, event):
        """Right-click in listbox to cycle visibility of selected keypoint.
        Visibility holds forward from this frame until another vis override."""
        if not self._edit_mode:
            return
        idx = self.kpt_listbox.nearest(event.y)
        if idx < 0:
            return
        self._push_undo()
        self._selected_kpt = idx
        self.kpt_listbox.selection_clear(0, tk.END)
        self.kpt_listbox.selection_set(idx)
        p, k = self._person_var.get(), idx
        frame = self.current_frame
        _, _, curr_vis = self._get_effective_kpt(p, k, frame)
        new_vis = (curr_vis + 1) % 3
        key = (p, k)
        if key not in self._vis_overrides:
            self._vis_overrides[key] = {}
        self._vis_overrides[key][frame] = new_vis
        self._is_dirty = True
        self._refresh_kpt_panel()
        self._refresh_frame()

    def _mark_keyframe(self):
        if not self._edit_mode or self._selected_kpt is None:
            return
        self._push_undo()
        p, k = self._person_var.get(), self._selected_kpt
        frame = self.current_frame
        x, y, vis = self._get_effective_kpt(p, k, frame)
        key = (p, k)
        if key not in self._keyframes:
            self._keyframes[key] = {}
        self._keyframes[key][frame] = (x, y, vis)
        self._is_dirty = True
        self._refresh_kpt_panel()

    def _clear_keyframe(self):
        if not self._edit_mode or self._selected_kpt is None:
            return
        self._push_undo()
        p, k = self._person_var.get(), self._selected_kpt
        frame = self.current_frame
        key = (p, k)
        if key in self._keyframes and frame in self._keyframes[key]:
            del self._keyframes[key][frame]
            if not self._keyframes[key]:
                del self._keyframes[key]
            self._is_dirty = True
        self._refresh_kpt_panel()
        self._refresh_frame()

    def _refresh_kpt_panel(self):
        """Repopulate the keypoint listbox for the current frame."""
        if self.pose_data is None:
            return
        self.kpt_listbox.delete(0, tk.END)
        kpt_names = self.cfg.pose.keypoint_names
        p = self._person_var.get()
        frame = self.current_frame
        _vis_labels = {0: "✕", 1: "●", 2: "○"}
        _vis_colors = {0: "#cc241d", 1: "#b8bb26", 2: "#458588"}
        for k, name in enumerate(kpt_names):
            _, _, vis = self._get_effective_kpt(p, k, frame)
            key = (p, k)
            is_kf = key in self._keyframes and frame in self._keyframes[key]
            kf_mark = " ⬥" if is_kf else "  "
            self.kpt_listbox.insert(tk.END, f"{_vis_labels[vis]}{kf_mark} {name}")
            self.kpt_listbox.itemconfig(tk.END, fg=_vis_colors.get(vis, "#d4d4d4"))
        if self._selected_kpt is not None:
            self.kpt_listbox.selection_clear(0, tk.END)
            self.kpt_listbox.selection_set(self._selected_kpt)
            self.kpt_listbox.see(self._selected_kpt)
        # Update keyframe button state
        if self._selected_kpt is not None:
            key = (self._person_var.get(), self._selected_kpt)
            has_kf = key in self._keyframes and frame in self._keyframes[key]
            self.btn_clear_kf.config(
                state=tk.NORMAL if has_kf else tk.DISABLED,
                bg="#6b1a1a" if has_kf else "#3c3c3c",
            )

    # ── Undo ──────────────────────────────────────────────────────────────

    def _push_undo(self):
        """Snapshot current edit state onto the undo stack."""
        if self._edit_pose is None:
            return
        import copy
        if len(self._undo_stack) >= 20:
            self._undo_stack.pop(0)
        self._undo_stack.append((
            self._edit_pose.copy(),
            copy.deepcopy(self._keyframes),
            copy.deepcopy(self._vis_overrides),
        ))

    def _undo(self):
        """Ctrl+Z: restore previous edit state."""
        if not self._undo_stack or not self._edit_mode:
            return
        entry = self._undo_stack.pop()
        if len(entry) == 3:
            self._edit_pose, self._keyframes, self._vis_overrides = entry
        else:
            self._edit_pose, self._keyframes = entry  # backward compat
        self._is_dirty = True
        self._refresh_kpt_panel()
        self._refresh_frame()

    # ── Auto save + file save ─────────────────────────────────────────────

    def _schedule_auto_save(self):
        self.root.after(10_000, self._auto_save)

    def _auto_save(self):
        if self._is_dirty and self._edit_mode:
            self._save_edits()
        self.root.after(10_000, self._auto_save)

    def _save_edits(self):
        """Materialise keyframe interpolation into _edit_pose and save to disk."""
        if self._edit_pose is None or self.current_scene is None:
            return
        arr = self._edit_pose.copy()
        conf_vals = {0: 0.0, 1: 0.9, 2: 3.0}  # 3.0 = occluded sentinel
        # Apply all keyframe interpolations
        for (p, k), kf_dict in self._keyframes.items():
            if not kf_dict:
                continue
            n_frames = arr.shape[0]
            for fi in range(n_frames):
                x, y, vis = self._get_effective_kpt(p, k, fi)
                arr[fi, p, k, 0] = x
                arr[fi, p, k, 1] = y
                arr[fi, p, k, 2] = conf_vals.get(vis, float(self._edit_pose[fi, p, k, 2]))
        # Apply visibility overrides (step function: hold forward from set frame)
        for (p, k), overrides in self._vis_overrides.items():
            if not overrides:
                continue
            sorted_override_frames = sorted(overrides.keys())
            n_frames = arr.shape[0]
            for fi in range(n_frames):
                before = [f for f in sorted_override_frames if f <= fi]
                if before:
                    arr[fi, p, k, 2] = conf_vals[overrides[max(before)]]
        # Write to file using the canonical new layout
        from src.data.curation import keypoints_path
        scene_dir = self.processed_dir / self.current_scene
        save_path = keypoints_path(scene_dir, self.cfg.pose.model_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Close any existing memcpy-backed mmap before overwriting on Windows.
        if self.pose_data is not None and hasattr(self.pose_data, "_mmap"):
            try:
                self.pose_data._mmap.close()
            except Exception:
                pass
        self.pose_data = None

        temp_path = save_path.with_suffix(".tmp.npy")
        np.save(str(temp_path), arr)
        temp_path.replace(save_path)

        # Reload mmap
        self._edit_pose = arr
        self.pose_data = np.load(str(save_path), mmap_mode="r")
        self._is_dirty = False
        import logging
        logging.getLogger(__name__).info("Auto-saved keypoints to %s", save_path)

    # ── Export ────────────────────────────────────────────────────────────

    def _export_pose_dataset(self):
        """Export current scene keypoints as a YOLO pose dataset."""
        if self.current_scene is None or self.pose_data is None:
            return
        out_dir = self.data_dir / "yolokpdataset" / self.current_scene
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "images").mkdir(exist_ok=True)
        (out_dir / "labels").mkdir(exist_ok=True)

        conf_thresh = self._kpt_conf_threshold
        pose = self._edit_pose if self._edit_mode and self._edit_pose is not None else self.pose_data
        n_frames = pose.shape[0]
        n_kpt = pose.shape[2]
        scene_id = self.current_scene
        exported = 0

        for fi in range(n_frames):
            # Export primary person (person 0) only
            p = 0
            kpts = np.array([self._get_effective_kpt(p, k, fi) for k in range(n_kpt)],
                            dtype=np.float32)
            # kpts: [n_kpt, 3] — x, y, vis (already threshold-applied via _get_effective_kpt)

            visible = kpts[kpts[:, 2] > 0]
            if len(visible) == 0:
                continue  # skip frames with no visible keypoints

            # Bounding box from visible keypoints
            xs, ys = visible[:, 0], visible[:, 1]
            pad = 0.03
            x1, y1 = max(0.0, xs.min() - pad), max(0.0, ys.min() - pad)
            x2, y2 = min(1.0, xs.max() + pad), min(1.0, ys.max() + pad)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = x2 - x1, y2 - y1

            # Save video frame as image
            img = self._get_video_frame(fi)
            if img is None:
                img = np.zeros((640, 640, 3), dtype=np.uint8)
            img_name = f"{scene_id}_{fi:05d}.jpg"
            cv2.imwrite(str(out_dir / "images" / img_name), img)

            # YOLO pose label: class cx cy w h  kx ky v  ...
            parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
            for k in range(n_kpt):
                kx, ky, kv = kpts[k]
                parts += [f"{kx:.6f}", f"{ky:.6f}", str(int(kv))]

            lbl_name = f"{scene_id}_{fi:05d}.txt"
            (out_dir / "labels" / lbl_name).write_text(" ".join(parts) + "\n")
            exported += 1

        # Write dataset.yaml
        kpt_names = self.cfg.pose.keypoint_names
        yaml_content = (
            f"path: {out_dir}\n"
            f"train: images\n"
            f"val: images\n"
            f"nc: 1\n"
            f"names: ['person']\n"
            f"kpt_shape: [{n_kpt}, 3]\n"
            f"skeleton: {self.cfg.pose.keypoint_bones}\n"
        )
        (out_dir / "dataset.yaml").write_text(yaml_content)

        import tkinter.messagebox as mb
        mb.showinfo("Export Complete",
                    f"Exported {exported} frames to:\n{out_dir}\n\n"
                    f"dataset.yaml written with {n_kpt} keypoints.")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize processed scene data")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to the data/ directory (default: data/)")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Config YAML path (default: configs/default.yaml)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    cfg = Config.from_yaml(str(config_path)) if config_path.exists() else Config()

    root = tk.Tk()
    root.minsize(900, 560)
    _sw = root.winfo_screenwidth()
    _sh = root.winfo_screenheight()
    root.maxsize(_sw, _sh - 46)

    app = DataVisualizer(root, data_dir, cfg)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
