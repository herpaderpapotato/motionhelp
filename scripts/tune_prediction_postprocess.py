"""Interactive tuner for wave-aware prediction postprocessing.

This tool is meant for dense, unsimplified funscript outputs written by
scripts/predict_disposition.py. It lets you inspect the raw script, tweak the
full WavePostprocessConfig surface, and see the postprocessed curve update live.
Optional synchronized video playback is supported when a matching video is
available or selected manually.
"""

from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageTk

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.funscript import load_funscript, save_funscript
from src.data.prediction_postprocess import WavePostprocessConfig, postprocess_predictions


DEFAULT_POSTPROCESS = WavePostprocessConfig()
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v")


@dataclass(frozen=True)
class ControlSpec:
    name: str
    label: str
    kind: str
    group: str
    minimum: float | int | None = None
    maximum: float | int | None = None
    choices: tuple[str, ...] = ()


CONTROL_SPECS: tuple[ControlSpec, ...] = (
    ControlSpec("lowpass_cutoff_hz", "Lowpass cutoff (Hz)", "float", "Primary", minimum=0.0),
    ControlSpec("trough_prominence", "Trough prominence", "float", "Primary", minimum=0.0),
    ControlSpec("trough_distance_seconds", "Trough distance (s)", "float", "Primary", minimum=0.0),
    ControlSpec("min_cycle_amplitude", "Min cycle amplitude", "float", "Primary", minimum=0.0),
    ControlSpec("min_cycle_frequency_hz", "Min cycle frequency (Hz)", "float", "Primary", minimum=0.0),
    ControlSpec(
        "gradient_smoothing",
        "Gradient smoothing",
        "choice",
        "Primary",
        choices=("none", "savgol", "gaussian"),
    ),
    ControlSpec("gradient_window_seconds", "Gradient window (s)", "float", "Primary", minimum=0.0),
    ControlSpec("min_gradient_length_seconds", "Min gradient length (s)", "float", "Primary", minimum=0.0),
    ControlSpec("min_gradient_range", "Min gradient range", "float", "Primary", minimum=0.0),
    ControlSpec(
        "stretch_mode",
        "Stretch mode",
        "choice",
        "Primary",
        choices=("linear", "tanh"),
    ),
    ControlSpec("stretch_gain", "Stretch gain", "float", "Primary", minimum=0.0),
    ControlSpec("chunk_seconds", "Chunk size (s, 0=off)", "float", "Primary", minimum=0.0),
    ControlSpec("lowpass_order", "Lowpass order", "int", "Advanced", minimum=1),
    ControlSpec("gradient_polyorder", "Gradient polyorder", "int", "Advanced", minimum=1),
    ControlSpec("gradient_sigma_seconds", "Gradient sigma (s)", "float", "Advanced", minimum=0.0),
    ControlSpec("gradient_floor", "Gradient floor", "float", "Advanced", minimum=0.0),
    ControlSpec("gradient_low_quantile", "Gradient low quantile", "float", "Advanced", minimum=0.0, maximum=1.0),
    ControlSpec("gradient_high_quantile", "Gradient high quantile", "float", "Advanced", minimum=0.0, maximum=1.0),
    ControlSpec(
        "min_gradient_sign_balance",
        "Min gradient sign balance",
        "float",
        "Advanced",
        minimum=0.0,
        maximum=1.0,
    ),
    ControlSpec("merge_gap_seconds", "Merge gap (s)", "float", "Advanced", minimum=0.0),
    ControlSpec("chunk_overlap_seconds", "Chunk overlap (s)", "float", "Advanced", minimum=0.0),
)

SPEC_BY_NAME = {spec.name: spec for spec in CONTROL_SPECS}


def _is_plain_entry(widget: tk.Widget | None) -> bool:
    return isinstance(widget, tk.Entry) and not isinstance(widget, ttk.Combobox)


def fit_image(frame_bgr: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    scale = min(max_w / max(float(w), 1.0), max_h / max(float(h), 1.0), 1.0)
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame_bgr


def estimate_fps_from_actions(actions: list[dict]) -> float:
    if len(actions) < 2:
        raise ValueError("Need at least two actions to infer fps; set it manually")

    timestamps_ms = np.asarray([int(action["at"]) for action in actions], dtype=np.float64)
    total_delta_ms = float(timestamps_ms[-1] - timestamps_ms[0])
    if total_delta_ms > 0.0:
        return 1000.0 * float(len(actions) - 1) / total_delta_ms

    diffs_ms = np.diff(timestamps_ms)
    positive_diffs = diffs_ms[diffs_ms > 0.0]
    if positive_diffs.size == 0:
        raise ValueError("Could not infer fps from action timestamps; set it manually")
    return 1000.0 / float(np.mean(positive_diffs))


def _candidate_stems(stem: str) -> list[str]:
    candidates = [stem]
    for suffix in (".raw", ".post", "_raw", "_post", "-raw", "-post"):
        if stem.endswith(suffix):
            candidates.append(stem[: -len(suffix)])
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return ordered


def auto_discover_video(script_path: Path, script_data: dict, data_dir: Path | None) -> Path | None:
    metadata = script_data.get("metadata", {}) if isinstance(script_data, dict) else {}
    candidates: list[Path] = []

    for key in ("source_video", "video_path"):
        value = metadata.get(key)
        if value:
            candidates.append(Path(value))

    for stem in _candidate_stems(script_path.stem):
        for extension in VIDEO_EXTENSIONS:
            candidates.append(script_path.with_name(f"{stem}{extension}"))

    if data_dir is not None:
        preprocessed_dir = data_dir / "preprocessed"
        for stem in _candidate_stems(script_path.stem):
            for extension in VIDEO_EXTENSIONS:
                candidates.append(preprocessed_dir / f"{stem}{extension}")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


class PostprocessTunerApp:
    def __init__(self, root: tk.Tk, data_dir: Path | None = None):
        self.root = root
        self.data_dir = data_dir

        self.input_path: Path | None = None
        self.video_path: Path | None = None
        self.script_data: dict = {}
        self.timestamps_ms = np.zeros((0,), dtype=np.int64)
        self.time_axis_seconds = np.zeros((0,), dtype=np.float32)
        self.raw_positions = np.zeros((0,), dtype=np.float32)
        self.processed_positions = np.zeros((0,), dtype=np.float32)
        self.inferred_script_fps = 0.0
        self.script_fps = 0.0
        self.current_config = WavePostprocessConfig()
        self.current_stats: dict[str, object] = {}

        self.cap: cv2.VideoCapture | None = None
        self.video_fps = 0.0
        self.video_frame_count = 0
        self.current_video_frame = -1
        self.video_photo: ImageTk.PhotoImage | None = None
        self.video_display_size = (640, 360)

        self.current_time_seconds = 0.0
        self._playing = False
        self._play_after_id: str | None = None
        self._reprocess_after_id: str | None = None
        self._suspend_reprocess = False

        self.script_fps_var = tk.StringVar(value="")
        self.speed_var = tk.StringVar(value="1.0")
        self.scrubber_var = tk.DoubleVar(value=0.0)
        self.file_info_var = tk.StringVar(value="No script loaded")
        self.video_info_var = tk.StringVar(value="No video loaded")
        self.time_info_var = tk.StringVar(value="t=0.000s")
        self.position_info_var = tk.StringVar(value="raw=--  post=--")
        self.status_var = tk.StringVar(value="Load a dense funscript to begin")

        self.control_vars: dict[str, tk.StringVar] = {}
        self.control_widgets: dict[str, tk.Widget] = {}
        self.entry_normal_bg = "#2d2d30"
        self.entry_invalid_bg = "#5a1a1a"

        self._build_ui()
        self._set_control_values(DEFAULT_POSTPROCESS)
        self.script_fps_var.trace_add("write", self._on_setting_changed)

    @property
    def total_duration_seconds(self) -> float:
        duration = 0.0
        if self.time_axis_seconds.size:
            duration = float(self.time_axis_seconds[-1])
        if self.video_fps > 0.0 and self.video_frame_count > 0:
            duration = max(duration, max(0.0, float(self.video_frame_count - 1) / self.video_fps))
        return duration

    def _build_ui(self) -> None:
        self.root.title("Wave Postprocess Tuner")
        self.root.configure(bg="#1e1e1e")
        self.root.minsize(1320, 860)

        top = tk.Frame(self.root, bg="#1e1e1e")
        top.pack(fill=tk.X, padx=8, pady=8)

        button_style = {
            "bg": "#3c3c3c",
            "fg": "#d4d4d4",
            "activebackground": "#505050",
            "activeforeground": "#ffffff",
            "relief": tk.FLAT,
            "padx": 8,
            "pady": 4,
        }
        tk.Button(top, text="Load Script", command=self._load_script_dialog, **button_style).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(top, text="Load Video", command=self._load_video_dialog, **button_style).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Auto Match Video", command=self._auto_match_video, **button_style).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Reset Defaults", command=self._reset_defaults, **button_style).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Save Postprocessed", command=self._save_processed_script, **button_style).pack(side=tk.LEFT, padx=4)

        info = tk.Frame(self.root, bg="#1e1e1e")
        info.pack(fill=tk.X, padx=8)
        tk.Label(info, textvariable=self.file_info_var, bg="#1e1e1e", fg="#9cdcfe", anchor="w").pack(fill=tk.X)
        tk.Label(info, textvariable=self.video_info_var, bg="#1e1e1e", fg="#ce9178", anchor="w").pack(fill=tk.X)

        body = tk.Frame(self.root, bg="#1e1e1e")
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 8))

        main = tk.Frame(body, bg="#1e1e1e")
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(body, bg="#252526", width=400)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        sidebar.pack_propagate(False)

        video_frame = tk.LabelFrame(main, text="Video", bg="#1e1e1e", fg="#d4d4d4", bd=1, relief=tk.GROOVE)
        video_frame.pack(fill=tk.BOTH)
        self.video_surface = tk.Frame(video_frame, bg="#111111", height=self.video_display_size[1] + 16)
        self.video_surface.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.video_surface.pack_propagate(False)
        self.video_surface.bind("<Configure>", self._on_video_surface_resize)
        self.video_label = tk.Label(
            self.video_surface,
            text="Script-only mode\nLoad a video for synchronized playback",
            bg="#111111",
            fg="#808080",
            anchor=tk.CENTER,
            justify=tk.CENTER,
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        controls = tk.Frame(main, bg="#1e1e1e")
        controls.pack(fill=tk.X, pady=(8, 0))
        tk.Button(controls, text="<<", command=self._step_back_10, **button_style).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(controls, text="<", command=self._step_back, **button_style).pack(side=tk.LEFT, padx=4)
        self.play_button = tk.Button(controls, text="Play", command=self._toggle_playback, **button_style)
        self.play_button.pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text=">", command=self._step_forward, **button_style).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text=">>", command=self._step_forward_10, **button_style).pack(side=tk.LEFT, padx=4)

        tk.Label(controls, text="Speed", bg="#1e1e1e", fg="#d4d4d4").pack(side=tk.LEFT, padx=(16, 4))
        speed_box = ttk.Combobox(controls, textvariable=self.speed_var, values=("0.25", "0.5", "1.0", "1.5", "2.0"), width=6, state="readonly")
        speed_box.pack(side=tk.LEFT)

        tk.Label(controls, textvariable=self.time_info_var, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 10)).pack(side=tk.LEFT, padx=(16, 8))
        tk.Label(controls, textvariable=self.position_info_var, bg="#1e1e1e", fg="#4ec9b0", font=("Consolas", 10, "bold")).pack(side=tk.LEFT)

        self.scrubber = ttk.Scale(main, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.scrubber_var, command=self._on_scrub)
        self.scrubber.pack(fill=tk.X, pady=(8, 8))

        plot_frame = tk.LabelFrame(main, text="Raw vs Postprocessed", bg="#1e1e1e", fg="#d4d4d4", bd=1, relief=tk.GROOVE)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(11, 7), facecolor="#1e1e1e")
        grid = self.figure.add_gridspec(2, 1, hspace=0.18)
        self.ax_raw = self.figure.add_subplot(grid[0])
        self.ax_processed = self.figure.add_subplot(grid[1], sharex=self.ax_raw)
        for axis in (self.ax_raw, self.ax_processed):
            axis.set_facecolor("#1e1e1e")
            axis.grid(True, alpha=0.22, color="#505050")
            axis.tick_params(colors="#a0a0a0")
            for spine in axis.spines.values():
                spine.set_edgecolor("#4a4a4a")
            axis.set_ylim(-0.05, 1.05)
        self.ax_raw.set_ylabel("Raw")
        self.ax_processed.set_ylabel("Post")
        self.ax_processed.set_xlabel("Time (s)")
        self.ax_raw.set_title("Original dense script")
        self.ax_processed.set_title("Postprocessed output")

        self.raw_line, = self.ax_raw.plot([], [], color="#4ec9b0", lw=1.0, marker="o", markersize=2, alpha=0.9)
        self.processed_raw_ghost, = self.ax_processed.plot([], [], color="#7f848e", lw=0.8, alpha=0.35)
        self.processed_line, = self.ax_processed.plot([], [], color="#d19a66", lw=1.15, marker="o", markersize=2, alpha=0.95)
        self.raw_cursor = self.ax_raw.axvline(0.0, color="#ffffff", lw=0.9, alpha=0.55)
        self.processed_cursor = self.ax_processed.axvline(0.0, color="#ffffff", lw=0.9, alpha=0.55)
        self.raw_marker, = self.ax_raw.plot([], [], marker="o", markersize=7, color="#9cdcfe", linestyle="None")
        self.processed_marker, = self.ax_processed.plot([], [], marker="o", markersize=7, color="#ffd580", linestyle="None")

        self.figure_canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.figure_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.figure_canvas.mpl_connect("button_press_event", self._on_plot_click)

        status = tk.Label(
            main,
            textvariable=self.status_var,
            bg="#1e1e1e",
            fg="#c5c5c5",
            anchor="w",
            justify=tk.LEFT,
        )
        status.pack(fill=tk.X, pady=(8, 0))

        self._build_sidebar(sidebar)

        self.root.bind("<space>", lambda _event: self._toggle_playback())
        self.root.bind("<Left>", lambda _event: self._step_back())
        self.root.bind("<Right>", lambda _event: self._step_forward())
        self.root.bind("<Shift-Left>", lambda _event: self._step_back_10())
        self.root.bind("<Shift-Right>", lambda _event: self._step_forward_10())

    def _build_sidebar(self, parent: tk.Frame) -> None:
        canvas = tk.Canvas(parent, bg="#252526", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        inner = tk.Frame(canvas, bg="#252526")
        canvas.create_window((0, 0), window=inner, anchor="nw")

        inner.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(1, width=event.width))

        input_frame = tk.LabelFrame(inner, text="Input", bg="#252526", fg="#d4d4d4", bd=1, relief=tk.GROOVE)
        input_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        tk.Label(input_frame, text="Script fps", bg="#252526", fg="#d4d4d4", anchor="w").pack(fill=tk.X, padx=8, pady=(8, 2))
        self.script_fps_entry = tk.Entry(
            input_frame,
            textvariable=self.script_fps_var,
            bg=self.entry_normal_bg,
            fg="#f0f0f0",
            insertbackground="#ffffff",
            relief=tk.FLAT,
        )
        self.script_fps_entry.pack(fill=tk.X, padx=8)
        tk.Button(input_frame, text="Use inferred fps", command=self._use_inferred_fps, bg="#3c3c3c", fg="#d4d4d4", activebackground="#505050", relief=tk.FLAT).pack(fill=tk.X, padx=8, pady=(6, 8))

        for group in ("Primary", "Advanced"):
            group_frame = tk.LabelFrame(inner, text=group, bg="#252526", fg="#d4d4d4", bd=1, relief=tk.GROOVE)
            group_frame.pack(fill=tk.X, padx=8, pady=4)
            for spec in CONTROL_SPECS:
                if spec.group != group:
                    continue
                self._build_control(group_frame, spec)

        summary_frame = tk.LabelFrame(inner, text="Summary", bg="#252526", fg="#d4d4d4", bd=1, relief=tk.GROOVE)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))
        self.summary_text = tk.Text(
            summary_frame,
            height=20,
            wrap="word",
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#ffffff",
            relief=tk.FLAT,
        )
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.summary_text.configure(state=tk.DISABLED)

    def _build_control(self, parent: tk.LabelFrame, spec: ControlSpec) -> None:
        row = tk.Frame(parent, bg="#252526")
        row.pack(fill=tk.X, padx=8, pady=4)

        tk.Label(row, text=spec.label, bg="#252526", fg="#d4d4d4", anchor="w").pack(fill=tk.X)
        var = tk.StringVar(value=str(getattr(DEFAULT_POSTPROCESS, spec.name)))
        var.trace_add("write", self._on_setting_changed)
        self.control_vars[spec.name] = var

        if spec.kind == "choice":
            widget = ttk.Combobox(row, textvariable=var, values=spec.choices, state="readonly")
            widget.pack(fill=tk.X, pady=(2, 0))
        else:
            widget = tk.Entry(
                row,
                textvariable=var,
                bg=self.entry_normal_bg,
                fg="#f0f0f0",
                insertbackground="#ffffff",
                relief=tk.FLAT,
            )
            widget.pack(fill=tk.X, pady=(2, 0))
        self.control_widgets[spec.name] = widget

    def _load_script_dialog(self) -> None:
        path = filedialog.askopenfilename(
            title="Select dense funscript",
            filetypes=[("Funscript", "*.funscript"), ("JSON", "*.json"), ("All files", "*.*")],
        )
        if path:
            try:
                self.load_script(Path(path))
            except Exception as exc:
                self.status_var.set(f"Failed to load script: {exc}")
                messagebox.showerror("Load Script", str(exc))

    def _load_video_dialog(self) -> None:
        path = filedialog.askopenfilename(
            title="Select source video",
            filetypes=[("Video", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v"), ("All files", "*.*")],
        )
        if path:
            try:
                self.load_video(Path(path))
            except Exception as exc:
                self.status_var.set(f"Failed to load video: {exc}")
                messagebox.showerror("Load Video", str(exc))

    def _auto_match_video(self) -> None:
        if self.input_path is None:
            messagebox.showinfo("Auto Match Video", "Load a funscript first.")
            return
        candidate = auto_discover_video(self.input_path, self.script_data, self.data_dir)
        if candidate is None:
            messagebox.showinfo("Auto Match Video", "No matching video was found. Load one manually if needed.")
            return
        try:
            self.load_video(candidate)
        except Exception as exc:
            self.status_var.set(f"Failed to load auto-matched video: {exc}")
            messagebox.showerror("Auto Match Video", str(exc))

    def _reset_defaults(self) -> None:
        self._set_control_values(DEFAULT_POSTPROCESS)
        if self.inferred_script_fps > 0.0 and not self.script_fps_var.get().strip():
            self.script_fps_var.set(f"{self.inferred_script_fps:.6f}")
        self._schedule_reprocess()

    def _use_inferred_fps(self) -> None:
        if self.inferred_script_fps <= 0.0:
            return
        self.script_fps_var.set(f"{self.inferred_script_fps:.6f}")

    def _set_control_values(self, config: WavePostprocessConfig) -> None:
        self._suspend_reprocess = True
        try:
            for spec in CONTROL_SPECS:
                value = getattr(config, spec.name)
                if value is None:
                    text = "0"
                else:
                    text = str(value)
                self.control_vars[spec.name].set(text)
                widget = self.control_widgets.get(spec.name)
                if _is_plain_entry(widget):
                    widget.configure(bg=self.entry_normal_bg)
        finally:
            self._suspend_reprocess = False

    def _on_setting_changed(self, *_args) -> None:
        if self._suspend_reprocess:
            return
        self._schedule_reprocess()

    def _schedule_reprocess(self) -> None:
        if self.raw_positions.size == 0:
            return
        if self._reprocess_after_id is not None:
            self.root.after_cancel(self._reprocess_after_id)
        self._reprocess_after_id = self.root.after(220, self._reprocess_now)

    def _reprocess_now(self) -> None:
        self._reprocess_after_id = None
        if self.raw_positions.size == 0:
            return

        try:
            script_fps = self._read_script_fps()
            config = self._read_config()
            processed, stats = postprocess_predictions(self.raw_positions, script_fps, config, return_stats=True)
        except Exception as exc:
            self.status_var.set(f"Postprocess settings are not valid yet: {exc}")
            return

        self.script_fps = script_fps
        self.current_config = config
        self.current_stats = stats
        self.processed_positions = processed
        self._update_plot_series()
        self._update_summary()
        self._update_current_view()
        self.status_var.set(
            "Postprocess updated live. Advanced knobs available here: lowpass order, quantiles, sign balance, merge gap, and chunk overlap."
        )

    def _read_script_fps(self) -> float:
        text = self.script_fps_var.get().strip()
        if not text:
            raise ValueError("script fps is required")
        try:
            value = float(text)
        except ValueError as exc:
            self.script_fps_entry.configure(bg=self.entry_invalid_bg)
            raise ValueError("script fps must be a number") from exc
        if value <= 0.0:
            self.script_fps_entry.configure(bg=self.entry_invalid_bg)
            raise ValueError("script fps must be greater than zero")
        self.script_fps_entry.configure(bg=self.entry_normal_bg)
        return value

    def _read_config(self) -> WavePostprocessConfig:
        values: dict[str, float | int | str | None] = {}
        for spec in CONTROL_SPECS:
            widget = self.control_widgets.get(spec.name)
            if _is_plain_entry(widget):
                widget.configure(bg=self.entry_normal_bg)
            text = self.control_vars[spec.name].get().strip()
            try:
                if spec.kind == "choice":
                    if text not in spec.choices:
                        raise ValueError(f"must be one of {', '.join(spec.choices)}")
                    value: float | int | str | None = text
                elif spec.kind == "int":
                    value = int(float(text))
                else:
                    number = float(text)
                    value = None if spec.name == "chunk_seconds" and number <= 0.0 else number
            except ValueError as exc:
                if _is_plain_entry(widget):
                    widget.configure(bg=self.entry_invalid_bg)
                raise ValueError(f"{spec.label}: invalid value") from exc

            if value is not None and spec.minimum is not None and value < spec.minimum:
                if _is_plain_entry(widget):
                    widget.configure(bg=self.entry_invalid_bg)
                raise ValueError(f"{spec.label}: must be >= {spec.minimum}")
            if value is not None and spec.maximum is not None and value > spec.maximum:
                if _is_plain_entry(widget):
                    widget.configure(bg=self.entry_invalid_bg)
                raise ValueError(f"{spec.label}: must be <= {spec.maximum}")
            values[spec.name] = value

        return WavePostprocessConfig(**values)

    def load_script(self, path: Path, fps_override: float | None = None) -> None:
        data = load_funscript(path)
        actions = sorted(data.get("actions", []), key=lambda action: action["at"])
        if not actions:
            raise ValueError(f"No actions found in {path}")

        self._stop_playback()
        self.input_path = path
        self.script_data = data
        self.timestamps_ms = np.asarray([int(action["at"]) for action in actions], dtype=np.int64)
        self.time_axis_seconds = self.timestamps_ms.astype(np.float32) / 1000.0
        self.raw_positions = np.asarray([float(action["pos"]) for action in actions], dtype=np.float32) / 100.0
        self.inferred_script_fps = float(fps_override) if fps_override is not None else self._infer_script_fps(data, actions)

        self.script_fps_var.set(f"{self.inferred_script_fps:.6f}")
        self.file_info_var.set(
            f"Script: {path} | actions={len(actions)} | duration={self.total_duration_seconds:.3f}s | inferred fps={self.inferred_script_fps:.6f}"
        )

        candidate_video = auto_discover_video(path, data, self.data_dir)
        if candidate_video is not None:
            self.load_video(candidate_video)
        else:
            self.clear_video()
            self.video_info_var.set("Video: none (script-only mode)")

        self._sync_scrubber_range()
        self._reprocess_now()
        self._set_current_time(0.0)

    def _infer_script_fps(self, data: dict, actions: list[dict]) -> float:
        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        for key in ("postprocess_fps", "output_fps", "fps"):
            value = metadata.get(key)
            if value is not None:
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                if parsed > 0.0:
                    return parsed
        return estimate_fps_from_actions(actions)

    def load_video(self, path: Path) -> None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Could not open video: {path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.clear_video()
        self.cap = cap
        self.video_path = path
        self.video_fps = fps
        self.video_frame_count = frame_count
        self.current_video_frame = -1
        duration = max(0.0, float(frame_count - 1) / fps) if frame_count > 0 else 0.0
        self.video_info_var.set(f"Video: {path} | fps={fps:.6f} | frames={frame_count} | duration={duration:.3f}s")
        self._sync_scrubber_range()
        self._update_current_view()

    def clear_video(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.video_path = None
        self.video_fps = 0.0
        self.video_frame_count = 0
        self.current_video_frame = -1
        self.video_photo = None
        self.video_label.configure(image="", text="Script-only mode\nLoad a video for synchronized playback")

    def _sync_scrubber_range(self) -> None:
        self.scrubber.configure(to=max(1e-6, self.total_duration_seconds))

    def _update_plot_series(self) -> None:
        duration = max(1.0, self.total_duration_seconds)
        if self.time_axis_seconds.size:
            self.raw_line.set_data(self.time_axis_seconds, self.raw_positions)
            self.processed_raw_ghost.set_data(self.time_axis_seconds, self.raw_positions)
            self.processed_line.set_data(self.time_axis_seconds, self.processed_positions)
        else:
            self.raw_line.set_data([], [])
            self.processed_raw_ghost.set_data([], [])
            self.processed_line.set_data([], [])
        self.ax_raw.set_xlim(0.0, duration)
        self.ax_processed.set_xlim(0.0, duration)
        self.figure_canvas.draw_idle()

    def _on_video_surface_resize(self, event) -> None:
        width = max(160, int(event.width) - 8)
        height = max(120, int(event.height) - 8)
        if (width, height) == self.video_display_size:
            return
        self.video_display_size = (width, height)
        if self.cap is not None:
            self._update_current_view()

    def _sample_series(self, series: np.ndarray, time_seconds: float) -> float:
        if self.time_axis_seconds.size == 0 or series.size == 0:
            return 0.0
        return float(np.interp(time_seconds, self.time_axis_seconds, series, left=series[0], right=series[-1]))

    def _update_current_view(self) -> None:
        raw_value = self._sample_series(self.raw_positions, self.current_time_seconds)
        processed_value = self._sample_series(self.processed_positions, self.current_time_seconds)

        self.time_info_var.set(f"t={self.current_time_seconds:.3f}s")
        self.position_info_var.set(f"raw={raw_value:0.4f}  post={processed_value:0.4f}")

        self.raw_cursor.set_xdata([self.current_time_seconds, self.current_time_seconds])
        self.processed_cursor.set_xdata([self.current_time_seconds, self.current_time_seconds])
        self.raw_marker.set_data([self.current_time_seconds], [raw_value])
        self.processed_marker.set_data([self.current_time_seconds], [processed_value])
        self.figure_canvas.draw_idle()
        self._refresh_video_frame(raw_value, processed_value)

    def _refresh_video_frame(self, raw_value: float, processed_value: float) -> None:
        if self.cap is None or self.video_fps <= 0.0 or self.video_frame_count <= 0:
            self.video_label.configure(image="", text="Script-only mode\nLoad a video for synchronized playback")
            return

        target_frame = int(round(self.current_time_seconds * self.video_fps))
        target_frame = max(0, min(self.video_frame_count - 1, target_frame))
        if target_frame == self.current_video_frame and self.video_photo is not None:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            return

        self.current_video_frame = target_frame
        annotated = self._annotate_frame(frame_bgr, raw_value, processed_value)
        fitted = fit_image(annotated, self.video_display_size[0], self.video_display_size[1])
        frame_rgb = cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        self.video_photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.video_photo, text="")

    def _annotate_frame(self, frame_bgr: np.ndarray, raw_value: float, processed_value: float) -> np.ndarray:
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]
        box_w = max(140, int(w * 0.22))
        box_h = 54
        x0 = 12
        y0 = max(12, h - box_h - 16)
        cv2.rectangle(annotated, (x0, y0), (x0 + box_w, y0 + box_h), (24, 24, 24), thickness=-1)
        cv2.rectangle(annotated, (x0, y0), (x0 + box_w, y0 + box_h), (96, 96, 96), thickness=1)
        raw_w = int(round((box_w - 24) * np.clip(raw_value, 0.0, 1.0)))
        processed_w = int(round((box_w - 24) * np.clip(processed_value, 0.0, 1.0)))
        cv2.putText(annotated, f"raw  {raw_value:0.3f}", (x0 + 8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"post {processed_value:0.3f}", (x0 + 8, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.rectangle(annotated, (x0 + 64, y0 + 9), (x0 + 64 + raw_w, y0 + 16), (176, 188, 38), thickness=-1)
        cv2.rectangle(annotated, (x0 + 64, y0 + 31), (x0 + 64 + processed_w, y0 + 38), (77, 166, 255), thickness=-1)
        return annotated

    def _on_plot_click(self, event) -> None:
        if event.xdata is None:
            return
        self._set_current_time(float(event.xdata))

    def _on_scrub(self, value: str) -> None:
        try:
            time_seconds = float(value)
        except ValueError:
            return
        self._set_current_time(time_seconds, update_scrubber=False)

    def _set_current_time(self, time_seconds: float, update_scrubber: bool = True) -> None:
        clamped = max(0.0, min(self.total_duration_seconds, time_seconds))
        self.current_time_seconds = clamped
        if update_scrubber:
            self.scrubber_var.set(clamped)
        self._update_current_view()

    def _frame_step_seconds(self) -> float:
        if self.video_fps > 0.0:
            return 1.0 / self.video_fps
        if self.script_fps > 0.0:
            return 1.0 / self.script_fps
        if self.inferred_script_fps > 0.0:
            return 1.0 / self.inferred_script_fps
        return 1.0 / 30.0

    def _step_back(self) -> None:
        self._set_current_time(self.current_time_seconds - self._frame_step_seconds())

    def _step_forward(self) -> None:
        self._set_current_time(self.current_time_seconds + self._frame_step_seconds())

    def _step_back_10(self) -> None:
        self._set_current_time(self.current_time_seconds - 10.0 * self._frame_step_seconds())

    def _step_forward_10(self) -> None:
        self._set_current_time(self.current_time_seconds + 10.0 * self._frame_step_seconds())

    def _toggle_playback(self) -> None:
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        if self.total_duration_seconds <= 0.0:
            return
        self._playing = True
        self.play_button.configure(text="Pause")
        self._schedule_next_frame()

    def _stop_playback(self) -> None:
        self._playing = False
        self.play_button.configure(text="Play")
        if self._play_after_id is not None:
            self.root.after_cancel(self._play_after_id)
            self._play_after_id = None

    def _schedule_next_frame(self) -> None:
        if not self._playing:
            return
        try:
            speed = float(self.speed_var.get())
        except ValueError:
            speed = 1.0
        speed = max(0.1, speed)
        base_fps = 1.0 / self._frame_step_seconds()
        delay_ms = max(1, int(round(1000.0 / (base_fps * speed))))
        self._play_after_id = self.root.after(delay_ms, self._advance_playback)

    def _advance_playback(self) -> None:
        if not self._playing:
            return
        next_time = self.current_time_seconds + self._frame_step_seconds()
        if next_time >= self.total_duration_seconds:
            self._set_current_time(self.total_duration_seconds)
            self._stop_playback()
            return
        self._set_current_time(next_time)
        self._schedule_next_frame()

    def _update_summary(self) -> None:
        if self.raw_positions.size == 0:
            text = "No script loaded"
        else:
            stats = self.current_stats
            config_json = json.dumps(self.current_config.to_dict(), indent=2)
            text = "\n".join(
                [
                    f"actions: {self.raw_positions.size}",
                    f"script fps: {self.script_fps:.6f}",
                    f"raw duration: {self.total_duration_seconds:.3f}s",
                    f"chunks: {int(stats.get('num_chunks', 0))}",
                    f"normalized cycles: {int(stats.get('num_normalized_cycles', 0))}",
                    f"gradient segments: {int(stats.get('num_gradient_segments', 0))}",
                    f"cycle samples: {int(stats.get('num_cycle_samples', 0))}",
                    f"gradient samples: {int(stats.get('num_gradient_samples', 0))}",
                    f"clipped samples: {int(stats.get('num_clipped_samples', 0))}",
                    f"gradient low threshold: {float(stats.get('gradient_low_threshold', 0.0)):.6f}",
                    f"gradient high threshold: {float(stats.get('gradient_high_threshold', 0.0)):.6f}",
                    "",
                    "Current config:",
                    config_json,
                ]
            )
        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", text)
        self.summary_text.configure(state=tk.DISABLED)

    def _save_processed_script(self) -> None:
        if self.input_path is None or self.processed_positions.size == 0:
            messagebox.showinfo("Save Postprocessed", "Load and process a funscript first.")
            return

        default_path = self.input_path.with_name(f"{self.input_path.stem}.post{self.input_path.suffix}")
        path = filedialog.asksaveasfilename(
            title="Save postprocessed funscript",
            initialfile=default_path.name,
            defaultextension=self.input_path.suffix,
            filetypes=[("Funscript", "*.funscript"), ("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        out_data = dict(self.script_data)
        metadata = dict(out_data.get("metadata", {}))
        metadata["postprocessed"] = True
        metadata["postprocess_method"] = "wave"
        metadata["postprocess_fps"] = self.script_fps
        metadata["postprocess_config"] = self.current_config.to_dict()
        metadata["source_funscript"] = str(self.input_path)
        if self.video_path is not None:
            metadata["source_video"] = str(self.video_path)
        out_data["metadata"] = metadata
        out_data["actions"] = [
            {
                "at": int(at_ms),
                "pos": int(np.clip(np.round(float(position) * 100.0), 0, 100)),
            }
            for at_ms, position in zip(self.timestamps_ms, self.processed_positions)
        ]
        save_funscript(out_data, Path(path))
        self.status_var.set(f"Saved postprocessed funscript to {path}")

    def save_preview(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(path, dpi=150)

    def on_close(self) -> None:
        self._stop_playback()
        if self._reprocess_after_id is not None:
            self.root.after_cancel(self._reprocess_after_id)
            self._reprocess_after_id = None
        self.clear_video()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive tuner for wave-aware prediction postprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=None, help="Dense funscript to load at startup")
    parser.add_argument("--video", type=Path, default=None, help="Optional video to load at startup")
    parser.add_argument("--fps", type=float, default=None, help="Override inferred script fps")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Data directory used for auto-matching preprocessed videos",
    )
    parser.add_argument(
        "--test-load",
        action="store_true",
        help="Construct the GUI, load the requested assets, print stats, and exit",
    )
    parser.add_argument(
        "--save-preview",
        type=Path,
        default=None,
        help="Optional plot image written after startup or test-load",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    app = PostprocessTunerApp(root, data_dir=args.data_dir)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    try:
        if args.input is not None:
            app.load_script(args.input, fps_override=args.fps)
        if args.video is not None:
            app.load_video(args.video)
        root.update_idletasks()

        if args.save_preview is not None and args.input is not None:
            app.save_preview(args.save_preview)
            print(f"Saved preview: {args.save_preview}")

        if args.test_load:
            print(f"Loaded actions: {app.raw_positions.size}")
            print(f"Script fps: {app.script_fps:.6f}")
            print(
                "Wave postprocess: "
                f"chunks={int(app.current_stats.get('num_chunks', 0))} "
                f"cycles={int(app.current_stats.get('num_normalized_cycles', 0))} "
                f"gradients={int(app.current_stats.get('num_gradient_segments', 0))}"
            )
            app.on_close()
            return

        root.mainloop()
    finally:
        try:
            root_exists = bool(root.winfo_exists())
        except tk.TclError:
            root_exists = False
        if root_exists:
            app.on_close()


if __name__ == "__main__":
    main()