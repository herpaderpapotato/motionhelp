"""Apply wave-aware postprocessing to dense prediction funscripts.

This script is intended for the unsimplified framewise funscript output written by
scripts/predict_disposition.py, where each action corresponds to one predicted
frame. It rewrites the action positions after applying the same cycle-aware and
gradient-aware postprocess that can now be enabled directly during prediction.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.funscript import load_funscript, save_funscript
from src.data.prediction_postprocess import WavePostprocessConfig, postprocess_predictions


DEFAULT_POSTPROCESS = WavePostprocessConfig()


def estimate_fps_from_actions(actions: list[dict]) -> float:
    if len(actions) < 2:
        raise ValueError("Need at least two actions to infer fps; pass --fps explicitly")

    timestamps_ms = np.asarray([int(action["at"]) for action in actions], dtype=np.float64)
    total_delta_ms = float(timestamps_ms[-1] - timestamps_ms[0])
    if total_delta_ms > 0.0:
        return 1000.0 * float(len(actions) - 1) / total_delta_ms

    diffs_ms = np.diff(timestamps_ms)
    positive_diffs = diffs_ms[diffs_ms > 0.0]
    if positive_diffs.size == 0:
        raise ValueError("Could not infer fps from action timestamps; pass --fps explicitly")
    return 1000.0 / float(np.mean(positive_diffs))


def build_postprocess_config(args: argparse.Namespace) -> WavePostprocessConfig:
    chunk_seconds = args.chunk_seconds
    if chunk_seconds is not None and chunk_seconds <= 0.0:
        chunk_seconds = None
    return WavePostprocessConfig(
        lowpass_cutoff_hz=args.lowpass_cutoff_hz,
        lowpass_order=args.lowpass_order,
        trough_prominence=args.trough_prominence,
        trough_distance_seconds=args.trough_distance_seconds,
        min_cycle_amplitude=args.min_cycle_amplitude,
        min_cycle_frequency_hz=args.min_cycle_frequency_hz,
        gradient_smoothing=args.gradient_smoothing,
        gradient_window_seconds=args.gradient_window_seconds,
        gradient_polyorder=args.gradient_polyorder,
        gradient_sigma_seconds=args.gradient_sigma_seconds,
        gradient_floor=args.gradient_floor,
        gradient_low_quantile=args.gradient_low_quantile,
        gradient_high_quantile=args.gradient_high_quantile,
        min_gradient_length_seconds=args.min_gradient_length_seconds,
        min_gradient_range=args.min_gradient_range,
        min_gradient_sign_balance=args.min_gradient_sign_balance,
        merge_gap_seconds=args.merge_gap_seconds,
        stretch_mode=args.stretch_mode,
        stretch_gain=args.stretch_gain,
        chunk_seconds=chunk_seconds,
        chunk_overlap_seconds=args.chunk_overlap_seconds,
    )


def print_postprocess_stats(stats: dict[str, object]) -> None:
    print(
        f"Wave postprocess: chunks={stats['num_chunks']} "
        f"cycles={stats['num_normalized_cycles']} gradients={stats['num_gradient_segments']} "
        f"cycle_samples={stats['num_cycle_samples']} gradient_samples={stats['num_gradient_samples']} "
        f"clipped={stats['num_clipped_samples']}"
    )
    print(
        f"  thresholds: gradient_low={float(stats['gradient_low_threshold']):.6f} "
        f"gradient_high={float(stats['gradient_high_threshold']):.6f}"
    )


def plot_predictions(
    raw_positions: np.ndarray,
    processed_positions: np.ndarray,
    fps: float,
    source_name: str,
    show_plot: bool,
    save_plot: Path | None,
) -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("TkAgg" if show_plot else "Agg")
    time_axis = np.arange(len(raw_positions), dtype=np.float32) / fps
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

    axes[0].plot(time_axis, raw_positions, lw=1.0, color="steelblue", label="Raw")
    axes[0].plot(time_axis, processed_positions, lw=1.2, color="darkorange", label="Postprocessed")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Position")
    axes[0].set_title(f"Prediction Postprocess Comparison — {source_name}")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    delta = processed_positions - raw_positions
    axes[1].fill_between(time_axis, delta, color="purple", alpha=0.35, label="Post - raw")
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_ylabel("Delta")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    if save_plot is not None:
        save_plot.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_plot, dpi=150)
        print(f"Saved plot: {save_plot}")
    if show_plot:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Postprocess dense prediction funscripts with wave-aware normalization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Dense funscript produced by predict_disposition.py")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output funscript path (defaults to <input>.post.funscript)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override inferred frame rate used for postprocessing")
    parser.add_argument("--save-npy", type=Path, default=None,
                        help="Optional path to save the postprocessed framewise positions as .npy")
    parser.add_argument("--plot", action="store_true",
                        help="Show a raw-vs-postprocessed comparison plot")
    parser.add_argument("--save-plot", type=Path, default=None,
                        help="Save the comparison plot to this file")
    parser.add_argument("--lowpass-cutoff-hz", type=float,
                        default=DEFAULT_POSTPROCESS.lowpass_cutoff_hz,
                        help="Zero-phase low-pass cutoff before wave detection")
    parser.add_argument("--lowpass-order", type=int,
                        default=DEFAULT_POSTPROCESS.lowpass_order,
                        help="Butterworth low-pass filter order")
    parser.add_argument("--trough-prominence", type=float,
                        default=DEFAULT_POSTPROCESS.trough_prominence,
                        help="Minimum trough prominence for cycle detection")
    parser.add_argument("--trough-distance-seconds", type=float,
                        default=DEFAULT_POSTPROCESS.trough_distance_seconds,
                        help="Minimum time between detected troughs")
    parser.add_argument("--min-cycle-amplitude", type=float,
                        default=DEFAULT_POSTPROCESS.min_cycle_amplitude,
                        help="Only normalize cycles at or above this peak-to-trough range")
    parser.add_argument("--min-cycle-frequency-hz", type=float,
                        default=DEFAULT_POSTPROCESS.min_cycle_frequency_hz,
                        help="Only normalize cycles faster than this frequency")
    parser.add_argument("--gradient-smoothing", type=str,
                        default=DEFAULT_POSTPROCESS.gradient_smoothing,
                        choices=["none", "savgol", "gaussian"],
                        help="Smoother used before slow-gradient detection")
    parser.add_argument("--gradient-window-seconds", type=float,
                        default=DEFAULT_POSTPROCESS.gradient_window_seconds,
                        help="Savgol window size for slow-gradient detection")
    parser.add_argument("--gradient-polyorder", type=int,
                        default=DEFAULT_POSTPROCESS.gradient_polyorder,
                        help="Savgol polynomial order for gradient smoothing")
    parser.add_argument("--gradient-sigma-seconds", type=float,
                        default=DEFAULT_POSTPROCESS.gradient_sigma_seconds,
                        help="Gaussian sigma when --gradient-smoothing gaussian is used")
    parser.add_argument("--gradient-floor", type=float,
                        default=DEFAULT_POSTPROCESS.gradient_floor,
                        help="Minimum absolute gradient threshold floor")
    parser.add_argument("--gradient-low-quantile", type=float,
                        default=DEFAULT_POSTPROCESS.gradient_low_quantile,
                        help="Lower gradient quantile used for slow-gradient detection")
    parser.add_argument("--gradient-high-quantile", type=float,
                        default=DEFAULT_POSTPROCESS.gradient_high_quantile,
                        help="Upper gradient quantile used for slow-gradient detection")
    parser.add_argument("--min-gradient-length-seconds", type=float,
                        default=DEFAULT_POSTPROCESS.min_gradient_length_seconds,
                        help="Minimum duration of a slow-gradient segment to stretch")
    parser.add_argument("--min-gradient-range", type=float,
                        default=DEFAULT_POSTPROCESS.min_gradient_range,
                        help="Minimum local range before a slow-gradient segment is stretched")
    parser.add_argument("--min-gradient-sign-balance", type=float,
                        default=DEFAULT_POSTPROCESS.min_gradient_sign_balance,
                        help="Require mostly one-sided gradients inside a candidate segment")
    parser.add_argument("--merge-gap-seconds", type=float,
                        default=DEFAULT_POSTPROCESS.merge_gap_seconds,
                        help="Merge nearby gradient runs separated by this gap")
    parser.add_argument("--stretch-mode", type=str,
                        default=DEFAULT_POSTPROCESS.stretch_mode,
                        choices=["linear", "tanh"],
                        help="Stretching function used on slow-gradient segments")
    parser.add_argument("--stretch-gain", type=float,
                        default=DEFAULT_POSTPROCESS.stretch_gain,
                        help="Steepness for tanh-based slow-gradient stretching")
    parser.add_argument("--chunk-seconds", type=float,
                        default=DEFAULT_POSTPROCESS.chunk_seconds,
                        help="Chunk size for long predictions; set to 0 to disable chunking")
    parser.add_argument("--chunk-overlap-seconds", type=float,
                        default=DEFAULT_POSTPROCESS.chunk_overlap_seconds,
                        help="Overlap between postprocess chunks")
    args = parser.parse_args()

    data = load_funscript(args.input)
    actions = sorted(data.get("actions", []), key=lambda action: action["at"])
    if not actions:
        raise ValueError(f"No actions found in {args.input}")

    fps = float(args.fps) if args.fps is not None else estimate_fps_from_actions(actions)
    print(f"Using fps: {fps:.6f}")

    timestamps_ms = np.asarray([int(action["at"]) for action in actions], dtype=np.int64)
    raw_positions = np.asarray([float(action["pos"]) for action in actions], dtype=np.float32) / 100.0
    config = build_postprocess_config(args)
    processed_positions, stats = postprocess_predictions(
        raw_positions,
        fps,
        config,
        return_stats=True,
    )
    print_postprocess_stats(stats)

    out_path = args.out
    if out_path is None:
        out_path = args.input.with_name(f"{args.input.stem}.post{args.input.suffix}")

    out_data = dict(data)
    metadata = dict(out_data.get("metadata", {}))
    metadata["postprocessed"] = True
    metadata["postprocess_method"] = "wave"
    metadata["postprocess_fps"] = fps
    metadata["postprocess_config"] = config.to_dict()
    metadata["source_funscript"] = str(args.input)
    out_data["metadata"] = metadata
    out_data["actions"] = [
        {
            "at": int(at_ms),
            "pos": int(np.clip(np.round(float(position) * 100.0), 0, 100)),
        }
        for at_ms, position in zip(timestamps_ms, processed_positions)
    ]
    save_funscript(out_data, out_path)
    print(f"Saved postprocessed funscript: {out_path}")

    if args.save_npy is not None:
        args.save_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_npy, processed_positions)
        print(f"Saved postprocessed positions: {args.save_npy}")

    if args.plot or args.save_plot:
        plot_predictions(
            raw_positions=raw_positions,
            processed_positions=processed_positions,
            fps=fps,
            source_name=str(args.input),
            show_plot=args.plot,
            save_plot=args.save_plot,
        )


if __name__ == "__main__":
    main()