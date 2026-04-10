"""Video preprocessing using ffmpeg with hardware acceleration.

Handles VR SBS/TB cropping and downscaling of high-resolution source videos
to a manageable size for feature extraction. Uses NVENC for GPU-accelerated
encoding and CUDA for GPU-accelerated decoding.
"""

import logging
import subprocess
import json
from pathlib import Path

log = logging.getLogger(__name__)

# Map VR projection types to their stereo layout
STEREO_LAYOUTS = {
    "180_sbs": "sbs",      # side-by-side, crop left half
    "fisheye190": "sbs",   # also SBS
    "mkx200": "sbs",       # also SBS
    "360_tb": "tb",        # top-bottom, crop top half
    "180_mono": "mono",    # mono, no stereo crop needed
    "": "sbs",             # assume SBS if unknown
}


def probe_video(video_path: Path) -> dict:
    """Get video stream info using ffprobe.

    Returns:
        Dict with keys: width, height, fps, duration, codec
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr}")

    data = json.loads(result.stdout)
    video_stream = next(
        (s for s in data.get("streams", []) if s["codec_type"] == "video"),
        None,
    )
    if not video_stream:
        raise RuntimeError(f"No video stream found in {video_path}")

    fps_str = video_stream.get("r_frame_rate", "30/1")
    fps_parts = fps_str.split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])

    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": fps,
        "duration": float(data.get("format", {}).get("duration", 0)),
        "codec": video_stream.get("codec_name", ""),
    }


def build_preprocess_command(
    input_path: Path,
    output_path: Path,
    projection: str = "180_sbs",
    target_height: int = 960,
    target_fps: int | None = None,
    eye: str = "left",
    use_hw_accel: bool = True,
    crf: int = 23,
    source_codec: str = "",
    source_width: int = 0,
    source_height: int = 0,
) -> list[str]:
    """Build an ffmpeg command for preprocessing a VR video.

    Crops the selected eye from stereo video, downscales, and re-encodes.
    When HW acceleration is enabled, uses CUVID decoder with -resize to
    downscale in the decoder (before CPU transfer), then light CPU filters
    for crop/fps, and NVENC for encoding.

    Args:
        input_path: Source video path.
        output_path: Destination path for preprocessed video.
        projection: VR projection type (180_sbs, fisheye190, mkx200, 360_tb, 180_mono).
        target_height: Output height in pixels. Width computed to maintain aspect.
        target_fps: Output frame rate. None = keep original.
        eye: Which eye to extract: "left" or "right".
        use_hw_accel: Use CUVID decoding + NVENC encoding.
        crf: Constant rate factor (quality). Lower = better. 23 is default.
        source_codec: Source video codec name (e.g. "hevc", "h264") for CUVID.
        source_width: Source video width (for decoder resize calculation).
        source_height: Source video height (for decoder resize calculation).

    Returns:
        ffmpeg command as a list of strings.
    """
    layout = STEREO_LAYOUTS.get(projection, "sbs")

    # Determine the decoder resize dimensions
    # We want the final output after crop to be target_height tall.
    # For SBS: crop halves the width, so we need decoder output at 2*target_height wide  
    # For TB: crop halves the height, so we need decoder output at 2*target_height tall
    # Exact dims depend on source aspect ratio. Strategy: resize to a moderate
    # intermediate size in the decoder, then let CPU crop + scale do the rest.
    decoder_resize = ""
    if use_hw_accel and source_width > 0 and source_height > 0:
        # Resize in decoder to reduce data transfer. Target: make the larger
        # dimension ~2x the final target to keep crop quality.
        if layout == "sbs":
            # After crop: half width × full height → scale to target_height
            # Decoder resize: keep aspect ratio, target height = target_height
            # (crop will then take left/right half of width)
            resize_h = target_height
            resize_w = int(source_width * target_height / source_height)
            # Ensure even dimensions
            resize_w = resize_w + (resize_w % 2)
            decoder_resize = f"{resize_w}x{resize_h}"
        elif layout == "tb":
            resize_w = target_height  # after crop top half, height = target_height
            resize_h = target_height * 2
            decoder_resize = f"{resize_w}x{resize_h}"

    # Build video filter chain
    vf_filters: list[str] = []

    # Step 1: Stereo crop
    if layout == "sbs":
        if eye == "left":
            vf_filters.append("crop=iw/2:ih:0:0")
        else:
            vf_filters.append("crop=iw/2:ih:iw/2:0")
    elif layout == "tb":
        if eye == "left":
            vf_filters.append("crop=iw:ih/2:0:0")
        else:
            vf_filters.append("crop=iw:ih/2:0:ih/2")
    # mono: no crop

    # Step 2: Scale to target height (if decoder didn't handle it or CPU mode)
    if not decoder_resize:
        vf_filters.append(f"scale=-2:{target_height}")

    # Step 3: FPS conversion
    if target_fps:
        vf_filters.append(f"fps={target_fps}")

    vf_string = ",".join(vf_filters) if vf_filters else None

    if use_hw_accel:
        # CUVID decoder with resize → small frames to CPU → light filters → NVENC
        cuvid_decoder = _get_cuvid_decoder(source_codec)
        cmd = ["ffmpeg", "-y"]

        if cuvid_decoder:
            cmd.extend(["-c:v", cuvid_decoder])
            if decoder_resize:
                cmd.extend(["-resize", decoder_resize])
        else:
            # Fallback: generic CUDA hwaccel (no resize in decoder)
            cmd.extend(["-hwaccel", "cuda"])
            # Need scale filter since decoder can't resize
            if vf_string and "scale=" not in vf_string:
                if vf_filters:
                    vf_filters.append(f"scale=-2:{target_height}")
                    vf_string = ",".join(vf_filters)

        cmd.extend(["-i", str(input_path)])
        if vf_string:
            cmd.extend(["-vf", vf_string])
        cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(crf)])
        cmd.extend(["-movflags", "+faststart"])
        cmd.extend(["-an"])
        cmd.append(str(output_path))
        return cmd

    # CPU fallback
    if not vf_string:
        vf_filters.append(f"scale=-2:{target_height}")
        vf_string = ",".join(vf_filters)
    cmd = ["ffmpeg", "-y", "-i", str(input_path)]
    cmd.extend(["-vf", vf_string])
    cmd.extend(["-c:v", "libx264", "-crf", str(crf), "-preset", "fast"])
    cmd.extend(["-movflags", "+faststart"])
    cmd.extend(["-an"])
    cmd.append(str(output_path))
    return cmd


# Map codec names to CUVID decoders
_CUVID_DECODERS = {
    "hevc": "hevc_cuvid",
    "h265": "hevc_cuvid",
    "h264": "h264_cuvid",
    "avc": "h264_cuvid",
    "vp9": "vp9_cuvid",
    "av1": "av1_cuvid",
}


def _get_cuvid_decoder(codec: str) -> str:
    """Return the CUVID decoder name for a given codec, or empty string."""
    return _CUVID_DECODERS.get(codec.lower(), "")


def preprocess_video(
    input_path: Path,
    output_path: Path,
    projection: str = "180_sbs",
    target_height: int = 960,
    target_fps: int | None = None,
    eye: str = "left",
    use_hw_accel: bool = True,
    crf: int = 23,
    overwrite: bool = False,
) -> Path:
    """Preprocess a VR video: crop eye, downscale, re-encode.

    Args:
        input_path: Source video.
        output_path: Destination for preprocessed video.
        projection: VR projection type.
        target_height: Output height in pixels.
        target_fps: Output FPS or None to keep original.
        eye: "left" or "right" eye to extract.
        use_hw_accel: Use NVENC + CUDA.
        crf: Quality factor (lower = better).
        overwrite: If False, skip if output already exists.

    Returns:
        Path to the preprocessed video.
    """
    if output_path.exists() and not overwrite:
        log.info("Preprocessed video exists, skipping: %s", output_path.name)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Probe source to get codec and resolution for CUVID decoder selection
    source_info = probe_video(input_path)

    cmd = build_preprocess_command(
        input_path, output_path, projection, target_height,
        target_fps, eye, use_hw_accel, crf,
        source_codec=source_info["codec"],
        source_width=source_info["width"],
        source_height=source_info["height"],
    )

    log.info("Preprocessing: %s → %s", input_path.name, output_path.name)
    log.info("Command: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour max for very long videos
    )

    if result.returncode != 0:
        # If NVENC failed, retry with CPU encoding
        if use_hw_accel and ("nvenc" in result.stderr.lower() or "cuda" in result.stderr.lower()):
            log.warning("HW acceleration failed, falling back to CPU encoding")
            return preprocess_video(
                input_path, output_path, projection, target_height,
                target_fps, eye, use_hw_accel=False, crf=crf, overwrite=True,
            )
        raise RuntimeError(
            f"ffmpeg preprocessing failed for {input_path.name}:\n{result.stderr[-500:]}"
        )

    # Verify output
    if not output_path.exists() or output_path.stat().st_size < 1000:
        raise RuntimeError(f"Preprocessed output is missing or too small: {output_path}")

    info = probe_video(output_path)
    log.info(
        "Preprocessed: %dx%d, %.1f fps, %.1f min",
        info["width"], info["height"], info["fps"], info["duration"] / 60,
    )
    return output_path
