#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEXVERSE_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DATASET_ROOT="$DEXVERSE_DIR/visionpro_test"
TASK_NAME="${TASK_NAME:-Dexverse-PickCube-v0}"
INPUT_DIR="$DATASET_ROOT/$TASK_NAME"
OUTPUT_ROOT="$DEXVERSE_DIR/demo/visionpro_test"
VIDEO_ROOT="$OUTPUT_ROOT/composite_videos/$TASK_NAME"
CONDA_SETUP="${CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"

if [[ ! -f "$CONDA_SETUP" ]]; then
    echo "Error: Conda initialization script not found: $CONDA_SETUP" >&2
    exit 1
fi
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory not found: $INPUT_DIR" >&2
    exit 1
fi

shopt -s nullglob
input_files=("$INPUT_DIR"/*.pkl)
if (( ${#input_files[@]} == 0 )); then
    echo "Error: No .pkl files found in: $INPUT_DIR" >&2
    exit 1
fi

source "$CONDA_SETUP"
conda activate dexverse

# Use the imageio-ffmpeg binary when ffmpeg is not available on PATH.
FFMPEG_SHIM_DIR=""
cleanup_ffmpeg_shim() {
    if [[ -n "$FFMPEG_SHIM_DIR" && -d "$FFMPEG_SHIM_DIR" ]]; then
        rm -f -- "$FFMPEG_SHIM_DIR/ffmpeg"
        rmdir -- "$FFMPEG_SHIM_DIR"
    fi
}
trap cleanup_ffmpeg_shim EXIT

if ! command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG_EXE="$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
    if [[ ! -x "$FFMPEG_EXE" ]]; then
        echo "Error: FFmpeg executable not found: $FFMPEG_EXE" >&2
        exit 1
    fi
    FFMPEG_SHIM_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dexverse-ffmpeg.XXXXXX")"
    ln -s -- "$FFMPEG_EXE" "$FFMPEG_SHIM_DIR/ffmpeg"
    export PATH="$FFMPEG_SHIM_DIR:$PATH"
    echo "Using imageio-ffmpeg: $FFMPEG_EXE"
fi

mkdir -p "$OUTPUT_ROOT" "$VIDEO_ROOT"
cd "$DEXVERSE_DIR"

echo "Converting ${#input_files[@]} Pickle file(s):"
printf '  %s\n' "${input_files[@]}"

# Convert each Pickle in a separate Isaac Sim process.
for input_file in "${input_files[@]}"; do
    input_name="${input_file##*/}"
    input_stem="${input_name%.pkl}"
    h5_file="$OUTPUT_ROOT/$TASK_NAME/$input_stem.3view_rgb.seq.demo.h5"

    if [[ -f "$h5_file" ]]; then
        echo "[SKIP] HDF5 already exists: $h5_file"
        continue
    fi

    echo "[CONVERT] $input_file"
    python scripts/demo_tools/create_demo_files_sequential.py \
        --demos-root "$DATASET_ROOT" \
        --file "$input_file" \
        --output-dir "$OUTPUT_ROOT" \
        --obs-groups 3view_rgb \
        --set-state \
        "$@"
done

# Render composite RGB videos after all HDF5 files are available.
for input_file in "${input_files[@]}"; do
    input_name="${input_file##*/}"
    input_stem="${input_name%.pkl}"
    h5_file="$OUTPUT_ROOT/$TASK_NAME/$input_stem.3view_rgb.seq.demo.h5"
    video_dir="$VIDEO_ROOT/$input_stem"

    if [[ ! -f "$h5_file" ]]; then
        echo "Error: Converted HDF5 not found: $h5_file" >&2
        exit 1
    fi

    expected_videos="$(python - "$h5_file" <<'PY'
import h5py
import sys

with h5py.File(sys.argv[1], "r") as dataset:
    print(len(dataset["data"]))
PY
)"
    existing_videos=("$video_dir"/*.mp4)

    if (( ${#existing_videos[@]} == expected_videos )); then
        echo "[SKIP RENDER] Found ${#existing_videos[@]} MP4 file(s): $video_dir"
    else
        python scripts/demo_tools/render_demo_video.py \
            --dataset_file "$h5_file" \
            --episode all \
            --output-dir "$video_dir" \
            --fps 60
        existing_videos=("$video_dir"/*.mp4)
    fi

    for video_file in "${existing_videos[@]}"; do
        video_info="$(ffmpeg -hide_banner -i "$video_file" 2>&1 || true)"
        if [[ "$video_info" == *"yuv420p"* ]]; then
            echo "[SKIP TRANSCODE] Already yuv420p: $video_file"
            continue
        fi

        compatible_file="${video_file%.mp4}.yuv420p.tmp.mp4"
        echo "[TRANSCODE] $video_file"
        ffmpeg -hide_banner -loglevel error -y \
            -i "$video_file" \
            -an \
            -c:v libx264 \
            -crf 18 \
            -pix_fmt yuv420p \
            -movflags +faststart \
            "$compatible_file"
        mv -f -- "$compatible_file" "$video_file"
    done
done

echo "Conversion complete. HDF5: $OUTPUT_ROOT/$TASK_NAME; composite videos: $VIDEO_ROOT"
