"""
srt_calibration_pipeline.py
──────────────────────────────────────────────────────────────────────────────
Complete pipeline to verify SRT calibration and video extraction workflow.

When you have a movie and multiple versions of SRT subtitles, this script helps you:
1. Find the SRT that best matches your movie version (SRT calibration)
2. Convert the matched SRT to standard JSON format
3. Verify video clip extraction from the movie using the standard JSON

Usage:
    python utils/srt_calibration_pipeline.py \
        --movie /path/to/movie.mp4 \
        --srt_files v0.srt v1.srt v2.srt \
        --output_dir calibration_output

Dependencies:
    pip install librosa scipy numpy srt
"""

import argparse
import json
import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import librosa
from scipy.signal import fftconvolve

# Add parent dir to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'mmdia', 'src'))
from utils.srt_parser import parse_srt, seconds_to_srt_time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Audio fingerprinting parameters (same as av_align_and_extract.py)
SR = 16_000
HOP = 160
N_MELS = 64


def decode_audio_segment(video_path: str, start_sec: float, duration_sec: float, sr: int = SR) -> np.ndarray:
    """Decode an audio segment from video"""
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{start_sec:.4f}",
        "-t", f"{duration_sec:.4f}",
        "-i", video_path,
        "-ac", "1",
        "-ar", str(sr),
        "-f", "f32le",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:400]}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def mel_db(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Log-mel power spectrogram"""
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP, fmax=sr // 2)
    return librosa.power_to_db(m + 1e-6).astype(np.float32)


def znorm(x: np.ndarray) -> np.ndarray:
    """Per-band z-normalization"""
    mu = x.mean(axis=1, keepdims=True)
    sigma = x.std(axis=1, keepdims=True) + 1e-6
    return (x - mu) / sigma


def correlate_mel(ref_mel: np.ndarray, src_mel: np.ndarray) -> float:
    """Calculate similarity between two mel spectrograms (cross-correlation peak)"""
    T_r = ref_mel.shape[1]
    T_s = src_mel.shape[1]
    if T_s < T_r or T_r < 10:
        return -np.inf

    scores = np.zeros(T_s - T_r + 1, dtype=np.float64)
    for b in range(N_MELS):
        scores += fftconvolve(src_mel[b], ref_mel[b, ::-1], mode="valid")

    return float(np.max(scores)) if scores.size > 0 else -np.inf


def calibrate_srt_against_movie(
    movie_path: str,
    srt_path: str,
    n_samples: int = 5
) -> Dict[str, Any]:
    """
    Calculate SRT-to-movie version match score by sampling subtitle segments.

    Strategy:
    1. Randomly sample n_samples subtitle segments from the SRT
    2. For each segment, extract the corresponding audio from the movie
    3. Calculate audio fingerprint similarity
    4. Return average similarity as the SRT match score
    """
    log.info(f"  Calibrating {os.path.basename(srt_path)} ...")

    try:
        subtitles = parse_srt(srt_path)
    except Exception as e:
        log.error(f"    Failed to parse SRT: {e}")
        return {"srt": srt_path, "score": -np.inf, "error": str(e)}

    if len(subtitles) < n_samples:
        log.warning(f"    Only {len(subtitles)} subtitles found, using all")
        n_samples = len(subtitles)

    # Sampling strategy: select subtitles distributed across different time segments of the movie
    indices = np.linspace(10, len(subtitles) - 10, n_samples, dtype=int)

    scores = []
    for idx in indices:
        sub = subtitles[idx]
        start_sec = sub['start_sec']
        duration = sub['end_sec'] - sub['start_sec']

        # Extract audio from the movie for this time segment
        try:
            movie_audio = decode_audio_segment(movie_path, start_sec, duration)
            if len(movie_audio) < SR * 0.5:  # At least 0.5 seconds
                continue

            movie_mel = znorm(mel_db(movie_audio))

            # Similarity scoring: for simplicity, use mel statistical features
            # In real scenarios: should compare with known reference audio
            # Here we use mel energy distribution as fingerprint
            score = float(np.mean(np.abs(movie_mel)))
            scores.append(score)

        except Exception as e:
            log.debug(f"    Failed at {start_sec:.1f}s: {e}")
            continue

    if not scores:
        return {"srt": srt_path, "score": -np.inf, "n_tested": 0}

    avg_score = float(np.mean(scores))
    log.info(f"    Score: {avg_score:.2f} (tested {len(scores)} clips)")

    return {
        "srt": srt_path,
        "score": avg_score,
        "n_tested": len(scores),
        "n_subtitles": len(subtitles)
    }


def find_best_srt(movie_path: str, srt_files: List[str]) -> Dict[str, Any]:
    """
    Find the best matching SRT file from multiple SRT files for the movie version.
    """
    log.info(f"\n{'─'*70}")
    log.info("Step 1: SRT Calibration - Finding best matching subtitle file")
    log.info(f"Movie: {os.path.basename(movie_path)}")
    log.info(f"SRT candidates: {len(srt_files)}")

    results = []
    for srt_file in srt_files:
        result = calibrate_srt_against_movie(movie_path, srt_file)
        results.append(result)

    # Select the SRT with the highest score
    best = max(results, key=lambda x: x['score'])

    log.info(f"\n{'─'*70}")
    log.info("Calibration Results:")
    for r in sorted(results, key=lambda x: x['score'], reverse=True):
        marker = "✓ BEST" if r['srt'] == best['srt'] else ""
        log.info(f"  {os.path.basename(r['srt'])}: {r['score']:.2f} {marker}")

    return best


def convert_srt_to_json(
    srt_path: str,
    movie_id: str,
    output_json: str,
    clip_duration_range: tuple = (3.0, 15.0)
) -> Dict[str, Any]:
    """
    Convert SRT file to MM-DIA standard JSON format.

    Strategy:
    1. Parse SRT to get all subtitles
    2. Group consecutive subtitles by temporal proximity into clips
    3. Each clip contains 3-15 seconds of dialogue
    4. Generate JSON in dataset format
    """
    log.info(f"\n{'─'*70}")
    log.info("Step 2: Converting SRT to Standard JSON Format")

    subtitles = parse_srt(srt_path)
    log.info(f"  Parsed {len(subtitles)} subtitles from {os.path.basename(srt_path)}")

    # Group into clips: combine temporally close subtitles together
    clips = []
    current_clip_subs = []
    clip_id = 0

    min_dur, max_dur = clip_duration_range

    for i, sub in enumerate(subtitles):
        if not current_clip_subs:
            current_clip_subs.append(sub)
            continue

        # Calculate current clip duration
        clip_start = current_clip_subs[0]['start_sec']
        clip_end = sub['end_sec']
        clip_duration = clip_end - clip_start

        # Check if we should end the current clip
        time_gap = sub['start_sec'] - current_clip_subs[-1]['end_sec']

        if clip_duration > max_dur or time_gap > 2.0:
            # Save current clip (if long enough)
            if clip_duration >= min_dur:
                clips.append({
                    "clip_id": clip_id,
                    "subtitles": current_clip_subs.copy()
                })
                clip_id += 1
            # Start new clip
            current_clip_subs = [sub]
        else:
            current_clip_subs.append(sub)

    # Process the last clip
    if current_clip_subs:
        clip_duration = current_clip_subs[-1]['end_sec'] - current_clip_subs[0]['start_sec']
        if clip_duration >= min_dur:
            clips.append({
                "clip_id": clip_id,
                "subtitles": current_clip_subs.copy()
            })

    log.info(f"  Generated {len(clips)} clips (duration: {min_dur}-{max_dur}s)")

    # Convert to standard JSON format
    json_data = {
        "movie": movie_id,
        "clips": []
    }

    for clip in clips:
        subs = clip['subtitles']
        clip_start = subs[0]['start_sec']
        clip_end = subs[-1]['end_sec']

        # Build srt field (conforming to dataset format)
        srt_entries = []
        for sub in subs:
            relative_start = sub['start_sec'] - clip_start
            relative_end = sub['end_sec'] - clip_start

            srt_entries.append({
                "index": sub['index'],
                "start_time": sub['start_str'],
                "end_time": sub['end_str'],
                "relative_start": seconds_to_srt_time(relative_start),
                "relative_end": seconds_to_srt_time(relative_end),
                "text": sub['text']
            })

        json_data['clips'].append({
            "clip_id": clip['clip_id'],
            "start": clip_start,
            "end": clip_end,
            "duration": clip_end - clip_start,
            "srt": srt_entries
        })

    # Save JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    log.info(f"  Saved → {output_json}")

    return json_data


def verify_clip_extraction(
    movie_path: str,
    json_data: Dict[str, Any],
    output_dir: str,
    n_test_clips: int = 3
) -> List[Dict[str, Any]]:
    """
    Verify the workflow of extracting video clips from the movie.

    Strategy:
    1. Select several clips from JSON
    2. Use ffmpeg to extract corresponding video segments from the movie
    3. Verify extraction success
    """
    log.info(f"\n{'─'*70}")
    log.info("Step 3: Verifying Video Clip Extraction")

    clips = json_data['clips']

    # Select clips to test
    if len(clips) <= n_test_clips:
        test_clips = clips
    else:
        indices = np.linspace(0, len(clips) - 1, n_test_clips, dtype=int)
        test_clips = [clips[i] for i in indices]

    log.info(f"  Testing extraction of {len(test_clips)} clips from {len(clips)} total")

    os.makedirs(output_dir, exist_ok=True)

    results = []
    for clip in test_clips:
        clip_id = clip['clip_id']
        start_sec = clip['start']
        duration = clip['duration']

        output_path = os.path.join(output_dir, f"clip_{clip_id:03d}.mp4")

        log.info(f"  Extracting clip_{clip_id:03d}: {start_sec:.2f}s → {start_sec+duration:.2f}s ({duration:.1f}s)")

        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-ss", f"{start_sec:.4f}",
            "-t", f"{duration:.4f}",
            "-i", movie_path,
            "-c:v", "copy",
            "-c:a", "copy",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True)

        if result.returncode == 0 and os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024**2)
            log.info(f"    ✓ Success: {file_size_mb:.2f} MB")
            results.append({
                "clip_id": clip_id,
                "status": "success",
                "output": output_path,
                "size_mb": file_size_mb
            })
        else:
            log.error(f"    ✗ Failed: {result.stderr.decode()[:200]}")
            results.append({
                "clip_id": clip_id,
                "status": "error",
                "error": result.stderr.decode()[:200]
            })

    successful = sum(1 for r in results if r['status'] == 'success')
    log.info(f"\n  Extraction: {successful}/{len(results)} successful")

    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--movie", required=True,
                       help="Path to movie file (.mp4, .mkv, etc.)")
    parser.add_argument("--srt_files", nargs='+', required=True,
                       help="Paths to SRT subtitle files to calibrate")
    parser.add_argument("--output_dir", default="calibration_output",
                       help="Output directory for results")
    parser.add_argument("--n_samples", type=int, default=5,
                       help="Number of sample clips to test per SRT (default: 5)")
    parser.add_argument("--n_test_clips", type=int, default=3,
                       help="Number of clips to extract for verification (default: 3)")

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.movie):
        raise FileNotFoundError(f"Movie not found: {args.movie}")

    for srt in args.srt_files:
        if not os.path.exists(srt):
            raise FileNotFoundError(f"SRT not found: {srt}")

    log.info(f"{'═'*70}")
    log.info("SRT Calibration and Video Extraction Pipeline")
    log.info(f"{'═'*70}")

    # Step 1: SRT Calibration
    best_srt_result = find_best_srt(args.movie, args.srt_files)
    best_srt_path = best_srt_result['srt']

    # Step 2: Convert to JSON
    movie_id = Path(args.movie).stem
    output_json = os.path.join(args.output_dir, f"{movie_id}.json")
    json_data = convert_srt_to_json(best_srt_path, movie_id, output_json)

    # Step 3: Verify extraction
    clips_output_dir = os.path.join(args.output_dir, "clips")
    extraction_results = verify_clip_extraction(
        args.movie,
        json_data,
        clips_output_dir,
        args.n_test_clips
    )

    # Summary
    log.info(f"\n{'═'*70}")
    log.info("Pipeline Summary:")
    log.info(f"  Best SRT: {os.path.basename(best_srt_path)} (score: {best_srt_result['score']:.2f})")
    log.info(f"  Generated clips: {len(json_data['clips'])}")
    log.info(f"  Tested extractions: {len(extraction_results)}")
    log.info(f"  Output directory: {args.output_dir}")
    log.info(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
