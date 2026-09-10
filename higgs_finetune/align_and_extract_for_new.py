"""
align_and_extract.py
──────────────────────────────────────────────────────────────────────────────
Aligns MM-DIA clips against a user-supplied source video, then extracts the
matching video segments.

The MM-DIA dataset does NOT release video files.  Instead, the released
*audio* files (denoised .wav) serve as fingerprint references.  This script
searches for each reference clip inside the user's source video using
mel-spectrogram cross-correlation, then extracts the corresponding video
segment via ffmpeg.

Strategy
────────
The source video's audio is decoded ONCE and its mel spectrogram is held in
memory.  Every clip reference is then searched against a slice of that shared
spectrogram — avoiding repeated ffmpeg seeks and making the per-clip cost
negligible.

Speed-offset handling
─────────────────────
Different rips of the same film can run at slightly different speeds (PAL/NTSC
4 % difference, variable encode rates).  For each clip we try a small set of
speed candidates.  Instead of the expensive phase-vocoder time_stretch, we
simply resample the reference audio in time (librosa.resample), which changes
duration without preserving pitch — sufficient for fingerprinting, ~10× faster.

Memory estimate (2-hour film at SR=16 kHz, hop=160):
  audio array : 7200 × 16000 × 4 B  ≈  440 MB
  mel (64 × T) : 64 × 720000 × 4 B  ≈  185 MB  (kept after audio is freed)

Usage
─────
# Single film
python align_and_extract.py \\
    --json_path    release/json/movie/tt0114369.json \\
    --audio_dir    release/audio/movie/tt0114369 \\
    --source_video my_sources/tt0114369.mkv \\
    --output_dir   extracted/movie/tt0114369

# Batch (all films in a batch)
python align_and_extract.py \\
    --batch_json_dir  release/json/movie \\
    --batch_audio_dir release/audio/movie \\
    --source_dir      my_sources/movie \\
    --batch_output_dir extracted/movie

Options
───────
  --search_window SEC   Half-width of per-clip search window (default 120 s).
                        Increase if sources have large missing/added scenes.
  --workers N           Parallel workers for clip extraction (default 4).
  --dry_run             Print detected offsets only; do not extract video.
  --results_json PATH   Save per-clip alignment log as JSON.

Dependencies
────────────
  pip install librosa scipy numpy tqdm
  ffmpeg on $PATH
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import fftconvolve
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Analysis parameters
# ──────────────────────────────────────────────────────────────────────────────

SR       = 16_000   # working sample rate — 16 kHz is plenty for fingerprinting
HOP      = 160      # hop length  →  0.01 s / frame  (SR / HOP = 100 fps)
N_MELS   = 64

# Speed candidates: cover PAL↔NTSC (~4 %), small encode-rate variations.
# Each value means "source runs at this speed relative to the dataset master".
SPEED_CANDIDATES = [0.96, 0.98, 1.00, 1.02, 1.04]

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Low-level audio helpers
# ──────────────────────────────────────────────────────────────────────────────

def decode_full_audio(video_path: str, sr: int = SR) -> np.ndarray:
    """
    Decode the entire audio track of a video file to a mono float32 array.
    Uses a single ffmpeg call with a pipe — no temp files.
    """
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", video_path,
        "-ac", "1",          # mono
        "-ar", str(sr),      # target sample rate
        "-f", "f32le",       # raw 32-bit float, little-endian
        "-",                 # stdout
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed decoding {video_path}:\n{result.stderr.decode()[:600]}"
        )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def load_ref_audio(path: str, sr: int = SR) -> np.ndarray:
    """Load a reference WAV file from the dataset."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def mel_db(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Log-mel power spectrogram, shape (N_MELS, T)."""
    m = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, hop_length=HOP, fmax=sr // 2
    )
    return librosa.power_to_db(m + 1e-6).astype(np.float32)


def znorm(x: np.ndarray) -> np.ndarray:
    """Per-band zero-mean unit-variance normalisation."""
    mu    = x.mean(axis=1, keepdims=True)
    sigma = x.std(axis=1, keepdims=True) + 1e-6
    return (x - mu) / sigma


# ──────────────────────────────────────────────────────────────────────────────
# Cross-correlation search
# ──────────────────────────────────────────────────────────────────────────────

def correlate_mel(ref_mel: np.ndarray, src_mel: np.ndarray) -> np.ndarray:
    """
    FFT-based cross-correlation between ref (N_MELS, T_r) and src (N_MELS, T_s).
    Returns scores array of length  T_s - T_r + 1  (one score per lag).
    Each score is the sum of per-band correlations.
    """
    T_r = ref_mel.shape[1]
    T_s = src_mel.shape[1]
    if T_s < T_r:
        return np.array([-np.inf])

    scores = np.zeros(T_s - T_r + 1, dtype=np.float64)
    for b in range(N_MELS):
        scores += fftconvolve(src_mel[b], ref_mel[b, ::-1], mode="valid")
    return scores


def search_in_movie_mel(
    ref_audio: np.ndarray,
    movie_mel: np.ndarray,          # full movie mel, shape (N_MELS, T_movie)
    approx_start_sec: float,
    search_window_sec: float,
    speed_candidates: list = SPEED_CANDIDATES,
) -> tuple[float, float, float]:
    """
    Find the best match for `ref_audio` inside `movie_mel`.

    Parameters
    ──────────
    ref_audio        : reference clip audio (1-D float32, sample rate = SR)
    movie_mel        : pre-computed normalised mel of the full source video
    approx_start_sec : expected start time in the source (seconds)
    search_window_sec: total search window (seconds) centred on approx_start

    Returns
    ───────
    (found_start_sec, best_speed, best_score)

    How speed candidates work
    ─────────────────────────
    If the source runs at speed s (e.g. s=1.04 for a 4 % faster encode),
    then the same content is compressed in time by factor s.  We simulate
    this by resampling the reference to be s× shorter, then searching.
    librosa.resample (sinc, not phase vocoder) is ~10× faster than
    time_stretch and is all we need for fingerprinting.
    """
    half        = search_window_sec / 2.0
    win_start_s = max(0.0, approx_start_sec - half)
    win_end_s   = approx_start_sec + half + len(ref_audio) / SR

    # Convert seconds → mel frames
    def s2f(sec):
        return max(0, min(movie_mel.shape[1] - 1, int(sec * SR / HOP)))

    f_start = s2f(win_start_s)
    f_end   = s2f(win_end_s)
    src_slice = movie_mel[:, f_start:f_end]   # view, no copy

    best_score = -np.inf
    best_start = approx_start_sec
    best_speed = 1.0

    for speed in speed_candidates:
        # Resample reference in time only — changes duration, not pitch.
        # orig_sr / target_sr = 1 / speed  ⟹  output is 1/speed × shorter.
        resampled = librosa.resample(
            ref_audio, orig_sr=SR, target_sr=int(SR * speed)
        )
        ref_mel = znorm(mel_db(resampled))

        scores = correlate_mel(ref_mel, src_slice)
        if scores.size == 0:
            continue

        idx   = int(np.argmax(scores))
        score = float(scores[idx])

        if score > best_score:
            best_score = score
            # frame index → absolute seconds in source
            best_start = win_start_s + (f_start + idx) * HOP / SR
            best_speed = speed

    return best_start, best_speed, best_score


# ──────────────────────────────────────────────────────────────────────────────
# Video extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_video_clip(
    source_video: str,
    start_sec: float,
    duration_sec: float,
    output_path: str,
    speed: float = 1.0,
) -> bool:
    """
    Extract [start_sec, start_sec + duration_sec/speed] from source_video.
    If speed != 1, re-time video and audio to natural 1× speed.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # We read duration/speed worth of source material, then scale back to
    # duration seconds so the output sounds/looks at normal speed.
    read_dur = duration_sec / speed

    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-ss", f"{start_sec:.4f}",
        "-t",  f"{read_dur:.4f}",
        "-i",  source_video,
    ]

    if abs(speed - 1.0) > 0.005:
        cmd += [
            "-filter_complex",
            f"[0:v]setpts={1.0/speed:.6f}*PTS[v];"
            f"[0:a]atempo={speed:.4f}[a]",
            "-map", "[v]",
            "-map", "[a]",
        ]
    else:
        cmd += ["-c:v", "copy", "-c:a", "copy"]

    cmd.append(output_path)

    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        log.error(f"ffmpeg failed → {output_path}\n{r.stderr.decode()[:400]}")
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Film-level orchestration
# ──────────────────────────────────────────────────────────────────────────────

def _find_ref_audio(audio_dir: str, clip_id: str) -> str | None:
    """Return the path to the released reference WAV for a clip, or None."""
    candidates = [
        os.path.join(audio_dir, clip_id, f"{clip_id}.wav"),
        os.path.join(audio_dir, f"{clip_id}.wav"),
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


def process_film(
    json_path: str,
    audio_dir: str,
    source_video: str,
    output_dir: str,
    search_window: float,
    workers: int,
    dry_run: bool,
) -> list[dict]:
    """
    Full pipeline for one film:
      1. Decode source video audio (one ffmpeg call).
      2. Compute full-movie mel spectrogram (once).
      3. For each clip, search and (optionally) extract in parallel.
    """
    log.info(f"  Loading source audio from {source_video} …")
    try:
        movie_audio = decode_full_audio(source_video)
    except RuntimeError as e:
        log.error(str(e))
        return []

    log.info(
        f"  Source audio: {len(movie_audio)/SR/60:.1f} min "
        f"({len(movie_audio)/SR:.0f} s)  —  computing mel spectrogram …"
    )
    movie_mel = znorm(mel_db(movie_audio))
    del movie_audio   # free ~440 MB; mel (~185 MB) stays for searches

    log.info(
        f"  Mel shape: {movie_mel.shape}  "
        f"({movie_mel.shape[1] * HOP / SR:.0f} s coverage)"
    )

    # Load clips from JSON
    with open(json_path) as f:
        data = json.load(f)
    clips = data if isinstance(data, list) else data.get("clips", data.get("dialogues", []))

    def _job(clip: dict) -> dict:
        clip_id  = clip.get("clip_id", clip.get("id", ""))
        start    = float(clip.get("start", clip.get("start_sec", 0)))
        end      = float(clip.get("end",   clip.get("end_sec",   0)))
        duration = end - start if end > start else float(clip.get("duration", 0))

        if not clip_id or duration <= 0:
            return {"clip_id": clip_id, "status": "skipped (no id or duration)"}

        out_path = os.path.join(output_dir, f"{clip_id}.mp4")
        if os.path.exists(out_path):
            return {"clip_id": clip_id, "status": "skipped (exists)"}

        ref_path = _find_ref_audio(audio_dir, clip_id)
        if ref_path is None:
            return {"clip_id": clip_id, "status": "error: ref audio missing"}

        try:
            ref_audio = load_ref_audio(ref_path)
        except Exception as e:
            return {"clip_id": clip_id, "status": f"error loading ref: {e}"}

        found_start, speed, score = search_in_movie_mel(
            ref_audio=ref_audio,
            movie_mel=movie_mel,
            approx_start_sec=start,
            search_window_sec=search_window,
        )

        delta = found_start - start
        log.debug(
            f"    {clip_id}: expected={start:.2f}s  found={found_start:.2f}s  "
            f"Δ={delta:+.3f}s  speed={speed:.3f}  score={score:.1f}"
        )

        result = {
            "clip_id":      clip_id,
            "expected_start": start,
            "found_start":  found_start,
            "offset_delta": delta,
            "speed":        speed,
            "score":        score,
            "duration":     duration,
        }

        if dry_run:
            result["status"] = "dry_run"
            return result

        ok = extract_video_clip(
            source_video=source_video,
            start_sec=found_start,
            duration_sec=duration,
            output_path=out_path,
            speed=speed,
        )
        result["status"] = "ok" if ok else "error: ffmpeg failed"
        result["output"] = out_path
        return result

    results = []
    # movie_mel is read-only in all threads; no lock needed
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_job, c): c.get("clip_id", "") for c in clips}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=os.path.basename(json_path)):
            results.append(fut.result())

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    single = p.add_argument_group("Single-film mode")
    single.add_argument("--json_path")
    single.add_argument("--audio_dir")
    single.add_argument("--source_video")
    single.add_argument("--output_dir")

    batch = p.add_argument_group("Batch mode")
    batch.add_argument("--batch_json_dir")
    batch.add_argument("--batch_audio_dir")
    batch.add_argument("--source_dir",
                       help="Dir with source videos named <imdb_id>.*")
    batch.add_argument("--batch_output_dir")

    p.add_argument("--search_window", type=float, default=120.0,
                   help="Total search window in seconds around expected timestamp "
                        "(default: 120 s).  Increase for heavily edited sources.")
    p.add_argument("--workers", type=int, default=4,
                   help="Thread workers for parallel clip searches (default: 4). "
                        "Note: mel spectrogram computation is single-threaded per film.")
    p.add_argument("--dry_run", action="store_true",
                   help="Detect offsets only; skip video extraction.")
    p.add_argument("--results_json", default="",
                   help="Save per-clip results to this JSON file.")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main():
    args = build_parser().parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if shutil.which("ffmpeg") is None:
        raise EnvironmentError("ffmpeg not found on $PATH")

    all_results: list[dict] = []

    # ── single-film ───────────────────────────────────────────────────────────
    if args.json_path:
        missing = [k for k in ("audio_dir", "source_video", "output_dir")
                   if not getattr(args, k)]
        if missing:
            raise ValueError(
                f"Single-film mode requires: {', '.join('--' + m for m in missing)}"
            )
        log.info(f"Single-film mode: {args.json_path}")
        all_results = process_film(
            json_path=args.json_path,
            audio_dir=args.audio_dir,
            source_video=args.source_video,
            output_dir=args.output_dir,
            search_window=args.search_window,
            workers=args.workers,
            dry_run=args.dry_run,
        )

    # ── batch ─────────────────────────────────────────────────────────────────
    elif args.batch_json_dir:
        missing = [k for k in ("batch_audio_dir", "source_dir", "batch_output_dir")
                   if not getattr(args, k)]
        if missing:
            raise ValueError(
                f"Batch mode requires: {', '.join('--' + m for m in missing)}"
            )
        for json_path in sorted(Path(args.batch_json_dir).glob("*.json")):
            imdb_id = json_path.stem
            src_matches = list(Path(args.source_dir).glob(f"{imdb_id}.*"))
            if not src_matches:
                log.warning(f"No source video for {imdb_id}, skipping.")
                continue
            log.info(f"\n{'─'*60}\n{imdb_id}  ←  {src_matches[0].name}")
            results = process_film(
                json_path=str(json_path),
                audio_dir=os.path.join(args.batch_audio_dir, imdb_id),
                source_video=str(src_matches[0]),
                output_dir=os.path.join(args.batch_output_dir, imdb_id),
                search_window=args.search_window,
                workers=args.workers,
                dry_run=args.dry_run,
            )
            all_results.extend(results)
    else:
        build_parser().print_help()
        return

    # ── summary ───────────────────────────────────────────────────────────────
    ok      = sum(1 for r in all_results if str(r.get("status","")).startswith("ok"))
    skipped = sum(1 for r in all_results if "skip"  in str(r.get("status","")))
    errors  = len(all_results) - ok - skipped
    log.info(
        f"\n{'═'*60}\n"
        f"Total: {len(all_results)} clips — "
        f"{ok} extracted, {skipped} skipped, {errors} errors"
    )

    if args.dry_run:
        deltas = [r["offset_delta"] for r in all_results if "offset_delta" in r]
        if deltas:
            log.info(
                f"Offset stats (s):  mean={np.mean(deltas):+.2f}  "
                f"std={np.std(deltas):.2f}  "
                f"min={np.min(deltas):+.2f}  max={np.max(deltas):+.2f}"
            )

    if args.results_json:
        with open(args.results_json, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        log.info(f"Results saved → {args.results_json}")


if __name__ == "__main__":
    main()
