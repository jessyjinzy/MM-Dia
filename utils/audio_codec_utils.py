"""
audio_codec_utils.py
──────────────────────────────────────────────────────────────────────────────
Audio encoding/decoding utilities for MM-DIA dataset release.

Due to copyright restrictions, we release audio in encoded form using
Descript Audio Codec (DAC) which provides high-quality lossy compression
with ~90:1 compression ratio while maintaining perceptual quality.

DAC is a neural audio codec that:
- Operates at 44.1kHz sampling rate
- Uses 8kbps bitrate (model_type="44khz")
- Provides near-transparent quality for speech and dialogue
- Is fully open-source and reproducible

Usage
─────
# Encode a batch of audio files
python utils/audio_codec_utils.py encode \
    --input_dir release/audio/movie/tt0114369 \
    --output_dir release/audio_encoded/movie/tt0114369

# Decode back to WAV for listening/training
python utils/audio_codec_utils.py decode \
    --input_dir release/audio_encoded/movie/tt0114369 \
    --output_dir decoded_audio/movie/tt0114369

# Batch process all clips in a batch
python utils/audio_codec_utils.py encode \
    --batch_input_dir release/audio/movie \
    --batch_output_dir release/audio_encoded/movie

Dependencies
────────────
pip install descript-audio-codec torch torchaudio
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional
import warnings

import torch
import torchaudio
from tqdm import tqdm

try:
    import dac
    from dac.utils import load_model
except ImportError:
    raise ImportError(
        "descript-audio-codec not installed. "
        "Please run: pip install descript-audio-codec"
    )

log = logging.getLogger(__name__)

# DAC model configuration
# "44khz" model: 44.1kHz, 8kbps, ~90:1 compression
# "24khz" model: 24kHz, 8kbps (alternative for 24kHz audio)
DEFAULT_MODEL_TYPE = "44khz"

warnings.filterwarnings("ignore", category=UserWarning, module="torch")


def get_device() -> str:
    """Auto-detect CUDA availability."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_dac_model(model_type: str = DEFAULT_MODEL_TYPE, device: Optional[str] = None):
    """
    Load DAC model.

    Args:
        model_type: "44khz" (default) or "24khz"
        device: "cuda" or "cpu" (auto-detected if None)

    Returns:
        Loaded DAC model
    """
    if device is None:
        device = get_device()

    log.info(f"Loading DAC model ({model_type}) on {device}...")
    model = load_model(model_type=model_type)
    model = model.to(device)
    model.eval()
    return model


def encode_audio_file(
    input_wav: str,
    output_dac: str,
    model,
    device: str,
) -> dict:
    """
    Encode a single WAV file to DAC format.

    Args:
        input_wav: Path to input .wav file
        output_dac: Path to output .dac file
        model: Loaded DAC model
        device: Device to use

    Returns:
        dict with encoding statistics
    """
    # Load audio
    signal, sr = torchaudio.load(input_wav)
    signal = signal.to(device)

    # DAC expects specific sample rate
    expected_sr = model.sample_rate
    if sr != expected_sr:
        log.debug(f"Resampling {input_wav} from {sr} to {expected_sr} Hz")
        resampler = torchaudio.transforms.Resample(sr, expected_sr).to(device)
        signal = resampler(signal)
        sr = expected_sr

    # Ensure mono → stereo if needed (DAC typically works with mono/stereo)
    if signal.shape[0] == 1 and model.encoder.in_channels == 1:
        pass  # mono is fine
    elif signal.shape[0] > 1:
        # Mix to mono for consistency
        signal = signal.mean(dim=0, keepdim=True)

    # Create DAC signal object
    dac_signal = dac.DACFile.from_audio(signal, sr)

    # Encode
    with torch.inference_mode():
        dac_signal = model.compress(dac_signal)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_dac)), exist_ok=True)
    dac_signal.save(output_dac)

    # Compute compression ratio
    input_size = os.path.getsize(input_wav)
    output_size = os.path.getsize(output_dac)
    ratio = input_size / output_size if output_size > 0 else 0

    return {
        "input": input_wav,
        "output": output_dac,
        "input_size_mb": input_size / (1024**2),
        "output_size_mb": output_size / (1024**2),
        "compression_ratio": ratio,
        "sample_rate": sr,
        "duration_sec": signal.shape[-1] / sr,
    }


def decode_audio_file(
    input_dac: str,
    output_wav: str,
    model,
    device: str,
) -> dict:
    """
    Decode a DAC file back to WAV.

    Args:
        input_dac: Path to input .dac file
        output_wav: Path to output .wav file
        model: Loaded DAC model
        device: Device to use

    Returns:
        dict with decoding statistics
    """
    # Load DAC file and decompress
    with torch.inference_mode():
        reconstructed = model.decompress(input_dac, verbose=False)

    # Save as WAV (AudioSignal uses .write() not .save())
    os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
    reconstructed.write(output_wav)

    return {
        "input": input_dac,
        "output": output_wav,
        "sample_rate": reconstructed.sample_rate,
        "duration_sec": reconstructed.duration,
    }


def process_clip_directory(
    clip_audio_dir: str,
    output_dir: str,
    model,
    device: str,
    mode: str = "encode",
) -> list[dict]:
    """
    Process all audio files in a clip directory.

    For MM-DIA structure: clip_XXX/ contains multiple .wav files
    (clip_XXX.wav, clip_XXX_44k.wav, clip_XXX_orig.wav, etc.)

    Args:
        clip_audio_dir: Directory containing audio files for one clip
        output_dir: Output directory
        model: DAC model
        device: Device to use
        mode: "encode" or "decode"

    Returns:
        List of processing results
    """
    results = []

    if mode == "encode":
        # Encode all .wav files
        wav_files = sorted(Path(clip_audio_dir).glob("*.wav"))
        for wav_path in wav_files:
            dac_path = os.path.join(output_dir, wav_path.stem + ".dac")
            try:
                result = encode_audio_file(str(wav_path), dac_path, model, device)
                results.append(result)
            except Exception as e:
                log.error(f"Failed to encode {wav_path}: {e}")
                results.append({"input": str(wav_path), "status": f"error: {e}"})

    else:  # decode
        # Decode all .dac files
        dac_files = sorted(Path(clip_audio_dir).glob("*.dac"))
        for dac_path in dac_files:
            wav_path = os.path.join(output_dir, dac_path.stem + ".wav")
            try:
                result = decode_audio_file(str(dac_path), wav_path, model, device)
                results.append(result)
            except Exception as e:
                log.error(f"Failed to decode {dac_path}: {e}")
                results.append({"input": str(dac_path), "status": f"error: {e}"})

    return results


def process_film(
    film_audio_dir: str,
    output_dir: str,
    model,
    device: str,
    mode: str = "encode",
) -> list[dict]:
    """
    Process all clips in a film directory.

    Args:
        film_audio_dir: Directory containing clip_XXX subdirectories
        output_dir: Output directory
        model: DAC model
        device: Device to use
        mode: "encode" or "decode"

    Returns:
        List of all processing results
    """
    all_results = []

    # Find all clip directories (or .dac files if decoding a flat structure)
    if mode == "encode":
        clip_dirs = sorted([d for d in Path(film_audio_dir).iterdir() if d.is_dir()])
    else:
        clip_dirs = sorted([d for d in Path(film_audio_dir).iterdir() if d.is_dir()])

    for clip_dir in tqdm(clip_dirs, desc=os.path.basename(film_audio_dir)):
        out_clip_dir = os.path.join(output_dir, clip_dir.name)
        results = process_clip_directory(
            str(clip_dir), out_clip_dir, model, device, mode
        )
        all_results.extend(results)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("mode", choices=["encode", "decode"],
                        help="Operation mode: encode WAV to DAC, or decode DAC to WAV")

    # Single film mode
    single = parser.add_argument_group("Single film mode")
    single.add_argument("--input_dir",
                        help="Input directory (film with clip_XXX subdirs)")
    single.add_argument("--output_dir",
                        help="Output directory")

    # Batch mode
    batch = parser.add_argument_group("Batch mode")
    batch.add_argument("--batch_input_dir",
                       help="Batch directory containing multiple film subdirs")
    batch.add_argument("--batch_output_dir",
                       help="Batch output directory")

    # Model config
    parser.add_argument("--model_type", default=DEFAULT_MODEL_TYPE,
                        choices=["44khz", "24khz"],
                        help="DAC model type (default: 44khz)")
    parser.add_argument("--device", default=None,
                        help="Device: 'cuda' or 'cpu' (auto-detect if not specified)")
    parser.add_argument("--results_json", default="",
                        help="Save processing results to JSON file")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    device = args.device if args.device else get_device()
    log.info(f"Using device: {device}")

    # Load model once
    model = load_dac_model(args.model_type, device)

    all_results = []

    # Single film mode
    if args.input_dir:
        if not args.output_dir:
            parser.error("--output_dir required for single film mode")

        log.info(f"Processing {args.mode}: {args.input_dir} → {args.output_dir}")
        all_results = process_film(
            args.input_dir, args.output_dir, model, device, args.mode
        )

    # Batch mode
    elif args.batch_input_dir:
        if not args.batch_output_dir:
            parser.error("--batch_output_dir required for batch mode")

        film_dirs = sorted([d for d in Path(args.batch_input_dir).iterdir() if d.is_dir()])
        log.info(f"Batch mode: processing {len(film_dirs)} films...")

        for film_dir in film_dirs:
            film_id = film_dir.name
            out_dir = os.path.join(args.batch_output_dir, film_id)
            log.info(f"\n{'─'*60}\n{film_id}")

            results = process_film(str(film_dir), out_dir, model, device, args.mode)
            all_results.extend(results)

    else:
        parser.print_help()
        return

    # Summary
    successful = sum(1 for r in all_results if "compression_ratio" in r or "duration_sec" in r)
    failed = len(all_results) - successful

    log.info(f"\n{'═'*60}")
    log.info(f"Total: {len(all_results)} files — {successful} successful, {failed} failed")

    if args.mode == "encode" and successful > 0:
        ratios = [r["compression_ratio"] for r in all_results if "compression_ratio" in r]
        if ratios:
            import numpy as np
            log.info(
                f"Compression ratio: mean={np.mean(ratios):.1f}x, "
                f"median={np.median(ratios):.1f}x, "
                f"range=[{np.min(ratios):.1f}x, {np.max(ratios):.1f}x]"
            )
            total_input = sum(r["input_size_mb"] for r in all_results if "input_size_mb" in r)
            total_output = sum(r["output_size_mb"] for r in all_results if "output_size_mb" in r)
            log.info(f"Total size: {total_input:.1f} MB → {total_output:.1f} MB")

    if args.results_json:
        with open(args.results_json, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        log.info(f"Results saved → {args.results_json}")


if __name__ == "__main__":
    main()
