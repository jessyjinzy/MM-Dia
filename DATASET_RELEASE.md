# MM-DIA Dataset Release Guide

This document explains the dataset release strategy, copyright considerations, and instructions for users to reconstruct the full dataset from their own source videos.

---

## Copyright & Release Strategy

### What We Release

Due to copyright restrictions on original movie/TV content, we release:

✅ **JSON annotations** – All dialogue-level metadata including:
- Utterance transcripts with timestamps
- Speaker information
- Affective triplet tags (Relationship, InteractionMode, EmotionalTone)
- Style descriptions
- Emotion intensity/volatility scores

✅ **Encoded audio** – Dialogue audio encoded with [Descript Audio Codec (DAC)](https://github.com/descriptinc/descript-audio-codec):
- High-quality neural audio codec (44.1kHz, 8kbps)
- ~90:1 compression ratio with near-transparent quality
- Fully open-source and reproducible
- Can be decoded back to WAV for training/listening

❌ **Video files** – NOT released due to copyright

### What Users Need to Provide

Users must obtain their own legal copies of the source movies/TV episodes. The dataset provides tools to automatically extract matching video clips from user-provided sources.

---

## Audio Encoding with DAC

We use **Descript Audio Codec (DAC)** for audio compression. DAC is a state-of-the-art neural audio codec that:

- Maintains perceptual quality nearly indistinguishable from original
- Reduces file size by ~90× (from several GB to tens of MB per batch)
- Is fully open-source (MIT license)
- Supports both encoding and lossless decoding

### Why DAC?

1. **Legal**: Compressed audio representations are less likely to be considered direct copies for copyright purposes
2. **Practical**: Dramatically reduces download size (360+ hours → manageable dataset)
3. **Quality**: Near-transparent quality for speech/dialogue (much better than MP3/AAC at equivalent bitrates)
4. **Research-friendly**: Fully open-source, reproducible, and well-documented

### Encoding Audio (for dataset creators)

```bash
# Encode a single film's audio
python audio_codec_utils.py encode \
    --input_dir audio/movie/tt0114369 \
    --output_dir audio_encoded/movie/tt0114369

# Encode an entire batch
python audio_codec_utils.py encode \
    --batch_input_dir audio/movie \
    --batch_output_dir audio_encoded/movie

# Check compression statistics
python audio_codec_utils.py encode \
    --batch_input_dir audio/movie \
    --batch_output_dir audio_encoded/movie \
    --results_json encoding_stats.json
```

### Decoding Audio (for users)

```bash
# Decode a single film's audio back to WAV
python audio_codec_utils.py decode \
    --input_dir audio_encoded/movie/tt0114369 \
    --output_dir audio/movie/tt0114369

# Decode an entire batch
python audio_codec_utils.py decode \
    --batch_input_dir audio_encoded/movie \
    --batch_output_dir audio/movie
```

The decoded audio can be used directly for model training or listening.

---

## Video Reconstruction from User Sources

Since we don't distribute video files, users must extract video clips from their own legally-obtained source files. We provide `av_align_and_extract.py` to automate this process.

### How It Works

The script uses **audio fingerprinting** to locate each dialogue clip in your source video:

1. **Mel-spectrogram matching**: Computes mel spectrograms of both reference (our audio) and source (your video)
2. **Cross-correlation search**: Finds exact temporal alignment via FFT-based correlation
3. **Speed tolerance**: Handles different video speeds (PAL/NTSC, encode rate variations) by testing multiple speed candidates
4. **Automatic extraction**: Uses ffmpeg to extract aligned video segments

### Why This Works Despite Different Sources

Different releases of the same film (Blu-ray, DVD, streaming, regional versions) may have:
- Different speeds (PAL vs NTSC: ~4% difference)
- Different cuts (extended/theatrical versions)
- Different timestamps (missing/added scenes)

Our script handles these differences by:
- Testing multiple speed candidates (0.96×, 0.98×, 1.0×, 1.02×, 1.04×)
- Using a configurable search window (default: ±120s around expected timestamp)
- Normalizing audio for robust matching

### Usage

#### Single Film

```bash
python av_align_and_extract.py \
    --json_path    json/movie/tt0114369.json \
    --audio_dir    audio/movie/tt0114369 \
    --source_video /path/to/your/tt0114369.mkv \
    --output_dir   video/movie/tt0114369
```

#### Batch Processing

```bash
python av_align_and_extract.py \
    --batch_json_dir  json/movie \
    --batch_audio_dir audio/movie \
    --source_dir      /path/to/your/sources/movie \
    --batch_output_dir video/movie
```

The script expects source videos to be named `<imdb_id>.*` (e.g., `tt0114369.mkv`, `tt0114369.mp4`)

#### Options

```bash
--search_window 240     # Increase search window for heavily edited sources
--workers 8             # Use more parallel workers for faster extraction
--dry_run               # Test alignment without extracting (see offsets)
--results_json log.json # Save per-clip alignment statistics
```

### Handling Multiple Source Versions

If you have multiple subtitle files (SRT) with slight differences, you can:

1. **Use the most accurate subtitle**: Our JSON annotations already include corrected transcripts
2. **Cross-validate timestamps**: The audio fingerprinting is robust to small timing differences
3. **Check alignment quality**: Use `--dry_run` to see offset statistics before extracting

Example workflow with multiple SRTs:

```bash
# Step 1: Test alignment with first source
python av_align_and_extract.py \
    --json_path json/movie/tt0114369.json \
    --audio_dir audio/movie/tt0114369 \
    --source_video sources/version_A/tt0114369.mkv \
    --output_dir video_test \
    --dry_run \
    --results_json alignment_A.json

# Step 2: Test with second source
python av_align_and_extract.py \
    --json_path json/movie/tt0114369.json \
    --audio_dir audio/movie/tt0114369 \
    --source_video sources/version_B/tt0114369.mkv \
    --output_dir video_test \
    --dry_run \
    --results_json alignment_B.json

# Step 3: Compare results and choose the source with better alignment
# (lower mean offset delta and higher correlation scores)

# Step 4: Extract from the best source
python av_align_and_extract.py \
    --json_path json/movie/tt0114369.json \
    --audio_dir audio/movie/tt0114369 \
    --source_video sources/version_A/tt0114369.mkv \
    --output_dir video/movie/tt0114369
```

### Expected Alignment Accuracy

With good-quality source videos, you should expect:
- **Mean offset**: < ±0.5 seconds
- **Success rate**: > 99% of clips
- **Speed detection**: Automatic detection of source speed within 0.02× tolerance

If alignment fails for many clips:
- Increase `--search_window` (default 120s)
- Check if your source is a significantly different cut (director's cut, etc.)
- Verify audio quality (some heavily compressed sources may not align well)

---

## Complete Dataset Preparation Workflow

For users who download the released dataset:

```bash
# 1. Download released data (JSON + encoded audio)
# Assume downloaded to: release_download/

# 2. Decode audio from DAC to WAV
python audio_codec_utils.py decode \
    --batch_input_dir release_download/audio_encoded/movie \
    --batch_output_dir audio/movie

# 3. Extract video from your own sources
python av_align_and_extract.py \
    --batch_json_dir  release_download/json/movie \
    --batch_audio_dir audio/movie \
    --source_dir      /path/to/your/movie/sources \
    --batch_output_dir video/movie

# 4. Now you have the complete dataset:
# - json/movie/          (annotations)
# - audio/movie/         (decoded audio)
# - video/movie/         (extracted from your sources)
```

---

## Dataset Structure After Reconstruction

```
dataset/
├── json/
│   └── movie/
│       ├── tt0114369.json
│       └── ...
├── audio/
│   └── movie/
│       └── tt0114369/
│           ├── clip_000/
│           │   ├── clip_000.wav        # Decoded from .dac
│           │   ├── clip_000_44k.wav
│           │   └── ...
│           └── ...
└── video/
    └── movie/
        └── tt0114369/
            ├── clip_000.mp4            # Extracted from your source
            ├── clip_001.mp4
            └── ...
```

---

## Technical Notes

### Audio Fingerprinting Robustness

The alignment script is robust to:
- ✅ Speed variations (PAL/NTSC conversion, encode rate differences)
- ✅ Different audio codecs in source video
- ✅ Small cuts/edits within search window
- ✅ Volume normalization differences

It will NOT work well with:
- ❌ Completely different language dubs
- ❌ Director's cuts with substantially different scene order
- ❌ Severely degraded audio quality (e.g., cam rips)

### DAC Technical Details

- **Model**: `dac_44khz` (44.1kHz, 8kbps)
- **Architecture**: VQ-VAE with residual quantization
- **Latency**: ~5ms (suitable for real-time applications)
- **Compression**: ~90:1 (44.1kHz 16-bit stereo WAV → 8kbps DAC)

Alternative: The script also supports `dac_24khz` for 24kHz audio files.

---

## Troubleshooting

### Audio Decoding Issues

**Problem**: `descript-audio-codec` import error

**Solution**: Install DAC:
```bash
pip install descript-audio-codec
```

**Problem**: CUDA out of memory during encoding/decoding

**Solution**: Use CPU or process smaller batches:
```bash
python audio_codec_utils.py encode --device cpu ...
```

### Video Extraction Issues

**Problem**: Many clips fail to align

**Solution 1**: Increase search window:
```bash
python av_align_and_extract.py --search_window 240 ...
```

**Solution 2**: Check if you have the wrong version (different cut):
```bash
python av_align_and_extract.py --dry_run --results_json check.json ...
# Inspect check.json for systematic offset patterns
```

**Problem**: ffmpeg not found

**Solution**: Install ffmpeg:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Conda
conda install ffmpeg
```

---

## License & Citation

The MM-DIA annotations and code are released for research purposes only. Users are responsible for ensuring they have legal access to source video content.

When using this dataset, please cite:

```bibtex
@inproceedings{jin2026mmdia,
  title     = {From Natural Alignment to Conditional Controllability in Multimodal Dialogue},
  author    = {Jin, Zeyu and Zhou, Songtao and Wang, Haoyu and Tian, Minghao and Yun, Kaifeng and Chen, Zhuo and Qin, Xiaoyu and Jia, Jia},
  booktitle = {The International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

For questions or issues, please open an issue on our GitHub repository.
