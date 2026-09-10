# MM-DIA: A Large-Scale Expressive Multimodal Dialogue Dataset

This is the official repository of the ICLR 2026 paper "From Natural Alignment to Conditional Controllability in Multimodal Dialogue".

For details of the dataset and Multimodal Dialogue Generation tasks, please refer to our [Paper](https://cloud.tsinghua.edu.cn/f/01f98fa0f9c34983887b/?dl=1) and [Demo Page](https://mmdiaiclr26.github.io/mmdiaiclr26)

---

## Overview

MM-DIA is a large-scale multimodal dialogue dataset curated from movies and TV series, featuring **360+ hours** and **54,700 dialogue clips** with fine-grained interaction-level annotations. It supports three dialogue generation tasks under explicit and implicit cross-modal control:

- **Task 1 – Style-Controllable Dialogue Speech Synthesis**: generate multi-speaker dialogue audio conditioned on text transcripts and expressive style prompts.
- **Task 2 – Vision-Conditioned Dialogue Speech Synthesis**: synthesize speech guided by visual dialogue context (speaker appearance, body language, scene).
- **Task 3 – Speech-Driven Dialogue Video Generation**: generate dialogue video from conversational audio and transcripts.

MM-DIA-BENCH (309 highly expressive dual-speaker dialogues) provides a rigorous benchmark for evaluating cross-modal style consistency.

---

## 🚀 Quick Start

Get started in 3 steps (~2 hours):

```bash
# 1. Install dependencies
conda create -n mmdia python=3.10 && conda activate mmdia
pip install -r requirements.txt

# 2. Decode audio from released DAC files
python utils/audio_codec_utils.py decode \
    --batch_input_dir audio_encoded/movie \
    --batch_output_dir audio/movie

# 3. (For video usage) Extract video from your legal sources
python av_align_and_extract.py \
    --batch_json_dir json/movie \
    --batch_audio_dir audio/movie \
    --source_dir /path/to/your/movies \
    --batch_output_dir video/movie
```

📖 See [QUICKSTART.md](QUICKSTART.md) for detailed 5-minute setup guide  
📦 See [DATASET_RELEASE.md](DATASET_RELEASE.md) for copyright & reconstruction details

---

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total duration** | 360+ hours |
| **Dialogue clips** | 54,700 |
| **Source films** | ~100 movies + TV episodes |
| **Languages** | English (primary) |
| **Sample rate** | 24kHz / 44.1kHz |
| **Avg clip duration** | 15-30 seconds |
| **Speakers per clip** | 2-4 (dialogue scenes) |
| **Annotation fields** | Transcripts, timestamps, affective tags, style descriptions, emotion scores |

---

## Repository Structure

```
.
├── mmdia/                        # Data curation pipeline
│   ├── extraction_main.py        # Entry point: dialogue extraction
│   ├── annotation_main.py        # Entry point: dialogue annotation
│   ├── run.sh                    # End-to-end pipeline script
│   ├── assets/                   # Example input (video + SRT)
│   ├── output/                   # Example output (clips + JSON)
│   └── src/
│       ├── extraction_pipeline.py
│       ├── annotation_pipeline.py
│       └── utils/
│           ├── annotation_prompts.py   # Prompts for Gemini-2.5-pro annotation
│           ├── video_ops.py            # FFmpeg-based clip extraction
│           ├── srt_parser.py           # Subtitle alignment
│           ├── llm_client.py           # API wrapper
│           └── ...
│
├── higgs_finetune/               # Model training (Higgs-Audio-V2 SFT)
│   ├── train_higgs.py            # Training script
│   ├── extract_higgs.py          # Audio token pre-extraction
│   ├── make_split.py             # Train/eval/test split
│   ├── infer_higgs.py            # Command-line inference
│   ├── accl_config.yaml          # Accelerate config (multi-GPU)
│   └── boson_multimodal/         # Model architecture (Higgs-Audio-V2)
│
├── higgs_infer/                  # Model inference & Gradio demo
│   ├── infer.py                  # Batch inference
│   ├── app.py                    # Gradio web demo
│   └── higgs_audio/              # Model architecture (inference copy)
│
├── evaluation/                   # Evaluation scripts
│   ├── evaluate.sh               # Full evaluation pipeline
│   ├── evaluate_wer_seedtts.py   # WER (Whisper-based)
│   ├── evaluate_utmos.py         # UTMOS speech quality
│   ├── evaluate_cpwer.py         # cp-WER (speaker turn-taking accuracy)
│   ├── evaluate_sasim.py         # sa-SIM (speaker-aware similarity, MFA-based)
│   ├── evaluate_gemini.py        # Gemini-as-Judge subjective scoring
│   └── gemini_diarization.py     # Gemini-based speaker diarization 
│
├── av_align_and_extract.py       # Video reconstruction from user sources
├── utils/
│   ├── audio_codec_utils.py      # Audio encoding/decoding with DAC
│   ├── make_json.py              # Per-batch JSON merge utility
│   ├── make_data.py              # Full dataset merge + speaker re-indexing
│   ├── format_utils.py           # Shared format helpers
│   └── srt_calibration_pipeline.py
├── requirements.txt
├── DATASET_RELEASE.md            # Detailed guide on copyright & data reconstruction
├── EULA.md                       # End User License Agreement
└── README.md                     # This file
```

---

## ⚠️ Copyright Notice & Dataset Release Strategy

**Due to copyright restrictions, we do NOT release original video files.**

### What We Release:
- ✅ **JSON annotations** (transcripts, timestamps, style descriptions, affective tags)
- ✅ **Encoded audio** (using Descript Audio Codec - DAC for compression)
- ❌ **Video files** (NOT included - users must provide their own sources)

### What Users Need to Do:
1. **Obtain legal copies** of source movies/TV episodes
2. **Decode audio** from released DAC files to WAV (using `audio_codec_utils.py`)
3. **Extract video clips** from your sources using audio fingerprinting (using `av_align_and_extract.py`)

See [DATASET_RELEASE.md](DATASET_RELEASE.md) for detailed instructions on:
- Audio encoding/decoding with DAC
- Video reconstruction from multiple source versions
- Handling different subtitle (SRT) files
- Troubleshooting alignment issues

### What to Download for Your Research

| Research Area | JSON | Audio | Video | What to Reconstruct | Use Cases |
|---------------|:----:|:-----:|:-----:|---------------------|-----------|
| **Spoken Dialogue Generation** | ✅ | ✅ | ❌ | Audio (decode only) | Multi-speaker TTS, conversational speech synthesis, dialogue modeling |
| **Dialogue Understanding** | ✅ | ✅ | ❌ | Audio (decode only) | Speaker diarization, emotion recognition, turn-taking analysis |
| **Audio-Visual Speech** | ✅ | ✅ | ✅ | Audio (decode) + Video (extract) | Audio-visual speech generation, lip-sync, talking face synthesis |
| **Video Generation** | ✅ | ✅ | ✅ | Audio (decode) + Video (extract) | Speech-driven animation, gesture generation, multimodal dialogue video |
| **Benchmark Evaluation** | ✅ | ✅ | ✅* | Audio + Video | Direct evaluation without video extraction (*MM-DIA-BENCH videos provided) |

**Storage Requirements:**
- JSON annotations: ~500 MB
- Encoded audio: ~15-20 GB  
- Decoded audio: ~40-50 GB
- Extracted video: ~300-500 GB (depends on source quality)
- MM-DIA-BENCH videos: ~5-8 GB (309 clips, provided directly)

**Time Estimates:**
- Decode audio: ~30 min per batch (GPU) or ~2 hours (CPU)
- Extract video: ~10-30 min per film (depends on length and search window)
- MM-DIA-BENCH: Ready to use, no extraction needed

---

## Dataset Layout

The dataset is organized into three modality directories that share a consistent three-level hierarchy: `batch → movie/episode → clip`.

```
# Released structure (what you download)
json/                             # ✅ Released
└── <batch_id>/
    └── <imdb_id_or_episode_id>.json    # Clip-level + dialogue-level annotations

audio_enc/                    # ✅ Released (DAC compressed)
└── <batch_id>/
    └── <imdb_id_or_episode_id>/
        ├── clip_000/
        │   ├── clip_000.dac            # Encoded with DAC codec
        │   ├── clip_000_44k.dac
        │   ├── clip_000_no_vocals.dac
        │   └── clip_000_orig.dac
        └── ...

# Reconstructed by user (after decoding + extraction)
audio/                            # Decoded from audio_enc/
└── <batch_id>/
    └── <imdb_id_or_episode_id>/
        ├── clip_000/
        │   ├── clip_000.wav            # Denoised audio (24 kHz)
        │   ├── clip_000_44k.wav        # Denoised audio (44.1 kHz)
        │   ├── clip_000_no_vocals.wav  # Separated background (BGM, ambience)
        │   └── clip_000_orig.wav       # Volume-normalised original audio
        └── ...

video/                            # ❌ NOT released - extract from your sources
└── <batch_id>/
    └── <imdb_id_or_episode_id>/
        ├── clip_000.mp4
        ├── clip_001.mp4
        └── ...
```

`video/<ID>/clip_k.mp4`, `audio/<ID>/clip_k/clip_k.wav`, and the corresponding entry in `json/<ID>.json` all refer to the same temporal segment.

### Quick Start: Reconstruct the Dataset

```bash
# Step 1: Decode audio from DAC to WAV
python utils/audio_codec_utils.py decode \
    --batch_input_dir audio_encoded/movie \
    --batch_output_dir audio/movie

# Step 2: Extract video from your own legally-obtained source files
python av_align_and_extract.py \
    --batch_json_dir  json/movie \
    --batch_audio_dir audio/movie \
    --source_dir      /path/to/your/movie/sources \
    --batch_output_dir video/movie
```

See [DATASET_RELEASE.md](DATASET_RELEASE.md) for detailed instructions.

### JSON Annotation Schema

Each `.json` file contains a list of dialogue clips. Key fields per clip:

| Field | Description |
|-------|-------------|
| `clip_id` | Unique clip identifier |
| `utterances` | List of utterances with speaker, text, timestamps, non-verbal sounds |
| `affective_triplet` | `{Relationship, InteractionMode, EmotionalTone}` tags |
| `description` | Free-style per-speaker turn-level style trajectory |
| `emotion_intensity` | Dialogue-level emotional intensity score (1–10) |
| `emotion_volatility` | Speaker-level emotional volatility score (1–10) |
| `speaker_visibility` | Whether main speakers are visible in keyframes |

---

## 📋 Prerequisites

- **Python**: 3.10+
- **PyTorch**: 2.5.1+ with CUDA 11.8+
- **FFmpeg**: Required for video/audio processing
- **Disk Space**: 
  - ~20 GB for encoded audio
  - ~50-100 GB for full dataset after reconstruction
- **Legal Access**: Source movies/TV episodes for video reconstruction
- **GPU**: Recommended for training and audio encoding/decoding

---

## Getting Started

### 1. Environment Setup

```bash
conda create -n mmdia python=3.10
conda activate mmdia
pip install -r requirements.txt
pip install flash-attn          # optional, recommended for training
```

### 2. Dataset Reconstruction

If you downloaded the released dataset, reconstruct the full dataset:

```bash
# Decode audio from DAC to WAV
python utils/audio_codec_utils.py decode \
    --batch_input_dir audio_encoded/movie \
    --batch_output_dir audio/movie

# Extract video clips from your own source movies
python av_align_and_extract.py \
    --batch_json_dir  json/movie \
    --batch_audio_dir audio/movie \
    --source_dir      /path/to/your/sources \
    --batch_output_dir video/movie
```

See [DATASET_RELEASE.md](DATASET_RELEASE.md) for detailed instructions on handling multiple source versions and troubleshooting.

---

## Data Pipeline (MM-DIA Curation)

**Note:** This section is for researchers who want to create their own dialogue datasets using our curation pipeline. If you're using the released MM-DIA dataset, you can skip to the [Training](#training) section after completing dataset reconstruction.

The curation pipeline processes raw movie/TV video files with subtitle (`.srt`) files and produces segmented dialogue clips with fine-grained annotations.

### Input

- Raw video file (`.mp4` or similar)
- Subtitle file (`.srt`, optionally multi-source for calibration)

### Run

```bash
cd mmdia

# Full pipeline: extraction + annotation for a single film
bash run.sh

# Or run steps individually:

# Step 1–3: Dialogue extraction (scene segmentation → speaker attribution → clip cutting)
python extraction_main.py \
    --video assets/tt0327137.mp4 \
    --srt   assets/tt0327137.srt \
    --output output/tt0327137

# Step 4: Dialogue-level expressiveness annotation (requires Gemini-2.5-pro API)
python annotation_main.py \
    --input  output/tt0327137/tt0327137.json \
    --output output/tt0327137/tt0327137_annotated.json
```

The pipeline implements:
- **Subtitle calibration** – aligns multi-source SRT files with ASR output to correct timestamps.
- **Tolerance-enhanced scene boundary detection** – VLM-based scene continuity check with a dynamic keyframe buffer (Algorithm 1 in the paper).
- **Multimodal speaker attribution** – Gemini-2.5-flash assigns speaker identities from synchronised audio-visual segments.
- **Affective annotation** – Gemini-2.5-pro generates Affective Triplet tags and free-style descriptions.

---

## Data Preprocessing (for Training)

Before training, merge the per-film JSON annotations and extract audio tokens.

### Step 1 – Merge JSON annotations per batch

```bash
# Single batch
python utils/make_json.py --prefix movie

# Custom paths
python utils/make_json.py --prefix movie \
  --annotation_dir ../release/json \
  --audio_root     ../release/audio \
  --output_dir     output

# All batches
for subdir in ../release/video/*/; do
  python utils/make_json.py --prefix "$(basename "$subdir")"
done
```

### Step 2 – Merge all batches and re-index speakers

```bash
# Output: output_data/mm-dia.jsonl
python utils/make_data.py
```

### Step 3 – Pre-extract Higgs audio tokens

```bash
cd higgs_finetune
python extract_higgs.py \
    --data_path  ../output_data/mm-dia.jsonl \
    --output_dir ../output_data/mm-dia

python make_split.py \
    --input_file  ../output_data/mm-dia \
    --meta_file   train_eval_test_hard_ids.json \
    --output_dir  ../output_data/mm_dia_splits/
```

---

## Training

We fine-tune **Higgs-Audio-V2** on MM-DIA to enable style-controllable dialogue speech synthesis (Task 1). Training uses Accelerate for multi-GPU distributed training.

```bash
cd higgs_finetune

accelerate launch --config_file accl_config.yaml train_higgs.py \
    --train_data_path ../output_data/mm_dia_splits/train \
    --val_data_path   ../output_data/mm_dia_splits/eval  \
    --batch_size            3    \
    --lr                    1e-5 \
    --accumulation_steps    8    \
    --epochs               25    \
    --instruction                \
    --freeze_text_encoder        \
    --semantic_amplification 4.0 \
    --output_dir ../exp/run1
```

### Classifier-Free Guidance (CFG)

During training, style conditions are randomly dropped to support CFG at inference. Recommended inference settings (from ablation study):

```
--pad_left --guidance_scale 2.0 --cfg_text
```

| Flag | Description |
|------|-------------|
| `--guidance_scale` | CFG strength; higher = stronger style adherence |
| `--pad_left` | Left-pad input sequences |
| `--cfg_text` | Also drop transcript condition in CFG mode |

---

## Inference

### Command-line inference

```bash
cd higgs_infer
python infer.py --model_path higgs-audio-v2-sft-mm-dia
```

### Gradio demo

```bash
cd higgs_infer
python app.py --model_path higgs-audio-v2-sft-mm-dia
```

---

## Evaluation

The evaluation suite covers all metrics reported in the paper. Run the full pipeline with:

```bash
cd evaluation
bash evaluate.sh
```

Set the following paths in `evaluate.sh` before running:

| Variable | Description |
|----------|-------------|
| `WAV_ROOT` | Directory of model output folders to evaluate |
| `MODEL_PATH` | Path to `whisper-large-v3` checkpoint |
| `TEST_LIST` | Reference TSV without speaker labels (for WER) |
| `TEST_LIST_SPEAKER` | Reference TSV with speaker labels (for MFA / sa-SIM) |
| `TEST_TRANS_LAB` | Directory of `.lab` files for MFA alignment |
| `TEST_SPEAKER_AWARE_LIST` | JSON mapping for cp-WER speaker-aware evaluation |

### Metrics

| Metric | Script | Description |
|--------|--------|-------------|
| **WER** | `evaluate_wer_seedtts.py` | Word error rate via Whisper-large-v3 |
| **UTMOS** | `evaluate_utmos.py` | Predicted MOS for speech quality |
| **cp-WER** | `evaluate_cpwer.py` | Speaker turn-taking accuracy |
| **sa-SIM** | `evaluate_sasim.py` | Speaker-aware similarity (requires MFA alignment) |
| **Gemini-as-Judge** | `evaluate_gemini.py` | Subjective scoring: Spontaneity, Coherence, Intelligibility, Similarity, Quality, Instruction Following |

---

## MM-DIA-BENCH

MM-DIA-BENCH is a curated subset of 309 highly expressive dual-speaker dialogues with guaranteed speaker visibility, used for Tasks 2 and 3 evaluation. It is included in the released JSON annotations.

---

## Citation

```bibtex
@inproceedings{jin2026mmdia,
  title     = {From Natural Alignment to Conditional Controllability in Multimodal Dialogue},
  author    = {Jin, Zeyu and Zhou, Songtao and Wang, Haoyu and Tian, Minghao and Yun, Kaifeng and Chen, Zhuo and Qin, Xiaoyu and Jia, Jia},
  booktitle = {The International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

---

## License

The code in this repository is released under the MIT License.  
The MM-DIA dataset is released under a custom research-only license — see [MM-DIA_EULA.pdf](MM-DIA_EULA.pdf) for terms and the dataset access request form.

---

