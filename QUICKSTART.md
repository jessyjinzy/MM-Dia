# Quick Start Guide for MM-DIA Dataset

This guide helps you get started with the MM-DIA dataset in 5 minutes.

## Prerequisites

- Python 3.10+
- Legal access to source movies/TV episodes
- ~20 GB disk space for encoded audio
- ~50-100 GB for full dataset (after video extraction)

## Installation

```bash
# Clone the repository
git clone https://github.com/jessyjinzy/MM-Dia.git
cd MM-Dia

# Create environment
conda create -n mmdia python=3.10
conda activate mmdia

# Install dependencies
pip install -r requirements.txt
pip install flash-attn  # optional, for faster training
```

## Download Dataset

Download from HuggingFace (access request required):
- JSON annotations: `json/` (~500 MB)
- Encoded audio: `audio_encoded/` (~15-20 GB)

See [MM-DIA_EULA.pdf](MM-DIA_EULA.pdf) for access request process.

## Reconstruct Dataset

### Step 1: Decode Audio

```bash
# Decode DAC-compressed audio to WAV
python audio_codec_utils.py decode \
    --batch_input_dir audio_encoded/movie \
    --batch_output_dir audio/movie
```

Expected time: ~30 min for one batch on GPU, ~2 hours on CPU

### Step 2: Extract Video from Your Sources

Place your source movie files in a directory:
```
my_sources/
├── tt0114369.mkv
├── tt0112641.mp4
└── ...
```

Run extraction:
```bash
python av_align_and_extract.py \
    --batch_json_dir  json/movie \
    --batch_audio_dir audio/movie \
    --source_dir      my_sources \
    --batch_output_dir video/movie
```

Expected time: ~10-30 min per film depending on length

### Verify Extraction

```bash
# Check extracted clips
ls video/movie/tt0114369/

# Verify alignment quality (should see low offset deltas)
python av_align_and_extract.py \
    --batch_json_dir json/movie \
    --batch_audio_dir audio/movie \
    --source_dir my_sources \
    --batch_output_dir video/movie \
    --dry_run \
    --results_json alignment_check.json
```

Good alignment: mean offset < ±0.5s, >99% success rate

## Train Model

### Step 1: Preprocess Data

```bash
# Merge JSON annotations
python make_json.py --prefix movie

# Merge all batches and re-index speakers
python make_data.py
# Output: output_data/mm-dia.jsonl
```

### Step 2: Extract Audio Tokens

```bash
cd higgs_finetune

python extract_higgs.py \
    --data_path ../output_data/mm-dia.jsonl \
    --output_dir ../output_data/mm-dia

python make_split.py \
    --input_file ../output_data/mm-dia \
    --meta_file train_eval_test_hard_ids.json \
    --output_dir ../output_data/mm_dia_splits/
```

### Step 3: Train

```bash
accelerate launch --config_file accl_config.yaml train_higgs.py \
    --train_data_path ../output_data/mm_dia_splits/train \
    --val_data_path   ../output_data/mm_dia_splits/eval \
    --batch_size 3 \
    --lr 1e-5 \
    --accumulation_steps 8 \
    --instruction \
    --freeze_text_encoder \
    --semantic_amplification 4.0 \
    --epochs 25 \
    --output_dir ../exp/my_run
```

Expected training time: ~3-5 days on 4×A100 GPUs

## Run Inference

### Gradio Demo

```bash
cd higgs_infer
python app.py --model_path /path/to/checkpoint
# Open browser to http://localhost:7860
```

### Command Line

```bash
cd higgs_infer
python infer.py \
    --model_path /path/to/checkpoint \
    --text "Hello, how are you today?" \
    --style "friendly and warm tone" \
    --output generated.wav
```

## Evaluate Model

```bash
cd evaluation

# Edit evaluate.sh to set paths
WAV_ROOT="path/to/generated/audio"
MODEL_PATH="path/to/whisper-large-v3"
TEST_LIST="./evaluation/examples/_gt_trans_wo_speaker.tsv"

# Run full evaluation suite
bash evaluate.sh
```

Metrics computed:
- WER (Word Error Rate)
- UTMOS (Speech Quality)
- cp-WER (Turn-taking Accuracy)
- sa-SIM (Speaker Similarity)
- Gemini-as-Judge scores

## Troubleshooting

### Audio decoding is slow
```bash
# Use GPU if available
python audio_codec_utils.py decode --device cuda ...
```

### Video extraction fails
```bash
# Increase search window
python av_align_and_extract.py --search_window 240 ...

# Check alignment first
python av_align_and_extract.py --dry_run ...
```

### Out of memory during training
```bash
# Reduce batch size
--batch_size 2 --accumulation_steps 12
```

### Import errors
```bash
# Ensure all dependencies installed
pip install -r requirements.txt
pip install flash-attn
```

## Next Steps

- Read [README.md](README.md) for complete documentation
- See [DATASET_RELEASE.md](DATASET_RELEASE.md) for advanced usage
- Check [examples/](higgs_finetune/examples/) for generation examples
- Review [evaluation/](evaluation/) for metrics details


## Citation

```bibtex
@inproceedings{jin2026mmdia,
  title     = {From Natural Alignment to Conditional Controllability in Multimodal Dialogue},
  author    = {Jin, Zeyu and Zhou, Songtao and Wang, Haoyu and Tian, Minghao and Yun, Kaifeng and Chen, Zhuo and Qin, Xiaoyu and Jia, Jia},
  booktitle = {The International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
