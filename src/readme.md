# Data Layout

```markdown
video/
    ├── <batch_id>/
    │   ├── <imdb_id_or_episode_id>/
    │   │   ├── clip_000.mp4
    │   │   ├── clip_001.mp4
    │   │   └── ...
    │   └── ...
    └── ...

audio/
    ├── <batch_id>/
    │   ├── <imdb_id_or_episode_id>/
    │   │   ├── clip_000
    │   │   |   ├── clip_000.wav            #去噪后的音频(24khz)
    │   │   |   ├── clip_000_44k.wav        #去噪后的音频(44.1khz)
    │   │   |   ├── clip_000_no_vocals.wav  #去除的噪声部分(bgm, 环境音等)
    │   │   |   └── clip_000_orig.wav       #未去噪的音频(音量归一化)
    │   │   ├── clip_1
    │   │   └── ...
    │   └── ...
    └── ...

json/
    ├── <batch_id>/
    │   ├── <imdb_id_or_episode_id>.json
    │   └── ...
    └── ...
```

数据根目录包含三个子目录：

- video/：视频片段（.mp4）

- audio/：去噪后的音频片段（.mp4）

- json/：结构化标注信息（.json）

- 数据按 batch → 影片/单集 → 片段（clip） 的三级结构进行组织，并在 video / audio / json 三种模态之间保持一致的命名和对齐关系。

### Batch 与影片组织方式

- 每个 batch（如 `movie_41`）表示一次数据发布批次，包含多部电影，或同一剧集的多个单集。

- batch ID 在 `video/`、`audio/`、`json/` 中保持一致

- 在 batch 内，数据按 **影片 ID**（`imdb_id`）或**单集 ID**（`episode_id`） 组织。

### Clip 对齐说明

- 每个影片 / 单集被切分为多个连续片段：`clip_0`, `clip_1`, …

- `video/<ID>/clip_k.mp4` 与 `audio/<ID>/clip_k/clip_k.wav` 表示同一时间区间

- 对应的 `json/<ID>.json` 提供影片级与片段级标注信息

# 模型训练与推理

## Step-0. Environment Settings

1. 创建 conda 环境并安装必要依赖
```bash
conda create -n test python=3.10
conda activate test
pip install -r requirements.txt
pip install flash-attn # optional
```

# Step-1. 数据预处理

- 针对每个 batch 的数据，合并原始 json 标注至单个 json 文件
```bash
# Merge JSON annotations into a single file for the given batch.
# By default, the script assumes the dataset is located at ../release.
python make_json.py --prefix movie_48

# (If needed) customize input/output paths.
python make_json.py --prefix movie_48 \
  --annotation_dir ../release/json \
  --audio_root ../release/audio \
  --output_dir output

# Run the commmand for all batches.
for subdir in ../release/video/*/; do
  subdir=$(basename "$subdir")
  python make_json.py --prefix "$subdir"
done
```
- 合并所有 batch 的数据，并对说话人ID重编号

```bash
# By default, the output data jsonl is stored in output_data/mm-dia.jsonl
python make_data.py
```

# Step-2. 提取token
```bash
cd higgs_finetune
python extract_higgs.py --data_path ../output_data/mm-dia.jsonl --output_dir ../output_data/mm-dia
python make_split.py --input_file ../output_data/mm-dia --meta_file train_eval_test_hard_ids.json --output_dir ../output_data/mm_dia_splits/
```

# Step-3. 训练模型
```bash
accelerate launch --config_file accl_config.yaml train_higgs.py \
    --train_data_path ../output_data/mm_dia_splits/train\
    --val_data_path   ../output_data/mm_dia_splits/eval\
    --batch_size 3 \
    --lr 1e-5 \
    --accumulation_steps 8 \
    --instruction \
    --freeze_text_encoder \
    --semantic_amplification 4.0 \
    --epochs 25\
    --output_dir ../exp/test
```

# Step-4. 模型推理
- 模型训练完成后，可通过命令行或 Gradio Demo 对模型进行推理测试。

```bash
# You can also experience our finetuned model
cd higgs_infer
python app.py --model_path higgs-audio-v2-sft-mm-dia
python infer.py --model_path higgs-audio-v2-sft-mm-dia
```

## Classifier-Free Guidance

在训练阶段，我们通过**随机置空输入条件（对话风格描述）**的方式实现 Classifier-Free Guidance (CFG)，从而提升模型在推理阶段的可控生成能力。对应的推理参数如下：

- `--guidance_scale`：CFG 引导强度系数，用于控制条件约束强度

- `--pad_left`：是否对输入序列进行左侧 padding

- `--cfg_text`：在 CFG 模式下，是否同时对 transcription 条件置空

通过消融实验（ablation study），我们推荐在生成阶段使用如下参数组合：`--pad_left --guidance_scale 2.0 --cfg_text`
