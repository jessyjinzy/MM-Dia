#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors:  Han Zhu
#                                                   Wei Kang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Calculate Speaker Aware Similarity.
SV model wavlm_large_finetune.pth is downloaded from
    https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification
SSL model wavlm_large.pt is downloaded from
    https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_large.pt
"""
import argparse
import json
import math
import os
from itertools import combinations
from typing import Dict, List, Tuple
from glob import glob

import librosa
import logging
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# part of the code is borrowed from https://github.com/lawlict/ECAPA-TDNN

""" Res2Conv1d + BatchNorm1d + ReLU
"""


class Res2Conv1dReluBn(nn.Module):
    """
    in_channels == out_channels == channels
    """

    def __init__(
        self,
        channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        scale=4,
    ):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    bias=bias,
                )
            )
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)

        return out


""" Conv1d + BatchNorm1d + ReLU
"""


class Conv1dReluBn(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


""" The SE connection of 1D case.
"""


class SE_Connect(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)

        return out


""" SE-Res2Block of the ECAPA-TDNN architecture.
"""


# def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
#     return nn.Sequential(
#         Conv1dReluBn(channels, 512, kernel_size=1, stride=1, padding=0),
#         Res2Conv1dReluBn(512, kernel_size, stride, padding, dilation, scale=scale),
#         Conv1dReluBn(512, channels, kernel_size=1, stride=1, padding=0),
#         SE_Connect(channels)
#     )


class SE_Res2Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        scale,
        se_bottleneck_dim,
    ):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(
            out_channels, kernel_size, stride, padding, dilation, scale=scale
        )
        self.Conv1dReluBn2 = Conv1dReluBn(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)

        return x + residual


""" Attentive weighted mean and standard deviation pooling.
"""


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear,
        #  then we don't need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(
                in_dim * 3, attention_channels, kernel_size=1
            )  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                in_dim, attention_channels, kernel_size=1
            )  # equals W and b in the paper
        self.linear2 = nn.Conv1d(
            attention_channels, in_dim, kernel_size=1
        )  # equals V and k in the paper

    def forward(self, x):

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-10
            ).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x_in))
        # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN_WAVLLM(nn.Module):
    def __init__(
        self,
        feat_dim=80,
        channels=512,
        emb_dim=192,
        global_context_att=False,
        sr=16000,
        ssl_model_path=None,
    ):
        super().__init__()
        self.sr = sr

        if ssl_model_path is None:
            self.feature_extract = torch.hub.load("s3prl/s3prl", "wavlm_large")
        else:
            self.feature_extract = torch.hub.load(
                os.path.dirname(ssl_model_path),
                "wavlm_local",
                source="local",
                ckpt=ssl_model_path,
            )

        if len(self.feature_extract.model.encoder.layers) == 24 and hasattr(
            self.feature_extract.model.encoder.layers[23].self_attn,
            "fp32_attention",
        ):
            self.feature_extract.model.encoder.layers[23].self_attn.fp32_attention = (
                False
            )
        if len(self.feature_extract.model.encoder.layers) == 24 and hasattr(
            self.feature_extract.model.encoder.layers[11].self_attn,
            "fp32_attention",
        ):
            self.feature_extract.model.encoder.layers[11].self_attn.fp32_attention = (
                False
            )

        self.feat_num = self.get_feat_num()
        self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))

        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        # self.channels = [channels] * 4 + [channels * 3]
        self.channels = [channels] * 4 + [1536]

        self.layer1 = Conv1dReluBn(feat_dim, self.channels[0], kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(
            self.channels[0],
            self.channels[1],
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            scale=8,
            se_bottleneck_dim=128,
        )
        self.layer3 = SE_Res2Block(
            self.channels[1],
            self.channels[2],
            kernel_size=3,
            stride=1,
            padding=3,
            dilation=3,
            scale=8,
            se_bottleneck_dim=128,
        )
        self.layer4 = SE_Res2Block(
            self.channels[2],
            self.channels[3],
            kernel_size=3,
            stride=1,
            padding=4,
            dilation=4,
            scale=8,
            se_bottleneck_dim=128,
        )

        # self.conv = nn.Conv1d(self.channels[-1], self.channels[-1], kernel_size=1)
        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, self.channels[-1], kernel_size=1)
        self.pooling = AttentiveStatsPool(
            self.channels[-1],
            attention_channels=128,
            global_context_att=global_context_att,
        )
        self.bn = nn.BatchNorm1d(self.channels[-1] * 2)
        self.linear = nn.Linear(self.channels[-1] * 2, emb_dim)

    def get_feat_num(self):
        self.feature_extract.eval()
        wav = [torch.randn(self.sr).to(next(self.feature_extract.parameters()).device)]
        with torch.no_grad():
            features = self.feature_extract(wav)
        select_feature = features["hidden_states"]
        if isinstance(select_feature, (list, tuple)):
            return len(select_feature)
        else:
            return 1

    def get_feat(self, x):
        with torch.no_grad():
            x = self.feature_extract([sample for sample in x])

        x = x["hidden_states"]
        if isinstance(x, (list, tuple)):
            x = torch.stack(x, dim=0)
        else:
            x = x.unsqueeze(0)
        norm_weights = (
            F.softmax(self.feature_weight, dim=-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        x = (norm_weights * x).sum(dim=0)
        x = torch.transpose(x, 1, 2) + 1e-6

        x = self.instance_norm(x)
        return x

    def forward(self, x):
        x = self.get_feat(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn(self.pooling(out))
        out = self.linear(out)

        return out


class SpeakerSimilarity:
    def __init__(
        self,
        sv_model_path="model/UniSpeech/wavlm_large_finetune.pth",
        ssl_model_path="model/s3prl/wavlm_large.pt",
        sample_rate=16000,
        device: torch.device = None,
    ):
        self.sample_rate = sample_rate
        self.channels = 1
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        logging.info("[Speaker Similarity] Using device: %s", self.device)

        self.model = ECAPA_TDNN_WAVLLM(
            feat_dim=1024,
            channels=512,
            emb_dim=256,
            sr=self.sample_rate,
            ssl_model_path=ssl_model_path,
        )
        state_dict = torch.load(sv_model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict["model"], strict=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_segments(self, segments: List[torch.Tensor]) -> List[torch.Tensor]:
        embds = []
        for seg in segments:
            seg = seg.to(self.device)
            emb = self.model([seg])  # Keep consistent with original forward (accepts list[tensor])
            # Normalize to 1D: [D]
            if emb.dim() > 1:
                emb = emb.squeeze()
            embds.append(emb.detach().cpu())
        return embds


def load_and_resample(wav_path: str, target_sr: int = 16000, dtype="float32") -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(wav_path, dtype=dtype)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return wav, sr


def slice_by_timestamps(
    wav: np.ndarray, sr: int, items: List[Dict], min_len_sec: float = 0.05
) -> Dict[str, List[torch.Tensor]]:
    spk2segs: Dict[str, List[torch.Tensor]] = {}
    n = len(wav)
    for it in items:
        spk = it["speaker"]
        t0 = max(0.0, float(it["start"]))
        t1 = max(t0, float(it["end"]))
        if t1 - t0 < min_len_sec:
            continue  # Skip very short segments to avoid numerical instability

        s0 = int(round(t0 * sr))
        s1 = int(round(t1 * sr))
        s0 = max(0, min(s0, n))
        s1 = max(0, min(s1, n))
        if s1 <= s0:
            continue

        seg_np = wav[s0:s1].astype(np.float32)
        seg_torch = torch.from_numpy(seg_np)
        spk2segs.setdefault(spk, []).append(seg_torch)
    return spk2segs


def pairwise_cosine_mean(emb_list: List[torch.Tensor]) -> float:
    if len(emb_list) < 2:
        return float("nan")

    X = torch.stack(emb_list, dim=0).float()         # [N, D]
    X = torch.nn.functional.normalize(X, p=2, dim=1) # L2
    sims = []
    for i, j in combinations(range(X.size(0)), 2):
        sims.append(F.cosine_similarity(X[i], X[j], dim=0).item())
    return float(np.mean(sims)) if sims else float("nan")


def speaker_aware_similarity(
    model: SpeakerSimilarity,
    wav_path: str,
    json_path: str,
    dtype: str = "float32",
    min_len_sec: float = 0.05,
    report_weight: str = "unweighted",
) -> Dict:

    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    wav, sr = load_and_resample(wav_path, target_sr=model.sample_rate, dtype=dtype)
    spk2segs = slice_by_timestamps(wav, sr, items, min_len_sec=min_len_sec)

    spk2embs: Dict[str, List[torch.Tensor]] = {}
    try:
        for spk, segs in spk2segs.items():
            emb_list = model.embed_segments(segs)
            spk2embs[spk] = emb_list

        spk2sim: Dict[str, float] = {}
        spk2count_pairs: Dict[str, int] = {}
        for spk, emb_list in spk2embs.items():
            if len(emb_list) < 2:
                spk2sim[spk] = float("nan")
                spk2count_pairs[spk] = 0
                continue

            n = len(emb_list)
            spk2count_pairs[spk] = n * (n - 1) // 2
            spk2sim[spk] = pairwise_cosine_mean(emb_list)

        valid = [(spk, spk2sim[spk], spk2count_pairs[spk]) for spk in spk2sim if not math.isnan(spk2sim[spk])]
        if not valid:
            overall = float("nan")
        else:
            if report_weight == "unweighted":
                overall = float(np.mean([x[1] for x in valid]))
            elif report_weight == "pairwise_weighted":
                sims = np.array([x[1] for x in valid], dtype=np.float64)
                weights = np.array([x[2] for x in valid], dtype=np.float64)
                overall = float(np.average(sims, weights=weights))
            else:
                raise ValueError("report_weight must be 'unweighted' or 'pairwise_weighted'")

        return {
            "per_speaker_similarity": spk2sim,           # {speaker: average cosine similarity}
            "per_speaker_pair_counts": spk2count_pairs,  # {speaker: number of pairs in average}
            "overall_similarity": overall,               # Overall speaker-aware similarity
            "speakers_found": list(spk2sim.keys()),
        }
    except Exception as e:
        return None

# ------------------ CLI ------------------ #
def main():
    parser = argparse.ArgumentParser(description="Compute speaker-aware similarity from a long conversation WAV and JSON timestamps.")
    parser.add_argument("--wav_dir", type=str, default="Path to the dir of your GT wavs", help="Path to long conversation wav.")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--min_len_sec", type=float, default=0.1, help="Drop segments shorter than this (seconds).")
    parser.add_argument("--report_weight", type=str, choices=["unweighted", "pairwise_weighted"], default="unweighted")
    args = parser.parse_args()
    
    sv_model_path = "../models/wavlm_large_finetune.pth"
    ssl_model_path = "../hub/s3prl_s3prl_main/wavlm_large.pt"

    spk_sim = SpeakerSimilarity(
        sv_model_path=sv_model_path,
        ssl_model_path=ssl_model_path,
        sample_rate=24000,
    )
    for json_dir in tqdm(os.listdir(args.wav_dir)):

        json_paths = glob(os.path.join(args.wav_dir, json_dir, "Aligned", "*.json"))
        print(os.path.join(args.wav_dir, json_dir, "Aligned"))
        results = {}

        for json_path in tqdm(json_paths):
            wav_path = os.path.join(args.wav_dir, json_dir, json_path.split('/')[-1].replace(".TextGrid.json", ".wav"))
            if not os.path.exists(wav_path):
                continue
            result = speaker_aware_similarity(
                spk_sim,
                wav_path=wav_path,
                json_path=json_path,
                dtype=args.dtype,
                min_len_sec=args.min_len_sec,
                report_weight=args.report_weight,
            )
            if result is not None:
                results[os.path.basename(json_path).replace(".json", "")] = result
        
        overall_similarity = []
        for item in results:
            if results[item]['per_speaker_pair_counts'].get('SPEAKER0', 0)>5:
                overall_similarity.append(results[item]['per_speaker_similarity']['SPEAKER0'])
            if results[item]['per_speaker_pair_counts'].get('SPEAKER1', 0)>5:
                overall_similarity.append(results[item]['per_speaker_similarity']['SPEAKER1'])

        sa_sim = round(sum(overall_similarity)/len(overall_similarity), 3)
        new_json = {"sa_sim": sa_sim, "details": results}
        print('SA-SIM: ', sa_sim, json_dir)
        json.dump(new_json, open(f"{args.wav_dir}/{json_dir}/_sa_sim.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
