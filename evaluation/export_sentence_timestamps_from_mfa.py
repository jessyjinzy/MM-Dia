#!/usr/bin/env python3
"""
export_sentence_timestamps_from_mfa.py

Given an MFA TextGrid (word tier) and a transcript containing [SPEAKERx] tags,
produce per-sentence start/end timestamps and a plain list of sentence boundary times.

This script:
  1) Parses your transcript, preserves [SPEAKERx], and splits into sentences.
  2) Parses the TextGrid's word tier into a token stream with times.
  3) Greedily aligns each sentence (normalized tokens) to the token stream in order.
  4) Writes:
      - <out_prefix>_sentences.csv             columns: speaker,text,start,end,matched,note
      - <out_prefix>_sentence_boundaries.txt   each line is the end time (sec) of the sentence

It avoids third-party network installs and uses a robust TextGrid regex parser.
If you already have a sentence tier, you can skip alignment by mapping times directly;
this script assumes word-tier input.

Usage:
  python export_sentence_timestamps_from_mfa.py \
    --textgrid mfa_out/audio.TextGrid \
    --transcript dialog_with_speakers.txt \
    --out_prefix mfa_alignment
"""
from __future__ import annotations
import os
import argparse
import difflib
import re
import shutil
from glob import glob
from tqdm import tqdm
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import json
import pandas as pd
import subprocess
# -------------------------
# Normalization & sentence splitting
# -------------------------
PUNCT_FOR_SENT = r"[。．.!?\?]+"  # English/Chinese common sentence-final punctuation
SPEAKER_TAG_RE = re.compile(r"\[(SPEAKER\d+)\]")

def normalize_token(s: str) -> str:
    """Lowercase, strip accents, keep letters/digits/'/-, collapse spaces."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s'’-]", " ", s)
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_sentences_keep_punct(text: str) -> List[str]:
    parts = re.split(f"({PUNCT_FOR_SENT})", text)
    sents, buf = [], ""
    for part in parts:
        if not part:
            continue
        buf += part
        if re.fullmatch(PUNCT_FOR_SENT, part):
            if buf.strip():
                sents.append(buf.strip())
            buf = ""
    if buf.strip():
        sents.append(buf.strip())
    # drop empties and ensure normalization not empty
    return [s for s in sents if s and normalize_token(s)]

def run_mfa(corpus_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cmd = [
    "mfa", "align",
    corpus_dir,
    "english_us_arpa",
    "english_us_arpa",
    output_dir,
    "--clean"
]
    subprocess.run(cmd, check=True)

@dataclass
class SentenceItem:
    speaker: str
    text: str
    norm_tokens: List[str]

def parse_transcript_with_speakers(raw: str) -> List[SentenceItem]:
    """Parse [SPEAKERx] blocks and split into sentences, preserving speaker per sentence."""
    chunks: List[Tuple[str, str]] = []
    last_speaker: Optional[str] = None
    pos = 0
    for m in SPEAKER_TAG_RE.finditer(raw):
        if m.start() > pos and last_speaker is not None:
            chunks.append((last_speaker, raw[pos:m.start()]))
        last_speaker = m.group(1)
        pos = m.end()
    if pos < len(raw) and last_speaker is not None:
        chunks.append((last_speaker, raw[pos:]))

    items: List[SentenceItem] = []
    for spk, chunk in chunks:
        for sent in split_sentences_keep_punct(chunk):
            toks = [t for t in normalize_token(sent).split() if t]
            if toks:
                items.append(SentenceItem(spk, sent.strip(), toks))
    return items

# -------------------------
# TextGrid parsing (pragmatic regex reader)
# -------------------------
@dataclass
class Token:
    text: str
    start: float
    end: float

def parse_textgrid_words_pragmatic(text: str, tier_names=("words","word","Words","Tokens","tokens")) -> List[Tuple[float,float,str]]:
    """
    Robustly locate a word-like tier and extract (xmin,xmax,text) from intervals.
    Compatible with MFA 3.x Praat TextGrid output.
    """
    # find item [i] blocks and choose one whose name matches candidates
    items_iter = list(re.finditer(r'item \[\d+\]:', text))
    spans = []
    for idx, m in enumerate(items_iter):
        start = m.start()
        end = items_iter[idx+1].start() if idx+1 < len(items_iter) else len(text)
        block = text[start:end]
        name_m = re.search(r'name\s*=\s*"([^"]+)"', block)
        if not name_m:
            continue
        name = name_m.group(1)
        if name in tier_names:
            spans.append(block)
    if not spans:
        # fallback to first interval tier (rare)
        spans = [text]

    block = spans[0]
    # Get intervals section (supports both 'size = N' and parenthesized forms)
    intervals_section_m = re.search(r'intervals\s*:\s*(?:size\s*=\s*\d+)?(.*)$', block, re.DOTALL)
    if not intervals_section_m:
        raise RuntimeError("Cannot find intervals section inside the chosen tier. "
                           "Check that the TextGrid has a word-level tier.")
    intervals_sec = intervals_section_m.group(1)

    rec_pat = re.compile(
        r"intervals \[\d+\]:\s*xmin\s*=\s*([0-9.]+)\s*xmax\s*=\s*([0-9.]+)\s*text\s*=\s*\"(.*?)\"",
        re.DOTALL,
    )
    out = []
    for xmin, xmax, textval in rec_pat.findall(intervals_sec):
        out.append((float(xmin), float(xmax), textval))
    return out

def build_token_stream_from_textgrid(tg_path: Path) -> List[Token]:
    tg_text = Path(tg_path).read_text(encoding="utf-8", errors="ignore")
    words_raw = parse_textgrid_words_pragmatic(tg_text)
    stream: List[Token] = []
    for xmin, xmax, w in words_raw:
        norm = normalize_token(w)
        if not norm or norm in {"sp", "<eps>", "sil", '""'}:
            continue
        # if normalization split a word, duplicate time span over subtokens
        for sub in norm.split():
            stream.append(Token(sub, xmin, xmax))
    if not stream:
        raise RuntimeError("No valid tokens parsed from TextGrid. "
                           "Confirm that the 'words' tier has non-empty labels.")
    return stream

# -------------------------
# Alignment (tolerant sequential search)
# -------------------------
def approx_eq(a: str, b: str, ratio: float = 0.84) -> bool:
    if a == b:
        return True
    return difflib.SequenceMatcher(a=a, b=b).ratio() >= ratio

def find_token_forward(hay: List[Token], needle: str, start_idx: int, max_lookahead: int = 500) -> Optional[int]:
    end = min(len(hay), start_idx + max_lookahead)
    for i in range(start_idx, end):
        if approx_eq(hay[i].text, needle):
            return i
    return None

def align_sentence_tokens(sent_toks: List[str], stream: List[Token], start_idx: int):
    """Greedy tolerant alignment: locate indices [i..j] in stream covering sent_toks in order.
       Returns (i, j, note). If fail, (None, None, reason).
    """
    if not sent_toks:
        return None, None, "empty"
    i = start_idx
    first = None
    last = None

    # find first
    idx = find_token_forward(stream, sent_toks[0], i)
    if idx is None:
        return None, None, f"first-not-found:{sent_toks[0]}"
    first = idx
    i = idx + 1

    # walk remaining tokens allowing skips in stream
    for tok in sent_toks[1:]:
        idx = find_token_forward(stream, tok, i)
        if idx is None:
            # partial coverage allowed: close at previous
            last = i - 1 if i > first else first
            return first, last, "partial"
        last = idx
        i = idx + 1

    if last is None:
        last = first
    return first, last, "ok"

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Export sentence timestamps from MFA TextGrid and a speaker-tagged transcript.")
    ap.add_argument("--wav_dir", default='', help="Path to MFA TextGrid (with a word tier).")
    ap.add_argument("--transcript", default='./evaluation/examples/gt_trans.tsv', help="Transcript text containing [SPEAKERx] tags.")
    ap.add_argument("--lab_dir", default="./evaluation/examples/gt_trans_lab", help="Groundtruth .lab files directory for MFA Alignment.")
    ap.add_argument("--round_digits", type=int, default=3, help="Decimal places for time rounding.")
    args = ap.parse_args()

    tr_path = args.transcript
    lab_dir = args.lab_dir
    with open(tr_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    print(len(lines))
    transcripts = {}
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
        file_id, transcription = parts
        transcripts[file_id] = transcription
    wav_dir = args.wav_dir
    wav_paths = os.listdir(wav_dir)
    # wav_paths = [f for f in wav_paths if 'tv_movie' in f]
    for wav_path in tqdm(wav_paths):
        # if 'test' not in wav_path:
        #     continue
        wav_files = glob(f'{wav_dir}/{wav_path}/output_*.wav')
        print(len(wav_files))
        for wav_file in wav_files:
            shutil.copy(wav_file, lab_dir)
        tg_path_dir = f'{wav_dir}/{wav_path}/Textgrid'
        # if os.path.exists(tg_path_dir):
        #     continue
        run_mfa(lab_dir, tg_path_dir)

        for tg_path in os.listdir(tg_path_dir):
            if not tg_path.endswith('.TextGrid'):
                continue
            # 1) transcript -> sentences
            raw = transcripts[tg_path[7:-9]]  #'output_XXXX.wav' -> 'XXXX'
            sentences = parse_transcript_with_speakers(raw)
            if not sentences:
                sys.exit("No sentences parsed from transcript. Check [SPEAKERx] tags and sentence punctuation.")

            # 2) TextGrid -> token stream
            stream_tokens = build_token_stream_from_textgrid(f'{tg_path_dir}/{tg_path}')

            # 3) Align
            results = []
            search_start = 0
            for s in sentences:
                st_idx, ed_idx, note = align_sentence_tokens(s.norm_tokens, stream_tokens, search_start)
                if st_idx is None:
                    prev_end = results[-1]["end"] if results else 0.0
                    results.append({
                        "speaker": s.speaker,
                        "text": s.text,
                        "start": round(prev_end, args.round_digits),
                        "end": round(prev_end, args.round_digits),
                        "matched": False,
                        "note": note
                    })
                else:
                    st_t = stream_tokens[st_idx].start
                    ed_t = stream_tokens[ed_idx].end
                    results.append({
                        "speaker": s.speaker,
                        "text": s.text,
                        "start": round(st_t, args.round_digits),
                        "end": round(ed_t, args.round_digits),
                        "matched": True if note in ("ok","partial") else False,
                        "note": note
                    })
                    search_start = ed_idx + 1

            # 4) Export
            if not os.path.exists(f"{wav_dir}/{wav_path}/Aligned"):
                os.makedirs(f"{wav_dir}/{wav_path}/Aligned")
            json.dump(results, open(f"{wav_dir}/{wav_path}/Aligned/{os.path.basename(tg_path)}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            
        # 5) remove temp wav files from lab dir
        temp_wav_files = glob(f'{lab_dir}/*.wav')
        for temp_wav_file in temp_wav_files:
            os.remove(temp_wav_file)

if __name__ == "__main__":
    main()
