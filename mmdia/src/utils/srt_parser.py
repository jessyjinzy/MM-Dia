import re
from datetime import datetime
from typing import List, Dict, Any

def parse_time_to_seconds(t_str: str) -> float:
    t_str = t_str.strip().replace(',', '.')
    try:
        dt = datetime.strptime(t_str, "%H:%M:%S.%f")
        return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    except ValueError:
        try:
            dt = datetime.strptime(t_str, "%H:%M:%S")
            return float(dt.hour * 3600 + dt.minute * 60 + dt.second)
        except ValueError:
            return 0.0

def seconds_to_srt_time(seconds: float) -> str:
    if seconds < 0: seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = seconds % 60
    ms = int((sec - int(sec)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(sec):02d},{ms:03d}"

def clean_srt_text(text: str) -> str:
    text = re.sub(r"\{.*?\}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def parse_srt(file_path: str) -> List[Dict[str, Any]]:
    """
    Parses SRT file into a list of dicts.
    Returns: [{'index': int, 'start_sec': float, 'end_sec': float, 'text': str, ...}, ...]
    """
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'gbk']
    content = None
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        raise ValueError(f"Could not read SRT file: {file_path}")

    content = content.replace('\r\n', '\n').replace('\r', '\n')
    raw_blocks = re.split(r'\n\s*\n', content.strip())
    
    blocks = []
    for block in raw_blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        
        # Parse Index
        if not lines[0].isdigit():
            continue
        idx = int(lines[0])
        
        # Parse Time
        if '-->' not in lines[1]:
            continue
        start_str, end_str = lines[1].split('-->')
        
        # Parse Text
        text = "\n".join(lines[2:])
        
        blocks.append({
            'index': idx,
            'start_str': start_str.strip(),
            'end_str': end_str.strip(),
            'start_sec': parse_time_to_seconds(start_str.strip()),
            'end_sec': parse_time_to_seconds(end_str.strip()),
            'text': text
        })
    return blocks

