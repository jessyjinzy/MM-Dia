import os
import subprocess
import tempfile
from typing import Optional
from src.utils.srt_parser import seconds_to_srt_time

def extract_frame_to_temp(video_path: str, timestamp_sec: float) -> Optional[str]:
    try:
        tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        output_path = tf.name
        tf.close()
        
        ts_str = seconds_to_srt_time(timestamp_sec).replace(',', '.')
        
        cmd = [
            "ffmpeg", "-ss", ts_str, "-i", video_path,
            "-frames:v", "1", "-q:v", "2", "-y", output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        return None
    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
        return None

def cut_video_ffmpeg(src: str, dst: str, start_sec: float, end_sec: float):
    duration = end_sec - start_sec
    if duration <= 0: 
        return

    cmd = [
        'ffmpeg', '-y',
        '-ss', f"{start_sec:.3f}",
        '-i', src,
        '-t', f"{duration:.3f}",
        '-map', '0:v:0', '-map', '0:a:0',
        '-c:v', 'libx264', '-c:a', 'aac',
        '-ac', '2', '-b:a', '192k',
        '-movflags', '+faststart',
        dst
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)