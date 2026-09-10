# main.py
import argparse
import os
from src.extraction_pipeline import DiaClipPipeline

def main():
    parser = argparse.ArgumentParser(...)

    parser.add_argument("--video_path", required=True, help="Path to source .mp4 file")
    parser.add_argument("--srt_path", required=True, help="Path to source .srt file")
    parser.add_argument("--output_dir", required=True, help="Root directory to save results")
    
    # API Configs
    parser.add_argument("--vlm_base_url", default="http://localhost:8000/v1", help="VLM API URL")
    parser.add_argument("--vlm_api_key", default="EMPTY", help="VLM API Key")
    parser.add_argument("--vlm_model", default="../pretrained_models/Qwen/Qwen2.5-VL-72B-Instruct", help="VLM Model Name")
    
    parser.add_argument("--llm_base_url", default=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"), help="LLM API URL")
    parser.add_argument("--llm_api_key", required=True, help="LLM API Key")
    parser.add_argument("--llm_model", default="gpt-5", help="LLM Model Name")

    parser.add_argument("--cut_lower_bound", type=float, default=90.0, help="Duration threshold to trigger LLM splitting (in seconds)")
    parser.add_argument("--min_lines", type=int, default=3, help="Minimum subtitle lines required for a valid clip")
    parser.add_argument("--vlm_step_size", type=int, default=10, help="Shift step for VLM frame sampling")
    parser.add_argument("--vlm_max_buffer", type=int, default=3, help="Context frame buffer size for VLM")


    args = parser.parse_args()

    # Checks
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")
    if not os.path.exists(args.srt_path):
        raise FileNotFoundError(f"SRT not found: {args.srt_path}")

    pipeline = DiaClipPipeline(args)
    pipeline.run()

if __name__ == "__main__":
    main()