#!/usr/bin/env python3
import argparse
import os
import logging
from src.annotation_pipeline import MovieAnnotationPipeline
from src.utils.llm_client import LLMService
from src.utils.file_utils import FileUploader

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Movie Clip Annotation Pipeline")
    
    # Paths
    parser.add_argument("--input_json", required=True, help="Path to clips_srt.json from previous step")
    parser.add_argument("--video_root", required=True, help="Folder containing mp4 clips")
    parser.add_argument("--output_json", required=True, help="Path to save annotated JSON")
    
    # API Configs
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--base_url", default=os.getenv("BASE_URL", "https://api.openai.com/v1"))

    # Upload Server Config
    parser.add_argument("--server_ip", default=os.getenv("SERVER_IP", "localhost"))
    parser.add_argument("--upload_dir", default=os.getenv("UPLOAD_DIR", "./uploads/"))

    args = parser.parse_args()

    # Initialize Services
    # Previously used models: gemini-2.5-flash (for both VLM and LLM)
    # Note: Assuming AIService has methods call_vlm and call_llm
    ai_service = LLMService(
        vlm_base_url=args.base_url, vlm_key=args.api_key, vlm_model="gpt-5",
        llm_base_url=args.base_url, llm_key=args.api_key, llm_model="gpt-5"
    )
    
    uploader = FileUploader(args.server_ip, args.upload_dir)

    # Run Pipeline
    pipeline = MovieAnnotationPipeline(args, ai_service, uploader)
    pipeline.run()

if __name__ == "__main__":
    main()