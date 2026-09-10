python extraction_main.py \
    --video_path ./assets/tt0327137.mp4 \
    --srt_path ./assets/tt0327137.srt \
    --output_dir ./output \
    --vlm_base_url http://localhost:8000/v1 \
    --vlm_api_key EMPTY \
    --vlm_model Path_to_Qwen2.5-VL-72B-Instruct \
    --llm_base_url "${LLM_BASE_URL:-https://api.openai.com/v1}" \
    --llm_api_key "${LLM_API_KEY}" \
    --llm_model gpt-5 \
    --cut_lower_bound 90.0 \
    --min_lines 3 \
    --vlm_step_size 10 \
    --vlm_max_buffer 3

python3 annotation_main.py \
    --input_json "./output/tt0327137/tt0327137.json" \
    --video_root "./output/tt0327137/tt0327137" \
    --output_json "./output/tt0327137/tt0327137-anno.json" \
    --api_key "${API_KEY}" \
    --base_url "${BASE_URL:-https://api.openai.com/v1}" \
    --server_ip "${SERVER_IP:-localhost}" \
    --upload_dir "${UPLOAD_DIR:-./uploads/}"
