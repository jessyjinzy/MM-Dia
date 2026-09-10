# Example Assets for MM-DIA Curation Pipeline

This directory should contain example input files for testing the data curation pipeline.

## Required Files

To run the pipeline example, you need:

1. **Video file**: `tt0327137.mp4` (or any test movie/episode)
   - Source: Legally obtained movie or TV episode
   - Format: Any ffmpeg-compatible video format (.mp4, .mkv, .avi, etc.)

2. **Subtitle file**: `tt0327137.srt`
   - Source: Subtitle file matching the video
   - Format: Standard SRT (SubRip) format
   - Note: Multiple SRT files from different sources can be used for better accuracy

## Example Usage

```bash
cd mmdia

# Full pipeline: extraction + annotation
bash run.sh

# Or run steps individually:

# Step 1-3: Dialogue extraction
python extraction_main.py \
    --video_path assets/tt0327137.mp4 \
    --srt_path   assets/tt0327137.srt \
    --output_dir output/tt0327137

# Step 4: Dialogue annotation (requires Gemini API)
python annotation_main.py \
    --input_json  output/tt0327137/tt0327137.json \
    --video_root  output/tt0327137/tt0327137 \
    --output_json output/tt0327137/tt0327137_annotated.json
```

## Notes

- Due to copyright restrictions, we cannot include actual movie files in this repository
- Users should provide their own legally-obtained content for testing
- The pipeline works with any movie/TV episode that has subtitles
- For best results, use high-quality video sources (Blu-ray, high-bitrate streaming)

## File Naming Convention

- Video files should be named by their IMDB ID (e.g., `tt0327137.mp4`)
- Subtitle files should have matching names (e.g., `tt0327137.srt`)
- Find IMDB IDs at: https://www.imdb.com/

## Minimal Test Setup

If you want to quickly test the pipeline, use a short clip (~5-10 minutes) with clear dialogue:
1. Extract a short segment from any movie
2. Obtain or create a matching SRT file
3. Place both in this directory
4. Run the pipeline as shown above
