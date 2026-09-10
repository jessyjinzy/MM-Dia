import os
import re
import shutil
import logging
import json
import base64
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

from src.utils.srt_parser import parse_srt, clean_srt_text, seconds_to_srt_time
from src.utils.video_ops import extract_frame_to_temp, cut_video_ffmpeg
from src.utils.llm_client import LLMService

logger = logging.getLogger(__name__)

class DiaClipPipeline:
    def __init__(self, args):
        self.args = args
        self.movie_name = os.path.splitext(os.path.basename(args.video_path))[0]
        
        self.ai_service = LLMService(
            args.vlm_base_url, args.vlm_api_key, args.vlm_model,
            args.llm_base_url, args.llm_api_key, args.llm_model
        )
        
        self.srt_blocks = []
        self.visual_ranges = [] # Format: [(start_index, end_index), ...]
        self.final_groups = []  # Format: [[index, index, ...], [index...]]
        self.decision_source = [] # "VLM" or "LLM" per final group
        self.final_metadata = []

    def run(self):
        logger.info(f"Processing Movie: {self.movie_name}")
        
        # 1. Parse Data
        self.srt_blocks = parse_srt(self.args.srt_path)
        logger.info(f"Loaded {len(self.srt_blocks)} subtitle blocks.")
        
        # 2. VLM Clustering
        self._step_vlm_clustering()
        
        # 3. LLM Refinement
        self._step_llm_refinement()
        
        # 4. Final Cut & Save
        self._step_final_production()
        
        logger.info("Pipeline Completed Successfully.")

    def _step_vlm_clustering(self):
            logger.info("Starting VLM Visual Clustering (Keyframe Pool Mode)...")
            blocks = self.srt_blocks
            if not blocks: 
                return

            ranges = []
            i = 0
            
            pbar = tqdm(total=len(blocks), desc="VLM Clustering")
            
            while i < len(blocks):
                start_i = i
                
                # Context Pool Initialize 
                active_ctx_paths = []
                for k in range(self.args.vlm_max_buffer):
                    if i + k < len(blocks):
                        f = extract_frame_to_temp(self.args.video_path, blocks[i + k]['start_sec'])
                        if f:
                            active_ctx_paths.append(f)
                
                if not active_ctx_paths:
                    i += 1
                    pbar.update(1)
                    continue

                while True:
                    found_extension = False
                    offsets = list(range(self.args.vlm_step_size, 0, -1))
                    
                    for offset in offsets:
                        j = i + offset
                        if j >= len(blocks):
                            continue
                        
                        candidate_frame = extract_frame_to_temp(self.args.video_path, blocks[j]['start_sec'])
                        if not candidate_frame:
                            continue
                            
                        match_idx = self.ai_service.check_visual_continuity(active_ctx_paths, candidate_frame)
            
                        if match_idx > 0:
                            replace_idx = match_idx - 1
                            
                            if 0 <= replace_idx < len(active_ctx_paths):
                                old_frame = active_ctx_paths[replace_idx]
                                active_ctx_paths[replace_idx] = candidate_frame
                                
                                if os.path.exists(old_frame) and old_frame != candidate_frame:
                                    os.remove(old_frame)
                                
                                i = j
                                found_extension = True
                                break 
                            else:
                                if os.path.exists(candidate_frame):
                                    os.remove(candidate_frame)
                        else:
                            if os.path.exists(candidate_frame):
                                os.remove(candidate_frame)
                    
                    if not found_extension:
                        break
                
                ranges.append((start_i, i))
                
                for f in active_ctx_paths:
                    if os.path.exists(f):
                        os.remove(f)

                pbar.update(i - start_i + 1)
                i += 1
                
            pbar.close()
            self.visual_ranges = ranges
            logger.info(f"VLM grouped {len(blocks)} subtitles into {len(ranges)} visual ranges.")

    def _llm_split_transcript(self, transcript: str) -> List[List[int]]:
        """
        Input: Multiline text of "Index: Content"
        Output: JSON list of list of SRT Indices e.g. [[1,2],[3,4]]
        """
        prompt = (
            "The following is a transcript of subtitle lines from a movie clip. "
            "Each line has an index and text. Segment this into complete, logical conversations. "
            "Return ONLY a JSON list of index groups (e.g. [[1,2,3],[4,5]]). "
            "Do not output markdown code blocks or explanations.\n\n"
            f"Transcript:\n{transcript}"
        )
        
        try:
            resp = self.ai_service.llm_client.chat.completions.create(
                model=self.args.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            content = resp.choices[0].message.content.strip()
            content = re.sub(r"```json", "", content)
            content = re.sub(r"```", "", content).strip()
            return json.loads(content)
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return []

    def _step_llm_refinement(self):
        logger.info("Refining ranges with LLM...")
        final_groups = []
        decision_source = []
        
        for (start_pos, end_pos) in tqdm(self.visual_ranges, desc="LLM Refinement"):
            # Get subset of blocks
            subset = self.srt_blocks[start_pos : end_pos + 1]
            if not subset: 
                continue
            
            duration = subset[-1]['end_sec'] - subset[0]['start_sec']
            
            if duration >= self.args.cut_lower_bound:
                transcript = "\n".join([f"{b['index']}: {b['text']}" for b in subset])
                
                split_indices = self._llm_split_transcript(transcript)
                
                valid_splits = []
                subset_ids = set(b['index'] for b in subset)
                
                if isinstance(split_indices, list):
                    for group in split_indices:
                        if isinstance(group, list) and len(group) >= self.args.min_lines:
                            if all(idx in subset_ids for idx in group):
                                valid_splits.append(sorted(group))
                
                if valid_splits:
                    final_groups.extend(valid_splits)
                    decision_source.extend(["LLM"] * len(valid_splits))
                else:
                    final_groups.append([b['index'] for b in subset])
                    decision_source.append("LLM")
            else:
                if len(subset) >= self.args.min_lines:
                    final_groups.append([b['index'] for b in subset])
                    decision_source.append("VLM")
                    
        self.final_groups = final_groups
        self.decision_source = decision_source
        logger.info(f"Refined into {len(final_groups)} final clips.")

    def _step_final_production(self):
        movie_out_dir = os.path.join(self.args.output_dir, self.movie_name)
        clips_out_dir = os.path.join(movie_out_dir, self.movie_name) # Subfolder for mp4s
        
        if os.path.exists(movie_out_dir):
            shutil.rmtree(movie_out_dir)
        os.makedirs(clips_out_dir)
        
        block_map = {b['index']: b for b in self.srt_blocks}
        
        clip_metadata_list = []
        
        pbar = tqdm(total=len(self.final_groups), desc="Cutting Clips")
        
        for i, (group_indices, decision) in enumerate(zip(self.final_groups, self.decision_source)):
            group_blocks = [block_map[idx] for idx in group_indices if idx in block_map]
            
            if len(group_blocks) < self.args.min_lines:
                pbar.update(1)
                continue
            
            group_blocks.sort(key=lambda x: x['start_sec'])
            
            start_sec = group_blocks[0]['start_sec']
            end_sec = group_blocks[-1]['end_sec'] + 0.1 # Slightly extend to include end
            
            clip_name = f"clip_{i:03d}.mp4"
            clip_path = os.path.join(clips_out_dir, clip_name)
            
            try:
                cut_video_ffmpeg(self.args.video_path, clip_path, start_sec, end_sec)
            except Exception as e:
                logger.error(f"Failed to cut {clip_name}: {e}")
                pbar.update(1)
                continue
            
            clip_srt_data = []
            for b in group_blocks:
                rel_start = max(0.0, b['start_sec'] - start_sec)
                rel_end = max(0.0, b['end_sec'] - start_sec)
                
                cleaned_text = clean_srt_text(b['text'])
                
                clip_srt_data.append({
                    "index": b['index'],
                    "abs_start_time": b['start_str'].replace('.', ','), # Format as HH:MM:SS,mmm
                    "abs_end_time": b['end_str'].replace('.', ','),
                    "relative_start": seconds_to_srt_time(rel_start),
                    "relative_end": seconds_to_srt_time(rel_end),
                    "text": cleaned_text
                })
            
            clip_metadata_list.append({
                "clip_id": i,
                "srt": clip_srt_data,
                "source": decision
            })
            pbar.update(1)
            
        pbar.close()
        
        json_path = os.path.join(movie_out_dir, f"{self.movie_name}.json")
        final_json = {
            "movie": self.movie_name,
            "clips": clip_metadata_list
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved Metadata: {json_path}")
        logger.info(f"Saved Clips: {clips_out_dir}")
        
