import json
import os
import re
import logging
from tqdm import tqdm
from src.utils.annotation_prompts import PROMPT_SPEAKER_DIARIZATION, PROMPT_ALIGNMENT, PROMPT_STYLE
from src.utils.text_cleaner import clean_json_response

logger = logging.getLogger(__name__)

class MovieAnnotationPipeline:
    def __init__(self, args, ai_service, uploader):
        self.args = args
        self.ai_service = ai_service
        self.uploader = uploader
        self.movie_name = os.path.basename(args.input_json).replace('_clips_srt.json', '').replace('.json', '')

    def run(self):
        with open(self.args.input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        clips = data.get('clips', [])
        logger.info(f"Loaded {len(clips)} clips for annotation.")
        
        valid_clips = []
        
        for clip in tqdm(clips, desc="Annotating Clips"):
            if 'style_annotation' in clip and 'speaker' in clip:
                valid_clips.append(clip)
                continue

            # --- Step 1: Speaker Diarization (VLM) ---
            transcription = [line['text'].strip().replace('\n',' ') for line in clip['srt']]
            video_path = os.path.join(self.args.video_root, f"clip_{str(clip['clip_id']).zfill(3)}.mp4")
            
            if not os.path.exists(video_path):
                logger.warning(f"Video not found: {video_path}")
                continue

            video_url = self.uploader.upload(video_path, prefix=self.movie_name)
            if not video_url: 
                continue

            print('video_url', video_url)

            try:
                raw_speaker_anno = self._diarize_speakers(video_url, transcription)
                
                if "No conversation" in raw_speaker_anno:
                    self.uploader.delete(os.path.basename(video_path), prefix=self.movie_name)
                    continue 
                
                # --- Step 2: Alignment & Cleaning (LLM) ---
                aligned_anno = self._align_annotations(raw_speaker_anno, transcription)
                
                if aligned_anno == "No Annotation" or len(aligned_anno) != len(clip['srt']):
                     self.uploader.delete(os.path.basename(video_path), prefix=self.movie_name)
                     continue 

                for i, line in enumerate(clip['srt']):
                    line['character_annotation'] = aligned_anno[i]
                
                speakers = self._extract_speakers(aligned_anno)
                clip['speaker'] = list(speakers)
                
                if not speakers:
                    self.uploader.delete(os.path.basename(video_path), prefix=self.movie_name)
                    continue

                # --- Step 3: Style Annotation (VLM) ---
                style_anno = self._analyze_style(video_url, list(speakers))
                clip['style_annotation'] = style_anno

                valid_clips.append(clip)

            except Exception as e:
                logger.error(f"Error processing clip {clip['clip_id']}: {e}")
            finally:
                self.uploader.delete(os.path.basename(video_path), prefix=self.movie_name)

        # --- Step 4: Final Save ---
        output_data = {
            "movie": data.get('movie', self.movie_name),
            "clips": valid_clips
        }
        with open(self.args.output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Annotation complete. Saved to {self.args.output_json}")

    def _diarize_speakers(self, video_url, transcription):
        prompt = PROMPT_SPEAKER_DIARIZATION.format(
            transcription='\n'.join(transcription),
            count=len(transcription),
            movie=self.movie_name
        )
        print('prompt', prompt)
        return self.ai_service.call_vlm(video_url, prompt)

    def _align_annotations(self, raw_anno_str, transcription):
        try:
            json_str = clean_json_response(raw_anno_str)
            raw_anno_list = json.loads(json_str)
        except:
            raw_anno_list = raw_anno_str.split('\n')

        prompt = PROMPT_ALIGNMENT.format(
            original_srt_lines=transcription,
            annotations=raw_anno_list,
            length=len(transcription)
        )
        resp = self.ai_service.call_llm(prompt)
        try:
            cleaned = json.loads(clean_json_response(resp))
            return cleaned
        except:
            return "No Annotation"

    def _extract_speakers(self, anno_list):
        speakers = set()
        for line in anno_list:
            # Simple heuristic: "Speaker: Dialogue"
            parts = line.split(':', 1)
            if len(parts) > 1:
                spk = parts[0].strip()
                if spk.lower() != "no annotation":
                    speakers.add(spk)
        return speakers

    def _analyze_style(self, video_url, speakers):
        speaker_str = ", ".join(speakers)
        prompt = PROMPT_STYLE.format(speaker=speaker_str, movie=self.movie_name)
        resp = self.ai_service.call_vlm(video_url, prompt)
        try:
            return json.loads(clean_json_response(resp))
        except:
            return {"Tags": [], "Summary": resp}