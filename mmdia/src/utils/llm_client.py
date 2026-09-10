import base64
import json
import re
import logging
from typing import List
from openai import OpenAI
import httpx
from src.utils.text_cleaner import clean_json_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, vlm_base_url, vlm_key, vlm_model, llm_base_url, llm_key, llm_model):
        self.vlm_client = OpenAI(
            api_key=vlm_key, 
            base_url=vlm_base_url,
            http_client=httpx.Client(
                base_url=vlm_base_url,
                follow_redirects=True),
            )
        self.llm_client = OpenAI(
            api_key=llm_key, 
            base_url=llm_base_url, 
            http_client=httpx.Client(
                base_url=vlm_base_url,
                follow_redirects=True)
        )
        self.vlm_model = vlm_model
        self.llm_model = llm_model

    def check_visual_continuity(self, context_frames: List[str], candidate_frame: str) -> bool:
        
        num_range = ", ".join(str(i+1) for i in range(len(context_frames)))
        
        """Asks VLM if the candidate frame belongs to the same scene as context frames."""
        prompt_text = (
            "You are a multimodal assistant. "
            f"You will be shown {len(context_frames)} context frames followed by one candidate frame. "
            "Based only on visual cues—such as characters, setting, lighting, camera angle, and composition—"
            "decide whether the candidate frame plausibly matches any of the context frames as part of the same scene or conversation. "
            f"If so, decide which context frame it matches best, choose a number from {num_range}. "
            "Returns only a digit number:"
            "0: Not continuous."
            "N: Continuous, and matches the N-th frame in context (from {num_range})."
        )
        
        content = [{"type": "text", "text": prompt_text}]
        
        # Add images
        for path in context_frames + [candidate_frame]:
            with open(path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
            })
            
        try:
            resp = self.vlm_client.chat.completions.create(
                model=self.vlm_model,
                messages=[{"role": "user", "content": content}]
            )
            ans = resp.choices[0].message.content.strip().lower()
            # If the answer is not a number, treat as "no"
            if not ans.isdigit():
                return 0
            else:
                return int(ans)
        except Exception as e:
            logger.error(f"VLM Error: {e}")
            return 0

    def split_transcript_semantics(self, transcript: str) -> List[List[int]]:
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
            resp = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            content = resp.choices[0].message.content.strip()
            content = clean_json_response(content)
            return json.loads(content)

        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return []

    def call_llm(self, prompt: str) -> str:
        """Generic LLM call."""
        try:
            resp = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return ""

    def call_vlm(self, video_url: str, prompt: str) -> str:
        """Generic VLM call."""

        try:
            resp = self.vlm_client.chat.completions.create(
                model=self.vlm_model,
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image_url", "image_url": video_url, "mimeType": "video/mp4"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            )
            generated_text = resp.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"VLM Error: {e}")
            return ""

        return generated_text
        