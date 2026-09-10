import argparse
import httpx
from openai import OpenAI
import glob
import re
import json
import os
from tqdm import tqdm
import shutil

SERVER_IP = os.getenv("SERVER_IP", "localhost")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads/")

def upload_file(local_path):
    filename = f"{local_path.split('/')[-2]}_{local_path.split('/')[-1]}"
    target_path = os.path.join(UPLOAD_DIR, filename)
    shutil.copy(local_path, target_path)
    return f"http://{SERVER_IP}/upload/{filename}"

def delete_file(local_path):
    filename = f"{local_path.split('/')[-2]}_{local_path.split('/')[-1]}"
    target_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(target_path):
        os.remove(target_path)
        return True
    return False


# for cp_wer evaluation
prompt_1 = """Please do the speaker diarization for the audio clip with the following dialogue transcript, and directly respond with the diarization result for each line of the given srt.
Dialogue Transcript:
{transcript}
Please directly respond with the diarization result for each line of the given srt, in the Json format of: 
[{{"start": start_time_in_seconds, "end": end_time_in_seconds, "speaker": speaker_id, "text": text}}, ...]
where speaker_id is an integer starting from 0.
Please ensure the following:
1. The start and end times are in seconds with one decimal place.
2. The text is exactly the same as in the given srt. Do not change any text, including punctuation and capitalization.
3. The diarization result should cover all lines in the given srt, without any omission or addition.
4. The speaker_id should be consistent for the same speaker throughout the dialogue.
5. The response should be a valid JSON array.
6. Do the speaker diarization ONLY by the detection of the timbre switch in the audio clip, DO NOT rely on the semantic of dialogue transcript.
"""

# for annotation
prompt_2 = """Please do the speaker diarization for the audio clip with the following dialogue transcript, and directly respond with the diarization result for each line of the given srt.
Dialogue Transcript:
{transcript}
Please directly respond with the diarization result for each line of the given srt, in the Json format of: 
[speaker_id_0, speaker_id_1, speaker_id_2...]
where speaker_id is an integer starting from 0.
Please ensure the following:
1. The diarization result should cover all lines in the given srt, without any omission or addition.
4. The speaker_id should be consistent for the same speaker throughout the dialogue.
5. The response should be a valid JSON array.
6. Do the speaker diarization ONLY by the detection of the timbre switch in the audio clip, DO NOT rely on the semantic of dialogue transcript.
"""

def llm_analysis(hypo, url, api_key, prompt_type=1, base_url="https://api.openai.com/v1"):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.Client(
            base_url=base_url,
            follow_redirects=True,
        ),
    )
    if prompt_type == 1:
        prompt = prompt_1.format(transcript=hypo)
    else:
        prompt = prompt_2.format(transcript=hypo)

    # Previously used model: gemini-2.5-flash
    model = 'gpt-5'
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": url, "mimeType": "audio/mp3"},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    generated_text = completion.choices[0].message.content.strip()
    print(completion)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Compute speaker-aware similarity from a long conversation WAV and JSON timestamps.")
    parser.add_argument("--wav_path", type=str, default="", help="Path to long conversation wav.")
    parser.add_argument("--prompt", type=int, default=1, help="Prompt template type, 1 or 2.")
    parser.add_argument("--api_key", type=str, default=os.getenv("API_KEY"), help="API key for Gemini/OpenAI")
    parser.add_argument("--base_url", default=os.getenv("BASE_URL", "https://api.openai.com/v1"), help="OpenAI API base URL")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("API key is required. Set via --api_key or API_KEY environment variable.")

    input_file = os.path.join(args.wav_path, "_wer_wo_speaker.txt")
    print(f"Processing file: {input_file}")
    api_key = args.api_key

    # Read and process
    json_output = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                wav_path, wer, truth, hypo = parts[:4]
                basename = os.path.basename(wav_path).split("output_")[-1].replace(".wav", "")
                url = upload_file(wav_path)
                try:
                    response = llm_analysis(hypo, url, api_key, args.prompt, args.base_url)
                except Exception as e:
                    response = "[]"
                json_output.append({"wav": basename, "wer": wer, "truth": truth, "hypo": hypo, "response": response})
                delete_file(wav_path)
                if len(json_output) % 5 == 0:
                    json.dump(json_output, open(input_file.replace(".txt", f"_cpwer.json"), "w"), indent=4)
    json.dump(json_output, open(input_file.replace(".txt", f"_cpwer.json"), "w"), indent=4)     

if __name__ == "__main__":
    main()
