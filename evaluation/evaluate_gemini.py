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


# for attribute recall
prompt_1 = """
    Given the above style prompts with the generated audio clips to evaluate, you will evaluate the audio based on six metrics and two additional tags. For each audio clip, provide six integer scores (1-5) separated by commas, followed by two tags: one for the relationship and one for the interaction type. The order should be as follows:

    1. **Gemini-Spontaneity**  
    Rate how spontaneous the audio sounds overall. Does the dialogue feel naturally improvised or does it sound overly scripted or mechanical?  
    1 = Extremely scripted, no spontaneity at all  
    3 = Somewhat spontaneous, but still feels somewhat rigid  
    5 = Completely spontaneous, sounds like a natural conversation with no noticeable script  

    2. **Gemini-Coherence**  
    Evaluate the coherence of transitions between segments or parts of the dialogue. Are the shifts between different parts of the conversation smooth and logical?  
    1 = Transitions are abrupt and confusing  
    3 = Transitions are somewhat natural but still noticeable  
    5 = Transitions are smooth and seamless, highly coherent  

    3. **Gemini-Intelligibility**  
    Rate the clarity of the speech. Is the dialogue easy to understand, or are there any pronunciation, speed, or clarity issues?  
    1 = Extremely hard to understand, major clarity issues  
    3 = Understandable with minor clarity issues  
    5 = Very clear and easy to understand, no issues  

    4. **Gemini-Similarity**  
    Assess the similarity of the speakers to what was intended. How well do the voices match the expected characteristics of the described characters or roles in the dialogue?  
    1 = Completely different from the described voices  
    3 = Somewhat similar, but noticeable differences  
    5 = Very similar, perfectly matches the described voices  

    5. **Gemini-Quality**  
    Evaluate the overall quality of the speech. This includes tone, texture, and emotional delivery. Does the voice quality sound natural, pleasant, and well-suited for the dialogue?  
    1 = Poor quality, unpleasant tone, or very robotic  
    3 = Average quality, some awkwardness or unnatural elements  
    5 = Excellent quality, natural and pleasant sounding  

    6. **Gemini-Instruction Following**  
    Evaluate how well the speech follows the intended instructions or description. Does the speech correspond accurately to the specified tone, mood, or context provided in the description?  
    1 = Completely off from the described instructions  
    3 = Somewhat adheres to the description, but noticeable differences  
    5 = Perfectly follows the description, highly accurate in tone and mood  

    7. **Relationship**  
    Choose the relationship between the characters involved in the dialogue:  
    - Workplace  
    - Friends  
    - Intimate  
    - Family  
    - Adversarial  
    - Individual  
    - Social  
    - Authority  

    8. **Interaction type**  
    Choose the type of interaction in the dialogue:  
    - Persuasion  
    - Conflict  
    - Questioning  
    - Storytelling  
    - Explanation  
    - Commands  
    - Dialogue Exchange  
    - Support Reassurance  
    - Dismissal Rejection  
    - Social Banter  
    - Authority Power  
    - Performance Public  
    - Introspection Reflection  
    - Emotion Release  
    - Invitation Social  

    For each audio clip, please provide the ratings for all six metrics as integers between 1 and 5, followed by the two tags for Relationship and Interaction type. Use the following format:

    **[Gemini-Spontaneity, Gemini-Coherence, Gemini-Intelligibility, Gemini-Similarity, Gemini-Quality, Gemini-Instruction Following, Relationship, Interaction type]**

    Constraints:
    - Only output numbers and commas, no text, no explanations.
    - Each audio clip's scores and tags must be on a separate line.
    - All scores must be integers between 1 and 5 inclusive.
    - Choose one option for both Relationship and Interaction type from the given lists.

    Please respond strictly in the specified format: six integers for the scores, followed by the two tags, one for Relationship and one for Interaction type, one audio clip per line, no extra text.
    """

# for character assignment
prompt_2 = """
    Given the above style prompts with the generated audio clips to evaluate, you will evaluate the audio based on six metrics. For each audio clip, provide six integer scores (1-5) separated by commas, in this exact order:

    1. **Gemini-Spontaneity**  
    Rate how spontaneous the audio sounds overall. Does the dialogue feel naturally improvised or does it sound overly scripted or mechanical?  
    1 = Extremely scripted, no spontaneity at all  
    3 = Somewhat spontaneous, but still feels somewhat rigid  
    5 = Completely spontaneous, sounds like a natural conversation with no noticeable script  

    2. **Gemini-Coherence**  
    Evaluate the coherence of transitions between segments or parts of the dialogue. Are the shifts between different parts of the conversation smooth and logical?  
    1 = Transitions are abrupt and confusing  
    3 = Transitions are somewhat natural but still noticeable  
    5 = Transitions are smooth and seamless, highly coherent  

    3. **Gemini-Intelligibility**  
    Rate the clarity of the speech. Is the dialogue easy to understand, or are there any pronunciation, speed, or clarity issues?  
    1 = Extremely hard to understand, major clarity issues  
    3 = Understandable with minor clarity issues  
    5 = Very clear and easy to understand, no issues  

    4. **Gemini-Similarity**  
    Assess the similarity of the speakers to what was intended. How well do the voices match the expected characteristics of the described characters or roles in the dialogue?  
    1 = Completely different from the described voices  
    3 = Somewhat similar, but noticeable differences  
    5 = Very similar, perfectly matches the described voices  

    5. **Gemini-Quality**  
    Evaluate the overall quality of the speech. This includes tone, texture, and emotional delivery. Does the voice quality sound natural, pleasant, and well-suited for the dialogue?  
    1 = Poor quality, unpleasant tone, or very robotic  
    3 = Average quality, some awkwardness or unnatural elements  
    5 = Excellent quality, natural and pleasant sounding  

    6. **Gemini-Instruction Following**  
    Evaluate how well the speech follows the intended conditional instructions or description. Does the speech correspond accurately to the specified tone, mood, or context provided in the description?  
    1 = Completely off from the described instructions  
    3 = Somewhat adheres to the description, but noticeable differences  
    5 = Perfectly follows the description, highly accurate in tone and mood  

    For each audio clip, please provide the ratings for all six metrics as integers between 1 and 5 in the following format:

    **[Gemini-Spontaneity, Gemini-Coherence, Gemini-Intelligibility, Gemini-Similarity, Gemini-Quality, Gemini-Instruction Following]**

    Constraints:
    - Only output numbers and commas, no text, no explanations.
    - Each audio clip's scores must be on a separate line.
    - All scores must be integers between 1 and 5 inclusive.

    Please respond strictly in the specified format: six integers per audio clip, one audio clip per line, no extra text.
"""


def llm_analysis(prompt, urls, instructions, api_key, base_url="https://api.openai.com/v1"):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.Client(
            base_url=base_url,
            follow_redirects=True,
        ),
    )

    messages=[
        {
            "role": "user",
            "content": [],
        }
    ]
    for i, url in enumerate(urls):
        messages[0]["content"].append({"type": "image_url", "image_url": url, "mimeType": "audio/mp3"})
        messages[0]["content"].append({"type": "text", "text": instructions[i]})
    messages[0]["content"].append({"type": "text", "text": prompt})

    # Previously used model: gemini-2.5-pro
    model = 'gpt-5'
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    generated_text = completion.choices[0].message.content.strip()
    print(completion)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Compute speaker-aware similarity from a long conversation WAV and JSON timestamps.")
    parser.add_argument("--wav_path", type=str, default="./evaluate/mos_data", help="Path to long conversation wav.")
    parser.add_argument("--prompt", type=int, default=1, help="Prompt template type, 1 or 2.")
    parser.add_argument("--api_key", type=str, default=os.getenv("API_KEY"), help="API key for Gemini/OpenAI")
    parser.add_argument("--style_json_file", type=str, default=os.getenv("STYLE_JSON_FILE", "./style_data.json"), help="Path to style JSON file")
    parser.add_argument("--base_url", default=os.getenv("BASE_URL", "https://api.openai.com/v1"), help="OpenAI API base URL")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("API key is required. Set via --api_key or API_KEY environment variable.")

    input_file = os.path.join(args.wav_path, '_wer_wo_speaker.txt')
    api_key = args.api_key

    json_output = []
    start_idx = 0
    output_file = os.path.join(args.wav_path, f"_gemini.json") # _2
    if os.path.exists(output_file):
        json_output = json.load(open(output_file))
        start_idx = len(json_output)

    print(f"Processing file: {output_file}")

    jsondata_style = json.load(open(args.style_json_file))

    style_dict = {}
    for item in jsondata_style:
        try:
            if 'tag' in args.wav_path:
                style_1 = item['annotation_tag'][0]
                style_2 = item['annotation_tag'][1]
                style_3 = item['annotation_tag'][2]

                style_dict[item['dialog_id']] = f'Relationship: {style_1}, Interactional Type: {style_2}, Emotional Tone: {style_3}'
            else:
                style_dict[item['dialog_id']] = item['annotation_text']
        except:
            continue

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in tqdm(range(start_idx, len(lines), 5)):
            urls = []
            instructions = []
            wav_paths = []
            for j in range(i, min(i+5, len(lines))):
                line = lines[j]
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    wav_path, wer, truth, hypo = parts[:4]
                    basename = os.path.basename(wav_path).split("output_")[-1].replace(".wav", "")
                    if basename not in style_dict:
                        continue
                    url = upload_file(wav_path)
                    urls.append(url)
                    instruction = style_dict[basename]
                    instructions.append(instruction)
                

            if args.prompt == 2:
                try:
                    response = llm_analysis(prompt_1, urls, instructions, api_key, args.base_url)
                except Exception as e:
                    response = "[]"
                responses = response.splitlines()

                for line, response in zip(lines[i:i+5], responses):
                    numbers = [int(x.strip()) for x in response.split(',')]
                    if len(numbers) == 6:
                        parts = line.strip().split("\t")
                        wav_path, wer, truth, hypo = parts[:4]
                        json_output.append({"wav": basename, "wer": wer, "truth": truth, "hypo": hypo, "response": numbers})
                    else:
                        print(f"⚠️ Warning: Unexpected output format in {line}")
                for wav_path in wav_paths:
                    delete_file(wav_path)
                json.dump(json_output, open(output_file, "w"), indent=4)

            else:
                try:
                    response = llm_analysis(prompt_1, urls, instructions, api_key, args.base_url)
                except Exception as e:
                    response = "[]"
                responses = response.splitlines()           
                     
                for line, response in zip(lines[i:i+5], responses):
                    parts = response.split(',')
                    
                    # Split the response into 6 numbers and 2 strings
                    try:
                        numbers = [int(x.strip()) for x in parts[:6]]
                        relationship = parts[6].strip()
                        interaction_type = parts[7].strip()
                        
                        if len(numbers) == 6 and relationship and interaction_type:
                            # Split the line into parts and retrieve the necessary information
                            parts_line = line.strip().split("\t")
                            wav_path, wer, truth, hypo = parts_line[:4]
                            
                            # Create the json output and append it
                            json_output.append({
                                "wav": wav_path, 
                                "wer": wer, 
                                "truth": truth, 
                                "hypo": hypo, 
                                "response": numbers, 
                                "relationship": relationship,
                                "interaction_type": interaction_type
                            })
                        else:
                            print(f"⚠️ Warning: Unexpected output format in {line}")
                    
                    except Exception as e:
                        print(f"⚠️ Error processing line: {line} - {e}")

                for wav_path in wav_paths:
                    delete_file(wav_path)
                json.dump(json_output, open(output_file, "w"), indent=4)

    json.dump(json_output, open(output_file, "w"), indent=4)            

if __name__ == "__main__":
    main()
