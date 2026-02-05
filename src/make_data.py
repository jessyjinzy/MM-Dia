
import os
import json
import re
import argparse

def multi_speaker_handler(text, speakers):
    #text = "Sizwe: Definitely works better with the silver. Sizwe: Right?"
    #speakers = ["Sizwe:", "Noni:"]

    pattern = f"({'|'.join(re.escape(s) for s in speakers)})"

    parts = re.split(pattern, text)
    dialogues = ''
    utt_spks = []
    for i in range(1, len(parts), 2):
        speaker = parts[i].strip()
        line = parts[i + 1].strip()
        speaker = speakers.index(speaker)
        #dialogues.append((speaker, line))
        dialogues += '[{}]{}'.format(speaker, line)
        utt_spks.append(speaker)

    return dialogues, utt_spks


def make_json_movies(input_dir, output_dir, output_name="mm-dia.jsonl"):
    orig_json_list = os.listdir(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    new_metas = []

    for orig_json in orig_json_list:
        orig_json = os.path.join(input_dir, orig_json)
        print(f"Processing {orig_json}...")
        if not os.path.exists(orig_json):
            print(f"File {orig_json} does not exist, skipping.")
            continue

        metas = json.load(open(orig_json, "r"))

        for meta in metas:
            try:
                dialog_info = {
                    "dialog_id": meta['movie_id'] + '_clip_' + str(meta['clip_id']), # 
                    "path": os.path.abspath(meta['audio_path']),
                    "video_path": meta['video_path'],
                    #"duration": meta['duration'],
                    "speakers": [],
                    "annotation_tag": [],
                    "annotation_text": '',
                    "segments": [],
                    "relationship": meta.get('relationship', None),
                    "interaction_type": meta.get('interaction_type', None),
                }
                dialog_info['speakers'] = list(set(seg['speaker'] + ':' for seg in meta['utterances'] if seg['speaker'].strip() != ''))

                if meta.get('annotation') is not None and isinstance(meta['annotation'], dict):
                    dialog_info["annotation_tag"] = meta['annotation'].get('Tags', [])
                    dialog_info["annotation_text"] = meta['annotation'].get('Summary', '')
                else:
                    print(f"Annotation missing processing dia {dialog_info['dialog_id']}")

                last_end = 0.0
                for i, utt in enumerate(meta['utterances']):
                    if utt["speaker"] == "Background music":
                        continue
                    utt_text, utt_spks = multi_speaker_handler(utt['text'], dialog_info['speakers'])
                    info = {
                        "segment_id": i,
                        "text": utt_text,# utt['text'].split(':', 1)[-1].strip(),  
                        "start": utt['start_time'],
                        "end": utt['end_time'],
                        "index": utt["index"],
                        "speaker": utt_spks, #dialog_info['speakers'].index(utt['speaker']),
                    }


                    assert last_end <= info['start'], f"Error in dia {dialog_info['dialog_id']}, segment {i}: last_end {last_end} > start {info['start']}"

                    last_end = info['end']

                    if info['end'] - info['start'] <= 0.2:
                        print(f"Warning: segment {i} in dia {dialog_info['dialog_id']} has duration: {info['end'] - info['start']:.2f} seconds")
                        continue
                    #assert info['start'] < info['end'], f"Error in dia {dialog_info['dialog_id']}, segment {i}: start {info['start']} >= end {info['end']}"

                    dialog_info["segments"].append(info)

                new_metas.append(dialog_info)
            except Exception as e:
                print(f"Error processing dia {meta.get('movie_id', 'unknown')}_clip_{meta.get('clip_id', 'unknown')}: {e}")
                continue

    with open(os.path.join(output_dir, output_name), "w") as f:
        for meta in new_metas:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=False, default="output_json")
    parser.add_argument("--output_dir", type=str, required=False, default="output_data")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    make_json_movies(input_dir, output_dir)
