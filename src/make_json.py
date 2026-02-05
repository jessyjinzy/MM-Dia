import os
import json
import srt
import re
from datetime import timedelta
import argparse

import torchaudio

def get_duration(audio_path):
    audio, sr = torchaudio.load(audio_path)
    return round(audio.size(1) / sr, 3) # Return duration in seconds


def process_annotation(annotation_dir, audio_root, prefix, output_dir):
    valid_clips = []

    dispose_movies = ['公主日记1', '公主日记2', '天生一对', '消失的爱人', '超凡蜘蛛侠1', '风雨哈佛路']

    for json_file in os.listdir(annotation_dir):
        json_path = os.path.join(annotation_dir, json_file)
        if not json_file.endswith('.json'):
            continue

        print(f"Processing {json_path}...")
        try:
            metas = json.load(open(json_path, 'r', encoding='utf-8'))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {json_path}: {e}")
            continue

        if metas['movie'] in dispose_movies:
            print(f"Skipping movie {metas['movie']} in {json_path}")
            continue

        movie_id = json_file.replace('.json', '')
        audio_dir = os.path.join(audio_root, movie_id)
        if not movie_id.startswith('tt'): # tv series
            movie_id = prefix + '_' + movie_id
        movie_id = movie_id.replace(' ', '_')

        for clip in metas['clips']:
            clip_id = int(clip['clip_id'])
            audiopath = os.path.join(audio_dir, f"clip_{clip_id:0>3d}", f"clip_{clip_id:0>3d}.wav")
            if 'mceiu' in prefix:
                audiopath = os.path.join(audio_root, f"dia_{clip_id}.wav")

            speakers = clip.get('speakers', clip.get('speaker', []))

            try:
                valid_clip = {
                    "movie_id": movie_id,
                    "clip_id": clip_id,
                    "audio_path": audiopath,
                    "video_path": clip.get('video_path', ''),
                    "key": clip.get('key', ''),
                    #"duration": get_duration(audiopath),
                    "speakers": speakers,
                    "annotation": clip.get('style_annotation', None),
                    "utterances": [],
                    "relationship": clip.get('relationship', None),
                    "interaction_type": clip.get('interaction_type', None),
                }

                # Skip clips in the beginning of the movie
                last_utt = clip['srt'][0]
                if ('abs_start_time' in last_utt or 'start_time' in last_utt) and srt.srt_timestamp_to_timedelta(last_utt.get('abs_start_time', last_utt.get('start_time', None))) < timedelta(seconds=30):
                    continue

                for i, utt in enumerate(clip['srt']):
                    ######## -- validation speaker for each utterance -- ##########
                    if 'mceiu' in prefix:
                        utt['character_annotation'] = utt["speaker"] + ": " + utt["text"]
                    if isinstance(utt['character_annotation'], dict):
                        utt['character_annotation'] = utt['character_annotation'].get('text', '')
                    speaker = utt.get('character_annotation', '').split(':')[0].strip()
                    assert speaker != '' and speaker in speakers, "Utt-Speaker {} not found in movie {} for clip_id: {}".format(speaker, json_path, clip_id)
                    ########### validation speaker for each utterance #############

                    # By far, we have ensured that speaker is defined
                    try:
                        start_time = srt.srt_timestamp_to_timedelta(utt['relative_start']).total_seconds() if isinstance(utt['relative_start'], str) else float(utt['relative_start'])
                        end_time = srt.srt_timestamp_to_timedelta(utt['relative_end']).total_seconds() if isinstance(utt['relative_end'], str) else float(utt['relative_end'])
                    except Exception as e:
                        print(f"Error converting time for clip_id {clip_id}, utterance index {i}: {e}, Skipping")
                        continue

                    valid_clip['utterances'].append({
                        "index": utt["index"],
                        "start_time": start_time,
                        "end_time": end_time,
                        "text": utt['character_annotation'],
                        "speaker": speaker
                    })

                if len(valid_clip['utterances']) > 0:
                    valid_clips.append(valid_clip)

            except KeyError as e:
                print(f"KeyError \tfor movie_id {json_path} clip_id {clip_id}:\t {e}")
                raise e
            except AssertionError as e:
                print(e)
                #print(f"AssertionError \tfor movie_id {json_path} clip_id {clip_id}:\t {e}")
                continue
            except Exception as e:
                print(f"Unexpected error \tfor movie_id {json_path} clip_id {clip_id}:\t {e}")
                raise e


    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, prefix + '.json')
    json.dump(valid_clips, open(output_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    return valid_clips

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process movie/tv show annotations and audio files.")
    parser.add_argument('--annotation_dir', type=str, default='../release/json', help='Path to the annotation directory.')
    parser.add_argument('--audio_root', type=str, default='../release/audio', help='Path to the audio root directory.')
    parser.add_argument('--prefix', type=str, default='movie_48', help='Prefix for different data batches.')
    parser.add_argument('--output_dir', type=str, default='output_json', help='Directory to save output json.')
    args = parser.parse_args()

    annotation_dir = os.path.join(args.annotation_dir, args.prefix)
    audio_root = os.path.join(args.audio_root, args.prefix)
    prefix = args.prefix

    valid_clips = process_annotation(annotation_dir, audio_root, prefix, args.output_dir)

    # Now do some statistics
    # 1. the count of clips
    print(f"Total valid clips: {len(valid_clips)}")
    # 2. the avg. count of speakers
    avg_speakers = sum(len(clip['speakers']) for clip in valid_clips) / len(valid_clips) if valid_clips else 0
    print(f"Average number of speakers per clip: {avg_speakers:.2f}")
    # Count of <=2 speakers clips
    count_2_speakers = sum(1 for clip in valid_clips if len(clip['speakers']) <= 2)
    print(f"Count of clips with 2 or fewer speakers: {count_2_speakers}")

    # 3. the avg. count of utterances
    avg_utterances = sum(len(clip['utterances']) for clip in valid_clips) / len(valid_clips) if valid_clips else 0
    print(f"Average number of utterances per clip: {avg_utterances:.2f}")
    # 4. the avg. duration of utterances
    avg_duration = sum((utt['end_time'] - utt['start_time'] for clip in valid_clips for utt in clip['utterances'])) / sum(len(clip['utterances']) for clip in valid_clips) if valid_clips else 0
    print(f"Average duration of utterances: {avg_duration:.2f} seconds")
    # 5. the avg. duration of clips
    avg_clip_duration = sum((clip['utterances'][-1]['end_time'] - clip['utterances'][0]['start_time'] for clip in valid_clips)) / len(valid_clips) if valid_clips else 0
    print(f"Average duration of clips: {avg_clip_duration:.2f} seconds")


