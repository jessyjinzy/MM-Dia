from evaluate_wer_seedtts import process_one
import json
import os
import itertools
from tqdm import tqdm
import argparse
import numpy as np

def compute_cpWER(spk0_content: str, spk1_content: str, spk0_content_gt: str, spk1_content_gt: str) -> dict:
    """
    Compute cpWER (cross speaker Word Error Rate) by considering all possible permutations of speakers.
    
    Parameters:
    - spk0_content: Hypothesis content from speaker 0
    - spk1_content: Hypothesis content from speaker 1
    - spk0_content_gt: Ground truth reference content from speaker 0
    - spk1_content_gt: Ground truth reference content from speaker 1
    
    Returns:
    - Dictionary containing the best permutation and corresponding cpWER (Word Error Rate).
    """

    # Initialize variables to track the best WER and the corresponding permutation
    best_wer = float('inf')
    best_permutation = None
    
    # Iterate through each permutation of the hypothesis
    _, hypo, total_wer_1, subs, dele, inse, word_num = process_one(f'{spk0_content} {spk1_content}', f'{spk0_content_gt} {spk1_content_gt}', 'en')
    _, hypo, total_wer_2, subs, dele, inse, word_num = process_one(f'{spk1_content} {spk0_content}', f'{spk0_content_gt} {spk1_content_gt}', 'en')

    # If this permutation results in a lower WER, update the best values
    if total_wer_1 < total_wer_2:
        best_wer = total_wer_1
    else:
        best_wer = total_wer_2

    # Return the result with the best permutation and its cpWER
    return best_wer

def main():
    parser = argparse.ArgumentParser(description="Compute speaker-aware similarity from a long conversation WAV and JSON timestamps.")
    parser.add_argument("--wav_path", type=str, default="", help="Path to long conversation wav.")
    parser.add_argument("--dialogues_dict", type=str, default="./evaluation/examples/gt_trans_spk_split.json")
    args = parser.parse_args()

    dialogues_dict = json.load(open(args.dialogues_dict, 'r', encoding='utf-8'))

    wav_p = (args.wav_path).split('/')[-1]
    new_list = json.load(open(os.path.join(args.wav_path, '_wer_wo_speaker_cpwer.json'), 'r', encoding='utf-8'))

    new_json = {}
    cpWERs = []
    for item in tqdm(new_list):
        spk0_content = []
        spk1_content = []
        if item['response'] == []:
            continue
        try:
            item['response'] = json.loads(item['response'].replace('```json', '').replace('```', '').strip())
        except:
            continue
        if not isinstance(item['response'], list):
            continue
        flag = True
        for part in item['response']:
            if 'speaker' not in part:
                flag = False
                break
            if part['speaker'] == 0:
                spk0_content.append(part['text'].strip())
            elif part['speaker'] == 1:
                spk1_content.append(part['text'].strip())
        if not flag:
            continue
        spk0_content = ' '.join(spk0_content)
        spk1_content = ' '.join(spk1_content)
        spk0_content_gt = dialogues_dict[item['wav']]['spk0']
        spk1_content_gt = dialogues_dict[item['wav']]['spk1']
        
        best_wer = compute_cpWER(spk0_content, spk1_content, spk0_content_gt, spk1_content_gt)
        item['cpWER'] = best_wer
        cpWERs.append(best_wer)
    cpWER_avg = round(np.mean(cpWERs) * 100, 3)
    new_json['cpWER_results'] = cpWER_avg
    new_json['details'] = new_list
    json.dump(new_json, open(os.path.join(args.wav_path, '_cpwer_clean.json'), 'w', encoding='utf-8'), indent=2)

    print('cpWER: ', cpWER_avg, wav_p)

if __name__ == "__main__":
    main()
