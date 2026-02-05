from pathlib import Path
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from time import time

import torchaudio
import torch
from tqdm import tqdm
import json
import torch.nn.functional as F
import datasets
import pickle


def load_tokenizer(name):
    if name == "dac":
        import dac
        dac_model_path = dac.utils.download()
        dac_model = dac.DAC.load(dac_model_path)
        dac_model.eval().cuda()
        return dac_model
    elif name == "higgs":
        from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
        AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

        audio_tokenizer = load_higgs_audio_tokenizer(AUDIO_TOKENIZER_PATH, device="cuda")
        return audio_tokenizer

def load_audio(audio_path, target_sr=44100):
    if isinstance(audio_path, Path):
        audio_path = str(audio_path)
    waveform, sr = torchaudio.load(audio_path)
    if sr != 44100:
        waveform = torchaudio.functional.resample(waveform, sr, 44100)
        sr = 44100
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    
    waveform = waveform.view(1, 1, -1)
    return waveform, sr

class HFDataset(Dataset):
    def __init__(self, dataset_path, sample_rate):
        super().__init__()
        try:
            self.dataset = datasets.load_dataset("json", data_files=dataset_path, split="train")
        except:
            self.dataset = datasets.load_from_disk(dataset_path)
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        file = self.dataset[item]['path'] #.replace('.wav', '_44k.wav')

        try:
            data, sr = load_audio(file, self.sample_rate)
        except Exception as e:
            print(file)
            raise e
        if data.shape[-1] == 0:
            print(file)
            return None

        if data.shape[-1] / sr > 120:
            print(f"Warning: {file} exceeds 120 seconds, skipping...")
            return None
        return data, file

        

def collecte_fn(samples):
    samples = list(filter(lambda sample: sample is not None, samples))
    if samples is None or len(samples) == 0:
        return None
    inputs = []
    out_files = []
    input_lens = [int(sample[0].shape[-1]) for sample in samples]
    max_len = max(input_lens)
    for i, sample in enumerate(samples):
        inputs.append(F.pad(sample[0], (0, max_len - input_lens[i]), mode='constant', value=0.))
        out_files.append(sample[1])
    inputs = torch.cat(inputs, dim=0)
    return (inputs, input_lens, out_files)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", 
                        type=str, 
                        required=True,
                        help="Audiofile folder path to be processed, make sure '*.wav' files exist in it or its subfolders.")
    
    parser.add_argument("--output_dir",
                        type=str,
                        default="output_data",
                        help="Output folder to save processed files.")

    args = parser.parse_args()

    input_folder = Path(args.data_path)
    output_folder = Path(args.output_dir)

    suffix = '.wav'
    device = 'cuda'
    sampling_rate = 24000
    channels = 1
    batch_size = 1
    num_workers = 8
    accelerate = Accelerator()

    codec_name = "higgs"
    model = load_tokenizer(codec_name)
    
    ds = HFDataset(args.data_path, sampling_rate)

    # select a subset for testing, let's say 10 samples
    #ds.dataset = ds.dataset.select(range(100))

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collecte_fn, num_workers=num_workers)
    (model, dl) = accelerate.prepare(model, dl)
    
    processed_data = {}

    for batch in tqdm(dl):
        if batch is None:
            continue
            
        with torch.no_grad():
            inputs, input_lens, out_files = batch

            for i, out_file in enumerate(out_files):
                codes = model.encode(out_file).unsqueeze(0)
                #out_file = Path(str(out_file).replace(input_folder, output_folder))
                out_len = codes.shape[-1]
                #print(out_file, codes[i].shape, out_len)
                processed_data[str(out_file)] = codes[i, :, :out_len]

    # saving processed data in pickle format
    pickle_file = os.path.join(output_folder, f'{codec_name}_codes', input_folder.stem + '.pkl')
    os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
    with open(pickle_file, 'wb') as f:
        pickle.dump(processed_data, f)
    print(f"Saved processed data to {pickle_file}")

    if isinstance(ds, HFDataset):
        ds = ds.dataset
        ds = ds.map(lambda x: {f"{codec_name}_codes": processed_data.get(str(x["path"]), None)})
        ds = ds.filter(lambda x: x[f"{codec_name}_codes"] is not None)
        #out_path = os.path.join(output_folder, input_folder.stem)
        out_path = output_folder
        print(f"Saving {len(ds)} dialogues to disk at {out_path}")
        ds.save_to_disk(out_path)

    exit(0)