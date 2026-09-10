import torch
import soundfile as sf
import librosa
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional, Tuple
import random
from tqdm import tqdm
import torchaudio
import numpy as np

from datasets import load_dataset, concatenate_datasets, load_from_disk, Sequence, Value

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from dataclasses import asdict
from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator, HiggsAudioBatchInput
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent

from transformers.loss.loss_utils import ForCausalLMLoss, fixed_cross_entropy

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
def calc_trainable_parameters(model):
    """
    Calculate the number of trainable parameters in the model.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}, Total parameters: {total_params}")
    return trainable_params, total_params

def load_dataset_from_list(data_list: List[Any], split: str = "train") -> Dict[str, Any]:
    dataset_lists = []
    idx2pairs = []
    for dataset_idx, dataset_path in enumerate(data_list):
        if dataset_path.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=dataset_path, split=split)
        else:
            dataset = load_from_disk(dataset_path)
        # Remove unnecessary columns if they exist
        if "duration" in dataset.column_names:
            dataset = dataset.remove_columns(["duration"])

        # Convert annotation_text to a sequence of strings if it's not already

        if isinstance(dataset.features["annotation_text"], Value):
            dataset = dataset.map(lambda x: {"annotation_text": [x["annotation_text"]]})

        dataset = dataset.cast_column("annotation_text", Sequence(Value("string")))
        dataset = dataset.cast_column("annotation_tag", Sequence(Value('string')))

        # filter out speakers with speakers > 5

        print(f"Original dataset size: {len(dataset)} for {dataset_path}")
        def agg_speakers(item):
            spks = []
            for seg in item["segments"]:
                if isinstance(seg["speaker"], list):
                    spks += seg["speaker"]
                else:
                    spks += [seg["speaker"]]
            return len(set(spks))
        dataset = dataset.map(lambda x: {"num_speakers": agg_speakers(x)})
        print("Max num: ", max(dataset["num_speakers"]))
        dataset = dataset.filter(lambda x: x["num_speakers"] <= 5)
        print("Filtered dataset size:", len(dataset))


        # Ensure the dataset has the required columns

        dataset_lists.append(dataset)
        idx2pairs += [(dataset_idx, i) for i in range(len(dataset))]
        print(f"Loaded dataset {dataset_idx} with {len(dataset)} samples from {dataset_path}")
    
    #conversational_dataset = concatenate_datasets(dataset_lists)
    
    return dataset_lists, idx2pairs

def text_normalization(text: str) -> str:
    """
    Normalize the text by removing extra spaces and converting to lowercase.
    """
    text_norm = ''
    for char in text:
        # remove emojis and icons
        if ord(char) > 127:
            continue
        text_norm += char
    
    return text_norm  # Convert to lowercase


def normalize_chinese_punctuation(text):
    """
    Convert Chinese (full-width) punctuation marks to English (half-width) equivalents.
    """
    # Mapping of Chinese punctuation to English punctuation
    chinese_to_english_punct = {
        "，": ", ",  # comma
        "。": ".",  # period
        "：": ":",  # colon
        "；": ";",  # semicolon
        "？": "?",  # question mark
        "！": "!",  # exclamation mark
        "（": "(",  # left parenthesis
        "）": ")",  # right parenthesis
        "【": "[",  # left square bracket
        "】": "]",  # right square bracket
        "《": "<",  # left angle quote
        "》": ">",  # right angle quote
        "“": '"',  # left double quotation
        "”": '"',  # right double quotation
        "‘": "'",  # left single quotation
        "’": "'",  # right single quotation
        "、": ",",  # enumeration comma
        "—": "-",  # em dash
        "…": "...",  # ellipsis
        "·": ".",  # middle dot
        "「": '"',  # left corner bracket
        "」": '"',  # right corner bracket
        "『": '"',  # left double corner bracket
        "』": '"',  # right double corner bracket
    }

    # Replace each Chinese punctuation with its English counterpart
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)

    return text

import re
def text_norm(text):
    pattern = re.findall(r'<[^>]+>|\([^)]*\)|\[(?!\d+\])[^][]+\]', text)
    tags = pattern.copy()
    for tag in pattern:
        text = text.replace(tag, '')

    text = text.replace('\n', ' ').replace('\r', ' ').replace('\\', '').replace('#', '')
    text = normalize_chinese_punctuation(text)

    unicodes = [c for c in text if ord(c) > 127]
    text = ''.join([c for c in text if ord(c) < 128])

    return text.strip(), tags, unicodes


def speaker_norm(texts, speaker_map):
    # extract speaker and text in the segment
    #   [S1] Hello [S2] Hi [S3] How are you?  --> [(S1, Hello), (S2, Hi), (S3, How are you?)]
    texts, _, _ = text_norm(texts)
    segments = re.findall(r'\[([^\]]+)\]([^\[]+)', texts)

    speaker_list = []
    normed_dialog = ""

    for speaker, text in segments:
        norm_t, tags, unicodes = text_norm(text)
        if norm_t == '':
            continue

        # re-order the speaker id
        if speaker in speaker_list:
            speaker_id = speaker_list.index(speaker)
        else:
            speaker_id = len(speaker_list)
            speaker_list += [speaker]

        normed_dialog += "{}{}".format(speaker_map[speaker_id], norm_t)
    
    return normed_dialog, speaker_list
            


class LocalHiggsDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_paths, model_path, duration_limit = 30, frame_rate = 25):

        self.data, self.idx = load_dataset_from_list(dataset_paths.split(","))

        self.frame_rate = frame_rate

        #length_dur = []
        #for row in tqdm(self.data):
        #    length_dur += [torch.tensor(row['dac_codes']).shape[-1]]

        ## bucket statistics
        #bound = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        #for i in range(len(bound) - 1):
        #    count = sum(1 for x in length_dur if x < bound[i + 1])
        #    print(f"Bucket {i}: {bound[i]} - {bound[i + 1]}: {count} samples, {count / len(length_dur) * 100:.2f}%")

        self.duration_limit = duration_limit  # seconds

        self.cache = {}
        # official prompt schema: [SPEAKER1]
        self.speaker_map = [f'[SPEAKER{i}]' for i in range(10)]

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def __len__(self) -> int:
        return len(self.idx)
    
    def convert_json_to_chatmlsample(self, feat: dict):
        scene_desc = feat.get("annotation_text", " ")
        messages = [
            Message(
                role="system",
                content=f"{MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>",
            ),
            Message(
                role="user",
                content=feat["text"],
            ),
            Message(
                role="assistant",
                content=AudioContent(audio_url=""),
            )
        ]
        chatml_sample = ChatMLSample(messages=messages)

        input_tokens, label_tokens, audio_contents, _ = prepare_chatml_sample(chatml_sample, self.tokenizer)
        
        audio_ids = feat["audio"]

        #print(f"Input tokens: {self.tokenizer.decode(input_tokens)}, Label tokens: {label_tokens}, Audio contents: {audio_contents}, Audio IDs shape: {audio_ids.shape}")

        context_audio_ids = audio_ids
        curr_sample = ChatMLDatasetSample(
            input_ids=torch.LongTensor(input_tokens),
            label_ids=torch.LongTensor(label_tokens) if label_tokens else None,
            audio_ids_concat=context_audio_ids,
            audio_ids_start=torch.LongTensor([0]),
            audio_waveforms_concat=None,
            audio_waveforms_start=None,
            audio_sample_rate=None,
            audio_speaker_indices=None,
        )
        return curr_sample
        

    def __getitem__(self, idx: int):

        data_idx, sample_idx = self.idx[idx]
        row = self.data[data_idx][sample_idx]

        # data augmentation
        # randomly choose a span with < 30seconds
        idx = random.choice(list(range(len(row["segments"]))))
        start_time = row["segments"][idx]["start"]
        end_time = row["segments"][idx]["end"]
    

        # 1. only keep a span of < 30 seconds
        # 2. remove tags wrapped in <>
        # 3. re-order the speaker id based on the first appearance
        # 4. add speaker id to the annotation

        text = ''

        for segment, next_seg in zip(row["segments"][idx:], row["segments"][idx+1:] + [None]):
            if segment["end"] - start_time > self.duration_limit:
                break

            if isinstance(segment["speaker"], list):
                #print(row["dialog_id"], segment["text"], segment["speaker"])
                # multi-speakers already embedded in the text, wrapped by []
                text += segment["text"]
            else:
                text += "[{}]{}".format(segment["speaker"], segment["text"])
            
            end_time = segment["end"]
            if next_seg is not None and end_time < next_seg['start']:
                end_time = min(next_seg['start'], segment['end'] + 2) # add small gap if possible, only 2 seconds

        text, speaker_list = speaker_norm(text, self.speaker_map)

        # text: [SPEAKER0] 
        # original text: speaker_list[0]
        # original speaker: row["speakers"][speaker_list[0]]
        try:
            orig_speaker_list = row.get("speakers", [])
            annotation_speaker = "SPEAKER_MAP: " + "\n".join([
                "{} -> {}".format(self.speaker_map[i][1:-1], orig_speaker_list[int(speaker)].strip().replace(':', ''))
                    for i, speaker in enumerate(speaker_list)])
        except:
            annotation_speaker = ''

        #print("speaker annotation: ", annotation_speaker)

        audio = torch.tensor(row["higgs_codes"])#.unsqueeze(0) # C x T

        assert audio.shape[-1] > 0, f"Audio data is empty for {row['dialog_id']} at index {idx}"

        audio = audio[..., int(start_time * self.frame_rate): int(end_time * self.frame_rate)]

        annotation_tag = row.get("annotation_tag", [])
        annotation_text = row.get("annotation_text", " ")
        if isinstance(annotation_tag, list) and len(annotation_tag) == 3:
            annotation_tag = 'Relationship: {}, Interaction Type: {}, Emotional State: {}'.format(annotation_tag[0], annotation_tag[1], annotation_tag[2])
        else:
            annotation_tag = ''

        if isinstance(annotation_text, list):
            annotation_text = random.choice(annotation_text) if annotation_text else ''
        
        if random.random() < 0.1 or text == '':
            text = ' '

        # annotation_speaker, annotation_tag, annotation_text
        # randomly drop 
        # stage-1 (40% of the steps): 50% annotation_tag, 50% annotation_text plus 15% dropout

        if annotation_tag and (not annotation_text or random.random() < 0.5):
            # randomly choose to use annotation tag or text
            annotation_text_stage1 = annotation_tag
        else:
            annotation_text_stage1 = annotation_speaker + '\n' + annotation_text

        if random.random() < 0.15 or annotation_text_stage1 == '':
            # randomly drop 
            annotation_text_stage1 = ' '

        ## stage-2 (30% of the steps): 50% annotation_tag concat text 
        #if random.random() < 0.5:
        #    annotation_text_stage2 = f"{annotation_tag}\n{annotation_text}"
        #else:
        #    annotation_text_stage2 = annotation_text_stage1

        ## stage-3 (30% of the steps): annotation_speaker + annotation_tag + annotation_text plus 15% dropout
        #if random.random() < 0.5:
        #    annotation_text_stage3 = f"{annotation_speaker}\n{annotation_text_stage2}"
        #else:
        #    annotation_text_stage3 = annotation_text_stage2

        #print("annotation_text: \n", annotation_text)

        #print(f"Audio shape: {audio.shape}, Text: {text}, Annotation: {annotation_text}")
        annotation_stages = {
            "stage1": annotation_text_stage1,
            #"stage2": annotation_text_stage2,
            #"stage3": annotation_text_stage3
        }

        feat = {
            #"input_ids": audio[0], # this is for length sampler
            "text": text,
            "annotation_text": annotation_text,
            "audio": audio
        }

        feat_dict = {}
        for stage_id, annotation_stage in annotation_stages.items():
            feat["annotation_text"] = annotation_stage
            feat_dict[stage_id] = self.convert_json_to_chatmlsample(feat)

        #return self.convert_json_to_chatmlsample(feat)
        return feat_dict

MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""


class InstructionalHiggsDataCollator:
    def __init__(self, config: HiggsAudioConfig):
        self._collator = HiggsAudioSampleCollator(
            whisper_processor=None,
            audio_in_token_id=config.audio_in_token_idx,
            audio_out_token_id=config.audio_out_token_idx,
            audio_stream_bos_id=config.audio_stream_bos_id,
            audio_stream_eos_id=config.audio_stream_eos_id,
            encode_whisper_embed=config.encode_whisper_embed,
            pad_token_id=config.pad_token_id,
            return_audio_in_tokens=config.encode_audio_in_tokens,
            use_delay_pattern=config.use_delay_pattern,
            round_to=1,
            audio_num_codebooks=config.audio_num_codebooks,
        )

    def __call__(self, features: List[ChatMLDatasetSample]):
        # features: List[Dict[ChatMLDatasetSample]]
        # change to Dict[List[ChatMLDatasetSample]]
        feat_list_dict = {}
        for feat in features:
            for stage_id in feat:
                if stage_id not in feat_list_dict:
                    feat_list_dict[stage_id] = []
                feat_list_dict[stage_id].append(feat[stage_id])
        

        batch_dict = {}
        for stage_id in feat_list_dict:
            batch_data = self._collator(feat_list_dict[stage_id])
            batch = asdict(batch_data)
            batch_dict[stage_id] = batch
        
        #print(type(features))
        #batch_data = self._collator(features)
        #batch = asdict(batch_data)
        if len(batch_dict) == 1:
            # if only one stage, return the inner dict directly
            return list(batch_dict.values())[0]

        return batch_dict


from utils import calculate_topk_accuracy, compute_metrics

def ForCausalLMLossReductionNone(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    unflatten_shape = shift_labels.shape
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = torch.nn.functional.cross_entropy(
        shift_logits, shift_labels, ignore_index=ignore_index, reduction='none')
    return loss.view(*unflatten_shape)

class HiggsAudioTrainer(Trainer):


    def stage_sampler(self, batch_dict):
        global_step = self.state.global_step
        total_steps = self.args.max_steps if self.args.max_steps > 0 else self.args.num_train_epochs * len(self.train_dataset) // (self.args.train_batch_size * self.args.gradient_accumulation_steps * max(1, self.args.world_size))

        #print("Global step: {}, Total steps: {}".format(global_step, total_steps))

        stage_id = "stage1"
        #if global_step < total_steps * 0.4:
        #    stage_id = "stage1"
        #elif global_step < total_steps * 0.7:
        #    stage_id = "stage2"
        #else:
        #    stage_id = "stage3"
        return batch_dict[stage_id]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        #inputs = self.stage_sampler(inputs)
        
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        #for key in inputs:
        #    print(f"Input key: {key}, Input shape: {inputs[key].shape if isinstance(inputs[key], torch.Tensor) else 'N/A'}, Input[0]: {inputs[key][0] if isinstance(inputs[key], torch.Tensor) and len(inputs[key]) > 0 else 'N/A'}")
        #print(f"Inputs: {inputs.keys()}")
        
        #print(f"audio_out_ids_start: {inputs.get('audio_out_ids_start', 'N/A')}, audio_out_ids_start_group_loc: {inputs.get('audio_out_ids_start_group_loc', 'N/A')}")
        try:
            outputs = model(**inputs)
        except Exception as e:
            print(f"Error during model forward pass: {e}")
            # saving the inputs for debugging with pickle
            import pickle
            with open(f'debug_inputs_rank_{self.accelerator.process_index}.pkl', 'wb') as f:
                pickle.dump(inputs, f)
            raise e

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        #for key in outputs:
        #    print(f"Output key: {key}, Output shape: {outputs[key].shape if isinstance(outputs[key], torch.Tensor) else 'N/A'}, Output[0]: {outputs[key][0] if isinstance(outputs[key], torch.Tensor) and len(outputs[key]) > 0 else 'N/A'}")

        # compute custom loss
        #loss = self.compute_loss_func(outputs)
        # text logits and labels
        logits = outputs["logits"]
        labels = outputs["expanded_labels"]
        # logits: (batch_size, seq_len, text_vocab_size)
        # labels: (batch_size, seq_len)
        text_vocab_size = logits.shape[-1]
        loss_text = ForCausalLMLoss(logits, labels, text_vocab_size)

        # ---------------------------------
        # [Un-Optimized]: non-parallel version
        # ---------------------------------
        #losses = [ForCausalLMLoss(logits, labels, text_vocab_size)]

        #num_audio_out = inputs["audio_out_ids_start"].shape[0]

        ##print("Audio labels: ", inputs["label_audio_ids"][0], inputs["label_audio_ids"][1])
        #for i in range(num_audio_out):
        #    segment_start = inputs["audio_out_ids_start"][i]
        #    segment_end = inputs["audio_out_ids_start"][i + 1] if i + 1 < num_audio_out else None

        #    segment_logits = outputs["audio_logits"][ segment_start:segment_end].transpose(0, 1).contiguous() # (seq, num_codebooks, vocab_size) -> (num_codebooks, seq, vocab_size)
        #    segment_ids = inputs["label_audio_ids"][...,segment_start:segment_end] # (num_codebooks, seq)

        #    audio_vocab_size = segment_logits.shape[-1]
        #    losses += [ForCausalLMLoss(segment_logits, segment_ids, audio_vocab_size)]

        # ---------------------------------
        # [Optimized]: parallel version
        # ---------------------------------

        audio_logits = outputs["audio_logits"].transpose(0, 1).contiguous()  # (seq, num_codebooks, vocab_size) -> (num_codebooks, seq, vocab_size)
        audio_labels = inputs["label_audio_ids"] # (num_codebooks, seq_len)
        audio_vocab_size = audio_logits.shape[-1]
        # ----------------------
        # Un-weighted version
        #loss_audio = ForCausalLMLoss(audio_logits, audio_labels, audio_vocab_size)
        #print("Direct compute:", loss_audio)

        loss_weights = [self.semantic_amplification] + [1] * 7

        num_channels = audio_logits.shape[0]
        num_items_in_batch = (audio_labels != -100).sum() / num_channels # all layers have the same number of valid tokens, calculate tokens from single layer here.
        loss_audio = ForCausalLMLossReductionNone(audio_logits, audio_labels, audio_vocab_size)

        loss_weights = torch.tensor(loss_weights, device=loss_audio.device, dtype=torch.float).view(num_channels, 1) 
        loss_audio = (loss_audio * loss_weights).sum() / (num_items_in_batch * loss_weights.sum() + 1e-6)

        loss = loss_text + loss_audio

        # ------- Override the following loss computation logic -----------
        #if labels is not None:
        #    unwrapped_model = self.accelerator.unwrap_model(model)
        #    if _is_peft_model(unwrapped_model):
        #        model_name = unwrapped_model.base_model.model._get_name()
        #    else:
        #        model_name = unwrapped_model._get_name()
        #    # User-defined compute_loss function
        #    if self.compute_loss_func is not None:
        #        loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
        #    elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #        loss = self.label_smoother(outputs, labels, shift_labels=True)
        #    else:
        #        loss = self.label_smoother(outputs, labels)
        #else:
        #    if isinstance(outputs, dict) and "loss" not in outputs:
        #        raise ValueError(
        #            "The model did not return a loss from the inputs, only the following keys: "
        #            f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        #        )
        #    # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # -----------------------------------------------------------------

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

    
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        #inputs = self.stage_sampler(inputs)

        if prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

        eval_inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**eval_inputs)

        # Extract text logits and labels 
        text_logits = outputs["logits"]
        text_labels = outputs["expanded_labels"]
        text_vocab_size = text_logits.shape[-1]

        loss_text = ForCausalLMLoss(text_logits, text_labels, text_vocab_size)
        text_metrics = calculate_topk_accuracy(text_logits, text_labels)

        # Extract audio logits and labels
        audio_logits = outputs["audio_logits"].transpose(0, 1).contiguous()
        audio_labels = inputs["label_audio_ids"]
        audio_vocab_size = audio_logits.shape[-1]

        loss_weights = [self.semantic_amplification] + [1] * 7

        num_channels = audio_logits.shape[0]
        num_items_in_batch = (audio_labels != -100).sum() / num_channels # all layers have the same number of valid tokens, calculate tokens from single layer here.
        loss_audio = ForCausalLMLossReductionNone(audio_logits, audio_labels, audio_vocab_size)
        audio_semantic_metrics = calculate_topk_accuracy(audio_logits[:1], audio_labels[:1])
        audio_acoustic_metrics = calculate_topk_accuracy(audio_logits[1:], audio_labels[1:])

        loss_weights = torch.tensor(loss_weights, device=loss_audio.device, dtype=torch.float).view(num_channels, 1) 
        loss_audio = (loss_audio * loss_weights).sum() / (num_items_in_batch * loss_weights.sum() + 1e-6)

        loss = loss_text + loss_audio

        device = model.device
        predictions = [
            torch.tensor(text_metrics["top1_accuracy_sum"], device=device),
            torch.tensor(text_metrics["top10_accuracy_sum"], device=device),
            torch.tensor(text_metrics["num_valid_tokens"], device=device),
            torch.tensor(audio_semantic_metrics["top1_accuracy_sum"], device=device),
            torch.tensor(audio_semantic_metrics["top10_accuracy_sum"], device=device),
            torch.tensor(audio_semantic_metrics["num_valid_tokens"], device=device),
            torch.tensor(audio_acoustic_metrics["top1_accuracy_sum"], device=device),
            torch.tensor(audio_acoustic_metrics["top10_accuracy_sum"], device=device),
            torch.tensor(audio_acoustic_metrics["num_valid_tokens"], device=device),
        ]
        labels = text_labels

        predictions = [v.to(model.device) for v in predictions]

        return (loss, predictions, labels)


def main(args):
    device = torch.device(args.device)

    set_seed(args.seed)
 
    # -- Load model and processor --
    print("Loading model and processor...")
    model = HiggsAudioModel.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    calc_trainable_parameters(model)
    model.freeze_text_head()
    calc_trainable_parameters(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable parameter: {name}, Shape: {param.shape}")

    print("Model and processor loaded successfully.")

    # -- Load and prepare dataset --
    print("Loading dataset...")
    # group by 'dialog_id' to ensure each sample is a complete conversation
    train_dataset = LocalHiggsDataset(args.train_data_path, model_path=args.model_id)
    val_dataset = LocalHiggsDataset(args.val_data_path, model_path=args.model_id)


    print("Dataset loaded successfully.")
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    # -- Data collator --
    #orig_collator = ConversationalDataCollator(processor=processor)
    #data_collator = InstructionalConversationalDataCollator(use_instruction=args.instruction)
    config = HiggsAudioConfig.from_pretrained(args.model_id)
    data_collator = InstructionalHiggsDataCollator(config)

    # -- Training arguments --
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size, 
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=args.accumulation_steps, 
        eval_strategy="epoch",
        #eval_steps=10,
        save_strategy="steps",
        learning_rate=args.lr,
        weight_decay=0.01,
        max_grad_norm=5,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=1000,
        bf16=True,
        fp16=False,
        #group_by_length=True,
        report_to="tensorboard",
        dataloader_num_workers=4,
        remove_unused_columns=False, # Keep original columns because we use group_by
        ddp_find_unused_parameters=False,
    )

    # -- Initialize Trainer --
    trainer = HiggsAudioTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.semantic_amplification = args.semantic_amplification

    # -- Start training --
    print("Starting training...")
    if args.resume_from_checkpoint:
        print("Resuming from the last checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("Starting training from scratch...")
        trainer.train()
    result = trainer.evaluate()
    print("Results:", result)

    # -- Save final model --
    trainer.save_model(args.output_dir)
    print(f"Final model and processor saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Dia model on conversational dataset.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproductivity.")
    parser.add_argument("--model_id", type=str, default="bosonai/higgs-audio-v2-generation-3B-base", help="Path to the pre-trained CSM model.")
    parser.add_argument("--train_data_path", type=str, default="../output_data/mm_dia_splits/train", help="Path to the training data in JSON format.")
    parser.add_argument("--val_data_path", type=str, default="../output_data/mm_dia_splits/train", help="Path to the training data in JSON format.")
    parser.add_argument("--output_dir", type=str, default="./exp/higgs_audio_hf-finetuned", help="Directory to save the fine-tuned model.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the training on (cuda or cpu).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--instruction", action='store_true', help="Whether to use instruction-based training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--freeze_text_encoder", action='store_true', help="Whether to freeze the text encoder during training.")
    parser.add_argument("--freeze_decoder", action='store_true', help="Whether to freeze the decoder during training.")
    parser.add_argument("--semantic_amplification", type=float, default=100.0, help="Semantic layer amplification factor.")
    parser.add_argument("--resume_from_checkpoint", action='store_true', help="Whether to resume training from the last checkpoint.")
    args = parser.parse_args()
    
    main(args)