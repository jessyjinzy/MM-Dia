import torch
import numpy as np

import torch
import numpy as np

def calculate_topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k_values: list = [1, 10], ignore_index: int = -100):
    """
    Calculate top-k accuracy for a single batch.
    Args:
        logits (torch.Tensor): Model output logits, shape (batch_size, seq_len, vocab_size).
        labels (torch.Tensor): Ground truth labels, shape (batch_size, seq_len).
        k_values (list): List of k values.
        ignore_index (int): Label value to ignore.
    Returns:
        dict: Dictionary containing accuracy metrics and valid token count.
    """
    # Ensure logits and labels are on CPU to avoid unnecessary GPU usage and sync overhead
    #logits = logits.cpu()
    #labels = labels.cpu()

    logits = logits[:, :-1, :]   # [B, L-1, V]
    labels = labels[:, 1:]

    # Get top-k predictions
    topk_preds = torch.topk(logits, max(k_values), dim=-1).indices

    # Flatten labels and predictions for calculation
    labels_flat = labels.flatten()

    # Filter out indices to ignore
    valid_indices = (labels_flat != ignore_index)
    labels_flat = labels_flat[valid_indices]

    # Get number of valid tokens
    num_valid_tokens = valid_indices.sum().item()

    if num_valid_tokens == 0:
        metrics = {f"top{k}_accuracy_sum": 0.0 for k in k_values}
        metrics["num_valid_tokens"] = 0
        return metrics

    metrics = {}
    for k in k_values:
        topk_preds_flat = topk_preds.view(-1, max(k_values))[:, :k]
        topk_preds_flat = topk_preds_flat[valid_indices]

        # Check if true label exists in top-k predictions
        correct_k = (topk_preds_flat == labels_flat.unsqueeze(-1)).any(dim=-1)

        # KEY: Return sum of correct predictions, not average
        metrics[f"top{k}_accuracy_sum"] = correct_k.sum().item()

    metrics["num_valid_tokens"] = num_valid_tokens
    return metrics

from transformers.trainer_utils import EvalPrediction

def compute_metrics(p: EvalPrediction):
    """
    Receive pre-computed metric sums and calculate final averages.
    """
    # p.predictions is a tuple containing metrics from all batches returned by prediction_step
    # For example, p.predictions[0] is a numpy array containing text_top1_sum from all batches
    text_top1_sum_all = p.predictions[0]
    text_top10_sum_all = p.predictions[1]
    text_tokens_all = p.predictions[2]
    audio_semantic_top1_sum_all = p.predictions[3]
    audio_semantic_top10_sum_all = p.predictions[4]
    audio_semantic_tokens_all = p.predictions[5]
    audio_acoustic_top1_sum_all = p.predictions[6]
    audio_acoustic_top10_sum_all = p.predictions[7]
    audio_acoustic_tokens_all = p.predictions[8]

    # Calculate totals
    total_text_top1_correct = np.sum(text_top1_sum_all)
    total_text_top10_correct = np.sum(text_top10_sum_all)
    total_text_tokens = np.sum(text_tokens_all)

    total_audio_semantic_top1_correct = np.sum(audio_semantic_top1_sum_all)
    total_audio_semantic_top10_correct = np.sum(audio_semantic_top10_sum_all)
    total_audio_semantic_tokens = np.sum(audio_semantic_tokens_all)

    total_audio_acoustic_top1_correct = np.sum(audio_acoustic_top1_sum_all)
    total_audio_acoustic_top10_correct = np.sum(audio_acoustic_top10_sum_all)
    total_audio_acoustic_tokens = np.sum(audio_acoustic_tokens_all)

    # Calculate final accuracy
    metrics = {}
    metrics["text_top1_accuracy"] = total_text_top1_correct / total_text_tokens if total_text_tokens > 0 else 0.0
    metrics["text_top10_accuracy"] = total_text_top10_correct / total_text_tokens if total_text_tokens > 0 else 0.0

    metrics["audio_semantic_top1_accuracy"] = total_audio_semantic_top1_correct / total_audio_semantic_tokens if total_audio_semantic_tokens > 0 else 0.0
    metrics["audio_semantic_top10_accuracy"] = total_audio_semantic_top10_correct / total_audio_semantic_tokens if total_audio_semantic_tokens > 0 else 0.0
    metrics["audio_acoustic_top1_accuracy"] = total_audio_acoustic_top1_correct / total_audio_acoustic_tokens if total_audio_acoustic_tokens > 0 else 0.0
    metrics["audio_acoustic_top10_accuracy"] = total_audio_acoustic_top10_correct / total_audio_acoustic_tokens if total_audio_acoustic_tokens > 0 else 0.0
    return metrics