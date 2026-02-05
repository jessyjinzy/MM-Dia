import torch
import numpy as np

import torch
import numpy as np

def calculate_topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k_values: list = [1, 10], ignore_index: int = -100):
    """
    计算单个批次的 top-k accuracy。
    Args:
        logits (torch.Tensor): 模型的输出 logits，形状为 (batch_size, seq_len, vocab_size)。
        labels (torch.Tensor): 真实的标签，形状为 (batch_size, seq_len)。
        k_values (list): 包含 k 值的列表。
        ignore_index (int): 忽略的标签值。
    Returns:
        dict: 一个包含各项 accuracy 和有效 token 数量的字典。
    """
    # 确保 logits 和 labels 在 CPU 上，以避免不必要的 GPU 占用和同步开销
    #logits = logits.cpu()
    #labels = labels.cpu()

    logits = logits[:, :-1, :]   # [B, L-1, V]
    labels = labels[:, 1:]  

    # 获取 top-k 预测结果
    topk_preds = torch.topk(logits, max(k_values), dim=-1).indices

    # 展平 labels 和 predictions 以便计算
    labels_flat = labels.flatten()
    
    # 过滤掉需要忽略的索引
    valid_indices = (labels_flat != ignore_index)
    labels_flat = labels_flat[valid_indices]
    
    # 获取有效 token 的数量
    num_valid_tokens = valid_indices.sum().item()
    
    if num_valid_tokens == 0:
        metrics = {f"top{k}_accuracy_sum": 0.0 for k in k_values}
        metrics["num_valid_tokens"] = 0
        return metrics

    metrics = {}
    for k in k_values:
        topk_preds_flat = topk_preds.view(-1, max(k_values))[:, :k]
        topk_preds_flat = topk_preds_flat[valid_indices]
        
        # 检查真实标签是否存在于 top-k 预测中
        correct_k = (topk_preds_flat == labels_flat.unsqueeze(-1)).any(dim=-1)
        
        # !!关键点!!：不计算平均值，而是返回正确预测的总数
        metrics[f"top{k}_accuracy_sum"] = correct_k.sum().item()
        
    metrics["num_valid_tokens"] = num_valid_tokens
    return metrics

from transformers.trainer_utils import EvalPrediction

def compute_metrics(p: EvalPrediction):
    """
    接收预先计算好的指标总和，并计算最终的平均值。
    """
    # p.predictions 是一个元组，包含了我们从 prediction_step 返回的所有批次的指标
    # 例如，p.predictions[0] 是一个包含所有批次 text_top1_sum 的 numpy 数组
    text_top1_sum_all = p.predictions[0]
    text_top10_sum_all = p.predictions[1]
    text_tokens_all = p.predictions[2]
    audio_semantic_top1_sum_all = p.predictions[3]
    audio_semantic_top10_sum_all = p.predictions[4]
    audio_semantic_tokens_all = p.predictions[5]
    audio_acoustic_top1_sum_all = p.predictions[6]
    audio_acoustic_top10_sum_all = p.predictions[7]
    audio_acoustic_tokens_all = p.predictions[8]

    # 计算总和
    total_text_top1_correct = np.sum(text_top1_sum_all)
    total_text_top10_correct = np.sum(text_top10_sum_all)
    total_text_tokens = np.sum(text_tokens_all)
    
    total_audio_semantic_top1_correct = np.sum(audio_semantic_top1_sum_all)
    total_audio_semantic_top10_correct = np.sum(audio_semantic_top10_sum_all)
    total_audio_semantic_tokens = np.sum(audio_semantic_tokens_all)

    total_audio_acoustic_top1_correct = np.sum(audio_acoustic_top1_sum_all)
    total_audio_acoustic_top10_correct = np.sum(audio_acoustic_top10_sum_all)
    total_audio_acoustic_tokens = np.sum(audio_acoustic_tokens_all)

    # 计算最终的 accuracy
    metrics = {}
    metrics["text_top1_accuracy"] = total_text_top1_correct / total_text_tokens if total_text_tokens > 0 else 0.0
    metrics["text_top10_accuracy"] = total_text_top10_correct / total_text_tokens if total_text_tokens > 0 else 0.0
    
    metrics["audio_semantic_top1_accuracy"] = total_audio_semantic_top1_correct / total_audio_semantic_tokens if total_audio_semantic_tokens > 0 else 0.0
    metrics["audio_semantic_top10_accuracy"] = total_audio_semantic_top10_correct / total_audio_semantic_tokens if total_audio_semantic_tokens > 0 else 0.0
    metrics["audio_acoustic_top1_accuracy"] = total_audio_acoustic_top1_correct / total_audio_acoustic_tokens if total_audio_acoustic_tokens > 0 else 0.0
    metrics["audio_acoustic_top10_accuracy"] = total_audio_acoustic_top10_correct / total_audio_acoustic_tokens if total_audio_acoustic_tokens > 0 else 0.0
    return metrics