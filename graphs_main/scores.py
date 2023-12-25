import torch

def dcg_at_k(scores, k):
    ranks = torch.log2(torch.arange(2, k + 2).float())
    if len(scores) < k:
        padding_length = k - len(scores)
        padding = torch.zeros(padding_length)
        scores = scores.flatten()
        scores = torch.cat([scores, padding])
    return (scores / ranks).sum()

def ndcg_at_k(pred_scores, true_scores, k):
  _, true_indices = true_scores.sort(descending=True)
  _, pred_indices = pred_scores.sort(descending=True)
  ideal_dcg = dcg_at_k(true_scores[true_indices][:k], k)
  actual_dcg = dcg_at_k(true_scores[pred_indices][:k], k)
  return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0


def hit_rate_at_k(pred_scores, true_scores, k):
  combined = list(zip(pred_scores, true_scores))
  combined.sort(key=lambda x: x[0], reverse=True)
  top_k = combined[:k]
  hits = sum(t for _, t in top_k)
  hit_rate = hits / k
  return hit_rate

