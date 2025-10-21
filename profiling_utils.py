import torch
import torch.nn.functional as F

def compute_policy_entropy(logits):
    """
    Computes the entropy of the policy distribution.
    logits: torch.Tensor of shape [num_actions] (before softmax)
    Returns a scalar entropy value.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs)
    return entropy.item()

def compute_kl_divergence(old_logits, new_logits):
    """
    Computes KL divergence between old and new policy distributions.
    Returns a scalar KL value.
    """
    old_probs = F.softmax(old_logits, dim=-1)
    old_log_probs = F.log_softmax(old_logits, dim=-1)
    new_log_probs = F.log_softmax(new_logits, dim=-1)
    kl = torch.sum(old_probs * (old_log_probs - new_log_probs))
    return kl.item()
