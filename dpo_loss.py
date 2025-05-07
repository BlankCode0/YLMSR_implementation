import torch
import torch.nn.functional as F

def dpo_loss(
    logp_policy_chosen,
    logp_ref_chosen,
    logp_policy_rejected,
    logp_ref_rejected,
    beta=0.1,
):
    """
    Computes the DPO loss:
    -log σ[β * ((log πθ(chosen) - log πref(chosen)) - (log πθ(rejected) - log πref(rejected)))]
    """
    diff = beta * (
        (logp_policy_chosen - logp_ref_chosen) -
        (logp_policy_rejected - logp_ref_rejected)
    )
    
    loss = F.binary_cross_entropy_with_logits(diff, torch.ones_like(diff))
    return loss
