import json

def load_preference_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


import torch

def compute_log_probs(model, tokenizer, prompt, response, device="cuda", detach=False):
    """
    Compute log π(y | x) for a response given a prompt.
    Returns a torch tensor containing the total log-prob of the response tokens.
    If detach=True, the result is detached from computation graph (used for ref model).
    """
    full_text = prompt + response
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Do NOT use torch.no_grad() here — we want gradients!
    outputs = model(input_ids, labels=input_ids)

    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    target_log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)

    # Calculate log-prob of the response part only
    prompt_len = len(tokenizer(prompt)["input_ids"])
    log_prob = target_log_probs[:, prompt_len:].sum(dim=1)

    if detach:
        log_prob = log_prob.detach()

    return log_prob
