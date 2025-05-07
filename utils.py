import json

def load_preference_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


import torch

def compute_log_probs(model, tokenizer, prompt, response, device="cuda"):
    """
    Compute log π(y | x) for a response given a prompt.
    Returns the total log-prob of response tokens conditioned on the prompt.
    """
    # Full input: prompt + response
    full_text = prompt + response

    # Tokenize with attention mask
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # outputs.loss = -average log likelihood
        # outputs.logits = [batch, seq_len, vocab]

    # Get log softmax over vocab
    logits = outputs.logits[:, :-1, :]  # cut last token prediction
    target_ids = input_ids[:, 1:]       # cut first input token (shifted)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Gather log_probs for target tokens
    target_log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)

    # Sum over response tokens only (not prompt)
    prompt_len = len(tokenizer(prompt)["input_ids"])
    response_log_prob = target_log_probs[:, prompt_len:].sum(dim=1).item()
    
    return response_log_prob
