import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from utils import load_preference_data, compute_log_probs
from dpo_loss import dpo_loss

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "gpt2"

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
policy_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

# Load dataset
dataset = load_preference_data("data/imdb_generated_pairs.json")

# Optimizer
optimizer = AdamW(policy_model.parameters(), lr=1e-5)
epochs = 50
beta = 0.1

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for sample in dataset:
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        # Compute log probabilities
        logp_pi_c = compute_log_probs(policy_model, tokenizer, prompt, chosen, device)
        logp_pi_r = compute_log_probs(policy_model, tokenizer, prompt, rejected, device)

        logp_ref_c = compute_log_probs(ref_model, tokenizer, prompt, chosen, device, detach=True)
        logp_ref_r = compute_log_probs(ref_model, tokenizer, prompt, rejected, device, detach=True)

        # Loss
        loss = dpo_loss(logp_pi_c, logp_ref_c, logp_pi_r, logp_ref_r, beta)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        ## to check gradient
        # print("Grad norm:", policy_model.transformer.h[0].mlp.c_fc.weight.grad.norm().item())

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

# Save trained model
policy_model.save_pretrained("models/policy_model")
tokenizer.save_pretrained("models/policy_model")

ref_model.save_pretrained("models/ref_model")
tokenizer.save_pretrained("models/ref_model")

print("✅ Model saved to models/policy_model")

