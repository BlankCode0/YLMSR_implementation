from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils import compute_log_probs
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
policy_model = AutoModelForCausalLM.from_pretrained("models/policy_model").to(device)
ref_model = AutoModelForCausalLM.from_pretrained("models/ref_model").to(device)
tokenizer = AutoTokenizer.from_pretrained("models/policy_model")

policy_model.eval()
ref_model.eval()

# Load data
with open("data/imdb_generated_pairs.json", "r") as f:
    dataset = json.load(f)

print("🔍 DPO reward evaluation on policy model vs reference model:\n")

for i, sample in enumerate(dataset):
    prompt = sample["prompt"]
    chosen = sample["chosen"]
    rejected = sample["rejected"]

    # Score log probs
    logp_policy_chosen = compute_log_probs(policy_model, tokenizer, prompt, chosen, device)
    logp_policy_rejected = compute_log_probs(policy_model, tokenizer, prompt, rejected, device)

    logp_ref_chosen = compute_log_probs(ref_model, tokenizer, prompt, chosen, device)
    logp_ref_rejected = compute_log_probs(ref_model, tokenizer, prompt, rejected, device)

    # Compute reward-style score
    reward_chosen = logp_policy_chosen - logp_ref_chosen
    reward_rejected = logp_policy_rejected - logp_ref_rejected

    print(f"Example {i+1}")
    print(f"  r(chosen)  = {reward_chosen.item():.4f}")
    print(f"  r(rejected)= {reward_rejected.item():.4f}")
    better = "CHOSEN ✅" if reward_chosen > reward_rejected else "REJECTED ❌"
    print(f"  Preferred: {better}\n")
