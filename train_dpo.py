from utils import load_preference_data
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
#calculate log probabilities
from utils import compute_log_probs
import torch
# Dpo loss
from dpo_loss import dpo_loss


dataset = load_preference_data("data/sample_data.json")
print(f"Loaded {len(dataset)} samples.")
print("Sample 1:")
print(f"Prompt:   {dataset[0]['prompt']}")
print(f"Chosen:   {dataset[0]['chosen']}")
print(f"Rejected: {dataset[0]['rejected']}")


# Loading GPT2

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
policy_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
ref_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
policy_model.eval()
ref_model.eval()

sample = dataset[0]
prompt = sample["prompt"]
chosen = sample["chosen"]
rejected = sample["rejected"]

logp_pi_c = compute_log_probs(policy_model, tokenizer, prompt, chosen, device)
logp_pi_r = compute_log_probs(policy_model, tokenizer, prompt, rejected, device)
logp_ref_c = compute_log_probs(ref_model, tokenizer, prompt, chosen, device)
logp_ref_r = compute_log_probs(ref_model, tokenizer, prompt, rejected, device)

# Convert to tensor
logp_pi_c = torch.tensor([logp_pi_c]).to(device)
logp_pi_r = torch.tensor([logp_pi_r]).to(device)
logp_ref_c = torch.tensor([logp_ref_c]).to(device)
logp_ref_r = torch.tensor([logp_ref_r]).to(device)

loss = dpo_loss(logp_pi_c, logp_ref_c, logp_pi_r, logp_ref_r, beta=0.1)

print(f"\n💣 DPO Loss: {loss.item():.4f}")


