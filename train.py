from utils import load_preference_data
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from utils import compute_log_probs
import torch

def main():
    dataset = load_preference_data("data/sample_data.json")
    print(f"Loaded {len(dataset)} samples.")
    print("Sample 1:")
    print(f"Prompt:   {dataset[0]['prompt']}")
    print(f"Chosen:   {dataset[0]['chosen']}")
    print(f"Rejected: {dataset[0]['rejected']}")


    # Loading GPT2
    model_name = "gpt2"
    tokennizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.eval()

    sample = dataset[0]
    prompt = sample["prompt"]
    chosen = sample["chosen"]
    rejected = sample["rejected"]

    logp_chosen = compute_log_probs(model, tokenizer, prompt, chosen, device)
    logp_rejected = compute_log_probs(model, tokenizer, prompt, rejected, device)

    print(f"\nLog π(chosen | prompt):   {logp_chosen:.4f}")
    print(f"Log π(rejected | prompt): {logp_rejected:.4f}")

if __name__ == "__main__":
    main()


