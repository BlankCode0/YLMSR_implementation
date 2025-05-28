import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load GPT-2 for generation
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# Load sentiment classifier
sentiment_pipe = pipeline("sentiment-analysis")

# Load prompts
with open("data/imdb_prompts.json", "r") as f:
    prompts = json.load(f)

# Output list of (prompt, chosen, rejected)
preference_pairs = []

for prompt in prompts:
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate 2 completions
    completions = []
    for _ in range(2):
        output = model.generate(
            **input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        generated = tokenizer.decode(output[0][input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
        completions.append(generated.strip())

    # Use sentiment classifier
    scores = sentiment_pipe(completions)
    pos_idx = 0 if scores[0]["label"] == "POSITIVE" else 1
    neg_idx = 1 - pos_idx

    # Save pair
    preference_pairs.append({
        "prompt": prompt,
        "chosen": completions[pos_idx],
        "rejected": completions[neg_idx]
    })

# Save preference dataset
with open("data/imdb_generated_pairs.json", "w") as f:
    json.dump(preference_pairs, f, indent=2)

print(f"✅ Saved {len(preference_pairs)} IMDb preference pairs to data/imdb_generated_pairs.json")
