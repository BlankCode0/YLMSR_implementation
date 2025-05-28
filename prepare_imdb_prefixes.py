from datasets import load_dataset

# Load IMDb dataset from HuggingFace (not Kaggle)
dataset = load_dataset("imdb")

# Use the 'train' split and extract review texts
reviews = dataset["train"]

# Optionally select 100 random samples
prefixes = []
for example in reviews.select(range(100)):
    text = example["text"]
    # Keep only the first 30-50 tokens as prefix
    prefix = " ".join(text.split()[:40])
    prefixes.append(prefix)

# Save to JSON
import json
with open("data/imdb_prompts.json", "w") as f:
    json.dump(prefixes, f, indent=2)

print("✅ IMDb prefixes saved to data/imdb_prompts.json")
