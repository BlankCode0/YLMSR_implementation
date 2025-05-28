# used to check the response of the gpt2 model

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "gpt2"  # or your trained model path: e.g., "models/policy_model"

# Load your current π_θ
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Prompt
prompt = "If only to avoid making this type of film in the future. This film is interesting as an experiment but tells no cogent story.<br /><br />One might feel virtuous for sitting thru it because it touches on so many IMPORTANT"

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate
outputs = model.generate(
    **inputs,
    max_length=100,
    do_sample=True,       # enable sampling
    top_k=50,             # top-k sampling
    top_p=0.95,           # nucleus sampling
    temperature=0.7,      # lower temp = safer, higher = diverse
    pad_token_id=tokenizer.eos_token_id
)

# Decode
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("🤖 Model Response:")
print(generated_text)
