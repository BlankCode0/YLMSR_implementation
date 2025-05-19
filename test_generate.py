from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_response(model_path, prompt, max_new_tokens=100, temperature=0.7):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated

if __name__ == "__main__":
    prompt = "Why is the sky blue?"

    print("\n🔵 Before fine-tuning (Reference Model):")
    print(generate_response("gpt2", prompt))

    print("\n🟢 After DPO fine-tuning (Policy Model):")
    print(generate_response("models/policy_model", prompt))
