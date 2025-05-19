# Your Language Model is Secretly a Reward Model - DPO Implementation (from scratch)

This repository contains a **from-scratch PyTorch implementation** of the Direct Preference Optimization (DPO) algorithm described in the paper:

> **Your Language Model is Secretly a Reward Model**  
> [ArXiv Link](https://arxiv.org/abs/2305.18290)

We fine-tune a policy language model (π_θ) using preference data **without training a separate reward model** or using reinforcement learning. Instead, we leverage a reparameterized reward based on log-probability differences from a reference model (π_ref).

---

## ✅ Features Completed

### ✅ Core Architecture
- Preference dataset format: `(prompt, chosen, rejected)`
- Loading GPT-2 policy and reference models via HuggingFace
- Custom `compute_log_probs()` to extract token-level log-probs
- DPO loss implemented:  
  \[
  \mathcal{L}_{\text{DPO}} = -\log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right)
  \]
- Fine-tuning policy model weights on preference examples
- Saving and loading local model checkpoints
- `generate()` script to compare π_ref and π_θ model outputs

---

## 🧪 Controlled Experiment (Overfitting Test)

We manually created a single preference pair with a high-quality `"chosen"` response to the prompt:

> `" "`

After training on this pair, the policy model was able to reproduce the preferred style and content when prompted, confirming that the DPO pipeline works.

---

## 📂 Project Structure


