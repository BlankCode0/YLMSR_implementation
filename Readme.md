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
  L_DPO = -log σ [ β ( log π_θ(y_w | x) - log π_ref(y_w | x) 
                        - log π_θ(y_l | x) + log π_ref(y_l | x) ) ]
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


# copy the code name YLMSR_I.py




# test_generated.py

{
    "prompt": "Without wishing to be a killjoy, Brad Sykes is responsible for at least two of the most dull and clich films i've ever seen - this being one of them, and Camp Blood being another. The acting is",
    "chosen": "spot on. Some of the best parts are performed by Mark Rylance in a show of love and care. The music is also good - in fact, Brad's vocal prowess proves more than capable of playing a harmonica, even to the point",
    "rejected": "mostly bad, although I did give it an 8 out of 10 - perhaps because of the way the film uses \"sadamune.\" The only character to do more than a short joke is the famous ganashi lady, and this is not"
  }

  We can say for sure that now our model is working correctly. Few reasons for it are:
   1. The response of DPO model was positive unlike the ref model which was a mixed of positive and negative
   2. Dpo model was also able to give similar response from the chosen response. like including Brad.
🔵 Before fine-tuning (Reference Model):
2025-05-29 04:54:21.685374: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1748494461.707558    8020 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1748494461.714326    8020 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-05-29 04:54:21.744286: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Without wishing to be a killjoy, Brad Sykes is responsible for at least two of the most dull and clich films i've ever seen - this being one of them, and Camp Blood being another. The acting is a bit over the top, but the film is clearly a work of art, and it's not much of a shocker to see him get his hands on the script, and the action sequences are extremely well done. It's a shame that there isn't a lot of action, as it's really just a glorified action scene with a few moments of action. The movie is also a bit of a work of art as far as the dialogue is concerned. The dialogue is a bit lacking,

🟢 After DPO fine-tuning (Policy Model):
Without wishing to be a killjoy, Brad Sykes is responsible for at least two of the most dull and clich films i've ever seen - this being one of them, and Camp Blood being another. The acting is nothing short of beautiful and the cinematography is flawless. With the help of a gorgeous cast of actors, Brad will be able to deliver a truly memorable film that will leave you wishing you could say "Hmmm, maybe I should have shot this more".
