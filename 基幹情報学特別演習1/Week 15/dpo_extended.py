import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from torch.optim import AdamW
import torch.nn.functional as F
from dpo_dataset_100 import preference_data
from tqdm import tqdm

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Model and Tokenizer Setup ---
model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Using a Qwen model for DPO

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load policy model (trainable)
policy_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
policy_model.to(device)
policy_model.train()

# Load reference model (non-trainable)
reference_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
reference_model.to(device)
reference_model.eval()
for param in reference_model.parameters():
    param.requires_grad = False

print(f"Loaded {model_name} on {device}")

# --- Preference Pair Data ---
pairs = preference_data
print(f"Generated {len(pairs)} preference pairs.")

# --- Utility Function for Log Probabilities ---
def get_log_probabilities(model, tokenizer, prompt, response):
    full_text = prompt + response
    inputs = tokenizer(
        full_text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length
    ).to(device)

    prompt_token_ids = tokenizer(prompt, return_tensors='pt').input_ids[0].size(0)
    outputs = model(**inputs, labels=inputs.input_ids)
    logits = outputs.logits
    response_logits = logits[:, prompt_token_ids-1:-1, :]
    response_labels = inputs.input_ids[:, prompt_token_ids:]

    log_probs = F.log_softmax(response_logits, dim=-1)
    target_len = min(log_probs.size(1), response_labels.size(1))
    lp = torch.gather(log_probs[:, :target_len, :], 2, response_labels[:, :target_len].unsqueeze(-1)).squeeze(-1)

    return lp.sum(dim=1)

# --- Evaluate Model Before Training ---
print("\n" + "="*50)
print("Policy Model Output BEFORE DPO Training:")
print("="*50)

policy_model.eval()
generation_config = GenerationConfig(
    max_new_tokens=20,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)
for pair in pairs:
    prompt = pair["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output_ids = policy_model.generate(**inputs, generation_config=generation_config)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    response_only = generated_text[len(prompt):].strip()

    print(f"Prompt: {prompt}")
    print(f"  Expected Preferred:  \"{pair['y_pos']}\"")
    print(f"  Expected Disfavored: \"{pair['y_neg']}\"")
    print(f"  Model Output (BEFORE): \"{response_only}\"")
    print("-" * 50)

policy_model.train()

# --- DPO Training Loop ---
optimizer = AdamW(policy_model.parameters(), lr=5e-6)
beta = 0.1
epochs = 5

print(f"\nStarting DPO training (beta={beta}, Epochs={epochs})...")

for epoch in range(epochs):
    total_loss = 0
    for pair in tqdm(pairs, desc=f"Training Epoch {epoch}", leave=False):
        prompt = pair["prompt"]
        y_pos = pair["y_pos"]
        y_neg = pair["y_neg"]

        pi_logp_pos = get_log_probabilities(policy_model, tokenizer, prompt, y_pos)
        pi_logp_neg = get_log_probabilities(policy_model, tokenizer, prompt, y_neg)

        with torch.no_grad():
            ref_logp_pos = get_log_probabilities(reference_model, tokenizer, prompt, y_pos)
            ref_logp_neg = get_log_probabilities(reference_model, tokenizer, prompt, y_neg)

        log_ratio_pos = pi_logp_pos - ref_logp_pos
        log_ratio_neg = pi_logp_neg - ref_logp_neg

        logits_diff = beta * (log_ratio_pos - log_ratio_neg)
        loss = -F.logsigmoid(logits_diff).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(pairs)
    print(f"Epoch {epoch}, Average DPO Loss: {avg_loss:.8f}")

print("\n--- DPO Training Finished ---")

# --- Evaluate Model After Training ---
print("\n" + "="*50)
print("Policy Model Output AFTER DPO Training:")
print("="*50)

policy_model.eval()
for pair in pairs:
    prompt = pair["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output_ids = policy_model.generate(**inputs, generation_config=generation_config)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    response_only = generated_text[len(prompt):].strip()

    print(f"Prompt: {prompt}")
    print(f"  Expected Preferred:  \"{pair['y_pos']}\"")
    print(f"  Expected Disfavored: \"{pair['y_neg']}\"")
    print(f"  Model Output (AFTER):  \"{response_only}\"")
    print("-" * 50)