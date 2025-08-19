# pip install -U transformers accelerate bitsandbytes peft
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
os.environ["HF_HOME"] = "/mnt/nfs/users/jalend/"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/nfs/users/jalend/transformers_cache"

checkpoint = "bigcode/starcoderbase-7b"
device_map = "auto"  # let Accelerate place layers across GPUs/CPU if needed


# 4-bit quantization config (NF4 + bfloat16 compute is a strong default on A6000)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# IMPORTANT: max_new_tokens is NOT a tokenizer arg; remove it from tokenizer init
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # safe pad for causal LM

if "3b" in checkpoint:
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )

# Data
source_lang = "rust"
dest_lang = "java"
df_source = pd.read_csv(f"data/{source_lang}.csv")  # expects 'prompt' and 'canonical_solution'
df_dest   = pd.read_csv(f"data/{dest_lang}.csv")    # you'll attach generations here later

# Prompt template (use your file)
with open(f"data/gpt_human_eval_{source_lang}_{dest_lang}.txt", "r") as f:
    prompt_template = f.read()

gen_kwargs = dict(
    max_new_tokens=250,
    do_sample=False,            # deterministic for Pass@1 eval
    temperature=0.0,
    top_p=1.0,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

def build_prompt(nl_prompt, src_code):
    # Your original code concatenated prompt+code with no separator; make it explicit.
    # If your template already contains placeholders, adapt accordingly.
    return f"{nl_prompt}\n\n# Source ({source_lang}):\n{src_code}\n\n# Target ({dest_lang}):\n"

@torch.inference_mode()
def run_inference(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, **gen_kwargs)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Return ONLY what the model added after the prompt
    return text[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
    # If your tokenizer detokenization trims whitespace differently, alternatively:
    # return text.split("# Target", 1)[-1]

# # Example single run
# example_prompt = build_prompt(df_source.iloc[0]["prompt"], df_source.iloc[0]["canonical_solution"])
# print(run_inference(example_prompt))

# Full loop + save predictions
pred = []
for i in tqdm(range(len(df_source))):
    nl = df_source.iloc[i]["prompt"]
    src = df_source.iloc[i]["canonical_solution"]
    prompt = build_prompt(nl, src)
    pred.append(run_inference(prompt))
    print("Model output:")
    print(pred[-1])
    break

df_out = df_dest.iloc[:len(pred)].copy()
df_out["generation"] = pred
out_path = f"{source_lang}_{dest_lang}_with_predictions_starcoderbase3b_4bit.csv"
df_out.to_csv(out_path, index=False)
print(f"Saved {out_path}")
