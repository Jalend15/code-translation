from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import os
# Ensure tqdm is set up for Pandas
tqdm.pandas()
os.environ["HF_HOME"] = "/mnt/nfs/users/jalend/"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/nfs/users/jalend/transformers_cache"


checkpoint = "bigcode/starcoderbase-3b"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

tokenizer_starcoderbase = AutoTokenizer.from_pretrained(checkpoint, max_new_tokens=1024)
model_starcoderbase = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

source_lang = "rust"
dest_lang = "java"
df_source = pd.read_csv(f"data/{source_lang}.csv")
df_dest = pd.read_csv(f"data/{dest_lang}.csv")


prompt_path = f"data/gpt_human_eval_{source_lang}_{dest_lang}.txt"
with open(prompt_path, "r") as prompt_file:
    prompt_obj = prompt_file.read()

count = 0
length = 0


# Function to load model and tokenizer
def load_model_and_tokenizer(checkpoint_dir):
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    return model, tokenizer


# Inference function
def run_inference(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=250)  # Generate text
    # Generate text

    # Decode the output tensor to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(generated_text)
    return generated_text



for i in range(len(df_source)):
    text = df_source.iloc[i]["prompt"]
    code = df_source.iloc[i]["canonical_solution"]
    output = run_inference(text+"\n\n"+code, model_starcoderbase, tokenizer_starcoderbase)
    print("Model's output:", output)
    break



# # Load only the first `len(pred)` rows from the CSV
# df_java = pd.read_csv(f"data/{dest_lang}.csv", nrows=len(pred))

# # Add the predictions to the DataFrame
# df_java["generation"] = pred

# # Save the updated DataFrame
# output_path = f"{source_lang}_{dest_lang}_with_predictions_starcoderbase7b_164.csv"
# df_java.to_csv(output_path, index=False)
