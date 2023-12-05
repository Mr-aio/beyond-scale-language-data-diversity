from tqdm import tqdm
import torch
import pandas as pd
from transformers import AutoTokenizer

# Load your CSV dataset
csv_path = 'docstring2/docstring2-train-4k.csv'
df = pd.read_csv(csv_path)

print('counting tokens...')

# Load your tokenizer
tokenizer_name_or_path = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

# Set a pad token for the tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenize the 'text' column in the DataFrame
tokenized_texts = tokenizer(df['text'].tolist(), padding=True, truncation=True, return_tensors='pt')

# Extract the input_ids from the tokenized texts
input_ids = tokenized_texts['input_ids']

# If using PyTorch, convert the input_ids to a PyTorch tensor
if isinstance(input_ids, torch.Tensor):
    input_ids = input_ids.cpu().numpy()

# Count the total number of tokens
total_tokens = input_ids.size

# Print the total token count
print(f"Total tokens in the dataset: {total_tokens}")
