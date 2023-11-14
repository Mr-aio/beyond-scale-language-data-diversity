import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

# Load the fine-tuned model and tokenizer
model_path = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a custom dataset class for your test data
class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        # Load the CSV file using pandas
        df = pd.read_csv(file_path)

        self.data = []
        for _, row in df.iterrows():
            # Tokenize the text using the provided tokenizer
            tokens = tokenizer.encode(row["text"], add_special_tokens=True)
            self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])

# Load your custom test dataset
test_dataset = CustomDataset("./af-test.csv", tokenizer)

# Define a DataLoader to handle batching and padding
batch_size = 4  # Adjust as needed
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=pad_sequence)

# Measure perplexity
max_length = model.config.n_positions
stride = 512
seq_len = len(test_dataset[0])  # Assuming test_dataset[0] contains the first sequence

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on the last loop
    input_ids = test_dataset[0][begin_loc:end_loc].unsqueeze(0).to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())

print(f"Perplexity: {ppl.item():.4f}")
