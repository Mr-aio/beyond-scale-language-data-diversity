import logging
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    CONFIG_MAPPING,
    Trainer,
    TrainingArguments,
    set_seed,  
    default_data_collator,
)

def sanity2_af_is_aligned_to_af():
    """ Sanity check that data from the same place has low. Prev work showed 0.05 is lower bound.
    so hopefully around that number. """
    batch_size = 8
    batch_size = 512
    remove_columns = []
    token = None

    # -- Get probe network
    from datasets import load_dataset
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
      tokenizer.pad_token = tokenizer.eos_token
    probe_network = GPT2LMHeadModel.from_pretrained("gpt2")
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    probe_network = probe_network.to(device)

    # -- Get batch from dataset
    from datasets import load_dataset
    # https://huggingface.co/datasets/brando/debug0_af/tree/main
    path1, name1 = 'brando/debug0_af', 'debug0_af'
    dataset1 = load_dataset(path1, name1, streaming=True, split="train", token=token).with_format(type="torch")

    batch = dataset1.take(batch_size)
    def preprocess_formalize(examples):
        """ link,formal statement,generated informal statement,solvable by sledgehammer,keep or not,informalization correct """
        print("EXAMPLES", examples.keys())
        informal_statement = examples["generated informal statement"]
        formal_statement = examples["formal statement"]
        text = f'informal statement {informal_statement} formal statement {formal_statement}'
        return tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    column_names = next(iter(batch)).keys()
    print(f'{column_names=}')

    preprocess = preprocess_formalize
    remove_columns = column_names  # remove everything except the tokenized fields in the dict
    print(f'{remove_columns=}')
    def map(batch):  # apply preprocess to batch to all examples in batch represented as a dataset
        return batch.map(preprocess, batched=True, remove_columns=remove_columns)
    tokenized_batch = map(batch)
    for i, tokenized_example in enumerate(tokenized_batch):
        print(f'Tokenized Example {i + 1}: {tokenized_example}')


sanity2_af_is_aligned_to_af()