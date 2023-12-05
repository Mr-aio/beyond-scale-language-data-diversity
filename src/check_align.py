import time
import datetime

from diversity.task2vec import Task2Vec
from diversity import task_similarity
from diversity.div_coeff import cross_diversity_coefficient

from pathlib import Path

import torch
import torch.nn as nn


def alignment_task2vec(dataset_target,
                        dataset_source,
                        map_target: callable,
                        map_source: callable,
                        probe_network: nn.Module,
                        tokenizer = None,
                        batch_size: int = 1024,
                        seed: int = 0,
                        buffer_size: int = 500_000,
                        distance = 'cosine',
                        verbose: bool = False,
                        debug: bool = False,
                        shuffle: bool = True,  # False for faster debugging/testing but it won't be shuffled
                        ) -> dict:
    """
    Alignment v2 - with Task2Vec

    Given two data sets, compute how aligned they are using probe network f_w
        alg_2 = Align_2(T, S, f_w) = 1 - d(e_{D_S}, e_{D_T})
    by comparing embedding the entire dataset or a large batch.

    Note: there is no sense of number of batches here, so num_batches = 1 effectively + if CIs needed need to be with wrt batch examples.
    """
    # - Get target shuffled data
    shuffled_dataset = dataset_source.shuffle(buffer_size=buffer_size, seed=seed) if shuffle else dataset
    raw_text_batch = shuffled_dataset.take(batch_size)

    # raw_text_batch = shuffled_dataset.take(batch_size) if streaming else shuffled_dataset.select(range(batch_size))
    tokenized_batch = map_target(raw_text_batch)
    if verbose:
        print(f'{raw_text_batch=}')
        print(f'{tokenized_batch=}')

    # - Get Task2Vec embedding for batch
    if not debug:
        embedding_target, loss_target = Task2Vec(probe_network, classifier_opts={'seed': seed}).embed(tokenized_batch)
    else:
        embedding_target, loss_target = Task2Vec(probe_network, classifier_opts={'break_early': True, 'seed': seed}).embed(tokenized_batch, epochs=1)  # only for debugging
    print(f'{loss_target=}\n{embedding_target=}\n') if verbose else None

    # - Get source shuffled data
    shuffled_dataset = dataset_target.shuffle(buffer_size=buffer_size, seed=seed)
    raw_text_batch = shuffled_dataset.take(batch_size)

    tokenized_batch = map_source(raw_text_batch)

    # - Get Task2Vec embedding for batch
    if not debug:
        embedding_source, loss_source = Task2Vec(probe_network, classifier_opts={'seed': seed}).embed(tokenized_batch)
    else:
        embedding_source, loss_source = Task2Vec(probe_network, classifier_opts={'break_early': True, 'seed': seed}).embed(tokenized_batch, epochs=1)  # only for debugging
    print(f'{loss_target=}\n{embedding_target=}\n') if verbose else None

    # - Compute results
    embeddings, losses = [], []
    embeddings.append({'embedding_target': embedding_target, 'embedding_source': embedding_source})
    losses.append({'loss_target': loss_target, 'loss_source': loss_source})

    # - Compute alignment
    distance_matrix = task_similarity.pdist([embedding_target, embedding_source], distance=distance)
    align = 1 - distance_matrix[0, 1]
    align_ci = task_similarity.stats_of_distance_matrix(distance_matrix)[1]

    # - Results
    results: dict = {'align': align, 'align_ci': align_ci,
                    'embeddings': embeddings,
                    'distance_matrix': distance_matrix,
                    'losses': losses,
                    "batch_size": batch_size}
    return results


def cross_test(dataset1, dataset2):
    """ Sanity check that data from the same place has low. Prev work showed 0.05 is lower bound.
    so hopefully around that number. """
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

    def preprocess_formalize(examples):
        print("EXAMPLES", examples.keys())
        if 'generated informal statement' in examples:
            informal_statement = examples["generated informal statement"]
            formal_statement = examples["formal statement"]
            text = f'informal statement {informal_statement} formal statement {formal_statement}'
        elif 'text' in examples:
            text = examples['text']
        elif 'nl_statement' in examples:
            informal_statement = examples['nl_statement']
            formal_statement = examples['nl_proof']
            text = f'informal statement {informal_statement} formal statement {formal_statement}'
        elif 'informal_stmt' in examples:
            text = f'informal statement {examples["informal_stmt"]} formal statement {examples["formal_statement"]}'
        elif 'body' in examples:
            text = f'code for {examples["docstring"]} is {examples["body"]}'
        elif 'canonical_solution' in examples:
            text = f'code for {examples["docstring"]} is {examples["canonical_solution"]}'
        else:
            print('Error, dataset columns messed up.')
        return tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")

    # -- Dataset 1
    path1, name1 = dataset1
    try:
        dataset1 = load_dataset(path1, name1, streaming=True, split="train", token=token).with_format(type="torch")
    except:
        dataset1 = load_dataset(path1, name1, streaming=True, split="test", token=token).with_format(type="torch")
    batch1 = dataset1.take(batch_size)
    column_names1 = next(iter(batch1)).keys()
    print(f'{column_names1=}')
    # - Prepare functions to tokenize batch
    preprocess1 = preprocess_formalize
    remove_columns1 = column_names1  # remove everything except the tokenized fields in the dict
    print(f'{remove_columns1=}')
    def map1(batch):  # apply preprocess to batch to all examples in batch represented as a dataset
        return batch1.map(preprocess1, batched=True, remove_columns=remove_columns1)
    
    # Dataset 2
    path2, name2 = dataset2
    try:
        dataset2 = load_dataset(path2, name2, streaming=True, split="train", token=token).with_format(type="torch")
    except:
        dataset2 = load_dataset(path2, name2, streaming=True, split="test", token=token).with_format(type="torch")
    batch2 = dataset2.take(batch_size)
    column_names2 = next(iter(batch2)).keys()
    print(f'{column_names2=}')
    # - Prepare functions to tokenize batch
    preprocess2 = preprocess_formalize
    remove_columns2 = column_names2  # remove everything except the tokenized fields in the dict
    print(f'{remove_columns2=}')
    def map2(batch):  # apply preprocess to batch to all examples in batch represented as a dataset
        return batch2.map(preprocess2, batched=True, remove_columns=remove_columns2)

    # -- Compute alignment
    print('-- Compute alignment...')
    print(f'{batch_size=}')
    results = alignment_task2vec(dataset1, dataset2, map1, map2, probe_network, verbose=True, debug=False, batch_size=batch_size)
    print(f'{results=}')


# def cross_test(dataset1, dataset2):
#     """ Sanity check that data from the same place has low. Prev work showed 0.05 is lower bound.
#     so hopefully around that number. """
#     batch_size = 512
#     remove_columns = []
#     token = None

#     # -- Get probe network
#     from datasets import load_dataset
#     import torch
#     from transformers import GPT2Tokenizer, GPT2LMHeadModel

#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     if tokenizer.pad_token_id is None:
#       tokenizer.pad_token = tokenizer.eos_token
#     probe_network = GPT2LMHeadModel.from_pretrained("gpt2")
#     device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
#     probe_network = probe_network.to(device)

#     # -- Get batch from dataset
#     from datasets import load_dataset
#     # https://huggingface.co/datasets/brando/debug0_af/tree/main

#     def preprocess_formalize1(examples):
#         print("EXAMPLES", examples.keys())
#         if 'generated informal statement' in examples:
#             informal_statement = examples["generated informal statement"]
#             formal_statement = examples["formal statement"]
#             text = f'{informal_statement}. {formal_statement}'
#         elif 'text' in examples:
#             text = examples['text']
#         elif 'nl_statement' in examples:
#             informal_statement = examples['nl_statement']
#             formal_statement = examples['nl_proof']
#             text = f'informal statement {informal_statement} formal statement {formal_statement}'
#         elif 'informal_stmt' in examples:
#             text = f'informal statement {examples["informal_stmt"]} formal statement {examples["formal_statement"]}'
#         else:
#             print('Error, dataset columns messed up.')
#         return tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    
#     def preprocess_formalize2(examples):
#         print("EXAMPLES", examples.keys())
#         if 'generated informal statement' in examples:
#             informal_statement = examples["generated informal statement"]
#             formal_statement = examples["formal statement"]
#             text = f'informal statement {informal_statement} formal statement {formal_statement}'
#         elif 'text' in examples:
#             text = examples['text']
#         elif 'nl_statement' in examples:
#             informal_statement = examples['nl_statement']
#             formal_statement = examples['nl_proof']
#             text = f'informal statement {informal_statement} formal statement {formal_statement}'
#         elif 'informal_stmt' in examples:
#             text = f'informal statement {examples["informal_stmt"]} formal statement {examples["formal_statement"]}'
#         elif 'body' in examples:
#             text = f'code for {examples["docstring"]} is {examples["body"]}'
#         else:
#             print('Error, dataset columns messed up.')
#         return tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")

#     # -- Dataset 1
#     path1, name1 = dataset1
#     try:
#         dataset1 = load_dataset(path1, name1, streaming=True, split="train", token=token).with_format(type="torch")
#     except:
#         dataset1 = load_dataset(path1, name1, streaming=True, split="validation", token=token).with_format(type="torch")
#     batch1 = dataset1.take(batch_size)
#     column_names1 = next(iter(batch1)).keys()
#     print(f'{column_names1=}')
#     # - Prepare functions to tokenize batch
#     preprocess1 = preprocess_formalize1
#     remove_columns1 = column_names1  # remove everything except the tokenized fields in the dict
#     print(f'{remove_columns1=}')
#     def map1(batch):  # apply preprocess to batch to all examples in batch represented as a dataset
#         return batch1.map(preprocess1, batched=True, remove_columns=remove_columns1)
    
#     # Dataset 2
#     path2, name2 = dataset2
#     try:
#         dataset2 = load_dataset(path2, name2, streaming=True, split="train", token=token).with_format(type="torch")
#     except:
#         dataset2 = load_dataset(path2, name2, streaming=True, split="validation", token=token).with_format(type="torch")
#     batch2 = dataset2.take(batch_size)
#     column_names2 = next(iter(batch2)).keys()
#     print(f'{column_names2=}')
#     # - Prepare functions to tokenize batch
#     preprocess2 = preprocess_formalize2
#     remove_columns2 = column_names2  # remove everything except the tokenized fields in the dict
#     print(f'{remove_columns2=}')
#     def map2(batch):  # apply preprocess to batch to all examples in batch represented as a dataset
#         return batch2.map(preprocess2, batched=True, remove_columns=remove_columns2)

#     # -- Compute alignment
#     print('-- Compute alignment...')
#     print(f'{batch_size=}')
#     results = alignment_task2vec(dataset1, dataset2, map1, map2, probe_network, verbose=True, debug=False, batch_size=batch_size)
#     print(f'{results=}')


if __name__ == '__main__':
    import time
    time_start = time.time()
    
    # path1, name1 = 'brando/debug1_af', 'debug1_af'
    # path1, name1 = 'hoskinson-center/proofnet', 'plain_text'
    # path1, name1 = 'c4', 'en'
    # path1, name1 = 'hoskinson-center/minif2f-lean4', 'minif2f-lean4'
    # path1, name1 = 'hoskinson-center/proof-pile', 'default'
    # path1, name1 = 'brando/debug1_af', 'debug1_af'
    # path2, name2 = 'brando/debug1_af', 'debug1_af'
    # path1, name1 = "wikitext", 'wikitext-2-raw-v1'
    path1, name1 = 'bigcode/humanevalpack', 'python'

    # path1, name1 = 'calum/the-stack-smol-python-docstrings', 'default'
    path2, name2 = 'calum/the-stack-smol-python-docstrings', 'default'


    cross_test(dataset1=(path1, name1), dataset2=(path2, name2))
    # -- End tests, report how long it took
    print(f'Time it took: {time.time() - time_start} seconds \a\n')