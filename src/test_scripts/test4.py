import logging
from dataclasses import dataclass, field
from typing import Optional
from itertools import chain
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="gpt2",  # Change to the path or name of your pre-trained GPT-2 model
        metadata={"help": "The model checkpoint for weights initialization."},
    )

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of your dataset from the datasets library."}
    )
    train_file: Optional[str] = field(
        default="af-train.csv",  # Replace with the actual path
        metadata={"help": "Path to your training data file."},
    )
    validation_file: Optional[str] = field(
        default="af-test.csv",  # Replace with the actual path
        metadata={"help": "Path to your validation data file."},
    )

@dataclass
class MyTrainingArguments(TrainingArguments):
    output_dir: Optional[str] = field(
        default="./gpt-2-af-fine_tuned",  # Replace with the desired output directory
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    # Load your dataset
    dataset = load_dataset("csv", data_files={"train": data_args.train_file, "validation": data_args.validation_file})

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config)

    # Tokenize and preprocess the datasets
    column_names = list(dataset["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], padding=True, truncation=True)

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,  # Adjust based on your available resources
        remove_columns=column_names,
    )

    # Group texts into chunks
    block_size = 128  # Adjust as needed
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4,  # Adjust based on your available resources
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"] if training_args.do_train else None,
        eval_dataset=lm_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Train the model
    if training_args.do_train:
        if lm_datasets["train"] is not None:
            trainer.train()
            # Save the model to the specified output directory with the desired name
            trainer.save_model(output_dir="./gpt-fine-tuned-af")
            print('done!')
        else:
            print("Training dataset is empty or None. Check your tokenization and grouping functions.")

    # Evaluate the model
    if training_args.do_eval:
        results = trainer.evaluate()
        print(results)

if __name__ == "__main__":
    main()
