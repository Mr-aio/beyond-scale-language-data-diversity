import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Define your two custom datasets (train_dataset1 and train_dataset2)

# Initialize the tokenizer and pre-trained GPT-2 model
model_name = "gpt2"  # You can choose a different GPT-2 variant if needed
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Tokenize your datasets
train_dataset1 = train_dataset1.map(lambda examples: tokenizer(examples['text'], return_special_tokens_mask=True), batched=True)
train_dataset2 = train_dataset2.map(lambda examples: tokenizer(examples['text'], return_special_tokens_mask=True), batched=True)

# Create a data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're not doing masked language modeling
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",  # Directory for saving the fine-tuned model
    overwrite_output_dir=True,  # Overwrite the output directory if it exists
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=2,  # Batch size
    save_steps=10_000,  # Save a checkpoint every X updates
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset1 + train_dataset2,  # Combine both datasets
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./gpt2-finetuned")

# You can now use this fine-tuned model for text generation tasks.
