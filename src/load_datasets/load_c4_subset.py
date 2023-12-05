from datasets import load_dataset
import pandas as pd

# Load the dataset from Hugging Face
dataset_name = "stas/c4-en-10k"
dataset = load_dataset(dataset_name)

# Function to truncate text to 100 characters
truncate_text = lambda x: x[:100] if isinstance(x, str) else x

# Select and truncate only 200 lines for training
train_subset = dataset['train'].select([i for i in range(300, 405)])
train_df = pd.DataFrame(train_subset)
train_df = train_df.map(truncate_text)

# Select and truncate only 50 lines for testing
test_subset = dataset['train'].select([i for i in range(40, 50)])
test_df = pd.DataFrame(test_subset)
test_df = test_df.map(truncate_text)

# Save the training set to c4-train.csv
train_csv_path = "../datasets/c4/c4-3-train-4k.csv"
train_df.to_csv(train_csv_path, index=False)
print(f"Training set saved to {train_csv_path}")

# Save the testing set to c4-test.csv
test_csv_path = "../datasets/c4/c4-3-test-4k.csv"
test_df.to_csv(test_csv_path, index=False)
print(f"Testing set saved to {test_csv_path}")
