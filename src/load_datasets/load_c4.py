from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset from Hugging Face
dataset_name = "stas/c4-en-10k"
dataset = load_dataset(dataset_name)

# Select only 200 lines for training
train_subset = dataset['train'].select([i for i in range(5)])

# Convert the subset to a Pandas DataFrame
train_df = pd.DataFrame(train_subset)

# Select only 50 lines for testing
test_subset = dataset['train'].select([i for i in range(30, 35)])

# Convert the subset to a Pandas DataFrame
test_df = pd.DataFrame(test_subset)

# Save the training set to c4-train.csv
train_csv_path = "c4-train.csv"
train_df.to_csv(train_csv_path, index=False)
print(f"Training set saved to {train_csv_path}")

# Save the testing set to c4-test.csv
test_csv_path = "c4-test.csv"
test_df.to_csv(test_csv_path, index=False)
print(f"Testing set saved to {test_csv_path}")
