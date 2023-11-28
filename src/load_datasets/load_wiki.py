from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import csv

# Load the dataset from Hugging Face
dataset_name = "wikitext"
dataset = load_dataset(dataset_name, 'wikitext-2-raw-v1')

# Select only 200 lines for training
train_subset = dataset['train'].select([i for i in range(50)])

# Convert the subset to a Pandas DataFrame
train_df = pd.DataFrame(train_subset)

# Clean the 'text' column by removing double quotes
train_df['text'] = train_df['text'].str.replace('"', '')

# Save the training set to wiki-train.csv
train_csv_path = "wiki-train-pre.csv"
train_df.to_csv(train_csv_path, index=False)
print(f"Training set saved to {train_csv_path}")

# Select only 50 lines for testing
test_subset = dataset['train'].select([i for i in range(50, 80)])

# Convert the subset to a Pandas DataFrame
test_df = pd.DataFrame(test_subset)

# Clean the 'text' column by removing double quotes
test_df['text'] = test_df['text'].str.replace('"', '')
test_df['text'] = test_df['text'].str.replace("'", '')

# Save the testing set to wiki-test.csv
test_csv_path = "wiki-test-pre.csv"
test_df.to_csv(test_csv_path, index=False)
print(f"Testing set saved to {test_csv_path}")


def remove_quotes(input_file, output_file):
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write the header (first row) to the output file
        header = next(reader)
        writer.writerow(header)

        for row in reader:
            # Remove quote marks from each value in the row
            cleaned_row = [value.replace('"', '').replace("'", '').strip('\n') for value in row]

            # Check if any value is left in the row after removing quotes
            if cleaned_row[0] != '':
                # Write the cleaned row to the output file
                writer.writerow(cleaned_row)

# Example usage
remove_quotes('wiki-train-pre.csv', 'wiki-train.csv')
remove_quotes('wiki-test-pre.csv', 'wiki-test.csv')