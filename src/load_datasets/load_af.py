import pandas as pd
from datasets import load_dataset

path1, name1 = 'brando/debug0_af', 'debug0_af'
dataset1 = load_dataset(path1, name1, streaming=True, split='test', token=None).with_format(type="torch")

# Assuming dataset1 is a PyTorch dataset
data = []

for sample in dataset1:
    # Extract data from the sample and append it to the data list
    # Modify this part based on the structure of your dataset
    text = f'informal statement {sample["generated informal statement"]} formal statement {sample["formal statement"]}'
    data.append({
        'text' : text
    })

# Convert the list of dictionaries to a Pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('../datasets/af/af-test.csv', index=False)
