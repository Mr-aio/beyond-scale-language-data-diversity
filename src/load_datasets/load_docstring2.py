import pandas as pd
from datasets import load_dataset

path1, name1 = 'trelent/the-stack-dedup-python-docstrings-1.0-percent-unified', 'default'
dataset1 = load_dataset(path1, name1, streaming=True, split='train', token=None).with_format(type="torch")

# Assuming dataset1 is a PyTorch dataset
data = []

i = 0
for sample in dataset1:
    
    # Extract data from the sample and append it to the data list
    # Modify this part based on the structure of your dataset
    i += 1
    if i < 6:
        continue
    if i == 12:
        break
    docstring = sample['docstring']
    body = sample['body']
    text = f'code for {docstring} is {body}'
    # text = f'informal statement {sample["generated informal statement"]} formal statement {sample["formal statement"]}'
    data.append({
        'text' : text
    })
    



# Convert the list of dictionaries to a Pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('../datasets/docstring2/docstring2-test-4k.csv', index=False)
