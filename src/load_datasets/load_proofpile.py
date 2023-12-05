import pandas as pd
from datasets import load_dataset

path1, name1 = 'hoskinson-center/proof-pile', 'default'
dataset1 = load_dataset(path1, name1, streaming=True, split='test', token=None).with_format(type="torch")

# Assuming dataset1 is a PyTorch dataset
data = []

i = 0
for sample in dataset1:
    print('HEARTBEAT')
    # Extract data from the sample and append it to the data list
    # Modify this part based on the structure of your dataset
    # text = f'informal statement {sample["informal_stmt"]} formal statement {sample["formal_statement"]}'
    text = sample['text']
    data.append({
        'text' : text
    })
    i += 1
    if i == 4:
        break

# Convert the list of dictionaries to a Pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('../datasets/proofpile/proofpile-test-4k.csv', index=False)
