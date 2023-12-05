import pandas as pd
from datasets import load_dataset

path1, name1 = 'brando/debug1_af', 'debug1_af'
dataset1 = load_dataset(path1, name1, streaming=True, split='test', token=None).with_format(type="torch")

# Assuming dataset1 is a PyTorch dataset
data = []

i = 0
for sample in dataset1:
    # Extract data from the sample and append it to the data list
    # Modify this part based on the structure of your dataset
    text1 = sample["generated informal statement"]
    text2 = sample["formal statement"]
    for txt in [text1, text2]:
        data.append({
            'text' : txt
        })
    i += 1
    if i == 30:
        break

# Convert the list of dictionaries to a Pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('../datasets/af/af-split-test-4k.csv', index=False)
