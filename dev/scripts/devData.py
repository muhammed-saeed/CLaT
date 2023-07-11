import pandas as pd



#tsv file path
tsv = "/local/musaeed/train.tsv"


# Initialize an empty list
lines_list = []

# Read the text file
with open('/local/musaeed/pcm_nsc-ud-dev.conllu', 'r') as file:
    # Iterate over each line in the file
    for line in file:
        # Check if the line starts with '# text_en ='
        if line.startswith('# text_en ='):
            # Remove the '# text_en =' prefix and append the line to the list
            lines_list.append(line[len('# text_en ='):].strip())

# Print the resulting list
print(lines_list)
df = pd.read_csv(tsv, sep='\t')

check_list = lines_list
filtered_df = df[~(df['input_text'].isin(check_list) | df['target_text'].isin(check_list))]
print(filtered_df.head())

print(f"the len difference ois {len(df) - len(filtered_df)}")

filtered_df.to_csv("/local/musaeed/CLaT/dev/data/trainDatawithoutDev.tsv",sep='\t')
