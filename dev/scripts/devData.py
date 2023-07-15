import pandas as pd



#tsv file path
tsv = "/local/musaeed/CLaT/train.tsv"


# Initialize an empty list

def returnTreenbank(file):
    # Read the text file
    lines_list = []
    with open(file, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Check if the line starts with '# text_en ='
            if line.startswith('# text_en ='):
                # Remove the '# text_en =' prefix and append the line to the list
                lines_list.append(line[len('# text_en ='):].strip())

    # Print the resulting list
    # print(lines_list)
    return lines_list
df = pd.read_csv(tsv, sep='\t')

check_list1 = returnTreenbank('/local/musaeed/pcm_nsc-ud-dev.conllu')
print(len(check_list1))

check_list1.extend(returnTreenbank('/local/musaeed/pcm_nsc-ud-test.conllu'))
print(len(check_list1))
filtered_df = df[~(df['input_text'].isin(check_list1) | df['target_text'].isin(check_list1))]
print(filtered_df.head())

print(f"the len difference ois {len(df) - len(filtered_df)}")

filtered_df.to_csv("/local/musaeed/CLaT/dev/data/trainDatawithoutDevNorTest.tsv",sep='\t')
