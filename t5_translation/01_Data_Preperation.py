import os
import pandas as pd
import argparse

def prepare_translation_datasets(data_path):
    with open(os.path.join(data_path, "train.pcm"), "r", encoding="utf-8") as f:
        pcm_text = f.readlines()
        pcm_text = [text.strip("\n") for text in pcm_text]

    with open(os.path.join(data_path, "train.en"), "r") as f:
        english_text = f.readlines()
        english_text = [text.strip("\n") for text in english_text]

    data = []
    for pcm, english in zip(pcm_text, english_text):
        data.append(["translate pcm to english", pcm, english])
        data.append(["translate english to pcm", english, pcm])

    train_df = pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])


    with open(os.path.join(data_path, "val.pcm"), "r", encoding="utf-8") as f:
        pcm_text = f.readlines()
        pcm_text = [text.strip("\n") for text in pcm_text]

    with open(os.path.join(data_path, "val.en"), "r") as f:
        english_text = f.readlines()
        english_text = [text.strip("\n") for text in english_text]

    data = []
    for pcm, english in zip(pcm_text, english_text):
        data.append(["translate pcm to english", pcm, english])
        data.append(["translate english to pcm", english, pcm])
    
    dev_df =  pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])


    with open("/local/musaeed/Naija-Pidgin/t5_translation/data/trainValAugmentationAppendNew/pcmAugment.txt", "r", encoding="utf-8") as f:
        pcm_text = f.readlines()
        pcm_text = [text.strip("\n") for text in pcm_text]

    with open("/local/musaeed/Naija-Pidgin/t5_translation/data/trainValAugmentationAppendNew/enAugment.txt", "r") as f:
        english_text = f.readlines()
        english_text = [text.strip("\n") for text in english_text]

    data = []
    for pcm, english in zip(pcm_text, english_text):
        data.append(["translate pcm to english", pcm, english])
        data.append(["translate english to pcm", english, pcm])
    
    augDf =  pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])





    with open("/local/musaeed/Naija-Pidgin/Treebank_test/testing_Data/pcm_parrellel.txt", "r", encoding="utf-8") as f:
        pcm_text = f.readlines()
        pcm_text = [text.strip("\n") for text in pcm_text]

    with open("/local/musaeed/Naija-Pidgin/Treebank_test/testing_Data/en_parrellel.txt", "r") as f:
        english_text = f.readlines()
        english_text = [text.strip("\n") for text in english_text]

    data = []
    for pcm, english in zip(pcm_text, english_text):
        data.append(["translate pcm to english", pcm, english])
        data.append(["translate english to pcm", english, pcm])

    TreeBank =  pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])


    train_df_  = pd.concat(
    [train_df, dev_df, augDf, TreeBank],
    axis=0,
    join="outer",
    ignore_index=False,
    keys=None,
    levels=None,
    names=None,
    verify_integrity=False,
    copy=True,
    )

    with open(os.path.join(data_path, "test.pcm"), "r", encoding="utf-8") as f:
        pcm_text = f.readlines()
        pcm_text = [text.strip("\n") for text in pcm_text]

    with open(os.path.join(data_path, "test.en"), "r") as f:
        english_text = f.readlines()
        english_text = [text.strip("\n") for text in english_text]

    data = []
    for pcm, english in zip(pcm_text, english_text):
        data.append(["translate pcm to english", pcm, english])
        data.append(["translate english to pcm", english, pcm])

    with open("/local/musaeed/Naija-Pidgin/t5_translation/data/trainValAugmentationAppendNew/testpcmAugment.txt", "r", encoding="utf-8") as f:
        pcm_text = f.readlines()
        pcm_text = [text.strip("\n") for text in pcm_text]

    with open("/local/musaeed/Naija-Pidgin/t5_translation/data/trainValAugmentationAppendNew/testenAugment.txt", "r") as f:
        english_text = f.readlines()
        english_text = [text.strip("\n") for text in english_text]

    eval_df = pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])

    data = []
    for pcm, english in zip(pcm_text, english_text):
        data.append(["translate pcm to english", pcm, english])
        data.append(["translate english to pcm", english, pcm])

    Augeval_df = pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])


    eval_df  = pd.concat(
    [eval_df,Augeval_df],
    axis=0,
    join="outer",
    ignore_index=False,
    keys=None,
    levels=None,
    names=None,
    verify_integrity=False,
    copy=True,
    )
    return train_df_, eval_df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepares translation datasets.")
    parser.add_argument("data_path", help="Path to the directory containing data files.")
    args = parser.parse_args()

    train_df, eval_df = prepare_translation_datasets(args.data_path)

    train_df = train_df.sample(frac=1)
    eval_df = eval_df.sample(frac=1)
    train_df.to_csv(os.path.join(args.data_path, "trainValAugmentationAppendNew/8_more_tsv/train.tsv"), sep="\t", index=False)
    eval_df.to_csv(os.path.join(args.data_path, "trainValAugmentationAppendNew/8_more_tsv/eval.tsv"), sep="\t", index=False)
