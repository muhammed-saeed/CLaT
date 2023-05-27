import os
import pandas as pd
import argparse

def prepare_translation_datasets(real_pcm_path, bt_en_path, bt_pcm_path, real_en):
    with open(bt_pcm_path, "r", encoding="utf-8") as f:
        pcm_text = f.readlines()
        pcm_text = [text.strip("\n") for text in pcm_text]

    with open(real_en, "r") as f:
        english_text = f.readlines()
        english_text = [text.strip("\n") for text in english_text]

    data = []
    for pcm, english in zip(pcm_text, english_text):
        data.append(["translate pcm to english", pcm, english])
        data.append(["translate english to pcm", english, pcm])

    train_df = pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])
    
    pcm_text = []
    english_text = []
    data=[]

    with open(real_pcm_path, "r", encoding="utf-8") as f:
        pcm_text = f.readlines()
        pcm_text = [text.strip("\n") for text in pcm_text]

    with open(bt_en_path, "r", encoding="utf-8") as f:
        english_text = f.readlines()
        english_text = [text.strip("\n") for text in english_text]

    data = []
    for pcm, english in zip(pcm_text, english_text):
        data.append(["translate pcm to english", pcm, english])
        data.append(["translate english to pcm", english, pcm])
    
    dev_df =  pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])

    train_df_  = pd.concat(
    [train_df, dev_df],
    axis=0,
    join="outer",
    ignore_index=False,
    keys=None,
    levels=None,
    names=None,
    verify_integrity=False,
    copy=True,
    )
    return train_df_

parser = argparse.ArgumentParser()
parser.add_argument("--real_pcm_path", help="Path to the real PCM data.")
parser.add_argument("--bt_en_path", help="Path to the back-translated English data.")
parser.add_argument("--bt_pcm_path", help="Path to the back-translated PCM data.")
parser.add_argument("--real_en", help="Path to the real English data.")
parser.add_argument("--output_file", help="Path to save the synthetic parallel data.")
args = parser.parse_args()

train_df_ = prepare_translation_datasets(args.real_pcm_path, args.bt_en_path, args.bt_pcm_path, args.real_en)
train_df_.to_csv(args.output_file, sep="\t",index = False)
