import os
import pandas as pd

real_pcm_path = "PATH_TO_MONOLINGUAL_lPCM.txt"
bt_en_path = "PATH_TO_THE_FAKE_ENGLISH_DATA_ASSOCIATED_WITH_REAL_MONO_PCM.txt"
bt_pcm_path = "PATH_TO_FAKE_PCM_DATA_ASSOCIATED_WITH_REAL_MONO_ENGLISH"
real_en = 'PATH_TO_REAL_ENGLISH.txt'
def prepare_translation_datasets():
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

train_df_ = prepare_translation_datasets()
train_df_.to_csv("PATH_TO_SYNTHETIC_PARALLEL_DATA.tsv", sep="\t",index = False)
