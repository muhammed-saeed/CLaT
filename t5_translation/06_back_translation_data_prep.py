import os
import pandas as pd

real_pcm_path = "/local/musaeed/OrthographicVariationModelT5Enbase/pcmRealEnBT/realPCM.txt"
bt_en_path = "/local/musaeed/OrthographicVariationModelT5Enbase/pcmRealEnBT/pcmreal2en.txt"
bt_pcm_path = "/local/musaeed/OrthographicVariationModelT5Enbase/EnglishRealPCMBT/enreal2pcm.txt"
real_en = '/local/musaeed/OrthographicVariationModelT5Enbase/EnglishRealPCMBT/realEnglish.txt'
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
train_df_.to_csv("/local/musaeed/OrthographicVariationModelT5Enbase/tsvData/T5Ortho_both_real_pcm_enreal_pcmbt_train.tsv", sep="\t",index = False)
# eval_df.to_csv("/home/CE/musaeed/t5_translation/data/tsv/eval.tsv", sep="\t", index= False)