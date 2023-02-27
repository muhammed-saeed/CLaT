import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import logging
import sacrebleu
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
import torch
torch.manual_seed(0)
import random 
random.seed(0)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# model_args = T5Args()
# model_args.max_length = 512
# model_args.length_penalty = 1
# model_args.num_beams = 10

model_args = T5Args()
model_args.max_length = 128
model_args.length_penalty = 1
model_args.num_beams = 5
model_args.eval_batch_size = 16

model_output_dir = "/home/CE/musaeed/checkpoint-128205-epoch-3"
# model_output_dir = "/local/musaeed/mt5_base_20_epoch/checkpoint-124900-epoch-5"
model_output_dir = "/local/musaeed/checkpoint-124900-epoch-8"
model_output_dir = "/local/musaeed/Naija-Pidgin/t5_translation/orthographicVariationReal/checkpoint-127520-epoch-20"

model = T5Model("t5", model_output_dir, args=model_args, )#cuda_devices=[6])

# eval_df = pd.read_csv("/home/CE/musaeed/Naija-Pidgin/t5_translation/data/tsv/eval.tsv", sep="\t").astype(str)
eval_df = pd.read_csv("/local/musaeed/Naija-Pidgin/t5_translation/data/trainValAugmentationAppendNew/tsv/eval.tsv", sep="\t").astype(str)

pcm_truth = [eval_df.loc[eval_df["prefix"] == "translate english to pcm"]["target_text"].tolist()]
to_pcm = eval_df.loc[eval_df["prefix"] == "translate english to pcm"]["input_text"].tolist()
pcm_truth_list = eval_df.loc[eval_df["prefix"] == "translate english to pcm"]["target_text"].tolist()

english_truth = [eval_df.loc[eval_df["prefix"] == "translate pcm to english"]["target_text"].tolist()]
to_english = eval_df.loc[eval_df["prefix"] == "translate pcm to english"]["input_text"].tolist()
english_truth_list = eval_df.loc[eval_df["prefix"] == "translate pcm to english"]["target_text"].tolist()
en2pcm = "translate english to pcm: "
pcm2en = "translate pcm to english: "
to_pcm_ = [en2pcm + s for s in to_pcm]
to_english_  = [pcm2en + s for s in to_english]


print(f"the english data is {to_english[:10]}")
print("#################################################")
# string_ = " ".join(pcm_truth[0])
# lines = pcm_truth[0].split(",")
print(f"the lenght of pcm_truth is {len(pcm_truth_list)}")
# print(f"examples of pcm_truth {pcm_truth}")

# Predict
pcm_preds = model.predict(to_pcm_)


en_pcm_bleu = sacrebleu.corpus_bleu(pcm_preds, pcm_truth)
print("--------------------------")
print("English to Pidgin: ", en_pcm_bleu.score)

print(f"the type of the prediction is type(pcm_preds)")

with open("/local/musaeed/Naija-Pidgin/t5_translation/eval_results/back_translation_real_pcm_pcm2en.txt", "w", encoding="utf-8") as fb:
    counter=0
    for index,pcm in enumerate(pcm_preds):
        source_line = "src: " + to_pcm[counter] + "\n"
        fb.write(source_line)
        real_line = "real: " + str(pcm_truth_list[counter]) + "\n"
        fb.write(real_line)
        pred_line = "pred: " + str(pcm) +"\n"
        
        fb.write(pred_line)
        sperator = " --------------     --------------  ----------------   -------------  \n "
        fb.write(sperator)
        counter +=1 


english_preds = model.predict(to_english_)

pcm_en_bleu = sacrebleu.corpus_bleu(english_preds, english_truth)
print("Pidgin to English: ", pcm_en_bleu.score)

counter=0
with open("/local/musaeed/Naija-Pidgin/t5_translation/eval_results/orthographicVariationen2en.txt", "w", encoding="utf-8") as fb:
    for index,en in enumerate(english_preds):
        source_line =  "src: " + to_english[counter] + "\n"
        fb.write(source_line)
        real_line = "real: " + str(english_truth_list[counter]) + "\n"
        fb.write(real_line)
        pred_line = "pred: " + str(en) +"\n"
        counter +=1
        fb.write(pred_line)
        sperator = " --------------     --------------  ----------------   -------------  \n "
        fb.write(sperator)


eval_df = pd.read_csv("/local/musaeed/Naija-Pidgin/t5_translation/data/tsv/eval.tsv", sep="\t").astype(str)

pcm_truth = [eval_df.loc[eval_df["prefix"] == "translate english to pcm"]["target_text"].tolist()]
to_pcm = eval_df.loc[eval_df["prefix"] == "translate english to pcm"]["input_text"].tolist()
pcm_truth_list = eval_df.loc[eval_df["prefix"] == "translate english to pcm"]["target_text"].tolist()

english_truth = [eval_df.loc[eval_df["prefix"] == "translate pcm to english"]["target_text"].tolist()]
to_english = eval_df.loc[eval_df["prefix"] == "translate pcm to english"]["input_text"].tolist()
english_truth_list = eval_df.loc[eval_df["prefix"] == "translate pcm to english"]["target_text"].tolist()
en2pcm = "translate english to pcm: "
pcm2en = "translate pcm to english: "
to_pcm_ = [en2pcm + s for s in to_pcm]
to_english_  = [pcm2en + s for s in to_english]


print(f"the english data is {to_english[:10]}")
print("#################################################")
# string_ = " ".join(pcm_truth[0])
# lines = pcm_truth[0].split(",")
print(f"the lenght of pcm_truth is {len(pcm_truth_list)}")
# print(f"examples of pcm_truth {pcm_truth}")

# Predict
pcm_preds = model.predict(to_pcm_)


en_pcm_bleu = sacrebleu.corpus_bleu(pcm_preds, pcm_truth)
print("--------------------------")
print("English to Pidgin: ", en_pcm_bleu.score)

print(f"the type of the prediction is type(pcm_preds)")

with open("/local/musaeed/Naija-Pidgin/t5_translation/eval_results/pcm2enDataRealTest.txt", "w", encoding="utf-8") as fb:
    counter=0
    for index,pcm in enumerate(pcm_preds):
        source_line = "src: " + to_pcm[counter] + "\n"
        fb.write(source_line)
        real_line = "real: " + str(pcm_truth_list[counter]) + "\n"
        fb.write(real_line)
        pred_line = "pred: " + str(pcm) +"\n"
        
        fb.write(pred_line)
        sperator = " --------------     --------------  ----------------   -------------  \n "
        fb.write(sperator)
        counter +=1 


english_preds = model.predict(to_english_)

pcm_en_bleu = sacrebleu.corpus_bleu(english_preds, english_truth)
print("Pidgin to English: ", pcm_en_bleu.score)

counter=0
with open("/local/musaeed/Naija-Pidgin/t5_translation/eval_results/en2pcmTranslationDataRealTest.txt", "w", encoding="utf-8") as fb:
    for index,en in enumerate(english_preds):
        source_line =  "src: " + to_english[counter] + "\n"
        fb.write(source_line)
        real_line = "real: " + str(english_truth_list[counter]) + "\n"
        fb.write(real_line)
        pred_line = "pred: " + str(en) +"\n"
        counter +=1
        fb.write(pred_line)
        sperator = " --------------     --------------  ----------------   -------------  \n "
        fb.write(sperator)

