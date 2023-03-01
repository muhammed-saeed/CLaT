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



model_args = T5Args()
model_args.max_length = 128
model_args.length_penalty = 1
model_args.num_beams = 5
model_args.eval_batch_size = 32


model_output_dir = "PATH_TO_YUOR_MODEL"

model = T5Model("t5", model_output_dir, args=model_args, )#cuda_devices=[6])

eval_df = pd.read_csv("PATH_TO_EVAL_DATASET/eval.tsv", sep="\t").astype(str)

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




# Predict
pcm_preds = model.predict(to_pcm_)


en_pcm_bleu = sacrebleu.corpus_bleu(pcm_preds, pcm_truth)
print("--------------------------")
print("English to Pidgin: ", en_pcm_bleu.score)

print(f"the type of the prediction is type(pcm_preds)")

with open("PAHT_TO_REPORT_RESULTS", "w", encoding="utf-8") as fb:
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
with open("PATH_TO_REPORT_RESULTS.txt", "w", encoding="utf-8") as fb:
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


