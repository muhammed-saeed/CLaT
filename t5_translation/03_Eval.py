import os
import logging
import sacrebleu
import pandas as pd
import random 
import argparse
from simpletransformers.t5 import T5Model, T5Args
import torch

def main(args):
    torch.manual_seed(0)
    random.seed(0)

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    model_args = T5Args()
    model_args.max_length = 128
    model_args.length_penalty = 1
    model_args.num_beams = 5
    model_args.eval_batch_size = 32

    model = T5Model("t5", args.model_path, args=model_args)

    eval_df = pd.read_csv(args.eval_data_path, sep="\t").astype(str)

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

    pcm_preds = model.predict(to_pcm_)

    en_pcm_bleu = sacrebleu.corpus_bleu(pcm_preds, pcm_truth)
    print("--------------------------")
    print("English to Pidgin: ", en_pcm_bleu.score)

    print(f"the type of the prediction is type(pcm_preds)")

    with open(args.en_to_pcm_results_path, "w", encoding="utf-8") as fb:
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
    with open(args.pcm_to_en_results_path, "w", encoding="utf-8") as fb:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a T5 Model.")
    parser.add_argument("--model_path", help="Path to your model directory.")
    parser.add_argument("--eval_data_path", help="Path to your evaluation dataset.")
    parser.add_argument("--pcm_to_en_results_path", help="Path to save your English to Pidgin results.")
    parser.add_argument("--en_to_pcm_results_path", help="Path to save your Pidgin to English results.")
    args = parser.parse_args()

    main(args)
