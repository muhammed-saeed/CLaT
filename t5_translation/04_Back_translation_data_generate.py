import os
import pandas as pd
import argparse
import logging
from simpletransformers.t5 import T5Model, T5Args


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Creating command line argument parser
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("model_output_dir", help="Path to your model directory.")
parser.add_argument("pcm_mono_path", help="Path to your PCM monolingual data.")
parser.add_argument("english_mono_path", help="Path to your English monolingual data.")
parser.add_argument("synthetic_english_path", help="Path to save your synthetic English results.")
parser.add_argument("synthetic_pcm_path", help="Path to save your synthetic PCM results.")
args = parser.parse_args()

# Setting CUDA_VISIBLE_DEVICES environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = "2,6"

model_args = T5Args()
model_args.max_length = 198
model_args.length_penalty = 1
model_args.fp16 = False
model_args.eval_batch_size = 16
model_args.num_beams = 10
model_args.n_gpu = 2

model = T5Model("mt5", args.model_output_dir, args=model_args, cuda_devices=[2,6])

pcm_data = open(args.pcm_mono_path,"r").readlines()
to_english = [line.lower() for line in pcm_data]

en2pcm = "translate english to pcm: "
pcm2en = "translate pcm to english: "
to_english_  = [pcm2en + s for s in to_english]

pcm_preds = model.predict(to_english_)

with open(args.synthetic_english_path,"w", encoding="utf-8") as fb:
    for line in pcm_preds:
        fb.write(line)
        fb.write("\n")

to_english = open(args.english_mono_path,"r").readlines()
to_pcm_  = [en2pcm + s for s in to_english]

english_preds = model.predict(to_pcm_)

with open(args.synthetic_pcm_path,"w", encoding="utf-8") as fb:
    for line in english_preds:
        fb.write(line)
        fb.write("\n")
