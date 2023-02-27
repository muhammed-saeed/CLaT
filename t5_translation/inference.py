import os

import logging
import sacrebleu
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


model_args = T5Args()
model_args.max_length = 512
model_args.length_penalty = 1
model_args.num_beams = 10
model_args.evaluation_batch_size=32


model_output_dir="PATH_TO_CHECKPOINT"
model = T5Model("mt5", model_output_dir, args=model_args, use_cuda=False)

 

sentence = ["translate english to pcm: what does Lord told his people and servants",
            "translate pcm to english: ‘ Make una be people wey like peace . ’ — MARK 9 : 50 .?"]
             




pcm_preds = model.predict(sentence)
print(f"the prediction is {pcm_preds}")
