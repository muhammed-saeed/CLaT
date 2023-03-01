import os
import pandas as pd

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,6"
import logging
import sacrebleu
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
[]

model_args = T5Args()
model_args.max_length = 198
model_args.length_penalty = 1
model_args.fp16=False
model_args.eval_batch_size=16
model_args.num_beams = 10
model_args.n_gpu=2
# model_args.eval_batch_size = 16






def prepare_translation_datasets(data_path):
    with open(os.path.join(data_path, "back_translation.pcm"), "r", encoding="utf-8") as f:
        pcm_text = f.readlines()
        pcm_text = [text.strip("\n") for text in pcm_text]

    with open(os.path.join(data_path, "back_translation.en"), "r") as f:
        english_text = f.readlines()
        english_text = [text.strip("\n") for text in english_text]

    data = []
    for pcm, english in zip(pcm_text, english_text):
        data.append(["translate pcm to english", pcm, english])
        data.append(["translate english to pcm", english, pcm])

    train_df = pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])


model_output_dir = "PATH_TO_CHECKPOINT"
model = T5Model("mt5", model_output_dir, args=model_args, cuda_devices=[2,6])


pcm_mono_path = "PATH_TO_MONOLINGUAL_PCM"

english_mono_path = ""
pcm_data = open(pcm_mono_path,"r").readlines()
to_english = [line.lower() for line in pcm_data]

en2pcm = "translate english to pcm: "
pcm2en = "translate pcm to english: "
# to_pcm_ = [en2pcm + s for s in to_pcm]
to_english_  = [pcm2en + s for s in to_english]

pcm_preds = model.predict(to_english_)


with open("PATH_TO_SyntheticEnglish","w", encoding="utf-8") as fb:
    for line in pcm_preds:
        fb.write(line)
        fb.write("\n")




english_mono_path = 'PATH_TO_MONO_ENGLISH'


en2pcm = "translate english to pcm: "
pcm2en = "translate pcm to english: "
# to_pcm_ = [en2pcm + s for s in to_pcm]
to_pcm_  = [en2pcm + s for s in to_english]



english_pred = model.predict(to_pcm_)

with open("PATH_TO_SyntheticPCM.txt","w", encoding="utf-8") as fb:
    for line in pcm_preds:
        fb.write(line)
        fb.write("\n")

