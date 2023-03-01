# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import logging
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_df= pd.read_csv("PATH_TO_SYNTHETIC_PARALLEL_DATA.tsv", sep="\t").astype(str)

# train_df["prefix"] = ""
# eval_df["prefix"] = ""
model_args = T5Args() 
model_args.max_seq_length = 128
model_args.train_batch_size = 8
model_args.eval_batch_size = 8
model_args.num_train_epochs = 20
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 30000
model_args.use_multiprocessing = False
model_args.fp16 = False
model_args.save_steps = -1
model_args.save_eval_checkpoints = False
model_args.no_cache = True
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.preprocess_inputs = False
# model_args.n_gpu = 2
model_args.num_return_sequences = 1
model_args.output_dir = "PATH_TO_OUTPUT_DIRECTORY"

model_args.wandb_project = "NAME OF THE PROJECT In WANDB"

model = T5Model("mt5", "google/mt5-base", args=model_args)

model = T5Model("m5", "t5-base", args=model_args)


model.train_model(train_df)