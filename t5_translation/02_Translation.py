import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import logging
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


train_df = pd.read_csv("PATH_TO_TRAINDATA.tsv", sep="\t").astype(str)



model_args = T5Args()
model_args.max_seq_length = 256
model_args.train_batch_size = 16
model_args.eval_batch_size = 16
model_args.num_train_epochs = 20
model_args.evaluate_during_training = False
model_args.evaluate_during_training_steps = 30000
model_args.use_multiprocessing = False
model_args.fp16 = False
model_args.save_steps = -1
model_args.save_eval_checkpoints = False
model_args.no_cache = True
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.preprocess_inputs = False
model_args.n_gpu = 3
model_args.num_return_sequences = 1
model_args.output_dir = "PATH_TO_CHECKPOINT/"

model_args.wandb_project = "NAME_OF_PROJECT_IN_WANDB"



model = T5Model("t5", "t5-base", args=model_args)

model.train_model(train_df )