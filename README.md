# Low-Resource Cross-Lingual Adaptive Training for Nigerian Pidgin
Low-Resource Cross-Lingual Adaptive Training for Nigerian Pidgin

## Installation

### Python version

* Python == 3.9.1

### Environment

Create an environment from file and activate the environment.

```
conda env create -n PROJECT_NAME python=3.9.1
conda activate PROJECT_NAME
```

Intalling the dependencies

```
pip install -f requirements.txt
```


Installing Fairseq

```
git clone https://github.com/pytorch/fairseq -q
cd fairseq
pip uninstall numpy -q -y
pip install wandb -q
pip install --editable ./ -q
cd ..
```

## Dataset


### Dataset Preprocessing and text partitioining
Process the bible books

```
python3 Pre-Processing Bible Code using BLocks/en_bible_pre_process.py --pdf_path "PATH TO English PDF File" --output_dir "PATH TO FOLDER IN WHICH THE SEPERATED VERSES WILL BE GENERATED"
python3 Pre-Processing Bible Code using BLocks/pcm_bible_process.py --pdf_path "PATH TO PCM PDF File" --output_dir "PATH TO FOLDER IN WHICH THE SEPERATED VERSES WILL BE GENERATED"
```


## Training
### Fairseq
-Preprocess
`bash FairseqTranslation/Machine Translation/joint_embeddings/fairseq_preprocess_bpe.sh `
- Training
`bash FairseqTranslation/Machine Translation/joint_embeddings/fairseq_train_bpe.sh`
- Generate 
`bash FairseqTranslation/Machine Translation/joint_embeddings/fairseq_generate_bpe.sh`

### T5 Translation
First Data preperation


`python3 t5_translation/01_Data_Preperation.py.py path_to_data_directory `


Second Training

`python3 /t5_translation/02_Translation.py.py train_data_path checkpoint_path wandb_project `

Third Evaluate the model

`python3 t5_translation/03_Eval.py model_path eval_data_path pcm_to_en_results_path en_to_pcm_results_path `

Fourth Back-translation data generation

`python3 t5_translation/04_Back_translation_data_generate.py.py model_output_dir pcm_mono_path english_mono_path synthetic_english_path synthetic_pcm_path `

Fifth Back-translation data preperation

`python3 t5_translation/05_Back_translation_data_prep.py --real_pcm_path PATH_TO_REAL_PCM --bt_en_path PATH_TO_BACK_TRANSLATED_ENGLISH --bt_pcm_path PATH_TO_BACK_TRANSLATED_PCM --real_en PATH_TO_REAL_ENGLISH --output_file PATH_TO_OUTPUT_FILE `

