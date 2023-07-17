# Low-Resource Cross-Lingual Adaptive Training for Nigerian Pidgin
Low-Resource Cross-Lingual Adaptive Training for Nigerian Pidgin

## Installation

### Python version

* Python == 3.9

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


## Data

| Prefix                  | Input Text                                                                                                                                                                     | Target Text                                                                                                                                                                  |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| translate english to pcm | What a blessing it is to be loved by ‘ the entire association of our brothers in the world ’ !                                                                         | Our ‘ brothers wey dey everywhere for this world ’ love us . This one no be big blessing ? ( 1 Pet .                                                                        |
| translate english to pcm | In a sense , today’s system of things is like that man on death row .                                                                                                         | We fit talk sey this world just be like that man wey go soon die . How ?                                                                                                    |
| translate pcm to english | Caiaphas wait make night reach before e send soldiers go arrest Jesus .                                                                                                       | Caiaphas sent soldiers to arrest Jesus under the cover of night .                                                                                                            |
| translate pcm to english | All dis story na true, (bikos na di pesin wey si dem happen, naim tok about dem and wetin e tok na true), so dat una go fit bilive.                               | And he who saw it has given witness (and his witness is true; he is certain that what he says is true) so that you may have belief.                                       |
| translate pcm to english | Around two years before e talk that one , e don first tell them say : “ As wuna de go , make wuna preach say : ‘ The Kingdom of heaven don near . ’ ” | Some two years earlier , Jesus had instructed his apostles : “ As you go , preach , saying : ‘ The Kingdom of the heavens has drawn near . ’ ”                               |
| translate pcm to english | Dem dey shame sey dem do dis kind wiked tins? ‘No!’ Dem nor dey shame at-all; dem nor even sabi form sef! Bikos of dis, just as pipol wey don die, na so too dem go die finish afta I don ponish dem.” Mi, wey bi God, don tok. | Let them be put to shame because they have done disgusting things. They had no shame, they were not able to become red with shame: so they will come down with those who are falling: in the time of their punishment they will be made low, says the Lord. |
| translate pcm to english | Thru-out dat day, Abimelek fight for di town. E kill all di pipol for di town kon seize am. Den e skata di town kon pour salt full di groun.                                  | And all that day Abimelech was fighting against the town; and he took it, and put to death the people who were in it, and had the town pulled down and covered with salt. |
| translate english to pcm | For the Levites have no part among you; to be the Lord's priests is their heritage; and Gad and Reuben and the half-tribe of Manasseh have had their heritage on the east side of Jordan, given to them by Moses, the servant of the Lord.   | But Levi pipol nor go get any land among una, bikos dia propaty na to serve God. Gad, Reuben and half for Manasseh tribe don already take dia part for Jordan River east wey Moses give dem.” |
| translate pcm to english | Evribody dey like good ansa, but e dey good to tok di rite tin for di rite time!                                                                                             | A man has joy in the answer of his mouth: and a word at the right time, how good it is!                                                                                   |


## Training
### Fairseq

Preprocess

`bash FairseqTranslation/Machine Translation/joint_embeddings/fairseq_preprocess_bpe.sh `

Training

`bash FairseqTranslation/Machine Translation/joint_embeddings/fairseq_train_bpe.sh`

Generate 

`bash FairseqTranslation/Machine Translation/joint_embeddings/fairseq_generate_bpe.sh`

### T5 Translation

Data preparation

`python3 t5_translation/01_Data_Preperation.py path_to_data_directory "PATH TO DATA DIRECTORY `

Training

```
python3 /t5_translation/02_Translation.py \
     train_data_path "PATH TO TRAIN DATA" \
     checkpoint_path "PATH TO STORE CHECKPOINT"
     wandb_project "WANDB PROJECT NAME"
```

Evaluate the model

```
python3 t5_translation/03_Eval.py \
    model_path "PATH TO MODEL" \
    eval_data_path "PATH TO EVAL DATA" \
    pcm_to_en_results_path  "PATH TO STORE THE PCM TO EN TRANSLATION RESULTS" \
    en_to_pcm_results_path "PATH TO STORE THE PCM TO EN TRANSLATION RESULTS"
 ```


Back-translation data generation

```
python3 t5_translation/04_Back_translation_data_generate.py \
    model_output_dir "PATH TO MODEL" \
    pcm_mono_path "PATH TO MONOLINGUAL PCM DATA" \
    english_mono_path "PATH TO MONOLINGUAL EN DATA" \ 
    synthetic_english_path "PATH TO SAVE THE EN DATA" \
    synthetic_pcm_path "PATH TO SAVE PCM DATA" 
```

Back-translation data preparation

```
python3 t5_translation/05_Back_translation_data_prep.py  \
    --real_pcm_path PATH_TO_REAL_PCM \
    --bt_en_path PATH_TO_BACK_TRANSLATED_ENGLISH \
    --bt_pcm_path PATH_TO_BACK_TRANSLATED_PCM \
    --real_en PATH_TO_REAL_ENGLISH \
    --output_file PATH_TO_OUTPUT_FILE
 
 ```


## Multi Class classification training and evaluation

```
python3 Sentiment Analysis/mcls.py \
    --tokenizer_folder "PATH_TO/tokenizer" \
    --train_path "PATH_TO/pidgin/pcm_train.csv" \
    --test_path "PATH_TO/pidgin/pcm_test.csv" \
    --dev_path "PATH_TO/pidgin/pcm_dev.csv" \
    --model_path "PATH_TO_CHECKPOINT" \
    --epochs 3 \
    --learning_rate 1e-05 \
    --train_batch_size 4 \
    --valid_batch_size 2 \
    --max_len 512
```



### Inference Examples


|      Input Sentence in English      |                     Refrence Translation in Pidgin                     |                    Model's predictions                    |
|--------------|----------------------------------------------|--------------------------------------------|
| Keep peace with one another. — MARK 9:50.      | Make una be people wey like peace. — MARK 9:50. | Make una dey do things with each other. — MARK 9:50. |
| What counsel did Jesus give to help us handle differences in a spirit of love? | Which advice Jesus give wey fit help us use love settle quarrel? | Wetin Jesus talk wey go help us use love settle quarrel? |
| What questions might a Christian ask himself when deciding how to settle differences with others? | Which question we fit ask ourself if we dey think about how we go take settle the quarrel wey we get with people? | Wetin person wey dey serve Jehovah fit ask imself when e dey settle quarrel with another person? |
| How can the three steps outlined at Matthew 18:15-17 be used to resolve some conflicts? | How the three things wey Jesus talk for Matthew 18:15-17 fit help us settle quarrel? | Wetin we fit do to settle any quarrel wey dey Matthew 18:15-17? |
| What human struggles are featured in Genesis, and why is this of interest? | Wetin be the quarrel wey some people get wey dey Genesis? And wetin make am dey Bible? | For Genesis, which kind fight human being dey fight, and why this one concern us? |
| What attitude spread throughout the world, and what has been the result? | How fight and quarrel take full everywhere today? And wetin this kind thing don cause? | How people for the whole world take dey behave, and wetin don come happen? |
| How did Jesus teach people to handle disagreements? | Which advice Jesus give about how to settle quarrel? | How Jesus take teach people to handle quarrel? |
