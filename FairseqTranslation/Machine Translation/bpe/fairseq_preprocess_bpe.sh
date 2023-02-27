SOURCE_LANGUAGE=pcm
TARGET_LANGUAGE=en
TRAIN_PREF="PATH_TO/train_bpe/train.bpe"
VALID_PREF="PATH_TO/val_bpe/val.bpe"
TEST_PREF="/PATH_TO/test_bpe/test.bpe"
DEST_DIR="PATH_TO/BPE_METHOD/pcm_en.tokenized.pcm-en"
SRC_THRES=0
TGT_THRES=0
EN_DICT_PATH="/PATH_TO/fairseq.en.vocab"
PCM_DICT_PATH="/PATH_TO/fairseq.pcm.vocab"
# --srcdict $PCM_DICT_PATH     --tgtdict $EN_DICT_PATH
fairseq-preprocess \
    --source-lang $SOURCE_LANGUAGE    --srcdict $PCM_DICT_PATH     --tgtdict $EN_DICT_PATH --target-lang $TARGET_LANGUAGE  --align-suffix align     --trainpref  $TRAIN_PREF        --validpref $VALID_PREF     --testpref $TEST_PREF   --destdir  $DEST_DIR     --thresholdsrc $SRC_THRES     --thresholdtgt $TGT_THRES
